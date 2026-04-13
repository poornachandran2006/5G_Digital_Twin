"""
5G Network Digital Twin — FastAPI Backend
WebSocket + REST API for live dashboard

Real wiring:
  - NetworkSimulation (SimPy) drives every tick
  - EnsemblePredictor (LSTM + XGBoost) produces congestion_predictions
  - PPOAgent queries the trained policy for ppo_actions
  - Falls back to mock generator if any component fails to load
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import pickle
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Iterator, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make project root importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("api")

# ── Constants ─────────────────────────────────────────────────────────────────
_SEQ_LEN = 10           # must match config.SEQUENCE_LENGTH
_N_FEATURES = 18        # must match FEATURE_COLUMNS in data_preprocessor.py
_TICK_INTERVAL_S = 1.0  # wall-clock seconds between broadcast ticks

# ── Global state ──────────────────────────────────────────────────────────────
sim_state: dict = {}
sim_history: deque = deque(maxlen=300)
sim_running: bool = False
active_connections: List[WebSocket] = []
start_time: float = time.time()

# Simulation components (set in lifespan)
simulator = None
ensemble = None
ppo_model = None          # raw SB3 PPO model (model.predict)
scaler = None             # fitted StandardScaler
_sim_step_iter: Optional[Iterator[Any]] = None
_bg_task: Optional[asyncio.Task] = None

# Rolling feature buffer for LSTM — stores last _SEQ_LEN feature rows
_feature_buffer: deque = deque(maxlen=_SEQ_LEN)


# ── Feature extraction ────────────────────────────────────────────────────────

def _state_to_feature_row(state: Any) -> np.ndarray:
    """
    Extract the 18 ML features from a SimulationState in the exact column
    order used during training (matches FEATURE_COLUMNS in data_preprocessor.py):

        cell0_load, cell1_load, cell2_load,
        cell0_throughput, cell1_throughput, cell2_throughput,
        cell0_ue_count, cell1_ue_count, cell2_ue_count,
        cell0_avg_sinr, cell1_avg_sinr, cell2_avg_sinr,
        system_throughput, system_avg_sinr, system_avg_latency_ms,
        handover_count, handover_rate, packet_loss_rate
    """
    gs = state.gnb_states
    us = state.ue_states
    n_ue = max(len(us), 1)

    loads = [float(gs[i]["load"]) for i in range(3)]

    tputs = [
        sum(float(u["throughput_mbps"]) for u in us if int(u["serving_gnb_id"]) == i)
        for i in range(3)
    ]
    ue_counts = [
        sum(1 for u in us if int(u["serving_gnb_id"]) == i)
        for i in range(3)
    ]
    sinrs = [
        float(np.mean([float(u["sinr_db"]) for u in us if int(u["serving_gnb_id"]) == i] or [0.0]))
        for i in range(3)
    ]

    system_throughput = sum(float(u["throughput_mbps"]) for u in us)
    system_avg_sinr = float(np.mean([float(u["sinr_db"]) for u in us])) if us else 0.0
    # Approximate latency: 1000 / throughput per UE (ms), clipped
    system_avg_latency = float(np.mean([
        min(1000.0 / max(float(u["throughput_mbps"]), 0.1), 200.0)
        for u in us
    ])) if us else 20.0
    handover_count = float(state.handover_count)
    handover_rate = handover_count / n_ue
    # Packet loss proxy: fraction of UEs below minimum SINR threshold
    packet_loss_rate = float(sum(1 for u in us if float(u["sinr_db"]) < -3.0) / n_ue)

    return np.array([
        *loads,
        *tputs,
        *ue_counts,
        *sinrs,
        system_throughput,
        system_avg_sinr,
        system_avg_latency,
        handover_count,
        handover_rate,
        packet_loss_rate,
    ], dtype=np.float32)


def _get_congestion_predictions() -> dict:
    """
    Run the trained ensemble on the current feature buffer.
    Returns per-cell congestion probability dict.
    Falls back to load-based estimate if ensemble/scaler not ready.
    """
    if ensemble is None or scaler is None or len(_feature_buffer) < _SEQ_LEN:
        # Not enough data yet — return neutral probabilities
        return {"0": 0.1, "1": 0.1, "2": 0.1}

    # Stack buffer into sequence: (seq_len, n_features)
    seq = np.stack(list(_feature_buffer), axis=0)  # (10, 18)

    # Scale flat features (last tick) for XGBoost
    flat_scaled = scaler.transform(seq[-1:, :]).astype(np.float32)  # (1, 18)

    # Scale sequence for LSTM
    seq_scaled = scaler.transform(seq).astype(np.float32)           # (10, 18)
    seq_tensor = torch.from_numpy(seq_scaled).unsqueeze(0)          # (1, 10, 18)

    # Ensemble returns system-level congestion probability
    system_prob = float(ensemble.predict_proba(seq_tensor, flat_scaled)[0])

    # Distribute to per-cell probabilities weighted by load
    # This gives the dashboard meaningful per-cell values
    if sim_state and "cells" in sim_state:
        loads = np.array([c["load_percent"] for c in sim_state["cells"]], dtype=np.float32)
        total = loads.sum()
        if total > 0:
            cell_probs = np.clip((loads / total) * system_prob * 3.0, 0.01, 0.99)
        else:
            cell_probs = np.full(3, system_prob, dtype=np.float32)
    else:
        cell_probs = np.full(3, system_prob, dtype=np.float32)

    return {str(i): round(float(cell_probs[i]), 3) for i in range(3)}


def _get_ppo_actions(state: Any) -> dict:
    """
    Query the PPO policy for an action given current cell loads and UE counts.
    Returns per-cell action dict for the dashboard RL Agent Panel.
    Falls back to zeros if PPO not loaded.
    """
    if ppo_model is None or state is None:
        return {"0": 0, "1": 0, "2": 0}

    gs = state.gnb_states
    us = state.ue_states
    import config as _cfg

    loads = np.clip([float(gs[i]["load"]) for i in range(3)], 0.0, 1.0)

    # Congestion probs from current predictions (already computed this tick)
    cong_vals = list(_get_congestion_predictions().values())
    cong_probs = np.clip(cong_vals, 0.0, 1.0)

    ue_counts = np.array([
        sum(1 for u in us if int(u["serving_gnb_id"]) == i) / max(_cfg.NUM_UE, 1)
        for i in range(3)
    ], dtype=np.float32)

    obs = np.concatenate([loads, cong_probs, ue_counts]).astype(np.float32)
    action, _ = ppo_model.predict(obs, deterministic=True)
    action_int = int(action)

    # Map single system action to per-cell display
    # Action meaning: 0=NoOp, 1=LoadBalance, 2=MassBalance, 3=EmergencyHandover
    return {"0": action_int, "1": action_int, "2": action_int}


# ── Simulation state → tick dict ──────────────────────────────────────────────

def _state_to_tick_dict(state: Any, tick_num: int) -> dict:
    """Convert SimulationState to the WebSocket payload shape."""
    gs = state.gnb_states
    us = state.ue_states

    cells = []
    for i, g in enumerate(gs):
        gid = int(g["gnb_id"])
        cell_tput = sum(
            float(u["throughput_mbps"]) for u in us if int(u["serving_gnb_id"]) == gid
        )
        load = float(g["load"])
        cells.append({
            "cell_id": gid,
            "load_percent": round(min(0.999, max(0.0, load)), 3),
            "throughput_mbps": round(cell_tput, 2),
            "latency_ms": round(15.0 + 85.0 * load, 2),
            "connected_ues": len(g.get("connected_ues", [])),
            "prb_used": int(g.get("allocated_prbs", 0)),
        })

    ues = [
        {
            "ue_id": int(u["ue_id"]),
            "x": round(float(u["position"][0]), 1),
            "y": round(float(u["position"][1]), 1),
            "connected_cell": int(u["serving_gnb_id"]),
            "sinr_db": round(float(u["sinr_db"]), 2),
            "throughput_mbps": round(float(u["throughput_mbps"]), 2),
            "is_handover": bool(u.get("is_handover", False)),
        }
        for u in us
    ]

    total_tput = sum(float(u["throughput_mbps"]) for u in us)
    mean_load = float(np.mean([float(g["load"]) for g in gs])) if gs else 0.0

    # Update global feature buffer for ML inference
    _feature_buffer.append(_state_to_feature_row(state))

    congestion_predictions = _get_congestion_predictions()
    ppo_actions = _get_ppo_actions(state)

    return {
        "tick": int(state.tick),
        "timestamp": time.time(),
        "cells": cells,
        "ues": ues,
        "kpis": {
            "total_throughput": round(total_tput, 2),
            "mean_latency": round(18.0 + 40.0 * mean_load, 2),
            "handover_count": int(state.handover_count),
            "active_ues": len(us),
        },
        "congestion_predictions": congestion_predictions,
        "ppo_actions": ppo_actions,
    }


# ── Real simulation step ───────────────────────────────────────────────────────

def _advance_real_simulation(tick_num: int) -> dict:
    """Advance SimPy engine by one tick and return dashboard payload."""
    global simulator, _sim_step_iter
    import config

    if simulator is None:
        raise RuntimeError("simulator not initialised")

    if _sim_step_iter is None:
        _sim_step_iter = iter(simulator.run(ticks=config.SIM_DURATION_S))

    try:
        state = next(_sim_step_iter)
    except StopIteration:
        # Simulation completed one full run — restart
        logger.info("Simulation completed full run — restarting")
        from simulation.engine import NetworkSimulation
        simulator = NetworkSimulation()
        _sim_step_iter = iter(simulator.run(ticks=config.SIM_DURATION_S))
        _feature_buffer.clear()
        state = next(_sim_step_iter)

    return _state_to_tick_dict(state, tick_num)


# ── Mock tick (fallback when real simulator unavailable) ──────────────────────

def generate_mock_tick(tick_num: int) -> dict:
    t = tick_num * 0.05
    return {
        "tick": tick_num,
        "timestamp": time.time(),
        "cells": [
            {
                "cell_id": i,
                "load_percent": round(
                    min(0.95, max(0.05,
                        0.45 + 0.30 * math.sin(t + i * 2.1) + 0.10 * np.random.randn())),
                    3,
                ),
                "throughput_mbps": round(80 + 20 * math.cos(t + i), 2),
                "latency_ms": round(20 + 10 * abs(math.sin(t + i)), 2),
                "connected_ues": 6 + i,
                "prb_used": 40 + i * 5,
            }
            for i in range(3)
        ],
        "ues": [
            {
                "ue_id": j,
                "x": round(500 + 350 * math.cos(t * 0.3 + j * 0.314), 1),
                "y": round(500 + 350 * math.sin(t * 0.3 + j * 0.314), 1),
                "connected_cell": j % 3,
                "sinr_db": round(15 + 5 * math.sin(t + j), 2),
                "throughput_mbps": round(10 + 5 * math.cos(t + j), 2),
                "is_handover": (tick_num % 50 == 0 and j < 2),
            }
            for j in range(20)
        ],
        "kpis": {
            "total_throughput": round(240 + 40 * math.sin(t), 2),
            "mean_latency": round(25 + 8 * math.cos(t), 2),
            "handover_count": int(abs(math.sin(t * 0.5)) * 3),
            "active_ues": 20,
        },
        "congestion_predictions": {
            "0": round(min(0.99, max(0.01, 0.25 + 0.20 * math.sin(t))), 3),
            "1": round(min(0.99, max(0.01, 0.35 + 0.25 * math.cos(t))), 3),
            "2": round(min(0.99, max(0.01, 0.20 + 0.15 * math.sin(t + 1))), 3),
        },
        "ppo_actions": {
            "0": tick_num % 4,
            "1": (tick_num + 1) % 4,
            "2": (tick_num + 2) % 4,
        },
    }


# ── Pydantic models ───────────────────────────────────────────────────────────

class SimStatus(BaseModel):
    running: bool
    tick: int
    connected_clients: int
    uptime_seconds: float
    mode: str  # "real" or "mock"


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global simulator, ensemble, ppo_model, scaler, _sim_step_iter, sim_running, _bg_task

    # 1. Load simulation engine
    try:
        from simulation.engine import NetworkSimulation
        simulator = NetworkSimulation()
        _sim_step_iter = None
        logger.info("✓ NetworkSimulation initialised")
    except Exception as e:
        logger.warning("✗ Simulator unavailable (%s) — mock mode active", e)
        simulator = None

    # 2. Load scaler
    try:
        scaler_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "scaler.pkl"
        )
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info("✓ Scaler loaded from %s", scaler_path)
    except Exception as e:
        logger.warning("✗ Scaler unavailable (%s)", e)
        scaler = None

    # 3. Load LSTM + XGBoost ensemble
    try:
        import config
        from ml.lstm_model import CongestionLSTM
        from ml.xgboost_model import XGBoostPredictor
        from ml.ensemble import EnsemblePredictor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )

        lstm = CongestionLSTM(
            input_size=_N_FEATURES,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
        )
        
        lstm.load_state_dict(
            torch.load(os.path.join(models_dir, "lstm_best.pt"), map_location=device)
        )
        lstm.eval()

        xgb = XGBoostPredictor()
        xgb.load(os.path.join(models_dir, "xgboost_model.json"))

        ensemble = EnsemblePredictor(
            lstm_model=lstm,
            xgb_predictor=xgb,
            device=device,
            lstm_weight=0.6,
            xgb_weight=0.4,
        )
        logger.info("✓ Ensemble (LSTM + XGBoost) loaded on %s", device)
    except Exception as e:
        logger.warning("✗ Ensemble unavailable (%s) — predictions will be load-based", e)
        ensemble = None

    # 4. Load PPO agent
    try:
        from stable_baselines3 import PPO as SB3PPO

        ppo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "ppo_agent.zip"
        )
        ppo_model = SB3PPO.load(ppo_path, device="cpu")
        logger.info("✓ PPO agent loaded from %s", ppo_path)
    except Exception as e:
        logger.warning("✗ PPO agent unavailable (%s) — actions will be zero", e)
        ppo_model = None

    # 5. Auto-start simulation loop
    sim_running = True
    _bg_task = asyncio.create_task(run_simulation_loop())
    logger.info("✓ Simulation loop auto-started")

    yield  # ── server runs ──

    # Shutdown
    sim_running = False
    if _bg_task and not _bg_task.done():
        _bg_task.cancel()
    logger.info("Shutdown complete")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="5G Digital Twin API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ── Broadcast ─────────────────────────────────────────────────────────────────

async def broadcast_tick(data: dict) -> None:
    dead = []
    for ws in active_connections:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in active_connections:
            active_connections.remove(ws)


# ── Simulation loop ───────────────────────────────────────────────────────────

async def run_simulation_loop() -> None:
    global sim_state, sim_running
    tick = 0
    while sim_running:
        try:
            if simulator is not None:
                tick_data = _advance_real_simulation(tick)
            else:
                tick_data = generate_mock_tick(tick)

            sim_state = tick_data
            sim_history.append(tick_data)
            await broadcast_tick({"type": "tick_update", "payload": tick_data})

            if tick % 10 == 0:
                mode = "real" if simulator is not None else "mock"
                logger.info(
                    "Tick %d [%s] | clients=%d | tput=%.1f Mbps",
                    tick_data.get("tick", tick),
                    mode,
                    len(active_connections),
                    tick_data["kpis"]["total_throughput"],
                )
            tick += 1
            await asyncio.sleep(_TICK_INTERVAL_S)

        except Exception as e:
            logger.error("Sim loop error at tick %d: %s", tick, e, exc_info=True)
            await asyncio.sleep(_TICK_INTERVAL_S)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    active_connections.append(websocket)
    logger.info("WS connected. Total=%d", len(active_connections))

    try:
        # Send recent history to newly connected client
        history_payload = list(sim_history)[-60:]
        await websocket.send_json({"type": "history", "payload": {"ticks": history_payload}})

        # Keep connection alive; server pushes data via broadcast_tick
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WS error: %s", e)
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info("WS disconnected. Total=%d", len(active_connections))


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/api/status", response_model=SimStatus)
async def get_status() -> SimStatus:
    return SimStatus(
        running=sim_running,
        tick=sim_state.get("tick", 0),
        connected_clients=len(active_connections),
        uptime_seconds=round(time.time() - start_time, 1),
        mode="real" if simulator is not None else "mock",
    )


@app.post("/api/simulation/start")
async def start_simulation() -> dict:
    global sim_running, _bg_task
    if sim_running:
        return {"message": "already running"}
    sim_running = True
    _bg_task = asyncio.create_task(run_simulation_loop())
    logger.info("Simulation started via REST")
    return {"message": "started"}


@app.post("/api/simulation/stop")
async def stop_simulation() -> dict:
    global sim_running
    sim_running = False
    logger.info("Simulation stopped via REST")
    return {"message": "stopped"}


@app.get("/api/history")
async def get_history(limit: int = 100) -> dict:
    history = list(sim_history)
    return {"ticks": history[-limit:], "count": len(history)}


@app.get("/api/kpis/summary")
async def get_kpi_summary() -> dict:
    if not sim_history:
        return {"error": "no data yet"}
    ticks = list(sim_history)
    throughputs = [t["kpis"]["total_throughput"] for t in ticks]
    latencies = [t["kpis"]["mean_latency"] for t in ticks]
    handovers = [t["kpis"]["handover_count"] for t in ticks]
    congestion_counts = {str(i): 0 for i in range(3)}
    for t in ticks:
        for cell_id, prob in t.get("congestion_predictions", {}).items():
            if prob > 0.7:
                congestion_counts[str(cell_id)] += 1
    return {
        "mean_throughput_mbps": round(float(np.mean(throughputs)), 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "total_handovers": int(sum(handovers)),
        "ticks_sampled": len(ticks),
        "congestion_events_per_cell": congestion_counts,
    }


@app.get("/api/cells/{cell_id}/metrics")
async def get_cell_metrics(cell_id: int) -> dict:
    if cell_id not in (0, 1, 2):
        return {"error": "cell_id must be 0, 1, or 2"}
    result = [
        {
            "tick": t["tick"],
            "load_percent": c["load_percent"],
            "throughput_mbps": c["throughput_mbps"],
            "latency_ms": c["latency_ms"],
        }
        for t in sim_history
        for c in t.get("cells", [])
        if c["cell_id"] == cell_id
    ]
    return {"cell_id": cell_id, "metrics": result}


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "uptime": round(time.time() - start_time, 1),
        "mode": "real" if simulator is not None else "mock",
        "ensemble_loaded": ensemble is not None,
        "ppo_loaded": ppo_model is not None,
    }
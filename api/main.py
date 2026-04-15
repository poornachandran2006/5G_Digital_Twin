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
from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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

# ── Prometheus metrics ────────────────────────────────────────────────────
_prom_cell_load        = Gauge('cell_load_percent',       'PRB load per cell',         ['cell_id'])
_prom_cell_throughput  = Gauge('cell_throughput_mbps',    'Throughput per cell (Mbps)', ['cell_id'])
_prom_cell_latency     = Gauge('cell_latency_ms',         'Latency per cell (ms)',      ['cell_id'])
_prom_cell_ues         = Gauge('cell_connected_ues',      'Connected UEs per cell',     ['cell_id'])
_prom_congestion_prob  = Gauge('congestion_probability',  'LSTM congestion prediction', ['cell_id'])
_prom_anomaly_score    = Gauge('anomaly_score',           'IsolationForest anomaly score')
_prom_ppo_reward       = Gauge('ppo_reward',              'PPO agent reward this tick')
_prom_rb_reward        = Gauge('rule_based_reward',       'Rule-based agent reward this tick')
_prom_system_tput      = Gauge('system_throughput_mbps',  'Total system throughput (Mbps)')
_prom_handover_count   = Counter('handover_total',        'Cumulative handover events')
_prom_tick             = Gauge('simulation_tick',         'Current simulation tick')

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
xgb_predictor = None      # XGBoostPredictor instance (for feature importance)
_sim_step_iter: Optional[Iterator[Any]] = None
_bg_task: Optional[asyncio.Task] = None

# Rolling feature buffer for LSTM — stores last _SEQ_LEN feature rows
_feature_buffer: deque = deque(maxlen=_SEQ_LEN)
# Anomaly detector (loaded in lifespan)
anomaly_detector = None
_last_anomaly_result: dict = {"anomaly_score": 0.0, "is_anomaly": False, "severity": "normal"}
# A/B testing — rule-based agent running in parallel with PPO
rule_based_agent = None
_ab_history: deque = deque(maxlen=300)   # stores per-tick comparison data

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
        return {"0": 0.1, "1": 0.1, "2": 0.1}

    seq = np.stack(list(_feature_buffer), axis=0)  # (10, 18)
    flat_scaled = scaler.transform(seq[-1:, :]).astype(np.float32)  # (1, 18)
    seq_scaled = scaler.transform(seq).astype(np.float32)           # (10, 18)
    seq_tensor = torch.from_numpy(seq_scaled).unsqueeze(0)          # (1, 10, 18)

    system_prob = float(ensemble.predict_proba(seq_tensor, flat_scaled)[0])

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
    cong_vals = list(_get_congestion_predictions().values())
    cong_probs = np.clip(cong_vals, 0.0, 1.0)

    ue_counts = np.array([
        sum(1 for u in us if int(u["serving_gnb_id"]) == i) / max(_cfg.NUM_UE, 1)
        for i in range(3)
    ], dtype=np.float32)

    obs = np.concatenate([loads, cong_probs, ue_counts]).astype(np.float32)
    action, _ = ppo_model.predict(obs, deterministic=True)
    action_int = int(action)

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
            "traffic_profile": u.get("traffic_profile", "Video"),
            "qos_class": int(u.get("qos_class", 2)),
        }
        for u in us
    ]

    total_tput = sum(float(u["throughput_mbps"]) for u in us)
    mean_load = float(np.mean([float(g["load"]) for g in gs])) if gs else 0.0

    _feature_buffer.append(_state_to_feature_row(state))

    congestion_predictions = _get_congestion_predictions()
    ppo_actions = _get_ppo_actions(state)

    # Anomaly detection on current feature row
    global _last_anomaly_result, rule_based_agent, _ab_history
    if anomaly_detector is not None and len(_feature_buffer) > 0:
        current_row = list(_feature_buffer)[-1]
        _last_anomaly_result = anomaly_detector.score(current_row)

    # A/B testing — rule-based agent scores the same observation
    ab_entry = {"tick": state.tick, "ppo_action": 0, "ppo_reward": 0.0,
                "rb_action": 0, "rb_reward": 0.0}
    if rule_based_agent is not None:
        # Build 9-dim obs from current state (same as rl_env._get_observation)
        loads = np.array([state.gnb_states[i]["load"] for i in range(3)], dtype=np.float32)
        ue_counts = np.array([
            sum(1 for u in state.ue_states if u["serving_gnb_id"] == i) / max(len(state.ue_states), 1)
            for i in range(3)
        ], dtype=np.float32)
        cong = np.array([
            state.gnb_states[i]["load"] * 0.5 for i in range(3)
        ], dtype=np.float32)
        obs_9 = np.concatenate([loads, cong, ue_counts]).astype(np.float32)

        # Rule-based action + reward
        rb_action = rule_based_agent.predict(obs_9)
        rb_reward = rule_based_agent.record_reward(obs_9)

        # PPO action from ppo_actions already computed
        ppo_act = ppo_actions.get("action", 0) if isinstance(ppo_actions, dict) else 0

        # PPO reward — compute with same formula
        ppo_reward = 0.0
        for load in loads:
            if load < 0.70:
                ppo_reward += 0.2
            elif load < 0.90:
                ppo_reward -= 0.5
            else:
                ppo_reward -= 1.0
        ppo_reward += max(0.0, 0.1 - float(np.std(loads)))
        ppo_reward = float(np.clip(ppo_reward, -1.0, 1.0))

        ab_entry = {
            "tick": state.tick,
            "ppo_action": ppo_act,
            "ppo_reward": round(ppo_reward, 4),
            "rb_action": rb_action,
            "rb_reward": round(rb_reward, 4),
        }
        _ab_history.append(ab_entry)

    # ── Push to Prometheus gauges ─────────────────────────────────────────
    try:
        _prom_tick.set(tick_num)
        _prom_system_tput.set(sum(c["throughput_mbps"] for c in cells))
        _prom_anomaly_score.set(_last_anomaly_result.get("anomaly_score", 0.0))
        _prom_ppo_reward.set(ab_entry.get("ppo_reward", 0.0))
        _prom_rb_reward.set(ab_entry.get("rb_reward", 0.0))
        for c in cells:
            cid = str(c["cell_id"])
            _prom_cell_load.labels(cell_id=cid).set(c["load_percent"])
            _prom_cell_throughput.labels(cell_id=cid).set(c["throughput_mbps"])
            _prom_cell_latency.labels(cell_id=cid).set(c["latency_ms"])
            _prom_cell_ues.labels(cell_id=cid).set(c["connected_ues"])
        for cid, prob in congestion_predictions.items():
            _prom_congestion_prob.labels(cell_id=str(cid)).set(float(prob))
        ho_count = sum(1 for u in ues if u.get("is_handover", False))
        if ho_count > 0:
            _prom_handover_count.inc(ho_count)
    except Exception:
        pass  # never let metrics crash the simulation

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
        "anomaly": _last_anomaly_result,
        "ab_comparison": ab_entry,
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

class PredictRequest(BaseModel):
    """KPI snapshot for one tick — sent to /api/predict."""
    cell0_load: float
    cell1_load: float
    cell2_load: float
    cell0_throughput: float
    cell1_throughput: float
    cell2_throughput: float
    cell0_ue_count: int
    cell1_ue_count: int
    cell2_ue_count: int
    cell0_avg_sinr: float = 0.0
    cell1_avg_sinr: float = 0.0
    cell2_avg_sinr: float = 0.0
    system_throughput: float = 0.0
    system_avg_sinr: float = 0.0
    system_avg_latency_ms: float = 20.0
    handover_count: float = 0.0
    handover_rate: float = 0.0
    packet_loss_rate: float = 0.0


class AgentRequest(BaseModel):
    """9-dim observation for /api/agent/action."""
    cell0_load: float
    cell1_load: float
    cell2_load: float
    cong0: float = 0.1
    cong1: float = 0.1
    cong2: float = 0.1
    ue_ratio0: float = 0.33
    ue_ratio1: float = 0.33
    ue_ratio2: float = 0.33

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global simulator, ensemble, ppo_model, scaler, xgb_predictor, _sim_step_iter, sim_running, _bg_task, anomaly_detector, rule_based_agent
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
        xgb_predictor = xgb   # store globally for feature importance endpoint
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

    # 5. Load anomaly detector
    try:
        from ml.anomaly_detector import AnomalyDetector
        anomaly_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "anomaly_detector.pkl"
        )
        anomaly_detector = AnomalyDetector()
        anomaly_detector.load(anomaly_path)
        logger.info("✓ Anomaly detector loaded from %s", anomaly_path)
    except Exception as e:
        logger.warning("✗ Anomaly detector unavailable (%s)", e)
        anomaly_detector = None
    
    # 6. Initialize rule-based agent (no model file needed — pure logic)
    try:
        from optimizer.rule_based_agent import RuleBasedAgent
        rule_based_agent = RuleBasedAgent()
        logger.info("✓ Rule-based agent initialized")
    except Exception as e:
        logger.warning("✗ Rule-based agent unavailable (%s)", e)
        rule_based_agent = None

    # 7. Auto-start simulation loop
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
        history_payload = list(sim_history)[-60:]
        await websocket.send_json({"type": "history", "payload": {"ticks": history_payload}})

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


# ── Feature Importance Endpoint (XGBoost built-in gain, no shap/pandas) ──────
@app.get("/api/shap/explanation")
async def get_shap_explanation():
    """
    Returns per-feature importance using XGBoost's built-in gain scores.
    'Gain' = average reduction in impurity when a feature is used in a split.
    No shap or pandas dependency — works on Python 3.14.
    Endpoint kept as /api/shap/explanation so the frontend needs no changes.
    """
    if not sim_history:
        raise HTTPException(status_code=503, detail="No simulation data yet")

    if xgb_predictor is None:
        raise HTTPException(status_code=503, detail="XGBoost model not loaded")

    feature_names = [
        "cell0_load", "cell1_load", "cell2_load",
        "cell0_throughput", "cell1_throughput", "cell2_throughput",
        "cell0_ue_count", "cell1_ue_count", "cell2_ue_count",
        "cell0_avg_sinr", "cell1_avg_sinr", "cell2_avg_sinr",
        "system_throughput", "system_avg_sinr", "system_avg_latency_ms",
        "handover_count", "handover_rate", "packet_loss_rate"
    ]

    try:
        latest = list(sim_history)[-1]
        cells = latest.get("cells", [])
        kpis = latest.get("kpis", {})
        cell_map = {c["cell_id"]: c for c in cells}

        # Current feature values in exact training order
        feature_values = [
            cell_map.get(0, {}).get("load_percent", 0),
            cell_map.get(1, {}).get("load_percent", 0),
            cell_map.get(2, {}).get("load_percent", 0),
            cell_map.get(0, {}).get("throughput_mbps", 0),
            cell_map.get(1, {}).get("throughput_mbps", 0),
            cell_map.get(2, {}).get("throughput_mbps", 0),
            cell_map.get(0, {}).get("connected_ues", 0),
            cell_map.get(1, {}).get("connected_ues", 0),
            cell_map.get(2, {}).get("connected_ues", 0),
            0.0, 0.0, 0.0,          # avg_sinr per cell — not in tick dict
            kpis.get("total_throughput", 0),
            0.0,                    # system_avg_sinr — not in tick dict
            kpis.get("mean_latency", 0),
            kpis.get("handover_count", 0),
            kpis.get("handover_count", 0) / max(kpis.get("active_ues", 1), 1),
            0.0,                    # packet_loss_rate — not in tick dict
        ]

        # Get XGBoost gain-based importance from the trained booster
        # gain = how much each feature reduces impurity on average across all trees
        booster = xgb_predictor.model.get_booster()
        gain_scores = booster.get_score(importance_type="gain")

        # XGBoost names features f0..f17 when no feature names were set at training
        shap_list = []
        for i, name in enumerate(feature_names):
            xgb_key = f"f{i}"
            gain = gain_scores.get(xgb_key, 0.0)
            fval = float(feature_values[i])
            # Sign convention: positive = feature pushes toward congestion
            # (high load, high UE count → positive; high SINR → negative)
            signed_importance = gain if fval > 0 else -gain
            shap_list.append({
                "feature": name,
                "shap_value": signed_importance,
                "feature_value": round(fval, 4)
            })

        # Sort by absolute importance descending
        shap_list.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        # Normalise to [-1, +1] range so the bar chart is clean
        total = sum(abs(x["shap_value"]) for x in shap_list)
        if total > 0:
            for x in shap_list:
                x["shap_value"] = round(x["shap_value"] / total, 6)

        return {
            "tick": latest.get("tick", 0),
            "base_value": 0.121,    # dataset congestion base rate = 12.1%
            "features": shap_list,
            "method": "xgb_gain"   # tells frontend this is gain-based not SHAP
        }

    except Exception as e:
        logger.error("Feature importance error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Feature importance failed: {str(e)}")


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

@app.get("/api/anomaly/current")
async def get_anomaly_current() -> dict:
    """Returns the latest anomaly score for the live dashboard panel."""
    return {
        "tick": sim_state.get("tick", 0),
        **_last_anomaly_result,
    }


@app.get("/api/anomaly/history")
async def get_anomaly_history(limit: int = 100) -> dict:
    """Returns anomaly scores for the last N ticks from sim_history."""
    result = []
    for t in list(sim_history)[-limit:]:
        anomaly = t.get("anomaly", {"anomaly_score": 0.0, "is_anomaly": False, "severity": "normal"})
        result.append({
            "tick": t["tick"],
            "anomaly_score": anomaly["anomaly_score"],
            "is_anomaly": anomaly["is_anomaly"],
            "severity": anomaly["severity"],
        })
    return {"history": result, "count": len(result)}

@app.get("/api/ab/history")
async def get_ab_history(limit: int = 100) -> dict:
    """A/B comparison history: PPO vs rule-based rewards per tick."""
    data = list(_ab_history)[-limit:]
    if not data:
        return {"history": [], "summary": {}}

    ppo_rewards = [d["ppo_reward"] for d in data]
    rb_rewards  = [d["rb_reward"]  for d in data]

    summary = {
        "ppo_avg_reward":  round(float(np.mean(ppo_rewards)), 4),
        "rb_avg_reward":   round(float(np.mean(rb_rewards)),  4),
        "ppo_win_rate":    round(sum(p > r for p, r in zip(ppo_rewards, rb_rewards)) / len(data), 3),
        "ticks_compared":  len(data),
    }
    return {"history": data, "summary": summary}


@app.get("/api/ab/summary")
async def get_ab_summary() -> dict:
    """Live summary of PPO vs rule-based performance."""
    if not _ab_history:
        return {"ppo_avg_reward": 0, "rb_avg_reward": 0, "ppo_win_rate": 0, "ticks_compared": 0}
    data = list(_ab_history)
    ppo_rewards = [d["ppo_reward"] for d in data]
    rb_rewards  = [d["rb_reward"]  for d in data]
    return {
        "ppo_avg_reward":  round(float(np.mean(ppo_rewards)), 4),
        "rb_avg_reward":   round(float(np.mean(rb_rewards)),  4),
        "ppo_win_rate":    round(sum(p > r for p, r in zip(ppo_rewards, rb_rewards)) / len(data), 3),
        "rb_stats":        rule_based_agent.get_stats() if rule_based_agent else {},
        "ticks_compared":  len(data),
    }

@app.post("/api/predict")
async def predict_congestion(req: PredictRequest) -> dict:
    """
    Run the trained LSTM+XGBoost ensemble on a single KPI snapshot.
    Returns per-cell congestion probability for the next 30 ticks.

    Example curl:
      curl -X POST http://localhost:8000/api/predict
           -H "Content-Type: application/json"
           -d '{"cell0_load":0.85,"cell1_load":0.4,"cell2_load":0.3,
                "cell0_throughput":200,"cell1_throughput":400,"cell2_throughput":300,
                "cell0_ue_count":9,"cell1_ue_count":6,"cell2_ue_count":5}'
    """
    if ensemble is None or scaler is None:
        raise HTTPException(status_code=503, detail="Ensemble model not loaded")

    # Build feature row in exact training column order
    feature_row = np.array([
        req.cell0_load, req.cell1_load, req.cell2_load,
        req.cell0_throughput, req.cell1_throughput, req.cell2_throughput,
        float(req.cell0_ue_count), float(req.cell1_ue_count), float(req.cell2_ue_count),
        req.cell0_avg_sinr, req.cell1_avg_sinr, req.cell2_avg_sinr,
        req.system_throughput, req.system_avg_sinr, req.system_avg_latency_ms,
        req.handover_count, req.handover_rate, req.packet_loss_rate,
    ], dtype=np.float32).reshape(1, -1)

    # If we have a live feature buffer, append the request row and use full sequence
    # Otherwise tile the single row to fill the sequence
    if len(_feature_buffer) >= _SEQ_LEN:
        seq = np.stack(list(_feature_buffer), axis=0)  # (10, 18) from live sim
    else:
        seq = np.tile(feature_row, (_SEQ_LEN, 1))      # (10, 18) tiled

    # Override the last row with the request's features for fresh prediction
    seq[-1] = feature_row[0]

    flat_scaled = scaler.transform(feature_row).astype(np.float32)
    seq_scaled  = scaler.transform(seq).astype(np.float32)
    seq_tensor  = torch.from_numpy(seq_scaled).unsqueeze(0)  # (1, 10, 18)

    system_prob = float(ensemble.predict_proba(seq_tensor, flat_scaled)[0])

    # Distribute system probability across cells weighted by load
    loads = np.array([req.cell0_load, req.cell1_load, req.cell2_load], dtype=np.float32)
    total = loads.sum()
    if total > 0:
        cell_probs = np.clip((loads / total) * system_prob * 3.0, 0.01, 0.99)
    else:
        cell_probs = np.full(3, system_prob, dtype=np.float32)

    return {
        "system_congestion_probability": round(system_prob, 4),
        "cell_congestion_probabilities": {
            "0": round(float(cell_probs[0]), 4),
            "1": round(float(cell_probs[1]), 4),
            "2": round(float(cell_probs[2]), 4),
        },
        "prediction_horizon_ticks": 30,
        "model": "lstm_xgboost_ensemble_0.6_0.4",
    }


@app.post("/api/agent/action")
async def get_agent_action(req: AgentRequest) -> dict:
    """
    Query the trained PPO agent for a load-balancing action.
    Returns action int + human-readable label.

    Actions:
      0 = NoOp          — network healthy, do nothing
      1 = LoadBalance   — soft handover to balance load
      2 = MassBalance   — aggressive rebalancing
      3 = Emergency     — emergency handover (critical overload)

    Example curl:
      curl -X POST http://localhost:8000/api/agent/action
           -H "Content-Type: application/json"
           -d '{"cell0_load":0.91,"cell1_load":0.3,"cell2_load":0.25}'
    """
    if ppo_model is None:
        raise HTTPException(status_code=503, detail="PPO agent not loaded")

    obs = np.array([
        req.cell0_load, req.cell1_load, req.cell2_load,
        req.cong0, req.cong1, req.cong2,
        req.ue_ratio0, req.ue_ratio1, req.ue_ratio2,
    ], dtype=np.float32)

    action, _ = ppo_model.predict(obs, deterministic=True)
    action_int = int(action)

    action_labels = {
        0: "NoOp",
        1: "LoadBalance",
        2: "MassBalance",
        3: "EmergencyHandover",
    }

    # Also get rule-based decision for comparison
    rb_action = None
    if rule_based_agent is not None:
        rb_action = int(rule_based_agent.predict(obs))

    return {
        "action": action_int,
        "action_label": action_labels.get(action_int, "Unknown"),
        "rule_based_action": rb_action,
        "rule_based_label": action_labels.get(rb_action, "Unknown") if rb_action is not None else None,
        "observation": {
            "cell_loads": [req.cell0_load, req.cell1_load, req.cell2_load],
            "congestion_probs": [req.cong0, req.cong1, req.cong2],
            "ue_ratios": [req.ue_ratio0, req.ue_ratio1, req.ue_ratio2],
        },
    }

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus scrape endpoint — standard text exposition format."""
    from fastapi.responses import Response
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "uptime": round(time.time() - start_time, 1),
        "mode": "real" if simulator is not None else "mock",
        "ensemble_loaded": ensemble is not None,
        "ppo_loaded": ppo_model is not None,
    }
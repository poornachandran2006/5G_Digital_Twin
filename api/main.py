"""
5G Network Digital Twin — FastAPI Backend
WebSocket + REST API for live dashboard
"""

import asyncio
import logging
import time
import math
import os
import sys
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Iterator, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("api")

# ── Global state ──────────────────────────────────────────────────────────────
sim_state: dict = {}
sim_history: deque = deque(maxlen=300)
sim_running: bool = False
active_connections: List[WebSocket] = []
start_time: float = time.time()
simulator = None
_bg_task: Optional[asyncio.Task] = None
_sim_step_iter: Optional[Iterator[Any]] = None


def _state_to_tick_dict(state: Any) -> dict:
    """Map SimulationState to the same payload shape as generate_mock_tick."""
    cells: list[dict] = []
    for gs in state.gnb_states:
        gid = int(gs["gnb_id"])
        load = float(gs["load"])
        cell_tput = sum(
            float(u["throughput_mbps"])
            for u in state.ue_states
            if int(u["serving_gnb_id"]) == gid
        )
        n_conn = len(gs.get("connected_ues", []))
        cells.append({
            "cell_id": gid,
            "load_percent": round(min(0.999, max(0.0, load)), 3),
            "throughput_mbps": round(cell_tput, 2),
            "latency_ms": round(15.0 + 85.0 * load, 2),
            "connected_ues": n_conn,
            "prb_used": int(gs.get("allocated_prbs", 0)),
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
        for u in state.ue_states
    ]

    total_tput = sum(float(u["throughput_mbps"]) for u in state.ue_states)
    mean_load = float(np.mean([float(g["load"]) for g in state.gnb_states])) if state.gnb_states else 0.0

    congestion_predictions = {
        str(int(gs["gnb_id"])): round(min(0.99, max(0.01, float(gs["load"]))), 3)
        for gs in state.gnb_states
    }

    return {
        "tick": int(state.tick),
        "timestamp": time.time(),
        "cells": cells,
        "ues": ues,
        "kpis": {
            "total_throughput": round(total_tput, 2),
            "mean_latency": round(18.0 + 40.0 * mean_load, 2),
            "handover_count": int(state.handover_count),
            "active_ues": len(state.ue_states),
        },
        "congestion_predictions": congestion_predictions,
        "ppo_actions": {"0": 0, "1": 0, "2": 0},
    }


def _advance_real_simulation() -> dict:
    """Advance :class:`NetworkSimulation` by one tick (SimPy), return dashboard dict."""
    global simulator, _sim_step_iter
    import config
    from simulation.engine import NetworkSimulation

    if simulator is None:
        raise RuntimeError("simulator not initialised")

    if _sim_step_iter is None:
        _sim_step_iter = iter(simulator.run(ticks=config.SIM_DURATION_S))
    try:
        state = next(_sim_step_iter)
    except StopIteration:
        simulator = NetworkSimulation()
        _sim_step_iter = iter(simulator.run(ticks=config.SIM_DURATION_S))
        state = next(_sim_step_iter)
    return _state_to_tick_dict(state)


# ── Pydantic models ───────────────────────────────────────────────────────────
class SimStatus(BaseModel):
    running: bool
    tick: int
    connected_clients: int
    uptime_seconds: float


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global simulator, _sim_step_iter
    try:
        from simulation.engine import NetworkSimulation

        simulator = NetworkSimulation()
        _sim_step_iter = None
        logger.info("NetworkSimulation initialised")
    except Exception as e:
        logger.warning("Simulator unavailable (%s) — mock mode active", e)
        simulator = None
        _sim_step_iter = None
    yield
    global sim_running
    sim_running = False
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


# ── Mock tick (used when real simulator unavailable) ──────────────────────────
def generate_mock_tick(tick_num: int) -> dict:
    t = tick_num * 0.05
    return {
        "tick": tick_num,
        "timestamp": time.time(),
        "cells": [
            {
                "cell_id": i,
                "load_percent": round(
                    min(
                        0.95,
                        max(
                            0.05,
                            0.45
                            + 0.30 * math.sin(t + i * 2.1)
                            + 0.10 * np.random.randn(),
                        ),
                    ),
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


# ── Broadcast ─────────────────────────────────────────────────────────────────
async def broadcast_tick(data: dict):
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
async def run_simulation_loop():
    global sim_state, sim_running
    tick = 0
    while sim_running:
        try:
            if simulator is not None:
                tick_data = _advance_real_simulation()
            else:
                tick_data = generate_mock_tick(tick)

            sim_state = tick_data
            sim_history.append(tick_data)
            await broadcast_tick({"type": "tick_update", "payload": tick_data})

            if tick % 10 == 0:
                logger.info("Tick %s | clients=%d", tick_data.get("tick", tick), len(active_connections))
            tick += 1
            await asyncio.sleep(1.0)
        except Exception as e:
            logger.error("Sim loop error at tick %s: %s", tick, e)
            await asyncio.sleep(1.0)


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
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
async def get_status():
    return SimStatus(
        running=sim_running,
        tick=sim_state.get("tick", 0),
        connected_clients=len(active_connections),
        uptime_seconds=round(time.time() - start_time, 1),
    )


@app.post("/api/simulation/start")
async def start_simulation():
    global sim_running, _bg_task
    if sim_running:
        return {"message": "already running"}
    sim_running = True
    _bg_task = asyncio.create_task(run_simulation_loop())
    logger.info("Simulation started")
    return {"message": "started"}


@app.post("/api/simulation/stop")
async def stop_simulation():
    global sim_running
    sim_running = False
    logger.info("Simulation stopped")
    return {"message": "stopped"}


@app.get("/api/history")
async def get_history(limit: int = 100):
    history = list(sim_history)
    return {"ticks": history[-limit:], "count": len(history)}


@app.get("/api/kpis/summary")
async def get_kpi_summary():
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
async def get_cell_metrics(cell_id: int):
    if cell_id not in (0, 1, 2):
        return {"error": "cell_id must be 0, 1, or 2"}
    result = []
    for t in sim_history:
        cells = t.get("cells", [])
        match = next((c for c in cells if c["cell_id"] == cell_id), None)
        if match:
            result.append({
                "tick": t["tick"],
                "load_percent": match["load_percent"],
                "throughput_mbps": match["throughput_mbps"],
                "latency_ms": match["latency_ms"],
            })
    return {"cell_id": cell_id, "metrics": result}


@app.get("/health")
async def health():
    return {"status": "ok", "uptime": round(time.time() - start_time, 1)}

# 5G Network Digital Twin

> A physics-accurate, AI-augmented software twin of a 5G NR urban cellular network — simulating real SINR propagation, predicting congestion 30 seconds ahead using an LSTM+XGBoost ensemble, and autonomously optimizing load balancing with a PPO reinforcement learning agent.

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Dashboard (Vite)                   │
│  Overview │ Network Map (D3) │ KPIs │ Predictions │ RL Agent│
└──────────────────────┬──────────────────────────────────────┘
                       │  WebSocket  ws://localhost:8000/ws/simulation
┌──────────────────────▼──────────────────────────────────────┐
│                  FastAPI Backend                            │
│  Auto-starts simulation loop on server start                │
│  Real-time broadcast at 1 tick/second                       │
└────┬──────────────────┬────────────────────┬────────────────┘
     │                  │                    │
┌────▼────────┐  ┌──────▼──────┐  ┌─────────▼──────────┐
│ simulation/ │  │    ml/       │  │    optimizer/     │
│ SimPy + NumPy│  │ LSTM + XGB  │  │  PPO (SB3)        │
│ 3 gNB 20 UE │  │ SHAP expla. │  │  Gymnasium env     │
│ SINR physics│  │ Ensemble 0.6/0.4│ │  200k steps trained│
└─────────────┘  └─────────────┘  └────────────────────┘
```

---

## Key Results

| Metric | Value |
|---|---|
| Dataset rows | 10,800 |
| LSTM F1 score | 0.829 |
| Ensemble F1 score | **0.871** |
| Ensemble AUC-ROC | **0.984** |
| Ensemble Precision | 1.0 |
| PPO training steps | 200,000 |
| PPO action diversity | NoOp 17% / LoadBalance 19% / PowerCtrl 37% / Handover 28% |
| Total tests passing | 24 |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Simulation | Python 3.11, SimPy 4, NumPy (zero for-loops, fully vectorised) |
| ML — Sequential | PyTorch LSTM (seq_len=10, horizon=30 ticks) |
| ML — Tabular | XGBoost with SHAP feature importance |
| ML — Ensemble | Weighted blend: 60% LSTM + 40% XGBoost |
| RL | Stable-Baselines3 PPO, Gymnasium environment |
| Backend | FastAPI, WebSocket, SQLite |
| Frontend | React 18, D3.js (live network map), Recharts, TailwindCSS |
| DevOps | Docker, Docker Compose, AWS EC2, NGINX |

---

## Project Structure

```
5G_Digital_Twin/
├── config.py                    # All constants: grid, gNB, UE, ML params
├── requirements.txt
│
├── simulation/                  # Phase 2 — Physics engine
│   ├── engine.py                # SimPy orchestrator, RL override API
│   ├── gnb.py                   # gNB base station model
│   ├── ue.py                    # UE mobile device model
│   ├── channel.py               # Vectorised SINR + path loss (NumPy)
│   └── mobility.py              # Random Waypoint mobility
│
├── kpi/                         # Phase 3 — KPI engine
│   ├── calculator.py            # Per-tick KPI computation
│   ├── data_generator.py        # 10,800-row dataset with congestion injection
│   └── storage.py               # SQLite persistence
│
├── ml/                          # Phase 4 — ML models
│   ├── lstm_model.py            # PyTorch LSTM (BCEWithLogitsLoss)
│   ├── xgboost_model.py         # XGBoost classifier
│   ├── ensemble.py              # Weighted ensemble predictor
│   ├── shap_explainer.py        # SHAP feature importance
│   ├── data_preprocessor.py     # Feature engineering, scaling, splitting
│   └── train.py                 # Training pipeline
│
├── optimizer/                   # Phase 5 — RL agent
│   ├── rl_env.py                # Gymnasium NetworkOptimizationEnv
│   └── agent.py                 # PPOAgent wrapper (SB3)
│
├── api/                         # Phase 6 — Backend
│   ├── main.py                  # FastAPI WebSocket + REST (real engine wired)
│   └── run.py                   # uvicorn launcher
│
├── dashboard/                   # Phase 6 — Frontend
│   └── src/
│       ├── context/SimContext.jsx
│       ├── hooks/useSimSocket.js
│       └── components/panels/
│           ├── OverviewPanel.jsx
│           ├── NetworkMapPanel.jsx   # D3 live UE positions
│           ├── KPIPanel.jsx
│           ├── PredictionPanel.jsx
│           └── RLAgentPanel.jsx
│
├── models/                      # Trained artifacts
│   ├── lstm_best.pt
│   ├── xgboost_model.json
│   ├── scaler.pkl
│   └── ppo_agent.zip
│
├── data/
│   ├── kpi_dataset.csv          # 10,800-row training dataset
│   └── kpi_data.db
│
├── reports/
│   ├── phase4_results.json
│   ├── phase5_results.json
│   └── shap_summary.png
│
├── scripts/
│   ├── train_models.py
│   ├── train_rl_agent.py
│   └── evaluate_ppo.py
│
└── tests/                       # 24 tests passing
    ├── test_simulation.py
    ├── test_kpi.py
    ├── test_phase4.py
    └── test_phase5.py
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+

### 1. Clone & install Python dependencies

```bash
git clone https://github.com/poornachandran2006/5G_Digital_Twin.git
cd 5G_Digital_Twin
pip install -r requirements.txt
```

### 2. Start the FastAPI backend

```bash
cd api
python run.py
# Server starts at http://localhost:8000
# Simulation auto-starts — no manual trigger needed
```

### 3. Start the React dashboard

```bash
cd dashboard
npm install
npm run dev
# Dashboard available at http://localhost:5173
```

### 4. Run all tests

```bash
PYTHONPATH="." python -m pytest tests/ -v
```

---

## How It Works

### Physics Engine

The simulation runs a 1km × 1km urban grid with 3 gNB base stations and 20 mobile UEs. SINR is computed at every tick using:

```
SINR_i = P_tx + G_ant - PathLoss(d_i) - 10·log10(Σ_j≠i 10^((P_tx + G_ant - PathLoss(d_j))/10) + N₀)
```

where path loss uses exponent 3.5 (dense urban, 3GPP TR 38.901) and f = 3.5 GHz. UE mobility follows the Random Waypoint model. All computation is vectorised with NumPy — zero Python for-loops in the hot path.

### Congestion Prediction

A sequence of 10 ticks of 18 KPI features is fed to both an LSTM (PyTorch) and XGBoost. Their probability outputs are blended 60/40:

```
P_congestion = 0.6 × σ(LSTM_logit) + 0.4 × XGBoost_proba
```

The ensemble predicts congestion 30 ticks (30 seconds) ahead. SHAP values identify the 3 most important features: `cell_load`, `system_throughput`, and `packet_loss_rate`.

### RL Load Balancer

A PPO agent (Stable-Baselines3) observes a 9-dimensional state: [cell_loads × 3, congestion_probs × 3, UE_counts × 3] and chooses from 4 discrete actions:

| Action | Behaviour |
|---|---|
| 0 | NoOp — monitor only |
| 1 | LoadBalance — move 1 UE from most to least loaded cell |
| 2 | MassBalance — move up to 3 UEs from overloaded cells |
| 3 | EmergencyHandover — evacuate all UEs off critically loaded cells |

The agent was trained for 200,000 steps with PPO (γ=0.99, lr=3e-4) and achieves a diverse action distribution, confirming it learned non-trivial policies.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/ws/simulation` | WS | Live tick stream (1 tick/sec) |
| `/api/status` | GET | Running state, tick count, mode |
| `/api/history?limit=N` | GET | Last N tick payloads |
| `/api/kpis/summary` | GET | Aggregated KPI statistics |
| `/api/cells/{id}/metrics` | GET | Per-cell time-series metrics |
| `/api/simulation/start` | POST | Start simulation loop |
| `/api/simulation/stop` | POST | Stop simulation loop |
| `/health` | GET | Health check with component status |

---

## Dashboard Panels

| Panel | What it shows |
|---|---|
| Overview | Real-time KPI cards: throughput, latency, handovers, active UEs |
| Network Map | Live D3 canvas — UE positions update every tick, colour-coded by serving cell, red ring on handover |
| KPIs | Time-series charts for cell load, throughput, latency |
| Predictions | Per-cell congestion probability from the LSTM+XGBoost ensemble |
| RL Agent | PPO action history and reward trend |

---

## Author

**Poornachandran** — ECE Engineering Student  

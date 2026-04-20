<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00d4ff,100:8b5cf6&height=120&section=header&text=5G%20Network%20Digital%20Twin&fontSize=36&fontColor=ffffff&fontAlignY=40&desc=Physics-Based%20Simulation%20%E2%80%A2%20ML%20Congestion%20Prediction%20%E2%80%A2%20RL%20Optimization&descSize=14&descAlignY=65" />

<br/>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-WebSocket-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/AWS-EC2-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Ensemble%20AUC-0.984-00d4aa?style=flat-square&logo=checkmarx&logoColor=white"/>
  <img src="https://img.shields.io/badge/Ensemble%20F1-0.871-00d4aa?style=flat-square"/>
  <img src="https://img.shields.io/badge/Precision-1.000%20(zero%20FP)-00d4aa?style=flat-square"/>
  <img src="https://img.shields.io/badge/PPO%20Steps-200K-8b5cf6?style=flat-square"/>
  <img src="https://img.shields.io/badge/Simulation%20Ticks-10%2C800-8b5cf6?style=flat-square"/>
  <img src="https://img.shields.io/badge/Tests%20Passing-24-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
</p>

<br/>

> **A software-only digital twin of a real urban 5G cellular network.**
>
> Physics-based simulation → real-time KPI streaming → ML congestion prediction → RL load balancing → live dashboard.
> Every component is grounded in actual **3GPP standards** and production engineering practices.

<br/>

| 📡 **3 gNB Base Stations** | 📱 **20 Mobile UEs** | 🧠 **AUC 0.984** | 🤖 **PPO · 200K Steps** | ⚡ **30s Early Warning** |
|:---:|:---:|:---:|:---:|:---:|
| SimPy discrete-event | Random Waypoint mobility | LSTM + XGBoost Ensemble | Outperforms rule-based | Predicts congestion ahead |

<br/>

</div>

## 🌐 Live Demo
**Dashboard:** http://3.110.210.142  
**API Health:** http://3.110.210.142:8000/health

<img width="1919" height="854" alt="image" src="https://github.com/user-attachments/assets/c45e8be3-5911-4c6d-b4cd-6ffb93de0a29" />


---

## 📋 Table of Contents

1. [What Is a Digital Twin?](#-what-is-a-digital-twin)
2. [Project Overview](#-project-overview)
3. [System Architecture](#-system-architecture)
4. [Complete Data Flow](#-complete-data-flow)
5. [Feature Highlights](#-feature-highlights)
6. [Tech Stack](#-tech-stack)
7. [Project Structure](#-project-structure)
8. [Core Components — Deep Dive](#-core-components--deep-dive)
   - [1 · 5G Network Simulation Engine](#1--5g-network-simulation-engine)
   - [2 · KPI Engine & Data Generation](#2--kpi-engine--data-generation)
   - [3 · ML Congestion Prediction](#3--ml-congestion-prediction--lstm--xgboost-ensemble)
   - [4 · Reinforcement Learning Agent (PPO)](#4--reinforcement-learning-agent-ppo)
   - [5 · Anomaly Detection (IsolationForest)](#5--anomaly-detection-isolationforest)
   - [6 · A/B Testing Framework](#6--ab-testing-framework)
   - [7 · FastAPI Backend & WebSocket](#7--fastapi-backend--websocket)
   - [8 · React Live Dashboard](#8--react-live-dashboard)
9. [Key Metrics & Results](#-key-metrics--results)
10. [API Reference](#-api-reference)
11. [Installation & Setup](#-installation--setup)
12. [How to Run](#-how-to-run)
13. [Dashboard Walkthrough](#-dashboard-walkthrough)
14. [5G Physics — The Science Behind It](#-5g-physics--the-science-behind-it)
15. [ML Architecture — How the Models Work](#-ml-architecture--how-the-models-work)
16. [RL Environment — How the Agent Learns](#-rl-environment--how-the-agent-learns)
17. [Interview Q&A — What Recruiters Will Ask](#-interview-qa--what-recruiters-will-ask)
18. [Deployment](#-deployment)
19. [Roadmap](#-roadmap)
20. [Author](#-author)

---

## 🌐 What Is a Digital Twin?

A **Digital Twin** is a live software replica of a physical system that mirrors its real-world behaviour in real time. Originally developed for industrial machinery (GE, Siemens), digital twins are now a cornerstone of modern telecom engineering — used by Ericsson, Nokia, and Huawei to model, predict, and optimise live networks without touching hardware.

In the context of **5G networks**, every physical component has a software counterpart:

| Physical World | This Digital Twin |
|---|---|
| 3 real gNB base stations in a city | 3 simulated gNBs at fixed grid positions |
| 20 mobile users walking around | 20 UEs with Random Waypoint mobility |
| Radio signals attenuating over distance | Path loss model: `PL = 20·log(d) + 20·log(f) + 92.4 + 10·3.5·log(d/1000)` |
| Network engineers monitoring dashboards | React dashboard with live WebSocket feed |
| NOC predicting congestion before it happens | LSTM+XGBoost ensemble predicting 30 seconds ahead |
| Network automatically rebalancing load | PPO reinforcement learning agent issuing handover commands |

This project is **100% software** — no hardware required. The entire 5G network exists in Python memory, simulated tick by tick using SimPy's discrete-event simulation engine.

---

## 🎯 Project Overview

This project builds every layer of a production-grade telecom intelligence system from scratch:

```
INPUT                  PROCESSING                          OUTPUT
──────                 ──────────                          ──────

Physics config    →    SimPy simulation engine         →   Per-tick network state
(3.5 GHz, 43 dBm)     (10,800 ticks, 1s each)             (SINR, load, UE positions)
                            │
                            ▼
                       KPI Calculator                   →   10,800-row dataset
                       (load, throughput,                   (22 columns, labelled)
                        latency, handovers)
                            │
                  ┌─────────┼──────────────┐
                  ▼         ▼              ▼
              LSTM       XGBoost      IsolationForest →   Congestion prob (0–1)
              (seq=10)   (n=300)      (unsupervised)  →   Anomaly score
                  └─────────┘
                       Ensemble (0.6 + 0.4)
                            │
                            ▼
                    PPO Agent (200K steps)              →   Action: NoOp /
                    Rule-Based Agent (A/B)                  LoadBalance /
                                                            MassBalance /
                                                            EmergencyHandover
                            │
                            ▼
                    FastAPI + WebSocket                  →   1 tick/second broadcast
                    REST endpoints                           to all dashboard clients
                    Prometheus /metrics
                            │
                            ▼
                    React Dashboard (8 panels)          →   Live NOC-style interface
                    D3.js · Recharts · TailwindCSS          with dark/light theme
```

### What Makes This Project Stand Out

| Property | Detail |
|---|---|
| **Real 5G physics** | SINR from first principles using the 3GPP UMa path loss model at 3.5 GHz |
| **3GPP-compliant traffic** | UEs assigned Video/Gaming/IoT/VoIP profiles with 5QI values from 3GPP TS 23.501 |
| **Production ML pipeline** | Not just a model — full ensemble with preprocessing, sequence buffering, live inference |
| **RL that actually works** | PPO trained 200K steps on a custom Gymnasium env, beats rule-based in live A/B tests |
| **Observability built in** | Prometheus `/metrics` exposes 11 live gauges; any Grafana instance connects instantly |
| **Historical replay** | Record any time window, replay at 1×/2×/4×/8× speed across all 8 panels simultaneously |
| **Zero false positives** | Ensemble Precision = 1.000 — every congestion alarm is a real event |

---

## 🏗 System Architecture

The project is organised into **five distinct layers**, each with a single responsibility:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SIMULATION LAYER                             │
│                                                                     │
│   engine.py ──► gnb.py        ue.py        channel.py               │
│   (SimPy loop)  (3 gNBs,      (20 UEs,     (Vectorized SINR         │
│   10,800 ticks  43 dBm TX,    RWP mobility, path loss 3.5 GHz,      │
│   1s per tick)  100 PRBs)     5QI profiles) exponent = 3.5)         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ per-tick state
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          KPI LAYER                                  │
│                                                                     │
│   calculator.py: cell_load · throughput · latency · handover_rate   │
│   data_generator.py: 10,800 rows with injected congestion events    │
│   storage.py: SQLite persistence                                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ 10,800-row labelled dataset
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           ML LAYER                                  │
│                                                                     │
│   lstm_model.py        xgboost_model.py      ensemble.py            │
│   input: (1, 10, 18)   input: (1, 18)        0.6 × LSTM             │
│   hidden: 64           n_estimators: 300   + 0.4 × XGBoost          │
│   layers: 2            max_depth: 6          F1=0.871 AUC=0.984     │
│   dropout: 0.3                                                      │
│                                                                     │
│   anomaly_detector.py: IsolationForest (contamination = 0.05)       │
│   shap_explainer.py:   XGBoost gain-based feature importance        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ congestion prob + anomaly score
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OPTIMIZER LAYER                              │
│                                                                     │
│   rl_env.py: Gymnasium env — obs: (9,) · action: Discrete(4)        │
│   agent.py:  PPO (Stable-Baselines3) — 200,000 training steps       │
│   rule_based_agent.py: threshold policy for A/B comparison          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ actions + rewards
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    API + DASHBOARD LAYER                            │
│                                                                     │
│   FastAPI:  WS /ws/simulation · POST /api/predict                   │
│             POST /api/agent/action · GET /metrics (Prometheus)      │
│             GET /api/replay/* (record · play · stop)                │
│                                                                     │
│   React 18: Overview · Network Map · KPIs · Predictions             │
│             SHAP · Anomaly · A/B Test · RL Agent                    │
│             D3.js live map · Recharts time-series · TailwindCSS     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Complete Data Flow

Every second, this is exactly what happens:

```
  Tick N starts
       │
       ├─► SimPy advances 1 second
       │        │
       │        ├─► Each UE moves (Random Waypoint, max 3 m/s)
       │        ├─► SINR computed for every UE↔gNB pair (NumPy vectorised)
       │        ├─► Handover check: if neighbour SINR > serving + 3 dB → switch
       │        └─► PRBs allocated proportional to UE demand
       │
       ├─► KPI Calculator runs
       │        ├─► cell_load    = allocated_PRBs / 100
       │        ├─► throughput   = Σ UE throughputs per cell
       │        ├─► latency      = 15 + 85 × load  (ms)
       │        └─► handover_rate = events / active_UEs
       │
       ├─► ML Inference
       │        ├─► Sequence buffer updated (last 10 ticks × 18 features)
       │        ├─► LSTM forward pass → P_lstm ∈ [0,1]
       │        ├─► XGBoost predict_proba → P_xgb ∈ [0,1]
       │        ├─► Ensemble: P = 0.6·P_lstm + 0.4·P_xgb
       │        └─► IsolationForest scores current tick
       │
       ├─► RL Agents (both run every tick for A/B)
       │        ├─► PPO observes 9-dim state → action ∈ {0,1,2,3}
       │        ├─► Rule-based checks thresholds → action ∈ {0,1,2,3}
       │        └─► Both rewards computed and logged
       │
       └─► FastAPI broadcasts JSON payload over WebSocket
                └─► React dashboard updates all 8 panels simultaneously
```

---

## ✨ Feature Highlights

| Feature | Details | Why It Matters |
|---|---|---|
| **Physics-Based SINR** | Path loss exponent 3.5, 3.5 GHz carrier, 43 dBm TX, noise −104 dBm | Real 5G urban propagation model (3GPP TR 38.901 UMa) |
| **5QI Traffic Profiles** | Video (5QI-2), Gaming (5QI-3), IoT (5QI-5), VoIP (5QI-1) per 3GPP TS 23.501 | 3GPP-compliant demand simulation |
| **LSTM+XGBoost Ensemble** | 60/40 weighted, F1=0.871, AUC=0.984, Precision=1.0 | Zero false-positive alarms |
| **30-Tick Prediction Horizon** | Predicts congestion 30 seconds ahead of occurrence | Proactive vs reactive network management |
| **PPO RL Agent** | Trained 200K steps, Discrete(4) actions, custom reward function | Autonomous load balancing that outperforms hand-coded rules |
| **A/B Testing** | PPO vs rule-based running in parallel every single tick | Quantifies RL advantage with live evidence |
| **IsolationForest Anomaly** | Trained on 10,800 rows, contamination=0.05 | Catches unknown failure modes LSTM was never trained on |
| **Prometheus /metrics** | 11 live gauges: cell load, throughput, latency, SINR, anomaly score | Production observability — any Grafana connects instantly |
| **REST Model APIs** | `POST /api/predict` and `POST /api/agent/action` | Models callable as microservices from any system |
| **Historical Replay** | Record any time window, replay at 1×/2×/4×/8× speed | Demonstrate congestion events to stakeholders on demand |
| **Live WebSocket** | 1-second tick broadcast, auto-reconnect, 60-tick history on connect | Real-time dashboard that survives network drops |
| **Dark + Light Theme** | Full CSS variable system, one-click toggle | Professional NOC interface for any environment |

---

## 🛠 Tech Stack

### Simulation
| Technology | Version | Role |
|---|---|---|
| Python | 3.11 | Core language |
| SimPy | 4.1.1 | Discrete-event simulation engine |
| NumPy | latest | Vectorised SINR and path loss calculations |

### Machine Learning
| Technology | Version | Role |
|---|---|---|
| PyTorch | latest | LSTM model (2 layers, hidden=64, dropout=0.3) |
| XGBoost | latest | Gradient boosting (300 estimators, max_depth=6) |
| Stable-Baselines3 | latest | PPO reinforcement learning |
| Gymnasium | latest | Custom 5G RL environment |
| scikit-learn | latest | IsolationForest anomaly detection, StandardScaler |
| SHAP | latest | XGBoost gain-based feature importance |

### Backend
| Technology | Version | Role |
|---|---|---|
| FastAPI | latest | Async REST + WebSocket server |
| Uvicorn | latest | ASGI server |
| prometheus-client | 0.25.0 | Metrics exposition (11 live gauges) |
| SQLite | built-in | KPI data persistence |

### Frontend
| Technology | Version | Role |
|---|---|---|
| React | 18 | UI framework with hooks and context |
| D3.js | latest | Live network topology map |
| Recharts | latest | KPI time-series charts |
| TailwindCSS | latest | Utility-first styling |
| Vite | latest | Build tool and hot reload |

### DevOps
| Technology | Role |
|---|---|
| Docker | Container image build |
| Docker Compose | Multi-service orchestration (backend + frontend + nginx) |
| AWS EC2 | Cloud deployment target |
| NGINX | Reverse proxy and static file serving |

---

## 📁 Project Structure

```
5G_Digital_Twin/
│
├── config.py                           ← Single source of truth for all constants
├── requirements.txt
│
├── simulation/                         ← Core 5G network simulation
│   ├── engine.py                       ← NetworkSimulation class, SimPy event loop
│   ├── gnb.py                          ← gNB base station (PRB allocation, load)
│   ├── ue.py                           ← UE model (mobility, 5QI profiles, handover)
│   ├── channel.py                      ← Vectorised SINR + path loss (NumPy)
│   └── mobility.py                     ← Random Waypoint mobility model
│
├── kpi/                                ← KPI computation and storage
│   ├── calculator.py                   ← Per-tick KPI calculation
│   ├── data_generator.py               ← 10,800-row dataset with congestion injection
│   └── storage.py                      ← SQLite persistence layer
│
├── ml/                                 ← All machine learning models
│   ├── lstm_model.py                   ← PyTorch LSTM (input=18, hidden=64, layers=2)
│   ├── xgboost_model.py                ← XGBoost classifier wrapper
│   ├── ensemble.py                     ← 0.6×LSTM + 0.4×XGBoost weighted ensemble
│   ├── data_preprocessor.py            ← 18 features, StandardScaler pipeline
│   ├── train.py                        ← Training script (LSTM + XGBoost together)
│   ├── anomaly_detector.py             ← IsolationForest wrapper
│   └── shap_explainer.py               ← XGBoost gain-based feature importance
│
├── optimizer/                          ← Reinforcement learning
│   ├── rl_env.py                       ← Custom Gymnasium env (9-dim obs, Discrete(4))
│   ├── agent.py                        ← PPO training wrapper (Stable-Baselines3)
│   └── rule_based_agent.py             ← Threshold policy for A/B testing baseline
│
├── api/                                ← FastAPI backend
│   ├── main.py                         ← All endpoints: WS, REST, Prometheus, Replay
│   └── run.py                          ← Uvicorn launcher
│
├── dashboard/                          ← React 18 frontend
│   └── src/
│       ├── App.jsx                     ← Panel routing + LandingPage gate
│       ├── context/
│       │   ├── SimContext.jsx          ← Global simulation state (useReducer)
│       │   └── ThemeContext.jsx        ← Dark/light theme provider
│       ├── hooks/
│       │   └── useSimSocket.js         ← WebSocket connection + auto-reconnect
│       └── components/
│           ├── LandingPage.jsx         ← Hero page (animated stats, tech badges)
│           ├── PanelWrapper.jsx        ← Shared header + modal info system
│           ├── layout/
│           │   ├── Sidebar.jsx         ← Navigation with live cell indicators
│           │   └── TopBar.jsx          ← Tick counter + network health chip
│           ├── panels/
│           │   ├── OverviewPanel.jsx   ← System-wide KPI summary + sparklines
│           │   ├── NetworkMapPanel.jsx ← D3.js live map, UEs by 5QI profile
│           │   ├── KPIPanel.jsx        ← Per-cell throughput / latency / load
│           │   ├── PredictionPanel.jsx ← LSTM+XGBoost congestion probability
│           │   ├── ShapPanel.jsx       ← Feature importance waterfall chart
│           │   ├── AnomalyPanel.jsx    ← IsolationForest score gauge + history
│           │   ├── ABTestPanel.jsx     ← PPO vs rule-based reward comparison
│           │   └── RLAgentPanel.jsx    ← PPO action distribution bar chart
│           └── shared/
│               └── StatCard.jsx        ← Threshold-coloured KPI card
│
├── models/                             ← Trained model artifacts (git-tracked)
│   ├── lstm_best.pt                    ← Best LSTM checkpoint (PyTorch)
│   ├── xgboost_model.json              ← XGBoost booster (JSON format)
│   ├── scaler.pkl                      ← Fitted StandardScaler
│   ├── ppo_agent.zip                   ← Trained PPO policy (Stable-Baselines3)
│   └── anomaly_detector.pkl            ← Trained IsolationForest
│
├── data/
│   └── kpi_dataset.csv                 ← 10,800 rows × 22 columns
│
├── reports/
│   ├── phase4_results.json             ← ML evaluation metrics
│   ├── phase5_results.json             ← RL training results
│   └── shap_summary.png                ← Feature importance visualisation
│
├── scripts/
│   ├── train_anomaly_detector.py       ← One-time anomaly model training
│   └── train_rl_agent.py               ← PPO training script (≈10 min)
│
└── tests/                              ← 24 passing unit tests
    ├── test_simulation.py
    ├── test_kpi.py
    ├── test_phase4.py
    └── test_phase5.py
```

---

## 🔬 Core Components — Deep Dive

### 1 · 5G Network Simulation Engine

**Files:** `simulation/engine.py` · `gnb.py` · `ue.py` · `channel.py`

The simulation models a **1 km × 1 km urban grid** with 3 gNB base stations and 20 mobile UEs. It runs as a SimPy discrete-event simulation, advancing in 1-second ticks for 3 hours (10,800 ticks total).

#### gNB Configuration

```
gNB-0: position (200 m, 500 m)   ← West sector
gNB-1: position (500 m, 200 m)   ← North sector
gNB-2: position (800 m, 700 m)   ← East sector

TX Power:       43 dBm
Antenna Gain:   15 dB
Max PRBs:       100  (Physical Resource Blocks per gNB)
Carrier:        3.5 GHz  (5G NR n78 Sub-6 band)
Noise Power:   −104 dBm
```

#### SINR Calculation

The path loss model follows **3GPP TR 38.901 Urban Macro (UMa)**:

```
Step 1 — Path Loss
──────────────────
PL(d, f) = 20·log₁₀(d) + 20·log₁₀(f) + 92.4 + 10·α·log₁₀(d / 1000)

  where:
    d  = distance between UE and gNB (metres)
    f  = 3.5  (GHz)
    α  = 3.5  (urban path loss exponent)

Step 2 — Received Power
───────────────────────
P_rx = P_tx + G_ant − PL(d, f)
     = 43 dBm + 15 dB − PL

Step 3 — SINR
─────────────
SINR = P_rx(serving gNB)
       ─────────────────────────────────────────────
       P_noise + Σ P_rx(all interfering gNBs)

       clipped to [−6 dB, +30 dB]
```

#### UE Traffic Profiles (3GPP TS 23.501)

| Profile | 5QI Class | Demand Range | Speed Factor | Population |
|---|---|---|---|---|
| Video Streaming | 5QI-2 | 10 – 20 Mbps | 1.0× | 35 % |
| Gaming | 5QI-3 | 4 – 10 Mbps | 0.3× | 30 % |
| IoT Sensor | 5QI-5 | 0.1 – 1 Mbps | 0.5× | 20 % |
| VoIP | 5QI-1 | 0.5 – 2 Mbps | 0.8× | 15 % |

**Mobility:** Random Waypoint (RWP) — each UE picks a random destination and moves at up to 3 m/s. On arrival it pauses briefly, then picks a new waypoint.

**Handover logic:** A UE triggers a handover when a neighbouring gNB offers SINR that is more than 3 dB higher than the current serving gNB (hysteresis-based, standard in LTE/5G NR).

---

### 2 · KPI Engine & Data Generation

**Files:** `kpi/calculator.py` · `kpi/data_generator.py`

Every simulation tick produces the following KPIs **per cell**:

| KPI | Formula | Unit |
|---|---|---|
| Cell Load | `allocated_PRBs / max_PRBs` | 0.0 – 1.0 |
| Cell Throughput | `Σ UE_throughput` for all UEs on this cell | Mbps |
| Cell Latency | `15 + 85 × load` (linear model) | ms |
| Handover Rate | `handover_events / active_UEs` | events/UE |
| Packet Loss Rate | `UEs below SINR threshold / total_UEs` | 0.0 – 1.0 |

The data generator runs the full 10,800-tick simulation and **injects artificial congestion events** to create a balanced training dataset:

| Event Type | Load Range | Duration |
|---|---|---|
| Normal operation | 20 – 70 % | baseline |
| Warning event | 70 – 90 % | 30 – 60 ticks |
| Critical event | 90 – 100 % | 10 – 20 ticks |

This produces `data/kpi_dataset.csv` — **10,800 rows × 22 columns** — with a class balance of ~88 % normal / ~12 % congested.

---

### 3 · ML Congestion Prediction — LSTM + XGBoost Ensemble

**Files:** `ml/lstm_model.py` · `ml/xgboost_model.py` · `ml/ensemble.py` · `ml/data_preprocessor.py`

#### Feature Engineering — 18 Input Features

```python
FEATURE_COLUMNS = [
    # Per-cell load (primary signal — 3 features)
    'cell0_load', 'cell1_load', 'cell2_load',

    # Per-cell throughput — saturation indicator (3 features)
    'cell0_throughput', 'cell1_throughput', 'cell2_throughput',

    # Per-cell UE count — density pressure (3 features)
    'cell0_ue_count', 'cell1_ue_count', 'cell2_ue_count',

    # Per-cell average SINR — radio quality (3 features)
    'cell0_avg_sinr', 'cell1_avg_sinr', 'cell2_avg_sinr',

    # System-wide aggregates (3 features)
    'system_throughput', 'system_avg_sinr', 'system_avg_latency_ms',

    # Event counters (3 features)
    'handover_count', 'handover_rate', 'packet_loss_rate',
]
```

#### LSTM Architecture

```
Input tensor: (batch=1, sequence=10 ticks, features=18)
                              │
                    ┌─────────▼──────────┐
                    │   LSTM Layer 1     │  hidden_size=64, dropout=0.3
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │   LSTM Layer 2     │  hidden_size=64, dropout=0.3
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Linear(64 → 1)   │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │     Sigmoid        │  output ∈ [0, 1]
                    └────────────────────┘
```

**Why LSTM?** Congestion is a temporal phenomenon. A load of 60 % right now means very different things depending on whether it has been rising for 5 ticks (→ congestion likely) or falling from 85 % (→ recovering). LSTM's hidden state captures this trajectory across the last 10 ticks — something a point-in-time classifier fundamentally cannot do.

#### XGBoost Configuration

```python
XGBClassifier(
    n_estimators    = 300,
    max_depth       = 6,
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    scale_pos_weight= 7.3,   # balances 88/12 class split
    eval_metric     = 'logloss',
)
```

**Why XGBoost?** It captures sharp feature interaction thresholds — e.g. "when `cell0_load > 0.7` AND `ue_count > 8`, congestion spikes" — that LSTM's continuous hidden state tends to smooth over.

#### Ensemble Combination

```
Final probability = 0.6 × P_LSTM  +  0.4 × P_XGBoost
```

Weights tuned empirically: LSTM dominates because its temporal modelling catches drift earlier. XGBoost anchors precision by catching threshold crossings the LSTM may mistime.

#### Results

| Model | F1 Score | AUC-ROC | Precision | Recall |
|---|---|---|---|---|
| LSTM alone | 0.829 | — | — | — |
| XGBoost alone | 0.847 | 0.971 | — | — |
| **Ensemble** | **0.871** | **0.984** | **1.000** | 0.771 |

> **Precision = 1.000** means the ensemble has **zero false positives** across the entire test set. It never raises a congestion alarm when the network is healthy — the critical metric for production use, where false alarms erode operator trust and trigger unnecessary handovers that degrade service.

---

### 4 · Reinforcement Learning Agent (PPO)

**Files:** `optimizer/rl_env.py` · `optimizer/agent.py`

#### Custom Gymnasium Environment

```python
class FiveGNetworkEnv(gymnasium.Env):

    observation_space = Box(low=0, high=1, shape=(9,), dtype=np.float32)
    # Observation vector (9 dimensions):
    #   [cell0_load, cell1_load, cell2_load,        ← current network stress
    #    cong_prob0, cong_prob1, cong_prob2,         ← ML prediction per cell
    #    ue_ratio0,  ue_ratio1,  ue_ratio2]          ← UE distribution pressure

    action_space = Discrete(4)
    # Action meanings:
    #   0 = NoOp             — network healthy, do nothing
    #   1 = LoadBalance      — soft handover: move UEs from busiest to least loaded
    #   2 = MassBalance      — aggressive rebalancing across all cells
    #   3 = EmergencyHandover — immediate hard handover for critical overload
```

#### Reward Function

```python
def compute_reward(cell_loads):
    reward = 0.0

    for load in cell_loads:
        if   load < 0.70:  reward += 0.2    # healthy cell → reward stability
        elif load < 0.90:  reward -= 0.5    # warning zone → penalise
        else:              reward -= 1.0    # critical zone → heavy penalty

    # Bonus for balanced distribution across cells
    reward += max(0.0, 0.1 - std(cell_loads))

    return clip(reward, -1.0, 1.0)
```

#### Training Configuration

```
Algorithm:      PPO  (Proximal Policy Optimization)
Library:        Stable-Baselines3
Total steps:    200,000
Policy network: MlpPolicy  (2 × 64 hidden layers)
Learning rate:  3e-4
Clip range:     0.2
```

**Key finding:** The PPO agent consistently chooses `NoOp` (action 0) when the network is healthy — demonstrating it has learned a nuanced "do not overreact" policy. The rule-based agent, by contrast, frequently triggers `LoadBalance` unnecessarily in response to normal load variance. This difference is visible live in the A/B Test panel.

---

### 5 · Anomaly Detection (IsolationForest)

**File:** `ml/anomaly_detector.py`

The IsolationForest provides a **second line of defence** beyond the supervised LSTM ensemble — it detects anomalies that were never seen in training data: hardware faults, unusual flash-crowd events, coordinated attacks.

```python
IsolationForest(
    n_estimators  = 100,
    contamination = 0.05,   # expects 5% of live data to be anomalous
    random_state  = 42,
)
```

| Anomaly Score Range | Severity | Dashboard Colour |
|---|---|---|
| score > −0.10 | Normal | Green |
| −0.20 < score ≤ −0.10 | Warning | Amber |
| score ≤ −0.20 | Critical | Red |

Trained on the full 10,800-row KPI dataset. Scores every incoming tick in the live simulation loop.

---

### 6 · A/B Testing Framework

**Files:** `optimizer/rule_based_agent.py` · integrated in `api/main.py`

Every tick, **both agents observe the same 9-dimensional state** and independently produce an action and a reward. The results are logged and streamed to the A/B Test dashboard panel.

**Rule-Based Agent policy (for comparison baseline):**

```python
if   max(loads) > 0.90:           action = EmergencyHandover  (3)
elif max(loads) > 0.80:           action = MassBalance        (2)
elif max(loads) − min(loads) > 0.30: action = LoadBalance     (1)
else:                             action = NoOp               (0)
```

The live comparison quantifies — with actual evidence, not theoretical argument — that the PPO agent's learned policy produces higher cumulative rewards than the hand-coded threshold policy.

---

### 7 · FastAPI Backend & WebSocket

**File:** `api/main.py`

The backend is a single FastAPI application with a **lifespan startup sequence** that loads all components in order:

```
Startup sequence (api/main.py lifespan):
─────────────────────────────────────────
  1.  NetworkSimulation          (SimPy engine)
  2.  StandardScaler             (scaler.pkl)
  3.  LSTM + XGBoost Ensemble    (lstm_best.pt + xgboost_model.json)
  4.  PPO Agent                  (ppo_agent.zip)
  5.  Anomaly Detector           (anomaly_detector.pkl)
  6.  Rule-Based Agent           (pure Python logic, no file)
  7.  Simulation loop            (asyncio background task, auto-start)
```

**WebSocket tick payload — emitted every 1 second:**

```json
{
  "tick": 1432,
  "timestamp": 1776234785.953,
  "cells": [
    {
      "cell_id": 0,
      "load_percent": 0.49,
      "throughput_mbps": 437.11,
      "latency_ms": 56.65,
      "connected_ues": 7,
      "prb_used": 49
    }
  ],
  "ues": [
    {
      "ue_id": 3,
      "x": 208.0, "y": 317.0,
      "connected_cell": 0,
      "sinr_db": 7.96,
      "throughput_mbps": 57.19,
      "is_handover": false,
      "traffic_profile": "Gaming",
      "qos_class": 3
    }
  ],
  "kpis": {
    "total_throughput": 1456.03,
    "mean_latency": 31.2,
    "handover_count": 0,
    "active_ues": 20
  },
  "congestion_predictions": { "0": 0.12, "1": 0.08, "2": 0.31 },
  "ppo_actions":            { "0": 2,    "1": 0,    "2": 0    },
  "anomaly": {
    "anomaly_score": 0.3852,
    "is_anomaly": false,
    "severity": "normal"
  },
  "ab_comparison": {
    "tick": 1432,
    "ppo_action": 0,   "ppo_reward": 0.6,
    "rb_action": 1,    "rb_reward": 0.3
  }
}
```

---

### 8 · React Live Dashboard

**Directory:** `dashboard/src/`

Eight panels, all driven by a single WebSocket connection. State is managed globally in `SimContext.jsx` via `useReducer`. The `useSimSocket.js` hook handles connection, auto-reconnect, and 60-tick history replay on connect.

| Panel | What It Shows | Technology |
|---|---|---|
| **Overview** | System-wide KPIs — throughput, latency, per-cell load | Recharts AreaChart |
| **Network Map** | Live UE positions + gNB coverage + handover events | D3.js SVG |
| **KPIs** | Per-cell throughput / latency / load time-series (last 60 ticks) | Recharts LineChart |
| **Predictions** | LSTM+XGBoost congestion probability (0–1) per cell | Recharts AreaChart |
| **SHAP** | Feature importance waterfall chart (XGBoost gain-based) | Recharts BarChart |
| **Anomaly** | IsolationForest score gauge + score history with severity colouring | Recharts LineChart |
| **A/B Test** | PPO vs rule-based reward per tick + cumulative win rate | Recharts ComposedChart |
| **RL Agent** | PPO action distribution over last 100 ticks | Recharts BarChart |

**Network Map UE colour coding:**

| Colour | Traffic Profile | 5QI Class |
|---|---|---|
| 🔵 Blue | Video Streaming | 5QI-2 |
| 🟢 Green | Gaming | 5QI-3 |
| 🟡 Yellow | IoT Sensor | 5QI-5 |
| 🟣 Purple | VoIP | 5QI-1 |
| 🔴 Red ring | Active handover event | — |

---

## 📊 Key Metrics & Results

### ML Model Performance

| Metric | Value | Notes |
|---|---|---|
| Dataset size | 10,800 rows | One row per simulation tick |
| Feature count | 18 | Engineered from raw KPIs |
| Class balance | 88 % normal / 12 % congested | |
| LSTM F1 | 0.829 | Sequence model alone |
| XGBoost AUC | 0.971 | Point-in-time model alone |
| **Ensemble F1** | **0.871** | Combined: F1 +4.3% vs XGBoost alone |
| **Ensemble AUC-ROC** | **0.984** | |
| **Ensemble Precision** | **1.000** | Zero false-positive congestion alarms |
| Ensemble Recall | 0.771 | |
| Prediction horizon | 30 ticks | 30 seconds of advance warning |

### Simulation Parameters

| Parameter | Value |
|---|---|
| Grid size | 1,000 m × 1,000 m |
| gNB count | 3 |
| UE count | 20 |
| TX Power | 43 dBm |
| Carrier frequency | 3.5 GHz (n78 band) |
| Path loss exponent | 3.5 (urban) |
| Noise power | −104 dBm |
| Simulation duration | 10,800 seconds (3 hours) |
| Tick duration | 1 second |
| SINR range | −6 dB to +30 dB |
| Max UE speed | 3 m/s |

### RL Training

| Parameter | Value |
|---|---|
| Algorithm | PPO (Proximal Policy Optimization) |
| Total training steps | 200,000 |
| Observation dimensions | 9 |
| Action space | Discrete(4) |
| Policy network | MLP — 2 × 64 hidden |
| Learning rate | 3e-4 |
| Clip range | 0.2 |

---

## 📡 API Reference

### WebSocket

| Endpoint | Description |
|---|---|
| `WS /ws/simulation` | Subscribe to live tick stream. Sends 60-tick history on connect, then 1 tick/second. |

### Simulation Control

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/simulation/start` | Start the simulation loop |
| `POST` | `/api/simulation/stop` | Stop the simulation loop |
| `GET` | `/api/status` | Current status: tick, mode, connected clients |
| `GET` | `/health` | Health check: ensemble_loaded, ppo_loaded, mode |

### Data & Analytics

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/history?limit=100` | Last N ticks from simulation history |
| `GET` | `/api/kpis/summary` | Aggregate KPI statistics |
| `GET` | `/api/cells/{cell_id}/metrics` | Per-cell metric history (cell_id: 0, 1, 2) |
| `GET` | `/api/shap/explanation` | XGBoost feature importance (gain-based) |
| `GET` | `/api/anomaly/current` | Latest IsolationForest anomaly score |
| `GET` | `/api/anomaly/history?limit=100` | Anomaly score history |
| `GET` | `/api/ab/history?limit=100` | PPO vs rule-based comparison history |
| `GET` | `/api/ab/summary` | Aggregate A/B test statistics |

### ML Inference — Models as a Service

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | KPI snapshot → congestion probability from live ensemble |
| `POST` | `/api/agent/action` | 9-dim observation → PPO action + rule-based comparison |

**Example — `/api/predict`:**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cell0_load": 0.85, "cell1_load": 0.40, "cell2_load": 0.30,
    "cell0_throughput": 200, "cell1_throughput": 400, "cell2_throughput": 300,
    "cell0_ue_count": 9,  "cell1_ue_count": 6,  "cell2_ue_count": 5
  }'
```

```json
{
  "system_congestion_probability": 0.3412,
  "cell_congestion_probabilities": { "0": 0.5613, "1": 0.2641, "2": 0.1981 },
  "prediction_horizon_ticks": 30,
  "model": "lstm_xgboost_ensemble_0.6_0.4"
}
```

**Example — `/api/agent/action`:**

```bash
curl -X POST http://localhost:8000/api/agent/action \
  -H "Content-Type: application/json" \
  -d '{"cell0_load": 0.91, "cell1_load": 0.30, "cell2_load": 0.25}'
```

```json
{
  "action": 2,
  "action_label": "MassBalance",
  "rule_based_action": 3,
  "rule_based_label": "EmergencyHandover"
}
```

### Observability

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/metrics` | Prometheus scrape endpoint — 11 live gauges, standard text format |

```
# HELP cell_load_percent PRB load per cell
# TYPE cell_load_percent gauge
cell_load_percent{cell_id="0"} 0.49
cell_load_percent{cell_id="1"} 0.29
cell_load_percent{cell_id="2"} 0.21

# HELP system_throughput_mbps Total system throughput
# TYPE system_throughput_mbps gauge
system_throughput_mbps 1451.75

# HELP anomaly_score IsolationForest anomaly score
# TYPE anomaly_score gauge
anomaly_score 0.3852
```

### Replay

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/replay/record/start` | Begin buffering live ticks |
| `POST` | `/api/replay/record/stop` | Stop recording, freeze buffer |
| `POST` | `/api/replay/play?speed=4.0` | Stream buffer over WebSocket at N× speed |
| `POST` | `/api/replay/stop` | Abort active replay |
| `GET` | `/api/replay/status` | Recording state, buffer size, replay active |

---

## 🚀 Installation & Setup

### Prerequisites

```
Python  3.11+
Node.js 18+
Git
```

### 1 · Clone

```bash
git clone https://github.com/poornachandran2006/5G_Digital_Twin.git
cd 5G_Digital_Twin
```

### 2 · Python Dependencies

```bash
pip install -r requirements.txt
```

Core packages:
```
simpy==4.1.1
numpy
torch
xgboost
stable-baselines3[extra]
gymnasium
scikit-learn
shap
fastapi
uvicorn
prometheus-client==0.25.0
```

### 3 · Frontend Dependencies

```bash
cd dashboard && npm install && cd ..
```

### 4 · Verify Trained Models

All five model artifacts must be present in `models/`:

```
models/
├── lstm_best.pt           ← PyTorch LSTM checkpoint
├── xgboost_model.json     ← XGBoost booster
├── scaler.pkl             ← Fitted StandardScaler
├── ppo_agent.zip          ← Trained PPO policy
└── anomaly_detector.pkl   ← Trained IsolationForest
```

If any are missing, regenerate:

```bash
python -m kpi.data_generator           # generates kpi_dataset.csv
python -m ml.train                     # trains LSTM + XGBoost → saves artifacts
python scripts/train_anomaly_detector.py
python scripts/train_rl_agent.py       # ~10 minutes on CPU
```

---

## ▶️ How to Run

```bash
# Terminal 1 — backend
cd api && python run.py
```

```bash
# Terminal 2 — frontend
cd dashboard && npm run dev
```

**Expected backend output:**
```
✓ NetworkSimulation initialised
✓ Scaler loaded
✓ Ensemble (LSTM + XGBoost) loaded on cpu
✓ PPO agent loaded
✓ Anomaly detector loaded
✓ Rule-based agent initialized
✓ Simulation loop auto-started
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Dashboard: **http://localhost:5173**
API docs:  **http://localhost:8000/docs**
---

## 🖥 Dashboard Walkthrough

### Landing Page
Animated hero with live stat counters (gNBs, UEs, AUC, prediction window). Explains the project in plain English before entering the technical dashboard.

### Overview Panel
System-wide KPIs at a glance: total throughput (Mbps), mean latency (ms), per-cell load bars with threshold colouring (green → amber at 70 %, red at 90 %). First thing an NOC engineer looks at.

### Network Map Panel
Live D3.js visualisation of the 1 km × 1 km grid. Triangle markers = gNBs with coverage rings. Coloured dots = UEs moving in real time. Lines = active UE↔gNB connections. Red ring = active handover. Hover any UE for SINR and throughput tooltip.

### KPIs Panel
Three tabbed time-series (Throughput / Latency / Cell Load) for all 3 cells over the last 60 ticks. Use this panel to spot per-cell divergence and load trends.

### Predictions Panel
LSTM+XGBoost congestion probability (0–1) per cell, updated every tick. Threshold line at 0.7. Each value is a 30-second look-ahead — giving operators time to act before congestion occurs.

### SHAP Panel
XGBoost gain-based feature importance waterfall chart. Shows which of the 18 input features most influenced the current prediction. `cell_load` features typically dominate, followed by `ue_count` and `avg_sinr`.

### Anomaly Panel
IsolationForest score gauge (−1 to 0) and time-series. Severity bands: green (normal) / amber (warning) / red (critical). Catches failure modes the supervised model was never trained on.

### A/B Test Panel
Live comparison of PPO agent vs rule-based agent. Reward chart per tick, action distribution bars, and cumulative win-rate. Demonstrates RL advantage with live evidence, not theoretical claims.

### RL Agent Panel
Distribution of PPO actions (NoOp / LoadBalance / MassBalance / EmergencyHandover) over the last 100 ticks. A healthy network shows ~80 % NoOp — the agent has learned when *not* to act.

### Replay Controls (TopBar)
1. Click **● REC** → start recording tick data to buffer
2. Click **■ STOP** → freeze buffer
3. Select speed (1× / 2× / 4× / 8×)
4. Click **▶ PLAY** → replay across all 8 panels simultaneously
5. **⏪ REPLAY** badge appears in the top bar during playback

---

## ⚡ 5G Physics — The Science Behind It

### Why Path Loss Exponent α = 3.5?

In free space, signal power decreases as `1/d²` (α = 2.0). In urban environments, buildings cause reflections, diffraction, and shadowing that increase effective attenuation. The exponent α = 3.5 is the standard value for **dense urban environments** (3GPP TR 38.901 UMa model), used in real deployments by Ericsson and Nokia.

| Environment | Typical α |
|---|---|
| Free space | 2.0 |
| Suburban | 2.7 – 3.0 |
| Urban | 3.0 – 3.5 |
| Dense urban (this project) | **3.5** |
| Indoor obstructed | 4.0 – 6.0 |

### Why 3.5 GHz?

The **n78 band (3.3 – 3.8 GHz)** is the primary 5G NR mid-band globally — deployed by:
- Jio (India, nationwide 5G rollout)
- Deutsche Telekom (Germany)
- SK Telecom, KT (South Korea)
- BT/EE (UK)

It offers the critical balance: better coverage than mmWave (28 GHz), better capacity than sub-1 GHz. Simulating at 3.5 GHz makes this digital twin directly applicable to real live networks.

### Why −104 dBm Noise Power?

Derived from first principles using Johnson–Nyquist thermal noise:

```
Thermal noise power:
P_noise = k · T · B
        = 1.38×10⁻²³ · 290 · 20×10⁶
        = 8.0×10⁻¹⁴ W
        = −101 dBm

Add 3 dB noise figure (receiver imperfections):
P_noise = −101 − 3 = −104 dBm  ✓
```

---

## 🧠 ML Architecture — How the Models Work

### Why LSTM for Congestion Prediction?

Congestion is a **trajectory problem**, not a snapshot problem. Consider two UEs both at 60 % load:

- UE A: load has been 45 % → 50 % → 55 % → 60 % for the last 4 ticks → **likely congested in 30 seconds**
- UE B: load has been 85 % → 75 % → 68 % → 60 % for the last 4 ticks → **recovering, no action needed**

An LSTM sees this 10-tick history as a sequence and learns to distinguish these cases. A feed-forward network or XGBoost model receiving only the current snapshot cannot.

### Why XGBoost in the Ensemble?

XGBoost excels at capturing **hard threshold interactions** that the LSTM's continuous hidden state tends to smooth over. Rules like "when `cell0_load > 0.72` AND `ue_count > 8` simultaneously → congestion probability jumps" are naturally expressed as decision tree splits. The ensemble combines LSTM's temporal awareness with XGBoost's threshold precision.

### Why Precision = 1.0 Matters More Than Recall

In a real telecom NOC:

- A **false positive** (alarm when network is healthy) → engineers investigate → may trigger unnecessary handovers → degrades service for all users on that cell
- A **false negative** (missed alarm) → can still be caught reactively by other monitoring

Precision = 1.000 across the test set means every alarm is a genuine event. The 77.1 % recall means we miss some events — acceptable for a proactive system that still has reactive fallback.

### Why IsolationForest for Anomaly Detection?

The LSTM ensemble is **supervised** — it can only detect patterns it was trained on. IsolationForest is **unsupervised** — it learns the "normal" multivariate distribution and flags anything outside it, regardless of whether that pattern appeared in training data. This is the key difference:

| Scenario | LSTM Ensemble | IsolationForest |
|---|---|---|
| Known congestion pattern | ✅ Detects | ✅ Detects |
| Hardware RF interference | ❌ Not in training | ✅ Detects (anomalous SINR) |
| Flash crowd (stadium event) | ❌ Not in training | ✅ Detects (anomalous UE count) |
| Coordinated DoS traffic | ❌ Not in training | ✅ Detects |

The two models together provide defence-in-depth.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI bind port |
| `WS_URL` | `ws://localhost:8000/ws/simulation` | Frontend WebSocket target |
| `SIM_TICK_RATE` | `1.0` | Simulation speed (seconds per tick) |

## 👤 Author

<div align="center">

**Poornachandran**
3rd Year ECE Engineering Student
Specialising in 5G/Telecom Systems · ML for Networks · RL Optimisation

[![GitHub](https://img.shields.io/badge/GitHub-poornachandran2006-181717?style=flat-square&logo=github)](https://github.com/poornachandran2006/5G_Digital_Twin)

> *"Every component in this project is grounded in actual 5G physics, real ML engineering practices, and production observability patterns"*

</div>

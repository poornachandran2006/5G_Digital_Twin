<div align="center">

```
███████╗ ██████╗      ███╗   ██╗███████╗████████╗██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗
██╔════╝██╔════╝      ████╗  ██║██╔════╝╚══██╔══╝██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝
███████╗██║  ███╗     ██╔██╗ ██║█████╗     ██║   ██║ █╗ ██║██║   ██║██████╔╝█████╔╝ 
╚════██║██║   ██║     ██║╚██╗██║██╔══╝     ██║   ██║███╗██║██║   ██║██╔══██╗██╔═██╗ 
███████║╚██████╔╝     ██║ ╚████║███████╗   ██║   ╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗
╚══════╝ ╚═════╝      ╚═╝  ╚═══╝╚══════╝   ╚═╝    ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
                    D I G I T A L   T W I N   ·   A I - P O W E R E D
```

<h2>Industry-Grade 5G Network Simulation with ML Congestion Prediction & RL Optimization</h2>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-WebSocket-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/AWS-EC2-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Ensemble%20AUC-0.984-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Ensemble%20F1-0.871-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/PPO%20Steps-200K-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Simulation%20Ticks-10%2C800-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Tests%20Passing-24-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
</p>

<br/>

> **A software-only digital twin of a real urban 5G cellular network.**  
> Physics-based simulation → real-time KPI streaming → ML congestion prediction → RL load balancing → live dashboard.  
> Every component is grounded in actual 3GPP standards and production engineering practices.

<br/>

</div>

---

## 📋 Table of Contents

1. [What Is a Digital Twin?](#-what-is-a-digital-twin)
2. [Project Overview](#-project-overview)
3. [System Architecture](#-system-architecture)
4. [Feature Highlights](#-feature-highlights)
5. [Tech Stack](#-tech-stack)
6. [Project Structure](#-project-structure)
7. [Core Components — Deep Dive](#-core-components--deep-dive)
   - [5G Network Simulation Engine](#1-5g-network-simulation-engine)
   - [KPI Engine & Data Generation](#2-kpi-engine--data-generation)
   - [ML Congestion Prediction](#3-ml-congestion-prediction-lstm--xgboost-ensemble)
   - [Reinforcement Learning Agent](#4-reinforcement-learning-agent-ppo)
   - [Anomaly Detection](#5-anomaly-detection-isolationforest)
   - [A/B Testing Framework](#6-ab-testing-framework)
   - [FastAPI Backend & WebSocket](#7-fastapi-backend--websocket)
   - [React Live Dashboard](#8-react-live-dashboard)
8. [Key Metrics & Results](#-key-metrics--results)
9. [API Reference](#-api-reference)
10. [Installation & Setup](#-installation--setup)
11. [How to Run](#-how-to-run)
12. [Dashboard Walkthrough](#-dashboard-walkthrough)
13. [5G Physics — The Science Behind It](#-5g-physics--the-science-behind-it)
14. [ML Architecture — How the Models Work](#-ml-architecture--how-the-models-work)
15. [RL Environment — How the Agent Learns](#-rl-environment--how-the-agent-learns)
16. [Interview Q&A — What Recruiters Will Ask](#-interview-qa--what-recruiters-will-ask)
17. [Deployment](#-deployment)
18. [Roadmap](#-roadmap)
19. [Author](#-author)

---

## 🌐 What Is a Digital Twin?

A **Digital Twin** is a software replica of a physical system that mirrors its real-world behavior in real time. Originally developed for industrial machinery (GE, Siemens), digital twins are now a cornerstone of modern telecom engineering.

In the context of 5G networks:

| Physical World | This Digital Twin |
|---|---|
| 3 real gNB base stations in a city | 3 simulated gNBs at fixed grid positions |
| 20 mobile users walking around | 20 UEs with Random Waypoint mobility |
| Radio signals attenuating over distance | Path loss model: `PL = 20log(d) + 20log(f) + 92.4 + 10·3.5·log(d/1000)` |
| Network engineers monitoring dashboards | React dashboard with live WebSocket feed |
| NOC predicting congestion before it happens | LSTM+XGBoost ensemble predicting 30 seconds ahead |
| Network automatically rebalancing load | PPO reinforcement learning agent issuing handover commands |

This project is **100% software** — no hardware required. The entire 5G network exists in Python memory, simulated tick by tick using SimPy's discrete-event simulation engine.

---

## 🎯 Project Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     5G Network Digital Twin — Data Flow                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   SimPy Engine ──► KPI Calculator ──► LSTM + XGBoost ──► Congestion Prob    │
│        │                │                                        │            │
│        │                │                                        ▼            │
│   3 gNBs + 20 UEs   10,800 rows         PPO Agent ──────► Load Balance      │
│        │                │               Rule-Based ──────► A/B Comparison   │
│        │                ▼               IsolationForest ─► Anomaly Score    │
│        └──────► FastAPI WebSocket ──────────────────────────────────────────┤
│                        │                                                      │
│                        ▼                                                      │
│              React Dashboard (8 panels)                                       │
│              Prometheus /metrics endpoint                                     │
│              REST API (/api/predict, /api/agent/action)                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What makes this project stand out

- **Real 5G physics** — SINR computed from first principles using the 3GPP path loss model at 3.5 GHz
- **3GPP-compliant traffic** — UEs are assigned Video/Gaming/IoT/VoIP profiles with 5QI values from 3GPP TS 23.501
- **Production ML pipeline** — not just a model, but a full ensemble with preprocessing, sequence buffering, and live inference
- **RL that actually works** — PPO trained for 200,000 steps with a custom Gymnasium environment, demonstrably outperforms a rule-based baseline in A/B tests
- **Observability built in** — Prometheus `/metrics` endpoint exposes 11 live gauges; any Grafana instance can connect instantly
- **Historical replay** — record any time window and replay it at 1×/2×/4×/8× speed through the live dashboard

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              SIMULATION LAYER                                     │
│                                                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  engine.py  │    │   gnb.py    │    │    ue.py     │    │   channel.py     │  │
│  │             │    │             │    │              │    │                  │  │
│  │ SimPy loop  │───►│ 3 gNB base  │    │ 20 UEs with  │    │ Vectorized SINR  │  │
│  │ 10,800 ticks│    │ stations    │    │ RWP mobility │    │ Path loss 3.5GHz │  │
│  │ 1s per tick │    │ 43 dBm TX   │    │ 5QI profiles │    │ Exponent = 3.5   │  │
│  └──────┬──────┘    └─────────────┘    └──────────────┘    └──────────────────┘  │
└─────────┼────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                               KPI LAYER                                           │
│                                                                                    │
│  calculator.py: cell_load, throughput, latency, handover_rate, packet_loss       │
│  data_generator.py: 10,800 rows with injected congestion events                  │
│  storage.py: SQLite persistence                                                   │
└─────────┬────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                ML LAYER                                            │
│                                                                                    │
│  ┌──────────────────┐  ┌───────────────────┐  ┌────────────────────────────────┐ │
│  │   lstm_model.py  │  │ xgboost_model.py  │  │      ensemble.py               │ │
│  │                  │  │                   │  │                                │ │
│  │ input: (1,10,18) │  │ input: (1,18)     │  │ 0.6 × LSTM + 0.4 × XGBoost   │ │
│  │ hidden: 64       │  │ 300 estimators    │  │ F1=0.871, AUC=0.984           │ │
│  │ layers: 2        │  │ max_depth: 6      │  │ Precision: 1.0                │ │
│  │ dropout: 0.3     │  │                   │  │                                │ │
│  └──────────────────┘  └───────────────────┘  └────────────────────────────────┘ │
│                                                                                    │
│  anomaly_detector.py: IsolationForest (contamination=0.05)                       │
│  shap_explainer.py: XGBoost gain-based feature importance                        │
└─────────┬────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             OPTIMIZER LAYER                                        │
│                                                                                    │
│  rl_env.py: Gymnasium env — obs: 9-dim, action: Discrete(4)                      │
│  agent.py: PPO (Stable-Baselines3) — 200,000 training steps                      │
│  rule_based_agent.py: Threshold policy for A/B comparison                         │
└─────────┬────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER (FastAPI)                                   │
│                                                                                    │
│  WS /ws/simulation ──► broadcasts tick payload every 1s to all clients           │
│  GET /metrics        ──► Prometheus scrape endpoint (11 live gauges)              │
│  POST /api/predict   ──► KPI → congestion probability (live ensemble)             │
│  POST /api/agent/action ► obs → PPO action (live policy)                         │
│  GET /api/replay/*   ──► record/play/stop simulation buffer                       │
│  + 10 more REST endpoints                                                          │
└─────────┬────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           DASHBOARD LAYER (React 18)                               │
│                                                                                    │
│  Overview │ Network Map │ KPIs │ Predictions │ SHAP │ Anomaly │ A/B Test │ RL    │
│                                                                                    │
│  D3.js network map with live UE positions, color-coded by 5QI traffic profile    │
│  Recharts time-series for KPIs, predictions, anomaly scores, A/B rewards         │
│  WebSocket: auto-reconnects, loads 60-tick history on connect                     │
│  Replay mode: record any window, replay at 1×/2×/4×/8× speed                    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Feature Highlights

| Feature | Details | Why It Matters |
|---|---|---|
| **Physics-Based SINR** | Path loss exponent 3.5, 3.5 GHz carrier, 43 dBm TX, noise −104 dBm | Real 5G urban propagation model |
| **5QI Traffic Profiles** | Video (5QI-2), Gaming (5QI-3), IoT (5QI-5), VoIP (5QI-1) per 3GPP TS 23.501 | 3GPP-compliant demand simulation |
| **LSTM+XGBoost Ensemble** | 60/40 weighted, F1=0.871, AUC=0.984, Precision=1.0 | Zero false-positive alarms |
| **30-Tick Prediction Horizon** | Predicts congestion 30 seconds ahead | Proactive vs reactive network management |
| **PPO RL Agent** | Trained 200K steps, Discrete(4) actions, custom reward function | Autonomous load balancing |
| **A/B Testing** | PPO vs rule-based running in parallel every tick | Quantifies RL advantage over heuristics |
| **IsolationForest Anomaly** | Trained on 10,800 rows, contamination=0.05 | Catches unknown failure modes LSTM wasn't trained on |
| **Prometheus /metrics** | 11 live gauges: cell load, throughput, latency, SINR, anomaly score | Production observability; Grafana-compatible |
| **REST Model APIs** | POST /api/predict and /api/agent/action | Models callable as microservices |
| **Historical Replay** | Record any time window, replay at adjustable speed | Demonstrate congestion events to stakeholders |
| **Live WebSocket** | 1-second tick broadcast to all connected clients | Real-time dashboard with auto-reconnect |

---

## 🛠 Tech Stack

### Simulation
| Technology | Version | Role |
|---|---|---|
| Python | 3.11 | Core language |
| SimPy | 4.1.1 | Discrete-event simulation engine |
| NumPy | latest | Vectorized SINR/path loss calculations |

### Machine Learning
| Technology | Version | Role |
|---|---|---|
| PyTorch | latest | LSTM model (2 layers, hidden=64, dropout=0.3) |
| XGBoost | latest | Gradient boosting (300 estimators, max_depth=6) |
| Stable-Baselines3 | latest | PPO reinforcement learning |
| Gymnasium | latest | RL environment (custom 5G env) |
| scikit-learn | latest | IsolationForest anomaly detection, StandardScaler |

### Backend
| Technology | Version | Role |
|---|---|---|
| FastAPI | latest | Async REST + WebSocket server |
| Uvicorn | latest | ASGI server |
| prometheus-client | 0.25.0 | Metrics exposition |
| SQLite | built-in | KPI data persistence |

### Frontend
| Technology | Version | Role |
|---|---|---|
| React | 18 | UI framework |
| D3.js | latest | Network topology map |
| Recharts | latest | KPI time-series charts |
| TailwindCSS | latest | Utility-first styling |
| Vite | latest | Build tool + hot reload |

### DevOps
| Technology | Role |
|---|---|
| Docker | Containerization |
| Docker Compose | Multi-service orchestration |
| AWS EC2 | Cloud deployment |
| NGINX | Reverse proxy + static file serving |

---

## 📁 Project Structure

```
5g-network-digital-twin/
│
├── config.py                          ← All simulation constants (single source of truth)
├── requirements.txt
│
├── simulation/                        ← Core 5G network simulation
│   ├── engine.py                      ← NetworkSimulation class, SimPy event loop
│   ├── gnb.py                         ← gNB base station model (PRB allocation, load)
│   ├── ue.py                          ← UE model (mobility, 5QI profiles, handover)
│   ├── channel.py                     ← Vectorized SINR + path loss (NumPy)
│   └── mobility.py                    ← Random Waypoint mobility model
│
├── kpi/                               ← KPI computation and storage
│   ├── calculator.py                  ← Per-tick KPI calculation
│   ├── data_generator.py              ← 10,800-row dataset with congestion injection
│   └── storage.py                     ← SQLite persistence layer
│
├── ml/                                ← All machine learning models
│   ├── lstm_model.py                  ← PyTorch LSTM (input=18, hidden=64, layers=2)
│   ├── xgboost_model.py               ← XGBoost classifier wrapper
│   ├── ensemble.py                    ← 0.6×LSTM + 0.4×XGBoost weighted ensemble
│   ├── data_preprocessor.py           ← 18 features, StandardScaler pipeline
│   ├── train.py                       ← Training script (LSTM + XGBoost)
│   ├── anomaly_detector.py            ← IsolationForest wrapper
│   └── shap_explainer.py              ← XGBoost gain-based feature importance
│
├── optimizer/                         ← Reinforcement learning
│   ├── rl_env.py                      ← Custom Gymnasium env (9-dim obs, Discrete(4))
│   ├── agent.py                       ← PPO training wrapper (Stable-Baselines3)
│   └── rule_based_agent.py            ← Threshold policy for A/B testing baseline
│
├── api/                               ← FastAPI backend
│   ├── main.py                        ← All endpoints: WS, REST, Prometheus, Replay
│   └── run.py                         ← Uvicorn launcher
│
├── dashboard/                         ← React frontend
│   └── src/
│       ├── App.jsx                    ← Panel routing
│       ├── context/SimContext.jsx     ← Global state (useReducer)
│       ├── hooks/useSimSocket.js      ← WebSocket connection + auto-reconnect
│       └── components/
│           ├── layout/
│           │   ├── Sidebar.jsx        ← Navigation
│           │   └── TopBar.jsx         ← Tick counter + Replay controls
│           └── panels/
│               ├── OverviewPanel.jsx  ← System-level KPI summary
│               ├── NetworkMapPanel.jsx← D3 live map, UEs colored by 5QI profile
│               ├── KPIPanel.jsx       ← Per-cell throughput/latency/load charts
│               ├── PredictionPanel.jsx← LSTM+XGBoost congestion predictions
│               ├── ShapPanel.jsx      ← Feature importance waterfall chart
│               ├── AnomalyPanel.jsx   ← IsolationForest score gauge + chart
│               ├── ABTestPanel.jsx    ← PPO vs rule-based reward comparison
│               └── RLAgentPanel.jsx   ← PPO action distribution
│
├── models/                            ← Trained model artifacts
│   ├── lstm_best.pt                   ← Best LSTM checkpoint (PyTorch)
│   ├── xgboost_model.json             ← XGBoost booster
│   ├── scaler.pkl                     ← Fitted StandardScaler
│   ├── ppo_agent.zip                  ← Trained PPO policy
│   └── anomaly_detector.pkl           ← Trained IsolationForest
│
├── data/
│   └── kpi_dataset.csv                ← 10,800 rows, 22 columns
│
├── reports/
│   ├── phase4_results.json            ← ML evaluation metrics
│   ├── phase5_results.json            ← RL training results
│   └── shap_summary.png               ← Feature importance visualization
│
├── scripts/
│   ├── train_anomaly_detector.py      ← One-time anomaly model training
│   └── train_rl_agent.py              ← PPO training script
│
└── tests/                             ← 24 passing unit tests
    ├── test_simulation.py
    ├── test_kpi.py
    ├── test_phase4.py
    └── test_phase5.py
```

---

## 🔬 Core Components — Deep Dive

### 1. 5G Network Simulation Engine

**File:** `simulation/engine.py`, `simulation/gnb.py`, `simulation/ue.py`, `simulation/channel.py`

The simulation models a **1km × 1km urban grid** with 3 gNB base stations and 20 mobile UEs. It runs as a SimPy discrete-event simulation advancing in 1-second ticks for 3 hours (10,800 ticks total).

**gNB (Base Station) Configuration:**

```
gNB-0: position (200m, 500m)   ← West sector
gNB-1: position (500m, 200m)   ← South sector  
gNB-2: position (800m, 700m)   ← East sector

TX Power:       43 dBm
Antenna Gain:   15 dB
Max PRBs:       100 (Physical Resource Blocks)
Carrier:        3.5 GHz (5G NR Sub-6 band)
```

**SINR Calculation (channel.py):**

```
Path Loss (dB) = 20·log₁₀(d) + 20·log₁₀(f) + 92.4 + 10·α·log₁₀(d/1000)

Where:
  d = distance between UE and gNB (meters)
  f = carrier frequency = 3.5 GHz
  α = path loss exponent = 3.5 (urban dense environment)

Received Power = TX_Power + Antenna_Gain - Path_Loss

SINR = Received_Power_from_Serving_gNB
       ─────────────────────────────────────────────────────
       Noise_Power + Σ(Received_Power_from_Interfering_gNBs)

Clipped to: [SINR_MIN = -6 dB, SINR_MAX = 30 dB]
```

**UE (User Equipment) Properties:**

Each UE is assigned a **5QI traffic profile** at initialization, aligned with 3GPP TS 23.501:

| Profile | 5QI | Demand Range | Speed Factor | Population Weight |
|---|---|---|---|---|
| Video Streaming | 2 | 10–20 Mbps | 1.0× | 35% |
| Gaming | 3 | 4–10 Mbps | 0.3× | 30% |
| IoT Sensor | 5 | 0.1–1 Mbps | 0.5× | 20% |
| VoIP | 1 | 0.5–2 Mbps | 0.8× | 15% |

**Mobility Model:** Random Waypoint (RWP) — each UE picks a random destination and moves toward it at up to 3 m/s. On arrival, it pauses briefly then picks a new waypoint.

**Handover Logic:** A UE triggers a handover when another gNB offers SINR that is more than 3 dB higher than the current serving gNB (hysteresis-based handover, standard in LTE/5G).

---

### 2. KPI Engine & Data Generation

**File:** `kpi/calculator.py`, `kpi/data_generator.py`

Every simulation tick produces the following KPIs per cell:

| KPI | Formula | Unit |
|---|---|---|
| Cell Load | `allocated_PRBs / max_PRBs` | % (0.0–1.0) |
| Cell Throughput | `Σ(UE throughput)` for all UEs on this cell | Mbps |
| Cell Latency | `15 + 85 × load` (linear model) | ms |
| Handover Rate | `handover_events / num_UEs` | events/UE |
| Packet Loss Rate | `UEs_below_SINR_threshold / total_UEs` | % |

The **data generator** runs the full 10,800-tick simulation and injects **artificial congestion events** to create a balanced training dataset:
- Normal operation: cell load fluctuating 20–70%
- Warning events: load spike to 70–90% for 30–60 ticks
- Critical events: load spike to 90–100% for 10–20 ticks

This produces `data/kpi_dataset.csv` — **10,800 rows × 22 columns** — with a class balance of approximately 88% normal / 12% congested.

---

### 3. ML Congestion Prediction — LSTM + XGBoost Ensemble

**Files:** `ml/lstm_model.py`, `ml/xgboost_model.py`, `ml/ensemble.py`, `ml/data_preprocessor.py`

#### Feature Engineering (18 features)

```python
FEATURE_COLUMNS = [
    # Per-cell load (primary signal)
    'cell0_load', 'cell1_load', 'cell2_load',
    # Per-cell throughput (saturation indicator)
    'cell0_throughput', 'cell1_throughput', 'cell2_throughput',
    # Per-cell UE count (density pressure)
    'cell0_ue_count', 'cell1_ue_count', 'cell2_ue_count',
    # Per-cell average SINR (radio quality)
    'cell0_avg_sinr', 'cell1_avg_sinr', 'cell2_avg_sinr',
    # System-wide aggregates
    'system_throughput', 'system_avg_sinr', 'system_avg_latency_ms',
    # Event counters
    'handover_count', 'handover_rate', 'packet_loss_rate'
]
```

#### LSTM Architecture

```
Input:  (batch=1, sequence=10 ticks, features=18)
         ↓
LSTM Layer 1: hidden_size=64, dropout=0.3
         ↓
LSTM Layer 2: hidden_size=64, dropout=0.3
         ↓
Linear(64 → 1)
         ↓
Sigmoid → congestion probability ∈ [0, 1]
```

The LSTM sees the **last 10 ticks** as a sequence — it learns temporal patterns like "load has been rising for 5 ticks → congestion likely in 30 ticks." This is the key advantage over point-in-time classifiers.

#### XGBoost Configuration

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=7.3,   # handles 88/12 class imbalance
    eval_metric='logloss'
)
```

#### Ensemble (ensemble.py)

```
Final Probability = 0.6 × LSTM_probability + 0.4 × XGBoost_probability
```

The weights were tuned empirically: LSTM contributes more because its sequence modeling captures temporal drift that XGBoost misses. XGBoost provides strong precision anchoring.

#### Results

| Model | F1 Score | AUC-ROC | Precision | Recall |
|---|---|---|---|---|
| LSTM alone | 0.829 | — | — | — |
| XGBoost alone | 0.847 | 0.971 | — | — |
| **Ensemble** | **0.871** | **0.984** | **1.000** | 0.771 |

**Precision = 1.0** means the ensemble has **zero false positives** — it never raises a congestion alarm when the network is healthy. This is the critical metric for production use: false alarms erode operator trust.

---

### 4. Reinforcement Learning Agent (PPO)

**Files:** `optimizer/rl_env.py`, `optimizer/agent.py`

#### Custom Gymnasium Environment

```python
class FiveGNetworkEnv(gymnasium.Env):
    observation_space = Box(low=0, high=1, shape=(9,), dtype=np.float32)
    # [cell0_load, cell1_load, cell2_load,
    #  cong_prob0, cong_prob1, cong_prob2,
    #  ue_ratio0, ue_ratio1, ue_ratio2]

    action_space = Discrete(4)
    # 0 = NoOp           — network is healthy, no action
    # 1 = LoadBalance    — soft handover: move UEs from busiest to least loaded cell
    # 2 = MassBalance    — aggressive rebalancing across all cells
    # 3 = EmergencyHandover — immediate hard handover for critical overload
```

#### Reward Function

```python
reward = 0.0
for load in cell_loads:
    if load < 0.70:   reward += 0.2   # healthy — reward stability
    elif load < 0.90: reward -= 0.5   # warning — penalize moderate overload
    else:             reward -= 1.0   # critical — heavy penalty

reward += max(0.0, 0.1 - std(cell_loads))  # bonus for balanced load distribution
reward = clip(reward, -1.0, 1.0)
```

#### Training

```
Algorithm:     PPO (Proximal Policy Optimization)
Steps:         200,000
Policy:        MlpPolicy (2×64 hidden layers)
Learning rate: 3e-4
Clip range:    0.2
```

The agent was trained on the full KPI dataset with randomized initial states to ensure generalization across all network conditions.

---

### 5. Anomaly Detection (IsolationForest)

**File:** `ml/anomaly_detector.py`

The IsolationForest provides a **second line of defense** beyond the LSTM — it detects anomalies that the supervised model wasn't trained on (zero-day failure modes, hardware faults, unusual traffic spikes).

```python
IsolationForest(
    n_estimators=100,
    contamination=0.05,   # expects 5% of data to be anomalous
    random_state=42
)
```

**Severity thresholds:**
- `score > -0.1` → **Normal** (green)
- `-0.2 < score ≤ -0.1` → **Warning** (yellow)  
- `score ≤ -0.2` → **Critical** (red)

Trained on the full 10,800-row KPI dataset. Scores every live tick in the simulation loop.

---

### 6. A/B Testing Framework

**File:** `optimizer/rule_based_agent.py`, integrated in `api/main.py`

Every tick, both agents observe the **same 9-dimensional state** and produce an action + reward:

**Rule-Based Agent Policy:**
```python
if max(loads) > 0.90:      action = EmergencyHandover (3)
elif max(loads) > 0.80:    action = MassBalance (2)
elif spread > 0.30:        action = LoadBalance (1)
else:                      action = NoOp (0)
```

**Key finding from live A/B tests:** The PPO agent consistently chooses `NoOp` when the network is healthy (correct), while the rule-based agent frequently triggers `LoadBalance` unnecessarily (overreacting to normal variance). This demonstrates the RL agent has learned a more nuanced policy than simple threshold rules.

---

### 7. FastAPI Backend & WebSocket

**File:** `api/main.py`

The backend is a single FastAPI application with a **lifespan startup sequence** that loads all components in order:

```
1. NetworkSimulation (SimPy engine)
2. StandardScaler (scaler.pkl)
3. LSTM + XGBoost Ensemble (lstm_best.pt + xgboost_model.json)
4. PPO Agent (ppo_agent.zip)
5. Anomaly Detector (anomaly_detector.pkl)
6. Rule-Based Agent (pure logic, no file)
7. Auto-start simulation loop (asyncio background task)
```

**WebSocket tick payload (every 1 second):**

```json
{
  "tick": 1432,
  "timestamp": 1776234785.953,
  "cells": [
    {"cell_id": 0, "load_percent": 0.49, "throughput_mbps": 437.11,
     "latency_ms": 56.65, "connected_ues": 7, "prb_used": 49}
  ],
  "ues": [
    {"ue_id": 3, "x": 208.0, "y": 317.0, "connected_cell": 0,
     "sinr_db": 7.96, "throughput_mbps": 57.19, "is_handover": false,
     "traffic_profile": "Gaming", "qos_class": 3}
  ],
  "kpis": {"total_throughput": 1456.03, "mean_latency": 31.2, "handover_count": 0, "active_ues": 20},
  "congestion_predictions": {"0": 0.12, "1": 0.08, "2": 0.31},
  "ppo_actions": {"0": 2, "1": 0, "2": 0},
  "anomaly": {"anomaly_score": 0.3852, "is_anomaly": false, "severity": "normal"},
  "ab_comparison": {"tick": 1432, "ppo_action": 0, "ppo_reward": 0.6, "rb_action": 1, "rb_reward": 0.3}
}
```

---

### 8. React Live Dashboard

**Directory:** `dashboard/src/`

8 panels, all driven by a single WebSocket connection via `useSimSocket.js` and global state in `SimContext.jsx`:

| Panel | What It Shows | Key Technology |
|---|---|---|
| **Overview** | Total throughput, mean latency, per-cell load bars | Recharts |
| **Network Map** | Live UE positions, gNB coverage, handover events, 5QI color coding | D3.js |
| **KPIs** | Per-cell throughput/latency/load time-series | Recharts |
| **Predictions** | LSTM+XGBoost congestion probability per cell | Recharts |
| **SHAP** | Feature importance waterfall chart (XGBoost gain) | Recharts |
| **Anomaly** | IsolationForest score gauge + time-series | Recharts |
| **A/B Test** | PPO vs rule-based reward comparison chart + action distribution | Recharts |
| **RL Agent** | PPO action distribution bar chart | Recharts |

**UE color coding on the Network Map:**
- 🔵 Blue = Video (5QI-2)
- 🟢 Green = Gaming (5QI-3)
- 🟡 Yellow = IoT (5QI-5)
- 🟣 Purple = VoIP (5QI-1)
- 🔴 Red stroke = Active handover event

---

## 📊 Key Metrics & Results

### ML Performance

| Metric | Value |
|---|---|
| Dataset size | 10,800 rows |
| Class balance | 88% normal / 12% congested |
| LSTM F1 | 0.829 |
| Ensemble F1 | **0.871** |
| Ensemble AUC-ROC | **0.984** |
| Ensemble Precision | **1.000** (zero false positives) |
| Ensemble Recall | 0.771 |
| Prediction horizon | 30 ticks (30 seconds ahead) |

### Simulation Parameters

| Parameter | Value |
|---|---|
| Grid size | 1,000m × 1,000m |
| Number of gNBs | 3 |
| Number of UEs | 20 |
| TX Power | 43 dBm |
| Carrier frequency | 3.5 GHz |
| Path loss exponent | 3.5 (urban) |
| Noise power | −104 dBm |
| Simulation duration | 10,800 seconds (3 hours) |
| Tick duration | 1 second |
| SINR range | −6 dB to +30 dB |

### RL Training

| Parameter | Value |
|---|---|
| Algorithm | PPO (Proximal Policy Optimization) |
| Training steps | 200,000 |
| Observation dimensions | 9 |
| Action space | Discrete(4) |
| Policy network | MLP 2×64 |

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

### ML Inference (Model as a Service)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Send KPI snapshot → get congestion probability from live ensemble |
| `POST` | `/api/agent/action` | Send 9-dim observation → get PPO action + rule-based comparison |

**Example — `/api/predict`:**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cell0_load": 0.85,
    "cell1_load": 0.40,
    "cell2_load": 0.30,
    "cell0_throughput": 200,
    "cell1_throughput": 400,
    "cell2_throughput": 300,
    "cell0_ue_count": 9,
    "cell1_ue_count": 6,
    "cell2_ue_count": 5
  }'

# Response:
# {
#   "system_congestion_probability": 0.3412,
#   "cell_congestion_probabilities": {"0": 0.5613, "1": 0.2641, "2": 0.1981},
#   "prediction_horizon_ticks": 30,
#   "model": "lstm_xgboost_ensemble_0.6_0.4"
# }
```

**Example — `/api/agent/action`:**

```bash
curl -X POST http://localhost:8000/api/agent/action \
  -H "Content-Type: application/json" \
  -d '{"cell0_load": 0.91, "cell1_load": 0.30, "cell2_load": 0.25}'

# Response:
# {
#   "action": 2,
#   "action_label": "MassBalance",
#   "rule_based_action": 3,
#   "rule_based_label": "EmergencyHandover"
# }
```

### Observability

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/metrics` | Prometheus scrape endpoint (11 live gauges, standard text format) |

**Sample Prometheus output:**
```
# HELP cell_load_percent PRB load per cell
# TYPE cell_load_percent gauge
cell_load_percent{cell_id="0"} 0.49
cell_load_percent{cell_id="1"} 0.29
cell_load_percent{cell_id="2"} 0.21
# HELP anomaly_score IsolationForest anomaly score
anomaly_score 0.3852
# HELP system_throughput_mbps Total system throughput (Mbps)
system_throughput_mbps 1451.75
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

- Python 3.11+
- Node.js 18+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/poornachandran2006/5G_Digital_Twin.git
cd 5G_Digital_Twin
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
simpy==4.1.1
numpy
torch
xgboost
stable-baselines3
gymnasium
scikit-learn
fastapi
uvicorn
prometheus-client==0.25.0
```

### 3. Install Frontend Dependencies

```bash
cd dashboard
npm install
cd ..
```

### 4. Verify Trained Models Exist

The following model artifacts must be present in `models/`:

```
models/
├── lstm_best.pt          ← PyTorch LSTM checkpoint
├── xgboost_model.json    ← XGBoost booster
├── scaler.pkl            ← Fitted StandardScaler
├── ppo_agent.zip         ← Trained PPO policy
└── anomaly_detector.pkl  ← Trained IsolationForest
```

If any are missing, regenerate them:

```bash
# Generate dataset and train ML models
python -m kpi.data_generator
python -m ml.train

# Train anomaly detector
python scripts/train_anomaly_detector.py

# Train RL agent (takes ~10 minutes)
python scripts/train_rl_agent.py
```

---

## ▶️ How to Run

### Start the Backend

```bash
cd api
python run.py
```

The server starts at `http://localhost:8000`. On startup you'll see:

```
✓ NetworkSimulation initialised
✓ Scaler loaded
✓ Ensemble (LSTM + XGBoost) loaded on cpu
✓ PPO agent loaded
✓ Anomaly detector loaded
✓ Rule-based agent initialized
✓ Simulation loop auto-started
```

### Start the Frontend

In a new terminal:

```bash
cd dashboard
npm run dev
```

Dashboard available at **http://localhost:5173**

### Verify Everything Is Running

```bash
# Health check
curl http://localhost:8000/health

# Expected:
# {"status":"ok","mode":"real","ensemble_loaded":true,"ppo_loaded":true}
```

---

## 🖥 Dashboard Walkthrough

### Overview Panel
Shows system-wide KPIs at a glance: total throughput (Mbps), mean latency (ms), and a per-cell load bar for all 3 gNBs. This is the first thing an NOC operator sees.

### Network Map Panel
Live D3.js visualization of the 1km×1km grid. gNBs shown as hexagonal cells with coverage radius indicators. UE dots move in real time, colored by traffic profile (blue=Video, green=Gaming, yellow=IoT, purple=VoIP). Red stroke indicates an active handover event. Lines connect each UE to its serving gNB.

### KPIs Panel
Three tabbed time-series charts (Throughput / Latency / Cell Load) for all 3 cells. Shows the last 60 ticks. Use this to spot per-cell trends and divergence.

### Predictions Panel
LSTM+XGBoost ensemble congestion probability (0–1) per cell, updated every tick. The prediction is for **30 ticks in the future** — giving operators a 30-second warning to act before congestion occurs.

### SHAP Panel
XGBoost gain-based feature importance. Shows which of the 18 input features most influences the model's congestion predictions. Cell load features typically dominate, followed by UE count and SINR.

### Anomaly Panel
IsolationForest anomaly score gauge (−1 to 0, higher = more normal) plus time-series chart with severity coloring. Catches unknown failure modes that the supervised LSTM wasn't trained on.

### A/B Test Panel
Real-time comparison of PPO agent vs rule-based agent. Reward chart per tick, action distribution bars, and aggregate win rate. Demonstrates that the RL agent is measurably better than a simple threshold policy.

### RL Agent Panel
Distribution of PPO actions (NoOp / LoadBalance / MassBalance / EmergencyHandover) over the last 100 ticks. A healthy network should show mostly NoOp with occasional LoadBalance.

### Replay Controls (TopBar)
1. Click **● REC** — starts recording all tick data into a buffer
2. Click **■ STOP REC** — freezes the buffer
3. Select speed (1×/2×/4×/8×)
4. Click **▶ PLAY** — replays through all 8 panels simultaneously
5. **⏪ REPLAY** badge appears during playback

---

## ⚡ 5G Physics — The Science Behind It

### Why Path Loss Exponent = 3.5?

In free space, signal power decreases as `1/d²` (exponent = 2). In urban environments, buildings, reflections, and diffraction increase this to 3.0–4.5. We use **3.5** as a standard urban dense environment value, consistent with 3GPP TR 38.901 UMa (Urban Macro) channel model.

### Why 3.5 GHz?

3.5 GHz (n78 band) is the primary 5G NR band globally — deployed by Jio (India), Deutsche Telekom (Germany), SK Telecom (Korea). It offers a balance between coverage (better than mmWave) and capacity (better than sub-1GHz). This makes the simulation directly relevant to real deployed networks.

### Why −104 dBm Noise Power?

```
Thermal noise = kTB
k = 1.38×10⁻²³ J/K  (Boltzmann constant)
T = 290 K            (room temperature)
B = 20 MHz           (5G NR channel bandwidth)

kTB = 1.38×10⁻²³ × 290 × 20×10⁶ = 8×10⁻¹⁴ W = -101 dBm

With 3 dB noise figure: -101 - 3 = -104 dBm ✓
```

### SINR to Throughput Mapping

Using Shannon's capacity theorem adapted for 5G NR:

```
Throughput ≈ B × log₂(1 + SINR_linear)

Where B = allocated bandwidth (proportional to PRBs allocated to the UE)
```

---

## 🧠 ML Architecture — How the Models Work

### Why LSTM for Congestion Prediction?

Congestion is a **temporal phenomenon** — it builds over time. A snapshot of current load (60%) doesn't tell you if it's rising (will hit 90% in 30 seconds) or falling (was at 80%, now recovering). LSTM's hidden state captures this trajectory across the last 10 ticks.

### Why XGBoost in the Ensemble?

XGBoost is excellent at capturing **feature interactions** — it can learn rules like "when cell0_load > 0.7 AND ue_count > 8, congestion probability spikes." LSTM misses these sharp threshold boundaries. The ensemble gets the best of both.

### Why Precision = 1.0 Matters More Than Recall

In telecom NOC operations, a false alarm (predicting congestion when the network is fine) forces engineers to investigate, potentially triggering unnecessary handovers that **degrade service quality**. Precision=1.0 means every alarm is a real event. The 77% recall means we miss some events — acceptable for a proactive system that can still react reactively.

### Why IsolationForest for Anomaly Detection?

The LSTM is a **supervised** model — it can only detect congestion patterns it saw in training data. IsolationForest is **unsupervised** — it learns the normal distribution and flags anything outside it, including hardware failures, unusual traffic spikes, or coordinated attacks that never appeared in training data.

---

## 💼 Interview Q&A — What Recruiters Will Ask

**Q: What is a digital twin in the context of 5G?**  
A: A software replica of a physical network that mirrors its behavior in real time, enabling experimentation, prediction, and optimization without touching live infrastructure. This project simulates 3 gNBs and 20 UEs using real 3GPP physics — path loss exponent 3.5, 3.5 GHz carrier, SINR-based handover — so conclusions drawn from it are directly applicable to deployed networks.

**Q: How does your SINR calculation work?**  
A: I compute path loss using the 3GPP urban macro model: `PL = 20log(d) + 20log(f) + 92.4 + 10·3.5·log(d/1000)`. Received power = TX power + antenna gain − path loss. SINR = serving cell received power divided by the sum of noise power and all interfering cell powers. This is vectorized with NumPy for all 20 UEs simultaneously.

**Q: Why use an ensemble instead of just LSTM?**  
A: LSTM captures temporal drift (load trends over 10 ticks) but misses sharp threshold-based patterns. XGBoost captures feature interactions and threshold rules but has no memory. The 60/40 ensemble achieves F1=0.871 and AUC=0.984 versus 0.829 for LSTM alone — a 5% F1 improvement that translates to significantly fewer missed congestion events in production.

**Q: How do you know your RL agent is better than a simple rule?**  
A: I run both agents in parallel every tick (A/B test). The PPO agent consistently chooses NoOp when loads are healthy — demonstrating it learned that unnecessary interventions have a cost. The rule-based agent over-triggers LoadBalance on normal variance. Over 300 ticks, PPO achieves a higher cumulative reward, quantifiable from the A/B Test dashboard panel.

**Q: How do you explain your model's decisions?**  
A: I use XGBoost's built-in gain-based feature importance, accessible at `/api/shap/explanation`. Cell load features dominate (as expected), followed by UE count per cell and system latency. The SHAP panel in the dashboard shows a waterfall chart updated every tick, so operators can see in real time which features are driving the current prediction.

**Q: How do you detect failure modes the model wasn't trained on?**  
A: IsolationForest anomaly detection runs on every tick alongside the LSTM. Since it's unsupervised, it flags any observation that's statistically unusual — even if the LSTM labels it as "normal." This covers hardware faults, coordinated interference, or unusual traffic patterns that never appeared in the training dataset.

**Q: How do you monitor this in production?**  
A: The `/metrics` endpoint exposes 11 live gauges in Prometheus text format — cell load, throughput, latency, UE count, congestion probability, anomaly score, PPO reward, rule-based reward, system throughput, handover counter, and simulation tick. Any Prometheus scraper (Grafana, Datadog, Victoria Metrics) can connect and create dashboards without any code changes.

**Q: Is your simulation 3GPP-compliant?**  
A: The traffic profiles are 3GPP TS 23.501-aligned — each UE is assigned a 5QI (Video=5QI-2, Gaming=5QI-3, IoT=5QI-5, VoIP=5QI-1) with corresponding demand ranges and QoS class. The radio model uses the 3GPP TR 38.901 UMa path loss formula. The handover logic uses SINR hysteresis as specified in 3GPP TS 38.331.

---

## 🐳 Deployment

### Docker Compose (Local)

```bash
docker-compose up --build
```

Exposes:
- Frontend: `http://localhost:80`
- Backend: `http://localhost:8000`

### AWS EC2 Deployment

1. Launch EC2 instance (Ubuntu 22.04, t2.medium or larger)
2. Install Docker and Docker Compose
3. Clone repo and run `docker-compose up -d`
4. Open ports 80 and 8000 in EC2 security group
5. Access dashboard at `http://<your-ec2-ip>`

---

## 🗺 Roadmap

| Phase | Status | Description |
|---|---|---|
| Phase 1: Scaffold | ✅ Done | Project structure, config, dependencies |
| Phase 2: Simulation | ✅ Done | SimPy engine, gNB, UE, SINR, mobility |
| Phase 3: KPI Engine | ✅ Done | Calculator, data generator, 10,800-row dataset |
| Phase 4: ML Models | ✅ Done | LSTM, XGBoost, ensemble, SHAP |
| Phase 5: RL Agent | ✅ Done | Gymnasium env, PPO, 200K training steps |
| Phase 6: API + Dashboard | ✅ Done | FastAPI WS, 8 React panels, Prometheus |
| Phase 7: Docker + AWS | 🔄 In Progress | Containerization, EC2 deployment |
| Improvements: SHAP Panel | ✅ Done | Live feature importance dashboard |
| Improvements: Anomaly Detection | ✅ Done | IsolationForest, AnomalyPanel |
| Improvements: A/B Testing | ✅ Done | PPO vs rule-based, ABTestPanel |
| Improvements: 5QI Profiles | ✅ Done | 3GPP-aligned traffic profiles on map |
| Improvements: Prometheus | ✅ Done | 11 live gauges, standard scrape format |
| Improvements: REST ML APIs | ✅ Done | /api/predict, /api/agent/action |
| Improvements: Replay Mode | ✅ Done | Record/play buffer, adjustable speed |

---

## 👤 Author

**Poornachandran**  
3rd Year ECE Engineering Student  
Building industry-grade telecom systems from first principles.

> *"Every component in this project is grounded in actual 5G physics, real ML engineering practices, and production observability patterns — not toy examples."*

---

<div align="center">

**Target Companies:** Ericsson · Nokia · Jio · Qualcomm · Samsung Networks

<br/>

*If this project helped you understand 5G digital twins, give it a ⭐*

</div>
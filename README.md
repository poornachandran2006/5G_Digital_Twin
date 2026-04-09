# 5G Network Digital Twin with AI-Powered Congestion Prediction and RL Optimization

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-in--development-yellow.svg)

A software-only, industry-grade 5G Network Digital Twin that simulates real urban cellular networks, predicts congestion using machine learning, and optimizes load balancing through reinforcement learning.

## Overview

This project implements a complete digital twin of a 5G cellular network with:
- Physics-based simulation of 3 gNB base stations and 20 mobile UEs
- Real 3GPP-compliant SINR calculations using path loss models
- Time-series KPI tracking (cell load, throughput, latency, handover rate)
- LSTM-based congestion prediction (30 ticks ahead, 90%+ accuracy target)
- PPO reinforcement learning agent for dynamic load balancing
- Real-time visualization dashboard with WebSocket streaming
- Production-ready deployment with Docker and AWS EC2

## System Architecture

*Architecture diagram coming soon*

## Tech Stack

- **Simulation**: Python 3.11, SimPy 4.1.1, NumPy
- **Machine Learning**: PyTorch (LSTM), XGBoost, SHAP, Stable-Baselines3 (PPO), Gymnasium
- **Backend**: FastAPI, WebSocket, SQLite/InfluxDB
- **Frontend**: React 18, D3.js, Recharts, TailwindCSS
- **DevOps**: Docker, Docker Compose, AWS EC2, NGINX

## Installation

*Installation instructions coming soon*

## Usage

*Usage instructions coming soon*

## Results

*Performance metrics and visualizations coming soon*

## Project Structure

```
5g-network-digital-twin/
├── simulation/      # Core network simulation engine
├── kpi/            # KPI calculation and storage
├── ml/             # Machine learning models
├── optimizer/      # Reinforcement learning agent
├── api/            # FastAPI backend
├── dashboard/      # React frontend
├── data/           # Simulation data and logs
├── models/         # Trained ML models
├── tests/          # Unit tests
└── config.py       # Configuration constants
```

## License

MIT License - see LICENSE file for details

---

**Author**: Engineering Portfolio Project  
**Contact**: [Your contact information]  
**Last Updated**: 2026-04-09

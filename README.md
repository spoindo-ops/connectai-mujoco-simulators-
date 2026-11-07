# MuJoCo Reinforcement Learning Simulation Suite (Physical AI & Gymnasium)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo](https://img.shields.io/badge/Physics%20Engine-MuJoCo%203.x-red.svg)](https://mujoco.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Environment-Farama%20Gymnasium-green.svg)](https://gymnasium.farama.org/)

Autonomous physical AI simulation and reinforcement learning environment built on **Google DeepMind MuJoCo** and **Farama Gymnasium**. Designed for deterministic physics validation, dynamic contact damping, and PPO-driven autonomous navigation.

---

## 🏛️ Project & Corporate Metadata

- **Author / Developer:** SEONG JIN (`spoindo`)
- **Lab / Project:** Connect AI LAB · Advanced Physical AI Engineering
- **Technical Supervisor:** Seong Jin AI Executive Officer
- **Version:** v1.0.0-release
- **Standard Date:** 2026-11-07

---

## 🚀 Architecture Overview

This repository provides an end-to-end Physical AI simulation pipeline:

```
4기공부방 MUJOKO 시뮬레이터(2026년 11월분)/
├── 01_농구공_바운스/               # Stage 1: FIBA Standard Basketball Contact Damping
│   ├── world.xml                  # MJCF Model (Radius: 0.12m, Mass: 0.62kg, Restitution: 0.82)
│   └── run.py                     # Real-time 60FPS Interactive Physics Viewer
├── 02_1000톤_강철공/              # Stage 2: Hyper-compressed Solid Steel Extreme Mass
│   ├── world.xml                  # High-density Steel (32.8kg) to 1,000-Ton Inelastic Impact
│   └── run.py                     # Kinetic Energy Dissipation Visualizer
├── 03_공에이전트_자율주행_아레나/     # Stage 3: Autonomous Ball Agent & Gymnasium Arena
│   ├── world.xml                  # 16m x 16m Bound Arena + 100kg Titanium Ball Agent
│   ├── ball_nav_env.py            # Official Gymnasium Environment (`BallNav-v0`)
│   ├── train_ball_ppo.py          # Proximal Policy Optimization (PPO) Training Engine
│   ├── setup.py                   # Custom Package Packaging Configuration
│   └── run.py                     # Trained Policy Deployment & Viewer
└── 20261107_강화학습_08강_...md   # Master Technical Specification Report
```

---

## ⚙️ Core Technical Specifications

### 1. Contact Dynamics & Damping Parameters
- **Basketball Simulation:** FIBA Size 7 규격 (`radius="0.12"`, `mass="0.62"`). 충돌 안정성을 위해 `solref="-15000 -12"`, `solimp="0.9 0.95 0.001"` 댐핑 파라미터 적용.
- **1,000-Ton Extreme Solid:** 완전 비탄성 충돌(`restitution=0.0`) 및 극한의 운동 에너지 감쇠를 구현하여 지면 침하 없는 즉각적 에너지 소산 실현.
- **Titanium Agent:** 100kg 구체 질량 및 $120\,\text{Nm}$ 정밀 액추에이터 토크 제한으로 고속 회전 시 슬립 현상 방지 및 안정적 궤적 추종.

### 2. Markov Decision Process (MDP) Definition
- **Observation Space ($\mathcal{S} \in \mathbb{R}^8$):**
  $$[x, y, v_x, v_y, x_{target}, y_{target}, \Delta x, \Delta y]$$
- **Action Space ($\mathcal{A} \in \mathbb{R}^2$):**
  $$[\tau_x, \tau_y] \in [-1.0, 1.0] \times 120.0\,\text{Nm}$$
- **Reward Function ($R_t$):**
  $$R_t = R_{goal}(+10.0) + R_{out\_of\_bounds}(-5.0) + 2.0 \times (D_{t-1} - D_t) - 0.01 \|\mathbf{a}_t\|^2$$

---

## 📦 Quick Start & Local Execution

### 1. Installation
```bash
pip install mujoco gymnasium stable-baselines3
cd "03_공에이전트_자율주행_아레나"
pip install -e .
```

### 2. Gymnasium Standard Invocation
```python
import gymnasium as gym
import ball_nav_env

env = gym.make('BallNav-v0', render_mode='human')
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### 3. PPO Training Execution
```bash
python train_ball_ppo.py
```

---

## 📜 Intellectual Property & Standards

- **Developer:** SEONG JIN (`spoindo`)
- **License:** Apache License 2.0


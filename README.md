# Adaptive Reinforcement Learning Agent via Theory of Mind Profiling

An advanced Deep Reinforcement Learning paradigm featuring an agent that dynamically discovers, clusters, and adapts to unseen opponent strategies in competitive environments. Inspired by ToMnet principles, the architecture isolates tactical environmental states from latent behavioral profiling.

---

## Overview

Modern reinforcement learning agents excel in stationary environments but display severe fragility when exposed to dynamic or unseen opponent strategies. Traditional counter-strategies often rely on hardcoded rule adjustments or massive self-play loops that overfit to specific strategic trajectories.

This project aims to automate dynamic adaptation.

Instead of treating opponent behavior as an unobservable environment element, we leverage an isolated Recurrent Opponent Profiler that maps opponent histories into a continuous embedding space, serving as a real-time diagnostic and strategic adjustment vector for a Proximal Policy Optimization (PPO) core.

The project investigates whether latent descriptions of enemy playstyles can explicitly guide downstream action-selection loops to achieve zero-shot generalization against completely novel strategies.

---

## Motivation

An RL agent trained on a game like Rock-Paper-Scissors with mixed strategies or Connect Four consistently relies on:
* Over-aggressive opening sequences
* Reactive tit-for-tat patterns
* Periodic structural trap placements
* Exploitative shifts after a loss

Usually, identifying these patterns requires manually updating reward weights or tracking statistical rule counts. 

Our goal is to automatically discover these hidden strategies and isolate them into functional latent profiles, adapting behavior mid-match without modifying core networks.

The calculated profiles are combined with structural game states to dynamically shift the agent’s policy real-time.

## Project Pipeline

Game Environment Initialization (API Call)
│
▼
Expose Agent to Diverse Opponent Pool (Rule-Based ➔ Fixed Bots ➔ Self-Play)
│
▼
Stream Dual Observations (Game States & Historic Opponent Actions)
│
▼
Extract Latent Profile Vector (z_opp via GRU/LSTM Profiler)
│
▼
Predict Opponent Actions & Group Visual Clustered Profiles (t-SNE / PCA)
│
▼
Condition Policy & Value Networks with z_opp
│
▼
PPO Clipped Surrogate Update Loop
│
▼
Evaluate Adaptation Speed & Win Rate on Held-Out Unseen Opponents


---

## Objectives

* **Train** baseline rule competency against classic procedural frameworks.
* **Isolate** opponent action histories to build distinct behavioral features.
* **Represent** strategy samples using fixed-size latent embeddings.
* **Discover** structural clusters of semantically grouped opponent profiles.
* **Automatically adapt** policy weight responses without fine-tuning networks inline.
* **Evaluate** if structural behavioral profiling yields zero-shot generalizations on unseen opponents.

---

## Features

### Automated Strategy Profiling
Detect recurring opponent behavioral tendencies without manually tagging playstyle types.

### Explainable Theory of Mind
Convert opaque structural play dynamics into visible spatial clusters grouping identical tactical variations together.

### Targeted Dataset Expansion
Track agent capabilities across an incremental curriculum of structural complexity (from simple games to highly complex multi-agent environments).

---

## Datasets & Environments

Supported environments pulled across interactive frameworks:
* **PettingZoo** (Multi-agent baseline tasks)
* **OpenSpiel** (Game-theoretic analysis environments)
* **OpenAI Gym** / **Gymnasium**
* **Kaggle Environments**

---

## Technology Stack

### Languages
* Python

### Deep Learning & Reinforcement Learning
* PyTorch
* Stable-Baselines3 (Custom PPO Core)
* TensorBoard

### Environment Frameworks
* PettingZoo
* OpenSpiel

### Mathematical Analysis & Visualization
* scikit-learn (t-SNE / K-Means Clustering)
* NumPy
* Matplotlib
* Seaborn

---

## Project Structure

adaptive-rl/
│
├── data/              # Saved game logs and strategy configurations
├── notebooks/         # EDA and strategy cluster visualizations
├── models/            # Model weights and network checkpoints
│   ├── profiler/      # GRU / LSTM Opponent Profilers
│   ├── policy/        # PPO Actor-Critic weights
│   └── baselines/     # Fixed strategy bot parameters
│
├── profiling/         # Sequential history sequence encoding scripts
├── ppo_agent/         # Custom PPO adaptation and rollout memory code
├── environments/      # Environment connection wraps and APIs
├── evaluation/        # Validation scripts checking win rates vs unseen bots
├── utils/             # Metric trackers and plotting tools
├── results/           # Generated clustering graphs and metrics
└── README.md          # Project documentation


---

## Curriculum References

Our training pipelines and architecture choices match current academic deep RL guidelines:
* **Foundational RL Formulations:** Stanford CS234 (Reinforcement Learning)
* **Deep Architectural Scaling:** Stanford CS224r (Deep Reinforcement Learning)
* **Evolutionary Strategy Mapping:** Deep Learning Papers Reading Roadmap Repo

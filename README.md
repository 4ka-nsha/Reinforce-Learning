# Adaptive Reinforcement Learning Agent via Theory-of-Mind Opponent Profiling

A three-module framework for opponent-conditioned policy learning in competitive,
imperfect-information games. A GRU-based Opponent Profiler compresses an opponent's recent action
history into a latent embedding `z_opp`, which conditions a PPO actor-critic's policy and value
networks alongside the raw game state.

## Overview

Standard single-policy reinforcement learning agents perform well in stationary environments but
degrade against opponents whose strategies differ from those seen during training, because the
policy has no explicit representation of *who it is currently playing against*. This project
tests whether giving a PPO agent an explicit, learned opponent embedding improves adaptation and
generalization in competitive, imperfect-information games, relative to a PPO agent without one.

## Pipeline

\`\`\`
OpenSpiel Game Load (curriculum stage)
        │
        ▼
Sample Opponent (rule-based / fixed-strategy bot pool, or self-play checkpoint)
        │
        ▼
Stream Dual Observations (game state & rolling opponent-action history)
        │
        ▼
Extract Latent Profile Vector (z_opp via GRU profiler, L2-normalized)
        │
        ▼
Condition Policy & Value Networks with z_opp (concatenated input)
        │
        ▼
Joint PPO Update (clipped surrogate + profiler auxiliary next-action loss, GAE-λ advantages)
        │
        ▼
Evaluate: win rate & t-SNE cluster separation vs. held-out opponents
\`\`\`

## Modules

- **Environment & Curriculum** — `environments/`. OpenSpiel-backed, four-stage curriculum
  (Tic-Tac-Toe → Kuhn Poker → Leduc Poker → Connect Four), seven-bot opponent pool, rotating
  self-play checkpoint pool.
- **Opponent Profiler** — `models/profiler/`. Single-layer GRU encoder over a rolling window of
  the opponent's last 10 one-hot actions, producing a 32-dim L2-normalized `z_opp`, trained with a
  next-action auxiliary loss and validated by t-SNE/PCA cluster visualization.
- **PPO Policy & Training Loop** — `ppo_agent/`. A game-agnostic actor-critic MLP conditioned on
  `[game state, z_opp]`, trained with a clipped-surrogate objective and GAE-λ advantage estimation,
  jointly optimized with the profiler's auxiliary loss.

## Curriculum & Opponent Pool

Four sequential, two-player OpenSpiel games, in order: **Tic-Tac-Toe → Kuhn Poker → Leduc Poker →
Connect Four**. Network dimensions (`max_observation_dim`, `max_action_dim`) are derived once from
the curriculum rather than hardcoded, so one set of network shapes works across all four games.

Seven opponent archetypes across two tiers:

| Tier | Bots |
|---|---|
| Rule-based | RandomBot, GreedyBot |
| Fixed-strategy / personality | AggressiveBot, DefensiveBot, MirrorBot, PeriodicBot, ExploitativeBot |

## Results

A trained joint checkpoint has been evaluated against all seven opponent archetypes at every
curriculum stage (50 games/bot, 75 for Connect Four).

| Stage | Best matchup | Worst matchup | Cluster separation |
|---|---|---|---|
| Tic-Tac-Toe | MirrorBot 60.0% | DefensiveBot 12.0% | Rich structure, not archetype-pure |
| Kuhn Poker | PeriodicBot 64.0% | MirrorBot 46.0% | Collapses to ~3 points (structural) |
| Leduc Poker | DefensiveBot 66.0% | AggressiveBot 38.0% | ExploitativeBot separates; other 6 don't |
| Connect Four | MirrorBot 80.0% | DefensiveBot 6.7% | Rich structure, not archetype-pure |

Across all four stages, the agent plays competently against reactive/non-adaptive opponents
(RandomBot, MirrorBot, GreedyBot) and consistently struggles against opponents built around active
counterplay — DefensiveBot and ExploitativeBot — most sharply in Connect Four. The profiler's
t-SNE cluster-separation exit criterion (one clean cluster per archetype) is not met at any stage
yet: Kuhn Poker's short episodes collapse the latent space almost entirely; Leduc Poker separates
one archetype from the rest but not all seven; Tic-Tac-Toe and Connect Four produce dense spatial
clustering that doesn't track opponent identity as cleanly as it tracks something else (likely
game-state/trajectory structure).

## Evaluation

`evaluation/eval.py` plays `--games_per_bot` (default 50) evaluation games per opponent archetype
against a saved checkpoint, reporting win rate per archetype and feeding `z_opp` history into a
t-SNE/PCA cluster visualizer (`models/profiler/analysis.py`).

## Project Structure

Reinforce-Learning/
│
├── environments/       # Curriculum ladder, OpenSpiel env wrapping, opponent pool
├── models/
│   ├── baselines/       # Rule-based & fixed-strategy bot implementations
│   └── profiler/        # GRU opponent profiler + t-SNE/PCA cluster analysis
├── ppo_agent/           # PPO actor-critic, buffer, training loop, self-play snapshots
├── evaluation/          # Win-rate evaluation vs. held-out bots + latent visualization
├── tests/               # pytest suite
├── example_usage.py
└── requirements.txt

## Technology Stack

- **Language:** Python
- **RL / Deep Learning:** PyTorch, custom PPO core
- **Environment framework:** OpenSpiel
- **Analysis / visualization:** scikit-learn (t-SNE / PCA), Matplotlib, Seaborn
- **Testing:** pytest

## References

- Foundational RL Formulations: Stanford CS234 (Reinforcement Learning)
- Deep Architectural Scaling: Stanford CS224R (Deep Reinforcement Learning)
- Machine Theory of Mind (Rabinowitz et al., 2018); Proximal Policy Optimization (Schulman et al.,
  2017); Generalized Advantage Estimation (Schulman et al., 2015); OpenSpiel (Lanctot et al., 2019)

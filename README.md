# Adaptive Reinforcement Learning Agent via Theory-of-Mind Opponent Profiling

An opponent-conditioned PPO agent for competitive, imperfect-information games. Instead of treating the opponent as an unobserved part of the environment, a GRU-based **Opponent Profiler** encodes an opponent's recent behavior into a latent embedding, `z_opp`, which conditions a PPO actor-critic's policy and value heads alongside the game state.

The central question this project is built to answer: **Does conditioning a PPO policy on a learned opponent-profile embedding produce measurably better adaptation and generalization against novel opponents than an equivalent PPO agent trained without it?**

## How it works

```
OpenSpiel game init → sample opponent (bot pool / self-play checkpoint)
    → stream [game state, opponent action history]
    → GRU profiler extracts z_opp (L2-normalized)
    → policy & value heads condition on [state, z_opp]
    → PPO clipped-surrogate update (GAE-λ), profiler trained jointly
    → evaluate win rate & z_opp cluster separation on held-out opponents
```

The system is three independently-owned modules:

| Module | Contents | Key file |
|---|---|---|
| Environment & Curriculum | Game arena, opponent pool, curriculum ladder, Gymnasium interface | `environments/env_wrapper.py` |
| Opponent Profiler | GRU encoder producing `z_opp`, trajectory tracking, auxiliary loss, cluster evaluation | `models/profiler/models.py` |
| PPO Policy | Actor-critic, rollout buffer, clipped-surrogate update, self-play snapshots | `ppo_agent/model.py` |

## Curriculum & opponent pool

Training proceeds through a three-stage difficulty ladder: **Kuhn Poker → Leduc Poker → Connect Four**. Every stage evaluates against the same seven bot archetypes:

`random` · `greedy` · `aggressive` · `defensive` · `mirror` · `periodic` · `exploitative`

`RandomBot` and `GreedyBot` form the rule-based tier used to establish basic competency; the remaining five are fixed-strategy personalities (2-ply lookahead for `DefensiveBot`, action-history mirroring for `MirrorBot`, cross-episode state for `PeriodicBot` and `ExploitativeBot`) used to teach the profiler to tell opponents apart.

## Repository structure

```
Reinforce-Learning/
├── environments/
│   ├── curriculum.py       # Curriculum ladder; derives max_action_dim / max_observation_dim
│   └── env_wrapper.py      # AdaptiveOpponentEnv — the Gymnasium interface
├── models/
│   ├── baselines/
│   │   ├── bots.py         # Seven-bot opponent pool
│   │   └── base_bot.py     # find_winning_action, opponent_has_winning_reply
│   ├── profiler/
│   │   ├── models.py       # OpponentProfiler (GRU encoder + auxiliary head)
│   │   ├── tracker.py      # TrajectoryTracker (rolling opponent-action window)
│   │   └── analysis.py     # LatentVisualizer (t-SNE / PCA cluster plots)
│   └── checkpoints/        # joint_<stage>_model.pt (gitignored)
├── ppo_agent/
│   ├── model.py             # PPOActorCritic
│   ├── buffer.py            # Rollout buffer
│   ├── ppo.py                # GAE-λ advantages + clipped-surrogate update
│   ├── model_snapshot.py     # Rotating self-play checkpoint pool
│   └── train.py               # Training entry point
├── evaluation/
│   └── eval.py                 # Per-archetype win rate + cluster evaluation
├── tests/
│   └── test_environment.py     # 32 pytest cases across 7 bots × 4 stages
├── results/                    # Generated win-rate tables & cluster plots (gitignored)
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/4ka-nsha/Reinforce-Learning.git
cd Reinforce-Learning
git checkout main
py -m pip install -r requirements.txt
```

## Usage

**Train** a joint profiler + PPO checkpoint on a curriculum stage:

```bash
py -m ppo_agent.train --stage kuhn_poker --episodes 4000 --rollout_size 128 --lambda_aux 0.6
```

**Evaluate** a checkpoint against all seven opponent archetypes:

```bash
py -m evaluation.eval --stage kuhn_poker --checkpoint models/checkpoints/joint_kuhn_poker_model.pt --games_per_bot 50
```

This writes a per-archetype win-rate table and a t-SNE projection of the collected `z_opp` vectors to `results/`.

**Run the test suite:**

```bash
py -m pytest tests/
```

## Results at a glance

Trained checkpoints evaluated against all seven archetypes, every stage (50 games/bot; 75 for Connect Four):

| Stage | Best matchup | Worst matchup |
|---|---|---|
| Kuhn Poker | `periodic` — 64.0% | `mirror` — 46.0% |
| Leduc Poker | `defensive` — 66.0% | `aggressive` — 38.0% |
| Connect Four | `mirror` — 80.0% | `defensive` — 6.7% |

The agent plays competently against reactive, non-adaptive bots (`random`, `mirror`, `greedy`) and consistently struggles against the two archetypes built around active counterplay — `defensive` and `exploitative` — most sharply in Connect Four.

The profiler's cluster-separation exit criterion (distinct, archetype-pure clusters in projected `z_opp` space) is **not yet met at any stage**: Kuhn Poker collapses toward a handful of points for structural reasons tied to its short episodes; Leduc Poker separates one archetype (`exploitative`) from the rest but not all seven; Connect Four produces rich spatial clustering that tracks game-state/trajectory structure at least as strongly as opponent identity.

## Current status

- **Environment**: Validated — 32/32 tests passing, ~1.8×–4× throughput improvement from profiling-driven optimization, confirmed picklable for parallel rollout collection.
- **Profiler + PPO mechanics**: Validated — both run end-to-end against real environment data across all four stages without numerical or shape errors.
- **Per-archetype win rate**: Measured at every stage on real evaluation runs.
- **Profiler cluster separation**: Measured, exit criterion not yet met (see Results above).
- **Ablation baseline** (PPO without `z_opp`): **not yet implemented** — No code path currently exists to drop `z_opp` from the policy input. This is the single highest-priority open item, since it's what the project's central research question depends on.

## Known gaps

No ablation configuration yet — blocks the profiled-vs-baseline comparison this project is ultimately meant to answer.

## Contributors

Akansha Das · Ayush Mishra · Archit Singal

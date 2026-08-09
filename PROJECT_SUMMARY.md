# MARL Project — Complete Summary
**Multi-Agent Deep Reinforcement Learning on PettingZoo MPE**
UCLA Mathematics Department · Advisor: Professor Tao Gao

---

## Overview

Built a complete research codebase implementing and extending the **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** algorithm (Lowe et al. 2017) in PyTorch, evaluated on nine PettingZoo MPE environments. The project's primary contribution is a novel **Pre-Trained Action Inference (PTAI)** mechanism that substantially improves multi-agent coordination in competitive environments. The work culminated in a full academic paper with quantitative multi-seed experimental results.

---

## Repository

**GitHub**: https://github.com/shaashwathsivakumar/MARL_Proj  
**Language**: Python · **Framework**: PyTorch  
**Committed**: 1,208 files including model checkpoints, reward logs, training curves, and paper

---

## Core MADDPG Implementation (`maddpg.py`, `networks.py`)

Implemented the full MADDPG algorithm from scratch in PyTorch using the **Centralised Training, Decentralised Execution (CTDE)** paradigm:

- **Actor networks**: 2-hidden-layer MLP (obs_dim → 64 → 64 → action_dim), Xavier-uniform initialisation, 0.01 bias constants. Discrete action selection via **Gumbel-Softmax** (differentiable) during training and argmax at evaluation.
- **Critic networks**: Centralised MLP taking the full joint observation-action vector as input (all agents' obs + all agents' actions → 64 → 64 → 1).
- **Target networks**: Maintained for both actor and critic; Polyak soft-updated at τ = 0.02 each learning step.
- **Replay buffer**: 1,000,000-step capacity; 1,024-sample minibatches.
- **Gradient clipping**: Global norm clipped at 0.5 for all networks.
- **Actor regularisation**: L2 logit penalty (1e-3 × mean(logits²)) added to actor loss to prevent policy collapse.
- **Training schedule**: 30,000 episodes; 50,000 warmup steps (random actions before any gradient updates); gradient updates every 100 steps; learning rate 0.01 for both actor and critic (Adam optimiser).
- **Checkpointing**: Model weights and reward logs saved at episodes 10,000, 20,000, and 30,000.

---

## Algorithm Extensions (all implemented as composable flags in `train.py`)

Six algorithmic extensions were implemented on top of baseline MADDPG, each independently testable and combinable:

### 1. Pre-Trained Action Inference — PTAI (`PTAI.py`) ← **Primary Contribution**

**Motivation**: Standard MADDPG actors see only the current local observation. In competitive settings, knowing what other agents did one step ago resolves ambiguous states and enables reactive strategies. PTAI makes this temporal context available to each actor without requiring communication.

**Architecture — AI_Net**:
- A dedicated neural network (`AI_Net`) takes two consecutive local observations (o_t, o_{t-1}) for agent i and outputs an estimate of the **full previous joint action vector** â_i = concat(onehot(a₁^{t-1}), …, onehot(aₙ^{t-1})).
- The observation vector is structurally **separable** into segments corresponding to different agent types. AI_Net exploits this to decompose inference into two types of lightweight specialised modules instead of one monolithic O(N²) network:
  - **Directional Social Awareness Modules**: one per (observer-type, observed-type) pair. Takes features of the observed agent visible to observer i plus self/global features, all at both time steps and as temporal differences. Predicts the observed agent's previous action.
  - **Directional Self-Awareness Module**: one per observer type. Takes self/global features and their temporal differences. Predicts the agent's own previous action.
- Observation sub-assignments (which observation vector indices belong to which agent) are hardcoded per environment for all 9 MPE environments via config dictionaries in `PTAI.py`.

**Pre-training**:
- AI_Net is pre-trained **offline** on 200 episodes of uniformly random interaction (all agents sample actions uniformly from their action spaces) using supervised MSE loss between predicted and true one-hot action vectors.
- A stochastic gate randomly drops a fraction of (observer, observed) training pairs per step for regularisation.
- After pre-training, **all AI_Net weights are frozen** for the entirety of MADDPG training. No online updates.

**Integration with MADDPG**:
- At each decision step t ≥ 2, the frozen AI_Net is called: â_i = AI_Net(o_t, o_{t-1}).
- The actor receives the extended input [o_t ‖ â_i] (concatenation), expanding its input dimension by Σ|A_k|.
- At t = 1 (no previous observation), â_i is replaced by a zero vector of the same dimension.
- The centralised **critic is unchanged** — it still receives [all_obs, all_current_actions]. Only the actor gains temporal context.
- Previous observations are stored in the replay buffer to allow AI_Net to be queried during the critic/actor update steps as well.

**Environments supported**: All 9 MPE environments, each with a bespoke sub-assignment config. Configs defined for: simple_v3, simple_spread_v3, simple_adversary_v3, simple_crypto_v3, simple_push_v3, simple_reference_v3, simple_speaker_listener_v4, simple_tag_v3, simple_world_comm_v3.

**Pre-trained checkpoints**: Saved as `AI_Net_<env_name>.pt` in the project root; auto-loaded or auto-trained at run start.

---

### 2. Geometric Replay Sampling (`buffer.py`)

**Motivation**: Uniform replay sampling in non-stationary multi-agent settings underrepresents recent, policy-relevant transitions.

**Implementation**: Replay buffer indices are sampled from a **geometric distribution** P(X = k) = (1-p)^{k-1} · p used as reverse indices (k=1 maps to most-recent entry). Higher p produces a sharper recency bias. Default p = 1e-5.

**Flag**: `--use_geometric_sampling`, tunable via `--geo_alpha`.

---

### 3. Shared Actor Networks (`maddpg.py`)

**Motivation**: Agents on the same team observe structurally identical information. Weight sharing reduces parameters and encourages policy consistency within teams.

**Implementation**: A single actor network is shared across all agents of the same team (agent team and adversary team separately). The shared actor and its target are maintained with a single shared Adam optimiser. Falls back to individual actors for environments with heterogeneous observation dimensions (`simple_speaker_listener_v4`, `simple_world_comm_v3`).

**Flag**: `--shared_actor`

---

### 4. Previous Joint Action Conditioning for Critic (`maddpg.py`, `buffer.py`)

Appends the true previous joint action vector to the centralised critic's input, expanding critic input dimension by Σ|A_k|. The previous joint action is stored in the replay buffer alongside each transition.

**Flag**: `--use_prev_action`

---

### 5. Previous Observation Conditioning for Actor (`maddpg.py`)

Appends the agent's previous observation to the actor's current observation input: actor receives [o_{t-1} ‖ o_t]. This provides the actor with an explicit velocity/displacement signal.

**Flag**: `--use_prev_observation`

---

### 6. TD3-Style Twin Critic (`maddpg.py`)

**Implementation**: Each agent gets two independently-initialised critic networks (critic1, critic2). The TD target uses the minimum of both target critics' Q-value predictions, reducing Q-value overestimation bias (Fujimoto et al. 2018). Actor and target-network updates are delayed to every `--policy_delay` (default 2) calls to `update()`. Target policy smoothing (the third TD3 component) is intentionally omitted — it has no clean analog for Gumbel-Softmax discrete actions.

**Flag**: `--use_twin_critic`, tunable via `--policy_delay`

---

### 7. Adversary Gating — Experimental (`maddpg.py`)

**Implementation**: Each opponent agent's observation and action features in the centralised critic input are scaled by a scalar "trust gate" in (0, 1]. The gate is derived from the KL-like divergence between the opponent's batch-mean action distribution in the current update step and a slow-moving exponential moving average (EMA) of past batch-mean distributions. When an opponent's policy is changing rapidly (high drift), its critic contribution is downweighted. Gate is computed per-update, applied element-wise to the opponent's slice of the critic input tensor. No-op for environments without an opposing team.

**Flag**: `--use_adversary_gating`, tunable via `--gating_temperature` and `--gating_ema_decay`

---

## Algorithm Naming System

Algorithm names are auto-built from active flags: e.g. `maddpg_shared_actor_geometric_prev_action`. Override with `--algorithm <name>`. All results are stored under `results/<algorithm>/<env_name>/<run_num>/`.

---

## Environments (9 PettingZoo MPE)

All 9 MPE environments tested: `simple_v3`, `simple_adversary_v3`, `simple_crypto_v3`, `simple_push_v3`, `simple_reference_v3`, `simple_speaker_listener_v4`, `simple_spread_v3`, `simple_tag_v3`, `simple_world_comm_v3`.

Agent-team / adversary-team splits are consistent across all training scripts. Agents named `adversary_*` form the adversary team; agents named `agent_*`, `alice`, or `bob` form the agent team.

---

## Baseline Algorithms (`train_baseline.py`, `train_reinforce.py`)

For comparison, four additional algorithms were trained using Stable-Baselines3 and sb3-contrib with SuperSuit wrappers:

- **DQN** (Deep Q-Network)
- **PPO** (Proximal Policy Optimisation)
- **A2C** (Advantage Actor-Critic)
- **TRPO** (Trust Region Policy Optimisation, via sb3-contrib)
- **REINFORCE** (Vanilla policy gradient with parameter sharing; custom implementation in `train_reinforce.py`)

All baselines use a single shared policy across all agents (parameter sharing), unlike MADDPG's per-agent approach.

---

## Metrics System (`metrics.py`)

A per-environment metrics registry tracks game-event statistics from reward signals (no direct environment internals access):

| Environment | Metric | Description |
|---|---|---|
| simple_tag_v3 | `num_catches` | Predator-prey contact events per episode |
| simple_push_v3 | `agent_wins` | Fraction of episodes where agent reaches its goal at termination |
| simple_world_comm_v3 | `adversary_at_goal_frames` | Timesteps per episode where ≥1 adversary is at the goal landmark |
| simple_crypto_v3 | `bob_success` | Bob's decoding success rate |

---

## Evaluation Infrastructure

- **`evaluate.py`**: Loads a saved model checkpoint, runs N evaluation episodes, generates GIF animations (`gif/episode_N.gif`), and saves evaluation result plots.
- **`compare_results.py`**: Reads all `rewards.pkl` files across all runs/environments/algorithms, computes per-seed last-1000-episode means, prints formatted comparison tables, and optionally exports CSV and plots.
- **`plot_convergence.py`**: Generates 2×3 convergence plots for all algorithms across competitive environments, with 500-episode smoothing window and ±1 std shaded bands across seeds.
- **`plot_paper_convergence.py`**: Paper-specific 2×2 convergence figure (MADDPG vs PTAI only, 4 target environments).
- **`train_all_variants.py`**: Batch training script that launches multiple algorithm variants sequentially.

---

## Paper: "Enhancing the MADDPG Algorithm via Pre-Trained Action Inference in Competitive Multi-Agent Environments"

**Authors**: Shaash Sivakumar, Marc Walden, Jason Liu, Ryan Liu, Hamza Khan  
**Files**: `paper_ptai.tex` (LaTeX source), `paper_ptai.pdf` (compiled PDF)

### Experimental Design

- **Scope**: PTAI vs baseline MADDPG; 4 competitive environments; 3–4 independent seeds per (algorithm, environment) pair.
- **Evaluation**: Last 1,000 of 30,000 training episodes averaged per seed; cross-seed mean ± sample standard deviation reported.
- **Environments**: `simple_adversary_v3`, `simple_push_v3`, `simple_tag_v3`, `simple_world_comm_v3`.

### Final Results

| Environment | Metric | Baseline MADDPG (mean ± std) | MADDPG + PTAI (mean ± std) | Improvement |
|---|---|---|---|---|
| simple_adversary_v3 | Agent reward | 9.91 ± 0.40 (n=4) | 20.22 ± 2.82 (n=4) | **+104%** |
| simple_push_v3 | Agent win rate | 0.18 ± 0.08 (n=3) | 0.33 ± 0.03 (n=4) | **+83%** |
| simple_tag_v3 | Catches / episode | 0.62 ± 0.08 (n=3) | 3.31 ± 1.59 (n=4) | **+437%** |
| simple_world_comm_v3 | Adv. frames at goal | 10.42 ± 1.72 (n=3) | 26.78 ± 2.60 (n=4) | **+157%** |

### Individual Seed Data

**simple_adversary_v3 — MADDPG**: 9.94, 10.40, 9.88, 9.42  
**simple_adversary_v3 — PTAI**: 23.35, 21.79, 18.35, 17.38

**simple_push_v3 — MADDPG**: 0.27, 0.11, 0.16 (runs 2–4; run 1 predates metric system)  
**simple_push_v3 — PTAI**: 0.37, 0.33, 0.29, 0.33

**simple_tag_v3 — MADDPG**: 0.527, 0.666, 0.657 (runs 3, 5, 6; runs 1–2 empty, run 4 predates metric system)  
**simple_tag_v3 — PTAI**: 3.109, 5.437, 1.578, 3.123

**simple_world_comm_v3 — MADDPG**: 9.07, 12.36, 9.83 (runs 2–4; run 1 predates metric system)  
**simple_world_comm_v3 — PTAI**: 30.62, 26.07, 25.02, 25.42

### Paper Structure

1. Introduction (motivation, contributions)
2. Background (RL framework, DQN/MARL, MADDPG with equations)
3. Method — PTAI (AI_Net architecture, pre-training, MADDPG integration)
4. Environments (descriptions of all 4 competitive environments)
5. Experiments (hyperparameter table, evaluation protocol)
6. Results (quantitative table, convergence figure, per-environment analysis)
7. Discussion (why PTAI helps, random pre-training rationale, limitations)
8. Conclusion
9. References
10. Appendix — 3 algorithm pseudocodes (MADDPG, PTAI pre-training, MADDPG+PTAI)

---

## Results Directory Structure

```
results/
  <algorithm>/
    <env_name>/
      <run_num>/
        args.json           # all CLI flags used
        model.pt            # final policy weights
        rewards.pkl         # {per_agent, agent_score, adversary_score, agent_team, adversary_team, metrics}
        training_curves.png
        checkpoint_10000/   model.pt + rewards.pkl
        checkpoint_20000/   model.pt + rewards.pkl
        checkpoint_30000/   model.pt + rewards.pkl
        gif/episode_N.gif   # generated by evaluate.py
```

**Algorithms with results on disk**:
- `maddpg` — baseline (all 9 envs, 3–4 seeds each)
- `maddpg_ptai` — PTAI (competitive envs, 4 seeds each)
- `maddpg_adv_gating` — adversary gating (sweep across temperatures)
- `maddpg_geometric` — geometric sampling
- `maddpg_prev_action` — prev action critic conditioning
- `maddpg_ptai_online` — online AI_Net variant
- `maddpg_ptai_velA` — PTAI with analytical velocity features
- `maddpg_ptai_velL` — PTAI with learned velocity features
- `maddpg_twin_critic` — TD3-style twin critic

---

## Technologies & Dependencies

| Category | Tools |
|---|---|
| Core ML | Python, PyTorch, NumPy |
| Environments | PettingZoo MPE, Gymnasium |
| Baselines | Stable-Baselines3, sb3-contrib, SuperSuit |
| Visualisation | Matplotlib, Pillow (GIF generation) |
| Version control | Git, GitHub |
| Paper | LaTeX (source), Chrome headless (PDF generation) |

---

## Scale of Work

- **9** PettingZoo MPE environments
- **9** algorithm variants implemented
- **~30** training runs completed across 2 sessions (each 30,000 episodes)
- **1,208** files committed to GitHub (model weights, reward logs, figures)
- **760 MB** of training results (model checkpoints, reward histories, GIFs, plots)
- **6** distinct algorithmic extensions beyond baseline MADDPG
- **5** baseline algorithms for comparison (DQN, PPO, A2C, TRPO, REINFORCE)
- **1** full academic paper written end-to-end

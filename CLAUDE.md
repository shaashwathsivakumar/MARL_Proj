# MARL Project — MADDPG on PettingZoo MPE

## Project Overview
Research implementation of MADDPG (Multi-Agent Deep Deterministic Policy Gradient) on PettingZoo MPE environments. Includes several algorithmic extensions (geometric sampling, prev-action conditioning, shared actors, PTAI) and baseline algorithms (DQN, PPO, A2C, TRPO, REINFORCE) for comparison.

Reference paper: Lowe et al. 2017, arXiv:1706.02275.

## Source Files

| File | Purpose |
|------|---------|
| `maddpg.py` | Core MADDPG algorithm — `MADDPGAgent`, `MADDPG` classes |
| `networks.py` | Neural net architectures — `Actor`, `Critic` |
| `buffer.py` | `ReplayBuffer` (single agent), `MultiAgentReplayBuffer` |
| `PTAI.py` | Pre-Trained Action Inference — `AINet`, `Awareness`, `pre_train_ai_net` |
| `metrics.py` | Per-environment metrics trackers, `METRICS_REGISTRY` |
| `train.py` | MADDPG training entrypoint |
| `train_baseline.py` | SB3 baseline training (DQN/PPO/A2C/TRPO) |
| `train_reinforce.py` | Vanilla REINFORCE with parameter sharing |
| `evaluate.py` | Evaluate saved models; generate GIFs |
| `compare_results.py` | Load `rewards.pkl` across runs, print/plot tables |
| `train_all.sh` | Bash script to train MADDPG on all 9 envs sequentially |

## Environments (9 PettingZoo MPE)
`simple_v3`, `simple_adversary_v3`, `simple_crypto_v3`, `simple_push_v3`,
`simple_reference_v3`, `simple_speaker_listener_v4`, `simple_spread_v3`,
`simple_tag_v3`, `simple_world_comm_v3`

Agent-team / adversary-team splits are defined in `AGENT_TEAMS` dicts in each training script (all identical). Agent names beginning with `adversary` are the adversary team; names beginning with `agent`, `alice`, or `bob` are the agent team.

## Network Architecture
- **Actor**: MLP `obs_dim → 64 → 64 → action_dim`, Xavier-uniform init, 0.01 bias
  - Discrete actions via Gumbel-Softmax (exploration) or argmax (eval)
- **Critic**: MLP `(all_obs + all_actions [+ prev_joint_action]) → 64 → 64 → 1`
  - Centralized — sees every agent's observation and action
- Gradient clipping at 0.5 for all networks
- Actor regularization: `1e-3 * mean(logits²)` added to actor loss

## MADDPG Algorithm Variants (train.py flags)

| Flag | Effect |
|------|--------|
| `--shared_actor` | Share actor weights within each team (agent/adversary separately) |
| `--use_geometric_sampling` | Weight replay sampling by recency; `--geo_alpha` sets decay |
| `--use_prev_action` | Append previous joint action to critic input |
| `--use_prev_observation` | Append previous observation to actor input (`[prev_obs, obs]`) |
| `--use_ai_net` | Use frozen AINet to infer previous joint actions (**simple_tag_v3 only**) |
| `--use_adversary_gating` | **Experimental, unvalidated.** Scales each opponent agent's obs/action features in the centralized critic input by a "trust" gate derived from how far that opponent's batch action distribution has drifted from its own slow-moving EMA (`--gating_temperature` sets attenuation sensitivity, `--gating_ema_decay` sets EMA speed). No-op for agents with no opposing team. Composes with all other flags. |
| `--use_twin_critic` | TD3-style (Fujimoto et al. 2018) clipped double Q-learning: each agent gets two independently-initialized critics; TD target uses `min` of both targets (reduces Q-value overestimation bias); actor + target-network updates delayed to every `--policy_delay` (default 2) calls to `update()`. Target policy smoothing (the third TD3 component) is omitted — no clean analog for discrete Gumbel-Softmax actions. Composes with all other flags. |

Algorithm name is auto-built from flags, e.g. `maddpg_shared_actor_geometric_prev_action`.
Pass `--algorithm <name>` explicitly to override the auto-detected name.

**Incompatible combinations** (code handles gracefully):
- `--shared_actor` + `simple_speaker_listener_v4` or `simple_world_comm_v3` → falls back to individual actors (heterogeneous obs dims)
- `--use_ai_net` requires `--env_name simple_tag_v3`

## PTAI (Pre-Trained Action Inference)
Defined in `PTAI.py`. Predicts the **previous joint action** from an agent's current+previous observations.
- Architecture: `Awareness` modules (Social per agent-type pair, Self for own agent)
- Config hardcoded for `simple_tag_v3` via `SIMPLE_TAG_CONFIG` and `SIMPLE_TAG_SUB_ASSIGNMENTS`
- Pre-trained on 200 random episodes supervised (MSE), then **frozen** during MADDPG
- Saved/loaded via `AI_Net.pt` in the project root; auto-pre-trained if missing at train start

## Training Defaults (train.py)
```
num_episodes:        30000
warmup_steps:        50000   # random actions before any gradient step
learn_interval:      100     # steps between gradient updates
batch_size:          1024
buffer_capacity:     1000000
gamma:               0.95
tau:                 0.02    # Polyak soft-update coefficient
actor_lr / critic_lr: 0.01
checkpoint_interval: 10000   # episodes
log_interval:        100     # episodes
```

## Results Directory Structure
```
results/
  <algorithm>/
    <env_name>/
      <run_num>/
        args.json           # all CLI args used
        model.pt            # final weights
        rewards.pkl         # {per_agent, agent_score, adversary_score, agent_team, adversary_team, metrics}
        training_curves.png
        checkpoint_<ep>/
          model.pt
          rewards.pkl
        gif/
          episode_N.gif     # generated by evaluate.py
```

## Common Commands
```bash
# Activate venv (Windows)
myenv\Scripts\activate

# Train MADDPG baseline on one env
python train.py --env_name simple_spread_v3

# Train with extensions
python train.py --env_name simple_tag_v3 --use_ai_net --use_prev_observation

# Train with shared actor and geometric sampling
python train.py --env_name simple_tag_v3 --shared_actor --use_geometric_sampling

# Train all 9 envs sequentially (Linux/bash)
bash train_all.sh

# Train SB3 baseline
python train_baseline.py --algorithm dqn --env_name simple_spread_v3 --timesteps 750000

# Train REINFORCE
python train_reinforce.py --env_name simple_spread_v3 --num_episodes 30000

# Evaluate a run (generates GIFs)
python evaluate.py maddpg simple_spread_v3 1

# Compare all results
python compare_results.py
python compare_results.py --plot --csv results.csv

# Pre-train AINet standalone
python PTAI.py
```

## Dependencies
```
torch, numpy, pettingzoo[mpe], gymnasium, matplotlib, pillow
```
Baseline training additionally requires: `stable-baselines3`, `sb3-contrib`, `supersuit`

Virtual environments: `myenv/` (primary, contains torch+pettingzoo), `.venv/` (secondary).

## Known Issues / Notes
- `evaluate.py` imports `from algorithms import create_team_algorithm, get_available_algorithms` for mixed-algorithm evaluation. The `algorithms.py` module does not exist yet — mixed-algorithm evaluation will fail at import if attempted.
- `train_baseline.py` and `train_reinforce.py` use parameter sharing (one shared policy for all agents); they do not implement centralized training.
- Metrics in `metrics.py` use reward heuristics to approximate game events (catches, collisions) rather than accessing env internals directly — some metrics are approximations.

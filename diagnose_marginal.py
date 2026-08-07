"""
Diagnostic: is the on-policy accuracy jump for fellow-adversary prediction
real learning, or just a baseline-shift artifact?

Steps:
  1. Run on-policy rollouts with the trained MADDPG+PTAI (run 2) model.
  2. Compute the empirical marginal action distribution for each agent
     (how often each action was actually taken under the converged policy).
  3. Compute the "naive baseline accuracy" = max(marginal frequencies)
     i.e. what a contextless predictor that always guesses the most common
     action would score -- purely from the skewed distribution, zero inference.
  4. Compare naive baseline vs. AINet's actual per-cell accuracy.
  5. Report per-cell collapse ratio (does AINet output vary with input?).

Interpretation:
  naive_baseline ≈ on_policy_accuracy  => artifact, not real learning
  naive_baseline ≈ 0.20 but accuracy ≈ 0.30  => real instance-specific learning

Usage:
    python diagnose_marginal.py
"""
import numpy as np
import torch

from PTAI import get_env_ptai_config, _make_env
from diagnose_ainet import (
    compute_action_offsets, load_frozen_ai_net, load_trained_maddpg,
    PredictionScorer,
)

ENV_NAME = 'simple_tag_v3'
RUN_ID = '2'
NUM_EPISODES = 100
EPISODE_LENGTH = 25
DEVICE = 'cpu'


def run_onpolicy_with_marginal(env_name, episode_length, num_episodes,
                                ai_net, sub_assignments, cfg, offsets,
                                maddpg, obs_dims, device):
    env = _make_env(env_name, episode_length)
    agent_ids = env.possible_agents
    scorer = PredictionScorer(agent_ids, cfg, offsets, device)

    # action_counts[j][a] = how many times agent j took action a
    n_agents = len(agent_ids)
    action_sizes = [offsets[j][1] - offsets[j][0] for j in range(n_agents)]
    action_counts = [np.zeros(action_sizes[j], dtype=np.int64) for j in range(n_agents)]

    for _ in range(num_episodes):
        observations, _ = env.reset()
        last_observations = None
        last_actions = None
        prev_observations = {agent: np.zeros(obs_dims[agent], dtype=np.float32)
                             for agent in agent_ids}
        while env.agents:
            actions = maddpg.select_actions(observations, prev_observations, explore=False)
            next_observations, _, _, _, _ = env.step(actions)

            if last_observations is not None:
                scorer.score_step(ai_net, sub_assignments, observations,
                                  last_observations, last_actions)
                # tally the actions taken at `last_observations` (= last_actions)
                for j, agent_id in enumerate(agent_ids):
                    if agent_id in last_actions:
                        a = last_actions[agent_id]
                        if 0 <= a < action_sizes[j]:
                            action_counts[j][a] += 1

            prev_observations = {agent: observations[agent].copy() for agent in agent_ids}
            last_observations = {k: v.copy() for k, v in observations.items()}
            last_actions = dict(actions)
            observations = next_observations

    env.close()
    return scorer, action_counts, agent_ids


def main():
    print(f"=== On-policy marginal diagnostic: {ENV_NAME} run {RUN_ID} ===\n")

    ai_net, sub_assignments, cfg = load_frozen_ai_net(ENV_NAME, DEVICE)
    offsets = compute_action_offsets(cfg)

    probe_env = _make_env(ENV_NAME, EPISODE_LENGTH)
    probe_env.reset()
    agent_ids = probe_env.possible_agents
    obs_dims = {a: probe_env.observation_space(a).shape[0] for a in agent_ids}
    action_dims = {a: probe_env.action_space(a).n for a in agent_ids}
    probe_env.close()

    maddpg = load_trained_maddpg(
        ENV_NAME, RUN_ID, ai_net, sub_assignments,
        agent_ids, obs_dims, action_dims, DEVICE
    )

    print(f"Running {NUM_EPISODES} on-policy episodes (deterministic, explore=False)...")
    scorer, action_counts, agent_ids = run_onpolicy_with_marginal(
        ENV_NAME, EPISODE_LENGTH, NUM_EPISODES,
        ai_net, sub_assignments, cfg, offsets,
        maddpg, obs_dims, DEVICE
    )

    # Print AINet accuracy per cell (reuse scorer.report internals)
    n = len(agent_ids)
    ainet_acc = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if scorer.total[i, j] > 0:
                ainet_acc[i, j] = scorer.correct[i, j] / scorer.total[i, j]

    # Marginal distributions and naive baseline
    marginals = []
    naive_baseline = []
    for j in range(n):
        total = action_counts[j].sum()
        if total > 0:
            marginal = action_counts[j] / total
        else:
            marginal = np.ones(len(action_counts[j])) / len(action_counts[j])
        marginals.append(marginal)
        naive_baseline.append(float(marginal.max()))

    # Collapse ratios per observer
    collapse_ratios = scorer.collapse_ratios()

    # ---- Report ----
    print("\n--- Marginal action distributions (on-policy) ---")
    for j, agent_id in enumerate(agent_ids):
        total = action_counts[j].sum()
        dist_str = "  ".join(f"a{a}:{marginals[j][a]:.3f}" for a in range(len(marginals[j])))
        print(f"  {agent_id[:20]:<20}  [{dist_str}]  total_steps={total}")

    print("\n--- Naive baseline vs AINet accuracy per (observer, observed) cell ---")
    print(f"  {'observer':<20}  {'observed':<20}  {'naive_baseline':>14}  {'ainet_acc':>10}  {'gap':>8}  {'interpretation'}")
    print(f"  {'-'*20}  {'-'*20}  {'-'*14}  {'-'*10}  {'-'*8}  {'-'*25}")
    for i in range(n):
        for j in range(n):
            if scorer.total[i, j] == 0:
                continue
            nb = naive_baseline[j]
            aa = ainet_acc[i, j]
            gap = aa - nb
            if i == j:
                interp = "self"
            elif agent_ids[j] == 'agent_0':
                interp = "prey"
            elif agent_ids[i].startswith('adversary') and agent_ids[j].startswith('adversary'):
                interp = "fellow-adversary <<<" if abs(gap) > 0.03 else "fellow-adversary"
            else:
                interp = "other"
            print(f"  {agent_ids[i][:20]:<20}  {agent_ids[j][:20]:<20}  "
                  f"{nb:>14.3f}  {aa:>10.3f}  {gap:>+8.3f}  {interp}")

    print("\n--- Summary: fellow-adversary cells only ---")
    fa_naive = []
    fa_ainet = []
    for i in range(n):
        for j in range(n):
            if (i != j and scorer.total[i, j] > 0
                    and agent_ids[i].startswith('adversary')
                    and agent_ids[j].startswith('adversary')):
                fa_naive.append(naive_baseline[j])
                fa_ainet.append(ainet_acc[i, j])

    if fa_naive:
        print(f"  Mean naive baseline (fellow-adv):  {np.mean(fa_naive):.3f}")
        print(f"  Mean AINet accuracy (fellow-adv):  {np.mean(fa_ainet):.3f}")
        print(f"  Mean gap (AINet - naive):          {np.mean(np.array(fa_ainet) - np.array(fa_naive)):+.3f}")
        print(f"  Uniform chance level:               0.200")
        print()
        if np.mean(fa_naive) > 0.25:
            print("  => Naive baseline >> 0.20: on-policy action distribution IS skewed.")
            print("     On-policy 'improvement' is likely a baseline-shift artifact.")
        else:
            print("  => Naive baseline ≈ 0.20: on-policy action distribution is roughly uniform.")
        if np.mean(np.array(fa_ainet) - np.array(fa_naive)) > 0.05:
            print("  => AINet accuracy meaningfully exceeds naive baseline: evidence of real learning.")
        else:
            print("  => AINet accuracy does NOT exceed naive baseline: no real inference beyond mode-guessing.")

    print("\n--- Collapse ratios per observer (overall) ---")
    for i, agent_id in enumerate(agent_ids):
        print(f"  {agent_id[:20]:<20}  collapse_ratio={collapse_ratios[i]:.4f}")
    print("  (ratio << 1 => near-constant output regardless of input)")


if __name__ == '__main__':
    main()

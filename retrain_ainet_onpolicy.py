"""
Stage 3: Retrain AINet on on-policy data from the run-2 trained MADDPG model.

Motivation: pre_train_ai_net trains on random rollouts. Random actions produce
noisy, unstructured position deltas, so AINet collapses to predicting the marginal
action distribution for fellow-adversaries (gap vs naive baseline = 0.000).

On-policy data from a converged policy has structured, coordinated behavior, which
may provide learnable position-delta -> action signal for fellow-adversaries.

What this script does:
  1. Loads the run-2 MADDPG+PTAI model (results/maddpg_ptai/simple_tag_v3/2/).
  2. Creates a fresh AINet (random init — avoids the collapsed weights from prior training).
  3. Runs 200 on-policy episodes using maddpg.select_actions(explore=True) for action
     diversity (explore=True uses Gumbel-softmax sampling, matching training distribution).
  4. Trains the fresh AINet on those transitions — identical loop to pre_train_ai_net,
     same hyperparameters (lr=0.001, percent_to_train_on=0.8, max_episode_length=20).
     Critically: uses last_actions as the target (NOT actions) — same Stage-1 fix.
  5. Saves the retrained AINet to AI_Net_simple_tag_v3.pt (overwrites existing).
  6. Runs diagnostics (random rollout + on-policy rollout) to confirm whether
     fellow-adversary accuracy now exceeds the naive baseline.

Usage:
    python retrain_ainet_onpolicy.py
"""
import random

import numpy as np
import torch
import torch.nn as nn

from PTAI import AINet, get_env_ptai_config, _make_env, one_hot
from diagnose_ainet import (
    compute_action_offsets,
    load_frozen_ai_net,
    load_trained_maddpg,
    run_random_rollout,
    run_on_policy_rollout,
)

ENV_NAME          = 'simple_tag_v3'
RUN_ID            = '2'
EPISODE_COUNT     = 200
MAX_EPISODE_LEN   = 20
PCT_TRAIN         = 0.8
LR                = 0.001
SAVE_PATH         = 'AI_Net_simple_tag_v3.pt'
DEVICE            = 'cpu'
DIAG_EPISODES     = 50


def retrain_on_policy(env_name, maddpg, obs_dims, sub_assignments, cfg,
                      episode_count, max_episode_length, percent_to_train_on,
                      lr, device):
    """
    Train a fresh AINet on on-policy transitions from `maddpg`.
    Mirrors pre_train_ai_net exactly — only action selection and prev_obs tracking differ.
    """
    agents = list(obs_dims.keys())  # fixed ordering (env.possible_agents)

    fresh_ai_net = AINet(
        cfg['n_agent_types'], cfg['agent_types'],
        cfg['other_sub_sizes'], cfg['slfgbl_sub_sizes'],
        cfg['action_sub_sizes'], lr=lr, device=device,
    )

    env = _make_env(env_name, max_episode_length)

    for episode in range(episode_count):
        observations, _   = env.reset()
        last_observations = None
        last_actions      = None
        prev_observations = {a: np.zeros(obs_dims[a], dtype=np.float32) for a in agents}

        while env.agents:
            actions = maddpg.select_actions(observations, prev_observations, explore=True)
            next_observations, _, _, _, _ = env.step(actions)

            if last_observations is not None:
                for observer_ind, agent_name in enumerate(agents):
                    if (agent_name not in observations
                            or agent_name not in last_observations):
                        continue

                    observer_type = cfg['agent_types'][observer_ind]
                    sub_assign    = sub_assignments[observer_ind]

                    curr_obs = torch.tensor(
                        observations[agent_name], dtype=torch.float32
                    ).to(device)
                    last_obs = torch.tensor(
                        last_observations[agent_name], dtype=torch.float32
                    ).to(device)

                    slfgbl_mask = sub_assign == -1
                    curr_slfgbl = curr_obs[slfgbl_mask]
                    last_slfgbl = last_obs[slfgbl_mask]

                    for observed_ind, observed_name in enumerate(agents):
                        if observed_name not in last_actions:
                            continue
                        if random.random() > percent_to_train_on:
                            continue

                        observed_type = cfg['agent_types'][observed_ind]
                        action_size   = cfg['action_sub_sizes'][observed_type]
                        target = one_hot(last_actions[observed_name], action_size, device=device)

                        if observed_ind == observer_ind:
                            bundle = torch.cat([curr_slfgbl, last_slfgbl,
                                                curr_slfgbl - last_slfgbl])
                            a_pred = fresh_ai_net.self_modules[observer_type](bundle)
                            loss   = nn.MSELoss()(a_pred, target)
                            fresh_ai_net.self_modules[observer_type].update(loss)
                        else:
                            other_mask  = sub_assign == observed_ind
                            curr_other  = curr_obs[other_mask]
                            last_other  = last_obs[other_mask]
                            curr_bundle = torch.cat([curr_other, curr_slfgbl])
                            last_bundle = torch.cat([last_other, last_slfgbl])
                            bundle      = torch.cat([curr_bundle, last_bundle,
                                                     curr_bundle - last_bundle])
                            a_pred = fresh_ai_net.social_modules[observer_type][observed_type](bundle)
                            loss   = nn.MSELoss()(a_pred, target)
                            fresh_ai_net.social_modules[observer_type][observed_type].update(loss)

            prev_observations = {a: observations[a].copy() for a in agents}
            last_observations = {k: v.copy() for k, v in observations.items()}
            last_actions      = dict(actions)
            observations      = next_observations

        if (episode + 1) % 20 == 0:
            print(f"  On-policy AINet training: {episode + 1}/{episode_count} episodes")

    env.close()
    fresh_ai_net.eval_mode()
    return fresh_ai_net


def main():
    print(f"=== Stage 3: Retrain AINet on-policy ({ENV_NAME} run {RUN_ID}) ===\n")

    # Load existing AINet (needed to construct MADDPG with correct architecture)
    existing_ai_net, sub_assignments, cfg = load_frozen_ai_net(ENV_NAME, DEVICE)
    offsets = compute_action_offsets(cfg)

    # Probe env for dims
    probe_env = _make_env(ENV_NAME, MAX_EPISODE_LEN)
    probe_env.reset()
    agent_ids  = probe_env.possible_agents
    obs_dims   = {a: probe_env.observation_space(a).shape[0] for a in agent_ids}
    action_dims = {a: probe_env.action_space(a).n for a in agent_ids}
    probe_env.close()

    # Load run-2 MADDPG policy (used for on-policy data collection only)
    print(f"Loading run-{RUN_ID} MADDPG model ...")
    maddpg = load_trained_maddpg(
        ENV_NAME, RUN_ID, existing_ai_net, sub_assignments,
        agent_ids, obs_dims, action_dims, DEVICE,
    )

    # Retrain a fresh AINet on on-policy data
    print(f"\nTraining fresh AINet on {EPISODE_COUNT} on-policy episodes "
          f"(explore=True, max_ep_len={MAX_EPISODE_LEN}) ...")
    fresh_ai_net = retrain_on_policy(
        ENV_NAME, maddpg, obs_dims, sub_assignments, cfg,
        EPISODE_COUNT, MAX_EPISODE_LEN, PCT_TRAIN, LR, DEVICE,
    )

    # Save (overwrites AI_Net_simple_tag_v3.pt)
    torch.save(
        {'ai_net': fresh_ai_net.state_dict(), 'sub_assignments': sub_assignments},
        SAVE_PATH,
    )
    print(f"\nSaved on-policy-retrained AINet to {SAVE_PATH}")

    # ---- Diagnostics ----
    print(f"\n[Diagnostic 1/2] Random-rollout accuracy (baseline: how well random-rollout training worked)")
    random_scorer = run_random_rollout(
        ENV_NAME, MAX_EPISODE_LEN, DIAG_EPISODES,
        fresh_ai_net, sub_assignments, cfg, offsets, DEVICE,
    )
    random_acc, random_mse, random_ratio = random_scorer.report("ON-POLICY-TRAINED AINet -- RANDOM rollouts")

    print(f"\n[Diagnostic 2/2] On-policy rollout accuracy (key check: does it now exceed naive baseline?)")
    maddpg_for_diag = load_trained_maddpg(
        ENV_NAME, RUN_ID, fresh_ai_net, sub_assignments,
        agent_ids, obs_dims, action_dims, DEVICE,
    )
    onpolicy_scorer = run_on_policy_rollout(
        ENV_NAME, MAX_EPISODE_LEN, DIAG_EPISODES,
        fresh_ai_net, sub_assignments, cfg, offsets,
        maddpg_for_diag, obs_dims, DEVICE,
    )
    onpolicy_acc, onpolicy_mse, onpolicy_ratio = onpolicy_scorer.report("ON-POLICY-TRAINED AINet -- ON-POLICY rollouts")

    print(f"\n=== Summary ===")
    print(f"  random   rollout: accuracy={random_acc:.3f}  mse={random_mse:.4f}  collapse={random_ratio:.4f}")
    print(f"  on-policy rollout: accuracy={onpolicy_acc:.3f}  mse={onpolicy_mse:.4f}  collapse={onpolicy_ratio:.4f}")
    print()

    # Fellow-adversary specific check
    n = len(agent_ids)
    fa_naive = []
    fa_acc   = []
    for i in range(n):
        for j in range(n):
            if (i != j
                    and onpolicy_scorer.total[i, j] > 0
                    and agent_ids[i].startswith('adversary')
                    and agent_ids[j].startswith('adversary')):
                cell_acc = onpolicy_scorer.correct[i, j] / onpolicy_scorer.total[i, j]
                fa_acc.append(cell_acc)

    if fa_acc:
        mean_fa = np.mean(fa_acc)
        print(f"  Fellow-adversary on-policy accuracy (mean): {mean_fa:.3f}")
        print(f"  Naive baseline (mode-guessing, from diagnose_marginal): ~0.320")
        gap = mean_fa - 0.320
        print(f"  Gap above naive baseline: {gap:+.3f}")
        if gap > 0.03:
            print("  => IMPROVEMENT: on-policy AINet exceeds naive baseline for fellow-adversaries.")
            print("     Proceeding to run-3 is warranted.")
        else:
            print("  => NO IMPROVEMENT: on-policy AINet still collapses to mode for fellow-adversaries.")
            print("     Discuss before launching run-3.")


if __name__ == '__main__':
    main()

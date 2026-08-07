"""
Diagnostic: measure AINet (PTAI) previous-joint-action prediction quality
under two rollout distributions:

  1. "random"   — fresh random-action rollouts (matches AINet's pre-training
                  distribution: pre_train_ai_net samples env.action_space.sample()).
  2. "on_policy" — rollouts driven by the *trained* MADDPG+PTAI policy from
                  results/maddpg_ptai/<env_name>/1/model.pt.

For each rollout we step through transitions, ask the frozen AINet to predict
the previous joint action from (curr_obs, last_obs) — exactly as it's queried
during MADDPG action-selection / updates (maddpg.py select_actions / update) —
and compare the prediction's per-agent argmax against the *true* previous
action that was actually taken, plus MSE against the one-hot target AINet was
trained to regress to.

This isolates "does AINet predict well?" from "does a good prediction help?",
which is the evidence gate for whether PTAI's cooperative-env underperformance
is a distribution-shift problem (AINet pretrained on random play, deployed
against a converged on-policy distribution) or something else.

Usage:
    python diagnose_ainet.py --env_name simple_spread_v3
    python diagnose_ainet.py --env_name simple_tag_v3 --num_episodes 50
"""
import argparse
import os

import numpy as np
import torch

from PTAI import AINet, get_env_ptai_config, _make_env, one_hot
from maddpg import MADDPG


def compute_action_offsets(cfg):
    """Per-observed-agent (start, end) slices into AINet's concatenated output."""
    offsets = []
    start = 0
    for idx, agent_type in enumerate(cfg['agent_types']):
        size = cfg['action_sub_sizes'][agent_type]
        offsets.append((start, start + size))
        start += size
    return offsets


def load_frozen_ai_net(env_name, device):
    cfg_sub_assignments, cfg = get_env_ptai_config(env_name)
    path = f'AI_Net_{env_name}.pt'
    if not os.path.exists(path) and env_name == 'simple_tag_v3' and os.path.exists('AI_Net.pt'):
        # Legacy filename from before per-env naming was introduced (simple_tag_v3 was
        # the original/only PTAI target).
        path = 'AI_Net.pt'
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python train.py --env_name {env_name} --use_ai_net` "
            f"once (or `python PTAI.py --env {env_name}`) to pre-train and save it."
        )
    checkpoint = torch.load(path, map_location=device)
    ai_net = AINet(
        cfg['n_agent_types'], cfg['agent_types'], cfg['other_sub_sizes'],
        cfg['slfgbl_sub_sizes'], cfg['action_sub_sizes'], device=device
    )
    ai_net.load_state_dict(checkpoint['ai_net'])
    ai_net.eval_mode()
    sub_assignments = checkpoint['sub_assignments']
    return ai_net, sub_assignments, cfg


def load_trained_maddpg(env_name, run_id, ai_net, sub_assignments, agent_ids, obs_dims, action_dims, device):
    model_dir = os.path.join('results', 'maddpg_ptai', env_name, run_id)
    maddpg = MADDPG(
        agent_ids=agent_ids, obs_dims=obs_dims, action_dims=action_dims,
        buffer_capacity=1, device=device, use_prev_action=False, use_prev_obs=False,
        shared_actor=False, env_name=env_name, ai_net=ai_net, sub_assignments=sub_assignments,
    )
    maddpg.load(model_dir)
    return maddpg


class PredictionScorer:
    """Accumulates argmax-accuracy and MSE-vs-one-hot for AINet predictions."""

    def __init__(self, agent_ids, cfg, offsets, device):
        self.agent_ids = agent_ids
        self.cfg = cfg
        self.offsets = offsets
        self.device = device
        n = len(agent_ids)
        self.correct = np.zeros((n, n), dtype=np.int64)   # [observer][observed]
        self.total = np.zeros((n, n), dtype=np.int64)
        self.sq_err_sum = np.zeros((n, n), dtype=np.float64)

        # Output-collapse tracking: running sum / sum-of-squares / count of a_hat
        # per observer, to compute per-dimension mean & std across many different
        # (curr_obs, last_obs) inputs. A low std-to-magnitude ratio means AINet
        # outputs nearly the same vector regardless of input ("collapsed to constant").
        action_dim = sum(cfg['action_sub_sizes'][t] for t in cfg['agent_types'])
        self.ahat_sum = np.zeros((n, action_dim), dtype=np.float64)
        self.ahat_sq_sum = np.zeros((n, action_dim), dtype=np.float64)
        self.ahat_count = np.zeros(n, dtype=np.int64)

    def score_step(self, ai_net, sub_assignments, curr_obs_np, last_obs_np, true_actions):
        """true_actions: dict agent_name -> int action index taken at `last_obs` -> `curr_obs`."""
        for i, observer_id in enumerate(self.agent_ids):
            curr_obs_t = torch.tensor(curr_obs_np[observer_id], dtype=torch.float32).to(self.device)
            last_obs_t = torch.tensor(last_obs_np[observer_id], dtype=torch.float32).to(self.device)
            sub_assign = sub_assignments[i].to(self.device)
            with torch.no_grad():
                a_hat = ai_net.forward(sub_assign, i, curr_obs_t, last_obs_t).cpu().numpy()

            self.ahat_sum[i] += a_hat
            self.ahat_sq_sum[i] += a_hat ** 2
            self.ahat_count[i] += 1

            for j, observed_id in enumerate(self.agent_ids):
                if observed_id not in true_actions:
                    continue
                start, end = self.offsets[j]
                pred_slice = a_hat[start:end]
                true_idx = true_actions[observed_id]
                target = np.zeros_like(pred_slice)
                target[true_idx] = 1.0

                self.total[i, j] += 1
                if int(np.argmax(pred_slice)) == true_idx:
                    self.correct[i, j] += 1
                self.sq_err_sum[i, j] += float(np.mean((pred_slice - target) ** 2))

    def collapse_ratios(self):
        """
        Per-observer (and overall) ratio of cross-instance std to mean magnitude of
        a_hat. Computed from running sums: var = E[x^2] - E[x]^2 per output dimension,
        then ratio = mean(std over dims) / mean(|mean| over dims).

        Low ratio (<<1) => AINet outputs ~the same vector regardless of input
        (collapsed to a near-constant "average" prediction, i.e. it has not learned
        instance-specific inference). Ratio near/above 1 => output varies meaningfully
        with the input relative to its own scale.
        """
        n = len(self.agent_ids)
        ratios = np.zeros(n, dtype=np.float64)
        for i in range(n):
            cnt = max(self.ahat_count[i], 1)
            mean = self.ahat_sum[i] / cnt
            var = np.maximum(self.ahat_sq_sum[i] / cnt - mean ** 2, 0.0)
            std = np.sqrt(var)
            mean_mag = np.mean(np.abs(mean))
            ratios[i] = np.mean(std) / mean_mag if mean_mag > 1e-12 else float('nan')
        return ratios

    def report(self, label):
        n = len(self.agent_ids)
        overall_correct = self.correct.sum()
        overall_total = self.total.sum()
        overall_acc = overall_correct / max(overall_total, 1)
        overall_mse = self.sq_err_sum.sum() / max(overall_total, 1)
        ratios = self.collapse_ratios()
        overall_ratio = float(np.nanmean(ratios))
        print(f"\n--- {label}: overall argmax-accuracy = {overall_acc:.3f}   "
              f"overall MSE-vs-one-hot = {overall_mse:.4f}   "
              f"overall collapse-ratio = {overall_ratio:.3f}   (n_pairs={overall_total})")
        header = "observer / observed"
        print(f"{header:<22}" + "".join(f"{aid[:14]:>16}" for aid in self.agent_ids) + f"{'collapse-ratio':>18}")
        for i in range(n):
            row = []
            for j in range(n):
                if self.total[i, j] == 0:
                    row.append(f"{'--':>16}")
                else:
                    acc = self.correct[i, j] / self.total[i, j]
                    row.append(f"{acc:>16.3f}")
            print(f"{self.agent_ids[i][:22]:<22}" + "".join(row) + f"{ratios[i]:>18.4f}")
        return overall_acc, overall_mse, overall_ratio


def run_random_rollout(env_name, episode_length, num_episodes, ai_net, sub_assignments, cfg, offsets, device):
    env = _make_env(env_name, episode_length)
    agent_ids = env.possible_agents
    scorer = PredictionScorer(agent_ids, cfg, offsets, device)

    for _ in range(num_episodes):
        observations, _ = env.reset()
        last_observations = None
        last_actions = None
        while env.agents:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            next_observations, _, _, _, _ = env.step(actions)
            if last_observations is not None:
                scorer.score_step(ai_net, sub_assignments, observations, last_observations, last_actions)
            last_observations = {k: v.copy() for k, v in observations.items()}
            last_actions = dict(actions)
            observations = next_observations
    env.close()
    return scorer


def run_on_policy_rollout(env_name, episode_length, num_episodes, ai_net, sub_assignments, cfg, offsets,
                          maddpg, obs_dims, device):
    env = _make_env(env_name, episode_length)
    agent_ids = env.possible_agents
    scorer = PredictionScorer(agent_ids, cfg, offsets, device)

    for _ in range(num_episodes):
        observations, _ = env.reset()
        last_observations = None
        last_actions = None
        prev_observations = {agent: np.zeros(obs_dims[agent], dtype=np.float32) for agent in agent_ids}
        while env.agents:
            actions = maddpg.select_actions(observations, prev_observations, explore=False)
            next_observations, _, _, _, _ = env.step(actions)
            if last_observations is not None:
                scorer.score_step(ai_net, sub_assignments, observations, last_observations, last_actions)
            prev_observations = {agent: observations[agent].copy() for agent in agent_ids}
            last_observations = {k: v.copy() for k, v in observations.items()}
            last_actions = dict(actions)
            observations = next_observations
    env.close()
    return scorer


def main():
    parser = argparse.ArgumentParser(description="Diagnose AINet (PTAI) prediction quality under distribution shift")
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--run_id', type=str, default='1')
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--episode_length', type=int, default=25)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print(f"=== AINet diagnostic: {args.env_name} ===")
    ai_net, sub_assignments, cfg = load_frozen_ai_net(args.env_name, args.device)
    offsets = compute_action_offsets(cfg)

    # Probe env once to get agent_ids / obs_dims / action_dims for MADDPG construction
    probe_env = _make_env(args.env_name, args.episode_length)
    probe_env.reset()
    agent_ids = probe_env.possible_agents
    obs_dims = {a: probe_env.observation_space(a).shape[0] for a in agent_ids}
    action_dims = {a: probe_env.action_space(a).n for a in agent_ids}
    probe_env.close()

    # Chance-level baseline per agent type (1 / action_sub_size), for reference.
    chance = {f"type_{t}": 1.0 / size for t, size in enumerate(cfg['action_sub_sizes'])}

    print(f"Agents: {agent_ids}")
    print(f"AINet total_action_dim = {ai_net.total_action_dim}, offsets = {offsets}")
    print(f"Chance-level accuracy per agent type (1/action_dim): {chance}")

    print("\n[1/2] Random-action rollouts (matches AINet pre-training distribution)...")
    random_scorer = run_random_rollout(
        args.env_name, args.episode_length, args.num_episodes,
        ai_net, sub_assignments, cfg, offsets, args.device
    )
    random_acc, random_mse, random_ratio = random_scorer.report("RANDOM rollouts")

    print("\n[2/2] On-policy rollouts (trained MADDPG+PTAI policy, deterministic)...")
    maddpg = load_trained_maddpg(
        args.env_name, args.run_id, ai_net, sub_assignments,
        agent_ids, obs_dims, action_dims, args.device
    )
    onpolicy_scorer = run_on_policy_rollout(
        args.env_name, args.episode_length, args.num_episodes,
        ai_net, sub_assignments, cfg, offsets, maddpg, obs_dims, args.device
    )
    onpolicy_acc, onpolicy_mse, onpolicy_ratio = onpolicy_scorer.report("ON-POLICY rollouts")

    print(f"\n=== SUMMARY for {args.env_name} (chance level per type: {chance}) ===")
    print(f"  random   : accuracy={random_acc:.3f}  mse={random_mse:.4f}  collapse_ratio={random_ratio:.4f}")
    print(f"  on_policy: accuracy={onpolicy_acc:.3f}  mse={onpolicy_mse:.4f}  collapse_ratio={onpolicy_ratio:.4f}")
    print(f"  accuracy drop (random -> on_policy): {random_acc - onpolicy_acc:+.3f}")
    print("  (collapse_ratio << 1 means AINet's output barely varies with input -- "
          "it has collapsed to a near-constant prediction rather than learning instance-specific inference)")


if __name__ == '__main__':
    main()

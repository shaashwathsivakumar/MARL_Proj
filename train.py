"""
Training script for MADDPG on PettingZoo MPE environments.
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from pettingzoo.mpe.simple import simple as simple_v3
from pettingzoo.mpe.simple_adversary import simple_adversary as simple_adversary_v3
from pettingzoo.mpe.simple_crypto import simple_crypto as simple_crypto_v3
from pettingzoo.mpe.simple_push import simple_push as simple_push_v3
from pettingzoo.mpe.simple_reference import simple_reference as simple_reference_v3
from pettingzoo.mpe.simple_speaker_listener import simple_speaker_listener as simple_speaker_listener_v4
from pettingzoo.mpe.simple_spread import simple_spread as simple_spread_v3
from pettingzoo.mpe.simple_tag import simple_tag as simple_tag_v3
from pettingzoo.mpe.simple_world_comm import simple_world_comm as simple_world_comm_v3

from maddpg import MADDPG
from metrics import create_metrics_tracker, get_all_metric_names


# Agent team definitions for computing "agent score" (non-adversary team reward)
# Used for comparing algorithms as in the MADDPG paper
AGENT_TEAMS = {
    # Cooperative environments: all agents are on the same team
    'simple_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_reference_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_speaker_listener_v4': lambda agents: agents,  # speaker + listener
    'simple_spread_v3': lambda agents: [a for a in agents if a.startswith('agent')],

    # Competitive environments: agents vs adversaries
    'simple_adversary_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_crypto_v3': lambda agents: [a for a in agents if a.startswith('alice') or a.startswith('bob')],
    'simple_push_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_tag_v3': lambda agents: [a for a in agents if a.startswith('agent')],  # prey
    'simple_world_comm_v3': lambda agents: [a for a in agents if a.startswith('agent')],
}


def get_agent_team(env_name, agent_ids):
    """Get list of agents in the 'agent' team (non-adversaries) for score computation."""
    if env_name in AGENT_TEAMS:
        return AGENT_TEAMS[env_name](agent_ids)
    # Default: all agents starting with 'agent'
    return [a for a in agent_ids if a.startswith('agent')]


def make_env(env_name, max_cycles=25):
    """Create environment and extract dimension info."""
    # Map environment names to PettingZoo modules
    # MADDPG paper environments:
    #   - simple_v3: Basic single-agent environment
    #   - simple_adversary_v3: Physical deception
    #   - simple_crypto_v3: Covert communication
    #   - simple_push_v3: Competitive push
    #   - simple_reference_v3: Reference game
    #   - simple_speaker_listener_v4: Cooperative communication
    #   - simple_spread_v3: Cooperative navigation
    #   - simple_tag_v3: Predator-prey
    #   - simple_world_comm_v3: Keep-away with communication
    env_map = {
        'simple_v3': simple_v3,
        'simple_adversary_v3': simple_adversary_v3,
        'simple_crypto_v3': simple_crypto_v3,
        'simple_push_v3': simple_push_v3,
        'simple_reference_v3': simple_reference_v3,
        'simple_speaker_listener_v4': simple_speaker_listener_v4,
        'simple_spread_v3': simple_spread_v3,
        'simple_tag_v3': simple_tag_v3,
        'simple_world_comm_v3': simple_world_comm_v3,
    }

    if env_name not in env_map:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(env_map.keys())}")

    env = env_map[env_name].parallel_env(max_cycles=max_cycles)
    env.reset()

    obs_dims = {agent: env.observation_space(agent).shape[0] for agent in env.agents}
    action_dims = {agent: env.action_space(agent).n for agent in env.agents}

    return env, obs_dims, action_dims


def compute_running_reward(rewards, window=100):
    """Compute running average of rewards."""
    running = np.zeros_like(rewards)
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        running[i] = np.mean(rewards[start:i + 1])
    return running


def get_algorithm_name(args):
    """Determine algorithm name based on flags or explicit argument."""
    if args.algorithm:
        return args.algorithm

    # Auto-detect based on flags
    parts = ['maddpg']

    if args.shared_actor:
        parts.append('shared_actor')
    if args.use_geometric_sampling:
        parts.append('geometric')
    if args.use_prev_action:
        parts.append('prev_action')
    if args.use_prev_observation:
        parts.append('prev_observation')
    if args.use_ai_net:
        if getattr(args, 'online_ai_net', False):
            parts.append('ptai_online')
        else:
            parts.append('ptai')
        fellow_mode = getattr(args, 'fellow_mode', 'action')
        if fellow_mode == 'velocity_analytical':
            parts.append('velA')
        elif fellow_mode == 'velocity_learned':
            parts.append('velL')
    if getattr(args, 'use_adversary_gating', False):
        parts.append('adv_gating')
    if getattr(args, 'use_twin_critic', False):
        parts.append('twin_critic')

    return '_'.join(parts)


def train(args):
    """Main training loop."""
    # Validate --fellow_mode
    fellow_mode = getattr(args, 'fellow_mode', 'action')
    if fellow_mode != 'action':
        if not args.use_ai_net:
            raise ValueError("--fellow_mode requires --use_ai_net")
        if args.env_name != 'simple_tag_v3':
            raise ValueError("--fellow_mode != 'action' is only supported for simple_tag_v3")
        if getattr(args, 'online_ai_net', False):
            raise ValueError("--online_ai_net is not supported with --fellow_mode != 'action'")

    # Determine algorithm name
    algorithm = get_algorithm_name(args)

    # Setup results directory: results/<algorithm>/<env_name>/<run_num>
    algo_dir = os.path.join('./results', algorithm)
    env_dir = os.path.join(algo_dir, args.env_name)
    os.makedirs(env_dir, exist_ok=True)
    run_num = len([d for d in os.listdir(env_dir) if os.path.isdir(os.path.join(env_dir, d))]) + 1
    result_dir = os.path.join(env_dir, str(run_num))
    os.makedirs(result_dir)

    # Save training arguments for reproducibility
    args_dict = vars(args).copy()
    args_dict['algorithm'] = algorithm  # Store the resolved algorithm name
    with open(os.path.join(result_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=2)

    print(f"Algorithm: {algorithm}")
    print(f"Results will be saved to: {result_dir}")

    # Create environment
    env, obs_dims, action_dims = make_env(args.env_name, args.episode_length)
    agent_ids = list(obs_dims.keys())

    # Identify agent team for score computation
    agent_team = get_agent_team(args.env_name, agent_ids)
    adversary_team = [a for a in agent_ids if a not in agent_team]

    print(f"Environment: {args.env_name}")
    print(f"Agents: {agent_ids}")
    print(f"Agent team (for scoring): {agent_team}")
    print(f"Adversary team: {adversary_team}")
    if getattr(args, 'use_adversary_gating', False):
        print(f"Adversary gating: enabled (temperature={args.gating_temperature}, "
              f"ema_decay={args.gating_ema_decay})")
    if getattr(args, 'use_twin_critic', False):
        print(f"Twin critic (TD3-style): enabled (policy_delay={args.policy_delay})")
    print(f"Observation dims: {obs_dims}")
    print(f"Action dims: {action_dims}")

    # Pre-train or load AI_Net if requested (Algorithm 2)
    ai_net        = None
    sub_assignments = None
    if args.use_ai_net:
        from PTAI import AINet, get_env_ptai_config, pre_train_ai_net
        _, cfg = get_env_ptai_config(args.env_name)
        self_pos_offsets = dict(enumerate(cfg.get('self_pos_offsets', [0] * cfg['n_agent_types'])))
        if fellow_mode == 'velocity_analytical':
            ai_net_path = f'AI_Net_{args.env_name}_velA.pt'
        elif fellow_mode == 'velocity_learned':
            ai_net_path = f'AI_Net_{args.env_name}_velL.pt'
        else:
            ai_net_path = f'AI_Net_{args.env_name}.pt'
        if os.path.exists(ai_net_path):
            print(f"Loading pre-trained AI_Net from {ai_net_path} ...")
            checkpoint = torch.load(ai_net_path, map_location=args.device)
            ai_net = AINet(
                cfg['n_agent_types'], cfg['agent_types'],
                cfg['other_sub_sizes'], cfg['slfgbl_sub_sizes'],
                cfg['action_sub_sizes'], device=args.device,
                fellow_mode=fellow_mode, self_pos_offsets=self_pos_offsets,
            )
            ai_net.load_state_dict(checkpoint['ai_net'])
            sub_assignments = checkpoint['sub_assignments']
            if not getattr(args, 'online_ai_net', False):
                ai_net.eval_mode()
        else:
            print(f"Pre-training AI_Net for {args.env_name} (this may take a few minutes) ...")
            ai_net, sub_assignments = pre_train_ai_net(
                env_name=args.env_name,
                episode_count=200,
                max_episode_length=20,
                percent_to_train_on=0.8,
                lr=0.001,
                device=args.device,
                fellow_mode=fellow_mode,
            )
            if getattr(args, 'online_ai_net', False):
                # pre_train_ai_net freezes the net; undo that for online training
                for mod in ai_net.self_modules:
                    mod.train()
                for row in ai_net.social_modules:
                    for mod in row:
                        mod.train()
            torch.save(
                {'ai_net': ai_net.state_dict(), 'sub_assignments': sub_assignments},
                ai_net_path
            )
            print(f"AI_Net saved to {ai_net_path}")

    # Initialize MADDPG
    maddpg = MADDPG(
        agent_ids=agent_ids,
        obs_dims=obs_dims,
        action_dims=action_dims,
        buffer_capacity=args.buffer_capacity,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        device=args.device,
        geometric_sampling=args.use_geometric_sampling,
        geo_alpha=args.geo_alpha,
        use_prev_action=args.use_prev_action,
        use_prev_obs=args.use_prev_observation,
        shared_actor=args.shared_actor,
        env_name=args.env_name,
        ai_net=ai_net,
        sub_assignments=sub_assignments,
        online_ai_net=getattr(args, 'online_ai_net', False),
        use_adversary_gating=getattr(args, 'use_adversary_gating', False),
        gating_temperature=getattr(args, 'gating_temperature', 0.5),
        gating_ema_decay=getattr(args, 'gating_ema_decay', 0.05),
        use_twin_critic=getattr(args, 'use_twin_critic', False),
        policy_delay=getattr(args, 'policy_delay', 2),
    )

    # Compute total action dim for prev_joint_action tracking
    total_action_dim = sum(action_dims.values())

    # Training metrics
    episode_rewards = {agent: np.zeros(args.num_episodes) for agent in agent_ids}
    agent_scores = np.zeros(args.num_episodes)  # Agent team score for algorithm comparison
    adversary_scores = np.zeros(args.num_episodes)  # Adversary team score
    global_step = 0

    # Environment-specific metrics tracker
    metrics_tracker = create_metrics_tracker(args.env_name, agent_ids)
    metric_names = get_all_metric_names(args.env_name)
    episode_metrics = {name: np.zeros(args.num_episodes) for name in metric_names}

    # Training loop
    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        episode_reward = {agent: 0.0 for agent in agent_ids}
        metrics_tracker.reset_episode()

        # Initialize previous joint action (zeros at episode start)
        prev_joint_action = np.zeros(total_action_dim, dtype=np.float32) if args.use_prev_action else None

        # Initialize previous observations (zeros at episode start)
        prev_observations = {agent: np.zeros(obs_dims[agent], dtype=np.float32) for agent in agent_ids} if args.use_prev_observation else None

        while env.agents:
            global_step += 1

            # Action selection: random exploration vs policy
            if global_step < args.warmup_steps:
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            else:
                actions = maddpg.select_actions(obs, prev_observations, explore=True)

            # Environment step
            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            # Compute done flags (use terminations keys since env.agents may have changed)
            dones = {agent: terminations[agent] or truncations[agent] for agent in terminations.keys()}

            # Update environment-specific metrics
            metrics_tracker.update(obs, actions, rewards, next_obs, dones, env)

            # Store transition (with previous joint action and previous observations if enabled)
            maddpg.store_transition(obs, actions, rewards, next_obs, dones, prev_joint_action, prev_observations)

            # Update prev_joint_action for next step (concatenate current actions as one-hot)
            if args.use_prev_action:
                prev_joint_action = []
                for aid in agent_ids:
                    action = actions[aid]
                    if isinstance(action, (int, np.integer)):
                        action_onehot = np.zeros(action_dims[aid], dtype=np.float32)
                        action_onehot[action] = 1.0
                        prev_joint_action.append(action_onehot)
                    else:
                        prev_joint_action.append(action)
                prev_joint_action = np.concatenate(prev_joint_action)

            # Update prev_observations for next step
            if args.use_prev_observation:
                prev_observations = {agent: obs[agent].copy() for agent in agent_ids}

            # Accumulate rewards
            for agent in rewards.keys():
                episode_reward[agent] += rewards[agent]

            # Learning step
            if global_step >= args.warmup_steps and global_step % args.learn_interval == 0:
                if len(maddpg.buffer) >= args.batch_size:
                    maddpg.update(args.batch_size, args.gamma)
                    maddpg.soft_update_targets(args.tau)

            obs = next_obs

        # Record episode rewards
        for agent in agent_ids:
            episode_rewards[agent][episode] = episode_reward[agent]

        # Compute agent team score (sum of agent team rewards)
        agent_scores[episode] = sum(episode_reward[a] for a in agent_team)
        adversary_scores[episode] = sum(episode_reward[a] for a in adversary_team)

        # Record environment-specific metrics
        ep_metrics = metrics_tracker.get_episode_metrics()
        for name, value in ep_metrics.items():
            if name in episode_metrics:
                episode_metrics[name][episode] = value

        # Logging
        if (episode + 1) % args.log_interval == 0:
            msg = f"Episode {episode + 1}/{args.num_episodes} | "
            for agent in agent_ids:
                r = episode_reward[agent]
                msg += f"{agent}: {r:.1f} | "
            msg += f"AgentScore: {agent_scores[episode]:.1f}"
            if adversary_team:
                msg += f" | AdvScore: {adversary_scores[episode]:.1f}"
            print(msg)

        # Save checkpoint
        if (episode + 1) % args.checkpoint_interval == 0:
            checkpoint_dir = os.path.join(result_dir, f'checkpoint_{episode + 1}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            maddpg.save(checkpoint_dir)
            checkpoint_data = {
                'per_agent': {k: v[:episode + 1].copy() for k, v in episode_rewards.items()},
                'agent_score': agent_scores[:episode + 1].copy(),
                'adversary_score': adversary_scores[:episode + 1].copy(),
                'agent_team': agent_team,
                'adversary_team': adversary_team,
                'metrics': {k: v[:episode + 1].copy() for k, v in episode_metrics.items()},
            }
            with open(os.path.join(checkpoint_dir, 'rewards.pkl'), 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"  -> Checkpoint saved: {checkpoint_dir}")

    env.close()

    # Save results
    maddpg.save(result_dir)

    # Save evolved AINet when online training was used
    if getattr(args, 'online_ai_net', False) and ai_net is not None:
        torch.save(
            {'ai_net': ai_net.state_dict(), 'sub_assignments': sub_assignments},
            os.path.join(result_dir, 'ai_net.pt')
        )

    # Save rewards with agent_score and metrics for algorithm comparison
    rewards_data = {
        'per_agent': episode_rewards,
        'agent_score': agent_scores,
        'adversary_score': adversary_scores,
        'agent_team': agent_team,
        'adversary_team': adversary_team,
        'metrics': episode_metrics,
    }
    with open(os.path.join(result_dir, 'rewards.pkl'), 'wb') as f:
        pickle.dump(rewards_data, f)

    # Plot training curves
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    x = range(1, args.num_episodes + 1)

    # Top plot: Agent Score (main metric for algorithm comparison)
    ax1 = axes[0]
    ax1.plot(x, agent_scores, alpha=0.3, color='blue', label='Agent Score (raw)')
    ax1.plot(x, compute_running_reward(agent_scores), color='blue', linewidth=2, label='Agent Score (smoothed)')
    if adversary_team:
        ax1.plot(x, adversary_scores, alpha=0.3, color='red', label='Adversary Score (raw)')
        ax1.plot(x, compute_running_reward(adversary_scores), color='red', linewidth=2, label='Adversary Score (smoothed)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title(f'MADDPG Training: {args.env_name} - Agent Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Per-agent rewards
    ax2 = axes[1]
    for agent in agent_ids:
        rewards = episode_rewards[agent]
        ax2.plot(x, compute_running_reward(rewards), label=f'{agent}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.set_title('Per-Agent Rewards (smoothed)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'training_curves.png'), dpi=150)
    plt.close()

    print(f"\nTraining complete! Results saved to: {result_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train MADDPG on MPE environments')

    # Environment
    parser.add_argument('--env_name', type=str, default='simple_tag_v3',
                        choices=[
                            'simple_v3',
                            'simple_adversary_v3',
                            'simple_crypto_v3',
                            'simple_push_v3',
                            'simple_reference_v3',
                            'simple_speaker_listener_v4',
                            'simple_spread_v3',
                            'simple_tag_v3',
                            'simple_world_comm_v3',
                        ],
                        help='Environment name')
    parser.add_argument('--episode_length', type=int, default=25,
                        help='Maximum steps per episode')

    # Training
    parser.add_argument('--num_episodes', type=int, default=30000,
                        help='Total number of training episodes')
    parser.add_argument('--warmup_steps', type=int, default=50000,
                        help='Random exploration steps before learning')
    parser.add_argument('--learn_interval', type=int, default=100,
                        help='Steps between learning updates')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for learning')
    parser.add_argument('--buffer_capacity', type=int, default=1000000,
                        help='Replay buffer capacity')

    # Algorithm
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.02,
                        help='Soft update coefficient')
    parser.add_argument('--actor_lr', type=float, default=0.01,
                        help='Actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=0.01,
                        help='Critic learning rate')

    # Enhancements
    parser.add_argument('--shared_actor', action='store_true',
                        help='Share actor network within agent/adversary teams')
    parser.add_argument('--use_geometric_sampling', action='store_true',
                        help='Use geometric distribution for replay sampling')
    parser.add_argument('--geo_alpha', type=float, default=1e-5,
                        help='Decay rate for geometric sampling (higher = more bias to recent)')
    parser.add_argument('--use_prev_action', action='store_true',
                        help='Condition critic on previous joint action')
    parser.add_argument('--use_prev_observation', action='store_true',
                        help='Condition actor on previous observation')
    parser.add_argument('--online_ai_net', action='store_true',
                        help='Keep AINet unfrozen and update it simultaneously with MADDPG '
                             'training on each replay buffer batch. Requires --use_ai_net. '
                             'Results saved under maddpg_ptai_online algorithm directory.')
    parser.add_argument('--use_ai_net', action='store_true',
                        help='Use pre-trained AI_Net for action inference (PTAI). '
                             'Supported for all 9 MPE environments. '
                             'Pre-trains and saves AI_Net_<env>.pt if not already present.')
    parser.add_argument('--fellow_mode', type=str, default='action',
                        choices=['action', 'velocity_analytical', 'velocity_learned'],
                        help='Fellow-same-type-agent AINet prediction target: "action" '
                             '(default, unchanged ~0.32-acc action prediction), '
                             '"velocity_analytical" (parameter-free 2D displacement), '
                             '"velocity_learned" (2D displacement via learned regression). '
                             'Only supported for --use_ai_net --env_name simple_tag_v3 '
                             'without --online_ai_net.')
    parser.add_argument('--use_adversary_gating', action='store_true',
                        help='Experimental: scale each opponent agent\'s obs/action '
                             'features in the centralized critic input by a "trust" gate '
                             'derived from how much that opponent\'s action distribution '
                             'has drifted from its own slow-moving EMA (i.e. distrust the '
                             'opponent model most right when its policy is changing '
                             'fastest). No effect on agents with no opposing team '
                             '(e.g. cooperative envs). Composes with any other flag.')
    parser.add_argument('--gating_temperature', type=float, default=0.5,
                        help='Sensitivity of the adversary-gating trust gate to drift '
                             '(smaller = more aggressive attenuation). Only used with '
                             '--use_adversary_gating.')
    parser.add_argument('--gating_ema_decay', type=float, default=0.05,
                        help='EMA update rate for the adversary-gating drift reference. '
                             'Only used with --use_adversary_gating.')
    parser.add_argument('--use_twin_critic', action='store_true',
                        help='TD3-style (Fujimoto et al. 2018) clipped double Q-learning: '
                             'each agent gets two critics, the TD target uses the min of '
                             'both targets, and actor/target-network updates are delayed '
                             'to every --policy_delay calls. Target policy smoothing (the '
                             'third TD3 component) is omitted -- no clean analog for '
                             'discrete Gumbel-Softmax actions. Composes with any other flag.')
    parser.add_argument('--policy_delay', type=int, default=2,
                        help='Number of update() calls between actor/target-network '
                             'updates. Only used with --use_twin_critic.')

    # Algorithm naming (auto-detected from flags if not specified)
    parser.add_argument('--algorithm', type=str, default=None,
                        choices=['maddpg', 'maddpg_geometric', 'maddpg_prev_action',
                                 'maddpg_prev_observation', 'maddpg_geometric_prev_action',
                                 'maddpg_geometric_prev_observation',
                                 'maddpg_prev_action_prev_observation',
                                 'maddpg_geometric_prev_action_prev_observation',
                                 'maddpg_shared_actor', 'maddpg_shared_actor_geometric',
                                 'maddpg_shared_actor_prev_action', 'maddpg_shared_actor_prev_observation',
                                 'maddpg_shared_actor_geometric_prev_action',
                                 'maddpg_shared_actor_geometric_prev_observation',
                                 'maddpg_shared_actor_prev_action_prev_observation',
                                 'maddpg_shared_actor_geometric_prev_action_prev_observation',
                                 'maddpg_ptai_online', 'maddpg_ptai_velA', 'maddpg_ptai_velL',
                                 'maddpg_adv_gating', 'maddpg_twin_critic'],
                        help='Algorithm name for results organization (auto-detected if not specified)')

    # Misc
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Episodes between log messages')
    parser.add_argument('--checkpoint_interval', type=int, default=10000,
                        help='Episodes between checkpoint saves')
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Number of times to repeat this training run '
                             '(each saved to its own auto-numbered run directory)')

    args = parser.parse_args()

    # Auto-detect CUDA
    if args.device == 'cuda' and not __import__('torch').cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    for run in range(args.n_runs):
        if args.n_runs > 1:
            print(f"\n===== Run {run + 1}/{args.n_runs} =====")
        train(args)


if __name__ == '__main__':
    main()

"""
Training script for MADDPG on PettingZoo MPE environments.
"""
import argparse
import json
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.mpe import (
    simple_v3,
    simple_adversary_v3,
    simple_crypto_v3,
    simple_push_v3,
    simple_reference_v3,
    simple_speaker_listener_v4,
    simple_spread_v3,
    simple_tag_v3,
    simple_world_comm_v3,
)

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
    if args.use_geometric_sampling and args.use_prev_action:
        return 'maddpg_geometric_prev_action'
    elif args.use_geometric_sampling:
        return 'maddpg_geometric'
    elif args.use_prev_action:
        return 'maddpg_prev_action'
    else:
        return 'maddpg'


def train(args):
    """Main training loop."""
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
    print(f"Observation dims: {obs_dims}")
    print(f"Action dims: {action_dims}")

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
        use_prev_action=args.use_prev_action
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

        while env.agents:
            global_step += 1

            # Action selection: random exploration vs policy
            if global_step < args.warmup_steps:
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            else:
                actions = maddpg.select_actions(obs, explore=True)

            # Environment step
            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            # Compute done flags (use terminations keys since env.agents may have changed)
            dones = {agent: terminations[agent] or truncations[agent] for agent in terminations.keys()}

            # Update environment-specific metrics
            metrics_tracker.update(obs, actions, rewards, next_obs, dones, env)

            # Store transition (with previous joint action if enabled)
            maddpg.store_transition(obs, actions, rewards, next_obs, dones, prev_joint_action)

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
    parser.add_argument('--use_geometric_sampling', action='store_true',
                        help='Use geometric distribution for replay sampling')
    parser.add_argument('--geo_alpha', type=float, default=1e-5,
                        help='Decay rate for geometric sampling (higher = more bias to recent)')
    parser.add_argument('--use_prev_action', action='store_true',
                        help='Condition critic on previous joint action')

    # Algorithm naming (auto-detected from flags if not specified)
    parser.add_argument('--algorithm', type=str, default=None,
                        choices=['maddpg', 'maddpg_geometric', 'maddpg_prev_action',
                                 'maddpg_geometric_prev_action'],
                        help='Algorithm name for results organization (auto-detected if not specified)')

    # Misc
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Episodes between log messages')
    parser.add_argument('--checkpoint_interval', type=int, default=10000,
                        help='Episodes between checkpoint saves')

    args = parser.parse_args()

    # Auto-detect CUDA
    if args.device == 'cuda' and not __import__('torch').cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    train(args)


if __name__ == '__main__':
    main()

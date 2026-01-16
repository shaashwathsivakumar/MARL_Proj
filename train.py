"""
Training script for MADDPG on PettingZoo MPE environments.
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_tag_v3, simple_adversary_v3, simple_spread_v3

from maddpg import MADDPG


def make_env(env_name, max_cycles=25):
    """Create environment and extract dimension info."""
    env_map = {
        'simple_tag_v3': simple_tag_v3,
        'simple_adversary_v3': simple_adversary_v3,
        'simple_spread_v3': simple_spread_v3,
    }

    if env_name not in env_map:
        raise ValueError(f"Unknown environment: {env_name}")

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


def train(args):
    """Main training loop."""
    import json

    # Setup results directory
    env_dir = os.path.join('./results', args.env_name)
    os.makedirs(env_dir, exist_ok=True)
    run_num = len(os.listdir(env_dir)) + 1
    result_dir = os.path.join(env_dir, str(run_num))
    os.makedirs(result_dir)

    # Save training arguments for reproducibility
    with open(os.path.join(result_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"Results will be saved to: {result_dir}")

    # Create environment
    env, obs_dims, action_dims = make_env(args.env_name, args.episode_length)
    agent_ids = list(obs_dims.keys())

    print(f"Environment: {args.env_name}")
    print(f"Agents: {agent_ids}")
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
        device=args.device
    )

    # Training metrics
    episode_rewards = {agent: np.zeros(args.num_episodes) for agent in agent_ids}
    global_step = 0

    # Training loop
    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        episode_reward = {agent: 0.0 for agent in agent_ids}

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

            # Store transition
            maddpg.store_transition(obs, actions, rewards, next_obs, dones)

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

        # Logging
        if (episode + 1) % args.log_interval == 0:
            msg = f"Episode {episode + 1}/{args.num_episodes} | "
            total = 0
            for agent in agent_ids:
                r = episode_reward[agent]
                msg += f"{agent}: {r:.1f} | "
                total += r
            msg += f"Total: {total:.1f}"
            print(msg)

        # Save checkpoint
        if (episode + 1) % args.checkpoint_interval == 0:
            checkpoint_dir = os.path.join(result_dir, f'checkpoint_{episode + 1}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            maddpg.save(checkpoint_dir)
            with open(os.path.join(checkpoint_dir, 'rewards.pkl'), 'wb') as f:
                pickle.dump({k: v[:episode + 1].copy() for k, v in episode_rewards.items()}, f)
            print(f"  -> Checkpoint saved: {checkpoint_dir}")

    env.close()

    # Save results
    maddpg.save(result_dir)

    with open(os.path.join(result_dir, 'rewards.pkl'), 'wb') as f:
        pickle.dump(episode_rewards, f)

    # Plot training curves
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(1, args.num_episodes + 1)

    for agent in agent_ids:
        rewards = episode_rewards[agent]
        ax.plot(x, rewards, alpha=0.3, label=f'{agent} (raw)')
        ax.plot(x, compute_running_reward(rewards), label=f'{agent} (smoothed)')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(f'MADDPG Training: {args.env_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'training_curves.png'), dpi=150)
    print(f"\nTraining complete! Results saved to: {result_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train MADDPG on MPE environments')

    # Environment
    parser.add_argument('--env_name', type=str, default='simple_tag_v3',
                        choices=['simple_tag_v3', 'simple_adversary_v3', 'simple_spread_v3'],
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

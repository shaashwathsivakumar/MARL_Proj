"""
Evaluation script for trained MADDPG and mixed-algorithm models.
Loads saved models and generates GIFs of agent behavior.

Supports both single-algorithm models (from train.py) and
mixed-algorithm models (from train_mixed.py).
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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
from algorithms import create_team_algorithm, get_available_algorithms


# Agent team definitions (same as train.py)
AGENT_TEAMS = {
    'simple_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_reference_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_speaker_listener_v4': lambda agents: agents,
    'simple_spread_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_adversary_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_crypto_v3': lambda agents: [a for a in agents if a.startswith('alice') or a.startswith('bob')],
    'simple_push_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_tag_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_world_comm_v3': lambda agents: [a for a in agents if a.startswith('agent')],
}


def get_agent_team(env_name, agent_ids):
    """Get list of agents in the 'agent' team for score computation."""
    if env_name in AGENT_TEAMS:
        return AGENT_TEAMS[env_name](agent_ids)
    return [a for a in agent_ids if a.startswith('agent')]


def make_env(env_name, max_cycles=25, render_mode='rgb_array'):
    """Create environment for evaluation."""
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

    env = env_map[env_name].parallel_env(max_cycles=max_cycles, render_mode=render_mode)
    env.reset()

    obs_dims = {agent: env.observation_space(agent).shape[0] for agent in env.agents}
    action_dims = {agent: env.action_space(agent).n for agent in env.agents}

    return env, obs_dims, action_dims


def is_mixed_algorithm(algorithm_name):
    """Check if algorithm name is a mixed-algorithm combination (e.g., 'maddpg_vs_ddpg')."""
    return '_vs_' in algorithm_name


def parse_mixed_algorithms(algorithm_name):
    """Parse a mixed-algorithm name into agent and adversary algorithms."""
    parts = algorithm_name.split('_vs_')
    if len(parts) != 2:
        raise ValueError(f"Invalid mixed algorithm name: {algorithm_name}")
    return parts[0], parts[1]


def evaluate(args):
    """Run evaluation episodes and generate GIFs."""
    # Verify model directory exists: results/<algorithm>/<env_name>/<run_id>
    model_dir = os.path.join('./results', args.algorithm, args.env_name, args.run_id)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Algorithm: {args.algorithm}")

    # Setup GIF directory
    gif_dir = os.path.join(model_dir, 'gif')
    os.makedirs(gif_dir, exist_ok=True)
    existing_gifs = len([f for f in os.listdir(gif_dir) if f.endswith('.gif')])

    # Create environment
    env, obs_dims, action_dims = make_env(
        args.env_name, args.episode_length, render_mode='rgb_array'
    )
    agent_ids = list(obs_dims.keys())

    # Load training args to get model configuration
    args_path = os.path.join(model_dir, 'args.json')
    use_prev_action = False
    use_prev_obs = False
    shared_actor = False
    env_name = args.env_name
    train_args = {}
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            train_args = json.load(f)
            use_prev_action = train_args.get('use_prev_action', False)
            use_prev_obs = train_args.get('use_prev_observation', False)
            shared_actor = train_args.get('shared_actor', False)
            env_name = train_args.get('env_name', args.env_name)

    # Determine if this is a mixed-algorithm model
    is_mixed = is_mixed_algorithm(args.algorithm)

    if is_mixed:
        # Load mixed-algorithm models
        agent_algo_name, adversary_algo_name = parse_mixed_algorithms(args.algorithm)
        print(f"  Agent algorithm: {agent_algo_name}")
        print(f"  Adversary algorithm: {adversary_algo_name}")

        # Get team assignments
        agent_team = get_agent_team(args.env_name, agent_ids)
        adversary_team = [a for a in agent_ids if a not in agent_team]

        # Create and load agent team algorithm
        agent_algo = create_team_algorithm(
            algorithm=agent_algo_name,
            team_agent_ids=agent_team,
            all_agent_ids=agent_ids,
            obs_dims=obs_dims,
            action_dims=action_dims,
            buffer_capacity=1,
            device=args.device
        )
        agent_model_dir = os.path.join(model_dir, 'agent_team')
        if os.path.exists(agent_model_dir):
            agent_algo.load(agent_model_dir)
            print(f"Loaded agent team model from: {agent_model_dir}")

        # Create and load adversary team algorithm (if there are adversaries)
        adversary_algo = None
        if adversary_team:
            adversary_algo = create_team_algorithm(
                algorithm=adversary_algo_name,
                team_agent_ids=adversary_team,
                all_agent_ids=agent_ids,
                obs_dims=obs_dims,
                action_dims=action_dims,
                buffer_capacity=1,
                device=args.device
            )
            adversary_model_dir = os.path.join(model_dir, 'adversary_team')
            if os.path.exists(adversary_model_dir):
                adversary_algo.load(adversary_model_dir)
                print(f"Loaded adversary team model from: {adversary_model_dir}")

        # Create action selection function for mixed algorithm
        # Note: prev_obs is not used for mixed algorithms currently
        def select_actions(obs, prev_obs=None, explore=False):
            actions = agent_algo.select_actions(obs, explore=explore)
            if adversary_algo is not None:
                adv_actions = adversary_algo.select_actions(obs, explore=explore)
                actions.update(adv_actions)
            return actions

        maddpg = None  # Not used for mixed algorithms
    else:
        # Load single-algorithm MADDPG model
        maddpg = MADDPG(
            agent_ids=agent_ids,
            obs_dims=obs_dims,
            action_dims=action_dims,
            buffer_capacity=1,  # Not used for evaluation
            device=args.device,
            use_prev_action=use_prev_action,
            use_prev_obs=use_prev_obs,
            shared_actor=shared_actor,
            env_name=env_name
        )
        maddpg.load(model_dir)
        print(f"Loaded model from: {model_dir}")
        if shared_actor:
            print("  Using shared actor networks within teams")
        if use_prev_obs:
            print("  Using previous observation conditioning for actor")

        # Create action selection function for single algorithm
        # Note: prev_observations will be passed from the evaluation loop if use_prev_obs
        def select_actions(obs, prev_obs=None, explore=False):
            return maddpg.select_actions(obs, prev_obs, explore=explore)

    # Identify agent team for scoring
    agent_team = get_agent_team(args.env_name, agent_ids)
    adversary_team = [a for a in agent_ids if a not in agent_team]
    print(f"Agent team: {agent_team}")

    # Track rewards
    episode_rewards = {agent: np.zeros(args.num_episodes) for agent in agent_ids}
    agent_scores = np.zeros(args.num_episodes)

    # Environment-specific metrics
    metrics_tracker = create_metrics_tracker(args.env_name, agent_ids)
    metric_names = get_all_metric_names(args.env_name)
    episode_metrics = {name: [] for name in metric_names}

    # Run evaluation episodes
    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        episode_reward = {agent: 0.0 for agent in agent_ids}
        metrics_tracker.reset_episode()
        frames = []

        # Initialize previous observations (zeros at episode start)
        prev_observations = {agent: np.zeros(obs_dims[agent], dtype=np.float32) for agent in agent_ids} if use_prev_obs else None

        while env.agents:
            # Select actions (deterministic for evaluation)
            actions = select_actions(obs, prev_observations, explore=False)

            # Step environment
            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            dones = {agent: terminations[agent] or truncations[agent] for agent in terminations.keys()}

            # Update metrics
            metrics_tracker.update(obs, actions, rewards, next_obs, dones, env)

            # Capture frame
            frames.append(Image.fromarray(env.render()))

            # Accumulate rewards
            for agent in env.agents:
                episode_reward[agent] += rewards[agent]

            # Update prev_observations for next step
            if use_prev_obs:
                prev_observations = {agent: obs[agent].copy() for agent in agent_ids}

            obs = next_obs

        # Record episode rewards
        for agent in agent_ids:
            episode_rewards[agent][episode] = episode_reward[agent]
        agent_scores[episode] = sum(episode_reward[a] for a in agent_team)

        # Record metrics
        ep_metrics = metrics_tracker.get_episode_metrics()
        for name, value in ep_metrics.items():
            if name in episode_metrics:
                episode_metrics[name].append(value)

        # Save GIF
        gif_path = os.path.join(gif_dir, f'episode_{existing_gifs + episode + 1}.gif')
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # ms per frame
            loop=0
        )

        # Print episode summary
        print(f"Episode {episode + 1}: Agent Score = {agent_scores[episode]:.1f} | Saved: {gif_path}")

    env.close()

    # Print evaluation summary
    mean_agent_score = np.mean(agent_scores)
    std_agent_score = np.std(agent_scores)
    print(f"\n{'='*60}")
    print(f"Evaluation Summary: {args.env_name}")
    print(f"{'='*60}")
    print(f"Agent Score: {mean_agent_score:.2f} +/- {std_agent_score:.2f}")
    print(f"Min: {np.min(agent_scores):.2f}, Max: {np.max(agent_scores):.2f}")

    # Print task-specific metrics
    if episode_metrics:
        print(f"\nTask-Specific Metrics:")
        print(f"-" * 40)
        for name, values in episode_metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {name}: {mean_val:.2f} +/- {std_val:.2f}")
    print(f"{'='*60}")

    # Plot evaluation results
    if args.num_episodes > 1:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        x = range(1, args.num_episodes + 1)

        # Top: Agent Score
        ax1 = axes[0]
        ax1.bar(x, agent_scores, color='blue', alpha=0.7)
        ax1.axhline(y=mean_agent_score, color='red', linestyle='--', label=f'Mean: {mean_agent_score:.2f}')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Agent Score')
        ax1.set_title(f'MADDPG Evaluation: {args.env_name} - Agent Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom: Per-agent rewards
        ax2 = axes[1]
        for agent in agent_ids:
            ax2.plot(x, episode_rewards[agent], marker='o', label=agent)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.set_title('Per-Agent Rewards')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(model_dir, 'evaluation_results.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\nEvaluation plot saved to: {plot_path}")

    print(f"GIFs saved to: {gif_dir}")


def get_valid_algorithms():
    """Get list of valid algorithm names including mixed combinations."""
    base_algos = ['maddpg', 'maddpg_geometric', 'maddpg_prev_action',
                  'maddpg_prev_observation', 'maddpg_geometric_prev_action',
                  'maddpg_geometric_prev_observation',
                  'maddpg_prev_action_prev_observation',
                  'maddpg_geometric_prev_action_prev_observation',
                  'maddpg_shared_actor', 'maddpg_shared_actor_geometric',
                  'maddpg_shared_actor_prev_action', 'maddpg_shared_actor_prev_observation',
                  'maddpg_shared_actor_geometric_prev_action',
                  'maddpg_shared_actor_geometric_prev_observation',
                  'maddpg_shared_actor_prev_action_prev_observation',
                  'maddpg_shared_actor_geometric_prev_action_prev_observation']
    team_algos = get_available_algorithms()

    # Add mixed algorithm combinations
    mixed_algos = []
    for agent_algo in team_algos:
        for adv_algo in team_algos:
            mixed_algos.append(f"{agent_algo}_vs_{adv_algo}")

    return base_algos + mixed_algos


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained MADDPG or mixed-algorithm model')

    parser.add_argument('algorithm', type=str,
                        help='Algorithm name (e.g., maddpg, maddpg_vs_ddpg). '
                             'For mixed algorithms use format: agent_algo_vs_adversary_algo')
    parser.add_argument('env_name', type=str,
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
    parser.add_argument('run_id', type=str,
                        help='Run ID (folder name in results/<algorithm>/<env_name>/)')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of evaluation episodes')
    parser.add_argument('--episode_length', type=int, default=50,
                        help='Maximum steps per episode')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')

    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()

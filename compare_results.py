"""
Script to compare results across different algorithms/runs.
Generates tables similar to those in the MADDPG paper (arXiv:1706.02275).

Usage:
    python compare_results.py                    # Print summary tables
    python compare_results.py --plot             # Also generate comparison plots
    python compare_results.py --latex            # Output LaTeX formatted tables
    python compare_results.py --csv results.csv  # Export to CSV
"""
import argparse
import os
import pickle
import sys
import numpy as np

# Handle numpy version compatibility for pickled files
# Newer numpy (2.x) uses numpy._core, older uses numpy.core
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
import matplotlib.pyplot as plt
from collections import defaultdict


def compute_running_reward(rewards, window=100):
    """Compute running average of rewards."""
    running = np.zeros_like(rewards)
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        running[i] = np.mean(rewards[start:i + 1])
    return running


def load_results(results_dir='./results'):
    """Load all results from the results directory.

    New structure: results/<algorithm>/<env_name>/<run_id>/
    Returns: {algorithm: {env_name: {run_id: data}}}
    """
    results = {}

    for algorithm in sorted(os.listdir(results_dir)):
        algo_path = os.path.join(results_dir, algorithm)
        if not os.path.isdir(algo_path):
            continue

        results[algorithm] = {}

        for env_name in sorted(os.listdir(algo_path)):
            env_path = os.path.join(algo_path, env_name)
            if not os.path.isdir(env_path):
                continue

            results[algorithm][env_name] = {}

            for run_id in sorted(os.listdir(env_path)):
                run_path = os.path.join(env_path, run_id)
                if not os.path.isdir(run_path):
                    continue

                rewards_path = os.path.join(run_path, 'rewards.pkl')
                if not os.path.exists(rewards_path):
                    continue

                with open(rewards_path, 'rb') as f:
                    data = pickle.load(f)

                # Handle both old and new format
                if isinstance(data, dict) and 'agent_score' in data:
                    results[algorithm][env_name][run_id] = data
                else:
                    print(f"Warning: Old format in {algorithm}/{env_name}/{run_id}, skipping")
                    continue

    return results


def print_agent_score_table(results, last_n=1000):
    """Print table of agent scores grouped by environment for algorithm comparison."""
    print("\n" + "=" * 100)
    print("TABLE: Agent Scores by Environment (mean ± std of last {} episodes)".format(last_n))
    print("=" * 100)

    # Get all environments across all algorithms
    all_envs = set()
    for algo_data in results.values():
        all_envs.update(algo_data.keys())

    for env_name in sorted(all_envs):
        print(f"\n--- {env_name} ---")
        print(f"{'Algorithm':<35} {'Agent Score':>25} {'Adversary Score':>25}")
        print("-" * 90)

        for algorithm in sorted(results.keys()):
            if env_name not in results[algorithm]:
                continue

            for run_id in sorted(results[algorithm][env_name].keys()):
                data = results[algorithm][env_name][run_id]
                agent_scores = data['agent_score']
                adversary_scores = data['adversary_score']

                n = min(last_n, len(agent_scores))
                mean_agent = np.mean(agent_scores[-n:])
                std_agent = np.std(agent_scores[-n:])

                agent_str = f"{mean_agent:>10.2f} ± {std_agent:<10.2f}"

                if len(data.get('adversary_team', [])) > 0:
                    mean_adv = np.mean(adversary_scores[-n:])
                    std_adv = np.std(adversary_scores[-n:])
                    adv_str = f"{mean_adv:>10.2f} ± {std_adv:<10.2f}"
                else:
                    adv_str = "N/A"

                algo_label = f"{algorithm} (run {run_id})"
                print(f"{algo_label:<35} {agent_str:>25} {adv_str:>25}")

    print("\n" + "=" * 100)


def print_metrics_table(results, last_n=1000):
    """Print table of task-specific metrics by environment."""
    # Group by environment type
    env_metrics = defaultdict(list)

    for algorithm in sorted(results.keys()):
        for env_name in sorted(results[algorithm].keys()):
            for run_id in sorted(results[algorithm][env_name].keys()):
                data = results[algorithm][env_name][run_id]
                metrics = data.get('metrics', {})
                if metrics and any(len(v) > 0 if hasattr(v, '__len__') else True for v in metrics.values()):
                    env_metrics[env_name].append((algorithm, run_id, metrics))

    if not env_metrics:
        print("\nNo task-specific metrics found. Run training with updated code to collect metrics.")
        return

    # Print metrics tables by environment
    for env_name, runs in env_metrics.items():
        print(f"\n{'='*80}")
        print(f"TABLE: {env_name} - Task-Specific Metrics")
        print(f"{'='*80}")

        # Get metric names from first run
        metric_names = list(runs[0][2].keys())

        # Header
        header = f"{'Algorithm':<35}"
        for name in metric_names:
            header += f" {name:>12}"
        print(header)
        print("-" * 80)

        # Data rows
        for algorithm, run_id, metrics in runs:
            row = f"{algorithm} (run {run_id})"[:35].ljust(35)
            for name in metric_names:
                if name in metrics:
                    values = metrics[name]
                    if hasattr(values, '__len__') and len(values) > 0:
                        n = min(last_n, len(values))
                        mean_val = np.mean(values[-n:])
                        row += f" {mean_val:>12.2f}"
                    else:
                        row += f" {'N/A':>12}"
                else:
                    row += f" {'N/A':>12}"
            print(row)

        print("=" * 80)


def print_predator_prey_table(results, last_n=1000):
    """Print predator-prey specific table (like Table 3 in MADDPG paper)."""
    # Check if any algorithm has simple_tag_v3
    has_data = any('simple_tag_v3' in algo_data for algo_data in results.values())
    if not has_data:
        return

    print("\n" + "=" * 70)
    print("TABLE: Predator-Prey (simple_tag_v3)")
    print("Average number of prey catches per episode")
    print("=" * 70)
    print(f"{'Algorithm':<35} {'Catches':>15} {'Prey Survival':>15}")
    print("-" * 70)

    for algorithm in sorted(results.keys()):
        if 'simple_tag_v3' not in results[algorithm]:
            continue
        for run_id, data in results[algorithm]['simple_tag_v3'].items():
            metrics = data.get('metrics', {})
            if 'num_catches' in metrics and len(metrics['num_catches']) > 0:
                catches = metrics['num_catches']
                n = min(last_n, len(catches))
                mean_catches = np.mean(catches[-n:])

                survival = metrics.get('prey_survival_rate', [])
                if len(survival) > 0:
                    mean_survival = np.mean(survival[-n:])
                    surv_str = f"{mean_survival:.2%}"
                else:
                    surv_str = "N/A"

                algo_label = f"{algorithm} (run {run_id})"
                print(f"{algo_label:<35} {mean_catches:>15.2f} {surv_str:>15}")

    print("=" * 70)


def print_crypto_table(results, last_n=1000):
    """Print covert communication table (like Table 5 in MADDPG paper)."""
    has_data = any('simple_crypto_v3' in algo_data for algo_data in results.values())
    if not has_data:
        return

    print("\n" + "=" * 70)
    print("TABLE: Covert Communication (simple_crypto_v3)")
    print("Bob (agent) and Eve (adversary) success metrics")
    print("=" * 70)
    print(f"{'Algorithm':<35} {'Bob Reward':>15} {'Eve Reward':>15}")
    print("-" * 70)

    for algorithm in sorted(results.keys()):
        if 'simple_crypto_v3' not in results[algorithm]:
            continue
        for run_id, data in results[algorithm]['simple_crypto_v3'].items():
            metrics = data.get('metrics', {})
            bob_reward = metrics.get('bob_reward', [])
            eve_reward = metrics.get('eve_reward', [])

            if len(bob_reward) > 0:
                n = min(last_n, len(bob_reward))
                mean_bob = np.mean(bob_reward[-n:])
                mean_eve = np.mean(eve_reward[-n:]) if len(eve_reward) > 0 else 0
                algo_label = f"{algorithm} (run {run_id})"
                print(f"{algo_label:<35} {mean_bob:>15.2f} {mean_eve:>15.2f}")

    print("=" * 70)


def print_adversary_table(results, last_n=1000):
    """Print physical deception table (like Table 4 in MADDPG paper)."""
    has_data = any('simple_adversary_v3' in algo_data for algo_data in results.values())
    if not has_data:
        return

    print("\n" + "=" * 70)
    print("TABLE: Physical Deception (simple_adversary_v3)")
    print("Agent vs Adversary performance")
    print("=" * 70)
    print(f"{'Algorithm':<35} {'Agent Wins':>15} {'Agent Reward':>15}")
    print("-" * 70)

    for algorithm in sorted(results.keys()):
        if 'simple_adversary_v3' not in results[algorithm]:
            continue
        for run_id, data in results[algorithm]['simple_adversary_v3'].items():
            metrics = data.get('metrics', {})
            agent_success = metrics.get('agent_success', [])
            agent_reward = metrics.get('agent_reward', [])

            if len(agent_success) > 0:
                n = min(last_n, len(agent_success))
                win_rate = np.mean(agent_success[-n:])
                mean_reward = np.mean(agent_reward[-n:]) if len(agent_reward) > 0 else 0
                algo_label = f"{algorithm} (run {run_id})"
                print(f"{algo_label:<35} {win_rate:>14.2%} {mean_reward:>15.2f}")

    print("=" * 70)


def print_table6_metrics(results, last_n=1000):
    """Print Table 6 style metrics: frame-level adversary performance."""
    print("\n" + "=" * 80)
    print("TABLE 6: Adversary Frame-Level Metrics (like MADDPG paper Table 6)")
    print("=" * 80)

    # (a) Keep-Away: adversary_at_goal_frames
    has_ka = any('simple_world_comm_v3' in algo_data for algo_data in results.values())
    if has_ka:
        print("\n(a) Keep-Away (simple_world_comm_v3)")
        print("    Average frames adversary occupies goal (higher = better for Adv)")
        print("-" * 60)
        print(f"    {'Algorithm':<35} {'Adv at Goal':>15}")
        print("-" * 60)
        for algorithm in sorted(results.keys()):
            if 'simple_world_comm_v3' not in results[algorithm]:
                continue
            for run_id, data in results[algorithm]['simple_world_comm_v3'].items():
                metrics = data.get('metrics', {})
                adv_frames = metrics.get('adversary_at_goal_frames', [])
                if len(adv_frames) > 0:
                    n = min(last_n, len(adv_frames))
                    mean_frames = np.mean(adv_frames[-n:])
                    algo_label = f"{algorithm} (run {run_id})"
                    print(f"    {algo_label:<35} {mean_frames:>15.2f}")

    # (b) Physical Deception: adversary_at_goal_frames
    has_pd = any('simple_adversary_v3' in algo_data for algo_data in results.values())
    if has_pd:
        print("\n(b) Physical Deception (simple_adversary_v3)")
        print("    Average frames adversary stays at goal (higher = better for Adv)")
        print("-" * 60)
        print(f"    {'Algorithm':<35} {'Adv at Goal':>15}")
        print("-" * 60)
        for algorithm in sorted(results.keys()):
            if 'simple_adversary_v3' not in results[algorithm]:
                continue
            for run_id, data in results[algorithm]['simple_adversary_v3'].items():
                metrics = data.get('metrics', {})
                adv_frames = metrics.get('adversary_at_goal_frames', [])
                if len(adv_frames) > 0:
                    n = min(last_n, len(adv_frames))
                    mean_frames = np.mean(adv_frames[-n:])
                    algo_label = f"{algorithm} (run {run_id})"
                    print(f"    {algo_label:<35} {mean_frames:>15.2f}")

    # (c) Predator-Prey: num_collisions
    has_pp = any('simple_tag_v3' in algo_data for algo_data in results.values())
    if has_pp:
        print("\n(c) Predator-Prey (simple_tag_v3)")
        print("    Average collisions per episode (lower = better for Adv)")
        print("-" * 60)
        print(f"    {'Algorithm':<35} {'Collisions':>15}")
        print("-" * 60)
        for algorithm in sorted(results.keys()):
            if 'simple_tag_v3' not in results[algorithm]:
                continue
            for run_id, data in results[algorithm]['simple_tag_v3'].items():
                metrics = data.get('metrics', {})
                collisions = metrics.get('num_collisions', [])
                if len(collisions) > 0:
                    n = min(last_n, len(collisions))
                    mean_collisions = np.mean(collisions[-n:])
                    algo_label = f"{algorithm} (run {run_id})"
                    print(f"    {algo_label:<35} {mean_collisions:>15.3f}")

    print("\n" + "=" * 80)


def export_to_csv(results, output_path, last_n=1000):
    """Export results to CSV file.

    Args:
        results: {algorithm: {env_name: {run_id: data}}}
        output_path: Path to write CSV file
        last_n: Number of last episodes to average
    """
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['Algorithm', 'Environment', 'Run', 'Agent Score Mean', 'Agent Score Std',
                  'Adversary Score Mean', 'Adversary Score Std']

        # Get all unique metric names across all algorithms
        all_metrics = set()
        for algo_data in results.values():
            for env_data in algo_data.values():
                for run_data in env_data.values():
                    if 'metrics' in run_data:
                        all_metrics.update(run_data['metrics'].keys())

        for metric in sorted(all_metrics):
            header.extend([f'{metric} Mean', f'{metric} Std'])

        writer.writerow(header)

        # Data rows
        for algorithm in sorted(results.keys()):
            for env_name in sorted(results[algorithm].keys()):
                for run_id in sorted(results[algorithm][env_name].keys()):
                    data = results[algorithm][env_name][run_id]
                    agent_scores = data['agent_score']
                    adversary_scores = data['adversary_score']

                    n = min(last_n, len(agent_scores))

                    row = [
                        algorithm,
                        env_name,
                        run_id,
                        np.mean(agent_scores[-n:]),
                        np.std(agent_scores[-n:]),
                        np.mean(adversary_scores[-n:]) if len(data.get('adversary_team', [])) > 0 else '',
                        np.std(adversary_scores[-n:]) if len(data.get('adversary_team', [])) > 0 else '',
                    ]

                    metrics = data.get('metrics', {})
                    for metric in sorted(all_metrics):
                        if metric in metrics and len(metrics[metric]) > 0:
                            values = metrics[metric]
                            n_m = min(last_n, len(values))
                            row.extend([np.mean(values[-n_m:]), np.std(values[-n_m:])])
                        else:
                            row.extend(['', ''])

                    writer.writerow(row)

    print(f"\nResults exported to: {output_path}")


def plot_comparison(results, output_path='comparison.png'):
    """Plot agent score comparison across algorithms and environments.

    Args:
        results: {algorithm: {env_name: {run_id: data}}}
        output_path: Path to save the comparison plot
    """
    # Get all algorithms and environments
    algorithms = sorted(results.keys())
    all_envs = set()
    for algo_data in results.values():
        all_envs.update(algo_data.keys())
    environments = sorted(all_envs)

    if not algorithms or not environments:
        print("No data to plot")
        return

    # Collect scores: {env: {algo: mean_score}}
    scores_by_env = {env: {} for env in environments}
    for algorithm in algorithms:
        for env_name in environments:
            if env_name in results[algorithm]:
                # Average across runs if multiple
                run_scores = []
                for run_id, data in results[algorithm][env_name].items():
                    n = min(1000, len(data['agent_score']))
                    run_scores.append(np.mean(data['agent_score'][-n:]))
                scores_by_env[env_name][algorithm] = np.mean(run_scores)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(environments))
    width = 0.8 / len(algorithms)  # Width of each bar

    # Color palette for algorithms
    colors = ['steelblue', 'indianred', 'seagreen', 'orange', 'purple', 'brown']

    for i, algorithm in enumerate(algorithms):
        scores = []
        for env in environments:
            scores.append(scores_by_env[env].get(algorithm, 0))

        offset = (i - len(algorithms) / 2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=algorithm,
                      color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel('Environment', fontsize=12)
    ax.set_ylabel('Agent Score (mean of last 1000 episodes)', fontsize=12)
    ax.set_title('Algorithm Comparison: Agent Score Across Environments', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([env.replace('_v3', '').replace('_v4', '').replace('simple_', '')
                        for env in environments],
                       rotation=45, ha='right', fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nComparison plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare MADDPG results across environments')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing results')
    parser.add_argument('--last_n', type=int, default=1000,
                        help='Number of last episodes to average for metrics')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plot')
    parser.add_argument('--output', type=str, default='comparison.png',
                        help='Output path for comparison plot')
    parser.add_argument('--csv', type=str, default=None,
                        help='Export results to CSV file')
    parser.add_argument('--latex', action='store_true',
                        help='Print LaTeX formatted tables')

    args = parser.parse_args()

    # Load results
    results = load_results(args.results_dir)

    if not results:
        print("No results found!")
        return

    # Print summary tables
    print_agent_score_table(results, args.last_n)
    print_metrics_table(results, args.last_n)

    # Print environment-specific tables
    print_predator_prey_table(results, args.last_n)
    print_crypto_table(results, args.last_n)
    print_adversary_table(results, args.last_n)
    print_table6_metrics(results, args.last_n)

    # Export to CSV
    if args.csv:
        export_to_csv(results, args.csv, args.last_n)

    # Generate plot
    if args.plot:
        plot_comparison(results, args.output)


if __name__ == '__main__':
    main()

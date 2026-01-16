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
import numpy as np
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
    """Load all results from the results directory."""
    results = {}

    for env_name in sorted(os.listdir(results_dir)):
        env_path = os.path.join(results_dir, env_name)
        if not os.path.isdir(env_path):
            continue

        results[env_name] = {}

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
                results[env_name][run_id] = data
            else:
                print(f"Warning: Old format in {env_name}/{run_id}, skipping")
                continue

    return results


def print_agent_score_table(results, last_n=1000):
    """Print table of agent scores (Table-style like MADDPG paper)."""
    print("\n" + "=" * 90)
    print("TABLE: Agent Scores (mean ± std of last {} episodes)".format(last_n))
    print("=" * 90)
    print(f"{'Environment':<30} {'Algorithm':<15} {'Agent Score':>20} {'Adversary Score':>20}")
    print("-" * 90)

    for env_name in sorted(results.keys()):
        for run_id in sorted(results[env_name].keys()):
            data = results[env_name][run_id]
            agent_scores = data['agent_score']
            adversary_scores = data['adversary_score']

            n = min(last_n, len(agent_scores))
            mean_agent = np.mean(agent_scores[-n:])
            std_agent = np.std(agent_scores[-n:])

            agent_str = f"{mean_agent:>8.2f} ± {std_agent:<8.2f}"

            if len(data.get('adversary_team', [])) > 0:
                mean_adv = np.mean(adversary_scores[-n:])
                std_adv = np.std(adversary_scores[-n:])
                adv_str = f"{mean_adv:>8.2f} ± {std_adv:<8.2f}"
            else:
                adv_str = "N/A"

            # Use run_id as algorithm name for now (can be customized)
            algorithm = f"MADDPG (run {run_id})"
            print(f"{env_name:<30} {algorithm:<15} {agent_str:>20} {adv_str:>20}")

    print("=" * 90)


def print_metrics_table(results, last_n=1000):
    """Print table of task-specific metrics by environment."""
    # Group by environment type
    env_metrics = defaultdict(list)

    for env_name in sorted(results.keys()):
        for run_id in sorted(results[env_name].keys()):
            data = results[env_name][run_id]
            metrics = data.get('metrics', {})
            if metrics:
                env_metrics[env_name].append((run_id, metrics))

    if not env_metrics:
        print("\nNo task-specific metrics found. Run training with updated code to collect metrics.")
        return

    # Print metrics tables by environment
    for env_name, runs in env_metrics.items():
        print(f"\n{'='*70}")
        print(f"TABLE: {env_name} - Task-Specific Metrics")
        print(f"{'='*70}")

        # Get metric names from first run
        metric_names = list(runs[0][1].keys())

        # Header
        header = f"{'Algorithm':<20}"
        for name in metric_names:
            header += f" {name:>15}"
        print(header)
        print("-" * 70)

        # Data rows
        for run_id, metrics in runs:
            row = f"{'MADDPG (run ' + run_id + ')':<20}"
            for name in metric_names:
                if name in metrics:
                    values = metrics[name]
                    n = min(last_n, len(values))
                    mean_val = np.mean(values[-n:])
                    row += f" {mean_val:>15.2f}"
                else:
                    row += f" {'N/A':>15}"
            print(row)

        print("=" * 70)


def print_predator_prey_table(results, last_n=1000):
    """Print predator-prey specific table (like Table 3 in MADDPG paper)."""
    if 'simple_tag_v3' not in results:
        return

    print("\n" + "=" * 60)
    print("TABLE: Predator-Prey (simple_tag_v3)")
    print("Average number of prey catches per episode")
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Catches':>15} {'Prey Survival':>15}")
    print("-" * 60)

    for run_id, data in results['simple_tag_v3'].items():
        metrics = data.get('metrics', {})
        if 'num_catches' in metrics:
            catches = metrics['num_catches']
            n = min(last_n, len(catches))
            mean_catches = np.mean(catches[-n:])

            survival = metrics.get('prey_survival_rate', [])
            if survival:
                mean_survival = np.mean(survival[-n:])
                surv_str = f"{mean_survival:.2%}"
            else:
                surv_str = "N/A"

            print(f"{'MADDPG (run ' + run_id + ')':<25} {mean_catches:>15.2f} {surv_str:>15}")

    print("=" * 60)


def print_crypto_table(results, last_n=1000):
    """Print covert communication table (like Table 5 in MADDPG paper)."""
    if 'simple_crypto_v3' not in results:
        return

    print("\n" + "=" * 60)
    print("TABLE: Covert Communication (simple_crypto_v3)")
    print("Bob (agent) and Eve (adversary) success metrics")
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Bob Reward':>15} {'Eve Reward':>15}")
    print("-" * 60)

    for run_id, data in results['simple_crypto_v3'].items():
        metrics = data.get('metrics', {})
        bob_reward = metrics.get('bob_reward', [])
        eve_reward = metrics.get('eve_reward', [])

        if bob_reward:
            n = min(last_n, len(bob_reward))
            mean_bob = np.mean(bob_reward[-n:])
            mean_eve = np.mean(eve_reward[-n:]) if eve_reward else 0
            print(f"{'MADDPG (run ' + run_id + ')':<25} {mean_bob:>15.2f} {mean_eve:>15.2f}")

    print("=" * 60)


def print_adversary_table(results, last_n=1000):
    """Print physical deception table (like Table 4 in MADDPG paper)."""
    if 'simple_adversary_v3' not in results:
        return

    print("\n" + "=" * 60)
    print("TABLE: Physical Deception (simple_adversary_v3)")
    print("Agent vs Adversary performance")
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Agent Wins':>15} {'Agent Reward':>15}")
    print("-" * 60)

    for run_id, data in results['simple_adversary_v3'].items():
        metrics = data.get('metrics', {})
        agent_success = metrics.get('agent_success', [])
        agent_reward = metrics.get('agent_reward', [])

        if agent_success:
            n = min(last_n, len(agent_success))
            win_rate = np.mean(agent_success[-n:])
            mean_reward = np.mean(agent_reward[-n:]) if agent_reward else 0
            print(f"{'MADDPG (run ' + run_id + ')':<25} {win_rate:>14.2%} {mean_reward:>15.2f}")

    print("=" * 60)


def export_to_csv(results, output_path, last_n=1000):
    """Export results to CSV file."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['Environment', 'Run', 'Agent Score Mean', 'Agent Score Std',
                  'Adversary Score Mean', 'Adversary Score Std']

        # Get all unique metric names
        all_metrics = set()
        for env_data in results.values():
            for run_data in env_data.values():
                if 'metrics' in run_data:
                    all_metrics.update(run_data['metrics'].keys())

        for metric in sorted(all_metrics):
            header.extend([f'{metric} Mean', f'{metric} Std'])

        writer.writerow(header)

        # Data rows
        for env_name in sorted(results.keys()):
            for run_id in sorted(results[env_name].keys()):
                data = results[env_name][run_id]
                agent_scores = data['agent_score']
                adversary_scores = data['adversary_score']

                n = min(last_n, len(agent_scores))

                row = [
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
    """Plot agent score comparison across environments."""
    envs = sorted(results.keys())

    # Collect scores
    scores = []
    labels = []
    for env in envs:
        for run_id in results[env]:
            data = results[env][run_id]
            n = min(1000, len(data['agent_score']))
            mean_score = np.mean(data['agent_score'][-n:])
            scores.append(mean_score)
            labels.append(f"{env}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(scores))
    colors = ['steelblue' if s >= 0 else 'indianred' for s in scores]
    bars = ax.bar(x, scores, color=colors, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Agent Score (mean of last 1000 episodes)')
    ax.set_title('MADDPG Agent Score Comparison Across Environments')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
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

    # Export to CSV
    if args.csv:
        export_to_csv(results, args.csv, args.last_n)

    # Generate plot
    if args.plot:
        plot_comparison(results, args.output)


if __name__ == '__main__':
    main()

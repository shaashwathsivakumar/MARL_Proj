"""
Paper convergence plots: MADDPG vs PTAI on 4 competitive environments.
2x2 grid, smoothed 500-episode window, ±1 std shaded bands.
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = './results'
OUT_PATH    = './results/paper_convergence.png'
SMOOTH_W    = 500

# (env_name, metric_key, y-label, panel_title)
# metric_key=None → use agent_score from rewards directly
ENVS = [
    ('simple_adversary_v3',  None,                       'Agent Reward',          'Physical Deception\n(simple_adversary_v3)'),
    ('simple_push_v3',       'agent_wins',               'Agent Win Rate',        'Cooperative Push\n(simple_push_v3)'),
    ('simple_tag_v3',        'num_catches',              'Prey Catches / Episode', 'Predator-Prey\n(simple_tag_v3)'),
    ('simple_world_comm_v3', 'adversary_at_goal_frames', 'Adv. Frames at Goal',  'Keep-Away\n(simple_world_comm_v3)'),
]

ALGOS = {
    'maddpg':      dict(label='Baseline MADDPG', color='#555555', lw=2.0, ls='--', zorder=5),
    'maddpg_ptai': dict(label='MADDPG + PTAI',  color='#2563EB', lw=2.2, ls='-',  zorder=10),
}


def smooth(arr, w):
    out = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        s = max(0, i - w + 1)
        out[i] = arr[s:i+1].mean()
    return out


def load_seeds(env_name, metric_key, algo):
    run_dir = os.path.join(RESULTS_DIR, algo, env_name)
    if not os.path.isdir(run_dir):
        return []
    seeds = []
    for run_id in sorted(os.listdir(run_dir)):
        pkl = os.path.join(run_dir, run_id, 'rewards.pkl')
        if not os.path.exists(pkl):
            continue
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or 'agent_score' not in data:
            continue
        if metric_key is None:
            arr = np.array(data['agent_score'], dtype=float)
        elif metric_key in data.get('metrics', {}):
            arr = np.array(data['metrics'][metric_key], dtype=float)
        else:
            continue
        seeds.append(arr)
    return seeds


def plot_env(ax, env_name, metric_key, panel_title):
    n_ep = 0
    algo_data = {}
    for algo in ALGOS:
        seeds = load_seeds(env_name, metric_key, algo)
        if seeds:
            algo_data[algo] = [smooth(s, SMOOTH_W) for s in seeds]
            n_ep = max(n_ep, max(len(s) for s in seeds))

    if not algo_data:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', color='grey')
        return

    x = np.arange(1, n_ep + 1)
    for algo, style in ALGOS.items():
        if algo not in algo_data:
            continue
        smoothed = algo_data[algo]
        min_len  = min(len(s) for s in smoothed)
        smoothed = [s[:min_len] for s in smoothed]
        xi = x[:min_len]
        mean = np.mean(smoothed, axis=0)

        n_seeds = len(smoothed)
        seed_label = f'{style["label"]} (n={n_seeds})'
        ax.plot(xi, mean, color=style['color'], lw=style['lw'],
                ls=style['ls'], label=seed_label, zorder=style['zorder'])
        if n_seeds > 1:
            std = np.std(smoothed, axis=0)
            ax.fill_between(xi, mean - std, mean + std,
                            color=style['color'], alpha=0.14, zorder=style['zorder'] - 1)

    ax.set_title(panel_title, fontsize=9.5, fontweight='bold', pad=6)
    ax.set_xlabel('Episode', fontsize=9)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f'{int(v / 1000)}k' if v >= 1000 else str(int(v))))
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.4, alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.85, edgecolor='#cccccc', loc='best')


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (env_name, metric_key, ylabel, title) in zip(axes, ENVS):
        plot_env(ax, env_name, metric_key, title)
        ax.set_ylabel(ylabel, fontsize=9)

    fig.suptitle(
        'Training Convergence: MADDPG vs MADDPG+PTAI\n'
        f'(500-episode smoothing window; shaded = ±1 std across seeds)',
        fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    print(f'Saved: {OUT_PATH}')


if __name__ == '__main__':
    main()

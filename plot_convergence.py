"""
Generate convergence plots comparing algorithms across competitive environments.
Shows smoothed training curves with shaded std bands for multi-seed runs.
"""
import os
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

RESULTS_DIR = './results'
OUT_DIR = './results'
SMOOTH_WINDOW = 500  # episodes for running average

# Competitive environments: (env_name, metric_key, metric_label, higher_is_better_for)
ENVS = [
    ('simple_tag_v3',       'num_catches',        'Prey Catches / Episode',    'Predators'),
    ('simple_push_v3',      'agent_wins',          'Agent Win Rate',             'Agent'),
    ('simple_crypto_v3',    'bob_success',         'Bob Success Rate',           'Alice/Bob'),
    ('simple_adversary_v3', None,                  'Agent Reward',               'Agent'),
    ('simple_world_comm_v3','adversary_at_goal_frames', 'Adv. Frames at Goal',   'Adversary'),
]

# Algorithms to plot, in display order, with label and color
ALGO_STYLE = {
    'maddpg':              dict(label='Baseline MADDPG',        color='#888888', lw=2.0, ls='--', zorder=5),
    'maddpg_ptai':         dict(label='PTAI (action)',          color='#2563EB', lw=2.2, ls='-',  zorder=10),
    'maddpg_ptai_velA':    dict(label='PTAI (vel. analytical)', color='#7C3AED', lw=1.8, ls='-',  zorder=9),
    'maddpg_ptai_velL':    dict(label='PTAI (vel. learned)',    color='#A855F7', lw=1.8, ls='-',  zorder=8),
    'maddpg_adv_gating':   dict(label='Adversary Gating',       color='#DC2626', lw=1.8, ls='-',  zorder=7),
    'maddpg_twin_critic':  dict(label='Twin Critic',            color='#D97706', lw=1.6, ls='-',  zorder=6),
    'maddpg_geometric':    dict(label='Geometric Sampling',     color='#059669', lw=1.4, ls='-',  zorder=4),
    'maddpg_prev_action':  dict(label='Prev Action',            color='#0891B2', lw=1.4, ls='-',  zorder=3),
}

# For adv_gating on push/tag, only use the default temp=0.5 runs (runs 1-4)
ADV_GATING_LIMIT = {
    'simple_push_v3': ['1', '2', '3', '4'],
    'simple_tag_v3':  ['1', '2', '3', '4'],
}


def smooth(arr, window):
    out = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        s = max(0, i - window + 1)
        out[i] = arr[s:i+1].mean()
    return out


def load_curves(env_name, metric_key):
    """Load per-episode curves for each algorithm in an environment.
    Returns: {algo: list of 1-D arrays (one per seed)}
    """
    curves = {}
    for algo in ALGO_STYLE:
        run_dir = os.path.join(RESULTS_DIR, algo, env_name)
        if not os.path.isdir(run_dir):
            continue

        run_ids = sorted(os.listdir(run_dir))

        # Filter to default-config runs for adv_gating sweep envs
        if algo == 'maddpg_adv_gating' and env_name in ADV_GATING_LIMIT:
            run_ids = [r for r in run_ids if r in ADV_GATING_LIMIT[env_name]]

        seeds = []
        for run_id in run_ids:
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

        if seeds:
            curves[algo] = seeds

    return curves


def plot_env(env_name, metric_key, metric_label, ax):
    curves = load_curves(env_name, metric_key)
    if not curves:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', color='grey')
        return

    n_episodes = max(len(s) for seeds in curves.values() for s in seeds)
    x = np.arange(1, n_episodes + 1)

    for algo, style in ALGO_STYLE.items():
        if algo not in curves:
            continue
        seeds = curves[algo]
        smoothed = [smooth(s, SMOOTH_WINDOW) for s in seeds]

        # Truncate to shortest run if mismatched
        min_len = min(len(s) for s in smoothed)
        smoothed = [s[:min_len] for s in smoothed]
        xi = x[:min_len]

        mean = np.mean(smoothed, axis=0)
        ax.plot(xi, mean,
                color=style['color'], lw=style['lw'],
                ls=style['ls'], label=style['label'],
                zorder=style['zorder'])

        if len(smoothed) > 1:
            std = np.std(smoothed, axis=0)
            ax.fill_between(xi, mean - std, mean + std,
                            color=style['color'], alpha=0.12, zorder=style['zorder'] - 1)

    ax.set_xlabel('Episode', fontsize=9)
    ax.set_ylabel(metric_label, fontsize=9)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f'{int(v/1000)}k' if v >= 1000 else str(int(v))
    ))
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (env_name, metric_key, metric_label, team) in enumerate(ENVS):
        ax = axes[i]
        short = env_name.replace('simple_', '').replace('_v3', '').replace('_v4', '')
        ax.set_title(f'{short}\n({metric_label})', fontsize=9.5, fontweight='bold', pad=6)
        plot_env(env_name, metric_key, metric_label, ax)

    # Shared legend on the last (empty) subplot
    ax_leg = axes[5]
    ax_leg.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    # Collect all handles/labels from all axes
    seen = set()
    all_handles, all_labels = [], []
    for ax in axes[:5]:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen:
                seen.add(label)
                all_handles.append(handle)
                all_labels.append(label)
    ax_leg.legend(all_handles, all_labels, loc='center', fontsize=9,
                  framealpha=0.9, edgecolor='#cccccc',
                  title='Algorithm', title_fontsize=9)

    fig.suptitle('Training Convergence — Competitive Environments\n'
                 f'(smoothed over {SMOOTH_WINDOW}-episode window, shaded = ±1 std across seeds)',
                 fontsize=10.5, y=1.01)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'convergence_competitive.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()

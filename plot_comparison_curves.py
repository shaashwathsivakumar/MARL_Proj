"""
Plot training curves for MADDPG variants on simple_tag_v3.
Compares: standard MADDPG, geometric sampling, and PTAI.
"""
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves file without opening a window
import matplotlib.pyplot as plt

# Handle numpy version compatibility for pickled files (numpy 1.x vs 2.x)
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

ALGORITHMS = ['maddpg', 'maddpg_geometric', 'maddpg_ptai', 'maddpg_prev_action']
LABELS = ['MADDPG (baseline)', 'MADDPG + Geometric Sampling', 'MADDPG + PTAI', 'MADDPG + Prev Action']
COLORS = ['steelblue', 'seagreen', 'indianred', 'darkorange']
RUNS = {'maddpg': '3', 'maddpg_geometric': '1', 'maddpg_ptai': '1', 'maddpg_prev_action': '2'}  # best completed run per algo
ENV = 'simple_tag_v3'
WINDOW = 100


def compute_running_reward(rewards, window=100):
    running = np.zeros_like(rewards)
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        running[i] = np.mean(rewards[start:i + 1])
    return running


fig, axes = plt.subplots(2, 1, figsize=(13, 10))

for algo, color, label in zip(ALGORITHMS, COLORS, LABELS):
    path = os.path.join('results', algo, ENV, RUNS[algo], 'rewards.pkl')
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping {algo}")
        continue
    with open(path, 'rb') as f:
        data = pickle.load(f)

    agent_scores = data['agent_score']
    x = np.arange(1, len(agent_scores) + 1)
    smoothed = compute_running_reward(agent_scores, WINDOW)

    # Top: smoothed only (clean comparison)
    axes[0].plot(x, smoothed, color=color, linewidth=2, label=label)

    # Bottom: raw (faint) + smoothed (bold)
    axes[1].plot(x, agent_scores, alpha=0.12, color=color)
    axes[1].plot(x, smoothed, color=color, linewidth=2, label=label)

axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Agent Score')
axes[0].set_title(f'Algorithm Comparison — {ENV}  (smoothed, window={WINDOW})')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Agent Score')
axes[1].set_title('Raw + Smoothed Training Curves')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out = 'comparison_curves.png'
plt.savefig(out, dpi=150)
print(f"Saved: {out}")

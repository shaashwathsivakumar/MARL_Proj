"""
Run MADDPG variants sequentially on all 9 MPE environments.

This trains enhancement variants (geometric sampling, prev_action, PTAI) on every
environment so we have full cross-environment comparison data for the paper.

Usage:
    python train_all_variants.py                          # run everything
    python train_all_variants.py --skip_done              # skip envs already done
    python train_all_variants.py --variants geometric     # single variant only
    python train_all_variants.py --variants ptai          # PTAI only
    python train_all_variants.py --envs simple_tag_v3 simple_adversary_v3
"""

import argparse
import subprocess
import sys
import os
import json
import time


ENVS = [
    'simple_v3',
    'simple_adversary_v3',
    'simple_crypto_v3',
    'simple_push_v3',
    'simple_reference_v3',
    'simple_speaker_listener_v4',
    'simple_spread_v3',
    'simple_tag_v3',
    'simple_world_comm_v3',
]

# Maps variant name -> extra train.py flags
VARIANTS = {
    'geometric':    ['--use_geometric_sampling'],
    'prev_action':  ['--use_prev_action'],
    'ptai':         ['--use_ai_net'],
}

# Maps variant name -> results directory name
VARIANT_DIR = {
    'geometric':   'maddpg_geometric',
    'prev_action': 'maddpg_prev_action',
    'ptai':        'maddpg_ptai',
}

TRAIN_FLAGS = [
    '--num_episodes', '30000',
    '--warmup_steps', '50000',
    '--log_interval', '1000',
    '--checkpoint_interval', '10000',
]


def already_done(variant: str, env: str) -> bool:
    """Return True if a complete rewards.pkl exists for this variant+env."""
    algo_dir = VARIANT_DIR[variant]
    env_dir   = os.path.join('results', algo_dir, env)
    if not os.path.isdir(env_dir):
        return False
    for run in sorted(os.listdir(env_dir)):
        run_path = os.path.join(env_dir, run)
        if os.path.isfile(os.path.join(run_path, 'rewards.pkl')):
            return True
    return False


def run_training(variant: str, env: str, python: str) -> bool:
    """
    Launch train.py for one (variant, env) combination.
    Returns True on success, False on failure.
    """
    cmd = [
        python, 'train.py',
        '--env_name', env,
    ] + VARIANTS[variant] + TRAIN_FLAGS

    print(f"\n{'='*70}")
    print(f"  START  variant={variant}  env={env}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}\n", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = (time.time() - t0) / 60

    if result.returncode == 0:
        print(f"\n  OK  ({elapsed:.1f} min)\n")
        return True
    else:
        print(f"\n  FAILED (exit {result.returncode}) after {elapsed:.1f} min\n")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()),
                        help='Which variants to run (default: all)')
    parser.add_argument('--envs', nargs='+', default=ENVS,
                        help='Which environments to run (default: all 9)')
    parser.add_argument('--skip_done', action='store_true',
                        help='Skip (variant, env) pairs that already have results')
    parser.add_argument('--python', type=str, default=sys.executable,
                        help='Python interpreter to use (default: same as this script)')
    args = parser.parse_args()

    tasks = [(v, e) for v in args.variants for e in args.envs]

    skipped, done, failed = [], [], []

    for variant, env in tasks:
        if args.skip_done and already_done(variant, env):
            print(f"  SKIP  {variant} / {env}  (results already exist)")
            skipped.append((variant, env))
            continue

        success = run_training(variant, env, args.python)
        if success:
            done.append((variant, env))
        else:
            failed.append((variant, env))

    # Summary
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"  Done:    {len(done)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Failed:  {len(failed)}")
    if failed:
        print(f"  Failed tasks:")
        for v, e in failed:
            print(f"    {v} / {e}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

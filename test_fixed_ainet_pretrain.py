"""
Stage 1 test: does fixing pre_train_ai_net's previous-action indexing bug alone
unlock real learning?

Pre-trains a fresh AINet on simple_tag_v3 using the now-fixed pre_train_ai_net
(identical hyperparameters to train.py's call site), saves it to an isolated
test-only filename (AI_Net_simple_tag_v3_fixed_test.pt -- never the canonical
AI_Net_simple_tag_v3.pt path), then measures it with diagnose_ainet.py's
existing (already-correct) PredictionScorer / run_random_rollout machinery --
the same methodology and num_episodes=50 used to produce the baseline numbers
in diagnostic_logs/simple_tag_v3.log (chance=0.20, accuracy=0.192, MSE=0.1612,
collapse_ratio=0.075).

No MADDPG training happens here -- this is a cheap, isolated measurement of
AINet's pre-trained prediction quality only.

Usage:
    python test_fixed_ainet_pretrain.py
"""
import torch

from PTAI import pre_train_ai_net, get_env_ptai_config
from diagnose_ainet import compute_action_offsets, run_random_rollout

ENV_NAME = 'simple_tag_v3'
SAVE_PATH = 'AI_Net_simple_tag_v3_fixed_test.pt'


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Pre-training fixed AINet for {ENV_NAME} on {device} ...")
    ai_net, sub_assignments = pre_train_ai_net(
        env_name=ENV_NAME,
        episode_count=200,
        max_episode_length=20,
        percent_to_train_on=0.8,
        lr=0.001,
        device=device,
    )

    torch.save(
        {'ai_net': ai_net.state_dict(), 'sub_assignments': sub_assignments},
        SAVE_PATH
    )
    print(f"Saved {SAVE_PATH}")

    _, cfg = get_env_ptai_config(ENV_NAME)
    offsets = compute_action_offsets(cfg)

    print(f"\n[Random-action rollouts] (matches AINet pre-training distribution)")
    scorer = run_random_rollout(
        ENV_NAME, episode_length=25, num_episodes=50,
        ai_net=ai_net, sub_assignments=sub_assignments, cfg=cfg,
        offsets=offsets, device=device
    )
    acc, mse, ratio = scorer.report("FIXED AINet -- RANDOM rollouts")

    print(f"\n=== Stage 1 result for {ENV_NAME} ===")
    print(f"  fixed AINet: accuracy={acc:.3f}  mse={mse:.4f}  collapse_ratio={ratio:.4f}")
    print(f"  baseline (buggy AINet, same methodology, diagnostic_logs/{ENV_NAME}.log):")
    print(f"    accuracy=0.192  mse=0.1612  collapse_ratio=0.075  (chance=0.20)")
    print(f"  accuracy delta (fixed - buggy baseline): {acc - 0.192:+.3f}")
    print(f"  collapse_ratio delta (fixed - buggy baseline): {ratio - 0.075:+.4f}")


if __name__ == '__main__':
    main()

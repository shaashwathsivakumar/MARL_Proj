#!/bin/bash
# Train MADDPG baseline on all 9 MPE environments sequentially

source venv/bin/activate

ENVS=(
    "simple_v3"
    "simple_adversary_v3"
    "simple_crypto_v3"
    "simple_push_v3"
    "simple_reference_v3"
    "simple_speaker_listener_v4"
    "simple_spread_v3"
    "simple_tag_v3"
    "simple_world_comm_v3"
)

echo "Starting sequential training for ${#ENVS[@]} environments..."
echo "=============================================="

for env in "${ENVS[@]}"; do
    echo ""
    echo ">>> Training $env ($(date))"
    echo "----------------------------------------------"
    python train.py \
        --env_name "$env" \
        --num_episodes 30000 \
        --warmup_steps 50000 \
        --log_interval 1000 \
        --checkpoint_interval 10000
    echo "<<< Finished $env ($(date))"
done

echo ""
echo "=============================================="
echo "All training complete! ($(date))"

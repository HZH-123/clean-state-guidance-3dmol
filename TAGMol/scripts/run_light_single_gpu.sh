#!/bin/bash
# Light trajectory version - Single GPU
# BA=0.5, SA=7.5, QED=15, Clean=1.0
# Saves full trajectory only for first 5 samples

set -e
export PYTHONPATH=".":$PYTHONPATH

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cep-qm9

CONFIG="configs/best_config_light.yml"
RESULT_PATH="experiments_multi/full_best_config_light"
BATCH_SIZE=100

mkdir -p "$RESULT_PATH"

echo "=========================================="
echo "Light Trajectory Run"
echo "BA: 0.5, SA: 7.5, QED: 15, Clean: 1.0"
echo "Device: cuda:0"
echo "Save full traj for first 5 samples only"
echo "=========================================="

for i in {0..99}; do
    echo ""
    printf ">>> [%s] Pocket %d/100 [" "$(date '+%H:%M:%S')" "$((i+1))"

    # Progress bar
    COMPLETED=$((i+1))
    for ((j=0; j<50; j++)); do
        if [ $j -lt $((COMPLETED / 2)) ]; then
            printf "="
        else
            printf " "
        fi
    done
    printf "] %d%%\n" "$((COMPLETED))"

    python -m scripts.sample_multi_guided_diffusion_light \
        "$CONFIG" \
        --data_id "$i" \
        --batch_size "$BATCH_SIZE" \
        --device cuda:0 \
        --result_path "$RESULT_PATH"

done

echo ""
echo "=========================================="
echo "Sampling Complete! Starting evaluation..."
echo "=========================================="

# Multi-process evaluation
python scripts/evaluate_diffusion_parallel.py \
    "$RESULT_PATH" \
    --protein_root data/test_set \
    --exhaustiveness 16 \
    --num_workers 32

echo ""
echo "=========================================="
echo "ALL DONE!"
echo "Results: $RESULT_PATH"
echo "=========================================="

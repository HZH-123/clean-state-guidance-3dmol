#!/bin/bash
# Single GPU sampling + multi-process evaluation
# BA=0.5, SA=7.5, QED=15, Clean=1.0

set -e
export PYTHONPATH=".":$PYTHONPATH

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cep-qm9

CONFIG="configs/best_config.yml"
RESULT_PATH="experiments_multi/full_best_config"
BATCH_SIZE=25

echo "=========================================="
echo "Single GPU Run with BEST Parameters"
echo "BA: 0.5, SA: 7.5, QED: 15, Clean: 1.0"
echo "Device: cuda:0 only"
echo "=========================================="

mkdir -p "$RESULT_PATH"

# Single GPU: run all 100 pockets sequentially, batch_size=100
BATCH_SIZE=100

for i in {0..99}; do
    echo ""
    printf ">>> [%s] Pocket %d/100 [" "$(date '+%H:%M:%S')" "$i"

    # Progress bar
    COMPLETED=$((i))
    for ((j=0; j<50; j++)); do
        if [ $j -lt $((COMPLETED / 2)) ]; then
            printf "="
        else
            printf " "
        fi
    done
    printf "] %d%%\n" "$((COMPLETED))"

    # One batch of 100 samples per pocket
    python -m scripts.sample_multi_guided_diffusion \
        "$CONFIG" \
        --data_id "$i" \
        --batch_size "$BATCH_SIZE" \
        --device cuda:0 \
        --result_path "$RESULT_PATH"

    echo "    [$(date '+%H:%M:%S')] Pocket $i done"
done

echo ""
printf ">>> [%s] Pocket 100/100 [" "$(date '+%H:%M:%S')"
printf "%50s" | tr " " "="
printf "] 100%%\n"

echo ""
echo "=========================================="
echo "Sampling Complete! Starting evaluation..."
echo "=========================================="

# Multi-process evaluation (use all CPU cores)
python scripts/evaluate_diffusion_parallel.py \
    "$RESULT_PATH" \
    --protein_root data/test_set \
    --exhaustiveness 16 \
    --num_workers 34

echo ""
echo "=========================================="
echo "ALL DONE!"
echo "Results: $RESULT_PATH"
echo "Best: BA=0.5, SA=7.5, QED=15, Clean=1.0"
echo "=========================================="

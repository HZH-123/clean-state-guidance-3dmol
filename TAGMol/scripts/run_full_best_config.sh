#!/bin/bash
# Full run with BEST parameters - DUAL GPU + AUTO DOCK EVALUATION
# BA=0.5, SA=7.5, QED=15, Clean=1.0

set -e
export PYTHONPATH=".":$PYTHONPATH

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cep-qm9

CONFIG="configs/best_config.yml"
RESULT_PATH="experiments_multi/full_best_config"
TOTAL_TASKS=100
BATCH_SIZE=100

echo "=========================================="
echo "Full Run with BEST Parameters - DUAL GPU"
echo "=========================================="
echo "BA:  0.5"
echo "SA:  7.5"
echo "QED: 15"
echo "Clean scale: 1.0"
echo ""
echo "Output: $RESULT_PATH"
echo "Pockets: 0-99 (100 total)"
echo "Samples per pocket: $BATCH_SIZE"
echo "=========================================="

mkdir -p "$RESULT_PATH"

# Function to run pockets on specific GPU
run_pockets_on_gpu() {
    local GPU_ID=$1
    local START=$2
    local END=$3

    for ((i=START; i<END; i++)); do
        echo "[GPU $GPU_ID] Pocket $i / $TOTAL_TASKS"
        python -m scripts.sample_multi_guided_diffusion \
            "$CONFIG" \
            --data_id "$i" \
            --batch_size "$BATCH_SIZE" \
            --device "cuda:$GPU_ID" \
            --result_path "$RESULT_PATH" 2>&1 | tail -3
    done
}

# Split 100 pockets: GPU 0 does 0-49, GPU 1 does 50-99
echo ""
echo "Starting sampling on dual GPUs..."
echo "  GPU 0: pockets 0-49"
echo "  GPU 1: pockets 50-99"
echo ""

run_pockets_on_gpu 0 0 50 &
PID0=$!
run_pockets_on_gpu 1 50 100 &
PID1=$!

# Wait for both GPUs to finish sampling
wait $PID0 $PID1

echo ""
echo "=========================================="
echo "Sampling Complete! Starting Vina Dock evaluation..."
echo "=========================================="

# Run multi-process vina_dock evaluation
echo "Using 32 CPU cores for parallel docking..."

python scripts/evaluate_diffusion_parallel.py \
    "$RESULT_PATH" \
    --protein_root data/test_set \
    --exhaustiveness 16 \
    --num_workers 32

echo ""
echo "=========================================="
echo "ALL DONE!"
echo "=========================================="
echo "Results: $RESULT_PATH"
echo "Eval: $RESULT_PATH/eval_results/metrics_parallel.pt"
echo ""
echo "Best config: BA=0.5, SA=7.5, QED=15, Clean=1.0"
echo "=========================================="

#!/bin/bash
# Full run with optimal parameters - DUAL GPU + AUTO DOCK EVALUATION
# 100 pockets x 100 samples on 2 GPUs, then auto vina_dock evaluation

set -e
export PYTHONPATH=".":$PYTHONPATH

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cep-qm9

CONFIG="experiments_multi/tuning_pipeline/optimal_config.yml"
RESULT_PATH="experiments_multi/full_optimal"
TOTAL_TASKS=100
BATCH_SIZE=100

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Optimal config not found. Run tuning pipeline first:"
    echo "  bash scripts/run_full_tuning_pipeline.sh"
    exit 1
fi

echo "=========================================="
echo "Full Run with Optimal Parameters - DUAL GPU"
echo "=========================================="
echo "Config: $CONFIG"
echo "Output: $RESULT_PATH"
echo "Pockets: 0-99 (100 total)"
echo "Samples per pocket: $BATCH_SIZE"
echo ""
echo "Optimal parameters:"
grep -A2 "name: qed\|name: sa\|name: binding_affinity" "$CONFIG" | grep -E "name:|gradient_scale_cord:"
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

# Run vina_dock evaluation (multi-process parallel)
echo "Starting multi-process Vina Dock evaluation..."
echo "Using $(nproc) CPU cores"

python scripts/evaluate_diffusion_parallel.py \
    "$RESULT_PATH" \
    --protein_root data/test_set \
    --exhaustiveness 16 \
    --num_workers 32  # Use 32 out of 36 cores

echo ""
echo "=========================================="
echo "ALL DONE!"
echo "=========================================="
echo "Results: $RESULT_PATH"
echo "Eval: $RESULT_PATH/eval_results/"
echo ""
echo "Key metrics:"
echo "  - QED, SA, Vina Score, Vina Min, Vina Dock"
echo "  - High Affinity %, Diversity, Hit Rate %"
echo "=========================================="

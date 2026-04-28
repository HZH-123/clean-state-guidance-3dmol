#!/bin/bash
# Light trajectory version - Dual GPU
# BA=0.5, SA=7.5, QED=15, Clean=1.0
# GPU 0: even pockets (0,2,4...), GPU 1: odd pockets (1,3,5...)
# Saves full trajectory only for first 5 samples

set -e
export PYTHONPATH=".":$PYTHONPATH

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cep-qm9

CONFIG="configs/best_config_light.yml"
RESULT_BASE="experiments_multi/full_best_config_light"
RESULT_GPU0="${RESULT_BASE}/_gpu0"
RESULT_GPU1="${RESULT_BASE}/_gpu1"
BATCH_SIZE=100

mkdir -p "$RESULT_GPU0" "$RESULT_GPU1"

echo "=========================================="
echo "Dual GPU Light Trajectory Run"
echo "BA: 0.5, SA: 7.5, QED: 15, Clean: 1.0"
echo "GPU 0: even pockets (0,2,4...)"
echo "GPU 1: odd pockets (1,3,5...)"
echo "Save full traj for first 5 samples only"
echo "=========================================="

# Function to run pockets for one GPU
run_gpu() {
    local GPU_ID=$1
    local START=$2
    local STEP=$3
    local RESULT_DIR=$4
    local TOTAL=50  # Each GPU does 50 pockets

    local count=0
    for ((i=START; i<100; i+=STEP)); do
        count=$((count+1))
        echo "[GPU $GPU_ID] Pocket $i ($count/$TOTAL)"

        python -m scripts.sample_multi_guided_diffusion_light \
            "$CONFIG" \
            --data_id "$i" \
            --batch_size "$BATCH_SIZE" \
            --device "cuda:$GPU_ID" \
            --result_path "$RESULT_DIR"
    done
}

# Start both GPUs in parallel
echo "Starting GPU 0 (even pockets)..."
run_gpu 0 0 2 "$RESULT_GPU0" &
PID0=$!

echo "Starting GPU 1 (odd pockets)..."
run_gpu 1 1 2 "$RESULT_GPU1" &
PID1=$!

# Wait for both to finish
echo ""
echo "Waiting for both GPUs to complete..."
wait $PID0
wait $PID1

echo ""
echo "=========================================="
echo "Merging results..."
echo "=========================================="

# Merge results from both GPUs
mv "$RESULT_GPU0"/result_*.pt "$RESULT_BASE/" 2>/dev/null || true
mv "$RESULT_GPU1"/result_*.pt "$RESULT_BASE/" 2>/dev/null || true
rmdir "$RESULT_GPU0" "$RESULT_GPU1" 2>/dev/null || true

echo ""
echo "=========================================="
echo "Sampling Complete! Starting evaluation..."
echo "=========================================="

# Multi-process evaluation
python scripts/evaluate_diffusion_parallel.py \
    "$RESULT_BASE" \
    --protein_root data/test_set \
    --exhaustiveness 16 \
    --num_workers 32

echo ""
echo "=========================================="
echo "ALL DONE!"
echo "Results: $RESULT_BASE"
echo "=========================================="

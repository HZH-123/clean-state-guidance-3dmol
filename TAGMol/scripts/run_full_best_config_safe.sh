#!/bin/bash
# Safe version: batch_size=10, sequential pockets per GPU
set -e
export PYTHONPATH=".":$PYTHONPATH

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cep-qm9

CONFIG="configs/best_config.yml"
RESULT_BASE="experiments_multi/full_best_config"
mkdir -p "$RESULT_BASE"

echo "Starting: BA=0.5, SA=7.5, QED=15, Clean=1.0"
echo "Strategy: batch_size=10, 10 batches per pocket, 2 pockets at a time (1 per GPU)"

# GPU 0: pockets 0-49
for i in {0..49}; do
    echo "[GPU0] Pocket $i (10 batches of 10)"
    for b in {0..9}; do
        python -m scripts.sample_multi_guided_diffusion "$CONFIG" --data_id $i --batch_size 10 --device cuda:0 --result_path "$RESULT_BASE" 2>&1 | tail -1
    done
done &
PID0=$!

# GPU 1: pockets 50-99
for i in {50..99}; do
    echo "[GPU1] Pocket $i (10 batches of 10)"
    for b in {0..9}; do
        python -m scripts.sample_multi_guided_diffusion "$CONFIG" --data_id $i --batch_size 10 --device cuda:1 --result_path "$RESULT_BASE" 2>&1 | tail -1
    done
done &
PID1=$!

wait $PID0 $PID1
echo "Sampling complete! Starting evaluation..."

python scripts/evaluate_diffusion_parallel.py "$RESULT_BASE" --protein_root data/test_set --exhaustiveness 16 --num_workers 32

echo "ALL DONE!"

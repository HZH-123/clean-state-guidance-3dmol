#!/bin/bash
# Full run with BEST parameters - FAST VERSION
# BA=0.5, SA=7.5, QED=15, Clean=1.0
# Strategy: batch_size=25, separate dirs per GPU, avoid file lock contention

set -e
export PYTHONPATH=".":$PYTHONPATH

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cep-qm9

CONFIG="configs/best_config.yml"
RESULT_BASE="experiments_multi/full_best_config"
RESULT_GPU0="${RESULT_BASE}/_gpu0"
RESULT_GPU1="${RESULT_BASE}/_gpu1"
TOTAL_TASKS=100
BATCH_SIZE=25  # Smaller batches for better parallelism

mkdir -p "$RESULT_GPU0" "$RESULT_GPU1"

echo "=========================================="
echo "Full Run with BEST Parameters - FAST VERSION"
echo "BA: 0.5, SA: 7.5, QED: 15, Clean: 1.0"
echo "Strategy: batch_size=25, separate GPU dirs"
echo "=========================================="

# Function to run pockets on specific GPU (batch_size=25, 4 batches per pocket)
run_pockets_on_gpu() {
    local GPU_ID=$1
    local START=$2
    local END=$3
    local RESULT_DIR=$4

    for ((i=START; i<END; i++)); do
        echo "[GPU $GPU_ID] Pocket $i (4 batches of 25)"

        # Run 4 batches of 25 samples each
        for batch in {0..3}; do
            python -m scripts.sample_multi_guided_diffusion \
                "$CONFIG" \
                --data_id "$i" \
                --batch_size "$BATCH_SIZE" \
                --device "cuda:$GPU_ID" \
                --result_path "$RESULT_DIR" 2>&1 | tail -2
        done

        # Merge 4 batches into 1 result file for this pocket
        python -c "
import torch, os, glob
files = sorted(glob.glob('${RESULT_DIR}/result_${i}_*.pt'))
if len(files) == 4:
    all_data = []
    for f in files:
        try:
            d = torch.load(f, map_location='cpu', weights_only=False)
            all_data.append(d)
        except:
            pass
    if all_data:
        merged = {
            'data': all_data[0]['data'],
            'pred_ligand_pos': [],
            'pred_ligand_v': [],
            'pred_ligand_pos_traj': [],
            'pred_ligand_v_traj': [],
        }
        for d in all_data:
            merged['pred_ligand_pos'].extend(d.get('pred_ligand_pos', []))
            merged['pred_ligand_v'].extend(d.get('pred_ligand_v', []))
            merged['pred_ligand_pos_traj'].extend(d.get('pred_ligand_pos_traj', []))
            merged['pred_ligand_v_traj'].extend(d.get('pred_ligand_v_traj', []))

        # Save merged
        torch.save(merged, '${RESULT_DIR}/result_${i}.pt')

        # Delete batch files
        for f in files:
            os.remove(f)
        print(f'  Merged pocket ${i}: {len(merged[\"pred_ligand_pos\"])} samples')
" 2>/dev/null || echo "  Skip merge for pocket $i"
    done
}

# Split 100 pockets: GPU 0 does 0-49, GPU 1 does 50-99
echo "Starting dual GPU sampling with batch_size=25..."

run_pockets_on_gpu 0 0 50 "$RESULT_GPU0" &
PID0=$!
run_pockets_on_gpu 1 50 100 "$RESULT_GPU1" &
PID1=$!

wait $PID0 $PID1

echo ""
echo "Merging GPU results..."
# Move all results to main directory
mv "$RESULT_GPU0"/result_*.pt "$RESULT_BASE/" 2>/dev/null || true
mv "$RESULT_GPU1"/result_*.pt "$RESULT_BASE/" 2>/dev/null || true
rmdir "$RESULT_GPU0" "$RESULT_GPU1" 2>/dev/null || true

echo "Sampling complete! Starting Vina Dock evaluation..."

python scripts/evaluate_diffusion_parallel.py \
    "$RESULT_BASE" \
    --protein_root data/test_set \
    --exhaustiveness 16 \
    --num_workers 32

echo ""
echo "=========================================="
echo "ALL DONE!"
echo "Results: $RESULT_BASE"
echo "Best config: BA=0.5, SA=7.5, QED=15, Clean=1.0"
echo "=========================================="

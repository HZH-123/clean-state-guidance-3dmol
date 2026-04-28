#!/bin/bash
# Full run with BEST parameters - MEMORY SAFE VERSION
# BA=0.5, SA=7.5, QED=15, Clean=1.0
# One pocket at a time per GPU, batch_size=10

set -e
export PYTHONPATH=".":$PYTHONPATH

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cep-qm9

CONFIG="configs/best_config.yml"
RESULT_BASE="experiments_multi/full_best_config"
RESULT_GPU0="${RESULT_BASE}/_gpu0"
RESULT_GPU1="${RESULT_BASE}/_gpu1"
TOTAL_TASKS=100
BATCH_SIZE=10  # Smaller to fit in memory

mkdir -p "$RESULT_GPU0" "$RESULT_GPU1"

echo "=========================================="
echo "Full Run with BEST Parameters - MEMORY SAFE"
echo "BA: 0.5, SA: 7.5, QED: 15, Clean: 1.0"
echo "Strategy: batch_size=10, one pocket at a time per GPU"
echo "=========================================="

# Function: run one pocket with multiple batches
run_pocket() {
    local GPU_ID=$1
    local POCKET_ID=$2
    local RESULT_DIR=$3

    echo "[GPU $GPU_ID] Pocket $POCKET_ID (10 batches of 10)"

    # Run 10 batches of 10 samples each
    for batch in {0..9}; do
        python -m scripts.sample_multi_guided_diffusion \
            "$CONFIG" \
            --data_id "$POCKET_ID" \
            --batch_size "$BATCH_SIZE" \
            --device "cuda:$GPU_ID" \
            --result_path "$RESULT_DIR" 2>&1 | tail -1
    done

    # Merge batches
    python -c "
import torch, os, glob
import sys
files = sorted(glob.glob('${RESULT_DIR}/result_${POCKET_ID}_*.pt'))
if len(files) >= 10:
    all_data = []
    for f in files:
        try:
            d = torch.load(f, map_location='cpu', weights_only=False)
            all_data.append(d)
        except:
            pass
    if all_data and len(all_data) >= 10:
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

        if len(merged['pred_ligand_pos']) == 100:
            torch.save(merged, '${RESULT_DIR}/result_${POCKET_ID}.pt')
            for f in files:
                os.remove(f)
            print(f'  [GPU ${GPU_ID}] Pocket ${POCKET_ID}: {len(merged[\"pred_ligand_pos\"])} samples - DONE')
        else:
            print(f'  [GPU ${GPU_ID}] Pocket ${POCKET_ID}: incomplete ({len(merged[\"pred_ligand_pos\"])}/100)')
            sys.exit(1)
" 2>/dev/null
}

# Assign pockets to GPUs alternately
echo "Starting..."
for ((i=0; i<50; i++)); do
    POCKET0=$i
    POCKET1=$((i + 50))

    echo ""
    echo "=== Round $((i+1))/50 ==="
    echo "GPU 0: pocket $POCKET0, GPU 1: pocket $POCKET1"

    # Run both pockets in parallel (one per GPU)
    run_pocket 0 "$POCKET0" "$RESULT_GPU0" &
    PID0=$!
    run_pocket 1 "$POCKET1" "$RESULT_GPU1" &
    PID1=$!

    # Wait for both to finish before next round
    wait $PID0 $PID1
    echo "=== Round $((i+1)) complete ==="
done

echo ""
echo "Merging results..."
mv "$RESULT_GPU0"/result_*.pt "$RESULT_BASE/" 2>/dev/null || true
mv "$RESULT_GPU1"/result_*.pt "$RESULT_BASE/" 2>/dev/null || true
rmdir "$RESULT_GPU0" "$RESULT_GPU1" 2>/dev/null || true

echo ""
echo "Starting Vina Dock evaluation..."
python scripts/evaluate_diffusion_parallel.py \
    "$RESULT_BASE" \
    --protein_root data/test_set \
    --exhaustiveness 16 \
    --num_workers 32

echo ""
echo "=========================================="
echo "ALL DONE!"
echo "=========================================="

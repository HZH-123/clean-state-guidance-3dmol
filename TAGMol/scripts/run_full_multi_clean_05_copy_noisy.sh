#!/bin/bash
# Full run: Multi-objective guidance (QED 0.33 + SA 0.33 + BA 0.34) with clean scale 0.5
# 100 pockets x 100 samples

set -e

export PYTHONPATH=".":$PYTHONPATH

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate cep-qm9

CONFIG="configs/noise_guide_multi/sampling_guided_qed_0.33_sa_0.33_ba_0.34_clean_scale_0.5_copy_noisy.yml"
RESULT_PATH="experiments_multi/full_multi_clean_05_copy_noisy"
TOTAL_TASKS=100
BATCH_SIZE=100

echo "=========================================="
echo "Full Run: Multi-Objective Clean Scale 0.5"
echo "Config: $CONFIG"
echo "Weights: QED 0.33 + SA 0.33 + BA 0.34"
echo "Pockets: 0-99 (100 total)"
echo "Samples per pocket: $BATCH_SIZE"
echo "Output: $RESULT_PATH"
echo "=========================================="

# Create output directory
mkdir -p "$RESULT_PATH"

# Run all pockets
for ((i=0; i<TOTAL_TASKS; i++)); do
    echo ""
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Pocket $i / $TOTAL_TASKS"

    python -m scripts.sample_multi_guided_diffusion \
        "$CONFIG" \
        --data_id "$i" \
        --batch_size "$BATCH_SIZE" \
        --result_path "$RESULT_PATH"

    echo "    [$(date '+%Y-%m-%d %H:%M:%S')] Pocket $i done"
done

echo ""
echo "=========================================="
echo "Sampling Complete!"
echo "Results: $RESULT_PATH"
echo "=========================================="
echo ""
echo "Next step - run evaluation:"
echo "  python scripts/evaluate_diffusion.py $RESULT_PATH --docking_mode vina_score --protein_root data/test_set"

#!/bin/bash
# Evaluate BA tuning results

set -e
export PYTHONPATH=".":$PYTHONPATH

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cep-qm9

BASE_RESULT="experiments_multi/tuning_ba"
BA_VALUES=("ba_0.5" "ba_1.0" "ba_1.5" "ba_2.0")

echo "=========================================="
echo "Evaluating BA Tuning Results"
echo "=========================================="

for ba_val in "${BA_VALUES[@]}"; do
    RESULT_DIR="$BASE_RESULT/$ba_val"

    if [ -d "$RESULT_DIR" ]; then
        echo ""
        echo ">>> Evaluating $ba_val..."

        python scripts/evaluate_diffusion_enhanced.py \
            "$RESULT_DIR" \
            --docking_mode vina_score \
            --protein_root data/test_set
    else
        echo "Warning: $RESULT_DIR not found"
    fi
done

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Check eval_results/ directories"
echo "=========================================="

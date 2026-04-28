#!/bin/bash
# Full run with optimal parameters from tuning pipeline
# 100 pockets x 100 samples

set -e
export PYTHONPATH=".":$PYTHONPATH

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cep-qm9

CONFIG="experiments_multi/tuning_pipeline/optimal_config.yml"
RESULT_PATH="experiments_multi/full_optimal"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Optimal config not found. Run tuning pipeline first:"
    echo "  bash scripts/run_full_tuning_pipeline.sh"
    exit 1
fi

echo "=========================================="
echo "Full Run with Optimal Parameters"
echo "=========================================="
echo "Config: $CONFIG"
echo "Output: $RESULT_PATH"
echo ""
echo "Optimal parameters:"
grep -A1 "binding_affinity\|synthetic_access\|drug_likeness" "$CONFIG" | grep gradient_scale_cord | head -3
echo "=========================================="

mkdir -p "$RESULT_PATH"

# Run all 100 pockets
for ((i=0; i<100; i++)); do
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] Pocket $i / 100"

    python -m scripts.sample_multi_guided_diffusion \
        "$CONFIG" \
        --data_id "$i" \
        --batch_size 100 \
        --device cuda:0 \
        --result_path "$RESULT_PATH"
done

echo ""
echo "=========================================="
echo "Full Run Complete!"
echo "Results: $RESULT_PATH"
echo "Next: Evaluate with vina_dock"
echo "=========================================="

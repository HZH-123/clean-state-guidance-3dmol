#!/bin/bash
# Full run: BA-only with selected scales (0.25, 0.5)
# Discarding 1.0 based on pilot results (31.7% success drop)

set -e
export PYTHONPATH=".":$PYTHONPATH

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cep-qm9

# All 100 test pockets
TOTAL_TASKS=100

echo "=========================================="
echo "Full Run: BA-only guidance"
echo "Scales: 0.25, 0.5 (1.0 discarded)"
echo "Pockets: 0-99 (100 total)"
echo "Samples: 100 per pocket"
echo "=========================================="

# Run 0.25
echo ">>> Running clean_0.25..."
for ((i=0; i<$TOTAL_TASKS; i++)); do
    python -m scripts.sample_multi_guided_diffusion \
        configs/noise_guide_multi/sampling_guided_ba_1_clean_scale_0.25.yml \
        --data_id $i \
        --batch_size 100 \
        --result_path "experiments_multi/full_ba_clean_0.25"
done

# Run 0.5
echo ">>> Running clean_0.5..."
for ((i=0; i<$TOTAL_TASKS; i++)); do
    python -m scripts.sample_multi_guided_diffusion \
        configs/noise_guide_multi/sampling_guided_ba_1_clean_scale_0.5.yml \
        --data_id $i \
        --batch_size 100 \
        --result_path "experiments_multi/full_ba_clean_0.5"
done

echo "Full run complete!"

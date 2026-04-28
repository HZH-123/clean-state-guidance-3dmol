#!/bin/bash
# 3-pocket pilot for clean guidance scale comparison
# 4 conditions: noisy, clean_0.25, clean_0.5, clean_1.0
# 3 pockets x 20 samples each

set -e

export PYTHONPATH=".":$PYTHONPATH

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate cep-qm9

# Pocket IDs to test (first 3 from test set)
POCKETS=(0 1 2)

# Configurations
CONFIGS=(
    "configs/noise_guide_multi/sampling_guided_ba_1_noisy_pilot.yml:ba_noisy"
    "configs/noise_guide_multi/sampling_guided_ba_1_clean_scale_0.25_pilot.yml:ba_clean_0.25"
    "configs/noise_guide_multi/sampling_guided_ba_1_clean_scale_0.5_pilot.yml:ba_clean_0.5"
    "configs/noise_guide_multi/sampling_guided_ba_1_clean_scale_1.0_pilot.yml:ba_clean_1.0"
)

echo "=========================================="
echo "Starting 3-Pocket Pilot Experiment"
echo "Pockets: ${POCKETS[@]}"
echo "Configs: 4 conditions x 20 samples"
echo "=========================================="

for config_pair in "${CONFIGS[@]}"; do
    IFS=':' read -r CONFIG_NAME EXP_NAME <<< "$config_pair"
    echo ""
    echo ">>> Running: $EXP_NAME"
    echo "Config: $CONFIG_NAME"

    for pid in "${POCKETS[@]}"; do
        echo "  - Pocket $pid ..."
        python -m scripts.sample_multi_guided_diffusion \
            "$CONFIG_NAME" \
            --data_id "$pid" \
            --batch_size 20 \
            --result_path "experiments_multi/pilot_3pockets/$EXP_NAME"
    done
    echo "  [DONE] $EXP_NAME"
done

echo ""
echo "=========================================="
echo "Pilot Complete! Results in: experiments_multi/pilot_3pockets/"
echo "=========================================="

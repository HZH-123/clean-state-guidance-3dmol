#!/bin/bash
# BA tuning with dual GPU - 4 pockets x 4 BA values (0.5, 1.0, 1.5, 2.0)
# Fixed: QED=20, SA=5, clean scale=0.5

set -e
export PYTHONPATH=".":$PYTHONPATH

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cep-qm9

# 4 pockets for tuning
POCKETS=(0 1 2 3)

# 4 BA configurations
BA_CONFIGS=(
    "configs/tuning_ba/ba_0.5.yml:ba_0.5"
    "configs/tuning_ba/ba_1.0.yml:ba_1.0"
    "configs/tuning_ba/ba_1.5.yml:ba_1.5"
    "configs/tuning_ba/ba_2.0.yml:ba_2.0"
)

BASE_RESULT="experiments_multi/tuning_ba"
mkdir -p "$BASE_RESULT"

# Function to run on specific GPU
run_on_gpu() {
    local GPU_ID=$1
    shift
    local TASKS=("$@")

    for task in "${TASKS[@]}"; do
        IFS=':' read -r CONFIG_NAME EXP_NAME POCKET_ID <<< "$task"
        echo "[GPU $GPU_ID] Running $EXP_NAME pocket $POCKET_ID"

        python -m scripts.sample_multi_guided_diffusion \
            "$CONFIG_NAME" \
            --data_id "$POCKET_ID" \
            --batch_size 50 \
            --device "cuda:$GPU_ID" \
            --result_path "$BASE_RESULT/$EXP_NAME"

        echo "[GPU $GPU_ID] Done $EXP_NAME pocket $POCKET_ID"
    done
}

# Build task list
declare -a ALL_TASKS
for config_pair in "${BA_CONFIGS[@]}"; do
    IFS=':' read -r CONFIG_NAME EXP_NAME <<< "$config_pair"
    for pid in "${POCKETS[@]}"; do
        ALL_TASKS+=("$CONFIG_NAME:$EXP_NAME:$pid")
    done
done

echo "=========================================="
echo "BA Tuning - Dual GPU"
echo "Total tasks: ${#ALL_TASKS[@]}"
echo "GPU 0: tasks 0-7"
echo "GPU 1: tasks 8-15"
echo "=========================================="

# Split tasks between GPUs
GPU0_TASKS=()
GPU1_TASKS=()
for i in "${!ALL_TASKS[@]}"; do
    if [ $i -lt 8 ]; then
        GPU0_TASKS+=("${ALL_TASKS[$i]}")
    else
        GPU1_TASKS+=("${ALL_TASKS[$i]}")
    fi
done

# Run in parallel
run_on_gpu 0 "${GPU0_TASKS[@]}" &
PID0=$!

run_on_gpu 1 "${GPU1_TASKS[@]}" &
PID1=$!

# Wait for completion
wait $PID0
wait $PID1

echo ""
echo "=========================================="
echo "BA Tuning Complete!"
echo "Results: $BASE_RESULT/"
echo "Next: bash scripts/run_tuning_ba_eval.sh"
echo "=========================================="

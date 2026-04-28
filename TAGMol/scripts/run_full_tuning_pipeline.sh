#!/bin/bash
# Full tuning pipeline: BA → SA → QED
# 4 pockets x 50 samples each, dual GPU
# Final output: optimal BA + SA + QED

set -e
export PYTHONPATH=".":$PYTHONPATH

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cep-qm9

BASE_RESULT="experiments_multi/tuning_pipeline"
mkdir -p "$BASE_RESULT"

POCKETS=(0 1 2 3)
BATCH_SIZE=50

echo "=========================================="
echo "FULL TUNING PIPELINE"
echo "=========================================="
echo "Round 1: BA tuning (QED=20, SA=5 fixed)"
echo "Round 2: SA tuning (optimal BA, QED=20 fixed)"
echo "Round 3: QED tuning (optimal BA+SA fixed)"
echo "=========================================="

# ============================================
# ROUND 1: BA TUNING
# ============================================
echo ""
echo "=========================================="
echo "ROUND 1: BA TUNING"
echo "=========================================="

BA_VALUES=(0.5 1.0 1.5 2.0)
declare -a ALL_TASKS

for ba in "${BA_VALUES[@]}"; do
    for pid in "${POCKETS[@]}"; do
        ALL_TASKS+=("configs/tuning_ba/ba_${ba}.yml:ba_${ba}:${pid}")
    done
done

echo "Total BA tasks: ${#ALL_TASKS[@]}"

# Split and run on dual GPU
GPU0_TASKS=()
GPU1_TASKS=()
for i in "${!ALL_TASKS[@]}"; do
    if [ $((i % 2)) -eq 0 ]; then
        GPU0_TASKS+=("${ALL_TASKS[$i]}")
    else
        GPU1_TASKS+=("${ALL_TASKS[$i]}")
    fi
done

run_on_gpu() {
    local GPU_ID=$1
    shift
    local TASKS=("$@")
    for task in "${TASKS[@]}"; do
        IFS=':' read -r CONFIG_NAME EXP_NAME POCKET_ID <<< "$task"
        echo "[GPU $GPU_ID] $EXP_NAME pocket $POCKET_ID"
        python -m scripts.sample_multi_guided_diffusion \
            "$CONFIG_NAME" \
            --data_id "$POCKET_ID" \
            --batch_size "$BATCH_SIZE" \
            --device "cuda:$GPU_ID" \
            --result_path "$BASE_RESULT/$EXP_NAME" 2>&1 | tail -5
    done
}

run_on_gpu 0 "${GPU0_TASKS[@]}" &
PID0=$!
run_on_gpu 1 "${GPU1_TASKS[@]}" &
PID1=$!
wait $PID0 $PID1

echo "BA tuning sampling complete. Evaluating..."

# Evaluate BA results
for ba in "${BA_VALUES[@]}"; do
    python scripts/evaluate_diffusion_enhanced.py \
        "$BASE_RESULT/ba_${ba}" \
        --docking_mode vina_score \
        --protein_root data/test_set 2>&1 | tail -20
done

# Analyze and get optimal BA
echo "Analyzing BA results..."
OPTIMAL_BA=$(python -c "
import torch, numpy as np
from pathlib import Path

base = Path('${BASE_RESULT}')
ba_values = [0.5, 1.0, 1.5, 2.0]
best_score = -float('inf')
best_ba = 1.0

for ba in ba_values:
    f = base / f'ba_{ba}' / 'eval_results' / 'metrics_-1.pt'
    if f.exists():
        data = torch.load(f, map_location='cpu', weights_only=False)
        if 'all_results' in data and data['all_results']:
            qed = [r['chem_results']['qed'] for r in data['all_results']]
            sa = [r['chem_results']['sa'] for r in data['all_results']]
            vina = [r['vina']['score_only'][0]['affinity'] for r in data['all_results'] if 'vina' in r]
            if qed and sa and vina:
                score = np.mean(qed) + np.mean(sa) + (-np.mean(vina) + 5) / 10
                print(f'BA={ba}: QED={np.mean(qed):.3f}, SA={np.mean(sa):.3f}, Vina={np.mean(vina):.3f}, Score={score:.3f}', flush=True)
                if score > best_score:
                    best_score = score
                    best_ba = ba

print(f'OPTIMAL_BA={best_ba}', flush=True)
print(best_ba)
" 2>&1 | grep "OPTIMAL_BA=" | cut -d'=' -f2)

if [ -z "$OPTIMAL_BA" ]; then
    echo "ERROR: Could not determine optimal BA. Using 1.0"
    OPTIMAL_BA=1.0
fi

echo ""
echo ">>> OPTIMAL BA = ${OPTIMAL_BA}"
echo ""

# ============================================
# ROUND 2: SA TUNING
# ============================================
echo ""
echo "=========================================="
echo "ROUND 2: SA TUNING (BA=${OPTIMAL_BA})"
echo "=========================================="

# Generate SA configs with optimal BA
cat > "${BASE_RESULT}/sa_5.yml" << EOF
model:
  checkpoint: ./pretrained_models/pretrained_diffusion.pt

guide_models:
  - name: qed
    checkpoint: ./logs/training_dock_guide_qed_2024_01_06__01_35_21/checkpoints/186000.pt
    weight: 0.33
    guide_kind: Kd
    gradient_scale_cord: 20
    gradient_scale_cord_clean: 0.5
    gradient_scale_categ: 0.0
    clamp_pred_max: 1.0
  - name: sa
    checkpoint: ./logs/training_dock_guide_sa_2024_01_20__15_38_49/checkpoints/162000.pt
    weight: 0.33
    guide_kind: Kd
    gradient_scale_cord: 5
    gradient_scale_cord_clean: 0.5
    gradient_scale_categ: 0.0
    clamp_pred_max: 1.0
  - name: binding_affinity
    checkpoint: ./logs/training_dock_guide_2023_12_17__06_23_35/checkpoints/184000.pt
    weight: 0.34
    guide_kind: Kd
    gradient_scale_cord: ${OPTIMAL_BA}
    gradient_scale_cord_clean: 0.5
    gradient_scale_categ: 0.0

sample:
  seed: 2021
  num_samples: 50
  num_steps: 1000
  pos_only: False
  center_pos_mode: protein
  guide_representation: clean
  save_traj_mode: last
  sample_num_atoms: prior
EOF

cat > "${BASE_RESULT}/sa_7.5.yml" << EOF
model:
  checkpoint: ./pretrained_models/pretrained_diffusion.pt

guide_models:
  - name: qed
    checkpoint: ./logs/training_dock_guide_qed_2024_01_06__01_35_21/checkpoints/186000.pt
    weight: 0.33
    guide_kind: Kd
    gradient_scale_cord: 20
    gradient_scale_cord_clean: 0.5
    gradient_scale_categ: 0.0
    clamp_pred_max: 1.0
  - name: sa
    checkpoint: ./logs/training_dock_guide_sa_2024_01_20__15_38_49/checkpoints/162000.pt
    weight: 0.33
    guide_kind: Kd
    gradient_scale_cord: 7.5
    gradient_scale_cord_clean: 0.5
    gradient_scale_categ: 0.0
    clamp_pred_max: 1.0
  - name: binding_affinity
    checkpoint: ./logs/training_dock_guide_2023_12_17__06_23_35/checkpoints/184000.pt
    weight: 0.34
    guide_kind: Kd
    gradient_scale_cord: ${OPTIMAL_BA}
    gradient_scale_cord_clean: 0.5
    gradient_scale_categ: 0.0

sample:
  seed: 2021
  num_samples: 50
  num_steps: 1000
  pos_only: False
  center_pos_mode: protein
  guide_representation: clean
  save_traj_mode: last
  sample_num_atoms: prior
EOF

cat > "${BASE_RESULT}/sa_10.yml" << EOF
model:
  checkpoint: ./pretrained_models/pretrained_diffusion.pt

guide_models:
  - name: qed
    checkpoint: ./logs/training_dock_guide_qed_2024_01_06__01_35_21/checkpoints/186000.pt
    weight: 0.33
    guide_kind: Kd
    gradient_scale_cord: 20
    gradient_scale_cord_clean: 0.5
    gradient_scale_categ: 0.0
    clamp_pred_max: 1.0
  - name: sa
    checkpoint: ./logs/training_dock_guide_sa_2024_01_20__15_38_49/checkpoints/162000.pt
    weight: 0.33
    guide_kind: Kd
    gradient_scale_cord: 10
    gradient_scale_cord_clean: 0.5
    gradient_scale_categ: 0.0
    clamp_pred_max: 1.0
  - name: binding_affinity
    checkpoint: ./logs/training_dock_guide_2023_12_17__06_23_35/checkpoints/184000.pt
    weight: 0.34
    guide_kind: Kd
    gradient_scale_cord: ${OPTIMAL_BA}
    gradient_scale_cord_clean: 0.5
    gradient_scale_categ: 0.0

sample:
  seed: 2021
  num_samples: 50
  num_steps: 1000
  pos_only: False
  center_pos_mode: protein
  guide_representation: clean
  save_traj_mode: last
  sample_num_atoms: prior
EOF

SA_VALUES=(5 7.5 10)
declare -a SA_TASKS

for sa in "${SA_VALUES[@]}"; do
    for pid in "${POCKETS[@]}"; do
        SA_TASKS+=("${BASE_RESULT}/sa_${sa}.yml:sa_${sa}:${pid}")
    done
done

echo "Total SA tasks: ${#SA_TASKS[@]}"

GPU0_TASKS=()
GPU1_TASKS=()
for i in "${!SA_TASKS[@]}"; do
    if [ $((i % 2)) -eq 0 ]; then
        GPU0_TASKS+=("${SA_TASKS[$i]}")
    else
        GPU1_TASKS+=("${SA_TASKS[$i]}")
    fi
done

run_on_gpu 0 "${GPU0_TASKS[@]}" &
PID0=$!
run_on_gpu 1 "${GPU1_TASKS[@]}" &
PID1=$!
wait $PID0 $PID1

echo "SA tuning sampling complete. Evaluating..."

for sa in "${SA_VALUES[@]}"; do
    python scripts/evaluate_diffusion_enhanced.py \
        "$BASE_RESULT/sa_${sa}" \
        --docking_mode vina_score \
        --protein_root data/test_set 2>&1 | tail -20
done

echo "Analyzing SA results..."
OPTIMAL_SA=$(python -c "
import torch, numpy as np
from pathlib import Path

base = Path('${BASE_RESULT}')
sa_values = [5, 7.5, 10]
best_score = -float('inf')
best_sa = 5

for sa in sa_values:
    f = base / f'sa_{sa}' / 'eval_results' / 'metrics_-1.pt'
    if f.exists():
        data = torch.load(f, map_location='cpu', weights_only=False)
        if 'all_results' in data and data['all_results']:
            qed = [r['chem_results']['qed'] for r in data['all_results']]
            sa_val = [r['chem_results']['sa'] for r in data['all_results']]
            vina = [r['vina']['score_only'][0]['affinity'] for r in data['all_results'] if 'vina' in r]
            if qed and sa_val and vina:
                score = np.mean(qed) + np.mean(sa_val) + (-np.mean(vina) + 5) / 10
                print(f'SA={sa}: QED={np.mean(qed):.3f}, SA={np.mean(sa_val):.3f}, Vina={np.mean(vina):.3f}, Score={score:.3f}', flush=True)
                if score > best_score:
                    best_score = score
                    best_sa = sa

print(f'OPTIMAL_SA={best_sa}', flush=True)
print(best_sa)
" 2>&1 | grep "OPTIMAL_SA=" | cut -d'=' -f2)

if [ -z "$OPTIMAL_SA" ]; then
    echo "ERROR: Could not determine optimal SA. Using 5"
    OPTIMAL_SA=5
fi

echo ""
echo ">>> OPTIMAL SA = ${OPTIMAL_SA}"
echo ""

# ============================================
# ROUND 3: QED TUNING
# ============================================
echo ""
echo "=========================================="
echo "ROUND 3: QED TUNING (BA=${OPTIMAL_BA}, SA=${OPTIMAL_SA})"
echo "=========================================="

for qed in 15 20 25 30; do
    cat > "${BASE_RESULT}/qed_${qed}.yml" << EOF
model:
  checkpoint: ./pretrained_models/pretrained_diffusion.pt

guide_models:
  - name: qed
    checkpoint: ./logs/training_dock_guide_qed_2024_01_06__01_35_21/checkpoints/186000.pt
    weight: 0.33
    guide_kind: Kd
    gradient_scale_cord: ${qed}
    gradient_scale_cord_clean: 0.5
    gradient_scale_categ: 0.0
    clamp_pred_max: 1.0
  - name: sa
    checkpoint: ./logs/training_dock_guide_sa_2024_01_20__15_38_49/checkpoints/162000.pt
    weight: 0.33
    guide_kind: Kd
    gradient_scale_cord: ${OPTIMAL_SA}
    gradient_scale_cord_clean: 0.5
    gradient_scale_categ: 0.0
    clamp_pred_max: 1.0
  - name: binding_affinity
    checkpoint: ./logs/training_dock_guide_2023_12_17__06_23_35/checkpoints/184000.pt
    weight: 0.34
    guide_kind: Kd
    gradient_scale_cord: ${OPTIMAL_BA}
    gradient_scale_cord_clean: 0.5
    gradient_scale_categ: 0.0

sample:
  seed: 2021
  num_samples: 50
  num_steps: 1000
  pos_only: False
  center_pos_mode: protein
  guide_representation: clean
  save_traj_mode: last
  sample_num_atoms: prior
EOF
done

QED_VALUES=(15 20 25 30)
declare -a QED_TASKS

for qed in "${QED_VALUES[@]}"; do
    for pid in "${POCKETS[@]}"; do
        QED_TASKS+=("${BASE_RESULT}/qed_${qed}.yml:qed_${qed}:${pid}")
    done
done

echo "Total QED tasks: ${#QED_TASKS[@]}"

GPU0_TASKS=()
GPU1_TASKS=()
for i in "${!QED_TASKS[@]}"; do
    if [ $((i % 2)) -eq 0 ]; then
        GPU0_TASKS+=("${QED_TASKS[$i]}")
    else
        GPU1_TASKS+=("${QED_TASKS[$i]}")
    fi
done

run_on_gpu 0 "${GPU0_TASKS[@]}" &
PID0=$!
run_on_gpu 1 "${GPU1_TASKS[@]}" &
PID1=$!
wait $PID0 $PID1

echo "QED tuning sampling complete. Evaluating..."

for qed in "${QED_VALUES[@]}"; do
    python scripts/evaluate_diffusion_enhanced.py \
        "$BASE_RESULT/qed_${qed}" \
        --docking_mode vina_score \
        --protein_root data/test_set 2>&1 | tail -20
done

echo "Analyzing QED results..."
OPTIMAL_QED=$(python -c "
import torch, numpy as np
from pathlib import Path

base = Path('${BASE_RESULT}')
qed_values = [15, 20, 25, 30]
best_score = -float('inf')
best_qed = 20

for qed in qed_values:
    f = base / f'qed_{qed}' / 'eval_results' / 'metrics_-1.pt'
    if f.exists():
        data = torch.load(f, map_location='cpu', weights_only=False)
        if 'all_results' in data and data['all_results']:
            qed_val = [r['chem_results']['qed'] for r in data['all_results']]
            sa = [r['chem_results']['sa'] for r in data['all_results']]
            vina = [r['vina']['score_only'][0]['affinity'] for r in data['all_results'] if 'vina' in r]
            if qed_val and sa and vina:
                score = np.mean(qed_val) + np.mean(sa) + (-np.mean(vina) + 5) / 10
                print(f'QED={qed}: QED={np.mean(qed_val):.3f}, SA={np.mean(sa):.3f}, Vina={np.mean(vina):.3f}, Score={score:.3f}', flush=True)
                if score > best_score:
                    best_score = score
                    best_qed = qed

print(f'OPTIMAL_QED={best_qed}', flush=True)
print(best_qed)
" 2>&1 | grep "OPTIMAL_QED=" | cut -d'=' -f2)

if [ -z "$OPTIMAL_QED" ]; then
    echo "ERROR: Could not determine optimal QED. Using 20"
    OPTIMAL_QED=20
fi

echo ""
echo ">>> OPTIMAL QED = ${OPTIMAL_QED}"
echo ""

# ============================================
# FINAL SUMMARY
# ============================================
echo ""
echo "=========================================="
echo "TUNING COMPLETE - OPTIMAL PARAMETERS"
echo "=========================================="
echo ""
echo "Optimal Configuration:"
echo "  BA  (binding_affinity): gradient_scale_cord = ${OPTIMAL_BA}"
echo "  SA  (synthetic_access): gradient_scale_cord = ${OPTIMAL_SA}"
echo "  QED (drug_likeness):  gradient_scale_cord = ${OPTIMAL_QED}"
echo "  Clean scale: 0.5 (all properties)"
echo ""
echo "Config file generated: ${BASE_RESULT}/optimal_config.yml"
echo ""

# Generate optimal config
cat > "${BASE_RESULT}/optimal_config.yml" << EOF
# Optimal configuration from tuning pipeline
# BA=${OPTIMAL_BA}, SA=${OPTIMAL_SA}, QED=${OPTIMAL_QED}

model:
  checkpoint: ./pretrained_models/pretrained_diffusion.pt

guide_models:
  - name: qed
    checkpoint: ./logs/training_dock_guide_qed_2024_01_06__01_35_21/checkpoints/186000.pt
    weight: 0.33
    guide_kind: Kd
    gradient_scale_cord: ${OPTIMAL_QED}
    gradient_scale_cord_clean: 0.5
    gradient_scale_categ: 0.0
    clamp_pred_max: 1.0
  - name: sa
    checkpoint: ./logs/training_dock_guide_sa_2024_01_20__15_38_49/checkpoints/162000.pt
    weight: 0.33
    guide_kind: Kd
    gradient_scale_cord: ${OPTIMAL_SA}
    gradient_scale_cord_clean: 0.5
    gradient_scale_categ: 0.0
    clamp_pred_max: 1.0
  - name: binding_affinity
    checkpoint: ./logs/training_dock_guide_2023_12_17__06_23_35/checkpoints/184000.pt
    weight: 0.34
    guide_kind: Kd
    gradient_scale_cord: ${OPTIMAL_BA}
    gradient_scale_cord_clean: 0.5
    gradient_scale_categ: 0.0

sample:
  seed: 2021
  num_samples: 100
  num_steps: 1000
  pos_only: False
  center_pos_mode: protein
  guide_representation: clean
  save_traj_mode: last
  sample_num_atoms: prior
EOF

echo "Full run command:"
echo "  bash scripts/run_full_optimal.sh"
echo ""
echo "Results saved to: ${BASE_RESULT}/"
echo "=========================================="

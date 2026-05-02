# Guidance Representation Matters for Controllable 3D Molecular Generation

This repository implements the main experimental code for studying guidance representation in controllable 3D molecular generation. The core question is whether test-time property evaluation and gradient feedback should act on the current noisy intermediate state or on the generator's predicted clean-state estimate.

The project focuses on clean-state guidance. For 3D generators with clean-state prediction, the code reuses the backbone's predicted clean coordinates at each sampling step, computes the property surrogate loss and gradient on that representation, applies a plug-in correction to the predicted clean coordinates, and feeds the corrected clean estimate back into the original flow-matching or diffusion update. The method does not retrain the generative backbone and does not change the discrete atom or bond sampling mechanism.

## Highlights

- QM9/SemlaFlow: compares clean-state guidance, noisy-state guidance, reranking, and base sampling on a flow-matching generator.
- Single-objective and dual-objective property control: supports QM9 properties including `mu`, `alpha`, `cv`, `ehomo`, `elumo`, and `deltae`.
- Mechanism ablations: switches only the guidance representation while keeping the backbone, predictor, target conditions, loss, sampling budget, and schedule fixed.
- Trajectory analysis: records property errors on both the noisy current state and the predicted clean-state estimate during sampling.

## Repository Layout

```text
.
|-- experiments/                 # QM9 experiment runners, ablations, and report scripts
|-- models/                      # QM9 property predictors, SchNet training, and guidance utilities
|-- results/                     # Generated protocols, experiment outputs, reports, and trace analysis
|-- semla-flow/                  # SemlaFlow backbone and clean/noisy guidance implementation
|   |-- semlaflow/qm9_guided.py  # Main QM9 guided generation entry point
|   |-- semlaflow/models/fm.py   # Integrator implementation for guidance_source
|   `-- environment.yaml         # SemlaFlow/QM9 environment
```

## Environment

For the QM9/SemlaFlow code, create the environment from `semla-flow/environment.yaml`:

```bash
conda env create -f semla-flow/environment.yaml
conda activate equinv
pip install -r semla-flow/extra_requirements.txt
```

## Data and Models

Default QM9 paths:

- Raw QM9 data: `data/QM9/raw/`
- MuDM-style protocol data: `results/qm9_mudm_protocol/`
- QM9 property predictor checkpoints: `results/qm9_mudm_protocol/predictors/`
- SemlaFlow QM9 backbone checkpoint: `semla-flow/models/qm9/last.ckpt`

To regenerate the QM9 protocol files:

```bash
python semla-flow/semlaflow/build_qm9_mudm_protocol.py \
  --output_dir results/qm9_mudm_protocol
```

To retrain a QM9 property predictor:

```bash
python models/train_qm9_property.py \
  --property_name mu \
  --protocol_dir results/qm9_mudm_protocol
```

Replace `mu` with `alpha`, `cv`, `ehomo`, `elumo`, or `deltae` as needed.

## QM9 Guided Generation

`semla-flow/semlaflow/qm9_guided.py` is the main entry point for a single QM9 guided generation run. The key argument is `--guidance_source`:

- `predicted`: evaluates properties and backpropagates gradients on the model-implied clean-state estimate, i.e. clean-state guidance.
- `curr`: evaluates properties and backpropagates gradients on the current noisy state, i.e. noisy-state guidance.

Example dual-objective clean-state guidance run:

```bash
python semla-flow/semlaflow/qm9_guided.py \
  --ckpt_path semla-flow/models/qm9/last.ckpt \
  --data_path results/qm9_mudm_protocol \
  --protocol_dir results/qm9_mudm_protocol \
  --predictor_dir results/qm9_mudm_protocol/predictors \
  --dataset qm9 \
  --dataset_split test \
  --n_molecules 10000 \
  --integration_steps 200 \
  --objectives elumo mu \
  --weights 1 1 \
  --relation_mode cascade \
  --guidance_scale 15 \
  --guidance_start_t 0.5 \
  --guidance_end_t 1.0 \
  --guidance_source predicted \
  --guidance_device cuda \
  --save_json results/manual/elumo_mu_clean.json
```

Change `--guidance_source predicted` to `--guidance_source curr` to run the noisy-state guidance counterpart.

## Reproducing Experiments

Run the QM9 single-objective and dual-objective main experiments:

```bash
python experiments/run_mudm_suite.py \
  --output_dir results/mudm_suite \
  --guidance_device cuda
```

Run the fixed-scale dual-objective experiment:

```bash
python experiments/run_mudm_multi_scale30.py \
  --output_dir results/mudm_multi_scale30 \
  --guided_scale 30 \
  --guidance_device cuda
```

Run the mechanism ablation comparing `Base`, `Rerank`, `Noisy-guidance`, and `Clean-guidance`:

```bash
python experiments/run_qm9_multi_ablation.py \
  --output_dir results/qm9_multi_ablation \
  --guidance_device cuda
```

Generate the ablation report:

```bash
python experiments/make_qm9_multi_ablation_report.py \
  --input_dirs results/qm9_multi_ablation \
  --output results/qm9_multi_ablation_summary/ablation_report.md
```

Generate the trajectory analysis report and plots:

```bash
python experiments/make_qm9_trace_analysis_report.py \
  --input_dirs results/qm9_multi_ablation \
  --output_dir results/qm9_trace_analysis
```

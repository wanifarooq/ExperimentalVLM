# Frequency Alignment Run Guide

## Current Profiles

- `frequency_alignment/configs/local_test.yaml`
  - Tuned for the local machine detected on March 13, 2026: `NVIDIA GeForce RTX 5070 Ti Laptop GPU` with `12 GB` VRAM.
  - Uses `Qwen/Qwen3-VL-2B-Instruct`.
  - Purpose: smoke-test the full pipeline with very small sample counts.
- `frequency_alignment/configs/default.yaml`
  - Tuned for a server with `2x+ NVIDIA A100` class GPUs.
  - Uses `Qwen/Qwen3-VL-30B-A3B-Instruct` with `device_map: auto`.
  - Purpose: full experimental runs with vision-token drift enabled.

## Model Families

The code now uses a shared Hugging Face VLM adapter for the current supported families:

- `Qwen/Qwen3-VL-*`
- `llava-hf/llava-v1.6-mistral-7b-hf`
- `OpenGVLab/InternVL3-8B-hf`

Qwen remains the primary fully documented path for the full experiment stack. LLaVA and InternVL are available through the same scoring / internals adapter path so they can be swapped through `model.primary`.

## Install

```bash
pip install torch torchvision transformers accelerate scipy matplotlib numpy Pillow requests pyyaml
```

Optional:

```bash
pip install bitsandbytes
pip install segment-anything-2
pip install pycocotools
```

`pycocotools` is only needed if your PartImageNet annotations are stored as COCO RLE instead of polygons.

## Quick Commands

```bash
# Local smoke run
python3 -m frequency_alignment.run_experiment \
  --experiment all \
  --config frequency_alignment/configs/local_test.yaml \
  -v

# Full server run
python3 -m frequency_alignment.run_experiment \
  --experiment all \
  --config frequency_alignment/configs/default.yaml \
  -v

# Plot-only regeneration
python3 -m frequency_alignment.run_experiment \
  --plot-only \
  --config frequency_alignment/configs/default.yaml
```

## What Each Experiment Does Now

### Experiment 1

- Uses the configured VQA dataset. For the main research path this should be `gqa`.
- Keeps the original same-image `L1-L4` labels, and also stores a continuous semantic complexity score per task based on a structured semantic program with a grounding-ambiguity term.
- Also stores an explicit `option_hardness_score` control for MCQ discrimination difficulty.
- Scores MCQ options with the parent repository’s assistant-continuation log-likelihood routine.
- Stores raw and relative image-space perturbation spectra: `delta_f`, `delta_f_relative`.
- When vision-token extraction is enabled, also stores raw and relative feature-space perturbation spectra: `delta_f_vision`, `delta_f_vision_relative`.
- Stores the correct-option score change used downstream as `loglik_drift`, plus optional cosine drift and Dirichlet deltas.
- Saves `complexity_points.json` so degradation can be analyzed as a continuous function of semantic complexity, while `prompt_complexity_score` and `option_hardness_score` remain available as controls.
- `hypothesis_tests.json` now includes within-image fixed-effects regressions and multivariate horse-race regressions.

### Experiment 2

- Extracts the effective language-to-vision attention map from the self-attention slice returned by the model.
- Keeps the original level-wise summaries, and also stores a continuous semantic complexity score for each `(image, level)` task, plus a separate prompt-load control score.
- Also stores `option_hardness_score` so semantic complexity can be tested against MCQ discrimination difficulty.
- Applies the configured attention FFT window before the 2D FFT to reduce spectral leakage from patch-grid borders. Supported values are `hann`, `hamming`, and the off modes `none` / `off` / `false`. The current configs default this to `none`.
- Computes radial power spectra, `W_t`, and bandwidth for `overall`, `early`, `mid`, and `late` layer groups.
- Runs prompt-only controls (`empty_language`, `random_language`) on the same images and stores divergence-to-task statistics.
- Saves both per-sample and average filters for downstream experiments.
- Saves `complexity_points.json` so effective bandwidth can be plotted against continuous semantic complexity.
- `hypothesis_tests.json` now includes both pooled and within-image horse-race regressions (`semantic`, `prompt load`, `option hardness`) in addition to the univariate trends.

### Experiment 3

- Loads the real `W_t` filters from Experiment 2.
- Computes task-agnostic pre-fusion band drift from vision tokens only, once per perturbation.
- Computes the task-conditioned post-fusion response as scalar drift over all aligned tokens from a late decoder hidden state.
- Groups perturbations by similar pre-fusion drift profiles so the controlled input is approximately held fixed.
- Reports controlled overlap-vs-response correlations for `overall`, `early`, `mid`, and `late`, with `late` as the main post-fusion test.
- Also runs continuous regressions from semantic complexity to internal response metrics (`ΔZ_all` and response amplification), including within-image fixed effects and horse-race controls over semantic complexity, prompt load, and option hardness.
- The plotting layer now includes pooled-vs-within-image coefficient plots and a dedicated two-factor "tug-of-war" bar chart for internal drift.

### Experiment 4

- Runs the low-pass / high-pass sweep on the same multilevel GQA samples.
- Reports the critical cutoff where accuracy crosses the configured threshold.
- Saves `complexity_points.json` so critical cutoffs can be analyzed against continuous semantic complexity, with prompt load and option hardness retained as control variables.
- `hypothesis_tests.json` now includes within-image fixed-effects and multivariate horse-race regressions for the continuous analysis.

### Experiment 5

- Uses real per-sample `W_t` filters from Experiment 2 for `overall`, `early`, `mid`, and `late`.
- Uses both image-space and vision-feature perturbation spectra from Experiment 1.
- Keeps both raw and relative-normalized overlap branches; the primary aliases `image_space` and `vision_feature_space` currently point to the relative branch, while raw branches remain available as controls.
- Reports grouped and sample-level overlap correlations against:
  - `accuracy_drop`
  - `loglik_erosion`
  - `net_drop`
  - grouped `relative_accuracy_drop`
- Treats the `late` filter group as the primary hypothesis test and the others as controls.
- Also saves a standardized "comparable bridge" view in parallel with the raw overlap analysis:
  - prediction uses `zscore(log1p(S_pred))`
  - observed targets use `zscore(actual)`
  - this is for effect-size comparability and visualization only; the raw overlap metrics remain intact
- Also tracks whether predicted overlap and calibrated prediction error vary continuously with semantic complexity.

## Two-Factor Interpretation

Across the continuous analyses, the intended interpretation is:

- `semantic complexity` measures **compositional precision**
- `prompt load` measures potential **linguistic anchoring**
- `option hardness` controls for MCQ discrimination difficulty

The fixed-effects and horse-race regressions are therefore the main evidence for whether semantic complexity is the real driver of frequency-based fragility after controlling for prompt length and answer-set difficulty.

### Experiment 6

- Requires `PartImageNet` and `SAM2` / `SAM3`.
- Uses GT segmentation masks from PartImageNet.
- Prompts `SAM3` with text and `SAM2` with GT boxes.
- Evaluates clean and perturbed `mIoU` against GT masks, not mask-to-mask self-consistency.

## Local Run Notes

The local profile is intentionally conservative:

- `Qwen/Qwen3-VL-2B-Instruct`
- severity level `1` only
- `exp6.enabled: false`
- small sample counts for all experiments

Suggested commands:

```bash
python3 -m frequency_alignment.run_experiment \
  --experiment 1 \
  --config frequency_alignment/configs/local_test.yaml \
  -v

python3 -m frequency_alignment.run_experiment \
  --experiment "1 2 5" \
  --config frequency_alignment/configs/local_test.yaml \
  -v
```

## Server Run Notes

The server profile assumes multiple visible GPUs and relies on model sharding:

```yaml
model:
  primary: Qwen/Qwen3-VL-30B-A3B-Instruct
  device_map: auto
```

If you only have a single large GPU, switch to:

```yaml
model:
  primary: Qwen/Qwen3-VL-8B-Instruct
  device_map: null
```

Suggested commands:

```bash
nohup python3 -m frequency_alignment.run_experiment \
  --experiment all \
  --config frequency_alignment/configs/default.yaml \
  -v > frequency_alignment_server.log 2>&1 &
```

## Dataset Expectations

- `gqa`
  - Main dataset for experiments `1-5`.
  - The code generates same-image L1-L4 tasks from scene graphs.
- `partimagenet`
  - Dataset for experiment `6`.
  - The code expects hierarchical object / part / subpart annotations.
- `seedbench`
  - The adapter exists, but it is L4-only.
  - It is not the primary research path for experiments `1-5`, which require same-image multilevel tasks.

## Output Files

```text
frequency_alignment_outputs/
  config_snapshot.json
  combined/all_results.json
  exp1/summary.json
  exp1/hypothesis_tests.json
  exp1/per_sample.jsonl
  exp1/complexity_points.json
  exp2/summary.json
  exp2/hypothesis_tests.json
  exp2/power_spectra.json
  exp2/complexity_points.json
  exp3/summary.json
  exp3/hypothesis_tests.json
  exp3/amplification.json
  exp4/summary.json
  exp4/accuracy_curves.json
  exp4/complexity_points.json
  exp5/summary.json
  exp5/hypothesis_tests.json
  exp5/scatter_data.json
  exp5/vision_scatter_data.json
  exp5/sample_scatter_data.json
  exp5/vision_sample_scatter_data.json
  exp5/scatter_data_by_group.json
  exp5/sample_scatter_data_by_group.json
  exp5/scatter_data_by_group_and_target.json
  exp5/sample_scatter_data_by_group_and_target.json
  exp6/summary.json
  exp6/per_sample.jsonl
  plots/*.png
```

## Troubleshooting

### Out of Memory

- Local machine: stay on `Qwen/Qwen3-VL-2B-Instruct`.
- Single A100: switch server profile from `30B-A3B` to `8B`.
- Increase `exp2.layer_stride` or reduce `max_samples`.

### Experiment 3 or 5 Missing Inputs

Run in dependency order:

```bash
python3 -m frequency_alignment.run_experiment \
  --experiment "1 2 3 5" \
  --config frequency_alignment/configs/local_test.yaml \
  -v
```

### Experiment 6 No Samples

Make sure PartImageNet exists in one of the searched locations or point the config at a local copy under `.hf_cache/partimagenet` / `~/datasets/PartImageNet` / `/data/PartImageNet`.

### Offline Runs

Use:

```bash
python3 -m frequency_alignment.run_experiment \
  --offline \
  --config frequency_alignment/configs/default.yaml
```

This forwards cache-only loading to the parent repository model loader.

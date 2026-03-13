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
- Scores MCQ options with the parent repository’s assistant-continuation log-likelihood routine.
- Stores actual per-perturbation `delta_f` spectra, optional cosine drift, and optional Dirichlet deltas.

### Experiment 2

- Extracts per-level attention tensors from the model output.
- Computes radial power spectra and per-sample `W_t` filters.
- Saves both per-sample and average filters for downstream experiments.

### Experiment 3

- Loads the real `W_t` filters from Experiment 2.
- Compares pre-fusion vision tokens with post-fusion vision-token slices from the decoder hidden states.
- Runs FFT-based drift on the actual patch grid instead of an implicit square reshape.

### Experiment 4

- Runs the low-pass / high-pass sweep on the same multilevel GQA samples.
- Reports the critical cutoff where accuracy crosses the configured threshold.

### Experiment 5

- Uses real `delta_f` vectors emitted by Experiment 1.
- Uses real per-sample `W_t` filters from Experiment 2.
- Reports grouped and sample-level overlap correlations.

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
  exp2/summary.json
  exp2/power_spectra.json
  exp2/filters/*.npy
  exp3/summary.json
  exp3/amplification.json
  exp4/summary.json
  exp4/accuracy_curves.json
  exp5/summary.json
  exp5/scatter_data.json
  exp5/sample_scatter_data.json
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

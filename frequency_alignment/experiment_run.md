# Experiment Run Guide

## Prerequisites

### Python Packages (Required)

```bash
# If using system Python (Ubuntu 24.04+)
pip install --user --break-system-packages torch torchvision transformers accelerate scipy matplotlib numpy Pillow requests pyyaml

# If using conda/venv
pip install torch torchvision transformers accelerate scipy matplotlib numpy Pillow requests pyyaml
```

### Optional Packages

```bash
# For 4-bit/8-bit quantization (needed for 8B model on <16GB VRAM)
pip install --user --break-system-packages bitsandbytes

# For Experiment 6 (segmentation) -- SAM models
pip install --user --break-system-packages segment-anything-2
```

### Verify GPU

```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"
```

---

## Quick Reference: All Commands

```bash
# Run single experiment locally
python3 -m frequency_alignment.run_experiment --experiment 1 --config frequency_alignment/configs/local_test.yaml -v

# Run all experiments locally
python3 -m frequency_alignment.run_experiment --experiment all --config frequency_alignment/configs/local_test.yaml -v

# Run on server (full scale)
python3 -m frequency_alignment.run_experiment --experiment all --config frequency_alignment/configs/default.yaml -v

# Regenerate plots from existing results
python3 -m frequency_alignment.run_experiment --plot-only --config frequency_alignment/configs/local_test.yaml

# Override samples from command line
python3 -m frequency_alignment.run_experiment --experiment 1 --config frequency_alignment/configs/local_test.yaml --max-samples 50 -v
```

---

## Local Testing (RTX 4070 / 5070 Ti / any 12-16 GB GPU)

### Config File: `frequency_alignment/configs/local_test.yaml`

This config uses the small 2B model, few samples, and severity 1 only.
Purpose: verify the pipeline works end-to-end, NOT to get meaningful results.

### Test Each Experiment Individually

#### Experiment 1: Task Granularity Spectrum

```bash
python3 -m frequency_alignment.run_experiment \
    --experiment 1 \
    --config frequency_alignment/configs/local_test.yaml \
    --max-samples 3 \
    -v
```

What happens:
- Downloads GQA scene graphs (~43 MB, one-time)
- Downloads 3 individual images from Visual Genome (~100 KB each)
- Loads Qwen3-VL-2B onto GPU (~4 GB VRAM)
- Generates 4-level questions per image
- Applies 13 perturbations at severity 1
- Scores each (level, perturbation) via log-likelihood
- Saves results to `frequency_alignment_outputs_local/exp1/`

Expected time: ~1-2 minutes
Check output:
```bash
cat frequency_alignment_outputs_local/exp1/summary.json | python3 -m json.tool | head -20
cat frequency_alignment_outputs_local/exp1/hypothesis_tests.json | python3 -m json.tool
```

#### Experiment 2: Cross-Attention Frequency Analysis

```bash
python3 -m frequency_alignment.run_experiment \
    --experiment 2 \
    --config frequency_alignment/configs/local_test.yaml \
    --max-samples 3 \
    -v
```

What happens:
- Loads model with attention hooks
- For each image x level, extracts cross-attention maps
- Computes 2D FFT of attention -> radial power spectrum
- Computes W_t(omega) filter and G(t) bandwidth per level
- Saves .npy filter files (needed by Exp 3 and 5)

Expected time: ~2-3 minutes (attention extraction is slower)
Check output:
```bash
cat frequency_alignment_outputs_local/exp2/summary.json | python3 -m json.tool | head -30
ls frequency_alignment_outputs_local/exp2/filters/
```

#### Experiment 3: Pre/Post Fusion Drift

**Must run Experiment 2 first** (needs W_t filters).

```bash
# Run Exp 2 then Exp 3 in sequence
python3 -m frequency_alignment.run_experiment \
    --experiment "2 3" \
    --config frequency_alignment/configs/local_test.yaml \
    --max-samples 3 \
    -v
```

What happens:
- Loads W_t filters from exp2/filters/
- Hooks vision encoder (pre-fusion) and decoder layer 2 (post-fusion)
- Computes band-decomposed drift at both stages
- Computes amplification ratio R(omega) per band
- Correlates R(omega) with W_t(omega)

Expected time: ~2-3 minutes
Check output:
```bash
cat frequency_alignment_outputs_local/exp3/summary.json | python3 -m json.tool
```

#### Experiment 4: Synthetic Frequency Ablation

```bash
python3 -m frequency_alignment.run_experiment \
    --experiment 4 \
    --config frequency_alignment/configs/local_test.yaml \
    --max-samples 3 \
    -v
```

What happens:
- Sweeps lowpass/highpass cutoff in 5 steps (0.02 to 0.50)
- At each cutoff x level, scores MCQ on filtered image
- Finds critical cutoff omega_c* where accuracy = 50%
- Tests if omega_c* increases with granularity

Expected time: ~3-5 minutes (many cutoff steps x levels)
Check output:
```bash
cat frequency_alignment_outputs_local/exp4/accuracy_curves.json | python3 -m json.tool | head -30
```

#### Experiment 5: Spectral Overlap Prediction

**Must run Experiments 1 AND 2 first** (needs both outputs).

```bash
# Run Exp 1, 2, then 5 in sequence
python3 -m frequency_alignment.run_experiment \
    --experiment "1 2 5" \
    --config frequency_alignment/configs/local_test.yaml \
    --max-samples 3 \
    -v
```

What happens:
- Loads W_t from Exp 2 and actual drops from Exp 1
- Computes predicted sensitivity = spectral overlap integral
- Correlates predicted vs actual (Pearson r)
- Saves scatter plot data

Expected time: ~30 seconds (pure computation, no model inference)
Check output:
```bash
cat frequency_alignment_outputs_local/exp5/summary.json | python3 -m json.tool
```

#### Experiment 6: Segmentation Granularity

**Requires**: PartImageNet dataset + SAM2/SAM3 model weights.
These are NOT downloaded automatically and are large (~5 GB).

```bash
# Only if you have PartImageNet and SAM models set up:
python3 -m frequency_alignment.run_experiment \
    --experiment 6 \
    --config frequency_alignment/configs/local_test.yaml \
    --max-samples 3 \
    -v
```

If you don't have the data, skip this experiment by setting in config:
```yaml
experiments:
  exp6:
    enabled: false
```

### Run All Experiments Locally (except Exp 6)

Edit `frequency_alignment/configs/local_test.yaml` to disable Exp 6:
```yaml
experiments:
  exp6:
    enabled: false
```

Then run:
```bash
python3 -m frequency_alignment.run_experiment \
    --experiment all \
    --config frequency_alignment/configs/local_test.yaml \
    --max-samples 3 \
    -v
```

The runner automatically handles dependency order: runs 1, 2, 4 first, then 3, then 5.

Expected total time: ~10-15 minutes

---

## Server Run (A100 / H100 / Multi-GPU)

### Config File: `frequency_alignment/configs/default.yaml`

This is the full-scale config for publishable results.

### What to Change for Server

#### Step 1: Edit `frequency_alignment/configs/default.yaml`

The defaults are already set for server. Verify these match your setup:

```yaml
# Line 3: device selection
device: auto              # auto-detects CUDA. Change to "cuda:0" for specific GPU

# Line 4: cache directory -- make sure this is on a fast disk with space
cache_dir: .hf_cache      # needs ~10 GB for model + ~1 GB for GQA scene graphs

# Line 5: output directory
out_dir: frequency_alignment_outputs   # needs ~2 GB for full results

# Line 8: model -- use 8B for single GPU, 72B for multi-GPU
model:
  primary: Qwen/Qwen3-VL-8B-Instruct
  quantization: null       # no quantization on A100 (80 GB is enough)
```

#### Step 2: Check VRAM Requirements

| Model | VRAM (fp16) | VRAM (4bit) | Recommended GPU |
|-------|------------|-------------|-----------------|
| Qwen3-VL-2B | ~5 GB | ~2 GB | RTX 4070 / 5070 Ti |
| Qwen3-VL-8B | ~18 GB | ~6 GB | RTX 4090 / A100 |
| Qwen2.5-VL-72B | ~150 GB | ~40 GB | 2-4x A100 (device_map: auto) |
| LLaVA-OV-7B | ~16 GB | ~5 GB | RTX 4090 / A100 |

If your GPU has less than 20 GB, use 4-bit quantization:
```yaml
model:
  primary: Qwen/Qwen3-VL-8B-Instruct
  quantization: 4bit     # <-- change this
```

#### Step 3: Run

```bash
# Full run (all 6 experiments, ~8-12 hours on single A100)
python3 -m frequency_alignment.run_experiment \
    --experiment all \
    --config frequency_alignment/configs/default.yaml \
    -v

# Or run specific experiments
python3 -m frequency_alignment.run_experiment \
    --experiment 1 \
    --config frequency_alignment/configs/default.yaml \
    -v

# Override samples from CLI (e.g., quick server test with 50 samples)
python3 -m frequency_alignment.run_experiment \
    --experiment all \
    --config frequency_alignment/configs/default.yaml \
    --max-samples 50 \
    -v
```

#### Step 4: Run in Background (Recommended for Long Runs)

```bash
# Using nohup
nohup python3 -m frequency_alignment.run_experiment \
    --experiment all \
    --config frequency_alignment/configs/default.yaml \
    -v > experiment_log.txt 2>&1 &

# Using screen
screen -S freq_exp
python3 -m frequency_alignment.run_experiment \
    --experiment all \
    --config frequency_alignment/configs/default.yaml \
    -v
# Ctrl+A, D to detach. screen -r freq_exp to reattach.

# Using tmux
tmux new -s freq_exp
python3 -m frequency_alignment.run_experiment \
    --experiment all \
    --config frequency_alignment/configs/default.yaml \
    -v
# Ctrl+B, D to detach. tmux attach -t freq_exp to reattach.
```

### Estimated Server Runtimes (Single A100, Qwen3-VL-8B)

| Experiment | Samples | Time Estimate |
|-----------|---------|---------------|
| Exp 1 | 1000 images x 51 perturbations x 4 levels | ~4-6 hours |
| Exp 2 | 200 images x 4 levels (attention extraction) | ~1-2 hours |
| Exp 3 | 200 images x 10 perturbations | ~1-2 hours |
| Exp 4 | 500 images x 20 cutoffs x 4 levels | ~2-3 hours |
| Exp 5 | Pure computation (no GPU) | ~1 minute |
| Exp 6 | 300 images x SAM2 + SAM3 | ~1-2 hours |
| **Total** | | **~8-12 hours** |

---

## How to Change the Model

### Option A: Change in YAML Config

Edit the config file you're using:

```yaml
# In frequency_alignment/configs/default.yaml or local_test.yaml

model:
  primary: Qwen/Qwen3-VL-8B-Instruct    # <-- change this line
  quantization: null                       # <-- adjust if needed
  trust_remote_code: true
```

Available models (tested):
```yaml
# Small (local testing, 5 GB VRAM)
primary: Qwen/Qwen3-VL-2B-Instruct

# Medium (single A100, 18 GB VRAM)
primary: Qwen/Qwen3-VL-8B-Instruct

# Medium alternative (single A100, 16 GB VRAM)
primary: llava-hf/llava-onevision-qwen2-7b-ov-hf

# Medium alternative (single A100, 18 GB VRAM)
primary: OpenGVLab/InternVL2-8B

# Large (multi-GPU, 150 GB VRAM)
primary: Qwen/Qwen2.5-VL-72B-Instruct

# Old Qwen (single A100, 5 GB VRAM)
primary: Qwen/Qwen2-VL-2B-Instruct
```

**NOTE**: LLaVA and InternVL adapters are not yet implemented. Currently only
Qwen-family models work. The framework will raise `NotImplementedError` for
LLaVA/InternVL until their adapter files are written.

### Option B: Override from Command Line

```bash
python3 -m frequency_alignment.run_experiment \
    --experiment 1 \
    --config frequency_alignment/configs/local_test.yaml \
    --model-id "Qwen/Qwen2-VL-2B-Instruct" \
    -v
```

The CLI `--model-id` flag overrides whatever is in the YAML config.

### Quantization Options

```yaml
model:
  quantization: null    # Full precision (fp16 on GPU, fp32 on CPU)
  quantization: 4bit    # 4-bit quantization via bitsandbytes (requires bitsandbytes package)
  quantization: 8bit    # 8-bit quantization via bitsandbytes
  quantization: none    # Same as null, no quantization
```

---

## How to Change the Dataset

### For Experiments 1-5 (VQA tasks)

Edit the config file:

```yaml
data:
  dataset: gqa           # <-- change this
  max_samples: 1000
```

Supported values for `dataset`:
- `gqa` -- GQA with auto-generated 4-level questions from scene graphs (recommended)
- `seedbench` -- SEEDBench dataset (uses existing loader from vlm_invariance_check.py)

**GQA is strongly recommended** because it's the only dataset where we
control all 4 granularity levels from the same image. SEEDBench has
pre-made questions that don't map cleanly to our L1-L4 levels.

#### Changing GQA Sample Count

```yaml
data:
  max_samples: 1000      # Total samples with all 4 levels

experiments:
  exp1:
    max_samples: 1000     # Can override per-experiment
  exp2:
    max_samples: 200      # Exp 2 needs fewer (attention extraction is slow)
```

Or from CLI:
```bash
--max-samples 50    # Overrides ALL experiment sample counts
```

#### Using a Pre-Downloaded GQA Dataset

If you already downloaded GQA images (20 GB), point to them:

```yaml
data:
  gqa_dir: /path/to/gqa/images    # Skip auto-download, use local images
```

Otherwise, images download on-demand (~100 KB per image) from Visual Genome.

### For Experiment 6 (Segmentation)

```yaml
experiments:
  exp6:
    dataset: partimagenet    # Only option for Exp 6
    max_samples: 300

data:
  partimagenet_dir: /path/to/PartImageNet    # Set if pre-downloaded
```

PartImageNet must be downloaded manually from the official source.
It contains hierarchical part annotations (whole/part/subpart) needed
for the 3-level segmentation granularity test.

---

## How to Change Perturbations

### Severity Levels

```yaml
perturbations:
  severity_levels:
    - 1          # Mild: 4px translate, 5% scale, 10 deg rotation
    - 2          # Moderate: 8px translate, 10% scale, 20 deg rotation
    - 3          # Severe: 12px translate, 15% scale, 30 deg rotation
```

For local testing, use `[1]` only. For server, use `[1, 2, 3]`.

### Enable/Disable Perturbation Families

```yaml
perturbations:
  include_natural: true      # Translation, rotation, scale, overlay, etc.
  include_frequency: true    # Lowpass, highpass, band-limited noise
```

Setting `include_frequency: false` skips the 5 frequency perturbations,
reducing from 17 to 12 perturbations per severity level.

### Custom Severity Parameters

```yaml
perturbations:
  severity_params:
    1:
      translate: 4         # Pixels to shift
      padcrop: 4           # Pixels to pad/crop
      scale: 0.95          # Scale factor (0.95 = 5% smaller)
      rotation: 10         # Degrees to rotate
      freq_cutoff: 0.18    # Normalized frequency cutoff [0-0.5]
      freq_epsilon: 0.03137  # Noise amplitude (8/255)
      text_scale: 0.3      # Text overlay size relative to image
    2:
      translate: 8
      # ... etc
```

---

## How to Enable/Disable Specific Experiments

In your YAML config:

```yaml
experiments:
  exp1:
    enabled: true       # Set to false to skip
  exp2:
    enabled: true
  exp3:
    enabled: true
  exp4:
    enabled: true
  exp5:
    enabled: true
  exp6:
    enabled: false      # Disabled (needs PartImageNet + SAM)
```

Or run specific experiments from CLI:

```bash
# Single experiment
--experiment 1

# Multiple experiments (space or comma separated)
--experiment "1 2 4"
--experiment "1,2,4"

# All experiments (respects enabled/disabled in config)
--experiment all
```

---

## Output Structure

After a full run, you'll find:

```
frequency_alignment_outputs/           # (or _local for local test)
|-- config_snapshot.json               # Exact config used for this run
|-- combined/
|   |-- all_results.json               # Merged metrics from all experiments
|-- exp1/
|   |-- summary.json                   # Per-level accuracy and degradation stats
|   |-- hypothesis_tests.json          # Statistical test results (pass/fail)
|   |-- per_sample.jsonl               # One JSON per line, full per-image detail
|   |-- degradation_by_level.json      # (level x perturbation) breakdown
|-- exp2/
|   |-- summary.json                   # W_t averages and G(t) stats per level
|   |-- filters/                       # .npy files: W_t per sample per level
|   |-- power_spectra.json
|-- exp3/
|   |-- summary.json                   # Amplification ratios and correlations
|   |-- hypothesis_tests.json
|   |-- amplification.json
|-- exp4/
|   |-- summary.json                   # Critical cutoffs per level
|   |-- accuracy_curves.json           # Full accuracy-vs-cutoff data
|   |-- hypothesis_tests.json
|-- exp5/
|   |-- summary.json                   # Pearson r (predicted vs actual)
|   |-- hypothesis_tests.json
|   |-- scatter_data.json              # (predicted, actual) pairs
|-- exp6/
|   |-- summary.json
|   |-- hypothesis_tests.json
|   |-- per_sample.jsonl
|-- plots/
    |-- exp1_granularity_curves.png
    |-- exp2_power_spectrum.png
    |-- exp2_bandwidth.png
    |-- exp3_amplification.png
    |-- exp4_threshold_lowpass.png
    |-- exp4_threshold_highpass.png
    |-- exp5_overlap_scatter.png
    |-- exp6_segmentation.png
```

---

## Regenerating Plots

If you want to re-generate plots (e.g., after tweaking plot code) without
re-running experiments:

```bash
python3 -m frequency_alignment.run_experiment \
    --plot-only \
    --config frequency_alignment/configs/local_test.yaml
```

This reads existing JSON results from the output directory and regenerates
all plots. No GPU needed.

---

## Troubleshooting

### "No module named 'torchvision'"
```bash
pip install --user --break-system-packages torchvision
```

### "No module named 'bitsandbytes'" (only if using quantization)
```bash
pip install --user --break-system-packages bitsandbytes
```

### "externally-managed-environment" error from pip
Add `--break-system-packages` flag, or use a conda/venv environment.

### Out of Memory (OOM) on GPU
- Reduce `max_samples` or use `quantization: 4bit`
- For Exp 2: increase `layer_stride` (e.g., 8 instead of 4) to extract fewer layers
- Use the 2B model instead of 8B

### "No samples loaded. Check GQA data availability."
- GQA scene graphs may not have downloaded. Check `.hf_cache/gqa/sceneGraphs/`
- If behind a firewall, download manually from:
  https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip
  Extract to `.hf_cache/gqa/sceneGraphs/`

### Experiment 3 or 5 fails with missing data
- These depend on prior experiments. Run them in order:
  `--experiment "1 2 3 5"` or `--experiment all`

### Plots are empty or show all zeros
- Normal with 3 samples at severity 1 (too few for meaningful results)
- Run with `--max-samples 100` and severity `[1, 2, 3]` for real data

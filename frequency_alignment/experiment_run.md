# Frequency Alignment Run Guide

Last updated: April 27, 2026.

## Current Scientific Status

The original hypothesis still stands: language conditioning acts like a task-specific frequency filter `W_t`, and robustness depends on the overlap between `W_t` and perturbation energy. After the 2026-04-23 paired 200-sample runs (one DC OFF, one DC ON), the framing has been corrected once more (see `full_project_explanation.md` status banner):

- **Three-stage layer trajectory.** `W_t` shape is **early = prompt-length prior** (longer prompts → more concentrated filter, levels cluster by token count), **mid = uniform compression** (`G_mid ≈ 2.06–2.59` across all 8 levels), **late = `Csem`-driven downward concentration** (clean granularity ladder, `ρ(G_late, rank) = −1.0`).
- **Granularity axis = downward concentration.** Late-layer top-2 mass *rises*, tail mass *falls*, `G_late` shrinks with granularity. The earlier "peak consolidation + tail growth" framing is retracted: tail thins, it does not fatten.
- **Wordiness axis = layer-trajectory asymmetry.** Wordy mirrors start *more* concentrated than their semantic counterparts at early layers (length prior) and end *more* relaxed at late layers (asymmetric relaxation). The wordy↔base gap *grows* with depth, it does not converge — see `exp2/hypothesis_tests.json::wordy_layerwise_divergence` (renamed from the retracted `wordy_layerwise_convergence`).
- **DC ON is canonical.** The 2026-04-23 paired ablation shows Exp 5 first-order overlap `r = 0.471, p ≈ 10⁻³⁶` with DC ON vs `r = 0.055` with DC OFF. All five configs now use `suppress_dc: false`.
- **First-order overlap** `⟨W_t, ΔF⟩ = Σ_ω W_t(ω)·ΔF_p(ω)` is the unifying drift predictor (stored in `exp5/summary.json::first_order_overlap_tables`).
- **Low-freq perturbations dominate drift.** Both `W_t` (late) and natural `ΔF` concentrate on DC and DC-adjacent bins, which is why DC ON is the load-bearing convention.

Existing two-factor machinery is retained and gains a direct mechanistic reading:

- primary semantic force: raw semantic program complexity (`Csem`), with primary `L1-L4` reported as marginal correlations because `Csem` and prompt load are collinear on the terse ladder
- control force: prompt load / linguistic anchoring — now interpreted as the **Theorem 4 prior** rather than just a statistical control
- MCQ control: option hardness
- prompt-load experimental controls: `L5-L8` wordy levels matched to `L1-L4`
- Exp 5 primary behavioral target: correct-answer log-likelihood volatility
- triple-view reporting: Primary (`L1-L4`), Wordy (`L5-L8`), and Pooled (`L1-L8`)

## Current Profiles

- `frequency_alignment/configs/local_test.yaml`
  - Tuned for the local machine detected on March 13, 2026: `NVIDIA GeForce RTX 5070 Ti Laptop GPU` with `12 GB` VRAM.
  - Uses `Qwen/Qwen3-VL-2B-Instruct`.
  - Purpose: smoke-test the full pipeline with very small sample counts.
- `frequency_alignment/configs/default.yaml`
  - Tuned for a server with `2x+ NVIDIA A100` class GPUs.
  - Uses `Qwen/Qwen3-VL-30B-A3B-Instruct` with `device_map: auto`.
  - Purpose: full experimental runs with vision-token drift enabled.
- `frequency_alignment/configs/server_severity1.yaml`
  - Same server model as `default.yaml` but `perturbations.severity_levels: [1]` only.
  - `analysis.suppress_dc: false` (DC ON, canonical).
  - Used for the 2026-04-22 500-sample first-pass smoke run on the server.
- `frequency_alignment/configs/server_rerun.yaml` *(introduced 2026-04-23)*
  - Multi-severity follow-up.
  - `perturbations.severity_levels: [1, 2, 3]` (Exp 4 cutoff pinning required severity > 1).
  - `analysis.suppress_dc: false` (DC ON, canonical setting after the 2026-04-23 paired ablation).
  - `perturbations.include_frequency: true` (bandpass perturbations restored).
  - `experiments.exp6.enabled: false` (keeps segmentation leg silent in this run).
  - Intended for verifying the three-stage trajectory + first-order overlap predictions at multiple severities.

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

For server setup from the lean environment files in the repo:

```bash
conda env create -f clean_env.yml
conda activate vlm-robustness
pip install -r requirements-extra.txt
```

## Quick Commands

```bash
# Local smoke run
python3 -m frequency_alignment.run_experiment \
  --experiment all \
  --config frequency_alignment/configs/local_test.yaml \
  --max-samples 5 \
  --out-dir "frequency_alignment_outputs_dryrun_5samples_$(date +%Y%m%d_%H%M%S)" \
  -v

# Full server run
python3 -m frequency_alignment.run_experiment \
  --experiment all \
  --config frequency_alignment/configs/default.yaml \
  -v

# Plot-only regeneration
python3 -m frequency_alignment.run_experiment \
  --plot-only \
  --config frequency_alignment/configs/default.yaml \
  --out-dir <existing_output_dir>
```

`--max-samples N` is a global runtime override. It writes `N` into `data.max_samples` and every configured `experiments.exp*.max_samples` for that run, so all enabled experiments and downstream plots/tests use the processed subset from that override.

## What Each Experiment Does Now

### Experiment 1

- Uses the configured VQA dataset. For the main research path this should be `gqa`.
- Builds same-image primary levels `L1-L4` plus matched wordy controls `L5-L8`:
  - `L5`: L1 semantics with inflated prompt load
  - `L6`: L2 semantics with inflated prompt load
  - `L7`: L3 semantics with inflated prompt load
  - `L8`: L4 semantics with inflated prompt load
- The GQA loader makes each wordy-control prompt longer than its matched base prompt with neutral filler text, without imposing a fixed word or token cap.
- Primary monotonic granularity tests remain on `L1-L4`; descriptive and control plots can include `L5-L8`.
- Stores a continuous semantic complexity score per task based on a structured semantic program with a grounding-ambiguity term.
- Stores `prompt_complexity_score` as question-only content-word load. MCQ option strings are excluded from prompt load because option richness is controlled separately by `option_hardness_score`.
- Stores `complexity_score_residual`, the residualized semantic-logic variable used for Cres scatter plots and dual-force diagnostics after regressing semantic complexity on prompt load.
- Also stores an explicit `option_hardness_score` control for MCQ discrimination difficulty.
- Stores clean-image `prediction_entropy`, computed as `H = -sum_i p_i log(p_i)` over softmaxed MCQ option scores, as a control for model-side answer uncertainty.
- Scores MCQ options with the parent repository’s assistant-continuation log-likelihood routine.
- Stores raw and relative image-space perturbation spectra: `delta_f`, `delta_f_relative`.
- When vision-token extraction is enabled, also stores raw and relative feature-space perturbation spectra: `delta_f_vision`, `delta_f_vision_relative`.
- Stores `is_binary` and `num_options` so binary yes/no tasks and 4-way MCQ tasks can be separated from semantic complexity. Horse-race regressions include `is_binary` whenever the analyzed slice mixes both formats.
- Stores the correct-option score change used downstream as `loglik_drift`, plus directional summaries:
  - `loglik_erosion`
  - `loglik_recovery`
  - `loglik_volatility`
- Optional cosine drift and Dirichlet deltas are only computed when `exp1.extract_vision_tokens: true`.
- Saves `complexity_points.json` so degradation can be analyzed as a continuous function of semantic complexity, while `prompt_complexity_score` and `option_hardness_score` remain available as controls.
- Accuracy-drop horse races are filtered to clean-correct points only, so they measure fragility of existing knowledge.
- `hypothesis_tests.json` includes primary marginal summaries, wordy/pooled multivariate horse-race regressions, perturbation-specific summaries for accuracy drop and log-likelihood outcomes, and paired mirror tests comparing `L1-L4` against their `L5-L8` wordy controls.

### Experiment 2

- Extracts the effective language-to-vision attention map from the self-attention slice returned by the model.
- Keeps level-wise summaries for all available VQA levels, including `L5-L8` wordy controls when present, and also stores a continuous semantic complexity score for each `(image, level)` task plus a separate prompt-load control score.
- Also stores `option_hardness_score` so semantic complexity can be tested against MCQ discrimination difficulty.
- Applies the configured attention FFT window before the 2D FFT to reduce spectral leakage from patch-grid borders. Supported values are `hann`, `hamming`, and the off modes `none` / `off` / `false`. The current configs default this to `none`.
- Computes radial power spectra, `W_t`, and bandwidth for `overall`, `early`, `mid`, and `late` layer groups.
- Runs prompt-only controls (`empty_language`, `random_language`) on the same images and stores divergence-to-task statistics.
- Saves both per-sample and average filters for downstream experiments.
- Saves `complexity_points.json` so effective bandwidth can be plotted against continuous semantic complexity.
- `hypothesis_tests.json` now includes primary marginal summaries and wordy/pooled horse-race regressions (`Csem`, `prompt load`, `option hardness`) in addition to the univariate trends.
- Bandwidth plots include the wordy controls when those levels are present; primary monotonic granularity tests remain `L1-L4`.

### Experiment 3 (retired)

Exp 3 (fusion drift / response amplification) was retired in the camera-ready
cleanup. Its analysis is subsumed by the Exp 5 overlap law on pixel-space ΔF
plus the per-layer-group W_t shape statistics already in Exp 2.

### Experiment 4

- Runs the low-pass / high-pass sweep on the same multilevel GQA samples.
- Reports the critical cutoff where accuracy crosses the configured threshold.
- Saves `complexity_points.json` so critical cutoffs can be analyzed against continuous semantic complexity, with prompt load and option hardness retained as control variables.
- `hypothesis_tests.json` now includes primary marginal summaries and wordy/pooled horse-race regressions for the continuous analysis.
- Level-wise cutoff plots and wordy-control comparisons include `L5-L8` when those levels are present.

### Experiment 5

- Uses real per-sample `W_t` filters from Experiment 2 for `overall`, `early`, `mid`, and `late`.
- Uses both image-space and vision-feature perturbation spectra from Experiment 1.
- Keeps both raw and relative-normalized overlap branches; the primary aliases `image_space` and `vision_feature_space` currently point to the relative branch, while raw branches remain available as controls.
- Reports grouped and sample-level overlap correlations against:
  - `accuracy_drop`
  - `loglik_erosion`
  - `loglik_volatility`
  - `net_drop`
  - grouped `relative_accuracy_drop`
- Treats correct-answer `loglik_volatility` as the primary behavioral target. Accuracy drop, erosion, net drop, and relative accuracy drop are still emitted as controls.
- Treats the `late` filter group as the primary layer-group hypothesis test and the others as controls.
- Also saves a standardized "comparable bridge" view in parallel with the raw overlap analysis:
  - prediction uses `zscore(log1p(S_pred_first_order))`
  - observed targets use `zscore(actual)`
  - this is for effect-size comparability and visualization only; the raw overlap metrics remain intact
- Adds prediction-factor summaries for observed `accuracy_drop`, using `zscore(log1p(S_pred_first_order))`, raw `question_complexity_score`, `prompt_complexity_score`, `option_hardness_score`, and `is_binary` when applicable as predictors. Primary view is marginal-only; wordy/pooled views keep multivariate regressions. This is saved under `prediction_factor_horse_race` and plotted as `exp5_coefficient_plot_prediction_factors*.png`.
- Also tracks whether predicted overlap and calibrated prediction error vary continuously with semantic complexity.
- Produces level-wise scatter grids and level-by-perturbation predicted-vs-observed plots. These use separate visual scales when prediction and observation magnitudes differ.
- Produces wordy-control comparison plots for predicted sensitivity and observed behavior.
- Summary aliases such as `primary_target`, `primary_group`, and `pearson_r_grouped` refer to the configured primary branch: by default `loglik_volatility` on the `late` group.

## Two-Factor Interpretation

Across the continuous analyses, the intended interpretation is:

- `question_complexity_score` / `complexity_score` measures raw **semantic complexity (Csem)**
- `complexity_score_residual` measures **residualized compositional precision (Cres)** for scatter/diagnostic plots
- `prompt load` measures potential **linguistic anchoring**
- `option hardness` controls for MCQ discrimination difficulty
- `is_binary` controls for the binary yes/no vs 4-way MCQ answer-format confound when both formats appear in the regression slice
- `prediction entropy` controls for clean-answer uncertainty in Exp 1

Primary marginal summaries, wordy/pooled horse-race regressions, and fixed-effects diagnostics are therefore the main evidence for whether semantic complexity is the real driver of frequency-based fragility after controlling for prompt length, answer-set difficulty, and clean-model uncertainty where available. The `L5-L8` wordy controls test the same idea experimentally by increasing prompt load while holding the semantic program fixed.

Most regression and grouped-correlation payloads now include three views:

- `primary`: only the semantic ladder `L1-L4`; emits marginal Pearson/Spearman, Kendall monotonicity, and VIF diagnostics instead of multivariate betas
- `wordy`: only the matched wordy mirrors `L5-L8`
- `pooled`: all levels `L1-L8`

Legacy unsuffixed keys remain available for older plot loaders. Monotonicity and Spearman granularity tests are not pooled, because combining terse and wordy ladders would mix two different interventions.

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
# Full default run
nohup python3 -m frequency_alignment.run_experiment \
  --experiment all \
  --config frequency_alignment/configs/default.yaml \
  -v > frequency_alignment_server.log 2>&1 &

# Theory-refresh rerun (2026-04-24 onward): severities 1..3, suppress_dc=false,
# include_frequency=true, exp6 silent
nohup python3 -m frequency_alignment.run_experiment \
  --experiment all \
  --config frequency_alignment/configs/server_rerun.yaml \
  --max-samples 500 \
  -v > frequency_alignment_server_rerun.log 2>&1 &
```

## Dataset Expectations

- `gqa`
  - Main dataset for experiments `1-5`.
  - The code generates same-image `L1-L4` primary tasks plus `L5-L8` wordy controls from scene graphs.
  - Wordy controls are regenerated with cache version `gqa_multilevel_v6` and use neutral filler text that increases prompt load without changing the structured semantic program.
  - Wordy controls are only required to be longer than the matched base prompt by at least the configured filler margin; they are not forced to a fixed 60-word length.
  - Verification levels are balanced to 50% yes and 50% no when possible.
  - The loader uses an in-memory and on-disk granularity cache under `.hf_cache/gqa/granularity_cache/`, with early stopping once enough balanced valid samples are found.
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
  exp1/perturbation_examples/<image_id>/*.png
  exp2/summary.json
  exp2/hypothesis_tests.json
  exp2/power_spectra.json
  exp2/complexity_points.json
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
  exp6/perturbation_examples/<image_id>/*.png
  plots/*.png
  plots/plot_manifest.json
  plots/primary_l1_l4/*.png
  plots/primary_l1_l4/plot_manifest.json
```

Representative plot families include:

- `exp1_coefficient_plot_*`: overall coefficient/marginal-summary plots.
- `exp1_coefficient_plot_*_by_perturbation.png`: perturbation-specific coefficient/marginal-summary plots.
- `exp1_linguistic_stabilization_effect.png`: paired mirror plot showing `base - wordy` drift/drop deltas for `L1/L5`, `L2/L6`, `L3/L7`, and `L4/L8`.
- `*_csem.png`: raw semantic-complexity copies of the corresponding Cres complexity scatter plots.
- `plots/primary_l1_l4/`: duplicate plot suite restricted to the primary `L1-L4` ladder.
- `exp*_wordy_control_*.png`: matched `L1/L5`, `L2/L6`, `L3/L7`, `L4/L8` comparisons.
- `exp5_level_perturbation_predicted_vs_observed*.png`: Exp 5 predicted-vs-observed bars by level and perturbation.
- `exp5_bridge_scatter*.png`: comparable bridge plots using `zscore(log1p(S_pred_first_order))` and z-scored observed targets.

### Theory-refresh plots (2026-04-24)

These are emitted both to `plots/` and to `plots/primary_l1_l4/` and verify the
downward-concentration / prompt-length-prior / first-order-overlap predictions:

- `exp2_per_level_W_t.png`: 8-panel bar plot of `W_t(ω)` per level (L1–L8) with concentration and tail-mass annotations. Visual test of late-layer downward concentration.
- `exp2_shape_curves.png`: top-3 mass, top-2 mass, tail-mass fraction, and normalised centroid vs level, primary + wordy ladders overlaid. Theorem 2 prediction: late top-2 ↑, late tail ↓, and `G_late` ↓ as granularity increases; top-3 and centroid are informational.
- `exp2_wordy_divergence.png`: per-layer-group Δ(wordy − base) for all four mirror pairs. Theorem 4 prediction: the absolute wordy/base gap grows from `early` to `late`.
- `exp5_overlap_integral_heatmap_{image_space,vision_feature_space}.png`: heatmap of `⟨W_t, ΔF_p⟩` over (level × perturbation-family). Verifies the low-freq-perturbations-dominate prediction and the tail-asymmetry on high-freq perturbations.
- `exp5_first_order_overlap_vs_drift_{source}.png`: scatter of first-order overlap vs mean log-likelihood drift with Pearson r. Primary H1d gate under the revised theory.

### Theory-refresh gates (2026-04-24)

Added to the existing `hypothesis_tests.json` structure:

- `exp2/hypothesis_tests.json::shape_consolidation.spearman_top2_mass_late_vs_granularity` — ρ > 0 for late-layer downward concentration.
- `exp2/hypothesis_tests.json::shape_consolidation.spearman_tail_mass_late_vs_granularity` — ρ < 0 for late-layer tail thinning.
- `exp2/hypothesis_tests.json::shape_consolidation.spearman_top3_mass_vs_granularity` and `spearman_centroid_vs_granularity` — informational, not load-bearing gates.
- `exp2/hypothesis_tests.json::wordy_layerwise_divergence.{top_3_mass,top_2_mass,tail_mass_fraction,centroid_normalised}.passed` — mean |Δ(wordy−base)| grows from `early` to `late`.
- `exp1/hypothesis_tests.json::wordy_volatility_reduction.passed` — `Var(loglik_drift | wordy) < Var(loglik_drift | base)` averaged over mirror pairs.
- `exp5/summary.json::first_order_overlap_tables` — `{levels, perturbations, matrix_first_order, cells}` per source; direct input to the new overlap heatmap plots.

## Analysis Settings

- `analysis.num_bands_mode: auto` probes the model patch grid and freezes one run-level linear radial bin count. This keeps `W_t`, `delta_f`, `delta_f_vision`, Exp 3 drift spectra, and Exp 5 overlap vectors aligned.
- `analysis.suppress_dc` controls whether the DC band is included in the actual math. It is `false` in `default.yaml`, `local_test.yaml`, `llava_test.yaml`, `server_severity1.yaml`, and `server_rerun.yaml`; `true` should only be used as a diagnostic ablation.
- `analysis.attention_fft_window` / `experiments.exp2.fft_window` controls optional attention-map windowing before FFT. Current configs set this to `none`.
- `experiments.exp5.primary_layer_group` defaults to `late`; the code-level Exp 5 primary target is `loglik_volatility`.
- Complexity scatter plots use `complexity_score_residual` on the x-axis when available and also emit `_csem` copies against raw `complexity_score`.
- Coefficient plots now include triple-view variants when the result JSON contains the `_views` regression payloads.
- Exp 5 predicted-vs-observed scatter plots include triple-view panels for Primary, Wordy, and Pooled.
- Spectral plots can use log scaling for visualization only; this does not change stored spectra or hypothesis tests.
- Plot generation intentionally skips the old discrete `exp1_granularity_curves.png` so the regression and paired-control figures stay primary.
- `plotting.max_sample_scatter_points` can cap dense Exp 5 sample-level scatter rendering for visualization only; stored correlations still come from the full result data.
- `perturbations.export_num_images` controls how many clean + perturbed example image folders are saved per supported experiment.
- Exp 3 (fusion drift) was retired during the camera-ready cleanup; its analysis is subsumed by the Exp 5 overlap law on pixel-space ΔF.

## Troubleshooting

### Out of Memory

- Local machine: stay on `Qwen/Qwen3-VL-2B-Instruct`.
- Single A100: switch server profile from `30B-A3B` to `8B`.
- Increase `exp2.layer_stride` or reduce `max_samples`.

### Experiment 5 Missing Inputs

Exp 5 depends on Exp 1 + Exp 2. Run in dependency order:

```bash
python3 -m frequency_alignment.run_experiment \
  --experiment "1 2 5" \
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

# Frequency Alignment Run Guide

Last updated: July 12, 2026.

## Current Scientific Status

The results of this pipeline are written up in the paper **"Cross-Modal Attention
Acts as a Frequency Filter: Why Verbose Prompts Improve Robustness in
Vision–Language Models"** (`paper/main.tex`, ARR May 2026 / EMNLP 2026 review
format). The paper is the authoritative statement of the framing; this runbook
and `full_project_explanation.md` are kept in sync with it.

The core hypothesis stands: question-conditioned cross-modal attention induces a
task-specific spectral filter `W_t(ω)` over image patches, and behavioural drift
under perturbation scales with the overlap between `W_t` and the perturbation's
spectral signature `ΔF_p`. Two opposite-signed language forces shape the filter:
**semantic complexity narrows it** (destabilising), **prompt verbosity broadens
it** (stabilising).

### Evidence base (production runs)

- **Four 500-sample Qwen runs** — the canonical `2 × 2` matrix
  {Qwen3-VL-2B-Instruct, Qwen3-VL-8B-Instruct} × {GQA, CLEVR}, severities
  `[1, 2, 3]` pooled, 78 perturbation slices per image, `suppress_dc: false`,
  312,000 scoring rows and 500 image clusters per run (run dirs below).
- **Two LLaVA-OneVision GQA runs** (0.5B and 7B, 500 samples each). Only the
  **0.5B** run is used in the paper — it feeds both the cross-architecture
  appendix and the "LLaVA" row of the accuracy table (fingerprint: its L1–L4
  mean clean accuracy is exactly the table's 78.65). The **7B** outputs
  completed 2026-05-29 but were only copied into this repo on 2026-07-12 and
  are **not yet used in the paper** (its numbers: Base 83.10 clean / 81.93
  pert, Verbose 83.45 / 82.12 by the plain L1–L4 / L5–L8 mean).
- **One prompt-template control run** (100 GQA images, Qwen3-VL-2B, 5 templates)
  for the paper's prompt-template robustness appendix.

### The five separable predictions (paper framing)

Each prediction is tested on a different observable, so they do not share noise:

1. **Tug-of-war regression (behaviour only).** Within-image fixed-effects
   regression on z-scored predictors, identified via Frisch–Waugh–Lovell:
   `β̃_Cres > 0` and `β̃_Cprompt < 0` on log-likelihood volatility in **all four
   runs at p ≤ 10⁻⁵** (R² 0.18–0.76). The CR1 long-format version (312,000 rows,
   500 clusters) gives `β_Csem > 0` at `p ≤ 10⁻⁶` and `β_Cprompt < 0` at
   `p ≤ 10⁻⁶⁴` in all runs. The signed-drift variant passes 3/4 (sign-reverses on
   CLEVR-8B under recovery dominance — the anomaly that motivates volatility as
   primary outcome).
2. **Drift-variance reduction under verbose paraphrase.** Per-mirror-pair ratio
   `σ²_verbose / σ²_terse` < 1 in **14/16 cells**; mean ratio per run 0.19–0.76;
   on the 8B models the verbose paraphrase cuts drift variance by **70–81%**
   (mean 0.19 CLEVR-8B, 0.30 GQA-8B).
3. **Four-stage layer trajectory (attention geometry, no behaviour).**
   - Layer 0: **length prior** — verbose mirrors start `G(t)` 0.11–0.74 nats
     *narrower* than terse (4/4 runs).
   - Early layers: **prior erasure** — the gap collapses.
   - Mid layers: **Csem-driven narrowing** — terse `G(t)` for fine tasks drops
     while verbose mirrors stay flatter.
   - Late stack + `last_2`: **verbosity widening** — verbose filters end
     *broader* than terse (the stabilisation mechanism); `last_2` widening holds
     in 3/4 runs, CLEVR-8B widens at its peak layer ≈ 16 instead.
   - Peak verbose-broader depth is capacity-shifted: **2B at `last_2`, 8B at
     layer ≈ 16**.
4. **Parameter-free overlap law.** First-order overlap
   `⟨W_t, ΔF_p⟩ = Σ_ω W_t(ω)·ΔF_p(ω)` at `last_2` predicts volatility with
   grouped Pearson **r = 0.26–0.42**, matched fixed-effects r = 0.14–0.38, and
   within-cell median r = 0.16–0.46 (positive rate 62–72%); sign positive in all
   12 (run × aggregation-scale) cells, all p ≈ 0. Positive in 20/20
   (run × layer-group) cells with `last_2` strongest in every run.
   LLaVA-OneVision-0.5B replicates with grouped r = +0.343. Magnitude is
   moderate; the paper presents this as one of five tests, not the sole
   load-bearing claim.
5. **Cutoff / band-mass reframing.** ~76% of perturbation rows are low-frequency
   `ΔF` (16% mid, 4% high, 4% broadband); `β_Csem > 0` on volatility in 12/12
   (run, family) cells (10 at p ≤ 10⁻⁵); `W_t` carries only **3–5% of its mass
   in the highest radial band** in every (run, level), which mechanically
   explains why high-frequency perturbations cause the smallest drift in 3/4
   runs. Signature-matrix corollary: narrow-`W_t` rows out-drift broadband rows
   in 9/16 cells (cleanly 4/4 on CLEVR-2B and GQA-8B; CLEVR-8B reverses under
   recovery dominance; GQA-2B ties within ±0.11).

### Companion observations

- **Verbosity sharpens discrimination on clean images** (not just calibration):
  median +11 nats on gold log-p (16/16 cells), gold-vs-best-distractor margin
  widens ≈ +5 nats (15/16), argmax flip-rate drops in 12/16 cells (mean relative
  reduction 24%), accuracy under perturbation +1.5pp mean (up to +5.8pp on
  L4/L8).
- **Verbose prompts raise plain accuracy** on clean and perturbed images across
  Qwen-2B/8B and LLaVA-OV-0.5B on GQA and CLEVR (paper Table
  `verbose_results_acc`; per-level accuracies collected via
  `scripts/collect_completed_accuracies.py`. The table's "LLaVA" row is the
  **0.5B** run — the 7B run is not in the paper).
- **Prompt-template control** (paper App. `ablation_prompt`): the stabilisation
  is a property of non-semantic linguistic framing, not word count. On 100 GQA
  images with Qwen3-VL-2B: terse volatility 1.067 / clean acc 0.718; current
  wordy mirror 0.542 / 0.828; structured instruction 0.563 / 0.808; random
  neutral wrapper 0.800 / 0.808; a length-matched *meaningful* wrapper (adds
  visual-analysis semantics) **hurts**: 0.762 / 0.670.

### Conventions that remain canonical

- **DC ON (`suppress_dc: false`)** in all configs and all production runs. The
  2026-04-23 paired 200-sample ablation pinned this: Exp 5 first-order
  `r = 0.471, p ≈ 10⁻³⁶` DC ON vs `r = 0.055` DC OFF. `suppress_dc: true` is a
  diagnostic ablation only.
- **First-order overlap** is the primary predictor (`predicted_first_order`);
  the quadratic form is a stored secondary diagnostic (weaker on GQA,
  comparable on CLEVR).
- **`loglik_volatility` = E_p[|Δ|]** is the primary behavioural target; signed
  drift is reported as a complement.
- **`last_2`** (second-from-last decoder attention layer) is the primary layer
  group for Exp 5 (`experiments.exp5.primary_layer_group: last_2`); `overall`,
  `early`, `mid`, `late` are controls.

### Superseded framings (do not cite)

The 2026-04-27 "three-stage trajectory" story (early length prior → mid uniform
compression → late Csem-driven downward concentration) was based on the
200-sample dev runs and is superseded by the **four-stage trajectory** above,
measured on the four 500-sample production runs. Earlier retracted framings
("peak consolidation + tail growth", "wordy layerwise convergence", "early
over-steer corrected by depth") stay retracted. Historical gate names in older
output dirs (`wordy_layerwise_convergence`) were renamed
(`wordy_layerwise_divergence`) and remain in the code.

## Current Profiles

The shipped configs are:

- `frequency_alignment/configs/local_test.yaml`
  - Local smoke-test profile (RTX 5070 Ti Laptop, 12 GB VRAM).
  - `Qwen/Qwen3-VL-2B-Instruct`, GQA, tiny sample counts, severity 1,
    `include_frequency: false`, `suppress_dc: false`.
- `frequency_alignment/configs/clevr_qwen2b_test.yaml`
  - CLEVR smoke test: `Qwen/Qwen3-VL-2B-Instruct`, `dataset: clevr`,
    `max_samples: 2`, fixed `num_bands: 12`, `exp5.primary_layer_group: last_2`.
- `frequency_alignment/configs/server_rerun.yaml`
  - The server production template. Multi-severity (`[1, 2, 3]`),
    `include_frequency: true`, `suppress_dc: false`, `num_bands_mode: auto`,
    `exp6.enabled: false`.
  - Currently points at `llava-hf/llava-onevision-qwen2-7b-ov-hf` with
    `max_samples: 1000` (the last LLaVA-OV-7B run used `--max-samples 500`).
    For a Qwen production run, switch `model.primary` back to
    `Qwen/Qwen3-VL-{2B,8B}-Instruct` — the production `config_snapshot.json`
    files (below) are the authoritative record of what production used.

Removed profiles (`default.yaml`, `server_severity1.yaml`, `llava_test.yaml`)
no longer exist; every historical claim about them is superseded by the
snapshots inside the run directories.

## Production Run Directories

All under the repository root. Each contains `config_snapshot.json`
(authoritative config for that run — read it before claiming anything about the
run's settings), `exp1/`, `exp2/`, `exp4/`, `exp5/`, `combined/`, and `plots*/`.

| Run | Directory |
|-----|-----------|
| Qwen3-VL-2B / GQA | `qwen_gpa_2B/frequency_alignment_outputs_gqa_2B_500_20260517_032513/` |
| Qwen3-VL-8B / GQA | `qwen_gpa_8B/frequency_alignment_outputs_gqa_8B_500_20260516_125653/` |
| Qwen3-VL-2B / CLEVR | `qwen_clevr_2B/frequency_alignment_outputs_clevr_2B_500_20260516_132931/` |
| Qwen3-VL-8B / CLEVR | `qwen_clevr_8B/frequency_alignment_outputs_clevr_8B_500_20260516_131942/` |
| LLaVA-OV-0.5B / GQA | `llava_0.5_gqa/frequency_alignment_outputs_llava_0.5_gqa_500_20260521_213657/` |
| LLaVA-OV-7B / GQA | `llava_7b_gqa/frequency_alignment_outputs_llava_7b_gqa_500_20260521_210730/` |
| Prompt-template control | `prompt_template_analysis_qwen2b_gqa_100_20260525_225339/` |

Resolved radial band counts (`analysis.num_bands_resolution` in each snapshot):
**B = 7** on the Qwen GQA runs, **B = 9** on the Qwen CLEVR runs, **B = 20** on
both LLaVA-OneVision runs.

## Model Families

`frequency_alignment/models/__init__.py::get_adapter` routes by model id:

- `Qwen/Qwen3-VL-*` and `OpenGVLab/InternVL3-*-hf` → `HFVLMAdapter`
  (`models/qwen_adapter.py`). Qwen is the primary fully documented path; Qwen VL
  has no separate cross-attention, so the adapter slices the self-attention
  matrix between the `<|vision_start|>` / `<|vision_end|>` brackets. InternVL
  has not been run in production.
- `llava-hf/llava-onevision-qwen2-{0.5b,7b}-ov-hf` → `LLaVAAdapter`
  (`models/llava_adapter.py`). SigLIP vision encoder + Qwen2 LM, **no CLS
  token**. Only the OneVision variants (0.5B / 7B / 72B exist) are in scope; six
  adapter audit fixes were applied 2026-05-21 before the production runs.

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

`pycocotools` is only needed if your PartImageNet annotations are stored as COCO
RLE instead of polygons.

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

# Full server production run (edit model.primary in the config first)
python3 -m frequency_alignment.run_experiment \
  --experiment all \
  --config frequency_alignment/configs/server_rerun.yaml \
  --max-samples 500 \
  -v

# Plot-only regeneration
python3 -m frequency_alignment.run_experiment \
  --plot-only \
  --config frequency_alignment/configs/server_rerun.yaml \
  --out-dir <existing_output_dir>
```

`--max-samples N` is a global runtime override. It writes `N` into
`data.max_samples` and every configured `experiments.exp*.max_samples` for that
run, so all enabled experiments and downstream plots/tests use the processed
subset from that override.

## What Each Experiment Does Now

### Experiment 1

- Uses the configured VQA dataset (`gqa` or `clevr` for the main research path).
- Builds same-image primary levels `L1-L4` plus matched verbose mirrors `L5-L8`
  (internal level names: `L1_COARSE`, `L2_MEDIUM`, `L3_FINE`, `L4_VERY_FINE`,
  `L5_WORDY_SIMPLETON`, `L6_WORDY_MEDIUM`, `L7_WORDY_FINE`,
  `L8_WORDY_VERY_FINE`):
  - `L5`: L1 semantics with inflated prompt load, etc.
  - The verbose template wraps the terse question byte-for-byte: *"Please read
    the following question carefully and answer the same question directly.
    [terse question] The extra wording is only polite framing and should not
    change what you are being asked to answer."*
- Primary monotonic granularity tests remain on `L1-L4`; descriptive and control
  plots include `L5-L8`.
- Stores per task: continuous `question_complexity_score` (Csem, from the
  structured semantic program + grounding-ambiguity term),
  `prompt_complexity_score` (Cprompt, question-only content-word count; MCQ
  option strings excluded), `complexity_score_residual` (Cres, Csem residualised
  on Cprompt), `option_hardness_score`, `is_binary`, `num_options`, and
  clean-image `prediction_entropy`.
- Scores MCQ options with the parent repository's assistant-continuation
  log-likelihood routine. **Option letters are hidden**: each option *value*
  ("yes", "white") is scored as a continuation of (image, question).
- Stores raw and relative image-space perturbation spectra: `delta_f`,
  `delta_f_relative`; vision-feature analogues when
  `exp1.extract_vision_tokens: true`.
- Stores `loglik_drift = clean_ll(gold) − pert_ll(gold)` (positive = erosion),
  plus `loglik_erosion`, `loglik_recovery`, `loglik_volatility` (`|drift|`; the
  per-cell mean over perturbations is the paper's `Vol`).
- Accuracy-drop horse races are filtered to clean-correct points only.
- `hypothesis_tests.json` includes primary marginal summaries, wordy/pooled
  multivariate horse-race regressions, per-perturbation summaries, paired
  mirror tests, and the long-format CR1 gates. The composite verdict now gates
  on `long_format_dual_force_loglik_volatility` (falling back to the
  `loglik_drift` variant only if the volatility gate is absent), plus
  `spearman_drop_vs_granularity`, `monotonicity_accuracy_drop`,
  `per_perturbation_csem_sign_stability`, and
  `delta_f_family_falsifiable_prediction`.

### Experiment 2

- Extracts the effective language-to-vision attention map from the
  self-attention slice returned by the model.
- Computes radial power spectra, `W_t`, and bandwidth `G(t)` (inverse
  participation ratio) for the `overall`, `early`, `mid`, `late` layer groups
  **plus the `last_2` depth probe** (second-from-last attention layer — the
  paper's primary extraction depth for the overlap law).
- Saves per-sample and average filters under `exp2/filters/` **and a parallel
  question-only extraction under `exp2/filters_question_only/`** (attention to
  question text without the option strings; used for the paper's option-
  conditioning control, mean |ΔG(t)| = 0.18, max 0.35 — no systematic shift).
- Applies the configured attention FFT window before the 2D FFT (`hann`,
  `hamming`, or `none`; all current configs use `none`).
- Runs prompt-only controls (`empty_language`, `random_language`) on the same
  images and stores divergence-to-task statistics.
- Stores the same complexity/control scores as Exp 1 per (image, level) task.
- `hypothesis_tests.json` includes the granularity trend on `G` per layer group,
  the `shape_consolidation` block, and the `wordy_layerwise_divergence` block
  (the wordy-vs-base shape gap grows with depth).

### Experiment 3 (retired)

Exp 3 (fusion drift / response amplification) was retired in the camera-ready
cleanup (2026-05-11). Its analysis is subsumed by the Exp 5 overlap law on
pixel-space ΔF plus the per-layer-group `W_t` shape statistics in Exp 2. The
vision-feature-space overlap is not load-bearing: it sign-flips
(r = −0.31 to −0.38), which the paper discloses in the overlap table caption.

### Experiment 4

- Runs the low-pass / high-pass sweep on the same multilevel samples.
- Reports the critical cutoff where accuracy crosses the configured threshold.
- Saves `complexity_points.json`; horse races as in Exp 1.
- Multi-severity (`[1,2,3]`) is required — under severity 1 alone the cutoff
  pins at Nyquist.

### Experiment 5

- Uses real per-sample `W_t` filters from Experiment 2 for `overall`, `early`,
  `mid`, `late`, and `last_2`; **`last_2` is the primary layer group**
  (`primary_group: last_2` in `exp5/summary.json`).
- Uses image-space (and optionally vision-feature) perturbation spectra from
  Experiment 1; the primary `delta_normalization` is `relative`, raw branches
  remain as controls.
- Computes three overlap variants per cell; **first-order
  `predicted_first_order = Σ_ω W_t(ω)·ΔF_p(ω)` is primary**
  (`primary_overlap_variant: first_order`), linear and quadratic are stored
  diagnostics.
- Primary behavioural target: `loglik_volatility`
  (`primary_target: loglik_volatility`).
- Reports grouped correlations (by image × level × perturbation-family),
  matched fixed-effects correlations (demeaned within
  image × perturbation × severity units), and within-cell correlations — the
  paper's three aggregation scales.
- Also saves the standardized "comparable bridge" view
  (`zscore(log1p(S_pred))` vs `zscore(actual)`), prediction-factor horse races
  (`prediction_factor_horse_race`), per-(level × perturbation-family)
  `first_order_overlap_tables`, and an `overlap_2d` block (orientation-
  preserving 2D pilot; production paper App. reports 1D r within 0.022 of 2D).

### Experiment 6

- Requires `PartImageNet` and `SAM2` / `SAM3`.
- Uses GT segmentation masks from PartImageNet; prompts `SAM3` with text and
  `SAM2` with GT boxes; evaluates clean and perturbed `mIoU` against GT masks.
- Disabled (`exp6.enabled: false`) in all production runs; not part of the
  paper's evidence base.

## Two-Factor Interpretation

Across the continuous analyses, the intended interpretation is:

- `question_complexity_score` / `complexity_score` measures raw **semantic
  complexity (Csem)** — destabilising (`β > 0` on volatility)
- `complexity_score_residual` measures **residualized semantic complexity
  (Cres)**, the FWL-identified variable used in the paper's fixed-effects table
- `prompt load` (Cprompt) measures **linguistic anchoring / verbosity** —
  stabilising (`β < 0` on volatility)
- `option hardness` controls for MCQ discrimination difficulty
- `is_binary` controls for the binary yes/no vs 4-way MCQ answer-format confound
- `prediction entropy` controls for clean-answer uncertainty in Exp 1

Most regression and grouped-correlation payloads include three views:

- `primary`: only the semantic ladder `L1-L4` (marginal stats + VIF diagnostics
  — the terse ladder is collinear in Csem and Cprompt by construction)
- `wordy`: only the matched verbose mirrors `L5-L8`
- `pooled`: all levels `L1-L8` (the paper's fixed-effects regression grain:
  500 images × 8 levels = 4,000 cells per run)

Monotonicity and Spearman granularity tests are not pooled, because combining
terse and wordy ladders would mix two different interventions.

## Paper-Figure and Diagnostic Scripts (`scripts/`)

The paper's figures and several appendix numbers are produced by standalone
scripts operating on the production run dirs (never by modifying the pipeline):

- `compose_paper_teaser_and_example.py`, `export_teaser_sample.py`,
  `build_interactive_teaser.py`, `build_teaser_candidate_browser.py` — teaser
  figure (worked-example image GQA 2410848).
- `compose_cross_run_figures.py` — the `n1_trajectory_grid`,
  `n3_overlap_law_bars`, `n4_band_mass` figure family.
- `plot_delta_wt_signature_matrix.py` — `n2_signature_matrices_grid`
  (writes `plots_delta_wt_signature_matrix/` into each run dir).
- `plot_perturbation_signature_volatility.py` — `n5_designed_vs_actual`
  (writes `plots_perturbation_signature_volatility/`).
- `diagnose_matched_overlap_fixed_effects.py` — matched-FE overlap correlations
  (writes `plots_overlap_matched_fe/`).
- `verify_drift_baseline_concerns.py` — the Concern-A/Concern-B baseline audit
  (`baseline_concerns_report.{json,md}` in each run dir; 99.25% / 0.25% numbers
  in paper §confounds).
- `refit_tugofwar_table.py` — the within-image FE regression table
  (`tab:betas-fe`).
- `extract_2d_vs_1d_overlap.py`, `extract_per_severity_overlap.py`,
  `extract_bandwidth_variability.py`, `extract_delta_f_image_variance.py` —
  appendix diagnostics.
- `analyze_prompt_templates_qwen2b_gqa.py` — **the prompt-template control**
  (produced `prompt_template_analysis_qwen2b_gqa_100_20260525_225339/` and the
  paper's template-comparison table; 5 templates × L1-L4 × 18 perturbation
  types on 100 GQA images). `analyze_gqa_prompt_templates_qwen2b.py` is an
  earlier variant kept for reference.
- `collect_completed_accuracies.py` — the clean/perturbed accuracy table
  (`tab:verbose_results_acc`).
- `sanity_check_llava.py` — LLaVA adapter smoke test.
- `run_exp5_streaming.py` — memory-lean Exp 5 re-run against cached Exp 1/2
  outputs.

## Dataset Expectations

- `gqa`
  - Main dataset for experiments `1-5`.
  - The code generates same-image `L1-L4` primary tasks plus `L5-L8` verbose
    mirrors from scene graphs (not the GQA question generator).
  - Current cache version: **`gqa_multilevel_v8`** (in-memory + on-disk
    granularity cache under `.hf_cache/gqa/granularity_cache/`).
  - v8 design guarantees: L1 alternates positive/negative presence questions by
    image index; L3 asks scene-graph relations with negative-direction
    variants (cardinal-opposite relations only, to avoid joint-truth label
    noise); **L4 reuses L2's attribute category, option set, and gold answer**
    so the L2→L4 contrast isolates the relation operator with `H_opt` constant.
  - Verification levels are balanced to 50% yes / 50% no when possible.
  - GQA images download on demand from Visual Genome.
- `clevr`
  - Same 8-level mirror construction on CLEVR scenes (`data/clevr.py`);
    synthetic imagery, median patch grid gives B = 9.
- `partimagenet`
  - Dataset for experiment `6` (hierarchical object / part / subpart).
- `seedbench`
  - Adapter exists but is L4-only; not part of the research path.

## Output Files

```text
<run_dir>/
  config_snapshot.json
  combined/all_results.json
  exp1/summary.json
  exp1/hypothesis_tests.json
  exp1/per_sample.jsonl
  exp1/complexity_points.json
  exp1/degradation_by_level.json
  exp1/perturbation_examples/<image_id>/*.png
  exp2/summary.json
  exp2/hypothesis_tests.json
  exp2/power_spectra.json
  exp2/complexity_points.json
  exp2/filters/                      # per-sample + average W_t (.npy), incl. last_2
  exp2/filters_question_only/        # question-only parallel extraction
  exp4/summary.json
  exp4/accuracy_curves.json
  exp4/complexity_points.json
  exp5/summary.json                  # incl. first_order_overlap_tables, overlap_2d
  exp5/hypothesis_tests.json
  exp5/scatter_data*.json, exp5/sample_scatter_data*.json (by group / target)
  exp6/summary.json                  # only when exp6 enabled
  plots/*.png
  plots/plot_manifest.json
  plots/primary_l1_l4/*.png
  # written by the standalone diagnostic scripts:
  plots_delta_wt_signature_matrix/
  plots_overlap_matched_fe/
  plots_perturbation_signature_volatility/
  baseline_concerns_report.json / .md
```

Note: the shipped production run dirs contain a *partial* pipeline-plot suite
(exp1/exp2/exp4 families; no `exp5_*.png` and no `primary_l1_l4/` subdir in
`plots/`). All families below are still generated by
`frequency_alignment/plotting/plots.py` and can be regenerated with
`--plot-only`; the paper's exp5 figures were produced by the standalone
scripts listed above instead.

Representative plot families include:

- `exp1_coefficient_plot_*` (+ `_views`, `_by_perturbation`): coefficient /
  marginal-summary plots per outcome.
- `exp1_linguistic_stabilization_effect.png`: paired mirror plot of
  `base − wordy` deltas for the four mirror pairs.
- `exp1_long_format_coefficient_forest_*`, `exp1_design_effect_*`,
  `exp1_per_perturbation_csem_*`, `exp1_delta_f_family_csem_*`: the long-format
  CR1 gate family.
- `*_csem.png`: raw semantic-complexity copies of the corresponding Cres
  complexity scatter plots.
- `exp2_per_level_W_t.png`, `exp2_shape_curves.png`,
  `exp2_wordy_divergence.png`: filter-shape diagnostics.
- `exp2_band_heatmap_{early,mid,late,last}.png`, `exp2_bandwidth*.png`,
  `exp2_group_*.png`, `exp2_mean_gt_curves.png`, `exp2_gt_vs_layer.png`.
- `exp5_overlap_integral_heatmap_*.png`,
  `exp5_first_order_overlap_vs_drift_*.png`, `exp5_bridge_scatter*.png`,
  `exp5_level_perturbation_predicted_vs_observed*.png`,
  `exp5_coefficient_plot_prediction_factors*.png`.
- `plots/primary_l1_l4/`: duplicate plot suite restricted to the primary
  `L1-L4` ladder.

## Analysis Settings

- `analysis.num_bands_mode: auto` probes the model patch grid and freezes one
  run-level radial bin count: `B = ⌈0.6·√N⌉` (N = vision tokens on the median
  probed grid), subject to a radial-annulus cap that drops empty bins and the
  configured `[min_bands, max_bands]` clamp. Resolution details are stored in
  `analysis.num_bands_resolution` of every output dir. Production: **B = 7
  (Qwen/GQA), B = 9 (Qwen/CLEVR), B = 20 (LLaVA-OV/GQA)**.
- `analysis.suppress_dc: false` (DC ON) in every shipped config
  (`local_test.yaml`, `clevr_qwen2b_test.yaml`, `server_rerun.yaml`) and every
  production run; `true` is a diagnostic ablation only.
- `analysis.attention_fft_window` / `experiments.exp2.fft_window`: `none` in
  all current configs.
- `experiments.exp5.primary_layer_group`: **`last_2`**; code-level primary
  target is `loglik_volatility`, primary overlap variant `first_order`, primary
  ΔF normalization `relative`.
- Complexity scatter plots use `complexity_score_residual` on the x-axis and
  emit `_csem` copies against raw `complexity_score`.
- Coefficient plots include triple-view variants when the result JSON contains
  the `_views` regression payloads.
- `plotting.max_sample_scatter_points` caps dense Exp 5 scatter rendering
  (visualization only).
- `perturbations.export_num_images` controls how many clean + perturbed example
  image folders are saved per supported experiment.

## Perturbation Bank (production)

21 operator types in 4 categories, each at severities `{1, 2, 3}` with
signed-direction variants where applicable → **78 slices per image**:

- **Geometric (6):** `Translation` (cyclic horizontal), `PadCrop`, `Scale`,
  `ScalePadBlack`, `ScalePadWhite`, `Rotation`.
- **Photometric (6):** `Brightness`, `Contrast`, `GaussianNoise`, `JPEG`,
  `GaussianBlur`, `MotionBlur`.
- **Occlusion (4):** `Occlusion`, `TextOverlay`, `BoxOverlay`, `RandomText`.
- **Spectral (5):** `LowPassKeep`, `HighPassKeep`, `LowBandNoise`,
  `HighBandNoise`, `AllBandNoise` (cutoffs ω_c ∈ {0.18, 0.28, 0.38}).

Known naming gotcha (paper App. `naming`): `LowBandNoise` / `HighBandNoise`
carry names inverted relative to their measured spectral effect (an
`fftshift` convention mismatch between `make_frequency_mask` and the shifted
data it is applied to). The pure-keep operators behave as named. All analyses
condition on the empirically measured `ΔF` signature, so no result is affected.

## Server Run Notes

The server profile relies on model sharding when needed:

```yaml
model:
  primary: Qwen/Qwen3-VL-8B-Instruct   # or llava-hf/llava-onevision-qwen2-7b-ov-hf
  device_map: null                      # auto for multi-GPU sharding
```

Suggested command:

```bash
nohup python3 -m frequency_alignment.run_experiment \
  --experiment all \
  --config frequency_alignment/configs/server_rerun.yaml \
  --max-samples 500 \
  -v > frequency_alignment_server_rerun.log 2>&1 &
```

## Troubleshooting

### Out of Memory

- Local machine: stay on `Qwen/Qwen3-VL-2B-Instruct`.
- Single A100: use the 8B model rather than 30B-A3B.
- Increase `exp2.layer_stride` or reduce `max_samples`.
- For Exp 5 memory pressure on large runs, use `scripts/run_exp5_streaming.py`.

### Experiment 5 Missing Inputs

Exp 5 depends on Exp 1 + Exp 2. Run in dependency order:

```bash
python3 -m frequency_alignment.run_experiment \
  --experiment "1 2 5" \
  --config frequency_alignment/configs/local_test.yaml \
  -v
```

### Experiment 6 No Samples

Make sure PartImageNet exists in one of the searched locations or point the
config at a local copy under `.hf_cache/partimagenet` / `~/datasets/PartImageNet`
/ `/data/PartImageNet`.

### Offline Runs

```bash
python3 -m frequency_alignment.run_experiment \
  --offline \
  --config frequency_alignment/configs/server_rerun.yaml
```

This forwards cache-only loading to the parent repository model loader.

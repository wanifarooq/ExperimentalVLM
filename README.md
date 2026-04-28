# VLM Robustness

Harness for testing visual-language model robustness to common image perturbations. The main entrypoint `vlm_invariance_check.py` scores multiple-choice options by log-likelihood (or free-text modes), applies perturbations (shift/scale/crop/pad/rotation/text overlay), and can analyze embedding drift with PCA/t-SNE visuals.

## Frequency Alignment Project

The active research pipeline lives in `frequency_alignment/`. It tests the hypothesis that language conditioning acts as a task-specific spectral filter `W_t`, and that robustness depends on the overlap between this filter and perturbation energy. The current implementation uses GQA same-image primary levels `L1-L4`, matched wordy mirrors `L5-L8`, question-only prompt-load controls, prediction-entropy and option-hardness controls, paired mirror tests, directional log-likelihood drift metrics, Exp 3 all-token post-fusion drift, and Exp 5 spectral-overlap prediction with correct-answer log-likelihood volatility as the primary target.

Current frequency-alignment details:
- `L5-L8` are longer-than-base wordy mirrors of `L1-L4`; they are not padded to a fixed 60-word target.
- Primary (`L1-L4`) view reports marginal Pearson/Spearman, Kendall monotonicity, and VIF diagnostics instead of multivariate betas because raw semantic complexity and prompt load are highly collinear on the terse ladder.
- Wordy (`L5-L8`) and Pooled (`L1-L8`) views keep full multivariate horse-race regressions using raw `question_complexity_score`, `prompt_complexity_score`, `option_hardness_score`, and an `is_binary` task-format control when the slice mixes yes/no and 4-way MCQ items, plus prediction entropy where available.
- `complexity_score_residual` is still computed for residualized scatter and dual-force diagnostic plots, and matching `_csem` plot copies are emitted against raw semantic complexity.
- Exp 2 computes `W_t` for `overall`, `early`, `mid`, and `late` layer groups, plus prompt-only controls.
- Exp 5 treats the `late` layer group, relative-normalized image-space spectra, and first-order overlap `S_pred = sum W_t * delta_F` as the main branch, while raw spectra, vision-feature spectra, other layer groups, and quadratic/linear overlap variants remain controls.
- Exp 5 also emits prediction-factor summaries for observed accuracy drop: `zscore(log1p(S_pred_first_order))`, raw semantic complexity, prompt load, and option hardness, with the same primary-marginal vs wordy/pooled-regression split.

For the scientific design, math, experiments, gates, and plot catalogue, see `full_project_explanation.md` at the repo root.

For commands, configs, outputs, and troubleshooting, see `frequency_alignment/experiment_run.md`.

For the paper, slides, and a structured literature review, see `paper/main.tex`, `paper/slides.tex`, and `paper/LITERATURE.md`.

## Features
- Compare predictions across perturbations with MCQ log-likelihood scoring or free-text representations (`label`, `text`, `text_mcq`, `semantic` placeholder).
- Perturbations: translation, pad/crop, scale, scale+pad (black/white), text overlays, rotation.
- Embedding analyses: context/answer similarity, visualization, and drift (perturbation vs. control).
- Multi-device/multi-worker execution with automatic model and dataset caching.

## Requirements
- Python 3.9+ and PyTorch (GPU recommended; CPU/MPS work). The script bootstraps common deps (`torch`, `transformers>=4.57.0`, `Pillow`, `pandas`, `huggingface_hub`, `requests`, `matplotlib`, `scikit-learn`) if missing.
- Cache root defaults to `.hf_cache/` for models, datasets, and HF hub downloads. Override with `--cache-dir`. Use `--offline` to force cache-only reads.

## Quick start
```bash
# Minimal run on a few samples, single device
python vlm_invariance_check.py --device auto

# Scale up on GPU(s) with parallel workers
python vlm_invariance_check.py \
  --device cuda \
  --num-workers 2 \
  --max-samples 2000 \
  --model-id <hf-model-id>
```

Key flags (model/dataset agnostic):
- `--model-id` / `--model-path`: Hugging Face id or local snapshot directory.
- `--compare-mode {label,text,text_mcq,semantic}`: representation for invariance checks.
- `--translate-steps`, `--padcrop-steps`, `--scale-factor`: perturbation strengths.
- `--invariance-mode {label,embedding,both}` and `--drift-only`: choose analyses to run.
- `--save-changed-dir ''`: disable saving prediction-flip image pairs.
- `--summary-file`: path for per-worker summaries (suffix `_workerN` is added automatically).

## Models
- Any `AutoModelForImageTextToText` + `AutoProcessor` pair should work. Swap `--model-id` or point `--model-path` to a local checkout; models are cached under `<cache-dir>/models/<safe_name>`.
- For offline use, pre-populate the cache/model path and pass `--offline`.

## Datasets
- The script ships with a SEEDBench adapter (`--dataset seedbench` plus `--seedbench-tsv` and `--image-root`). To add your own dataset, implement a loader in `vlm_invariance_check.py` (see `load_seedbench_samples`) and extend `load_samples_for_dataset` to dispatch on `--dataset <name>`.
- Dataset loaders should yield `Sample` objects with `image_path`, `question`, an option dict (`{"A": "...", "B": "...", ...}`), and optional ground-truth label/category/hint. Free-text modes work without options, but MCQ metrics require them.
- When adding a new adapter, add CLI arguments for its paths/preprocessing; reuse the existing caching helpers if applicable.

## Outputs
- `summary*.txt`: invariance metrics per worker/run (AVg flip rate, Ve per-image impact, confusion vs. GT). A combined `summary.txt` appears when `--summary-file summary.txt` is used.
- `changed_predictions/`: original vs. perturbed image pairs where the predicted label changed.
- `embedding_viz/worker_*/*.png`: PCA/t-SNE plots for context/answer embeddings.
- `embedding_drift/`: drift histograms and `drift_summary.txt` comparing perturbation drift to control drift.
- `logs/`: raw stdout/stderr from multi-worker runs.
- `ploter.py`: quick script to aggregate `summary_worker*.txt` into AVg/Ve bar plots.

## Reference run (artifacts in repo)
Current checked-in artifacts come from a SEEDBench + Qwen3-VL-2B run (~14k samples across 4 workers). Headline numbers (mean across workers):
- Base accuracy: ~0.611
- Flip rates (AVg / Ve): Translation 0.062 / 0.163; Pad/Crop 0.066 / 0.170; Scale 0.074 / 0.074; Scale+Pad 0.077 / 0.099; Rotation 0.119 / 0.165; TextOverlay 0.192 / 0.240; Any perturbation 0.086 / 0.377.
- Embedding drift (`embedding_drift/drift_summary.txt`): text overlays show the largest embedding shift; other perturbations show comparatively small cosine/L2 drift versus control pairs.

Use these outputs as references or replace them with your own model/dataset runs.

## Tips
- Reuse existing data by pointing to your local TSV/image root; otherwise the SEEDBench adapter will auto-download.
- If you only need embedding drift (no invariance), run with `--drift-only`.
- Large runs benefit from `--num-workers` across available devices; set it to `0` to stay single-process even if multiple GPUs are visible.

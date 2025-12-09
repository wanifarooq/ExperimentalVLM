# Repository Guidelines

## Project Structure & Module Organization
- `vlm_invariance_check.py`: entry point for perturbation sweeps, metric computation, caching, and dataset/model downloads (defaults to `.hf_cache/`, `models/`, and `seedbench/`).
- `backup.py`: legacy variant of the driver; keep changes minimal unless intentionally diverging.
- `ploter.py`: turns `summary_worker*.txt` files into AVg/Ve bar charts; expects the worker summaries in the repo root.
- Run artifacts land in `changed_predictions/`, `embedding_drift/`, `embedding_viz/`, `logs/`, `summary*.txt`; avoid committing new large artifacts.

## Build, Test, and Development Commands
- Python 3.9+, venv recommended (`python -m venv .venv && source .venv/bin/activate`). The driver auto-installs `torch`, `transformers>=4.57.0`, Pillow, pandas, matplotlib, scikit-learn, huggingface_hub, and requests.
- Smoke/local run (single process): `python vlm_invariance_check.py --device auto --num-workers 0 --dataset seedbench --seedbench-tsv /path/to/SEEDBench_IMG.tsv --image-root /path/to/SEEDBench_IMG --max-samples 20 --compare-mode label --save-changed-dir ''`.
- Drift-only mode: add `--drift-only` to skip invariance sweeps; use `--cache-dir <dir>` to control cache placement; `--offline` forces cache-only reads.
- Metrics plot: `python ploter.py` after you have `summary_worker*.txt` outputs in the root.

## Coding Style & Naming Conventions
- Python, 4-space indentation, PEP 8-ish formatting; prefer type hints and dataclasses where already used. Keep CLI flags kebab-case (e.g., `--seedbench-tsv`) and internal variables snake_case.
- Use `pathlib.Path` for filesystem operations and keep side effects behind clear helper functions. Match existing naming for metrics (`AVg`, `Ve`) and outputs (`summary_workerN.txt`, `embedding_*`).

## Testing Guidelines
- No formal test suite; rely on smoke runs with small `--max-samples` and `--num-workers 0` to validate new logic. Confirm `summary*.txt` appears, logs show no tracebacks, and optional `embedding_drift/drift_summary.txt` updates.
- For visualization tweaks, run `python ploter.py` to ensure plots render. Avoid generating or committing large image sets; use `--save-changed-dir ''` when a run does not need flip visualizations.

## Commit & Pull Request Guidelines
- Git history uses short, descriptive subjects (“updated readme”, “added the results”); keep commits concise and imperative.
- PRs should state the goal, key flags/datasets/models used for validation, the smoke-run command, and notable metrics/output paths. Link related issues, include plot screenshots when relevant, and update README/AGENTS when adding flags or workflows.

## Security & Configuration Tips
- Do not commit dataset images, cached models, or credentials; rely on local `.hf_cache/` and `--offline` when using existing assets. Keep Hugging Face tokens out of commits and logs.
- Multi-worker GPU runs can be memory-heavy; tune `--num-workers` to available devices and prefer a short `--max-samples` when iterating. The script already sets conservative OpenMP env vars; avoid overriding unless necessary.

## Response labeling
- Use the following tags to signal confidence or status:
  - `[uncertain]`: Unverified claim; briefly state why confidence is low.
  - `[todo]`: Required next steps or follow-ups that can’t be completed now.
  - `[blocked]`: Work cannot proceed due to missing info, approval, or an external dependency.
  - `[assumption]`: Explicitly note an assumption being made.
  - `[risk]`: Call out a potential failure mode or edge case not yet resolved.
  - `[source-needed]`: Authoritative reference is appropriate but not yet provided.

## Remove AI code slop
Check the diff against main, and remove all AI generated slop introduced in this branch.

This includes:
- Extra comments that a human wouldn't add or is inconsistent with the rest of the file;
- Extra defensive checks or try/catch blocks that are abnormal for that area of the codebase (especially if called by trusted / validated codepaths);
- Casts to any to get around type issues;
- Variables that are only used a single time right after declaration, prefer inlining the rhs;
- Any other style that is inconsistent with the file.

Report at the end with only a 1–3 sentence summary of what you changed.

## Dependency versions
If the version of a dependency is not specified, always use the latest version.

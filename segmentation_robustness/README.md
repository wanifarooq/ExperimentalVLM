# Segmentation Robustness (SAM2 vs SAM3)

End-to-end robustness harness for comparing SAM2 (vision-only, PVS) and SAM3 (vision-language, PCS/PVS) under matched natural and frequency-targeted perturbations. The driver applies identical geometric transforms to images, masks, and visual prompts, then evaluates mask quality, grounding, and video stability metrics.

Current project status: this directory remains the standalone segmentation robustness harness. The active frequency-alignment project integrates segmentation as optional Exp 6 under `frequency_alignment/`, where PartImageNet GT masks are used for clean and perturbed mIoU and SAM2 acts as the vision-only control. See `../frequency_alignment/experiment_run.md` and `../frequency_alignment/TECHNICAL_DOCUMENTATION.txt` for the current Exp 6 integration details.

## What this runs
- SAM2 (PVS): visual prompts (boxes/points) -> masks.
- SAM3 (PCS/PVS): text prompts (and optional exemplars) -> instance masks.
- If SAM3 weights are unavailable, fallback is GroundingDINO (text->box) + SAM2 (box->mask).

## Datasets
The harness can run end-to-end from a YAML config with a dataset preset, or accept a JSONL manifest you provide.

Presets:
- `mini_coco` (default): auto-downloads COCO val2017 + annotations, builds a 200-sample manifest with instance masks, boxes, and centroid points, and injects negative text prompts.
- `davis17` (optional): auto-downloads DAVIS-2017 val, builds video manifests with frame lists and per-frame masks/boxes/points.

### Manifest format (JSONL)
Each line is a sample. Single images are represented as a single-frame list.

```
{
  "id": "sample-001",
  "split": "val",
  "frames": [
    {
      "image": "images/0001.jpg",
      "mask": "masks/0001.png",
      "instance_id": 1,
      "boxes": [[120.0, 45.0, 410.0, 360.0]],
      "points": [[210.0, 140.0, 1]]
    }
  ],
  "text_prompt": "a bicycle",
  "concept": "bicycle",
  "concept_present": true
}
```

Notes:
- `mask` can be a binary mask or an instance-id mask; use `instance_id` to select the target instance when a mask contains multiple IDs.
- `boxes`/`points` are in pixel coordinates; points are `[x, y, label]` with label 1 (positive) or 0 (negative).
- `concept_present` can be `false` for negative examples (used for grounding error rates).

## Usage
```bash
python segmentation_robustness/segmentation_invariance_check.py \
  --config segmentation_robustness/configs/default.yaml
```

Key flags:
- `--config`: YAML config; CLI overrides YAML.
- `--manifest`: dataset JSONL manifest (optional; overrides preset).
- `--dataset-preset`: `mini_coco` (default) or `davis17`.
- `--cache-dir`: cache directory for datasets and checkpoints.
- `--sam2-config`: SAM2 config name within the `sam2` package (default `configs/sam2.1/sam2.1_hiera_b+.yaml`).
- `--sam2-checkpoint`: SAM2 checkpoint path (auto-downloaded if missing).
- `--sam3-backend`: `auto` (default), `sam3`, or `groundingdino_sam2`.
- `--sam3-module`, `--sam3-class`, `--sam3-checkpoint`: required for custom SAM3 adapter.
- `--save-visuals`: save overlay panels (GT vs SAM2 vs SAM3) for a subset of samples.
- `--visual-limit`: maximum number of panels to save.
- `--include-natural` / `--include-frequency`: toggle perturbation families.
- `--severity-levels`: severity levels (default `1 2 3`) matched across natural/frequency families.
- `--out-dir`: output directory for summaries and visualizations.

SAM3 is attempted first; if unavailable, the pipeline automatically falls back to GroundingDINO + SAM2.
SAM2 and GroundingDINO assets auto-download into the cache when enabled.
SAM3 checkpoints are gated on Hugging Face; request access at https://huggingface.co/facebook/sam3 and authenticate (`hf auth login`) to enable SAM3.

## Outputs
- `summary.json`: metrics by model, perturbation, and severity.
- `summary.txt`: human-readable table format.
- `plots/`: aggregate plots (metric vs severity for natural/frequency groups).
- `visuals/`: sample overlays showing GT vs SAM2 vs SAM3 predictions.

## Metrics
Mask quality:
- mIoU
- Boundary F-score
- Fragmentation (connected components)

SAM3 grounding (text prompts):
- Concept recall
- False positives
- Wrong-instance rate
- Presence errors

Video:
- J (IoU), F (boundary) per frame
- ID stability (IoU between consecutive predicted masks)

## Dependencies
Install as needed (latest versions): `torch`, `numpy`, `Pillow`, `matplotlib`, `PyYAML`, `requests`, `huggingface_hub`, `pycocotools`.
Optional model backends:
- `sam2` package for SAM2 predictors.
- `groundingdino` for the SAM3 fallback pipeline.

Custom SAM3 adapter:
- The class referenced by `--sam3-module` and `--sam3-class` should implement `predict(image, text_prompt)` and return `(masks, scores)` or `{\"masks\": ..., \"scores\": ...}`.

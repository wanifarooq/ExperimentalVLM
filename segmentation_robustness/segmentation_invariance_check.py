#!/usr/bin/env python3
"""
Segmentation robustness harness for comparing SAM2 and SAM3 under matched
natural and frequency-targeted perturbations.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

def ensure_package(pkg: str, version_spec: Optional[str] = None) -> None:
    try:
        import importlib.metadata as importlib_metadata

        ver = importlib_metadata.version(pkg)
        if version_spec is None:
            return
        if version_spec.startswith(">="):
            needed = tuple(int(x) for x in version_spec[2:].split("."))
            have = tuple(int(x) for x in ver.split(".")[: len(needed)])
            if have >= needed:
                return
    except Exception:
        pass

    spec = pkg if version_spec is None else f"{pkg}{version_spec}"
    print(f"[setup] Installing/upgrading {spec}...", file=sys.stderr)
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    base_cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if not in_venv:
        base_cmd.append("--user")
    try:
        subprocess.check_call(base_cmd + [spec])
    except subprocess.CalledProcessError:
        env = dict(**{**os.environ, "PIP_BREAK_SYSTEM_PACKAGES": "1"})
        fallback_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--break-system-packages",
            spec,
        ]
        subprocess.check_call(fallback_cmd, env=env)


ensure_package("torch")
ensure_package("numpy")
ensure_package("Pillow")
ensure_package("matplotlib")
ensure_package("requests")
ensure_package("huggingface_hub")
ensure_package("pycocotools")
ensure_package("PyYAML")
ensure_package("scipy")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import matplotlib  # noqa: E402
import requests  # noqa: E402
import yaml  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402
from huggingface_hub.utils import HfHubHTTPError  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class FrameRecord:
    image_path: Path
    mask_path: Optional[Path]
    instance_id: Optional[int]
    boxes: List[Tuple[float, float, float, float]]
    points: List[Tuple[float, float, int]]
    gt_boxes_all: List[Tuple[float, float, float, float]]
    gt_mask_paths_all: List[Path]


@dataclass
class SegSample:
    sample_id: str
    frames: List[FrameRecord]
    text_prompt: Optional[str]
    concept: Optional[str]
    concept_present: Optional[bool]
    split: Optional[str] = None


@dataclass
class LoadedFrame:
    image: Image.Image
    mask: Optional[np.ndarray]
    instance_id: Optional[int]
    boxes: List[Tuple[float, float, float, float]]
    points: List[Tuple[float, float, int]]
    gt_boxes_all: List[Tuple[float, float, float, float]]
    gt_masks_all: List[np.ndarray]


@dataclass
class LoadedSample:
    sample_id: str
    frames: List[LoadedFrame]
    text_prompt: Optional[str]
    concept: Optional[str]
    concept_present: Optional[bool]


@dataclass
class MaskPrediction:
    mask: np.ndarray
    score: float
    label: Optional[str] = None


@dataclass(frozen=True)
class SeverityPreset:
    level: int
    translate: int
    padcrop: int
    scale: float
    rotation: float
    freq_cutoff: float
    freq_epsilon: float
    text_scale: float


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    family: str
    severity: int
    kind: str
    params: Dict[str, Any]


DEFAULT_SEVERITIES = {
    1: SeverityPreset(
        level=1,
        translate=4,
        padcrop=4,
        scale=0.95,
        rotation=10.0,
        freq_cutoff=0.18,
        freq_epsilon=4.0 / 255.0,
        text_scale=0.8,
    ),
    2: SeverityPreset(
        level=2,
        translate=8,
        padcrop=8,
        scale=0.9,
        rotation=20.0,
        freq_cutoff=0.28,
        freq_epsilon=8.0 / 255.0,
        text_scale=1.0,
    ),
    3: SeverityPreset(
        level=3,
        translate=12,
        padcrop=12,
        scale=0.85,
        rotation=30.0,
        freq_cutoff=0.38,
        freq_epsilon=12.0 / 255.0,
        text_scale=1.2,
    ),
}

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
DEFAULT_SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DEFAULT_SAM2_CONFIG_URL = (
    "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/"
    "sam2.1/sam2.1_hiera_b%2B.yaml"
)
DEFAULT_SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_base_plus.pt"
)
DEFAULT_GDINO_CONFIG_URL = (
    "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/"
    "groundingdino/config/GroundingDINO_SwinB_cfg.py"
)
DEFAULT_GDINO_CHECKPOINT_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/"
    "groundingdino_swinb_cogcoor.pth"
)
DEFAULT_SAM3_REPO_ID = "facebook/sam3"
DEFAULT_SAM3_FILENAME = "sam3.pt"
COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
DAVIS_TRAINVAL_480P_URL = (
    "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
)
DAVIS_TRAINVAL_1080P_URL = (
    "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-1080p.zip"
)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path).expanduser().resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must be a YAML mapping at the top level.")
    return data


def flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for key in [
        "seed",
        "deterministic",
        "device",
        "out_dir",
        "save_visuals",
        "visual_limit",
        "iou_threshold",
        "include_natural",
        "include_frequency",
        "severity_levels",
        "max_samples",
        "manifest",
        "cache_dir",
        "run_sam2",
        "run_sam3",
        "run_dino",
        "run_gtbox",
    ]:
        if key in cfg:
            defaults[key] = cfg[key]

    experiment = cfg.get("experiment", {})
    if isinstance(experiment, dict):
        if "mode" in experiment:
            defaults["experiment_mode"] = experiment["mode"]

    eval_cfg = cfg.get("eval", {})
    if isinstance(eval_cfg, dict):
        if "iou_tau" in eval_cfg:
            defaults["eval_iou_tau"] = eval_cfg["iou_tau"]
        if "topk_boxes" in eval_cfg:
            defaults["eval_topk_boxes"] = eval_cfg["topk_boxes"]

    plots_cfg = cfg.get("plots", {})
    if isinstance(plots_cfg, dict):
        if "group_by" in plots_cfg:
            defaults["plots_group_by"] = plots_cfg["group_by"]

    dataset = cfg.get("dataset", {})
    if isinstance(dataset, dict):
        if "preset" in dataset:
            defaults["dataset_preset"] = dataset["preset"]
        if "cache_dir" in dataset:
            defaults["cache_dir"] = dataset["cache_dir"]
        if "max_samples" in dataset:
            defaults["max_samples"] = dataset["max_samples"]
        if "manifest" in dataset:
            defaults["manifest"] = dataset["manifest"]
        if "negative_ratio" in dataset:
            defaults["dataset_negative_ratio"] = dataset["negative_ratio"]
        if "rebuild_manifest" in dataset:
            defaults["dataset_rebuild_manifest"] = dataset["rebuild_manifest"]
        if "allow_download" in dataset:
            defaults["dataset_allow_download"] = dataset["allow_download"]
        if "davis_resolution" in dataset:
            defaults["dataset_davis_resolution"] = dataset["davis_resolution"]

    sam2 = cfg.get("sam2", {})
    if isinstance(sam2, dict):
        if "gtbox" in sam2:
            defaults["run_gtbox"] = sam2["gtbox"]
        if "config" in sam2:
            defaults["sam2_config"] = sam2["config"]
        if "checkpoint" in sam2:
            defaults["sam2_checkpoint"] = sam2["checkpoint"]
        if "auto_download" in sam2:
            defaults["sam2_auto_download"] = sam2["auto_download"]
        if "config_url" in sam2:
            defaults["sam2_config_url"] = sam2["config_url"]
        if "checkpoint_url" in sam2:
            defaults["sam2_checkpoint_url"] = sam2["checkpoint_url"]

    sam3 = cfg.get("sam3", {})
    if isinstance(sam3, dict):
        if "backend" in sam3:
            defaults["sam3_backend"] = sam3["backend"]
        if "module" in sam3:
            defaults["sam3_module"] = sam3["module"]
        if "class" in sam3:
            defaults["sam3_class"] = sam3["class"]
        if "checkpoint" in sam3:
            defaults["sam3_checkpoint"] = sam3["checkpoint"]
        if "repo_id" in sam3:
            defaults["sam3_repo_id"] = sam3["repo_id"]
        if "filename" in sam3:
            defaults["sam3_filename"] = sam3["filename"]
        if "hf_token" in sam3:
            defaults["sam3_token"] = sam3["hf_token"]
        if "auto_download" in sam3:
            defaults["sam3_auto_download"] = sam3["auto_download"]

    gdino = cfg.get("gdino", {})
    if isinstance(gdino, dict):
        if "enable" in gdino:
            defaults["run_dino"] = gdino["enable"]
        if "config" in gdino:
            defaults["gdino_config"] = gdino["config"]
        if "checkpoint" in gdino:
            defaults["gdino_checkpoint"] = gdino["checkpoint"]
        if "auto_download" in gdino:
            defaults["gdino_auto_download"] = gdino["auto_download"]
        if "config_url" in gdino:
            defaults["gdino_config_url"] = gdino["config_url"]
        if "checkpoint_url" in gdino:
            defaults["gdino_checkpoint_url"] = gdino["checkpoint_url"]
        if "box_threshold" in gdino:
            defaults["gdino_box_threshold"] = gdino["box_threshold"]
        if "text_threshold" in gdino:
            defaults["gdino_text_threshold"] = gdino["text_threshold"]

    run_cfg = cfg.get("run", {})
    if isinstance(run_cfg, dict):
        mode = run_cfg.get("mode")
        if mode:
            mode_str = str(mode).strip().lower()
            if mode_str == "sam2":
                defaults["run_sam2"] = True
                defaults["run_sam3"] = False
            elif mode_str == "sam3":
                defaults["run_sam2"] = False
                defaults["run_sam3"] = True
            elif mode_str == "both":
                defaults["run_sam2"] = True
                defaults["run_sam3"] = True

    return {k: v for k, v in defaults.items() if v is not None}


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest


def extract_zip(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(dest_dir)


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max()) + 1.0
    y1 = float(ys.max()) + 1.0
    return (x0, y0, x1, y1)


def mask_to_point(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def ensure_coco_assets(cache_dir: Path, allow_download: bool) -> tuple[Path, Path]:
    root = cache_dir / "datasets" / "mini_coco"
    images_dir = root / "val2017"
    ann_file = root / "annotations" / "instances_val2017.json"
    if not images_dir.exists():
        if not allow_download:
            raise FileNotFoundError("COCO images not found and auto-download is disabled.")
        zip_path = download_file(COCO_VAL_URL, root / "val2017.zip")
        extract_zip(zip_path, root)
    if not ann_file.exists():
        if not allow_download:
            raise FileNotFoundError("COCO annotations not found and auto-download is disabled.")
        zip_path = download_file(COCO_ANN_URL, root / "annotations_trainval2017.zip")
        extract_zip(zip_path, root)
    return images_dir, ann_file


def build_mini_coco_manifest(
    cache_dir: Path,
    max_samples: int,
    negative_ratio: float,
    seed: int,
    allow_download: bool,
    rebuild: bool,
) -> Path:
    from pycocotools.coco import COCO

    root = cache_dir / "datasets" / "mini_coco"
    manifest_path = root / "manifest.jsonl"
    if manifest_path.exists() and not rebuild:
        return manifest_path
    images_dir, ann_file = ensure_coco_assets(cache_dir, allow_download)
    coco = COCO(str(ann_file))
    img_ids = list(coco.imgs.keys())
    rng = random.Random(seed)
    rng.shuffle(img_ids)
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    all_cat_ids = list(cat_id_to_name.keys())
    mask_dir = root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    if max_samples is None:
        max_samples = 200
    if max_samples <= 0:
        positive_count = len(img_ids)
        total_target = int(math.ceil(positive_count / (1.0 - negative_ratio)))
        negative_count = max(0, total_target - positive_count)
        max_samples = positive_count + negative_count
    else:
        negative_count = int(round(max_samples * negative_ratio))
        positive_count = max_samples - negative_count

    entries: List[Dict[str, Any]] = []
    selected_img_ids: List[int] = []
    idx = 0
    while len(entries) < positive_count and idx < len(img_ids):
        img_id = img_ids[idx]
        idx += 1
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        if not ann_ids:
            continue
        anns = coco.loadAnns(ann_ids)
        ann = max(anns, key=lambda a: a.get("area", 0.0))
        mask = coco.annToMask(ann).astype(np.uint8)
        if mask.sum() == 0:
            continue
        image_info = coco.imgs[img_id]
        image_path = images_dir / image_info["file_name"]
        mask_path = mask_dir / f"{img_id}_{ann['id']}.png"
        Image.fromarray(mask).save(mask_path)
        concept_anns = [a for a in anns if a.get("category_id") == ann.get("category_id")]
        gt_masks_all: List[str] = []
        gt_boxes_all: List[List[float]] = []
        for c_ann in concept_anns:
            c_mask = coco.annToMask(c_ann).astype(np.uint8)
            if c_mask.sum() == 0:
                continue
            c_mask_path = mask_dir / f"{img_id}_{c_ann['id']}.png"
            if rebuild or not c_mask_path.exists():
                Image.fromarray(c_mask).save(c_mask_path)
            cx0, cy0, cw, ch = c_ann["bbox"]
            gt_boxes_all.append([float(cx0), float(cy0), float(cx0 + cw), float(cy0 + ch)])
            gt_masks_all.append(str(c_mask_path.relative_to(root)))
        x, y, w, h = ann["bbox"]
        bbox = [float(x), float(y), float(x + w), float(y + h)]
        centroid = mask_to_point(mask)
        if centroid is None:
            cx = float(x + w / 2.0)
            cy = float(y + h / 2.0)
        else:
            cx, cy = centroid
        entry = {
            "id": f"coco_{img_id}_{ann['id']}",
            "split": "val",
            "frames": [
                {
                    "image": str(image_path.relative_to(root)),
                    "mask": str(mask_path.relative_to(root)),
                    "gt_mask": str(mask_path.relative_to(root)),
                    "instance_id": 1,
                    "boxes": [bbox],
                    "gt_box": bbox,
                    "points": [[cx, cy, 1]],
                    "gt_points": [[cx, cy, 1]],
                    "gt_boxes_all": gt_boxes_all,
                    "gt_masks_all": gt_masks_all,
                }
            ],
            "text_prompt": cat_id_to_name.get(ann["category_id"], "object"),
            "concept": cat_id_to_name.get(ann["category_id"], "object"),
            "concept_present": True,
        }
        entries.append(entry)
        selected_img_ids.append(img_id)

    img_to_cats: Dict[int, set[int]] = {}
    for img_id in selected_img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        img_to_cats[img_id] = {int(a["category_id"]) for a in anns}

    if not selected_img_ids:
        raise RuntimeError("No COCO samples available to build the manifest.")

    neg_idx = 0
    attempts = 0
    max_attempts = max(1, len(selected_img_ids)) * 10
    while len(entries) < max_samples and attempts < max_attempts:
        img_id = selected_img_ids[neg_idx % len(selected_img_ids)]
        neg_idx += 1
        attempts += 1
        present = img_to_cats.get(img_id, set())
        absent = [cid for cid in all_cat_ids if cid not in present]
        if not absent:
            continue
        neg_cat = rng.choice(absent)
        image_info = coco.imgs[img_id]
        image_path = images_dir / image_info["file_name"]
        entry = {
            "id": f"coco_{img_id}_neg_{neg_cat}",
            "split": "val",
            "frames": [
                {
                    "image": str(image_path.relative_to(root)),
                    "boxes": [],
                    "points": [],
                    "gt_boxes_all": [],
                    "gt_masks_all": [],
                }
            ],
            "text_prompt": cat_id_to_name.get(neg_cat, "object"),
            "concept": cat_id_to_name.get(neg_cat, "object"),
            "concept_present": False,
        }
        entries.append(entry)

    root.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return manifest_path


def ensure_davis_assets(cache_dir: Path, allow_download: bool, resolution: str) -> Path:
    root = cache_dir / "datasets" / "davis17"
    data_root = root / "DAVIS"
    if data_root.exists():
        return data_root
    if not allow_download:
        raise FileNotFoundError("DAVIS data not found and auto-download is disabled.")
    url = DAVIS_TRAINVAL_480P_URL if resolution == "480p" else DAVIS_TRAINVAL_1080P_URL
    zip_path = download_file(url, root / Path(url).name)
    extract_zip(zip_path, root)
    return data_root


def build_davis_manifest(
    cache_dir: Path,
    max_samples: int,
    seed: int,
    allow_download: bool,
    rebuild: bool,
    resolution: str,
) -> Path:
    root = cache_dir / "datasets" / "davis17"
    manifest_path = root / "manifest.jsonl"
    if manifest_path.exists() and not rebuild:
        return manifest_path
    data_root = ensure_davis_assets(cache_dir, allow_download, resolution)
    split_file = data_root / "ImageSets" / "2017" / "val.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing DAVIS split file: {split_file}")
    with open(split_file, "r", encoding="utf-8") as f:
        sequences = [line.strip() for line in f if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(sequences)
    if max_samples > 0:
        sequences = sequences[:max_samples]

    mask_out_root = root / "masks"
    entries: List[Dict[str, Any]] = []
    for seq in sequences:
        img_dir = data_root / "JPEGImages" / resolution / seq
        ann_dir = data_root / "Annotations" / resolution / seq
        if not img_dir.exists() or not ann_dir.exists():
            continue
        frame_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})
        if not frame_files:
            continue
        first_mask = Image.open(ann_dir / (frame_files[0].stem + ".png"))
        first_arr = np.array(first_mask)
        instance_ids = [int(v) for v in np.unique(first_arr) if v != 0]
        if not instance_ids:
            continue
        target_id = instance_ids[0]
        frames: List[Dict[str, Any]] = []
        for frame_path in frame_files:
            ann_path = ann_dir / (frame_path.stem + ".png")
            ann_arr = np.array(Image.open(ann_path))
            bin_mask = (ann_arr == target_id).astype(np.uint8)
            mask_path = mask_out_root / seq / f"{frame_path.stem}.png"
            if rebuild or not mask_path.exists():
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(bin_mask).save(mask_path)
            bbox = mask_to_bbox(bin_mask)
            if bbox is None:
                boxes = []
                points = []
                gt_boxes_all: List[List[float]] = []
            else:
                centroid = mask_to_point(bin_mask)
                if centroid is None:
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                else:
                    cx, cy = centroid
                boxes = [[bbox[0], bbox[1], bbox[2], bbox[3]]]
                points = [[cx, cy, 1]]
                gt_boxes_all = [boxes[0]]
            frames.append(
                {
                    "image": str(frame_path.relative_to(root)),
                    "mask": str(mask_path.relative_to(root)),
                    "gt_mask": str(mask_path.relative_to(root)),
                    "instance_id": 1,
                    "boxes": boxes,
                    "gt_box": boxes[0] if boxes else None,
                    "points": points,
                    "gt_points": points,
                    "gt_boxes_all": gt_boxes_all,
                    "gt_masks_all": [str(mask_path.relative_to(root))] if boxes else [],
                }
            )
        entries.append(
            {
                "id": f"davis_{seq}",
                "split": "val",
                "frames": frames,
                "text_prompt": seq.replace("_", " "),
                "concept": seq.replace("_", " "),
                "concept_present": True,
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return manifest_path


def resolve_manifest_path(args: argparse.Namespace, cache_dir: Path) -> Path:
    if args.manifest:
        return Path(args.manifest).expanduser().resolve()
    preset = args.dataset_preset
    if preset == "mini_coco":
        return build_mini_coco_manifest(
            cache_dir=cache_dir,
            max_samples=args.max_samples,
            negative_ratio=args.dataset_negative_ratio,
            seed=args.seed,
            allow_download=args.dataset_allow_download,
            rebuild=args.dataset_rebuild_manifest,
        )
    if preset == "davis17":
        return build_davis_manifest(
            cache_dir=cache_dir,
            max_samples=args.max_samples,
            seed=args.seed,
            allow_download=args.dataset_allow_download,
            rebuild=args.dataset_rebuild_manifest,
            resolution=args.dataset_davis_resolution,
        )
    raise ValueError(f"Unsupported dataset preset: {preset}")


def resolve_path(root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_manifest(path: Path) -> List[SegSample]:
    samples: List[SegSample] = []
    root = path.parent
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            frames: List[FrameRecord] = []
            for frame in data.get("frames", []):
                image_path = resolve_path(root, frame.get("image"))
                mask_path = resolve_path(root, frame.get("mask"))
                if image_path is None:
                    raise ValueError("Frame is missing an image path.")
                boxes = [tuple(map(float, b)) for b in frame.get("boxes", [])]
                points = [
                    (float(p[0]), float(p[1]), int(p[2]))
                    for p in frame.get("points", [])
                ]
                gt_boxes_all = [tuple(map(float, b)) for b in frame.get("gt_boxes_all", [])]
                if not gt_boxes_all:
                    gt_boxes_all = list(boxes)
                gt_mask_paths_all = [
                    resolve_path(root, p) for p in frame.get("gt_masks_all", []) if p
                ]
                if not gt_mask_paths_all and mask_path is not None:
                    gt_mask_paths_all = [mask_path]
                instance_id = frame.get("instance_id")
                if instance_id is not None:
                    instance_id = int(instance_id)
                frames.append(
                    FrameRecord(
                        image_path=image_path,
                        mask_path=mask_path,
                        instance_id=instance_id,
                        boxes=boxes,
                        points=points,
                        gt_boxes_all=gt_boxes_all,
                        gt_mask_paths_all=gt_mask_paths_all,
                    )
                )
            sample = SegSample(
                sample_id=str(data.get("id", len(samples))),
                frames=frames,
                text_prompt=data.get("text_prompt"),
                concept=data.get("concept"),
                concept_present=data.get("concept_present"),
                split=data.get("split"),
            )
            samples.append(sample)
    return samples


def select_subset(samples: List[SegSample], max_samples: int, seed: int) -> List[SegSample]:
    if max_samples <= 0 or max_samples >= len(samples):
        return samples
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    return [samples[i] for i in indices[:max_samples]]


def load_mask(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    mask_img = Image.open(path)
    mask_arr = np.array(mask_img)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    return mask_arr


def load_sample(sample: SegSample) -> LoadedSample:
    frames: List[LoadedFrame] = []
    for frame in sample.frames:
        image = Image.open(frame.image_path).convert("RGB")
        mask = load_mask(frame.mask_path)
        gt_masks_all: List[np.ndarray] = []
        for mask_path in frame.gt_mask_paths_all:
            loaded = load_mask(mask_path)
            if loaded is not None:
                gt_masks_all.append(loaded)
        if not gt_masks_all and mask is not None:
            gt_masks_all = [mask]
        frames.append(
            LoadedFrame(
                image=image,
                mask=mask,
                instance_id=frame.instance_id,
                boxes=list(frame.boxes),
                points=list(frame.points),
                gt_boxes_all=list(frame.gt_boxes_all),
                gt_masks_all=gt_masks_all,
            )
        )
    text_prompt = sample.text_prompt or sample.concept
    return LoadedSample(
        sample_id=sample.sample_id,
        frames=frames,
        text_prompt=text_prompt,
        concept=sample.concept,
        concept_present=sample.concept_present,
    )


def _pil_from_mask(mask: np.ndarray) -> Image.Image:
    if mask.dtype != np.uint8:
        if mask.max() > 255:
            mask = mask.astype(np.uint16)
        else:
            mask = mask.astype(np.uint8)
    return Image.fromarray(mask)


def _mask_from_pil(mask_img: Image.Image) -> np.ndarray:
    return np.array(mask_img)


def _clip_boxes(
    boxes: List[Tuple[float, float, float, float]],
    width: int,
    height: int,
) -> List[Tuple[float, float, float, float]]:
    clipped: List[Tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in boxes:
        x0 = max(0.0, min(width, x0))
        x1 = max(0.0, min(width, x1))
        y0 = max(0.0, min(height, y0))
        y1 = max(0.0, min(height, y1))
        if x1 > x0 and y1 > y0:
            clipped.append((x0, y0, x1, y1))
    return clipped


def _clip_points(
    points: List[Tuple[float, float, int]],
    width: int,
    height: int,
) -> List[Tuple[float, float, int]]:
    kept: List[Tuple[float, float, int]] = []
    for x, y, label in points:
        if 0.0 <= x < width and 0.0 <= y < height:
            kept.append((x, y, label))
    return kept


def _shift_boxes(
    boxes: List[Tuple[float, float, float, float]],
    dx: float,
    dy: float,
) -> List[Tuple[float, float, float, float]]:
    return [(x0 + dx, y0 + dy, x1 + dx, y1 + dy) for x0, y0, x1, y1 in boxes]


def _shift_points(
    points: List[Tuple[float, float, int]],
    dx: float,
    dy: float,
) -> List[Tuple[float, float, int]]:
    return [(x + dx, y + dy, label) for x, y, label in points]


def _scale_boxes(
    boxes: List[Tuple[float, float, float, float]],
    scale: float,
) -> List[Tuple[float, float, float, float]]:
    return [(x0 * scale, y0 * scale, x1 * scale, y1 * scale) for x0, y0, x1, y1 in boxes]


def _scale_points(
    points: List[Tuple[float, float, int]],
    scale: float,
) -> List[Tuple[float, float, int]]:
    return [(x * scale, y * scale, label) for x, y, label in points]


def _rotate_points(
    points: List[Tuple[float, float, int]],
    angle_deg: float,
    width: int,
    height: int,
    new_w: int,
    new_h: int,
) -> List[Tuple[float, float, int]]:
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    cx, cy = width / 2.0, height / 2.0
    ncx, ncy = new_w / 2.0, new_h / 2.0
    out: List[Tuple[float, float, int]] = []
    for x, y, label in points:
        x0 = x - cx
        y0 = y - cy
        xr = x0 * cos_a - y0 * sin_a + ncx
        yr = x0 * sin_a + y0 * cos_a + ncy
        out.append((xr, yr, label))
    return out


def _rotate_boxes(
    boxes: List[Tuple[float, float, float, float]],
    angle_deg: float,
    width: int,
    height: int,
    new_w: int,
    new_h: int,
) -> List[Tuple[float, float, float, float]]:
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    cx, cy = width / 2.0, height / 2.0
    ncx, ncy = new_w / 2.0, new_h / 2.0
    rotated: List[Tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in boxes:
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        xs: List[float] = []
        ys: List[float] = []
        for x, y in corners:
            x0r = x - cx
            y0r = y - cy
            xr = x0r * cos_a - y0r * sin_a + ncx
            yr = x0r * sin_a + y0r * cos_a + ncy
            xs.append(xr)
            ys.append(yr)
        rotated.append((min(xs), min(ys), max(xs), max(ys)))
    return rotated


def _rotate_size(width: int, height: int, angle_deg: float) -> Tuple[int, int]:
    rad = math.radians(angle_deg)
    cos_a = abs(math.cos(rad))
    sin_a = abs(math.sin(rad))
    new_w = int(round(width * cos_a + height * sin_a))
    new_h = int(round(width * sin_a + height * cos_a))
    return new_w, new_h


def translate_frame(frame: LoadedFrame, dx: int, dy: int) -> LoadedFrame:
    image = Image.new(frame.image.mode, frame.image.size, "black")
    image.paste(frame.image, (dx, dy))
    mask = None
    if frame.mask is not None:
        mask_img = _pil_from_mask(frame.mask)
        mask_out = Image.new(mask_img.mode, mask_img.size, 0)
        mask_out.paste(mask_img, (dx, dy))
        mask = _mask_from_pil(mask_out)
    width, height = image.size
    boxes = _clip_boxes(_shift_boxes(frame.boxes, dx, dy), width, height)
    points = _clip_points(_shift_points(frame.points, dx, dy), width, height)
    gt_masks_all: List[np.ndarray] = []
    for gt_mask in frame.gt_masks_all:
        mask_img = _pil_from_mask(gt_mask)
        mask_out = Image.new(mask_img.mode, mask_img.size, 0)
        mask_out.paste(mask_img, (dx, dy))
        gt_masks_all.append(_mask_from_pil(mask_out))
    gt_boxes_all = _clip_boxes(_shift_boxes(frame.gt_boxes_all, dx, dy), width, height)
    return LoadedFrame(
        image=image,
        mask=mask,
        instance_id=frame.instance_id,
        boxes=boxes,
        points=points,
        gt_boxes_all=gt_boxes_all,
        gt_masks_all=gt_masks_all,
    )


def pad_or_crop_frame(frame: LoadedFrame, pad: int) -> LoadedFrame:
    width, height = frame.image.size
    if pad == 0:
        return LoadedFrame(
            image=frame.image.copy(),
            mask=frame.mask.copy() if frame.mask is not None else None,
            instance_id=frame.instance_id,
            boxes=list(frame.boxes),
            points=list(frame.points),
            gt_boxes_all=list(frame.gt_boxes_all),
            gt_masks_all=[m.copy() for m in frame.gt_masks_all],
        )
    if pad > 0:
        new_w, new_h = width + 2 * pad, height + 2 * pad
        image = Image.new(frame.image.mode, (new_w, new_h), "black")
        image.paste(frame.image, (pad, pad))
        mask = None
        if frame.mask is not None:
            mask_img = _pil_from_mask(frame.mask)
            mask_out = Image.new(mask_img.mode, (new_w, new_h), 0)
            mask_out.paste(mask_img, (pad, pad))
            mask = _mask_from_pil(mask_out)
        boxes = _clip_boxes(_shift_boxes(frame.boxes, pad, pad), new_w, new_h)
        points = _clip_points(_shift_points(frame.points, pad, pad), new_w, new_h)
        gt_masks_all: List[np.ndarray] = []
        for gt_mask in frame.gt_masks_all:
            mask_img = _pil_from_mask(gt_mask)
            mask_out = Image.new(mask_img.mode, (new_w, new_h), 0)
            mask_out.paste(mask_img, (pad, pad))
            gt_masks_all.append(_mask_from_pil(mask_out))
        gt_boxes_all = _clip_boxes(_shift_boxes(frame.gt_boxes_all, pad, pad), new_w, new_h)
        return LoadedFrame(
            image=image,
            mask=mask,
            instance_id=frame.instance_id,
            boxes=boxes,
            points=points,
            gt_boxes_all=gt_boxes_all,
            gt_masks_all=gt_masks_all,
        )
    crop = abs(pad)
    if width <= 2 * crop or height <= 2 * crop:
        return LoadedFrame(
            image=frame.image.copy(),
            mask=frame.mask.copy() if frame.mask is not None else None,
            instance_id=frame.instance_id,
            boxes=list(frame.boxes),
            points=list(frame.points),
            gt_boxes_all=list(frame.gt_boxes_all),
            gt_masks_all=[m.copy() for m in frame.gt_masks_all],
        )
    image = frame.image.crop((crop, crop, width - crop, height - crop))
    mask = None
    if frame.mask is not None:
        mask_img = _pil_from_mask(frame.mask)
        mask_out = mask_img.crop((crop, crop, width - crop, height - crop))
        mask = _mask_from_pil(mask_out)
    new_w, new_h = image.size
    boxes = _clip_boxes(_shift_boxes(frame.boxes, -crop, -crop), new_w, new_h)
    points = _clip_points(_shift_points(frame.points, -crop, -crop), new_w, new_h)
    gt_masks_all: List[np.ndarray] = []
    for gt_mask in frame.gt_masks_all:
        mask_img = _pil_from_mask(gt_mask)
        mask_out = mask_img.crop((crop, crop, width - crop, height - crop))
        gt_masks_all.append(_mask_from_pil(mask_out))
    gt_boxes_all = _clip_boxes(_shift_boxes(frame.gt_boxes_all, -crop, -crop), new_w, new_h)
    return LoadedFrame(
        image=image,
        mask=mask,
        instance_id=frame.instance_id,
        boxes=boxes,
        points=points,
        gt_boxes_all=gt_boxes_all,
        gt_masks_all=gt_masks_all,
    )


def scale_frame(frame: LoadedFrame, scale: float) -> LoadedFrame:
    width, height = frame.image.size
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    image = frame.image.resize((new_w, new_h), resample=Image.BICUBIC)
    mask = None
    if frame.mask is not None:
        mask_img = _pil_from_mask(frame.mask)
        mask_out = mask_img.resize((new_w, new_h), resample=Image.NEAREST)
        mask = _mask_from_pil(mask_out)
    boxes = _clip_boxes(_scale_boxes(frame.boxes, scale), new_w, new_h)
    points = _clip_points(_scale_points(frame.points, scale), new_w, new_h)
    gt_masks_all: List[np.ndarray] = []
    for gt_mask in frame.gt_masks_all:
        mask_img = _pil_from_mask(gt_mask)
        mask_out = mask_img.resize((new_w, new_h), resample=Image.NEAREST)
        gt_masks_all.append(_mask_from_pil(mask_out))
    gt_boxes_all = _clip_boxes(_scale_boxes(frame.gt_boxes_all, scale), new_w, new_h)
    return LoadedFrame(
        image=image,
        mask=mask,
        instance_id=frame.instance_id,
        boxes=boxes,
        points=points,
        gt_boxes_all=gt_boxes_all,
        gt_masks_all=gt_masks_all,
    )


def scale_pad_frame(frame: LoadedFrame, scale: float, background: str) -> LoadedFrame:
    width, height = frame.image.size
    scaled = scale_frame(frame, scale)
    canvas = Image.new(frame.image.mode, (width, height), background)
    offset_x = (width - scaled.image.size[0]) // 2
    offset_y = (height - scaled.image.size[1]) // 2
    canvas.paste(scaled.image, (offset_x, offset_y))
    mask = None
    if scaled.mask is not None:
        mask_img = _pil_from_mask(scaled.mask)
        mask_canvas = Image.new(mask_img.mode, (width, height), 0)
        mask_canvas.paste(mask_img, (offset_x, offset_y))
        mask = _mask_from_pil(mask_canvas)
    boxes = _clip_boxes(
        _shift_boxes(_scale_boxes(frame.boxes, scale), offset_x, offset_y),
        width,
        height,
    )
    points = _clip_points(
        _shift_points(_scale_points(frame.points, scale), offset_x, offset_y),
        width,
        height,
    )
    gt_masks_all: List[np.ndarray] = []
    for gt_mask in scaled.gt_masks_all:
        mask_img = _pil_from_mask(gt_mask)
        mask_canvas = Image.new(mask_img.mode, (width, height), 0)
        mask_canvas.paste(mask_img, (offset_x, offset_y))
        gt_masks_all.append(_mask_from_pil(mask_canvas))
    gt_boxes_all = _clip_boxes(
        _shift_boxes(_scale_boxes(frame.gt_boxes_all, scale), offset_x, offset_y),
        width,
        height,
    )
    return LoadedFrame(
        image=canvas,
        mask=mask,
        instance_id=frame.instance_id,
        boxes=boxes,
        points=points,
        gt_boxes_all=gt_boxes_all,
        gt_masks_all=gt_masks_all,
    )


def rotate_frame(frame: LoadedFrame, angle_deg: float) -> LoadedFrame:
    width, height = frame.image.size
    new_w, new_h = _rotate_size(width, height, angle_deg)
    image = frame.image.rotate(angle_deg, resample=Image.BICUBIC, expand=True, fillcolor="white")
    mask = None
    if frame.mask is not None:
        mask_img = _pil_from_mask(frame.mask)
        mask_out = mask_img.rotate(angle_deg, resample=Image.NEAREST, expand=True, fillcolor=0)
        mask = _mask_from_pil(mask_out)
    boxes = _clip_boxes(
        _rotate_boxes(frame.boxes, angle_deg, width, height, new_w, new_h),
        new_w,
        new_h,
    )
    points = _clip_points(
        _rotate_points(frame.points, angle_deg, width, height, new_w, new_h),
        new_w,
        new_h,
    )
    gt_masks_all: List[np.ndarray] = []
    for gt_mask in frame.gt_masks_all:
        mask_img = _pil_from_mask(gt_mask)
        mask_out = mask_img.rotate(angle_deg, resample=Image.NEAREST, expand=True, fillcolor=0)
        gt_masks_all.append(_mask_from_pil(mask_out))
    gt_boxes_all = _clip_boxes(
        _rotate_boxes(frame.gt_boxes_all, angle_deg, width, height, new_w, new_h),
        new_w,
        new_h,
    )
    return LoadedFrame(
        image=image,
        mask=mask,
        instance_id=frame.instance_id,
        boxes=boxes,
        points=points,
        gt_boxes_all=gt_boxes_all,
        gt_masks_all=gt_masks_all,
    )


def overlay_font(image: Image.Image, scale: float) -> ImageFont.ImageFont:
    h = image.size[1]
    size = max(12, int(round(h / 28 * scale)))
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def overlay_text_box(image: Image.Image, text: str, scale: float) -> Tuple[int, int, int, int]:
    font = overlay_font(image, scale)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    w, h = image.size
    x = max(0, (w - tw) // 2)
    y = max(0, (h - th) // 2)
    pad = max(4, int(round(4 * scale)))
    return (x - pad, y - pad, x + tw + pad, y + th + pad)


def overlay_text(
    image: Image.Image,
    text: str,
    scale: float,
    box: Optional[Tuple[int, int, int, int]] = None,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = overlay_font(image, scale)
    if box is None:
        box = overlay_text_box(image, text, scale)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    x = int(round(box[0] + (box_w - tw) / 2.0))
    y = int(round(box[1] + (box_h - th) / 2.0))
    draw.rectangle(list(box), fill="white")
    draw.text((x, y), text, fill="red", font=font)
    return out


def overlay_box(
    image: Image.Image,
    scale: float,
    box: Optional[Tuple[int, int, int, int]] = None,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    if box is None:
        w, h = image.size
        box_w = int(round(w * 0.45 * scale))
        box_h = int(round(h * 0.25 * scale))
        x0 = max(0, (w - box_w) // 2)
        y0 = max(0, (h - box_h) // 2)
        box = (x0, y0, x0 + box_w, y0 + box_h)
    draw.rectangle(list(box), fill="white")
    return out


def _rand_phrase(rng: random.Random, length: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(rng.choice(alphabet) for _ in range(length))


def make_frequency_mask(height: int, width: int, mode: str, cutoff: float) -> np.ndarray:
    fy = np.fft.fftfreq(height).reshape(height, 1)
    fx = np.fft.fftfreq(width).reshape(1, width)
    radius = np.sqrt(fx * fx + fy * fy)
    if mode == "high":
        mask = radius >= cutoff
    elif mode == "low":
        mask = radius <= cutoff
    else:
        mask = np.ones_like(radius, dtype=bool)
    return mask.astype(np.float32)


def fft_filter_keep(image: Image.Image, mode: str, cutoff: float) -> Image.Image:
    arr = np.asarray(image).astype(np.float32) / 255.0
    h, w, c = arr.shape
    mask = make_frequency_mask(h, w, mode, cutoff)
    out = np.zeros_like(arr)
    for ch in range(c):
        fft = np.fft.fft2(arr[:, :, ch])
        filtered = np.fft.ifft2(fft * mask).real
        out[:, :, ch] = filtered
    out = np.clip(out, 0.0, 1.0)
    return Image.fromarray((out * 255.0).round().astype(np.uint8))


def fft_band_noise(
    image: Image.Image,
    mode: str,
    cutoff: float,
    epsilon: float,
    rng: np.random.Generator,
) -> Image.Image:
    arr = np.asarray(image).astype(np.float32) / 255.0
    h, w, c = arr.shape
    mask = make_frequency_mask(h, w, mode, cutoff)
    noise = rng.normal(0.0, 1.0, size=arr.shape).astype(np.float32)
    filt = np.zeros_like(noise)
    for ch in range(c):
        fft = np.fft.fft2(noise[:, :, ch])
        filtered = np.fft.ifft2(fft * mask).real
        filt[:, :, ch] = filtered
    max_abs = float(np.max(np.abs(filt)))
    if max_abs > 0:
        filt = filt / max_abs * epsilon
    out = np.clip(arr + filt, 0.0, 1.0)
    return Image.fromarray((out * 255.0).round().astype(np.uint8))


def build_perturbations(
    severity_levels: List[int],
    include_natural: bool,
    include_frequency: bool,
) -> List[PerturbationSpec]:
    specs: List[PerturbationSpec] = [
        PerturbationSpec(name="Base", family="base", severity=0, kind="base", params={})
    ]
    for level in severity_levels:
        preset = DEFAULT_SEVERITIES.get(level)
        if preset is None:
            raise ValueError(f"Unknown severity level: {level}")
        if include_natural:
            specs.extend(
                [
                    PerturbationSpec(
                        name=f"Translation(+{preset.translate})",
                        family="natural",
                        severity=level,
                        kind="translate",
                        params={"dx": preset.translate, "dy": 0},
                    ),
                    PerturbationSpec(
                        name=f"Translation(-{preset.translate})",
                        family="natural",
                        severity=level,
                        kind="translate",
                        params={"dx": -preset.translate, "dy": 0},
                    ),
                    PerturbationSpec(
                        name=f"PadCrop(+{preset.padcrop})",
                        family="natural",
                        severity=level,
                        kind="padcrop",
                        params={"pad": preset.padcrop},
                    ),
                    PerturbationSpec(
                        name=f"PadCrop(-{preset.padcrop})",
                        family="natural",
                        severity=level,
                        kind="padcrop",
                        params={"pad": -preset.padcrop},
                    ),
                    PerturbationSpec(
                        name=f"Scale({preset.scale:.2f})",
                        family="natural",
                        severity=level,
                        kind="scale",
                        params={"scale": preset.scale},
                    ),
                    PerturbationSpec(
                        name=f"ScalePadBlack({preset.scale:.2f})",
                        family="natural",
                        severity=level,
                        kind="scale_pad",
                        params={"scale": preset.scale, "background": "black"},
                    ),
                    PerturbationSpec(
                        name=f"ScalePadWhite({preset.scale:.2f})",
                        family="natural",
                        severity=level,
                        kind="scale_pad",
                        params={"scale": preset.scale, "background": "white"},
                    ),
                    PerturbationSpec(
                        name=f"Rotation(+{preset.rotation:.0f})",
                        family="natural",
                        severity=level,
                        kind="rotate",
                        params={"angle": preset.rotation},
                    ),
                    PerturbationSpec(
                        name=f"Rotation(-{preset.rotation:.0f})",
                        family="natural",
                        severity=level,
                        kind="rotate",
                        params={"angle": -preset.rotation},
                    ),
                    PerturbationSpec(
                        name="TextOverlay",
                        family="natural",
                        severity=level,
                        kind="text_overlay",
                        params={"scale": preset.text_scale},
                    ),
                    PerturbationSpec(
                        name="BoxOverlay",
                        family="natural",
                        severity=level,
                        kind="box_overlay",
                        params={"scale": preset.text_scale},
                    ),
                    PerturbationSpec(
                        name="RandomText",
                        family="natural",
                        severity=level,
                        kind="random_text",
                        params={"scale": preset.text_scale},
                    ),
                ]
            )
        if include_frequency:
            specs.extend(
                [
                    PerturbationSpec(
                        name=f"LowPassKeep({preset.freq_cutoff:.2f})",
                        family="frequency",
                        severity=level,
                        kind="fft_keep",
                        params={"mode": "low", "cutoff": preset.freq_cutoff},
                    ),
                    PerturbationSpec(
                        name=f"HighPassKeep({preset.freq_cutoff:.2f})",
                        family="frequency",
                        severity=level,
                        kind="fft_keep",
                        params={"mode": "high", "cutoff": preset.freq_cutoff},
                    ),
                    PerturbationSpec(
                        name=f"LowBandNoise({preset.freq_cutoff:.2f})",
                        family="frequency",
                        severity=level,
                        kind="fft_noise",
                        params={
                            "mode": "low",
                            "cutoff": preset.freq_cutoff,
                            "epsilon": preset.freq_epsilon,
                        },
                    ),
                    PerturbationSpec(
                        name=f"HighBandNoise({preset.freq_cutoff:.2f})",
                        family="frequency",
                        severity=level,
                        kind="fft_noise",
                        params={
                            "mode": "high",
                            "cutoff": preset.freq_cutoff,
                            "epsilon": preset.freq_epsilon,
                        },
                    ),
                    PerturbationSpec(
                        name=f"AllBandNoise({preset.freq_cutoff:.2f})",
                        family="frequency",
                        severity=level,
                        kind="fft_noise",
                        params={
                            "mode": "all",
                            "cutoff": preset.freq_cutoff,
                            "epsilon": preset.freq_epsilon,
                        },
                    ),
                ]
            )
    return specs


def apply_perturbation(
    sample: LoadedSample,
    spec: PerturbationSpec,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> LoadedSample:
    if spec.kind == "base":
        return LoadedSample(
            sample_id=sample.sample_id,
            frames=[
                LoadedFrame(
                    image=frame.image.copy(),
                    mask=frame.mask.copy() if frame.mask is not None else None,
                    instance_id=frame.instance_id,
                    boxes=list(frame.boxes),
                    points=list(frame.points),
                    gt_boxes_all=list(frame.gt_boxes_all),
                    gt_masks_all=[m.copy() for m in frame.gt_masks_all],
                )
                for frame in sample.frames
            ],
            text_prompt=sample.text_prompt,
            concept=sample.concept,
            concept_present=sample.concept_present,
        )

    out_frames: List[LoadedFrame] = []
    if spec.kind == "translate":
        dx = int(spec.params["dx"])
        dy = int(spec.params.get("dy", 0))
        for frame in sample.frames:
            out_frames.append(translate_frame(frame, dx, dy))
    elif spec.kind == "padcrop":
        pad = int(spec.params["pad"])
        for frame in sample.frames:
            out_frames.append(pad_or_crop_frame(frame, pad))
    elif spec.kind == "scale":
        scale = float(spec.params["scale"])
        for frame in sample.frames:
            out_frames.append(scale_frame(frame, scale))
    elif spec.kind == "scale_pad":
        scale = float(spec.params["scale"])
        background = str(spec.params["background"])
        for frame in sample.frames:
            out_frames.append(scale_pad_frame(frame, scale, background))
    elif spec.kind == "rotate":
        angle = float(spec.params["angle"])
        for frame in sample.frames:
            out_frames.append(rotate_frame(frame, angle))
    elif spec.kind == "text_overlay":
        text = sample.concept or sample.text_prompt or "ANSWER"
        scale = float(spec.params.get("scale", 1.0))
        for frame in sample.frames:
            box = overlay_text_box(frame.image, text, scale)
            out_frames.append(
                LoadedFrame(
                    image=overlay_text(frame.image, text, scale, box),
                    mask=frame.mask.copy() if frame.mask is not None else None,
                    instance_id=frame.instance_id,
                    boxes=list(frame.boxes),
                    points=list(frame.points),
                    gt_boxes_all=list(frame.gt_boxes_all),
                    gt_masks_all=[m.copy() for m in frame.gt_masks_all],
                )
            )
    elif spec.kind == "box_overlay":
        scale = float(spec.params.get("scale", 1.0))
        text = sample.concept or sample.text_prompt or "ANSWER"
        for frame in sample.frames:
            box = overlay_text_box(frame.image, text, scale)
            out_frames.append(
                LoadedFrame(
                    image=overlay_box(frame.image, scale, box),
                    mask=frame.mask.copy() if frame.mask is not None else None,
                    instance_id=frame.instance_id,
                    boxes=list(frame.boxes),
                    points=list(frame.points),
                    gt_boxes_all=list(frame.gt_boxes_all),
                    gt_masks_all=[m.copy() for m in frame.gt_masks_all],
                )
            )
    elif spec.kind == "random_text":
        scale = float(spec.params.get("scale", 1.0))
        phrase = _rand_phrase(rng, 10)
        text = sample.concept or sample.text_prompt or "ANSWER"
        for frame in sample.frames:
            box = overlay_text_box(frame.image, text, scale)
            out_frames.append(
                LoadedFrame(
                    image=overlay_text(frame.image, phrase, scale, box),
                    mask=frame.mask.copy() if frame.mask is not None else None,
                    instance_id=frame.instance_id,
                    boxes=list(frame.boxes),
                    points=list(frame.points),
                    gt_boxes_all=list(frame.gt_boxes_all),
                    gt_masks_all=[m.copy() for m in frame.gt_masks_all],
                )
            )
    elif spec.kind == "fft_keep":
        mode = str(spec.params["mode"])
        cutoff = float(spec.params["cutoff"])
        for frame in sample.frames:
            out_frames.append(
                LoadedFrame(
                    image=fft_filter_keep(frame.image, mode, cutoff),
                    mask=frame.mask.copy() if frame.mask is not None else None,
                    instance_id=frame.instance_id,
                    boxes=list(frame.boxes),
                    points=list(frame.points),
                    gt_boxes_all=list(frame.gt_boxes_all),
                    gt_masks_all=[m.copy() for m in frame.gt_masks_all],
                )
            )
    elif spec.kind == "fft_noise":
        mode = str(spec.params["mode"])
        cutoff = float(spec.params["cutoff"])
        epsilon = float(spec.params["epsilon"])
        for frame in sample.frames:
            out_frames.append(
                LoadedFrame(
                    image=fft_band_noise(frame.image, mode, cutoff, epsilon, np_rng),
                    mask=frame.mask.copy() if frame.mask is not None else None,
                    instance_id=frame.instance_id,
                    boxes=list(frame.boxes),
                    points=list(frame.points),
                    gt_boxes_all=list(frame.gt_boxes_all),
                    gt_masks_all=[m.copy() for m in frame.gt_masks_all],
                )
            )
    else:
        raise ValueError(f"Unknown perturbation kind: {spec.kind}")

    return LoadedSample(
        sample_id=sample.sample_id,
        frames=out_frames,
        text_prompt=sample.text_prompt,
        concept=sample.concept,
        concept_present=sample.concept_present,
    )


class Sam2Adapter:
    def __init__(self, config_name: str, checkpoint: str, device: str) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import sam2 as sam2_pkg
        except ImportError as exc:
            raise ImportError("sam2 package is required for SAM2 inference.") from exc
        cfg_path = Path(sam2_pkg.__file__).resolve().parent / config_name
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"SAM2 config not found in package: {config_name} (resolved to {cfg_path})"
            )
        self.model = build_sam2(config_name, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(self.model)

    def predict(self, image: Image.Image, boxes: List[Tuple[float, float, float, float]], points: List[Tuple[float, float, int]]) -> List[MaskPrediction]:
        img_arr = np.asarray(image)
        self.predictor.set_image(img_arr)
        preds: List[MaskPrediction] = []
        point_coords = None
        point_labels = None
        if points:
            point_coords = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
            point_labels = np.array([p[2] for p in points], dtype=np.int32)
        if boxes:
            for box in boxes:
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=np.array(box, dtype=np.float32),
                    multimask_output=True,
                )
                for mask, score in zip(masks, scores):
                    preds.append(MaskPrediction(mask=mask.astype(bool), score=float(score)))
        elif point_coords is not None:
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            for mask, score in zip(masks, scores):
                preds.append(MaskPrediction(mask=mask.astype(bool), score=float(score)))
        return preds


class GroundingDinoAdapter:
    def __init__(self, config_path: str, checkpoint: str, device: str) -> None:
        try:
            from groundingdino.util.inference import load_model, load_image, predict
        except ImportError as exc:
            raise ImportError("groundingdino package is required for GroundingDINO fallback.") from exc
        self.model = load_model(config_path, checkpoint, device=device)
        self.load_image = load_image
        self.predict_fn = predict

    def predict_boxes(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        topk: Optional[int] = None,
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            image.save(tmp.name)
            image_source, image_tensor = self.load_image(tmp.name)
        boxes, logits, _ = self.predict_fn(
            model=self.model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        h, w = image_source.shape[:2]
        out_boxes: List[Tuple[float, float, float, float]] = []
        scores: List[float] = []
        for box, logit in zip(boxes, logits):
            x0, y0, x1, y1 = box
            out_boxes.append((float(x0 * w), float(y0 * h), float(x1 * w), float(y1 * h)))
            scores.append(float(logit))
        if out_boxes:
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            if topk is not None:
                order = order[:topk]
            out_boxes = [out_boxes[i] for i in order]
            scores = [scores[i] for i in order]
        return out_boxes, scores


class GroundingDinoSam2Adapter:
    def __init__(
        self,
        sam2: Sam2Adapter,
        gdino: GroundingDinoAdapter,
        box_threshold: float,
        text_threshold: float,
    ) -> None:
        self.sam2 = sam2
        self.gdino = gdino
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def predict(self, image: Image.Image, text_prompt: str) -> List[MaskPrediction]:
        boxes, scores = self.gdino.predict_boxes(
            image, text_prompt, self.box_threshold, self.text_threshold
        )
        preds: List[MaskPrediction] = []
        for box, score in zip(boxes, scores):
            masks = self.sam2.predict(image, [box], [])
            for mask_pred in masks:
                preds.append(
                    MaskPrediction(
                        mask=mask_pred.mask,
                        score=score * max(mask_pred.score, 1e-6),
                    )
                )
        return preds


def predict_dino_sam2(
    image: Image.Image,
    text_prompt: str,
    sam2: Sam2Adapter,
    gdino: GroundingDinoAdapter,
    box_threshold: float,
    text_threshold: float,
    topk: int,
) -> Tuple[List[MaskPrediction], List[Tuple[float, float, float, float]], List[float]]:
    boxes, scores = gdino.predict_boxes(
        image, text_prompt, box_threshold, text_threshold, topk=topk
    )
    preds: List[MaskPrediction] = []
    for box, score in zip(boxes, scores):
        masks = sam2.predict(image, [box], [])
        for mask_pred in masks:
            preds.append(
                MaskPrediction(
                    mask=mask_pred.mask,
                    score=score * max(mask_pred.score, 1e-6),
                )
            )
    return preds, boxes, scores


SAM3_VISUAL_KEYS = (
    "visual_tokens",
    "vision_tokens",
    "image_tokens",
    "visual_embeddings",
    "vision_embeddings",
    "image_embeddings",
    "visual_feats",
    "image_feats",
    "vision_features",
)
SAM3_TEXT_KEYS = (
    "text_tokens",
    "text_embeddings",
    "text_feats",
    "text_features",
    "text_embeds",
    "language_features",
    "language_embeds",
    "prompt_before_enc",
)
SAM3_FUSED_KEYS = (
    "fused_tokens",
    "fusion_tokens",
    "fused_embeddings",
    "fusion_embeddings",
    "fusion_feats",
    "conditioned_tokens",
    "cross_attended_tokens",
    "encoder_hidden_states",
    "prompt_after_enc",
)


def _as_tensor(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).float()
    return None


def _extract_output_tokens(output: Any, keys: Tuple[str, ...]) -> Optional[torch.Tensor]:
    if isinstance(output, dict):
        for key in keys:
            if key in output:
                return _as_tensor(output[key])
        for value in output.values():
            token = _extract_output_tokens(value, keys)
            if token is not None:
                return token
    if isinstance(output, (tuple, list)):
        for item in output:
            token = _extract_output_tokens(item, keys)
            if token is not None:
                return token
    return None


def extract_sam3_internals(output: Any, model: Any) -> Dict[str, torch.Tensor]:
    internals: Dict[str, torch.Tensor] = {}
    visual = _extract_output_tokens(output, SAM3_VISUAL_KEYS)
    if visual is None:
        for attr in ("vision_encoder_output", "visual_tokens", "image_tokens", "image_embeddings"):
            visual = _as_tensor(getattr(model, attr, None))
            if visual is not None:
                break
    if visual is not None:
        internals["visual_tokens"] = visual
    text = _extract_output_tokens(output, SAM3_TEXT_KEYS)
    if text is None:
        for attr in ("text_encoder_output", "text_tokens", "text_embeddings", "text_features"):
            text = _as_tensor(getattr(model, attr, None))
            if text is not None:
                break
    if text is not None:
        internals["text_tokens"] = text
    fused = _extract_output_tokens(output, SAM3_FUSED_KEYS)
    if fused is None:
        for attr in ("fusion_encoder_output", "fused_tokens", "fusion_tokens", "conditioned_tokens"):
            fused = _as_tensor(getattr(model, attr, None))
            if fused is not None:
                break
    if fused is not None:
        internals["fused_tokens"] = fused
    return internals


class Sam3Adapter:
    def __init__(self, module_path: str, class_name: str, checkpoint: str, device: str) -> None:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        if hasattr(cls, "from_pretrained"):
            self.model = cls.from_pretrained(checkpoint, device=device)
        else:
            self.model = cls(checkpoint=checkpoint, device=device)

    def _call_predict(self, image: Image.Image, text_prompt: str, return_internals: bool) -> Any:
        if return_internals:
            try:
                return self.model.predict(image, text_prompt, return_internals=True)
            except TypeError as exc:
                if "return_internals" not in str(exc):
                    raise
        return self.model.predict(image, text_prompt)

    def _parse_predictions(self, result: Any) -> List[MaskPrediction]:
        masks = None
        scores = None
        if isinstance(result, tuple) and len(result) >= 1:
            masks = result[0]
            scores = result[1] if len(result) > 1 else None
        elif isinstance(result, dict):
            masks = result.get("masks")
            scores = result.get("scores")
        if masks is None:
            return []
        masks_list = list(masks)
        if scores is None:
            scores = [1.0 for _ in masks_list]
        preds: List[MaskPrediction] = []
        for mask, score in zip(masks_list, scores):
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            preds.append(MaskPrediction(mask=mask > 0.5, score=float(score)))
        return preds

    def supports_visual_prompting(self) -> bool:
        return hasattr(self.model, "predict_visual")

    def predict_visual(
        self, image: Image.Image, boxes: List[Tuple[float, float, float, float]], points: List[Tuple[float, float, int]]
    ) -> List[MaskPrediction]:
        if not hasattr(self.model, "predict_visual"):
            return []
        result = self.model.predict_visual(image, boxes, points)
        return self._parse_predictions(result)

    def predict(self, image: Image.Image, text_prompt: str) -> List[MaskPrediction]:
        result = self._call_predict(image, text_prompt, return_internals=False)
        return self._parse_predictions(result)

    def predict_with_internals(
        self, image: Image.Image, text_prompt: str
    ) -> Tuple[List[MaskPrediction], Dict[str, torch.Tensor]]:
        result = self._call_predict(image, text_prompt, return_internals=True)
        preds = self._parse_predictions(result)
        internals = extract_sam3_internals(result, self.model)
        return preds, internals


class Sam3NativeAdapter:
    def __init__(self, checkpoint_path: Optional[Path], device: str) -> None:
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise ImportError("sam3 package is required for SAM3 native inference.") from exc
        ckpt = str(checkpoint_path) if checkpoint_path is not None else None
        model = build_sam3_image_model(
            checkpoint_path=ckpt,
            load_from_HF=checkpoint_path is None,
            device=device,
        )
        self.model = model
        self.processor = Sam3Processor(model)

    def supports_visual_prompting(self) -> bool:
        return self.model.inst_interactive_predictor is not None

    def _run(self, image: Image.Image, text_prompt: str) -> Any:
        state = self.processor.set_image(image)
        return self.processor.set_text_prompt(state=state, prompt=text_prompt)

    def _parse_predictions(self, output: Any) -> List[MaskPrediction]:
        masks = output.get("masks") if isinstance(output, dict) else None
        scores = output.get("scores") if isinstance(output, dict) else None
        if masks is None:
            return []
        if isinstance(masks, torch.Tensor):
            masks_tensor = masks.detach().cpu()
            masks_list = [masks_tensor[i].numpy() for i in range(masks_tensor.shape[0])]
        else:
            masks_list = list(masks)
        if scores is None:
            scores = [1.0 for _ in masks_list]
        elif isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().tolist()
        preds: List[MaskPrediction] = []
        for mask, score in zip(masks_list, scores):
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            preds.append(MaskPrediction(mask=mask > 0.5, score=float(score)))
        return preds

    def predict(self, image: Image.Image, text_prompt: str) -> List[MaskPrediction]:
        output = self._run(image, text_prompt)
        return self._parse_predictions(output)

    def predict_with_internals(
        self, image: Image.Image, text_prompt: str
    ) -> Tuple[List[MaskPrediction], Dict[str, torch.Tensor]]:
        output = self._run(image, text_prompt)
        preds = self._parse_predictions(output)
        internals = extract_sam3_internals(output, self.model)
        return preds, internals

    def predict_visual(
        self, image: Image.Image, boxes: List[Tuple[float, float, float, float]], points: List[Tuple[float, float, int]]
    ) -> List[MaskPrediction]:
        if self.model.inst_interactive_predictor is None:
            return []
        state = self.processor.set_image(image)
        point_coords = None
        point_labels = None
        if points:
            point_coords = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
            point_labels = np.array([p[2] for p in points], dtype=np.int32)
        box_arr = np.array(boxes, dtype=np.float32) if boxes else None
        masks, scores, _ = self.model.predict_inst(
            state,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_arr,
            multimask_output=True,
        )
        if masks is None:
            return []
        if masks.ndim == 4:
            flat_masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
            if scores is None:
                flat_scores = np.ones(flat_masks.shape[0], dtype=np.float32)
            else:
                flat_scores = scores.reshape(-1)
        else:
            flat_masks = masks
            if scores is None:
                flat_scores = np.ones(len(flat_masks), dtype=np.float32)
            else:
                flat_scores = scores
        preds: List[MaskPrediction] = []
        for mask, score in zip(flat_masks, flat_scores):
            preds.append(MaskPrediction(mask=mask.astype(bool), score=float(score)))
        return preds


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return inter / union if union > 0 else 0.0


def box_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def score_entropy(scores: List[float]) -> float:
    if not scores:
        return 0.0
    total = float(sum(max(s, 0.0) for s in scores))
    if total <= 0:
        return 0.0
    ent = 0.0
    for score in scores:
        p = max(score, 0.0) / total
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    return ent


def box_diagnostics(
    boxes: List[Tuple[float, float, float, float]],
    scores: List[float],
    gt_boxes: List[Tuple[float, float, float, float]],
    image_size: Tuple[int, int],
) -> Dict[str, float]:
    best_iou = 0.0
    center_shift = 0.0
    if boxes and gt_boxes:
        best_idx = 0
        best_match = gt_boxes[0]
        for idx, box in enumerate(boxes):
            for gt_box in gt_boxes:
                iou = box_iou(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
                    best_match = gt_box
        diag = math.hypot(image_size[0], image_size[1])
        if diag > 0:
            cx, cy = box_center(boxes[best_idx])
            gx, gy = box_center(best_match)
            center_shift = math.hypot(cx - gx, cy - gy) / diag
    score_margin = 0.0
    if scores:
        score_margin = scores[0] - scores[1] if len(scores) > 1 else scores[0]
    return {
        "best_box_iou_with_gt": best_iou,
        "box_center_shift": center_shift,
        "score_margin": score_margin,
        "score_entropy": score_entropy(scores),
    }


def _aggregate_tokens(tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tokens is None:
        return None
    if tokens.numel() == 0:
        return None
    if tokens.dim() == 1:
        return tokens
    dims = tuple(range(tokens.dim() - 1))
    return tokens.mean(dim=dims)


def cosine_alignment(
    visual_tokens: Optional[torch.Tensor], text_tokens: Optional[torch.Tensor]
) -> Optional[float]:
    vis = _aggregate_tokens(visual_tokens)
    text = _aggregate_tokens(text_tokens)
    if vis is None or text is None:
        return None
    if vis.shape != text.shape:
        return None
    vis = F.normalize(vis, dim=0)
    text = F.normalize(text, dim=0)
    return float(torch.dot(vis, text).item())


def alignment_metrics_from_internals(internals: Dict[str, torch.Tensor]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not internals:
        return metrics
    visual = internals.get("visual_tokens")
    text = internals.get("text_tokens")
    fused = internals.get("fused_tokens")
    pre = cosine_alignment(visual, text)
    if pre is not None:
        metrics["vl_alignment_pre"] = pre
    post = cosine_alignment(fused, text) if fused is not None else None
    if post is not None:
        metrics["vl_alignment_post"] = post
    if "vl_alignment_pre" in metrics and "vl_alignment_post" in metrics:
        metrics["vl_alignment_gain"] = metrics["vl_alignment_post"] - metrics["vl_alignment_pre"]
    return metrics


def _flatten_tokens_for_fft(tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tokens is None:
        return None
    if tokens.numel() == 0:
        return None
    if tokens.dim() == 1:
        return tokens[:, None]
    if tokens.dim() == 2:
        return tokens
    return tokens.reshape(-1, tokens.shape[-1])


def token_fft(tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    flat = _flatten_tokens_for_fft(tokens)
    if flat is None or flat.shape[0] < 2:
        return None
    return torch.fft.rfft(flat, dim=0)


def spectral_drift_from_fft(
    base_fft: Optional[torch.Tensor],
    pert_fft: Optional[torch.Tensor],
    low_frac: float = 0.25,
    high_frac: float = 0.25,
) -> Optional[Dict[str, float]]:
    if base_fft is None or pert_fft is None:
        return None
    n = min(base_fft.shape[0], pert_fft.shape[0])
    c = min(base_fft.shape[1], pert_fft.shape[1])
    if n < 2 or c < 1:
        return None
    base_fft = base_fft[:n, :c]
    pert_fft = pert_fft[:n, :c]
    diff = (pert_fft - base_fft).abs()
    n_freq = diff.shape[0]
    low_bins = max(1, int(n_freq * low_frac))
    high_bins = max(1, int(n_freq * high_frac))
    low = float(diff[:low_bins].mean().item())
    high = float(diff[-high_bins:].mean().item())
    ratio = high / (low + 1e-8)
    return {
        "visual_low_freq_drift": low,
        "visual_high_freq_drift": high,
        "visual_drift_ratio": ratio,
    }


def _boundary_from_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    tmask = torch.from_numpy(mask.astype(np.float32))[None, None]
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32)
    eroded = F.conv2d(tmask, kernel, padding=1)
    eroded = (eroded == 9.0).squeeze(0).squeeze(0)
    boundary = (tmask.squeeze(0).squeeze(0) > 0.5) & (~eroded)
    return boundary.cpu().numpy()


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    tmask = torch.from_numpy(mask.astype(np.float32))[None, None]
    k = 2 * radius + 1
    kernel = torch.ones((1, 1, k, k), dtype=torch.float32)
    dilated = F.conv2d(tmask, kernel, padding=radius)
    return (dilated.squeeze(0).squeeze(0) > 0).cpu().numpy()


def boundary_f_score(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    pred_b = _boundary_from_mask(pred)
    gt_b = _boundary_from_mask(gt)
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0
    pred_d = _dilate_mask(pred_b, tolerance)
    gt_d = _dilate_mask(gt_b, tolerance)
    precision = float((pred_b & gt_d).sum()) / float(pred_b.sum())
    recall = float((gt_b & pred_d).sum()) / float(gt_b.sum())
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def fragmentation(mask: np.ndarray) -> int:
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    mask = mask.astype(bool)
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    count = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
                count += 1
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in (
                        (cy - 1, cx),
                        (cy + 1, cx),
                        (cy, cx - 1),
                        (cy, cx + 1),
                    ):
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
    return count


def extract_gt_masks(mask: Optional[np.ndarray], target_id: Optional[int]) -> Tuple[List[np.ndarray], Optional[int]]:
    if mask is None:
        return [], None
    ids = [int(v) for v in np.unique(mask) if v != 0]
    if not ids:
        return [], None
    masks = [(mask == v) for v in ids]
    if target_id is None:
        return masks, None
    if target_id in ids:
        return masks, ids.index(target_id)
    return masks, None


def choose_best_prediction(preds: List[MaskPrediction]) -> Optional[np.ndarray]:
    if not preds:
        return None
    best = max(preds, key=lambda p: p.score)
    return best.mask


def choose_best_pred(preds: List[MaskPrediction]) -> Optional[MaskPrediction]:
    if not preds:
        return None
    return max(preds, key=lambda p: p.score)


def best_iou_match(
    pred_mask: np.ndarray, gt_masks: List[np.ndarray]
) -> Tuple[float, Optional[np.ndarray]]:
    best_iou = 0.0
    best_gt = None
    for gt in gt_masks:
        iou = mask_iou(pred_mask, gt)
        if iou > best_iou:
            best_iou = iou
            best_gt = gt
    return best_iou, best_gt


def grounding_eval(
    preds: List[MaskPrediction],
    gt_masks_all: List[np.ndarray],
    concept_present: Optional[bool],
    iou_threshold: float,
) -> Tuple[Dict[str, float], Optional[MaskPrediction], Optional[np.ndarray], float]:
    has_pred = bool(preds)
    if concept_present is False:
        return (
            {
                "concept_recall": 0.0,
                "false_positive": 1.0 if has_pred else 0.0,
                "wrong_instance_rate": 0.0,
                "no_prediction_rate": 0.0 if has_pred else 1.0,
                "presence_error": 1.0 if has_pred else 0.0,
            },
            None,
            None,
            0.0,
        )
    if not gt_masks_all:
        return (
            {
                "concept_recall": 0.0,
                "false_positive": 1.0 if has_pred else 0.0,
                "wrong_instance_rate": 0.0,
                "no_prediction_rate": 0.0 if has_pred else 1.0,
                "presence_error": 1.0 if has_pred else 0.0,
            },
            None,
            None,
            0.0,
        )
    best_pred = choose_best_pred(preds)
    best_iou = 0.0
    best_gt = None
    if best_pred is not None:
        best_iou, best_gt = best_iou_match(best_pred.mask, gt_masks_all)
    recall = 1.0 if best_iou >= iou_threshold else 0.0
    wrong_instance = 1.0 if has_pred and recall == 0.0 else 0.0
    false_positive = 0.0
    if has_pred:
        fp_count = 0
        for pred in preds:
            max_iou = max(mask_iou(pred.mask, gt) for gt in gt_masks_all)
            if max_iou < iou_threshold:
                fp_count += 1
        false_positive = fp_count / float(len(preds))
    presence_error = 1.0 - recall
    return (
        {
            "concept_recall": recall,
            "false_positive": false_positive,
            "wrong_instance_rate": wrong_instance,
            "no_prediction_rate": 0.0 if has_pred else 1.0,
            "presence_error": presence_error,
        },
        best_pred,
        best_gt,
        best_iou,
    )


def concept_metrics(
    preds: List[MaskPrediction],
    gt_masks: List[np.ndarray],
    target_idx: Optional[int],
    concept_present: Optional[bool],
    iou_threshold: float,
) -> Dict[str, float]:
    if concept_present is False:
        has_pred = bool(preds)
        return {
            "concept_recall": 0.0,
            "false_positive": 1.0 if has_pred else 0.0,
            "wrong_instance": 0.0,
            "presence_error": 1.0 if has_pred else 0.0,
        }
    if not gt_masks:
        return {
            "concept_recall": 0.0,
            "false_positive": 1.0 if preds else 0.0,
            "wrong_instance": 0.0,
            "presence_error": 1.0 if preds else 0.0,
        }

    target_mask = gt_masks[target_idx] if target_idx is not None else gt_masks[0]
    best_target = 0.0
    best_other = 0.0
    for pred in preds:
        ious = [mask_iou(pred.mask, gt) for gt in gt_masks]
        best_target = max(best_target, mask_iou(pred.mask, target_mask))
        if target_idx is not None:
            for idx, val in enumerate(ious):
                if idx != target_idx:
                    best_other = max(best_other, val)
    recall = 1.0 if best_target >= iou_threshold else 0.0
    wrong_instance = 1.0 if recall == 0.0 and best_other >= iou_threshold else 0.0
    false_positive = 0.0
    if preds:
        fp_count = 0
        for pred in preds:
            max_iou = max(mask_iou(pred.mask, gt) for gt in gt_masks)
            if max_iou < iou_threshold:
                fp_count += 1
        false_positive = fp_count / float(len(preds))
    presence_error = 1.0 - recall
    return {
        "concept_recall": recall,
        "false_positive": false_positive,
        "wrong_instance": wrong_instance,
        "presence_error": presence_error,
    }


def compute_video_metrics(
    pred_masks: List[Optional[np.ndarray]],
    gt_masks: List[Optional[np.ndarray]],
) -> Dict[str, float]:
    if not pred_masks or not gt_masks:
        return {"j": 0.0, "f": 0.0, "id_stability": 0.0}
    j_vals: List[float] = []
    f_vals: List[float] = []
    id_vals: List[float] = []
    for pred, gt in zip(pred_masks, gt_masks):
        if pred is None or gt is None:
            j_vals.append(0.0)
            f_vals.append(0.0)
        else:
            j_vals.append(mask_iou(pred, gt))
            f_vals.append(boundary_f_score(pred, gt))
    for idx in range(1, len(pred_masks)):
        prev = pred_masks[idx - 1]
        cur = pred_masks[idx]
        if prev is None or cur is None:
            id_vals.append(0.0)
        else:
            id_vals.append(mask_iou(prev, cur))
    return {
        "j": float(np.mean(j_vals)) if j_vals else 0.0,
        "f": float(np.mean(f_vals)) if f_vals else 0.0,
        "id_stability": float(np.mean(id_vals)) if id_vals else 0.0,
    }


def overlay_mask(image: Image.Image, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> Image.Image:
    if mask is None:
        return image.copy()
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    base = np.asarray(image).astype(np.float32)
    overlay = base.copy()
    overlay[mask.astype(bool)] = np.array(color, dtype=np.float32)
    boundary = _boundary_from_mask(mask)
    overlay[boundary] = np.array(color, dtype=np.float32)
    mixed = base * (1.0 - alpha) + overlay * alpha
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.ndim > 2:
        pred = np.squeeze(pred)
    if gt.ndim > 2:
        gt = np.squeeze(gt)
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    h, w = pred.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    out[tp] = (0, 200, 0)
    out[fp] = (200, 0, 0)
    out[fn] = (0, 120, 255)
    return out


def save_panel(
    out_path: Path,
    base_image: Image.Image,
    pert_image: Image.Image,
    gt_mask: Optional[np.ndarray],
    sam2_mask: Optional[np.ndarray],
    sam3_mask: Optional[np.ndarray],
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    axes[0].imshow(base_image)
    axes[0].set_title("Base")
    axes[1].imshow(pert_image)
    axes[1].set_title("Perturbed")
    if gt_mask is not None:
        axes[2].imshow(overlay_mask(pert_image, gt_mask, (0, 200, 0), 0.5))
    else:
        axes[2].imshow(pert_image)
    axes[2].set_title("GT")
    if sam2_mask is not None:
        axes[3].imshow(overlay_mask(pert_image, sam2_mask, (0, 180, 255), 0.5))
    else:
        axes[3].imshow(pert_image)
    axes[3].set_title("SAM2")
    if sam3_mask is not None:
        axes[4].imshow(overlay_mask(pert_image, sam3_mask, (255, 160, 0), 0.5))
    else:
        axes[4].imshow(pert_image)
    axes[4].set_title("SAM3")
    if gt_mask is not None and sam3_mask is not None:
        axes[5].imshow(error_map(sam3_mask, gt_mask))
    else:
        axes[5].imshow(pert_image)
    axes[5].set_title("Errors")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


class MetricBook:
    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Dict[Tuple[str, int], List[float]]]] = {}

    def add(self, model: str, metric: str, perturb: str, severity: int, value: float) -> None:
        self.data.setdefault(model, {}).setdefault(metric, {}).setdefault((perturb, severity), []).append(value)

    def summary(self) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
        out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
        for model, metrics in self.data.items():
            out[model] = {}
            for metric, buckets in metrics.items():
                out[model][metric] = {}
                for (pert, severity), values in buckets.items():
                    key = f"{pert}|sev{severity}"
                    out[model][metric][key] = {
                        "mean": float(np.mean(values)) if values else 0.0,
                        "count": len(values),
                    }
        return out


MODEL_LABELS = {
    "SAM2": "SAM2 (PVS)",
    "SAM3": "SAM3 (PCS)",
    "SAM3_PVS": "SAM3 (PVS)",
    "DINO_SAM2": "DINO->SAM2",
    "SAM2_GTBOX": "SAM2 (GT boxes)",
}

GROUP_LABELS = {
    "natural": "Natural",
    "lowfreq": "Low freq",
    "highfreq": "High freq",
    "lowfreq_keep": "Low-pass keep",
    "highfreq_keep": "High-pass keep",
    "lowfreq_noise": "Low-band noise",
    "highfreq_noise": "High-band noise",
    "allfreq_noise": "All-band noise",
}

METRIC_LABELS = {
    "miou": "mIoU",
    "boundary_f": "Boundary F",
    "fragmentation": "Fragmentation",
    "concept_recall": "Concept recall",
    "false_positive": "False positive rate",
    "wrong_instance_rate": "Wrong-instance rate",
    "no_prediction_rate": "No-prediction rate",
    "presence_error": "Presence error",
    "conditional_miou": "Conditional mIoU",
    "conditional_boundary_f": "Conditional boundary F",
    "conditional_fragmentation": "Conditional fragmentation",
    "vl_alignment_pre": "VL alignment (pre)",
    "vl_alignment_post": "VL alignment (post)",
    "vl_alignment_gain": "VL alignment gain",
    "vl_alignment_pre_drift": "VL alignment drift (pre)",
    "vl_alignment_post_drift": "VL alignment drift (post)",
    "vl_alignment_gain_drift": "VL alignment gain drift",
    "visual_low_freq_drift": "Visual drift (low freq)",
    "visual_high_freq_drift": "Visual drift (high freq)",
    "visual_drift_ratio": "Visual drift ratio (high/low)",
    "best_box_iou_with_gt": "Best box IoU vs GT",
    "box_center_shift": "Box center shift",
    "score_margin": "Score margin",
    "score_entropy": "Score entropy",
    "sam3_score_margin": "SAM3 score margin",
    "sam3_score_entropy": "SAM3 score entropy",
    "j": "J (IoU)",
    "f": "F (Boundary)",
    "id_stability": "ID stability",
}


def display_model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def display_group_label(group: str) -> str:
    return GROUP_LABELS.get(group, group.replace("_", " ").title())


def display_metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def plot_metric_trends(
    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    out_dir: Path,
    metric: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 5))
    has_data = False
    for model, metrics in summary.items():
        entries = metrics.get(metric, {})
        by_sev: Dict[int, List[float]] = {}
        for key, stats in entries.items():
            if "sev" not in key:
                continue
            sev = int(key.split("sev", maxsplit=1)[1])
            by_sev.setdefault(sev, []).append(stats["mean"])
        if not by_sev:
            continue
        xs = sorted(by_sev.keys())
        ys = [float(np.mean(by_sev[sev])) for sev in xs]
        plt.plot(xs, ys, marker="o", label=display_model_label(model))
        has_data = True
    if not has_data:
        plt.close()
        return
    plt.title(title or f"{display_metric_label(metric)} vs Severity")
    plt.xlabel("Severity")
    plt.ylabel(display_metric_label(metric))
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_severity.png")
    plt.close()


def parse_summary_key(key: str) -> Tuple[Optional[str], Optional[int]]:
    if "|sev" not in key:
        return None, None
    pert, sev_str = key.rsplit("|sev", maxsplit=1)
    try:
        sev = int(sev_str)
    except ValueError:
        return None, None
    return pert, sev


def build_group_map(
    perturbations: List[PerturbationSpec],
    split_frequency: bool = False,
) -> Dict[str, str]:
    groups: Dict[str, str] = {}
    for spec in perturbations:
        if spec.family == "natural":
            groups[spec.name] = "natural"
        elif spec.family == "frequency":
            mode = str(spec.params.get("mode", "all"))
            if split_frequency:
                if spec.kind == "fft_keep":
                    group = "lowfreq_keep" if mode == "low" else "highfreq_keep"
                else:
                    if mode == "low":
                        group = "lowfreq_noise"
                    elif mode == "high":
                        group = "highfreq_noise"
                    else:
                        group = "allfreq_noise"
                groups[spec.name] = group
            else:
                groups[spec.name] = "lowfreq" if mode == "low" else "highfreq"
        else:
            groups[spec.name] = spec.family
    return groups


def compute_relative_drop(
    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    base_key: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for model, metrics in summary.items():
        for metric, entries in metrics.items():
            base = entries.get(base_key, {})
            if not base:
                continue
            base_mean = base.get("mean")
            if base_mean is None:
                continue
            for key, stats in entries.items():
                if key == base_key:
                    continue
                out.setdefault(model, {}).setdefault(metric, {})[key] = {
                    "mean": float(base_mean) - float(stats.get("mean", 0.0)),
                    "count": int(stats.get("count", 0)),
                }
    return out


def cohens_d(sample_a: List[float], sample_b: List[float]) -> Optional[float]:
    if not sample_a or not sample_b:
        return None
    arr_a = np.asarray(sample_a, dtype=np.float32)
    arr_b = np.asarray(sample_b, dtype=np.float32)
    mean_a = float(arr_a.mean())
    mean_b = float(arr_b.mean())
    var_a = float(arr_a.var(ddof=1)) if arr_a.size > 1 else 0.0
    var_b = float(arr_b.var(ddof=1)) if arr_b.size > 1 else 0.0
    denom = arr_a.size + arr_b.size - 2
    if denom <= 0:
        return 0.0
    pooled = math.sqrt(((arr_a.size - 1) * var_a + (arr_b.size - 1) * var_b) / denom)
    if pooled == 0:
        return 0.0
    return (mean_a - mean_b) / pooled


def compute_drift_effect_sizes(
    metric_book: MetricBook,
    base_key: Tuple[str, int],
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for model, metrics in metric_book.data.items():
        for metric, buckets in metrics.items():
            if "drift" not in metric:
                continue
            base_vals = buckets.get(base_key, [])
            if not base_vals:
                continue
            for (pert, sev), values in buckets.items():
                if (pert, sev) == base_key:
                    continue
                effect = cohens_d(values, base_vals)
                if effect is None:
                    continue
                key = f"{pert}|sev{sev}"
                out.setdefault(model, {}).setdefault(metric, {})[key] = {
                    "mean": float(effect),
                    "count": len(values),
                }
    return out

def extract_metric_values(
    metric_book: MetricBook,
    model: str,
    metric: str,
    perturbation_filter: Optional[Callable[[str, int], bool]] = None,
) -> List[float]:
    """Extract all values for a given model and metric, optionally filtered."""
    values: List[float] = []
    model_data = metric_book.data.get(model, {})
    metric_data = model_data.get(metric, {})
    
    for (pert_name, severity), vals in metric_data.items():
        if perturbation_filter is None or perturbation_filter(pert_name, severity):
            values.extend(vals)
    
    return values


def _compute_drop_series(
    metric_book: MetricBook,
    model: str,
    metric: str,
    base_key: Tuple[str, int] = ("Base", 0),
) -> Tuple[Optional[float], Dict[Tuple[str, int], List[float]]]:
    metric_data = metric_book.data.get(model, {}).get(metric, {})
    base_vals = metric_data.get(base_key, [])
    if not base_vals:
        return None, {}
    base_mean = float(np.mean(base_vals))
    drops: Dict[Tuple[str, int], List[float]] = {}
    for key, vals in metric_data.items():
        if key == base_key or not vals:
            continue
        drops[key] = [base_mean - v for v in vals]
    return base_mean, drops


def _paired_effect_size(values_a: List[float], values_b: List[float]) -> float:
    diffs = [a - b for a, b in zip(values_a, values_b)]
    diff_std = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    return float(np.mean(diffs)) / diff_std if diff_std else 0.0


def test_differential_sensitivity(
    metric_book: MetricBook,
    sam2_model: str = "SAM2",
    sam3_model: str = "SAM3",
    metric: str = "miou",
    perturbation_family: str = "highfreq",
    group_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Test H1: SAM3 degrades more than SAM2 under high-frequency perturbations.
    
    Returns statistical test results with p-value and effect size.
    """
    from scipy import stats
    
    def is_target_perturbation(pert_name: str, severity: int) -> bool:
        if group_map is None:
            return perturbation_family.lower() in pert_name.lower()
        return group_map.get(pert_name) == perturbation_family and severity > 0
    
    sam2_base = extract_metric_values(
        metric_book, sam2_model, metric, lambda p, s: p == "Base" and s == 0
    )
    sam3_base = extract_metric_values(
        metric_book, sam3_model, metric, lambda p, s: p == "Base" and s == 0
    )
    
    if not sam2_base or not sam3_base:
        return {
            "hypothesis_supported": False,
            "reason": "insufficient_data",
            "sam2_n": 0,
            "sam3_n": 0,
            "n_pairs": 0,
        }
    
    sam2_base_mean = float(np.mean(sam2_base))
    sam3_base_mean = float(np.mean(sam3_base))

    sam2_metric = metric_book.data.get(sam2_model, {}).get(metric, {})
    sam3_metric = metric_book.data.get(sam3_model, {}).get(metric, {})
    sam2_drops_by_key: Dict[Tuple[str, int], float] = {}
    sam3_drops_by_key: Dict[Tuple[str, int], float] = {}

    for (pert_name, severity), vals in sam2_metric.items():
        if not vals or not is_target_perturbation(pert_name, severity):
            continue
        sam2_drops_by_key[(pert_name, severity)] = sam2_base_mean - float(np.mean(vals))

    for (pert_name, severity), vals in sam3_metric.items():
        if not vals or not is_target_perturbation(pert_name, severity):
            continue
        sam3_drops_by_key[(pert_name, severity)] = sam3_base_mean - float(np.mean(vals))

    shared_keys = sorted(set(sam2_drops_by_key) & set(sam3_drops_by_key))
    if len(shared_keys) < 2:
        return {
            "hypothesis_supported": False,
            "reason": "insufficient_data",
            "sam2_n": len(shared_keys),
            "sam3_n": len(shared_keys),
            "n_pairs": len(shared_keys),
        }

    sam2_drops = [sam2_drops_by_key[key] for key in shared_keys]
    sam3_drops = [sam3_drops_by_key[key] for key in shared_keys]
    t_stat, p_value = stats.ttest_rel(sam3_drops, sam2_drops)
    effect_size = _paired_effect_size(sam3_drops, sam2_drops)

    sam2_drop = float(np.mean(sam2_drops))
    sam3_drop = float(np.mean(sam3_drops))

    return {
        "hypothesis_supported": bool(sam3_drop > sam2_drop and p_value < 0.05),
        "sam2_drop": sam2_drop,
        "sam3_drop": sam3_drop,
        "differential_drop": sam3_drop - sam2_drop,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": effect_size,
        "sam2_n": len(sam2_drops),
        "sam3_n": len(sam3_drops),
        "n_pairs": len(shared_keys),
        "interpretation": _interpret_effect_size(effect_size),
    }


def test_differential_sensitivity_overall(
    metric_book: MetricBook,
    sam2_model: str = "SAM2",
    sam3_model: str = "SAM3",
    metric: str = "miou",
) -> Dict[str, Any]:
    from scipy import stats

    sam2_base = extract_metric_values(
        metric_book, sam2_model, metric, lambda p, s: p == "Base" and s == 0
    )
    sam3_base = extract_metric_values(
        metric_book, sam3_model, metric, lambda p, s: p == "Base" and s == 0
    )
    if not sam2_base or not sam3_base:
        return {
            "hypothesis_supported": False,
            "reason": "insufficient_data",
            "sam2_n": 0,
            "sam3_n": 0,
            "n_pairs": 0,
        }

    sam2_base_mean = float(np.mean(sam2_base))
    sam3_base_mean = float(np.mean(sam3_base))
    sam2_metric = metric_book.data.get(sam2_model, {}).get(metric, {})
    sam3_metric = metric_book.data.get(sam3_model, {}).get(metric, {})
    sam2_drops_by_key: Dict[Tuple[str, int], float] = {}
    sam3_drops_by_key: Dict[Tuple[str, int], float] = {}

    for (pert_name, severity), vals in sam2_metric.items():
        if (pert_name, severity) == ("Base", 0) or not vals:
            continue
        sam2_drops_by_key[(pert_name, severity)] = sam2_base_mean - float(np.mean(vals))

    for (pert_name, severity), vals in sam3_metric.items():
        if (pert_name, severity) == ("Base", 0) or not vals:
            continue
        sam3_drops_by_key[(pert_name, severity)] = sam3_base_mean - float(np.mean(vals))

    shared_keys = sorted(set(sam2_drops_by_key) & set(sam3_drops_by_key))
    if len(shared_keys) < 2:
        return {
            "hypothesis_supported": False,
            "reason": "insufficient_data",
            "sam2_n": len(shared_keys),
            "sam3_n": len(shared_keys),
            "n_pairs": len(shared_keys),
        }

    sam2_drops = [sam2_drops_by_key[key] for key in shared_keys]
    sam3_drops = [sam3_drops_by_key[key] for key in shared_keys]
    t_stat, p_value = stats.ttest_rel(sam3_drops, sam2_drops)
    effect_size = _paired_effect_size(sam3_drops, sam2_drops)

    sam2_drop = float(np.mean(sam2_drops))
    sam3_drop = float(np.mean(sam3_drops))
    return {
        "hypothesis_supported": bool(sam3_drop > sam2_drop and p_value < 0.05),
        "sam2_drop": sam2_drop,
        "sam3_drop": sam3_drop,
        "differential_drop": sam3_drop - sam2_drop,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": effect_size,
        "sam2_n": len(sam2_drops),
        "sam3_n": len(sam3_drops),
        "n_pairs": len(shared_keys),
        "interpretation": _interpret_effect_size(effect_size),
    }


def test_differential_sensitivity_by_family(
    metric_book: MetricBook,
    families: List[str],
    group_map: Dict[str, str],
    sam2_model: str = "SAM2",
    sam3_model: str = "SAM3",
    metric: str = "miou",
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for family in families:
        results[family] = test_differential_sensitivity(
            metric_book,
            sam2_model=sam2_model,
            sam3_model=sam3_model,
            metric=metric,
            perturbation_family=family,
            group_map=group_map,
        )
    return results


def test_differential_sensitivity_by_perturbation(
    metric_book: MetricBook,
    group_map: Dict[str, str],
    sam2_model: str = "SAM2",
    sam3_model: str = "SAM3",
    metric: str = "miou",
) -> Dict[str, Dict[str, Any]]:
    from scipy import stats

    _, sam2_drops = _compute_drop_series(metric_book, sam2_model, metric)
    _, sam3_drops = _compute_drop_series(metric_book, sam3_model, metric)
    results: Dict[str, Dict[str, Any]] = {}

    shared_keys = sorted(set(sam2_drops) & set(sam3_drops), key=lambda x: (x[0], x[1]))
    for pert, sev in shared_keys:
        drops2 = sam2_drops.get((pert, sev), [])
        drops3 = sam3_drops.get((pert, sev), [])
        key_str = f"{pert}|sev{sev}"
        family = group_map.get(pert)
        if len(drops2) < 2 or len(drops3) < 2:
            results[key_str] = {
                "hypothesis_supported": False,
                "reason": "insufficient_data",
                "sam2_n": len(drops2),
                "sam3_n": len(drops3),
                "family": family,
            }
            continue
        if len(drops2) != len(drops3):
            results[key_str] = {
                "hypothesis_supported": False,
                "reason": "length_mismatch",
                "sam2_n": len(drops2),
                "sam3_n": len(drops3),
                "family": family,
            }
            continue
        t_stat, p_value = stats.ttest_rel(drops3, drops2)
        sam2_drop = float(np.mean(drops2))
        sam3_drop = float(np.mean(drops3))
        effect_size = _paired_effect_size(drops3, drops2)
        results[key_str] = {
            "hypothesis_supported": bool(sam3_drop > sam2_drop and p_value < 0.05),
            "sam2_drop": sam2_drop,
            "sam3_drop": sam3_drop,
            "differential_drop": sam3_drop - sam2_drop,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": effect_size,
            "sam2_n": len(drops2),
            "sam3_n": len(drops3),
            "family": family,
        }
    return results


def test_alignment_performance_correlation(
    metric_book: MetricBook,
    model: str = "SAM3",
    alignment_metric: str = "vl_alignment_pre_drift",
    performance_metric: str = "miou",
) -> Dict[str, Any]:
    """
    Test H2: Vision-language alignment drift correlates with performance drop.
    
    Returns correlation coefficient and p-value.
    """
    from scipy import stats
    
    model_data = metric_book.data.get(model, {})
    perf_data = model_data.get(performance_metric, {})
    align_data = model_data.get(alignment_metric, {})
    
    base_perf = []
    for (pert, sev), vals in perf_data.items():
        if pert == "Base" and sev == 0:
            base_perf.extend(vals)

    if not base_perf:
        return {
            "hypothesis_supported": False,
            "reason": "insufficient_data",
            "n_pairs": 0,
        }

    base_perf_mean = float(np.mean(base_perf))

    # Match alignment drift to performance drop for same perturbations
    paired_align: List[float] = []
    paired_drop: List[float] = []
    
    for (pert, sev), align_vals in align_data.items():
        if sev == 0:
            continue
        perf_vals = perf_data.get((pert, sev))
        if not align_vals or not perf_vals:
            continue
        paired_align.append(float(np.mean(align_vals)))
        paired_drop.append(base_perf_mean - float(np.mean(perf_vals)))
    
    if len(paired_align) < 10:
        return {
            "hypothesis_supported": False,
            "reason": "insufficient_data",
            "n": len(paired_align),
        }
    
    # Correlation: higher alignment drift should correlate with larger drop
    r_pearson, p_pearson = stats.pearsonr(paired_align, paired_drop)
    r_spearman, p_spearman = stats.spearmanr(paired_align, paired_drop)
    
    # We expect positive correlation (more drift = larger drop)
    supported = bool(r_pearson > 0.6 and p_pearson < 0.01)
    
    return {
        "hypothesis_supported": supported,
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(r_spearman),
        "spearman_p": float(p_spearman),
        "n_pairs": len(paired_align),
        "interpretation": _interpret_correlation(r_pearson),
    }


def test_frequency_specificity(
    metric_book: MetricBook,
    model: str = "SAM3",
    drift_metric: str = "visual_drift_ratio",
    group_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Test H3: High-frequency perturbations cause greater drift than low-frequency.
    
    Compares visual_drift_ratio (high_freq_drift / low_freq_drift) across families.
    """
    from scipy import stats
    
    highfreq_drifts = extract_metric_values(
        metric_book, model, drift_metric,
        lambda p, s: group_map.get(p) == "highfreq" if group_map else "High" in p
    )
    
    lowfreq_drifts = extract_metric_values(
        metric_book, model, drift_metric,
        lambda p, s: group_map.get(p) == "lowfreq" if group_map else "Low" in p
    )
    
    if not highfreq_drifts or not lowfreq_drifts:
        return {
            "hypothesis_supported": False,
            "reason": "insufficient_data",
        }
    
    high_mean = float(np.mean(highfreq_drifts))
    low_mean = float(np.mean(lowfreq_drifts))
    
    # Test: high-freq drift ratio should be > low-freq drift ratio
    t_stat, p_value = stats.ttest_ind(highfreq_drifts, lowfreq_drifts)
    
    return {
        "hypothesis_supported": bool(high_mean > low_mean and p_value < 0.05),
        "high_freq_drift_ratio": high_mean,
        "low_freq_drift_ratio": low_mean,
        "difference": high_mean - low_mean,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d(highfreq_drifts, lowfreq_drifts) or 0.0,
    }


def test_nonlocal_drift(
    drift_effect_sizes: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    model: str = "SAM3",
    comparison_model: str = "SAM2",
    drift_metric: str = "vl_alignment_pre_drift",
) -> Dict[str, Any]:
    """
    Test H4: SAM3 shows more non-local drift (Cohen's d closer to 0) than SAM2.
    
    From VLM paper: d ≈ -0.3 indicates non-local drift (comparable to inter-image).
    """
    sam3_data = drift_effect_sizes.get(model, {}).get(drift_metric, {})
    sam2_data = drift_effect_sizes.get(comparison_model, {}).get(drift_metric, {})
    
    if not sam3_data:
        return {
            "hypothesis_supported": False,
            "reason": "no_sam3_drift_data",
        }
    
    # Extract Cohen's d values
    sam3_cohens_d = [v["mean"] for v in sam3_data.values()]
    
    if not sam3_cohens_d:
        return {
            "hypothesis_supported": False,
            "reason": "insufficient_data",
        }
    
    sam3_d_mean = float(np.mean(sam3_cohens_d))
    
    # From VLM paper Table 4:
    # Local drift: d < -3.0 (perturbation drift << inter-image drift)
    # Non-local drift: d ≈ -0.3 (perturbation drift ~ inter-image drift)
    
    # SAM3 should show d closer to 0 (more non-local)
    # SAM2 should show d more negative (more local)
    
    is_nonlocal = sam3_d_mean > -1.0  # Threshold for "non-local"
    
    result = {
        "hypothesis_supported": is_nonlocal,
        "sam3_cohens_d_mean": sam3_d_mean,
        "sam3_cohens_d_values": sam3_cohens_d,
        "interpretation": "non-local" if is_nonlocal else "local",
    }
    
    if sam2_data:
        sam2_cohens_d = [v["mean"] for v in sam2_data.values()]
        sam2_d_mean = float(np.mean(sam2_cohens_d))
        result["sam2_cohens_d_mean"] = sam2_d_mean
        result["sam3_more_nonlocal"] = sam3_d_mean > sam2_d_mean
    
    return result


def _interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def _interpret_correlation(r: float) -> str:
    """Interpret correlation coefficient."""
    abs_r = abs(r)
    if abs_r < 0.3:
        return "weak"
    elif abs_r < 0.7:
        return "moderate"
    else:
        return "strong"


def analyze_hypothesis(
    metric_book: MetricBook,
    summary: Dict[str, Any],
    drift_effect_sizes: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    group_map: Dict[str, str],
    families: Optional[List[str]] = None,
    alignment_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Complete hypothesis evaluation with all statistical tests.
    
    Returns verdict on whether hypothesis is supported by data.
    """
    
    # Test 1: Differential Sensitivity
    print("[test] Running differential sensitivity test (SAM3 vs SAM2)...")
    diff_sens_highfreq = test_differential_sensitivity(
        metric_book, "SAM2", "SAM3", "miou", "highfreq", group_map
    )
    diff_sens_lowfreq = test_differential_sensitivity(
        metric_book, "SAM2", "SAM3", "miou", "lowfreq", group_map
    )
    diff_sens_natural = test_differential_sensitivity(
        metric_book, "SAM2", "SAM3", "miou", "natural", group_map
    )

    family_list = families or ["natural", "lowfreq", "highfreq"]
    diff_sens_by_family = test_differential_sensitivity_by_family(
        metric_book,
        families=family_list,
        group_map=group_map,
        sam2_model="SAM2",
        sam3_model="SAM3",
        metric="miou",
    )
    diff_sens_by_pert = test_differential_sensitivity_by_perturbation(
        metric_book,
        group_map=group_map,
        sam2_model="SAM2",
        sam3_model="SAM3",
        metric="miou",
    )
    diff_sens_overall = test_differential_sensitivity_overall(
        metric_book,
        sam2_model="SAM2",
        sam3_model="SAM3",
        metric="miou",
    )
    
    # Test 2: Alignment-Performance Correlation
    print("[test] Running alignment-performance correlation test...")
    align_corr_pre = test_alignment_performance_correlation(
        metric_book, "SAM3", "vl_alignment_pre_drift", "miou"
    )
    align_corr_post = test_alignment_performance_correlation(
        metric_book, "SAM3", "vl_alignment_post_drift", "miou"
    )
    
    # Test 3: Frequency Specificity
    print("[test] Running frequency specificity test...")
    freq_spec = test_frequency_specificity(metric_book, "SAM3", "visual_drift_ratio", group_map)
    
    # Test 4: Non-local Drift
    print("[test] Running non-local drift test (Cohen's d)...")
    nonlocal_test = test_nonlocal_drift(drift_effect_sizes, "SAM3", "SAM2")
    
    # Overall verdict
    criteria = {
        "differential_sensitivity_highfreq": bool(
            diff_sens_highfreq.get("hypothesis_supported", False)
        ),
        "alignment_correlation": bool(align_corr_pre.get("hypothesis_supported", False)),
        "frequency_specificity": bool(freq_spec.get("hypothesis_supported", False)),
        "nonlocal_drift": bool(nonlocal_test.get("hypothesis_supported", False)),
    }
    
    supported = sum(criteria.values()) >= 3  # At least 3 out of 4 tests pass
    
    return {
        "hypothesis_supported": supported,
        "criteria_passed": int(sum(criteria.values())),
        "criteria_total": len(criteria),
        "tests": {
            "differential_sensitivity": {
                "highfreq": diff_sens_highfreq,
                "lowfreq": diff_sens_lowfreq,
                "natural": diff_sens_natural,
                "overall": diff_sens_overall,
            },
            "differential_sensitivity_by_family": diff_sens_by_family,
            "differential_sensitivity_by_perturbation": diff_sens_by_pert,
            "alignment_correlation": {
                "pre_fusion": align_corr_pre,
                "post_fusion": align_corr_post,
            },
            "frequency_specificity": freq_spec,
            "nonlocal_drift": nonlocal_test,
        },
        "summary": _generate_summary(
            criteria,
            diff_sens_highfreq,
            align_corr_pre,
            freq_spec,
            diff_sens_overall,
            alignment_status,
        ),
    }


def _generate_summary(
    criteria: Dict[str, bool],
    diff_sens: Dict[str, Any],
    align_corr: Dict[str, Any],
    freq_spec: Dict[str, Any],
    overall_sens: Optional[Dict[str, Any]] = None,
    alignment_status: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate human-readable summary of results."""
    lines = []
    
    if criteria["differential_sensitivity_highfreq"]:
        lines.append(
            f"✓ SAM3 degrades {diff_sens['differential_drop']:.3f} more than SAM2 "
            f"under high-freq perturbations (p={diff_sens['p_value']:.4f})"
        )
    else:
        lines.append("✗ No significant differential sensitivity detected")
    
    if criteria["alignment_correlation"]:
        lines.append(
            f"✓ Alignment drift correlates with performance "
            f"(r={align_corr['pearson_r']:.3f}, p={align_corr['pearson_p']:.4f})"
        )
    else:
        lines.append("✗ Weak alignment-performance correlation")
    
    if criteria["frequency_specificity"]:
        lines.append(
            f"✓ High-freq drift ratio ({freq_spec['high_freq_drift_ratio']:.3f}) "
            f"> low-freq ({freq_spec['low_freq_drift_ratio']:.3f})"
        )
    else:
        lines.append("✗ No frequency-specific drift pattern")

    if overall_sens:
        if overall_sens.get("reason") == "insufficient_data":
            lines.append("✗ Overall sensitivity not tested (insufficient data)")
        elif overall_sens.get("hypothesis_supported"):
            lines.append(
                f"✓ Overall sensitivity: SAM3 drops more than SAM2 "
                f"(p={overall_sens['p_value']:.4f})"
            )
        else:
            lines.append(
                f"✗ Overall sensitivity not supported (p={overall_sens['p_value']:.4f})"
            )

    if alignment_status is not None:
        if not alignment_status.get("sam3_alignment_metrics_available", True):
            lines.append("✗ Alignment metrics unavailable (SAM3 internals missing tokens)")
        else:
            lines.append("✓ Alignment metrics available")
    
    return "\n".join(lines)

def plot_alignment_vs_performance_scatter(
    metric_book: MetricBook,
    out_dir: Path,
    model: str = "SAM3",
) -> None:
    """Create scatter plot: alignment drift (x) vs performance drop (y)."""
    from scipy import stats
    
    model_data = metric_book.data.get(model, {})
    align_data = model_data.get("vl_alignment_pre_drift", {})
    miou_data = model_data.get("miou", {})
    
    # Get baseline mIoU
    base_miou = []
    for (pert, sev), vals in miou_data.items():
        if pert == "Base" and sev == 0:
            base_miou.extend(vals)
    
    if not base_miou:
        return
    
    base_miou_mean = float(np.mean(base_miou))
    
    # Collect paired data
    align_drifts: List[float] = []
    perf_drops: List[float] = []
    
    for (pert, sev), align_vals in align_data.items():
        if sev == 0:
            continue
        if (pert, sev) not in miou_data:
            continue
        
        miou_vals = miou_data[(pert, sev)]
        if not align_vals or not miou_vals:
            continue
        align_drifts.append(float(np.mean(align_vals)))
        perf_drops.append(base_miou_mean - float(np.mean(miou_vals)))
    
    if len(align_drifts) < 2:
        return
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(align_drifts, perf_drops, alpha=0.5, s=30)
    
    # Regression line
    z = np.polyfit(align_drifts, perf_drops, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(align_drifts), max(align_drifts), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    # Statistics
    r, p_val = stats.pearsonr(align_drifts, perf_drops)
    plt.text(
        0.05, 0.95,
        f"Pearson r = {r:.3f}\np-value = {p_val:.4f}\nn = {len(align_drifts)}",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.xlabel("Vision-Language Alignment Drift (pre-fusion)")
    plt.ylabel("Performance drop (mIoU)")
    plt.title(f"{display_model_label(model)}: Alignment drift vs performance loss")
    plt.grid(True, alpha=0.3)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{model}_alignment_vs_performance.png", dpi=150)
    plt.close()


def plot_differential_sensitivity_comparison(
    metric_book: MetricBook,
    out_dir: Path,
    group_map: Dict[str, str],
    metric: str = "miou",
) -> None:
    """Bar plot comparing SAM2 vs SAM3 degradation across perturbation families."""
    families = ["natural", "lowfreq", "highfreq"]
    models = ["SAM2", "SAM3"]
    
    # Compute mean drops for each family
    results: Dict[str, Dict[str, float]] = {}
    
    for model in models:
        results[model] = {}
        model_data = metric_book.data.get(model, {})
        metric_data = model_data.get(metric, {})
        
        # Get baseline
        base_vals = []
        for (pert, sev), vals in metric_data.items():
            if pert == "Base" and sev == 0:
                base_vals.extend(vals)
        
        if not base_vals:
            continue
        
        base_mean = float(np.mean(base_vals))
        
        for family in families:
            family_vals = []
            for (pert, sev), vals in metric_data.items():
                if sev == 0:
                    continue
                if group_map.get(pert) == family:
                    family_vals.extend(vals)
            
            if family_vals:
                drop = base_mean - float(np.mean(family_vals))
                results[model][family] = drop
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(families))
    width = 0.35
    
    sam2_drops = [results["SAM2"].get(f, 0) for f in families]
    sam3_drops = [results["SAM3"].get(f, 0) for f in families]
    
    ax.bar(x - width/2, sam2_drops, width, label=display_model_label("SAM2"), alpha=0.8)
    ax.bar(x + width/2, sam3_drops, width, label=display_model_label("SAM3"), alpha=0.8)
    
    ax.set_xlabel("Perturbation family")
    ax.set_ylabel(f"{display_metric_label(metric)} drop")
    ax.set_title("Differential sensitivity: SAM2 vs SAM3")
    ax.set_xticks(x)
    ax.set_xticklabels([display_group_label(f) for f in families])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "differential_sensitivity_comparison.png", dpi=150)
    plt.close()


def plot_differential_sensitivity_per_perturbation(
    perturbation_tests: Dict[str, Dict[str, Any]],
    out_dir: Path,
    metric: str = "miou",
) -> None:
    items_by_family: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for key, stats in perturbation_tests.items():
        if "differential_drop" not in stats:
            continue
        family = stats.get("family") or "other"
        items_by_family.setdefault(family, []).append((key, stats))

    if not items_by_family:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for family, items in items_by_family.items():
        items_sorted = sorted(items, key=lambda x: x[1]["differential_drop"])
        labels = [key for key, _ in items_sorted]
        diffs = [float(stats["differential_drop"]) for _, stats in items_sorted]
        colors = []
        for _, stats in items_sorted:
            p_val = stats.get("p_value")
            if p_val is not None and p_val < 0.05:
                colors.append("tab:red")
            else:
                colors.append("tab:gray")
        height = max(4, len(labels) * 0.3)
        fig, ax = plt.subplots(figsize=(10, height))
        y = np.arange(len(labels))
        ax.barh(y, diffs, color=colors, alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f"SAM3 drop - SAM2 drop ({display_metric_label(metric)})")
        ax.set_title(f"Differential drop by perturbation ({display_group_label(family)})")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / f"differential_drop_{family}_{metric}.png", dpi=150)
        plt.close(fig)


def plot_cohens_d_heatmap(
    drift_effect_sizes: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    out_dir: Path,
) -> None:
    """Heatmap of Cohen's d values for drift metrics."""
    import matplotlib.pyplot as plt
    
    models = list(drift_effect_sizes.keys())
    if not models:
        return
    
    # Collect all drift metrics
    all_metrics = set()
    for model_data in drift_effect_sizes.values():
        all_metrics.update(model_data.keys())
    
    metrics = sorted(all_metrics)
    
    if not metrics:
        return
    
    # Build matrix
    matrix = []
    for model in models:
        row = []
        for metric in metrics:
            metric_data = drift_effect_sizes.get(model, {}).get(metric, {})
            if metric_data:
                mean_d = float(np.mean([v["mean"] for v in metric_data.values()]))
                row.append(mean_d)
            else:
                row.append(0.0)
        matrix.append(row)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, max(4, len(models) * 0.8)))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-3, vmax=0)
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([display_metric_label(m) for m in metrics], rotation=35, ha="right")
    ax.set_yticklabels([display_model_label(m) for m in models])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cohen's d (vs baseline)", rotation=270, labelpad=20)
    
    # Add values in cells
    for i in range(len(models)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f"{matrix[i][j]:.2f}",
                         ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title("Drift Effect Sizes (Cohen's d)\nNegative = local drift, ~0 = non-local drift")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "cohens_d_heatmap.png", dpi=150)
    plt.close()

def aggregate_relative_drop_by_group(
    relative_drop: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    group_map: Dict[str, str],
) -> Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, float]]]]]:
    out: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, float]]]]] = {}
    for model, metrics in relative_drop.items():
        for metric, entries in metrics.items():
            grouped: Dict[str, Dict[int, List[float]]] = {}
            for key, stats in entries.items():
                pert, sev = parse_summary_key(key)
                if pert is None or sev is None:
                    continue
                group = group_map.get(pert)
                if group is None:
                    continue
                grouped.setdefault(group, {}).setdefault(sev, []).append(float(stats.get("mean", 0.0)))
            for group, by_sev in grouped.items():
                for sev, values in by_sev.items():
                    out.setdefault(model, {}).setdefault(metric, {}).setdefault(group, {})[sev] = {
                        "mean": float(np.mean(values)) if values else 0.0,
                        "count": len(values),
                    }
    return out


def compute_delta_grounding(
    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    text_pipeline: str,
    visual_pipeline: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    text_metrics = summary.get(text_pipeline, {})
    visual_metrics = summary.get(visual_pipeline, {})
    for metric, entries in text_metrics.items():
        if metric not in visual_metrics:
            continue
        for key, stats in entries.items():
            visual_stats = visual_metrics.get(metric, {}).get(key)
            if not visual_stats:
                continue
            delta = float(stats.get("mean", 0.0)) - float(visual_stats.get("mean", 0.0))
            out.setdefault(metric, {})[key] = {"mean": delta, "count": int(stats.get("count", 0))}
    return out


def plot_grouped_metric_trends(
    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    out_dir: Path,
    metric: str,
    group_map: Dict[str, str],
    groups: List[str],
) -> None:
    for model, metrics in summary.items():
        entries = metrics.get(metric, {})
        if not entries:
            continue
        plt.figure(figsize=(8, 5))
        for group in groups:
            by_sev: Dict[int, List[float]] = {}
            for key, stats in entries.items():
                pert, sev = parse_summary_key(key)
                if pert is None or sev is None:
                    continue
                if group_map.get(pert) != group:
                    continue
                by_sev.setdefault(sev, []).append(float(stats.get("mean", 0.0)))
            if not by_sev:
                continue
            xs = sorted(by_sev.keys())
            ys = [float(np.mean(by_sev[sev])) for sev in xs]
            plt.plot(xs, ys, marker="o", label=display_group_label(group))
        if not plt.gca().has_data():
            plt.close()
            continue
        plt.title(f"{display_model_label(model)}: {display_metric_label(metric)}")
        plt.xlabel("Severity")
        plt.ylabel(display_metric_label(metric))
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{model}_{metric}_family.png")
        plt.close()


def plot_relative_drop_trends(
    relative_drop: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, float]]]]],
    out_dir: Path,
    metric: str,
    groups: List[str],
) -> None:
    for model, metrics in relative_drop.items():
        entries = metrics.get(metric, {})
        if not entries:
            continue
        plt.figure(figsize=(8, 5))
        for group in groups:
            sev_map = entries.get(group, {})
            if not sev_map:
                continue
            xs = sorted(sev_map.keys())
            ys = [float(sev_map[sev].get("mean", 0.0)) for sev in xs]
            plt.plot(xs, ys, marker="o", label=display_group_label(group))
        if not plt.gca().has_data():
            plt.close()
            continue
        plt.title(f"{display_model_label(model)}: relative drop ({display_metric_label(metric)})")
        plt.xlabel("Severity")
        plt.ylabel(f"Delta {display_metric_label(metric)}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{model}_{metric}_relative_drop.png")
        plt.close()


def ensure_sam2_assets(args: argparse.Namespace, cache_dir: Path) -> tuple[str, Path]:
    ckpt_root = cache_dir / "checkpoints" / "sam2"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    config_name = args.sam2_config or DEFAULT_SAM2_CONFIG_NAME
    checkpoint_path = (
        Path(args.sam2_checkpoint).expanduser().resolve()
        if args.sam2_checkpoint
        else ckpt_root / "sam2_checkpoint.pt"
    )
    if not checkpoint_path.exists():
        if not args.sam2_auto_download:
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")
        download_file(args.sam2_checkpoint_url, checkpoint_path)
    return str(config_name), checkpoint_path


def ensure_gdino_assets(args: argparse.Namespace, cache_dir: Path) -> tuple[Path, Path]:
    ckpt_root = cache_dir / "checkpoints" / "groundingdino"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    default_cfg_name = Path(args.gdino_config_url).name
    default_ckpt_name = Path(args.gdino_checkpoint_url).name
    config_path = (
        Path(args.gdino_config).expanduser().resolve()
        if args.gdino_config
        else ckpt_root / default_cfg_name
    )
    checkpoint_path = (
        Path(args.gdino_checkpoint).expanduser().resolve()
        if args.gdino_checkpoint
        else ckpt_root / default_ckpt_name
    )
    if not config_path.exists():
        if not args.gdino_auto_download:
            raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")
        download_file(args.gdino_config_url, config_path)
    if not checkpoint_path.exists():
        if not args.gdino_auto_download:
            raise FileNotFoundError(f"GroundingDINO checkpoint not found: {checkpoint_path}")
        download_file(args.gdino_checkpoint_url, checkpoint_path)
    return config_path, checkpoint_path


def _format_sam3_download_error(exc: Exception) -> str:
    msg = str(exc)
    if isinstance(exc, HfHubHTTPError):
        msg = str(getattr(exc, "response", "")) or msg
    if "403" in msg or "gated" in msg or "401" in msg:
        return (
            "SAM3 checkpoint is gated on Hugging Face. "
            "Request access at https://huggingface.co/facebook/sam3 and run `hf auth login`."
        )
    if "404" in msg or "not found" in msg:
        return "SAM3 checkpoint not found on Hugging Face."
    return f"SAM3 download failed ({exc})."


def ensure_sam3_checkpoint(
    args: argparse.Namespace, cache_dir: Path
) -> tuple[Optional[Path], Optional[str]]:
    if args.sam3_checkpoint:
        path = Path(args.sam3_checkpoint).expanduser().resolve()
        if path.exists():
            return path, None
        if not args.sam3_auto_download:
            return None, f"SAM3 checkpoint not found: {path}"
    if not args.sam3_auto_download:
        return None, "SAM3 auto-download disabled."
    repo_id = args.sam3_repo_id
    if not repo_id:
        return None, "SAM3 repo id not configured."
    token = args.sam3_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        if args.sam3_filename:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=args.sam3_filename,
                cache_dir=str(cache_dir),
                token=token,
            )
            return Path(path), None
        repo_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            allow_patterns=["*.safetensors", "*.bin", "*.pt"],
            token=token,
        )
        return Path(repo_path), None
    except Exception as exc:
        return None, _format_sam3_download_error(exc)


def try_build_sam3(
    args: argparse.Namespace, cache_dir: Path, device: str
) -> tuple[Optional[object], Optional[str]]:
    if args.sam3_module and args.sam3_class:
        checkpoint_path, err = ensure_sam3_checkpoint(args, cache_dir)
        if checkpoint_path is None:
            return None, err or "SAM3 checkpoint unavailable."
        try:
            return (
                Sam3Adapter(
                    args.sam3_module,
                    args.sam3_class,
                    str(checkpoint_path),
                    device,
                ),
                None,
            )
        except Exception as exc:
            return None, f"SAM3 init failed ({exc})."

    checkpoint_path, err = ensure_sam3_checkpoint(args, cache_dir)
    if checkpoint_path is None and err:
        return None, err
    try:
        return Sam3NativeAdapter(checkpoint_path, device), None
    except Exception as exc:
        return None, f"SAM3 init failed ({exc})."


def parse_args() -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    known, _ = base.parse_known_args()
    cfg = load_yaml_config(known.config)

    p = argparse.ArgumentParser(description="Segmentation robustness for SAM2 vs SAM3")
    p.add_argument("--config", type=str, default=known.config, help="Path to YAML config.")
    p.add_argument("--manifest", type=str, default=None, help="Path to JSONL manifest.")
    p.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="Cache directory.")
    p.add_argument("--max-samples", type=int, default=200, help="Limit number of samples.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use deterministic torch ops.",
    )
    p.add_argument("--device", type=str, default="auto", help="Device for model inference.")
    p.add_argument(
        "--experiment-mode",
        type=str,
        default="both",
        choices=["both", "pvs_only", "pcs_only", "both_same_task"],
        help="Experiment mode (visual prompts, text prompts, or both).",
    )

    p.add_argument(
        "--include-natural",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run natural perturbations.",
    )
    p.add_argument(
        "--include-frequency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run frequency perturbations.",
    )
    p.add_argument("--severity-levels", type=int, nargs="+", default=[1, 2, 3], help="Severity levels to run.")

    p.add_argument("--dataset-preset", type=str, default="mini_coco", choices=["mini_coco", "davis17"], help="Dataset preset.")
    p.add_argument("--dataset-negative-ratio", type=float, default=0.2, help="Negative sample ratio for mini_coco.")
    p.add_argument(
        "--dataset-rebuild-manifest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild dataset manifest.",
    )
    p.add_argument(
        "--dataset-allow-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow dataset auto-downloads.",
    )
    p.add_argument(
        "--dataset-davis-resolution",
        type=str,
        default="480p",
        choices=["480p", "1080p"],
        help="DAVIS resolution preset.",
    )

    p.add_argument(
        "--sam2-config",
        type=str,
        default=DEFAULT_SAM2_CONFIG_NAME,
        help="SAM2 config name within the sam2 package.",
    )
    p.add_argument("--sam2-checkpoint", type=str, default=None, help="SAM2 checkpoint file.")
    p.add_argument(
        "--sam2-auto-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-download SAM2 assets.",
    )
    p.add_argument(
        "--sam2-config-url",
        type=str,
        default=DEFAULT_SAM2_CONFIG_URL,
        help="SAM2 config download URL (unused when using package configs).",
    )
    p.add_argument(
        "--sam2-checkpoint-url",
        type=str,
        default=DEFAULT_SAM2_CHECKPOINT_URL,
        help="SAM2 checkpoint download URL.",
    )
    p.add_argument(
        "--run-sam2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run SAM2.",
    )
    p.add_argument(
        "--run-gtbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run SAM2 with GT boxes (oracle baseline).",
    )

    p.add_argument(
        "--sam3-backend",
        type=str,
        default="auto",
        choices=["auto", "sam3", "groundingdino_sam2"],
        help="SAM3 backend.",
    )
    p.add_argument("--sam3-module", type=str, default=None, help="Python module path for SAM3 adapter.")
    p.add_argument("--sam3-class", type=str, default=None, help="Class name for SAM3 adapter.")
    p.add_argument("--sam3-checkpoint", type=str, default=None, help="SAM3 checkpoint or model path.")
    p.add_argument("--sam3-repo-id", type=str, default=DEFAULT_SAM3_REPO_ID, help="SAM3 Hugging Face repo id.")
    p.add_argument("--sam3-filename", type=str, default=DEFAULT_SAM3_FILENAME, help="SAM3 filename within repo.")
    p.add_argument("--sam3-token", type=str, default=None, help="Hugging Face token for gated SAM3 downloads.")
    p.add_argument(
        "--sam3-auto-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-download SAM3 checkpoint from HF.",
    )
    p.add_argument(
        "--run-sam3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run SAM3.",
    )

    p.add_argument("--gdino-config", type=str, default=None, help="GroundingDINO config file.")
    p.add_argument("--gdino-checkpoint", type=str, default=None, help="GroundingDINO checkpoint.")
    p.add_argument(
        "--run-dino",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run GroundingDINO -> SAM2 pipeline.",
    )
    p.add_argument(
        "--gdino-auto-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-download GroundingDINO assets.",
    )
    p.add_argument(
        "--gdino-config-url",
        type=str,
        default=DEFAULT_GDINO_CONFIG_URL,
        help="GroundingDINO config download URL.",
    )
    p.add_argument(
        "--gdino-checkpoint-url",
        type=str,
        default=DEFAULT_GDINO_CHECKPOINT_URL,
        help="GroundingDINO checkpoint download URL.",
    )
    p.add_argument("--gdino-box-threshold", type=float, default=0.3, help="GroundingDINO box threshold.")
    p.add_argument("--gdino-text-threshold", type=float, default=0.25, help="GroundingDINO text threshold.")

    p.add_argument("--out-dir", type=str, default="segmentation_robustness_outputs", help="Output directory.")
    p.add_argument(
        "--save-visuals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save visual overlays.",
    )
    p.add_argument("--visual-limit", type=int, default=20, help="Max panels to save.")
    p.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for grounding metrics.")
    p.add_argument("--eval-iou-tau", type=float, default=None, help="IoU threshold for grounding correctness.")
    p.add_argument("--eval-topk-boxes", type=int, default=5, help="Top-k boxes for grounding diagnostics.")
    p.add_argument(
        "--plots-group-by",
        type=str,
        default=None,
        help="Comma-separated family groups for grouped plots.",
    )

    p.set_defaults(**flatten_config(cfg))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, args.deterministic)

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_manifest_path(args, cache_dir)
    samples = load_manifest(manifest_path)
    samples = select_subset(samples, args.max_samples, args.seed)

    perturbations = build_perturbations(
        severity_levels=args.severity_levels,
        include_natural=args.include_natural,
        include_frequency=args.include_frequency,
    )
    group_map = build_group_map(perturbations)
    plot_group_map = build_group_map(perturbations, split_frequency=True)
    family_groups: List[str] = ["natural", "lowfreq", "highfreq"]
    plot_groups: List[str] = [
        "natural",
        "lowfreq_keep",
        "highfreq_keep",
        "lowfreq_noise",
        "highfreq_noise",
        "allfreq_noise",
    ]
    if args.plots_group_by:
        if isinstance(args.plots_group_by, dict):
            plot_groups = list(args.plots_group_by.get("family", plot_groups))
        elif isinstance(args.plots_group_by, list):
            plot_groups = list(args.plots_group_by)
        elif isinstance(args.plots_group_by, str):
            plot_groups = [v.strip() for v in args.plots_group_by.split(",") if v.strip()]

    device = select_device(args.device)

    mode = args.experiment_mode
    run_pvs = mode in ("both", "pvs_only")
    run_pcs = mode in ("both", "pcs_only", "both_same_task")
    iou_tau = args.eval_iou_tau if args.eval_iou_tau is not None else args.iou_threshold
    topk_boxes = args.eval_topk_boxes

    sam2_model = None
    sam3_model = None
    gdino = None
    sam3_reason = None

    if args.run_sam3 and run_pcs and args.sam3_backend in ("auto", "sam3"):
        sam3_model, sam3_reason = try_build_sam3(args, cache_dir, device)
        if sam3_model is None and args.sam3_backend == "sam3":
            raise RuntimeError(sam3_reason or "SAM3 backend unavailable.")
        if sam3_model is None and sam3_reason:
            print(f"[warn] {sam3_reason} SAM3 text pipeline disabled.")
    elif args.run_sam3 and args.sam3_backend == "groundingdino_sam2":
        sam3_reason = "SAM3 backend set to groundingdino_sam2; native SAM3 disabled."

    needs_sam2 = run_pvs or run_pcs or args.run_sam2 or args.run_gtbox
    if needs_sam2:
        config_name, checkpoint_path = ensure_sam2_assets(args, cache_dir)
        sam2_model = Sam2Adapter(config_name, str(checkpoint_path), device)

    if run_pcs and args.run_dino:
        try:
            gdino_config, gdino_checkpoint = ensure_gdino_assets(args, cache_dir)
            gdino = GroundingDinoAdapter(str(gdino_config), str(gdino_checkpoint), device)
        except Exception as exc:
            print(f"[warn] GroundingDINO unavailable ({exc}). DINO->SAM2 pipeline disabled.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = out_dir / "visuals"

    metrics = MetricBook()
    panels_saved = 0
    diagnostics: List[Dict[str, Any]] = []
    sam3_pvs_supported = (
        sam3_model is not None
        and hasattr(sam3_model, "supports_visual_prompting")
        and sam3_model.supports_visual_prompting()
    )
    alignment_available = (
        sam3_model is not None
        and args.run_sam3
        and run_pcs
        and hasattr(sam3_model, "predict_with_internals")
    )
    alignment_warned = False
    alignment_error = None
    alignment_metrics_available = alignment_available
    alignment_metrics_warned = False
    alignment_metrics_error = None
    gdino_failed = False
    meta: Dict[str, Any] = {
        "experiment_mode": mode,
        "sam3_text_available": sam3_model is not None,
        "sam3_pvs_supported": sam3_pvs_supported,
        "sam3_skip_reason": sam3_reason,
        "gdino_available": gdino is not None,
        "gdino_enabled": args.run_dino,
        "sam2_gtbox_enabled": args.run_gtbox,
        "iou_tau": iou_tau,
        "topk_boxes": topk_boxes,
        "plot_groups": plot_groups,
        "plot_group_map": "frequency_detail",
    }
    if args.run_sam3 and run_pcs and sam3_model is not None and not alignment_available:
        print("[warn] SAM3 does not expose internals; alignment analysis disabled.")
        alignment_error = "sam3_internals_unavailable"
    if run_pvs and args.run_sam3 and not sam3_pvs_supported:
        print("[warn] SAM3 visual prompting is not available; skipping SAM3 in PVS.")

    pipeline_counts = {
        "total_samples": len(samples),
        "samples_with_text_prompt": 0,
        "samples_without_text_prompt": 0,
        "sam2": 0,
        "sam2_gtbox": 0,
        "sam3": 0,
        "dino_sam2": 0,
        "all_three": 0,
        "only_sam2": 0,
        "only_sam3": 0,
        "only_dino_sam2": 0,
        "sam2_sam3_only": 0,
        "sam2_dino_only": 0,
        "sam3_dino_only": 0,
    }

    total_samples = len(samples)
    progress_every = max(1, total_samples // 10) if total_samples else 0

    for idx, sample in enumerate(samples):
        if progress_every:
            completed = idx + 1
            if completed == 1 or completed == total_samples or completed % progress_every == 0:
                remaining = total_samples - completed
                print(f"[progress] {completed}/{total_samples} samples (remaining {remaining})")
        loaded = load_sample(sample)
        has_text_prompt = bool(loaded.text_prompt)
        if has_text_prompt:
            pipeline_counts["samples_with_text_prompt"] += 1
        else:
            pipeline_counts["samples_without_text_prompt"] += 1
        sam2_active = sam2_model is not None and (run_pvs or run_pcs or args.run_sam2)
        sam3_active = sam3_model is not None and args.run_sam3 and run_pcs and has_text_prompt
        dino_active = (
            gdino is not None
            and args.run_dino
            and sam2_model is not None
            and run_pcs
            and has_text_prompt
            and not gdino_failed
        )
        sam2_gtbox_active = (
            sam2_model is not None
            and args.run_gtbox
            and any(frame.gt_boxes_all for frame in loaded.frames)
        )
        if sam2_active:
            pipeline_counts["sam2"] += 1
        if sam2_gtbox_active:
            pipeline_counts["sam2_gtbox"] += 1
        if sam3_active:
            pipeline_counts["sam3"] += 1
        if dino_active:
            pipeline_counts["dino_sam2"] += 1
        if sam2_active and sam3_active and dino_active:
            pipeline_counts["all_three"] += 1
        if sam2_active and not sam3_active and not dino_active:
            pipeline_counts["only_sam2"] += 1
        if sam3_active and not sam2_active and not dino_active:
            pipeline_counts["only_sam3"] += 1
        if dino_active and not sam2_active and not sam3_active:
            pipeline_counts["only_dino_sam2"] += 1
        if sam2_active and sam3_active and not dino_active:
            pipeline_counts["sam2_sam3_only"] += 1
        if sam2_active and dino_active and not sam3_active:
            pipeline_counts["sam2_dino_only"] += 1
        if sam3_active and dino_active and not sam2_active:
            pipeline_counts["sam3_dino_only"] += 1
        sample_seed = args.seed + idx * 1000
        rng = random.Random(sample_seed)
        np_rng = np.random.default_rng(sample_seed)
        baseline_conditional_success = {
            "SAM3": [False for _ in loaded.frames],
            "DINO_SAM2": [False for _ in loaded.frames],
        }
        base_alignment: List[Dict[str, float]] = []
        base_visual_fft: List[Optional[torch.Tensor]] = []
        if alignment_available and loaded.text_prompt:
            try:
                for frame in loaded.frames:
                    _, internals = sam3_model.predict_with_internals(frame.image, loaded.text_prompt)
                    if not internals:
                        alignment_available = False
                        alignment_error = "sam3_internals_missing"
                        if not alignment_warned:
                            print(
                                "[warn] SAM3 internals missing from output; alignment analysis disabled."
                            )
                            alignment_warned = True
                        break
                    align_metrics = alignment_metrics_from_internals(internals)
                    if not align_metrics and alignment_metrics_available:
                        alignment_metrics_available = False
                        alignment_metrics_error = "sam3_alignment_tokens_missing"
                        if not alignment_metrics_warned:
                            print(
                                "[warn] SAM3 internals did not expose alignment tokens; "
                                "alignment metrics disabled."
                            )
                            alignment_metrics_warned = True
                    base_alignment.append(align_metrics)
                    base_visual_fft.append(token_fft(internals.get("visual_tokens") if internals else None))
            except Exception as exc:
                alignment_available = False
                alignment_error = str(exc)
                if not alignment_warned:
                    print(f"[warn] SAM3 internals unavailable ({exc}). Alignment metrics disabled.")
                    alignment_warned = True

        for spec in perturbations:
            perturbed = apply_perturbation(loaded, spec, rng, np_rng)
            is_base = spec.kind == "base"

            sam2_preds: List[List[MaskPrediction]] = []
            sam2_gtbox_preds: List[List[MaskPrediction]] = []
            sam3_text_preds: List[List[MaskPrediction]] = []
            sam3_text_internals: List[Optional[Dict[str, torch.Tensor]]] = []
            sam3_pvs_preds: List[List[MaskPrediction]] = []
            dino_preds: List[List[MaskPrediction]] = []
            dino_boxes: List[List[Tuple[float, float, float, float]]] = []
            dino_scores: List[List[float]] = []

            for frame in perturbed.frames:
                if sam2_model is not None and (run_pvs or run_pcs or args.run_sam2):
                    sam2_preds.append(sam2_model.predict(frame.image, frame.boxes, frame.points))
                else:
                    sam2_preds.append([])
                if sam2_model is not None and args.run_gtbox and frame.gt_boxes_all:
                    sam2_gtbox_preds.append(sam2_model.predict(frame.image, frame.gt_boxes_all, []))
                else:
                    sam2_gtbox_preds.append([])
                if sam3_model is not None and args.run_sam3 and run_pcs and perturbed.text_prompt:
                    if alignment_available:
                        try:
                            preds, internals = sam3_model.predict_with_internals(
                                frame.image, perturbed.text_prompt
                            )
                        except Exception as exc:
                            alignment_available = False
                            alignment_error = str(exc)
                            if not alignment_warned:
                                print(
                                    f"[warn] SAM3 internals unavailable ({exc}). "
                                    "Alignment metrics disabled."
                                )
                                alignment_warned = True
                            preds = sam3_model.predict(frame.image, perturbed.text_prompt)
                            internals = None
                        sam3_text_preds.append(preds)
                        sam3_text_internals.append(internals)
                    else:
                        sam3_text_preds.append(sam3_model.predict(frame.image, perturbed.text_prompt))
                        sam3_text_internals.append(None)
                else:
                    sam3_text_preds.append([])
                    sam3_text_internals.append(None)
                if sam3_pvs_supported and args.run_sam3 and run_pvs:
                    sam3_pvs_preds.append(
                        sam3_model.predict_visual(frame.image, frame.boxes, frame.points)
                    )
                else:
                    sam3_pvs_preds.append([])
                if (
                    gdino is not None
                    and args.run_dino
                    and sam2_model is not None
                    and run_pcs
                    and perturbed.text_prompt
                ):
                    if gdino_failed:
                        dino_preds.append([])
                        dino_boxes.append([])
                        dino_scores.append([])
                    else:
                        try:
                            preds, boxes, scores = predict_dino_sam2(
                                frame.image,
                                perturbed.text_prompt,
                                sam2_model,
                                gdino,
                                args.gdino_box_threshold,
                                args.gdino_text_threshold,
                                topk_boxes,
                            )
                        except Exception as exc:
                            print(f"[warn] GroundingDINO inference failed ({exc}). Skipping DINO->SAM2.")
                            gdino_failed = True
                            preds, boxes, scores = [], [], []
                        dino_preds.append(preds)
                        dino_boxes.append(boxes)
                        dino_scores.append(scores)
                else:
                    dino_preds.append([])
                    dino_boxes.append([])
                    dino_scores.append([])

            gt_masks: List[Optional[np.ndarray]] = []
            gt_masks_all: List[List[np.ndarray]] = []
            gt_boxes_all: List[List[Tuple[float, float, float, float]]] = []
            for frame in perturbed.frames:
                masks, target_idx = extract_gt_masks(frame.mask, frame.instance_id)
                if masks:
                    target_mask = masks[target_idx] if target_idx is not None else masks[0]
                else:
                    target_mask = None
                gt_masks.append(target_mask)
                if frame.gt_masks_all:
                    gt_masks_all.append(frame.gt_masks_all)
                else:
                    gt_masks_all.append(masks)
                if frame.gt_boxes_all:
                    gt_boxes_all.append(list(frame.gt_boxes_all))
                else:
                    gt_boxes_all.append(list(frame.boxes))

            if sam2_model is not None and (run_pvs or run_pcs or args.run_sam2):
                pred_masks = [choose_best_prediction(preds) for preds in sam2_preds]
                for pmask, gmask in zip(pred_masks, gt_masks):
                    if gmask is None:
                        continue
                    miou = mask_iou(pmask, gmask) if pmask is not None else 0.0
                    bf = boundary_f_score(pmask, gmask) if pmask is not None else 0.0
                    frag = float(fragmentation(pmask)) if pmask is not None else 0.0
                    metrics.add("SAM2", "miou", spec.name, spec.severity, miou)
                    metrics.add("SAM2", "boundary_f", spec.name, spec.severity, bf)
                    metrics.add("SAM2", "fragmentation", spec.name, spec.severity, frag)
                if len(gt_masks) > 1:
                    video_stats = compute_video_metrics(pred_masks, gt_masks)
                    metrics.add("SAM2", "j", spec.name, spec.severity, video_stats["j"])
                    metrics.add("SAM2", "f", spec.name, spec.severity, video_stats["f"])
                    metrics.add("SAM2", "id_stability", spec.name, spec.severity, video_stats["id_stability"])

            if sam2_model is not None and args.run_gtbox and sam2_gtbox_preds:
                pred_masks = [choose_best_prediction(preds) for preds in sam2_gtbox_preds]
                for pmask, gmask in zip(pred_masks, gt_masks):
                    if gmask is None:
                        continue
                    miou = mask_iou(pmask, gmask) if pmask is not None else 0.0
                    bf = boundary_f_score(pmask, gmask) if pmask is not None else 0.0
                    frag = float(fragmentation(pmask)) if pmask is not None else 0.0
                    metrics.add("SAM2_GTBOX", "miou", spec.name, spec.severity, miou)
                    metrics.add("SAM2_GTBOX", "boundary_f", spec.name, spec.severity, bf)
                    metrics.add("SAM2_GTBOX", "fragmentation", spec.name, spec.severity, frag)
                if len(gt_masks) > 1:
                    video_stats = compute_video_metrics(pred_masks, gt_masks)
                    metrics.add("SAM2_GTBOX", "j", spec.name, spec.severity, video_stats["j"])
                    metrics.add("SAM2_GTBOX", "f", spec.name, spec.severity, video_stats["f"])
                    metrics.add("SAM2_GTBOX", "id_stability", spec.name, spec.severity, video_stats["id_stability"])

            if sam3_pvs_supported and args.run_sam3 and run_pvs:
                pred_masks = [choose_best_prediction(preds) for preds in sam3_pvs_preds]
                for pmask, gmask in zip(pred_masks, gt_masks):
                    if gmask is None:
                        continue
                    miou = mask_iou(pmask, gmask) if pmask is not None else 0.0
                    bf = boundary_f_score(pmask, gmask) if pmask is not None else 0.0
                    frag = float(fragmentation(pmask)) if pmask is not None else 0.0
                    metrics.add("SAM3_PVS", "miou", spec.name, spec.severity, miou)
                    metrics.add("SAM3_PVS", "boundary_f", spec.name, spec.severity, bf)
                    metrics.add("SAM3_PVS", "fragmentation", spec.name, spec.severity, frag)
                if len(gt_masks) > 1:
                    video_stats = compute_video_metrics(pred_masks, gt_masks)
                    metrics.add("SAM3_PVS", "j", spec.name, spec.severity, video_stats["j"])
                    metrics.add("SAM3_PVS", "f", spec.name, spec.severity, video_stats["f"])
                    metrics.add("SAM3_PVS", "id_stability", spec.name, spec.severity, video_stats["id_stability"])

            if sam3_model is not None and args.run_sam3 and run_pcs and perturbed.text_prompt:
                pred_masks = [choose_best_prediction(preds) for preds in sam3_text_preds]
                for pmask, gmask in zip(pred_masks, gt_masks):
                    if gmask is None:
                        continue
                    miou = mask_iou(pmask, gmask) if pmask is not None else 0.0
                    bf = boundary_f_score(pmask, gmask) if pmask is not None else 0.0
                    frag = float(fragmentation(pmask)) if pmask is not None else 0.0
                    metrics.add("SAM3", "miou", spec.name, spec.severity, miou)
                    metrics.add("SAM3", "boundary_f", spec.name, spec.severity, bf)
                    metrics.add("SAM3", "fragmentation", spec.name, spec.severity, frag)
                if len(gt_masks) > 1:
                    video_stats = compute_video_metrics(pred_masks, gt_masks)
                    metrics.add("SAM3", "j", spec.name, spec.severity, video_stats["j"])
                    metrics.add("SAM3", "f", spec.name, spec.severity, video_stats["f"])
                    metrics.add("SAM3", "id_stability", spec.name, spec.severity, video_stats["id_stability"])
                for frame_idx, frame in enumerate(perturbed.frames):
                    grounding, best_pred, best_gt, best_iou = grounding_eval(
                        sam3_text_preds[frame_idx],
                        gt_masks_all[frame_idx],
                        perturbed.concept_present,
                        iou_tau,
                    )
                    for k, v in grounding.items():
                        metrics.add("SAM3", k, spec.name, spec.severity, v)
                    if sam3_text_preds[frame_idx]:
                        scores = sorted(
                            [float(p.score) for p in sam3_text_preds[frame_idx]],
                            reverse=True,
                        )
                        margin = scores[0] - scores[1] if len(scores) > 1 else scores[0]
                        metrics.add("SAM3", "sam3_score_margin", spec.name, spec.severity, margin)
                        metrics.add(
                            "SAM3",
                            "sam3_score_entropy",
                            spec.name,
                            spec.severity,
                            score_entropy(scores[:topk_boxes]),
                        )
                    baseline_ok = baseline_conditional_success["SAM3"][frame_idx]
                    if best_pred is not None and best_gt is not None and best_iou >= iou_tau:
                        if is_base:
                            baseline_conditional_success["SAM3"][frame_idx] = True
                            baseline_ok = True
                        if baseline_ok or is_base:
                            metrics.add(
                                "SAM3",
                                "conditional_miou",
                                spec.name,
                                spec.severity,
                                mask_iou(best_pred.mask, best_gt),
                            )
                            metrics.add(
                                "SAM3",
                                "conditional_boundary_f",
                                spec.name,
                                spec.severity,
                                boundary_f_score(best_pred.mask, best_gt),
                            )
                            metrics.add(
                                "SAM3",
                                "conditional_fragmentation",
                                spec.name,
                                spec.severity,
                                float(fragmentation(best_pred.mask)),
                            )
                    elif baseline_ok and not is_base:
                        metrics.add("SAM3", "conditional_miou", spec.name, spec.severity, 0.0)
                        metrics.add("SAM3", "conditional_boundary_f", spec.name, spec.severity, 0.0)
                        metrics.add("SAM3", "conditional_fragmentation", spec.name, spec.severity, 0.0)
                    internals = sam3_text_internals[frame_idx]
                    if alignment_available and internals:
                        if alignment_metrics_available:
                            align = alignment_metrics_from_internals(internals)
                            if not align and alignment_metrics_available:
                                alignment_metrics_available = False
                                alignment_metrics_error = "sam3_alignment_tokens_missing"
                                if not alignment_metrics_warned:
                                    print(
                                        "[warn] SAM3 internals did not expose alignment tokens; "
                                        "alignment metrics disabled."
                                    )
                                    alignment_metrics_warned = True
                            for key, val in align.items():
                                metrics.add("SAM3", key, spec.name, spec.severity, val)
                                base_vals = base_alignment[frame_idx] if frame_idx < len(base_alignment) else {}
                                base_val = base_vals.get(key) if base_vals else None
                                if base_val is not None:
                                    metrics.add(
                                        "SAM3",
                                        f"{key}_drift",
                                        spec.name,
                                        spec.severity,
                                        float(base_val) - val,
                                    )
                        base_fft = base_visual_fft[frame_idx] if frame_idx < len(base_visual_fft) else None
                        pert_fft = token_fft(internals.get("visual_tokens"))
                        drift = spectral_drift_from_fft(base_fft, pert_fft)
                        if drift:
                            for key, val in drift.items():
                                metrics.add("SAM3", key, spec.name, spec.severity, val)

            if (
                gdino is not None
                and args.run_dino
                and sam2_model is not None
                and run_pcs
                and perturbed.text_prompt
            ):
                pred_masks = [choose_best_prediction(preds) for preds in dino_preds]
                for pmask, gmask in zip(pred_masks, gt_masks):
                    if gmask is None:
                        continue
                    miou = mask_iou(pmask, gmask) if pmask is not None else 0.0
                    bf = boundary_f_score(pmask, gmask) if pmask is not None else 0.0
                    frag = float(fragmentation(pmask)) if pmask is not None else 0.0
                    metrics.add("DINO_SAM2", "miou", spec.name, spec.severity, miou)
                    metrics.add("DINO_SAM2", "boundary_f", spec.name, spec.severity, bf)
                    metrics.add("DINO_SAM2", "fragmentation", spec.name, spec.severity, frag)
                if len(gt_masks) > 1:
                    video_stats = compute_video_metrics(pred_masks, gt_masks)
                    metrics.add("DINO_SAM2", "j", spec.name, spec.severity, video_stats["j"])
                    metrics.add("DINO_SAM2", "f", spec.name, spec.severity, video_stats["f"])
                    metrics.add("DINO_SAM2", "id_stability", spec.name, spec.severity, video_stats["id_stability"])
                for frame_idx, frame in enumerate(perturbed.frames):
                    grounding, best_pred, best_gt, best_iou = grounding_eval(
                        dino_preds[frame_idx],
                        gt_masks_all[frame_idx],
                        perturbed.concept_present,
                        iou_tau,
                    )
                    for k, v in grounding.items():
                        metrics.add("DINO_SAM2", k, spec.name, spec.severity, v)
                    baseline_ok = baseline_conditional_success["DINO_SAM2"][frame_idx]
                    if best_pred is not None and best_gt is not None and best_iou >= iou_tau:
                        if is_base:
                            baseline_conditional_success["DINO_SAM2"][frame_idx] = True
                            baseline_ok = True
                        if baseline_ok or is_base:
                            metrics.add(
                                "DINO_SAM2",
                                "conditional_miou",
                                spec.name,
                                spec.severity,
                                mask_iou(best_pred.mask, best_gt),
                            )
                            metrics.add(
                                "DINO_SAM2",
                                "conditional_boundary_f",
                                spec.name,
                                spec.severity,
                                boundary_f_score(best_pred.mask, best_gt),
                            )
                            metrics.add(
                                "DINO_SAM2",
                                "conditional_fragmentation",
                                spec.name,
                                spec.severity,
                                float(fragmentation(best_pred.mask)),
                            )
                    elif baseline_ok and not is_base:
                        metrics.add("DINO_SAM2", "conditional_miou", spec.name, spec.severity, 0.0)
                        metrics.add("DINO_SAM2", "conditional_boundary_f", spec.name, spec.severity, 0.0)
                        metrics.add("DINO_SAM2", "conditional_fragmentation", spec.name, spec.severity, 0.0)
                    diag = box_diagnostics(
                        dino_boxes[frame_idx],
                        dino_scores[frame_idx],
                        gt_boxes_all[frame_idx],
                        frame.image.size,
                    )
                    for key, val in diag.items():
                        metrics.add("DINO_SAM2", key, spec.name, spec.severity, val)
                    diagnostics.append(
                        {
                            "sample_id": sample.sample_id,
                            "perturbation": spec.name,
                            "severity": spec.severity,
                            "frame_index": frame_idx,
                            "boxes": [list(b) for b in dino_boxes[frame_idx]],
                            "scores": list(dino_scores[frame_idx]),
                        }
                    )

            if args.save_visuals and panels_saved < args.visual_limit:
                base_frame = loaded.frames[0]
                pert_frame = perturbed.frames[0]
                gt_mask = gt_masks[0]
                sam2_mask = None
                sam3_mask = None
                if sam2_model is not None and sam2_preds:
                    sam2_mask = choose_best_prediction(sam2_preds[0])
                if sam3_model is not None and args.run_sam3 and sam3_text_preds:
                    sam3_mask = choose_best_prediction(sam3_text_preds[0])
                out_path = visual_dir / f"{sample.sample_id}_{spec.name.replace(' ', '_')}.png"
                save_panel(
                    out_path,
                    base_frame.image,
                    pert_frame.image,
                    gt_mask,
                    sam2_mask,
                    sam3_mask,
                    title=f"{sample.sample_id} | {spec.name}",
                )
                panels_saved += 1

    meta["sam3_alignment_available"] = alignment_available
    meta["sam3_alignment_metrics_available"] = alignment_metrics_available
    if alignment_error:
        meta["sam3_alignment_error"] = alignment_error
    if alignment_metrics_error:
        meta["sam3_alignment_metrics_error"] = alignment_metrics_error

    alignment_status = {
        "sam3_alignment_available": bool(alignment_available),
        "sam3_alignment_metrics_available": bool(alignment_metrics_available),
    }
    if alignment_error:
        alignment_status["sam3_alignment_error"] = alignment_error
    if alignment_metrics_error:
        alignment_status["sam3_alignment_metrics_error"] = alignment_metrics_error

    control_status = {
        "sam3_pvs_supported": bool(sam3_pvs_supported),
        "sam3_pvs_enabled": bool(args.run_sam3 and run_pvs),
        "sam3_pvs_run": bool(args.run_sam3 and run_pvs and sam3_pvs_supported),
        "gdino_enabled": bool(args.run_dino),
        "sam2_gtbox_enabled": bool(args.run_gtbox),
    }

    summary_core = metrics.summary()
    base_key = "Base|sev0"
    relative_drop = compute_relative_drop(summary_core, base_key)
    relative_drop_by_family = aggregate_relative_drop_by_group(relative_drop, plot_group_map)
    delta_grounding = compute_delta_grounding(summary_core, "DINO_SAM2", "SAM2")
    conditional_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    grounding_drift: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    alignment_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    drift_keys = {
        "best_box_iou_with_gt",
        "box_center_shift",
        "score_margin",
        "score_entropy",
    }
    for model, metrics_dict in summary_core.items():
        for metric, entries in metrics_dict.items():
            if metric.startswith("conditional_"):
                conditional_metrics.setdefault(model, {})[metric] = entries
            if metric in drift_keys:
                grounding_drift.setdefault(model, {})[metric] = entries
            if metric.startswith("vl_alignment") or metric.startswith("visual_"):
                alignment_metrics.setdefault(model, {})[metric] = entries
    drift_effect_sizes = compute_drift_effect_sizes(metrics, ("Base", 0))
    pipeline_modes = {
        "SAM2": "pvs_only",
        "SAM3_PVS": "pvs_only",
        "SAM3": "pcs_only",
        "DINO_SAM2": "pcs_only",
        "SAM2_GTBOX": "pvs_only",
    }
    per_mode: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {"pvs_only": {}, "pcs_only": {}}
    for model, metrics_dict in summary_core.items():
        mode_key = pipeline_modes.get(model)
        if mode_key:
            per_mode[mode_key][model] = metrics_dict

    summary: Dict[str, Any] = dict(summary_core)
    summary["per_mode"] = per_mode
    summary["relative_drop"] = relative_drop
    summary["relative_drop_by_family"] = relative_drop_by_family
    summary["delta_grounding"] = delta_grounding
    summary["conditional_metrics"] = conditional_metrics
    summary["grounding_drift"] = grounding_drift
    summary["alignment_metrics"] = alignment_metrics
    summary["drift_effect_size"] = drift_effect_sizes
    summary["meta"] = meta
    summary["alignment_status"] = alignment_status
    summary["control_status"] = control_status
    summary["pipeline_sample_counts"] = pipeline_counts

    diagnostics_path = out_dir / "grounding_diagnostics.jsonl"
    if diagnostics:
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            for entry in diagnostics:
                f.write(json.dumps(entry) + "\n")
        summary["grounding_diagnostics_path"] = str(diagnostics_path)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    text_path = out_dir / "summary.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        for model, metrics_dict in summary_core.items():
            f.write(f"Model: {model}\n")
            for metric, entries in metrics_dict.items():
                f.write(f"  {metric}\n")
                for key, stats in entries.items():
                    f.write(f"    {key}: mean={stats['mean']:.4f} (n={stats['count']})\n")
            f.write("\n")
        f.write("Per-mode metrics\n")
        for mode_key, metrics_dict in per_mode.items():
            f.write(f"Mode: {mode_key}\n")
            for model, entries in metrics_dict.items():
                f.write(f"  {model}: {len(entries)} metrics\n")
        f.write("\n")
        f.write("Pipeline sample coverage\n")
        f.write(f"  total_samples: {pipeline_counts['total_samples']}\n")
        f.write(f"  samples_with_text_prompt: {pipeline_counts['samples_with_text_prompt']}\n")
        f.write(f"  samples_without_text_prompt: {pipeline_counts['samples_without_text_prompt']}\n")
        f.write(f"  sam2: {pipeline_counts['sam2']}\n")
        f.write(f"  sam2_gtbox: {pipeline_counts['sam2_gtbox']}\n")
        f.write(f"  sam3: {pipeline_counts['sam3']}\n")
        f.write(f"  dino_sam2: {pipeline_counts['dino_sam2']}\n")
        f.write(f"  all_three: {pipeline_counts['all_three']}\n")
        f.write(f"  only_sam2: {pipeline_counts['only_sam2']}\n")
        f.write(f"  only_sam3: {pipeline_counts['only_sam3']}\n")
        f.write(f"  only_dino_sam2: {pipeline_counts['only_dino_sam2']}\n")
        f.write(f"  sam2_sam3_only: {pipeline_counts['sam2_sam3_only']}\n")
        f.write(f"  sam2_dino_only: {pipeline_counts['sam2_dino_only']}\n")
        f.write(f"  sam3_dino_only: {pipeline_counts['sam3_dino_only']}\n")
        f.write("\n")
        f.write("Relative drop (Base severity=0)\n")
        for model, metrics_dict in relative_drop.items():
            f.write(f"Model: {model}\n")
            for metric, entries in metrics_dict.items():
                f.write(f"  {metric}: {len(entries)} entries\n")
        f.write("\n")
        f.write("Delta grounding (DINO_SAM2 - SAM2)\n")
        for metric, entries in delta_grounding.items():
            f.write(f"  {metric}: {len(entries)} entries\n")
        f.write("\n")
        f.write("Conditional mask quality\n")
        for model, metrics_dict in conditional_metrics.items():
            f.write(f"Model: {model}\n")
            for metric, entries in metrics_dict.items():
                f.write(f"  {metric}: {len(entries)} entries\n")
        f.write("\n")
        f.write("Grounding drift diagnostics\n")
        for model, metrics_dict in grounding_drift.items():
            f.write(f"Model: {model}\n")
            for metric, entries in metrics_dict.items():
                f.write(f"  {metric}: {len(entries)} entries\n")
        f.write("\n")
        f.write("Alignment metrics\n")
        for model, metrics_dict in alignment_metrics.items():
            f.write(f"Model: {model}\n")
            for metric, entries in metrics_dict.items():
                f.write(f"  {metric}: {len(entries)} entries\n")
        f.write("\n")
        f.write("Alignment availability\n")
        f.write(f"  sam3_alignment_available: {alignment_status['sam3_alignment_available']}\n")
        f.write(
            f"  sam3_alignment_metrics_available: "
            f"{alignment_status['sam3_alignment_metrics_available']}\n"
        )
        if alignment_status.get("sam3_alignment_error"):
            f.write(f"  sam3_alignment_error: {alignment_status['sam3_alignment_error']}\n")
        if alignment_status.get("sam3_alignment_metrics_error"):
            f.write(
                f"  sam3_alignment_metrics_error: "
                f"{alignment_status['sam3_alignment_metrics_error']}\n"
            )
        f.write("\n")
        f.write("Control status\n")
        f.write(f"  sam3_pvs_supported: {control_status['sam3_pvs_supported']}\n")
        f.write(f"  sam3_pvs_enabled: {control_status['sam3_pvs_enabled']}\n")
        f.write(f"  sam3_pvs_run: {control_status['sam3_pvs_run']}\n")
        f.write(f"  gdino_enabled: {control_status['gdino_enabled']}\n")
        f.write(f"  sam2_gtbox_enabled: {control_status['sam2_gtbox_enabled']}\n")
        f.write("\n")
        f.write("Drift effect sizes (Cohen's d)\n")
        for model, metrics_dict in drift_effect_sizes.items():
            f.write(f"Model: {model}\n")
            for metric, entries in metrics_dict.items():
                f.write(f"  {metric}: {len(entries)} entries\n")
        f.write("\n")

    
    
    # ==================================================================
    # HYPOTHESIS TESTING AND EVALUATION
    # ==================================================================
    print("\n" + "="*60)
    print("HYPOTHESIS TESTING")
    print("="*60)
    
    # Run comprehensive hypothesis tests
    verdict = analyze_hypothesis(metrics, summary_core, drift_effect_sizes, group_map, family_groups)
    verdict["alignment_status"] = alignment_status
    verdict["control_status"] = control_status
    
    # Save verdict to JSON
    verdict_path = out_dir / "hypothesis_verdict.json"
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)
    
    # Print verdict summary
    print(f"\n{'='*60}")
    if verdict['hypothesis_supported']:
        print("HYPOTHESIS VERDICT: ✓ SUPPORTED")
    else:
        print("HYPOTHESIS VERDICT: ✗ NOT SUPPORTED")
    print(f"{'='*60}")
    print(f"Criteria passed: {verdict['criteria_passed']}/{verdict['criteria_total']}")
    print(f"\n{verdict.get('summary', 'No summary available')}")
    print(f"\nDetailed results saved to: {verdict_path}")
    
    # ==================================================================
    # VISUALIZATIONS (both hypothesis-specific and standard metrics)
    # ==================================================================
    plots_dir = out_dir / "plots"
    
    print("\n[plot] Generating hypothesis test visualizations...")
    try:
        plot_alignment_vs_performance_scatter(metrics, plots_dir, "SAM3")
        print("[plot] ✓ Alignment vs performance scatter")
    except Exception as e:
        print(f"[warn] Failed to generate alignment scatter: {e}")
    
    try:
        plot_differential_sensitivity_comparison(metrics, plots_dir, group_map, "miou")
        print("[plot] ✓ Differential sensitivity comparison")
    except Exception as e:
        print(f"[warn] Failed to generate sensitivity comparison: {e}")

    try:
        perturbation_tests = verdict.get("tests", {}).get("differential_sensitivity_by_perturbation", {})
        plot_differential_sensitivity_per_perturbation(perturbation_tests, plots_dir, "miou")
        print("[plot] ✓ Differential sensitivity per perturbation")
    except Exception as e:
        print(f"[warn] Failed to generate per-perturbation sensitivity: {e}")
    
    try:
        plot_cohens_d_heatmap(drift_effect_sizes, plots_dir)
        print("[plot] ✓ Cohen's d heatmap")
    except Exception as e:
        print(f"[warn] Failed to generate Cohen's d heatmap: {e}")
    print("\n[plot] Generating standard metric visualizations...")

    
    for metric in [
        "miou",
        "boundary_f",
        "fragmentation",
        "concept_recall",
        "false_positive",
        "wrong_instance_rate",
        "no_prediction_rate",
        "presence_error",
        "conditional_miou",
        "conditional_boundary_f",
        "conditional_fragmentation",
        "vl_alignment_pre",
        "vl_alignment_post",
        "vl_alignment_gain",
        "vl_alignment_pre_drift",
        "vl_alignment_post_drift",
        "vl_alignment_gain_drift",
        "visual_low_freq_drift",
        "visual_high_freq_drift",
        "visual_drift_ratio",
        "best_box_iou_with_gt",
        "box_center_shift",
        "score_margin",
        "score_entropy",
        "j",
        "f",
        "id_stability",
    ]:
        plot_metric_trends(summary_core, plots_dir, metric, f"{display_metric_label(metric)} vs Severity")
    for metric in [
        "wrong_instance_rate",
        "concept_recall",
        "conditional_miou",
        "vl_alignment_post_drift",
        "vl_alignment_pre_drift",
        "visual_high_freq_drift",
        "visual_low_freq_drift",
        "visual_drift_ratio",
        "best_box_iou_with_gt",
        "box_center_shift",
        "score_entropy",
    ]:
        plot_grouped_metric_trends(summary_core, plots_dir, metric, plot_group_map, plot_groups)
        plot_relative_drop_trends(relative_drop_by_family, plots_dir, metric, plot_groups)

    print(f"Summary written to {summary_path}")
    print(f"Text summary written to {text_path}")
    print(f"Plots saved to {plots_dir}")
    if args.save_visuals:
        print(f"Visual panels saved to {visual_dir}")


if __name__ == "__main__":
    main()

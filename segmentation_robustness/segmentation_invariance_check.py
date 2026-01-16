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
from typing import Any, Dict, List, Optional, Tuple


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
    ]:
        if key in cfg:
            defaults[key] = cfg[key]

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

    max_samples = max_samples or 200
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
                    "instance_id": 1,
                    "boxes": [bbox],
                    "points": [[cx, cy, 1]],
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
            "frames": [{"image": str(image_path.relative_to(root)), "boxes": [], "points": []}],
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
            else:
                centroid = mask_to_point(bin_mask)
                if centroid is None:
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                else:
                    cx, cy = centroid
                boxes = [[bbox[0], bbox[1], bbox[2], bbox[3]]]
                points = [[cx, cy, 1]]
            frames.append(
                {
                    "image": str(frame_path.relative_to(root)),
                    "mask": str(mask_path.relative_to(root)),
                    "instance_id": 1,
                    "boxes": boxes,
                    "points": points,
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
        frames.append(
            LoadedFrame(
                image=image,
                mask=mask,
                instance_id=frame.instance_id,
                boxes=list(frame.boxes),
                points=list(frame.points),
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
    return LoadedFrame(
        image=image,
        mask=mask,
        instance_id=frame.instance_id,
        boxes=boxes,
        points=points,
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
        return LoadedFrame(
            image=image,
            mask=mask,
            instance_id=frame.instance_id,
            boxes=boxes,
            points=points,
        )
    crop = abs(pad)
    if width <= 2 * crop or height <= 2 * crop:
        return LoadedFrame(
            image=frame.image.copy(),
            mask=frame.mask.copy() if frame.mask is not None else None,
            instance_id=frame.instance_id,
            boxes=list(frame.boxes),
            points=list(frame.points),
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
    return LoadedFrame(
        image=image,
        mask=mask,
        instance_id=frame.instance_id,
        boxes=boxes,
        points=points,
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
    return LoadedFrame(
        image=image,
        mask=mask,
        instance_id=frame.instance_id,
        boxes=boxes,
        points=points,
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
    return LoadedFrame(
        image=canvas,
        mask=mask,
        instance_id=frame.instance_id,
        boxes=boxes,
        points=points,
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
    return LoadedFrame(
        image=image,
        mask=mask,
        instance_id=frame.instance_id,
        boxes=boxes,
        points=points,
    )


def overlay_font(image: Image.Image, scale: float) -> ImageFont.ImageFont:
    h = image.size[1]
    size = max(12, int(round(h / 28 * scale)))
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def overlay_text(image: Image.Image, text: str, scale: float) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = overlay_font(image, scale)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    w, h = image.size
    x = max(0, (w - tw) // 2)
    y = max(0, (h - th) // 2)
    pad = max(4, int(round(4 * scale)))
    box = (x - pad, y - pad, x + tw + pad, y + th + pad)
    draw.rectangle(box, fill="white")
    draw.text((x, y), text, fill="red", font=font)
    return out


def overlay_box(image: Image.Image, scale: float) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = image.size
    box_w = int(round(w * 0.45 * scale))
    box_h = int(round(h * 0.25 * scale))
    x0 = max(0, (w - box_w) // 2)
    y0 = max(0, (h - box_h) // 2)
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill="white")
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
        text = sample.text_prompt or sample.concept or "ANSWER"
        scale = float(spec.params.get("scale", 1.0))
        for frame in sample.frames:
            out_frames.append(
                LoadedFrame(
                    image=overlay_text(frame.image, text, scale),
                    mask=frame.mask.copy() if frame.mask is not None else None,
                    instance_id=frame.instance_id,
                    boxes=list(frame.boxes),
                    points=list(frame.points),
                )
            )
    elif spec.kind == "box_overlay":
        scale = float(spec.params.get("scale", 1.0))
        for frame in sample.frames:
            out_frames.append(
                LoadedFrame(
                    image=overlay_box(frame.image, scale),
                    mask=frame.mask.copy() if frame.mask is not None else None,
                    instance_id=frame.instance_id,
                    boxes=list(frame.boxes),
                    points=list(frame.points),
                )
            )
    elif spec.kind == "random_text":
        scale = float(spec.params.get("scale", 1.0))
        phrase = _rand_phrase(rng, 10)
        for frame in sample.frames:
            out_frames.append(
                LoadedFrame(
                    image=overlay_text(frame.image, phrase, scale),
                    mask=frame.mask.copy() if frame.mask is not None else None,
                    instance_id=frame.instance_id,
                    boxes=list(frame.boxes),
                    points=list(frame.points),
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


class Sam3Adapter:
    def __init__(self, module_path: str, class_name: str, checkpoint: str, device: str) -> None:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        if hasattr(cls, "from_pretrained"):
            self.model = cls.from_pretrained(checkpoint, device=device)
        else:
            self.model = cls(checkpoint=checkpoint, device=device)

    def predict(self, image: Image.Image, text_prompt: str) -> List[MaskPrediction]:
        result = self.model.predict(image, text_prompt)
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
        self.processor = Sam3Processor(model)

    def predict(self, image: Image.Image, text_prompt: str) -> List[MaskPrediction]:
        state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(state=state, prompt=text_prompt)
        masks = output.get("masks")
        scores = output.get("scores")
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


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return inter / union if union > 0 else 0.0


def _boundary_from_mask(mask: np.ndarray) -> np.ndarray:
    tmask = torch.from_numpy(mask.astype(np.float32))[None, None]
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32)
    eroded = F.conv2d(tmask, kernel, padding=1)
    eroded = (eroded == 9.0).squeeze(0).squeeze(0)
    boundary = (tmask.squeeze(0).squeeze(0) > 0.5) & (~eroded)
    return boundary.cpu().numpy()


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
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
    base = np.asarray(image).astype(np.float32)
    overlay = base.copy()
    overlay[mask.astype(bool)] = np.array(color, dtype=np.float32)
    boundary = _boundary_from_mask(mask)
    overlay[boundary] = np.array(color, dtype=np.float32)
    mixed = base * (1.0 - alpha) + overlay * alpha
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
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


def plot_metric_trends(
    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    out_dir: Path,
    metric: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 5))
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
        plt.plot(xs, ys, marker="o", label=model)
    plt.title(title)
    plt.xlabel("Severity")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_severity.png")
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
    token = args.sam3_token or None
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

    device = select_device(args.device)

    sam2_model = None
    sam3_model = None

    if args.run_sam3 and args.sam3_backend in ("auto", "sam3"):
        sam3_model, sam3_reason = try_build_sam3(args, cache_dir, device)
        if sam3_model is None and args.sam3_backend == "sam3":
            raise RuntimeError(sam3_reason or "SAM3 backend unavailable.")
        if sam3_model is None and sam3_reason:
            print(f"[warn] {sam3_reason} Falling back to GroundingDINO+SAM2.")

    needs_sam2 = args.run_sam2 or (args.run_sam3 and sam3_model is None)
    if needs_sam2:
        config_name, checkpoint_path = ensure_sam2_assets(args, cache_dir)
        sam2_model = Sam2Adapter(config_name, str(checkpoint_path), device)

    if args.run_sam3 and sam3_model is None:
        if args.sam3_backend == "sam3":
            raise RuntimeError("SAM3 backend requested but unavailable.")
        gdino_config, gdino_checkpoint = ensure_gdino_assets(args, cache_dir)
        gdino = GroundingDinoAdapter(str(gdino_config), str(gdino_checkpoint), device)
        if sam2_model is None:
            raise RuntimeError("GroundingDINO fallback requires SAM2.")
        sam3_model = GroundingDinoSam2Adapter(
            sam2=sam2_model,
            gdino=gdino,
            box_threshold=args.gdino_box_threshold,
            text_threshold=args.gdino_text_threshold,
        )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = out_dir / "visuals"

    metrics = MetricBook()
    panels_saved = 0

    for idx, sample in enumerate(samples):
        loaded = load_sample(sample)
        sample_seed = args.seed + idx * 1000
        rng = random.Random(sample_seed)
        np_rng = np.random.default_rng(sample_seed)

        for spec in perturbations:
            perturbed = apply_perturbation(loaded, spec, rng, np_rng)

            sam2_preds: List[List[MaskPrediction]] = []
            sam3_preds: List[List[MaskPrediction]] = []

            for frame in perturbed.frames:
                if sam2_model is not None and args.run_sam2:
                    sam2_preds.append(sam2_model.predict(frame.image, frame.boxes, frame.points))
                if sam3_model is not None and args.run_sam3 and perturbed.text_prompt:
                    sam3_preds.append(sam3_model.predict(frame.image, perturbed.text_prompt))

            gt_masks: List[Optional[np.ndarray]] = []
            for frame in perturbed.frames:
                masks, target_idx = extract_gt_masks(frame.mask, frame.instance_id)
                if masks:
                    target_mask = masks[target_idx] if target_idx is not None else masks[0]
                else:
                    target_mask = None
                gt_masks.append(target_mask)

            if sam2_model is not None and args.run_sam2:
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

            if sam3_model is not None and args.run_sam3 and perturbed.text_prompt:
                pred_masks = [choose_best_prediction(preds) for preds in sam3_preds]
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
                    masks, target_idx = extract_gt_masks(frame.mask, frame.instance_id)
                    if not masks:
                        continue
                    grounding = concept_metrics(
                        sam3_preds[frame_idx],
                        masks,
                        target_idx,
                        perturbed.concept_present,
                        args.iou_threshold,
                    )
                    for k, v in grounding.items():
                        metrics.add("SAM3", k, spec.name, spec.severity, v)

            if args.save_visuals and panels_saved < args.visual_limit:
                base_frame = loaded.frames[0]
                pert_frame = perturbed.frames[0]
                gt_mask = gt_masks[0]
                sam2_mask = None
                sam3_mask = None
                if sam2_model is not None and args.run_sam2 and sam2_preds:
                    sam2_mask = choose_best_prediction(sam2_preds[0])
                if sam3_model is not None and args.run_sam3 and sam3_preds:
                    sam3_mask = choose_best_prediction(sam3_preds[0])
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

    summary = metrics.summary()
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    text_path = out_dir / "summary.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        for model, metrics_dict in summary.items():
            f.write(f"Model: {model}\n")
            for metric, entries in metrics_dict.items():
                f.write(f"  {metric}\n")
                for key, stats in entries.items():
                    f.write(f"    {key}: mean={stats['mean']:.4f} (n={stats['count']})\n")
            f.write("\n")

    plots_dir = out_dir / "plots"
    for metric in ["miou", "boundary_f", "fragmentation", "concept_recall", "false_positive", "wrong_instance", "presence_error", "j", "f", "id_stability"]:
        plot_metric_trends(summary, plots_dir, metric, f"{metric} vs severity")

    print(f"Summary written to {summary_path}")
    print(f"Text summary written to {text_path}")
    print(f"Plots saved to {plots_dir}")
    if args.save_visuals:
        print(f"Visual panels saved to {visual_dir}")


if __name__ == "__main__":
    main()

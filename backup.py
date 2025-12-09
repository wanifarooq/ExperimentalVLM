#!/usr/bin/env python3
"""
Stability / invariance probe for Qwen3-VL on SEEDBench_IMG (VLMEvalKit format),
using log-likelihood-based option scoring (SEED-Bench-style).

- Dataset: SEEDBench_IMG.tsv + images
- Model: Qwen/Qwen3-VL-2B-Instruct (or any Qwen3-VL HF model)
- For each sample:
    * Build a SEEDBench-style MCQ prompt.
    * For each visual perturbation of the image:
        - Compute log P(option | image, prompt) for each option.
        - Choose argmax option label (A/B/C/...).
    * Compare labels to the base (original image) label.

Metrics per perturbation type v':
    AVg_v' = (# of perturbation instances that change the label) / (total instances)
    Ve_v'  = (# of images where ANY instance of this type changes the label) / (# images used)

compare_mode:
    - "label"   -> use log-likelihood argmax label (current main mode).
    - "text"    -> (hook) can be used later with free-text invariance.
    - "semantic"-> (hook) for future embedding / LLM judge.

Usage example:

python qwen3_vl_seedbench_invariance_ll.py \
  --seedbench-tsv /path/to/SEEDBench_IMG.tsv \
  --image-root /path/to/SEEDBench_IMG \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --device cuda \
  --max-samples 200 \
  --compare-mode label
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import base64
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Avoid Intel OpenMP shared-memory init failures in restricted environments.
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

# ---------------------------------------------------------------------
# Dependency bootstrap
# ---------------------------------------------------------------------


def ensure_package(pkg: str, version_spec: str | None = None) -> None:
    """Install a package if it is missing or version is too old."""
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
        env = os.environ.copy()
        env["PIP_BREAK_SYSTEM_PACKAGES"] = env.get(
            "PIP_BREAK_SYSTEM_PACKAGES", "1"
        )
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
ensure_package("transformers", ">=4.57.0")
ensure_package("Pillow")
ensure_package("pandas")
ensure_package("huggingface_hub")
ensure_package("requests")

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForImageTextToText,
    AutoProcessor,
)
from huggingface_hub import hf_hub_download  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
import tarfile  # noqa: E402
import zipfile  # noqa: E402

RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC

# ---------------------------------------------------------------------
# Utilities: device, normalization
# ---------------------------------------------------------------------


def pick_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infer_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def normalize_text(text: str) -> str:
    """Rough normalization: lowercase, strip punctuation/extra spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------
# Data downloading / caching
# ---------------------------------------------------------------------

DEFAULT_SEEDBENCH_REPO = "AILab-CVC/SEED-Bench"
DEFAULT_SEEDBENCH_TSV = "SEEDBench_IMG.tsv"
DEFAULT_SEEDBENCH_ARCHIVE = "SEEDBench_IMG.zip"
DEFAULT_SEEDBENCH_DIRNAME = "seedbench"
DEFAULT_MODELS_DIRNAME = "models"
DEFAULT_SEEDBENCH_VLMEVALKIT_URL = "https://opencompass.openxlab.space/utils/benchmarks/SEEDBench/SEEDBench_IMG.tsv"


def has_images(path: Path) -> bool:
    """Return True if path exists and has at least one child (file or folder)."""
    return path.exists() and any(path.iterdir())


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract a zip/tar archive into dest_dir, flattening a single top-level folder."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif suffix in {".tar", ".gz", ".tgz", ".bz2", ".xz"}:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    # If extraction created a single subfolder, move its contents up one level.
    children = list(dest_dir.iterdir())
    if len(children) == 1 and children[0].is_dir():
        inner = children[0]
        for p in inner.iterdir():
            target = dest_dir / p.name
            if target.exists():
                continue
            p.rename(target)
        inner.rmdir()


def find_existing_seedbench(
    candidates: Iterable[tuple[Path, Path]]
) -> Optional[tuple[Path, Path]]:
    """Return the first (tsv, images) pair where both exist."""
    for tsv_path, image_root in candidates:
        if tsv_path.exists() and has_images(image_root):
            return tsv_path, image_root
    return None


def count_seedbench_rows(tsv_path: Path) -> int:
    """Return the number of rows in the TSV, or 0 if unreadable."""
    if not tsv_path.exists():
        return 0
    try:
        return int(pd.read_csv(tsv_path, sep="\t").shape[0])
    except Exception as exc:
        print(f"[warn] Could not read TSV to count rows: {tsv_path} ({exc})", file=sys.stderr)
        return 0


def missing_images_for_rows(
    tsv_path: Path,
    image_root: Path,
    limit_rows: Optional[int],
) -> List[Path]:
    """Return list of missing image paths for the first N rows (or all if None)."""
    missing: List[Path] = []
    if not tsv_path.exists():
        return missing
    try:
        df = pd.read_csv(tsv_path, sep="\t", nrows=limit_rows)
    except Exception as exc:
        print(f"[warn] Could not read TSV to check images: {tsv_path} ({exc})", file=sys.stderr)
        return missing

    if "image_path" not in df.columns:
        print(f"[warn] TSV missing 'image_path' column: {tsv_path}", file=sys.stderr)
        return missing

    for rel in df["image_path"]:
        if pd.isna(rel) or str(rel).strip() == "":
            continue
        img_path = image_root / str(rel)
        if not img_path.exists():
            missing.append(img_path)
    return missing


def ensure_seedbench_coverage(
    tsv_path: Path,
    image_root: Path,
    *,
    required_rows: Optional[int],
    repo_id: str,
    tsv_filename: str,
    image_archive: str,
    cache_dir: Path,
    offline: bool,
) -> int:
    """
    Ensure we have at least `required_rows` TSV entries and matching images.
    Will refresh/download the TSV (overwriting) and image archive if needed.
    """
    current_rows = count_seedbench_rows(tsv_path)
    if required_rows and current_rows < required_rows:
        if offline:
            raise FileNotFoundError(
                f"Only {current_rows} rows available in {tsv_path}, but {required_rows} requested in offline mode."
            )
        print(
            f"[setup] TSV has {current_rows} rows; refreshing to satisfy requested {required_rows} samples...",
            file=sys.stderr,
        )
        current_rows = _download_seedbench_from_vlmevalkit(
            tsv_path,
            image_root,
            required_rows=required_rows,
        )
        if current_rows < required_rows:
            print(
                f"[warn] Even after refresh, only {current_rows} rows are available (requested {required_rows}).",
                file=sys.stderr,
            )

    rows_to_check = required_rows if required_rows is not None else current_rows
    missing = missing_images_for_rows(tsv_path, image_root, rows_to_check)
    if missing:
        if offline:
            print(
                f"[warn] {len(missing)} images missing for the first {rows_to_check} rows, offline mode prevents download.",
                file=sys.stderr,
            )
        else:
            print(
                f"[setup] Found {len(missing)} missing images; refreshing from VLMEvalKit source...",
                file=sys.stderr,
            )
            current_rows = _download_seedbench_from_vlmevalkit(
                tsv_path,
                image_root,
                required_rows=required_rows,
            )
            missing = missing_images_for_rows(tsv_path, image_root, rows_to_check)
            if missing:
                print(
                    f"[warn] {len(missing)} images still missing after refresh; some samples may be skipped.",
                    file=sys.stderr,
                )

    return current_rows


def download_seedbench(
    tsv_path: Path,
    image_root: Path,
    *,
    repo_id: str,
    tsv_filename: str,
    image_archive: str,
    cache_dir: Path,
    offline: bool,
    download_images: bool,
    force_tsv: bool = False,
    force_images: bool = False,
) -> None:
    """
    Download SEEDBench TSV (and optional images archive) via huggingface_hub
    into the cache_dir/seedbench layout.
    """
    if offline:
        raise FileNotFoundError(
            f"Missing data at {tsv_path} / {image_root}, and offline mode is enabled."
        )

    data_dir = cache_dir / DEFAULT_SEEDBENCH_DIRNAME
    data_dir.mkdir(parents=True, exist_ok=True)

    if force_tsv and tsv_path.exists():
        tsv_path.unlink()

    if force_images and image_root.exists() and any(image_root.iterdir()):
        print(
            "[setup] Refreshing SEEDBench images from archive (force_images=True)...",
            file=sys.stderr,
        )

    if force_tsv or not tsv_path.exists():
        print(
            f"[setup] Downloading SEEDBench TSV {tsv_filename} from {repo_id}...",
            file=sys.stderr,
        )
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=tsv_filename,
            repo_type="dataset",
            cache_dir=str(cache_dir),
            local_dir=str(tsv_path.parent),
            local_dir_use_symlinks=False,
        )

    need_images = force_images or (not image_root.exists() or not any(image_root.iterdir()))
    if download_images and need_images:
        print(
            f"[setup] Downloading SEEDBench images archive {image_archive} from {repo_id}...",
            file=sys.stderr,
        )
        image_root.mkdir(parents=True, exist_ok=True)
        archive_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=image_archive,
                repo_type="dataset",
                cache_dir=str(cache_dir),
                local_dir=str(data_dir),
                local_dir_use_symlinks=False,
            )
        )
        _extract_archive(archive_path, image_root)


def _download_seedbench_from_vlmevalkit(
    tsv_path: Path,
    image_root: Path,
    *,
    required_rows: Optional[int],
) -> int:
    """
    Download SEEDBench_IMG.tsv from VLMEvalKit's hosted URL, decode base64 images
    to disk, and rewrite the TSV with an image_path column pointing to the files.
    Returns number of rows in the refreshed TSV.
    """
    # If we already have a processed TSV and images on disk, reuse them instead
    # of re-downloading. The caller will handle subsetting rows as needed.

    tmp_path = tsv_path.with_suffix(".download")
    use_cached_download = tmp_path.exists()
    if use_cached_download:
        try:
            df = pd.read_csv(tmp_path, sep="\t")
            print(f"[info] Using cached TSV download at {tmp_path}", file=sys.stderr)
        except Exception as exc:
            print(f"[warn] Failed to read cached TSV at {tmp_path}: {exc}; re-downloading...", file=sys.stderr)
            use_cached_download = False

    if not use_cached_download:
        url = DEFAULT_SEEDBENCH_VLMEVALKIT_URL
        print(f"[setup] Fetching SEEDBench TSV from VLMEvalKit at {url}...", file=sys.stderr)
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        image_root.mkdir(parents=True, exist_ok=True)

        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(resp.content)
        df = pd.read_csv(tmp_path, sep="\t")

    if required_rows is not None:
        df = df.head(required_rows)

    image_paths: List[str] = []
    keep_mask: List[bool] = []
    for i, row in df.iterrows():
        img_b64 = row.get("image")
        if pd.isna(img_b64):
            image_paths.append("")
            keep_mask.append(False)
            continue
        idx_val = row.get("index", i)
        filename = f"{idx_val}.jpg"
        out_path = image_root / filename
        if not out_path.exists():
            try:
                raw = base64.b64decode(img_b64, validate=False)
                with open(out_path, "wb") as f:
                    f.write(raw)
            except Exception as exc:
                print(f"[warn] Failed to decode image for index {idx_val}: {exc}", file=sys.stderr)
                image_paths.append("")
                keep_mask.append(False)
                continue
        image_paths.append(filename)
        keep_mask.append(True)

    df["image_path"] = image_paths
    df = df[[bool(x) for x in keep_mask]]
    if "image" in df.columns:
        df = df.drop(columns=["image"])

    df.to_csv(tsv_path, sep="\t", index=False)
    return int(len(df))


def resolve_seedbench_paths(
    seedbench_tsv: Optional[str],
    image_root: Optional[str],
    data_dir: Path,
    tsv_filename: str,
) -> tuple[Path, Path]:
    """
    Resolve TSV and image root paths. If not provided, default to cache_dir/seedbench.
    If a custom TSV is provided without an image_root, try common sibling folders first.
    """
    data_dir = data_dir.expanduser().resolve()
    if seedbench_tsv:
        tsv_path = Path(seedbench_tsv).expanduser().resolve()
        if image_root:
            img_root = Path(image_root).expanduser().resolve()
        else:
            # Infer image root next to the TSV if present to avoid needless re-downloads.
            sibling_candidates = [
                tsv_path.parent / "SEEDBench_IMG",
                tsv_path.parent / "images",
            ]
            img_root = next(
                (p for p in sibling_candidates if p.exists()), data_dir / "images"
            )
    else:
        tsv_path = data_dir / tsv_filename
        img_root = (
            Path(image_root).expanduser().resolve()
            if image_root
            else data_dir / "images"
        )
    return tsv_path, img_root


def prepare_seedbench_data(args, cache_dir: Path) -> tuple[Path, Path, int]:
    """
    Resolve SEEDBench paths, download if missing, and refresh TSV/images to
    satisfy the requested sample count.
    """
    data_dir = (
        Path(args.data_dir).expanduser().resolve()
        if args.data_dir
        else cache_dir / DEFAULT_SEEDBENCH_DIRNAME
    )

    requested_tsv, requested_img_root = resolve_seedbench_paths(
        args.seedbench_tsv,
        args.image_root,
        data_dir=data_dir,
        tsv_filename=args.seedbench_tsv_filename,
    )

    default_tsv = data_dir / args.seedbench_tsv_filename
    default_img_root = data_dir / "images"
    repo_data_root = Path(__file__).resolve().parent / "data" / "SEED-Bench"
    repo_tsv = repo_data_root / args.seedbench_tsv_filename
    repo_img_root = repo_data_root / "SEEDBench_IMG"

    candidates = [
        (requested_tsv, requested_img_root),
        (repo_tsv, repo_img_root),
        (default_tsv, default_img_root),
    ]
    found = find_existing_seedbench(candidates)

    if found:
        tsv_path, image_root = found
        if (tsv_path, image_root) == (default_tsv, default_img_root):
            print(
                f"[info] Using cached data at {tsv_path} / {image_root}",
                file=sys.stderr,
            )
    else:
        tsv_path, image_root = default_tsv, default_img_root
        print(
            "[info] SEEDBench not found locally; downloading from VLMEvalKit source...",
            file=sys.stderr,
        )
        _download_seedbench_from_vlmevalkit(
            tsv_path,
            image_root,
            required_rows=args.max_samples,
        )

    total_rows = ensure_seedbench_coverage(
        tsv_path,
        image_root,
        required_rows=args.max_samples,
        repo_id=args.seedbench_repo,
        tsv_filename=args.seedbench_tsv_filename,
        image_archive=args.seedbench_image_archive,
        cache_dir=cache_dir,
        offline=args.offline,
    )
    return tsv_path, image_root, total_rows


def safe_model_dir_name(model_id: str) -> str:
    """Make a filesystem-friendly folder name for a model id."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)


def ensure_model_available(
    model_id: str,
    model_path: Optional[str],
    cache_dir: Path,
    offline: bool,
) -> Path:
    """
    Resolve a local model directory to load from. Precedence:
      1) explicit model_path if provided
      2) cached under <cache_dir>/models/<safe_name>
      3) download via snapshot_download if allowed
    """
    if model_path:
        local = Path(model_path).expanduser().resolve()
        if local.exists():
            return local
        else:
            print(
                f"[warn] Provided model path not found: {local}. Falling back to cache.",
                file=sys.stderr,
            )

    models_root = cache_dir / DEFAULT_MODELS_DIRNAME
    models_root.mkdir(parents=True, exist_ok=True)
    local_dir = models_root / safe_model_dir_name(model_id)

    if local_dir.exists() and any(local_dir.iterdir()):
        return local_dir

    if offline:
        raise FileNotFoundError(
            f"Model not found locally at {local_dir}, and offline mode is enabled."
        )

    print(
        f"[setup] Downloading model {model_id} to {local_dir}...",
        file=sys.stderr,
    )
    snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir),
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return local_dir


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class Sample:
    index: int
    image_path: Path
    question: str
    options: Dict[str, str]  # e.g. {"A": "...", "B": "..."}
    hint: Optional[str] = None
    category: Optional[str] = None  # SEEDBench type if present


@dataclass
class PerturbStats:
    changed_instances: int = 0
    total_instances: int = 0
    images_affected: int = 0


# ---------------------------------------------------------------------
# SEEDBench loading & prompt construction
# ---------------------------------------------------------------------


def load_seedbench_samples(
    tsv_path: Path,
    image_root: Path,
    max_samples: Optional[int] = None,
) -> List[Sample]:
    """
    Load SEEDBench_IMG.tsv in the style used by VLMEvalKit.

    Expected columns:
      - image_path
      - question
      - options in columns 'A', 'B', 'C', ...
      - optional: 'hint', 'category'/'type'/'ability', 'index'
    """
    df = pd.read_csv(tsv_path, sep="\t")
    if max_samples is not None:
        df = df.head(max_samples)

    samples: List[Sample] = []
    for i, row in df.iterrows():
        img_rel = str(row["image_path"])
        img_path = image_root / img_rel

        question = str(row["question"])
        hint = None
        if "hint" in df.columns and not pd.isna(row["hint"]):
            hint = str(row["hint"])

        options: Dict[str, str] = {}
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if letter in df.columns and not pd.isna(row[letter]):
                opt = str(row[letter])
                if opt.strip():
                    options[letter] = opt

        category = None
        for col in ["category", "type", "ability"]:
            if col in df.columns and not pd.isna(row[col]):
                category = str(row[col])
                break

        idx_val = (
            int(row["index"])
            if "index" in df.columns and not pd.isna(row["index"])
            else int(i)
        )

        samples.append(
            Sample(
                index=idx_val,
                image_path=img_path,
                question=question,
                options=options,
                hint=hint,
                category=category,
            )
        )
    return samples


def build_seedbench_prompt(sample: Sample) -> str:
    """
    Build a VLMEvalKit-style prompt for the SEEDBench multi-choice question.
    """
    parts: List[str] = []
    if sample.hint:
        parts.append(f"Hint: {sample.hint}")
    parts.append(f"Question: {sample.question}")
    if sample.options:
        parts.append("Options:")
        for key, val in sample.options.items():
            parts.append(f"{key}. {val}")
        parts.append("Please select the correct answer from the options above.")
    else:
        parts.append("Please answer the question concisely.")
    return "\n".join(parts)


# ---------------------------------------------------------------------
# Image operations: resizing + perturbations
# ---------------------------------------------------------------------


def resize_max_side(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """Resize so that max(width, height) <= max_side (preserve aspect)."""
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), RESAMPLE_BICUBIC)


def cyclic_horizontal_shift(img: Image.Image, n: int) -> Image.Image:
    """
    Cyclic horizontal shift by n pixels (wrap-around).
    """
    if n == 0:
        return img.copy()
    w, h = img.size
    new = Image.new(img.mode, (w, h))
    n_mod = n % w
    if n_mod == 0:
        return img.copy()

    left = img.crop((0, 0, w - n_mod, h))
    right = img.crop((w - n_mod, 0, w, h))
    new.paste(right, (0, 0))
    new.paste(left, (n_mod, 0))
    return new


def pad_or_crop(img: Image.Image, n: int) -> Image.Image:
    """
    Pad or crop:
      - n > 0: pad with black border of width n on each side.
      - n < 0: crop |n| pixels from each side (zoom-in).
    """
    w, h = img.size
    if n == 0:
        return img.copy()
    if n > 0:
        new_w, new_h = w + 2 * n, h + 2 * n
        canvas = Image.new(img.mode, (new_w, new_h), "black")
        canvas.paste(img, (n, n))
        return canvas
    else:
        k = abs(n)
        if w <= 2 * k or h <= 2 * k:
            return img.copy()
        return img.crop((k, k, w - k, h - k))


def scale_image(img: Image.Image, scale: float = 0.9) -> Image.Image:
    """Uniform bicubic scaling."""
    w, h = img.size
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), RESAMPLE_BICUBIC)


def scale_and_pad(
    img: Image.Image,
    scale: float = 0.9,
    background: str = "black",
) -> Image.Image:
    """
    Scale by <scale> and pad back to original size with given background.
    """
    w, h = img.size
    scaled = scale_image(img, scale)
    sw, sh = scaled.size
    canvas = Image.new(img.mode, (w, h), background)
    offset_x = (w - sw) // 2
    offset_y = (h - sh) // 2
    canvas.paste(scaled, (offset_x, offset_y))
    return canvas




def text_overlay(img: Image.Image, text: str) -> Image.Image:
    """
    Overlay red, instruction-like text in the center.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, h // 32))
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(0, (w - tw) // 2)
    y = max(0, (h - th) // 2)
    pad = 4
    draw.rectangle((x - pad, y - pad, x + tw + pad, y + th + pad), fill="white")
    draw.text((x, y), text, fill="red", font=font)
    return img


def rotate_image(img: Image.Image, angle: float) -> Image.Image:
    """Rotate by given angle with expand=True, fill white."""
    return img.rotate(angle, resample=RESAMPLE_BICUBIC, expand=True, fillcolor="white")


# ---------------------------------------------------------------------
# Model preparation
# ---------------------------------------------------------------------


def prepare_model(
    model_id: str,
    device: str,
    cache_dir: Optional[str] = None,
    *,
    local_files_only: bool = False,
):
    dtype = infer_dtype(device)
    model_kwargs: Dict[str, Any] = {"dtype": dtype}
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir

    print(
        f"[setup] Loading model {model_id} on {device} with dtype={dtype} "
        f"(local_only={local_files_only})",
        file=sys.stderr,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        **model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return model, processor


# ---------------------------------------------------------------------
# Option scoring via log-likelihood (SEED-Bench style)
# ---------------------------------------------------------------------


@torch.inference_mode()
def score_option_loglik(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    option_text: str,
    device: str,
    context_len: int,
) -> float:
    """
    Compute log P(option_text | image, prompt) using the causal LM structure:

    - We build messages = [user(image+prompt), assistant(option_text)].
    - We apply the chat template to get full input_ids.
    - We know context_len = length of user-only input_ids with add_generation_prompt=True.
    - For j in [context_len, full_len-1]:
        log p(token_j | <all previous>) = log softmax(logits[j-1])[token_j]
    - Return sum of these log-probs.

    This gives a scalar log-likelihood for the whole option_text.
    """
    # Build full conversation with assistant = option_text
    messages_full = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": option_text},
            ],
        },
    ]
    inputs_full = processor.apply_chat_template(
        messages_full,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
    )
    inputs_full = {
        k: (v.to(device) if hasattr(v, "to") else v)
        for k, v in inputs_full.items()
    }

    input_ids = inputs_full["input_ids"]  # [1, L]
    full_len = input_ids.shape[1]

    outputs = model(**inputs_full)
    logits = outputs.logits  # [1, L, vocab]

    # log_probs at positions 0..L-2 predicting tokens 1..L-1
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

    sum_logprob = 0.0
    # tokens j from context_len .. full_len-1 are part of the assistant reply
    for j in range(context_len, full_len):
        token_id = input_ids[0, j]
        # predicted at position j-1
        lp = log_probs[0, j - 1, token_id].item()
        sum_logprob += lp

    return sum_logprob


@torch.inference_mode()
def get_context_length(
    processor,
    image: Image.Image,
    prompt: str,
    device: str,
) -> int:
    """
    Get the token length of the user-only context (image + prompt), using
    add_generation_prompt=True so the model is ready to generate assistant output.

    context_len is the length of input_ids for:
        messages = [user(image+prompt)]
        add_generation_prompt=True
    """
    messages_user = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs_user = processor.apply_chat_template(
        messages_user,
        tokenize=True,
        add_generation_prompt=True,  # model expects assistant next
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = inputs_user["input_ids"]  # [1, L_ctx]
    return int(input_ids.shape[1])


@torch.inference_mode()
def choose_label_via_loglik(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    options: Dict[str, str],
    device: str,
) -> Optional[str]:
    """
    For this image + prompt, compute log-likelihood for each option_text
    and return the argmax option label (A/B/C/...).

    Returns:
        label (e.g. "A", "B", ...) or None if no options.
    """
    if not options:
        return None

    context_len = get_context_length(processor, image, prompt, device=device)

    best_label = None
    best_score = -float("inf")

    for lab, opt_text in options.items():
        s = score_option_loglik(
            model,
            processor,
            image,
            prompt,
            opt_text,
            device=device,
            context_len=context_len,
        )
        if s > best_score:
            best_score = s
            best_label = lab

    return best_label


# ---------------------------------------------------------------------
# Hooks for other compare modes (free-text / semantic) to use later
# ---------------------------------------------------------------------


@torch.inference_mode()
def generate_free_answer(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> str:
    """
    Free-text answer generation with Qwen3-VL, same style as your earlier script.
    Currently only used if compare_mode != 'label'.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False

    out = model.generate(**inputs, **gen_kwargs)
    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()


def get_representation_text_or_semantic(
    answer_text: str,
    compare_mode: str,
) -> str:
    """
    Map free-text answer to a representation for invariance in modes:
      - 'text'     : normalized surface-form
      - 'semantic' : placeholder (currently same as 'text', later: embeddings)
    """
    if compare_mode == "text":
        return normalize_text(answer_text)
    elif compare_mode == "semantic":
        # Placeholder: later replace with embedding / LLM-judge.
        return normalize_text(answer_text)
    else:
        raise ValueError(f"Unsupported compare_mode for free-text: {compare_mode}")


def entropy_from_answers(reprs: Iterable[Optional[str]]) -> float:
    """
    Shannon entropy over discrete representations (labels or texts).
    """
    vals = [r for r in reprs if r is not None]
    if not vals:
        return 0.0
    freq: Dict[str, int] = {}
    for r in vals:
        freq[r] = freq.get(r, 0) + 1
    total = float(sum(freq.values()))
    ent = 0.0
    for c in freq.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def run_experiment(
    model,
    processor,
    device: str,
    samples: List[Sample],
    translate_pixels: List[int],
    padcrop_pixels: List[int],
    scale_factor: float,
    max_new_tokens: int,
    temperature: float,
    compare_mode: str,
    save_changed_dir: Optional[Path],
    save_changed_limit: int,
) -> None:
    """
    Invariance experiment:

    If compare_mode == 'label':
        - Use log-likelihood-based argmax label as representation.
    Else:
        - Use free-text generation + normalization (for 'text' / 'semantic').

    For each image:
      - Compute base representation R_base.
      - For each perturbation family v':
          - For each instance, compute R_v.
          - Count label/representation changes.
      - Compute visual entropy over all representations for that sample.
    """
    perturb_types = [
        "Translation",
        "Pad/Crop",
        "Scale",
        "Scale+Pad",
        "TextOverlay",
        "Rotation",
    ]
    stats: Dict[str, PerturbStats] = {t: PerturbStats() for t in perturb_types}
    any_stats = PerturbStats()
    saved_examples = 0

    N_total = len(samples)
    N_used = 0  # samples with a valid base representation

    print(f"[info] Running invariance experiment on {N_total} samples...")
    for i, sample in enumerate(samples, start=1):
        if not sample.image_path.exists():
            print(f"[warn] Missing image: {sample.image_path}", file=sys.stderr)
            continue

        img = Image.open(sample.image_path).convert("RGB")
        img = resize_max_side(img, max_side=1024)
        prompt = build_seedbench_prompt(sample)

        # ---------------- Base representation ----------------
        if compare_mode == "label":
            base_repr = choose_label_via_loglik(
                model,
                processor,
                img,
                prompt,
                sample.options,
                device=device,
            )
            base_text_for_log = f"[loglik argmax] {base_repr}"
        else:
            base_text = generate_free_answer(
                model,
                processor,
                img,
                prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            base_repr = get_representation_text_or_semantic(
                base_text,
                compare_mode,
            )
            base_text_for_log = base_text

        print(f"\n=== Sample {i}/{N_total} (index={sample.index}) ===")
        print(f"Q: {sample.question}")
        if sample.options:
            print("Options: " + " | ".join(f"{k}: {v}" for k, v in sample.options.items()))
        print(f"[base answer] {base_text_for_log}")
        print(f"[base repr ({compare_mode})] {base_repr}")

        if base_repr is None:
            print("[info] Base representation is None; skipping this sample.")
            continue

        N_used += 1

        per_image_changed_any = False
        per_image_changed_type: Dict[str, bool] = {t: False for t in perturb_types}
        reprs_for_entropy: List[Optional[str]] = [base_repr]

        # Helper to compute repr for perturbed image
        def compute_repr_for_image(vimg: Image.Image) -> Optional[str]:
            if compare_mode == "label":
                return choose_label_via_loglik(
                    model,
                    processor,
                    vimg,
                    prompt,
                    sample.options,
                    device=device,
                )
            else:
                ans = generate_free_answer(
                    model,
                    processor,
                    vimg,
                    prompt,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                return get_representation_text_or_semantic(ans, compare_mode)

        def maybe_save_changed_pair(
            tag: str, vimg: Image.Image, changed_label: Optional[str]
        ) -> None:
            """Save original + perturbed image when the prediction changes."""
            nonlocal saved_examples
            if save_changed_dir is None:
                return
            limit_reached = save_changed_limit > 0 and saved_examples >= save_changed_limit
            if limit_reached:
                return
            try:
                save_changed_dir.mkdir(parents=True, exist_ok=True)
                base_tag = f"base-{base_repr}"
                pert_tag = f"perturbed-{changed_label or 'none'}"
                base_out = (
                    save_changed_dir
                    / f"{sample.index}_{tag}_{saved_examples+1}_{base_tag}.jpg"
                )
                pert_out = (
                    save_changed_dir
                    / f"{sample.index}_{tag}_{saved_examples+1}_{pert_tag}.jpg"
                )
                img.save(base_out)
                vimg.convert("RGB").save(pert_out)
                print(
                    "[save] Saved changed prediction pair "
                    f"(base={base_repr} -> perturbed={changed_label}): "
                    f"{base_out.name}, {pert_out.name}"
                )
                saved_examples += 1
            except Exception as exc:
                print(f"[warn] Failed to save changed pair for sample {sample.index}: {exc}", file=sys.stderr)

        # ---------------- Translation ----------------
        for n in translate_pixels:
            if n == 0:
                continue
            vimg = cyclic_horizontal_shift(img, n)
            r = compute_repr_for_image(vimg)
            stats["Translation"].total_instances += 1
            any_stats.total_instances += 1

            if r is not None:
                reprs_for_entropy.append(r)
                if r != base_repr:
                    stats["Translation"].changed_instances += 1
                    any_stats.changed_instances += 1
                    per_image_changed_type["Translation"] = True
                    per_image_changed_any = True
                    maybe_save_changed_pair("translation", vimg, r)

        # ---------------- Pad/Crop -------------------
        for n in padcrop_pixels:
            if n == 0:
                continue
            vimg = pad_or_crop(img, n)
            r = compute_repr_for_image(vimg)
            stats["Pad/Crop"].total_instances += 1
            any_stats.total_instances += 1

            if r is not None:
                reprs_for_entropy.append(r)
                if r != base_repr:
                    stats["Pad/Crop"].changed_instances += 1
                    any_stats.changed_instances += 1
                    per_image_changed_type["Pad/Crop"] = True
                    per_image_changed_any = True
                    maybe_save_changed_pair("padcrop", vimg, r)

        # ---------------- Scale ----------------------
        vimg_scale = scale_image(img, scale_factor)
        r = compute_repr_for_image(vimg_scale)
        stats["Scale"].total_instances += 1
        any_stats.total_instances += 1
        if r is not None:
            reprs_for_entropy.append(r)
            if r != base_repr:
                stats["Scale"].changed_instances += 1
                any_stats.changed_instances += 1
                per_image_changed_type["Scale"] = True
                per_image_changed_any = True
                maybe_save_changed_pair("scale", vimg_scale, r)

        # ---------------- Scale+Pad (black/white) ---
        for bg in ["black", "white"]:
            vimg_sp = scale_and_pad(img, scale_factor, background=bg)
            r = compute_repr_for_image(vimg_sp)
            stats["Scale+Pad"].total_instances += 1
            any_stats.total_instances += 1
            if r is not None:
                reprs_for_entropy.append(r)
                if r != base_repr:
                    stats["Scale+Pad"].changed_instances += 1
                    any_stats.changed_instances += 1
                    per_image_changed_type["Scale+Pad"] = True
                    per_image_changed_any = True
                    maybe_save_changed_pair(f"scale_pad_{bg}", vimg_sp, r)

        # ---------------- TextOverlay ----------------
        # Create text-overlay phrases based on the base representation.
        # E.g., if base_repr == "A", produce ["Answer is B", "Answer is C", "Answer is D"].
        other_labels = [lab for lab in sample.options.keys() if lab != base_repr]
        other_labels = other_labels[:3]  # take at most three alternatives
        TEXT_OVERLAY_PHRASES = [f"Answer is {lab}" for lab in other_labels]
        if not TEXT_OVERLAY_PHRASES:
            # fallback if options are missing for some reason
            TEXT_OVERLAY_PHRASES = ["Answer is B", "Answer is C", "Answer is D"]
        
        for phrase in TEXT_OVERLAY_PHRASES:
            vimg_txt = text_overlay(img, phrase)
            r = compute_repr_for_image(vimg_txt)
            stats["TextOverlay"].total_instances += 1
            any_stats.total_instances += 1
            if r is not None:
                reprs_for_entropy.append(r)
                if r != base_repr:
                    stats["TextOverlay"].changed_instances += 1
                    any_stats.changed_instances += 1
                    per_image_changed_type["TextOverlay"] = True
                    per_image_changed_any = True
                    maybe_save_changed_pair("text_overlay", vimg_txt, r)

        # ---------------- Rotation -------------------
        for angle in (-30.0, 30.0):
            vimg_rot = rotate_image(img, angle)
            r = compute_repr_for_image(vimg_rot)
            stats["Rotation"].total_instances += 1
            any_stats.total_instances += 1
            if r is not None:
                reprs_for_entropy.append(r)
                if r != base_repr:
                    stats["Rotation"].changed_instances += 1
                    any_stats.changed_instances += 1
                    per_image_changed_type["Rotation"] = True
                    per_image_changed_any = True
                    maybe_save_changed_pair(f"rotation_{int(angle)}", vimg_rot, r)

        # Update per-image affected counts
        for t in perturb_types:
            if per_image_changed_type[t]:
                stats[t].images_affected += 1
        if per_image_changed_any:
            any_stats.images_affected += 1

        Hv = entropy_from_answers(reprs_for_entropy)
        changed_types = [t for t in perturb_types if per_image_changed_type[t]]
        print(f"[info] Repr changed for types: {changed_types or 'None'}")
        print(f"[info] Visual entropy H_v ({compare_mode}) = {Hv:.4f}")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n========== SUMMARY ==========")
    print(f"Total samples in TSV: {N_total}")
    print(f"Samples used (with valid base representation): {N_used}")

    def safe_div(a: int, b: int) -> float:
        return float(a) / float(b) if b > 0 else 0.0

    header = f"{'Type':<12}  {'AVg':>8}  {'Ve':>8}  (#img_affected / #img_used, #changed / #total)"
    print(header)
    print("-" * len(header))

    for t in perturb_types:
        st = stats[t]
        avg = safe_div(st.changed_instances, st.total_instances)
        ve = safe_div(st.images_affected, N_used)
        print(
            f"{t:<12}  {avg:8.3f}  {ve:8.3f}  "
            f"({st.images_affected}/{N_used}, {st.changed_instances}/{st.total_instances})"
        )

    avg_any = safe_div(any_stats.changed_instances, any_stats.total_instances)
    ve_any = safe_div(any_stats.images_affected, N_used)
    print(
        f"{'Any':<12}  {avg_any:8.3f}  {ve_any:8.3f}  "
        f"({any_stats.images_affected}/{N_used}, "
        f"{any_stats.changed_instances}/{any_stats.total_instances})"
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3-VL invariance on SEEDBench_IMG under visual perturbations (log-likelihood scoring).",
    )
    p.add_argument( "--model-id", default="Qwen/Qwen3-VL-2B-Instruct", help="Hugging Face model id to use.")
    p.add_argument( "--model-path", type=str, default=None, help="Local path to a model directory (overrides model-id).")
    
    p.add_argument( "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Computation device.")
    p.add_argument( "--seedbench-tsv", type=str, default=None, help="Path to SEEDBench_IMG.tsv. If omitted, defaults to cache dir and auto-download if enabled.")
    p.add_argument( "--image-root", type=str, default=None, help="Root directory for SEEDBench images. If omitted, defaults to cache dir and auto-download if enabled.")
    
    p.add_argument( "--seedbench-repo", type=str, default=DEFAULT_SEEDBENCH_REPO, help="Hugging Face dataset repo id for SEEDBench.")
    p.add_argument( "--seedbench-tsv-filename", type=str, default=DEFAULT_SEEDBENCH_TSV, help="Filename of the SEEDBench TSV inside the repo or cache.")
    p.add_argument( "--seedbench-image-archive", type=str, default=DEFAULT_SEEDBENCH_ARCHIVE, help="Archive filename for SEEDBench images inside the repo (zip/tar).")
    
    p.add_argument( "--data-dir", type=str, default=None, help="Base directory to cache SEEDBench data (defaults to <cache_dir>/seedbench).")
    p.add_argument( "--max-samples", type=int, default=200, help="Maximum number of samples to evaluate.")
    p.add_argument( "--max-new-tokens", type=int, default=32, help="Max new tokens (only used for non-label compare modes).")
    p.add_argument( "--temperature", type=float, default=0.0, help="Sampling temperature for free-text modes.")

    p.add_argument( "--translate-steps", type=int, nargs="+", default=[-16, -12, -8, -4, 4, 8, 12, 16], help="Horizontal cyclic shifts (pixels).")
    p.add_argument( "--padcrop-steps", type=int, nargs="+", default=[-16, -12, -8, -4, 4, 8, 12, 16], help="Pad/Crop sizes (pixels; negative=crop).")
    p.add_argument( "--scale-factor", type=float, default=0.9, help="Scale factor for Scale and Scale+Pad.")
    p.add_argument( "--compare-mode", type=str, default="label", choices=["label", "text", "semantic"], help=(
                                                                    "Representation used for invariance comparison:\n"
                                                                    "  label   -> log-likelihood argmax MCQ label (SEED-Bench style)\n"
                                                                    "  text    -> normalized free-text (for later ablations)\n"
                                                                    "  semantic-> placeholder for embedding/LLM-based semantic invariance") )
    p.add_argument( "--cache-dir", type=str, default=str((Path(__file__).resolve().parent / ".hf_cache")), help="HF cache directory." )
    p.add_argument( "--offline",action="store_true", help="Use HF cache only (no internet)." )
    p.add_argument( "--save-changed-dir", type=str, default="changed_predictions", help="Directory to save original/perturbed image pairs when predictions change. Set empty to disable." )
    p.add_argument( "--save-changed-limit", type=int, default=20, help="Max number of changed-prediction pairs to save (set <=0 for no limit)." )
    
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = pick_device(args.device)
    torch.manual_seed(0)

    print(f"[info] Using device={device}")
    print(f"[info] HF cache dir={cache_dir}")

    tsv_path, image_root, total_rows = prepare_seedbench_data(args, cache_dir)
    if not tsv_path.exists() or not image_root.exists():
        raise FileNotFoundError(
            f"SEEDBench data unavailable after download attempts: {tsv_path}, {image_root}"
        )

    samples = load_seedbench_samples(tsv_path, image_root, max_samples=args.max_samples)
    print(f"[info] Loaded {len(samples)} samples from {tsv_path.name} (rows available: {total_rows})")
    if args.max_samples is not None and len(samples) < args.max_samples:
        print(
            f"[warn] Only {len(samples)} samples available; requested {args.max_samples}.",
            file=sys.stderr,
        )

    save_dir = Path(args.save_changed_dir).expanduser().resolve() if args.save_changed_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    model_source = ensure_model_available(
        args.model_id,
        args.model_path,
        cache_dir=cache_dir,
        offline=args.offline,
    )
    print(f"[info] Using model from {model_source}")

    model, processor = prepare_model(
        str(model_source),
        device=device,
        cache_dir=str(cache_dir),
        local_files_only=args.offline,
    )

    run_experiment(
        model,
        processor,
        device=device,
        samples=samples,
        translate_pixels=args.translate_steps,
        padcrop_pixels=args.padcrop_steps,
        scale_factor=args.scale_factor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        compare_mode=args.compare_mode,
        save_changed_dir=save_dir,
        save_changed_limit=args.save_changed_limit,
    )


if __name__ == "__main__":
    from pathlib import Path
    main()

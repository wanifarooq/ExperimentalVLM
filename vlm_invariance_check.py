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
    - "label"    -> use log-likelihood argmax label (current main mode).
    - "text"     -> free-text invariance with an open prompt.
    - "text_mcq" -> free-text invariance while keeping options visible.
    - "semantic" -> (hook) for future embedding / LLM judge.

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
import random
import requests
from dataclasses import dataclass, field
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
ensure_package("matplotlib")
ensure_package("scikit-learn")

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForImageTextToText,
    AutoProcessor,
)
from huggingface_hub import hf_hub_download  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
import tarfile  # noqa: E402
import zipfile  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
BATCH_LOGPROB_FALLBACK_WARNED = False

# ---------------------------------------------------------------------
# Utilities: device, normalization
# ---------------------------------------------------------------------


def infer_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def normalize_text(text: str) -> str:
    """Rough normalization: lowercase, strip punctuation/extra spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def append_line(path: Path, line: str) -> None:
    """Append a single line to a file, creating parents if needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{line}\n")
    except Exception as exc:
        print(f"[warn] Failed to append to {path}: {exc}", file=sys.stderr)


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
    answer_label: Optional[str] = None  # Ground-truth option label if available


@dataclass
class PerturbStats:
    changed_instances: int = 0
    total_instances: int = 0
    images_affected: int = 0
    gt_evaluable: int = 0
    right_to_wrong: int = 0
    wrong_to_right: int = 0
    right_to_right: int = 0
    wrong_to_wrong: int = 0


@dataclass
class SimilarityAccumulator:
    cos_sum: float = 0.0
    l2_sum: float = 0.0
    count: int = 0

    def update(self, cos_val: float, l2_val: float) -> None:
        self.cos_sum += cos_val
        self.l2_sum += l2_val
        self.count += 1

    def mean_cos(self) -> float:
        return self.cos_sum / self.count if self.count else 0.0

    def mean_l2(self) -> float:
        return self.l2_sum / self.count if self.count else 0.0


@dataclass
class EmbeddingStats:
    context_open: SimilarityAccumulator = field(default_factory=SimilarityAccumulator)
    context_mcq: SimilarityAccumulator = field(default_factory=SimilarityAccumulator)
    answer_open: SimilarityAccumulator = field(default_factory=SimilarityAccumulator)
    answer_mcq: SimilarityAccumulator = field(default_factory=SimilarityAccumulator)
    answer_mcq_free: SimilarityAccumulator = field(default_factory=SimilarityAccumulator)


@dataclass
class EmbeddingVizConfig:
    enabled: bool
    limit_samples: int
    out_dir: Path
    perplexity: float = 30.0


@dataclass
class DriftSampleRecord:
    sample_idx: int
    base_embs: Dict[str, torch.Tensor]
    per_type_dist: Dict[str, Dict[str, Dict[str, float]]]  # channel -> type -> metric -> val
    control_dist: Dict[str, Dict[str, float]]  # channel -> metric -> val


# ---------------------------------------------------------------------
# SEEDBench loading & prompt construction
# ---------------------------------------------------------------------


def _normalize_answer_label(
    raw: Any,
    options: Dict[str, str],
) -> Optional[str]:
    """
    Try to coerce a ground-truth column value into an option label.
    Supports letters, numeric indices (0/1-based), or matching option text.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None

    val = str(raw).strip()
    if not val:
        return None

    # Direct letter match (case-insensitive)
    letter = val.upper()
    if len(letter) == 1 and letter in options:
        return letter

    # Numeric index (0- or 1-based)
    if val.isdigit():
        idx = int(val)
        labels = list(options.keys())
        if 0 <= idx < len(labels):
            return labels[idx]
        if 1 <= idx <= len(labels):
            return labels[idx - 1]

    # Match by option text (exact or normalized)
    val_norm = normalize_text(val)
    for lab, opt_text in options.items():
        if val == opt_text or normalize_text(opt_text) == val_norm:
            return lab
    return None


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

        answer_label = None
        for col in ["answer", "answer_label", "label", "gt", "ground_truth", "answer_idx", "answer_index"]:
            if col in df.columns and not pd.isna(row[col]):
                answer_label = _normalize_answer_label(row[col], options)
                if answer_label:
                    break

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
                answer_label=answer_label,
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


def build_open_prompt(sample: Sample) -> str:
    """
    Open-ended prompt variant (no options).
    """
    parts: List[str] = []
    if sample.hint:
        parts.append(f"Hint: {sample.hint}")
    parts.append(f"Question: {sample.question}")
    parts.append("Please answer the question concisely.")
    return "\n".join(parts)


def build_open_with_options_prompt(sample: Sample) -> str:
    """
    Free-form answer prompt that still surfaces the options for context.
    """
    parts: List[str] = []
    if sample.hint:
        parts.append(f"Hint: {sample.hint}")
    parts.append(f"Question: {sample.question}")
    if sample.options:
        parts.append("Options (for your reference; answer freely):")
        for key, val in sample.options.items():
            parts.append(f"{key}. {val}")
    parts.append("Provide the best answer in your own words.")
    return "\n".join(parts)


def build_mcq_prompt(sample: Sample) -> str:
    """
    MCQ-style prompt variant (with options explicitly listed).
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


def _cpu_slots() -> List[str]:
    """Return a list of CPU slots for process-based parallelism."""
    cpu_count = max(1, os.cpu_count() or 1)
    return [f"cpu:{i}" for i in range(cpu_count)]


def canonical_device_id(device_id: str) -> str:
    """Map cpu:* slots back to 'cpu' for torch device construction."""
    return "cpu" if device_id.startswith("cpu") else device_id


def list_available_devices(
    requested: str | None,
    max_workers: Optional[int],
) -> List[str]:
    """
    Return a prioritized device list honoring an explicit request, otherwise:
      cuda GPUs (all), else mps, else CPU slots.
    """
    devs: List[str] = []
    req = (requested or "auto").lower()

    if req != "auto":
        if req.startswith("cuda"):
            if torch.cuda.is_available():
                if ":" in req:
                    devs = [req]
                else:
                    devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        elif req == "mps":
            if torch.backends.mps.is_available():
                devs = ["mps"]
        elif req == "cpu":
            devs = _cpu_slots()
        else:
            devs = [req]
    else:
        if torch.cuda.is_available():
            devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        elif torch.backends.mps.is_available():
            devs = ["mps"]
        else:
            devs = _cpu_slots()

    if not devs:
        devs = _cpu_slots()

    if max_workers is not None and max_workers > 0:
        devs = devs[:max_workers]
    return devs


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


def worker_process(
    worker_idx: int,
    device_id: str,
    args: argparse.Namespace,
    cache_dir: Path,
    samples: List[Sample],
    save_dir: Optional[Path],
    run_label_invariance: bool,
    run_embedding_invariance: bool,
    viz_config: Optional[EmbeddingVizConfig],
) -> None:
    """
    Top-level worker entrypoint for multiprocessing (must be picklable).
    """
    local_save_dir = save_dir / f"worker_{worker_idx}" if save_dir else None
    if local_save_dir is not None:
        local_save_dir.mkdir(parents=True, exist_ok=True)

    local_device = canonical_device_id(device_id)
    local_model_source = ensure_model_available(
        args.model_id,
        args.model_path,
        cache_dir=cache_dir,
        offline=args.offline,
    )
    print(
        f"[worker {worker_idx}] Using model from {local_model_source} on {local_device}",
        file=sys.stderr,
    )

    local_model, local_processor = prepare_model(
        str(local_model_source),
        device=local_device,
        cache_dir=str(cache_dir),
        local_files_only=args.offline,
    )

    summary_path = None
    if args.summary_file:
        base = Path(args.summary_file).expanduser().resolve()
        # Avoid clashes between workers by appending worker id
        if base.suffix:
            summary_path = base.with_name(f"{base.stem}_worker{worker_idx}{base.suffix}")
        else:
            summary_path = base.with_name(f"{base.name}_worker{worker_idx}")
        append_line(summary_path, f"[worker {worker_idx}] start")

    local_viz_config = None
    if viz_config is not None and viz_config.enabled:
        local_viz_config = EmbeddingVizConfig(
            enabled=True,
            limit_samples=viz_config.limit_samples,
            out_dir=viz_config.out_dir / f"worker_{worker_idx}",
            perplexity=viz_config.perplexity,
        )

    if run_label_invariance:
        run_experiment(
            local_model,
            local_processor,
            device=local_device,
            samples=samples,
            translate_pixels=args.translate_steps,
            padcrop_pixels=args.padcrop_steps,
            scale_factor=args.scale_factor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            compare_mode=args.compare_mode,
            save_changed_dir=local_save_dir,
            save_changed_limit=args.save_changed_limit,
            summary_path=summary_path,
            progress_prefix=f"[worker {worker_idx}] ",
        )
    if run_embedding_invariance:
        run_embedding_invariance_analysis(
            local_model,
            local_processor,
            device=local_device,
            samples=samples,
            translate_pixels=args.translate_steps,
            padcrop_pixels=args.padcrop_steps,
            scale_factor=args.scale_factor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            summary_path=summary_path,
            viz_config=local_viz_config,
            progress_prefix=f"[worker {worker_idx}] ",
        )


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


def get_context_length_cached(
    processor,
    image: Image.Image,
    prompt: str,
    device: str,
    cache: Dict[tuple[str, tuple[int, int]], int],
) -> int:
    """
    Cache context lengths keyed by (prompt, image.size) to avoid repeated
    tokenization for every perturbation of the same prompt/size.
    """
    key = (prompt, image.size)
    if key in cache:
        return cache[key]
    ctx = get_context_length(processor, image, prompt, device)
    cache[key] = ctx
    return ctx


@torch.inference_mode()
def get_context_embedding(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: str,
    context_len_cache: Optional[Dict[tuple[str, tuple[int, int]], int]] = None,
) -> Optional[torch.Tensor]:
    """
    Compute the context embedding (last hidden state at position context_len-1)
    for a prompt (open or MCQ).
    """
    cache = context_len_cache if context_len_cache is not None else {}
    context_len = get_context_length_cached(
        processor,
        image,
        prompt,
        device=device,
        cache=cache,
    )
    messages_user = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages_user,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]  # [1, L, H]
    if hidden.shape[1] < context_len:
        return None
    return hidden[0, context_len - 1].detach().cpu()


@torch.inference_mode()
def get_answer_embedding(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    answer_text: str,
    device: str,
    context_len_cache: Optional[Dict[tuple[str, tuple[int, int]], int]] = None,
) -> Optional[torch.Tensor]:
    """
    Mean-pooled hidden state over answer tokens (positions >= context_len).
    """
    cache = context_len_cache if context_len_cache is not None else {}
    context_len = get_context_length_cached(
        processor,
        image,
        prompt,
        device=device,
        cache=cache,
    )
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
            "content": [{"type": "text", "text": answer_text}],
        },
    ]
    inputs = processor.apply_chat_template(
        messages_full,
        tokenize=True,
        add_generation_prompt=False,
        padding=False,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]  # [1, L, H]
    seq_len = hidden.shape[1]
    if seq_len <= context_len:
        return None
    answer_states = hidden[0, context_len:seq_len]  # [ans_len, H]
    pooled = answer_states.mean(dim=0)
    return pooled.detach().cpu()


@torch.inference_mode()
def score_options_loglik_batch(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    options: Dict[str, str],
    device: str,
    context_len: int,
) -> Dict[str, float]:
    """
    Batch version of option scoring to reduce forward passes.
    Returns a mapping {label: loglikelihood}.
    """
    if not options:
        return {}

    messages_batch = []
    labels: List[str] = []
    for lab, opt_text in options.items():
        labels.append(lab)
        messages_batch.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": opt_text}],
                },
            ]
        )

    try:
        inputs_full = processor.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=False,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
    except Exception as exc:
        # Fallback to per-option scoring if batching is unsupported by the processor.
        global BATCH_LOGPROB_FALLBACK_WARNED
        if not BATCH_LOGPROB_FALLBACK_WARNED:
            print(f"[warn] Batch loglik scoring failed ({exc}); falling back to sequential.", file=sys.stderr)
            BATCH_LOGPROB_FALLBACK_WARNED = True
        scores: Dict[str, float] = {}
        for lab, opt_text in options.items():
            scores[lab] = score_option_loglik(
                model,
                processor,
                image,
                prompt,
                opt_text,
                device=device,
                context_len=context_len,
            )
        return scores
    inputs_full = {
        k: (v.to(device) if hasattr(v, "to") else v)
        for k, v in inputs_full.items()
    }

    input_ids = inputs_full["input_ids"]  # [B, L]
    attn_mask = inputs_full.get("attention_mask")
    outputs = model(**inputs_full)
    logits = outputs.logits  # [B, L, vocab]
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

    scores: Dict[str, float] = {}
    batch_size = input_ids.shape[0]
    for i in range(batch_size):
        seq_len = (
            int(attn_mask[i].sum().item())
            if attn_mask is not None
            else int(input_ids.shape[1])
        )
        if seq_len <= context_len:
            continue
        target_ids = input_ids[i, context_len:seq_len]
        pred_lp = log_probs[i, context_len - 1 : seq_len - 1, :]
        score = (
            pred_lp.gather(-1, target_ids.unsqueeze(-1))
            .squeeze(-1)
            .sum()
            .item()
        )
        scores[labels[i]] = score
    return scores


@torch.inference_mode()
def choose_label_via_loglik(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    options: Dict[str, str],
    device: str,
    context_len_cache: Optional[Dict[tuple[str, tuple[int, int]], int]] = None,
) -> Optional[str]:
    """
    For this image + prompt, compute log-likelihood for each option_text
    and return the argmax option label (A/B/C/...).

    Returns:
        label (e.g. "A", "B", ...) or None if no options.
    """
    if not options:
        return None

    cache = context_len_cache if context_len_cache is not None else {}
    context_len = get_context_length_cached(
        processor,
        image,
        prompt,
        device=device,
        cache=cache,
    )

    scores = score_options_loglik_batch(
        model,
        processor,
        image,
        prompt,
        options,
        device=device,
        context_len=context_len,
    )
    if not scores:
        return None

    best_label = max(scores, key=scores.get)
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
    if compare_mode in {"text", "text_mcq"}:
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
    summary_path: Optional[Path],
    progress_prefix: str = "",
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
    context_len_cache: Dict[tuple[str, tuple[int, int]], int] = {}

    N_total = len(samples)
    N_used = 0  # samples with a valid base representation
    samples_with_gt = 0  # samples where a ground-truth label is available
    base_correct_samples = 0

    print(f"[info] Running invariance experiment on {N_total} samples...")
    if summary_path is not None:
        append_line(summary_path, f"{progress_prefix}start label invariance on {N_total} samples")
    for i, sample in enumerate(samples, start=1):
        if not sample.image_path.exists():
            print(f"[warn] Missing image: {sample.image_path}", file=sys.stderr)
            continue

        img = Image.open(sample.image_path).convert("RGB")
        img = resize_max_side(img, max_side=1024)
        if compare_mode == "text_mcq":
            prompt = build_open_with_options_prompt(sample)
        else:
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
                context_len_cache=context_len_cache,
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

        gt_label = sample.answer_label
        base_correct: Optional[bool] = None
        if gt_label is not None:
            base_correct = base_repr == gt_label
            samples_with_gt += 1
            if base_correct:
                base_correct_samples += 1
            correctness_text = "correct" if base_correct else "WRONG"
            print(f"[base vs GT] {base_repr} vs {gt_label} ({correctness_text})")
        else:
            print("[base vs GT] ground truth unavailable for this sample.")

        N_used += 1

        per_image_changed_any = False
        per_image_changed_type: Dict[str, bool] = {t: False for t in perturb_types}
        reprs_for_entropy: List[Optional[str]] = [base_repr]

        def update_gt_confusion(ps: PerturbStats, pert_repr: Optional[str]) -> None:
            """Track correctness transitions vs. ground truth for one perturbation."""
            if base_correct is None or gt_label is None or pert_repr is None:
                return
            ps.gt_evaluable += 1
            any_stats.gt_evaluable += 1
            if base_correct:
                if pert_repr == gt_label:
                    ps.right_to_right += 1
                    any_stats.right_to_right += 1
                else:
                    ps.right_to_wrong += 1
                    any_stats.right_to_wrong += 1
            else:
                if pert_repr == gt_label:
                    ps.wrong_to_right += 1
                    any_stats.wrong_to_right += 1
                else:
                    ps.wrong_to_wrong += 1
                    any_stats.wrong_to_wrong += 1

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
                    context_len_cache=context_len_cache,
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
                update_gt_confusion(stats["Translation"], r)
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
                update_gt_confusion(stats["Pad/Crop"], r)
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
            update_gt_confusion(stats["Scale"], r)
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
                update_gt_confusion(stats["Scale+Pad"], r)
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
                update_gt_confusion(stats["TextOverlay"], r)
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
                update_gt_confusion(stats["Rotation"], r)
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
    def safe_div(a: int, b: int) -> float:
        return float(a) / float(b) if b > 0 else 0.0

    summary_lines: List[str] = []
    summary_lines.append("\n========== SUMMARY ==========")
    summary_lines.append(f"Total samples in TSV: {N_total}")
    summary_lines.append(f"Samples used (with valid base representation): {N_used}")
    summary_lines.append(f"Samples with ground truth: {samples_with_gt}")
    summary_lines.append(
        f"Base accuracy vs ground truth: {safe_div(base_correct_samples, samples_with_gt):.3f} "
        f"({base_correct_samples}/{samples_with_gt})"
    )

    header = f"{'Type':<12}  {'AVg':>8}  {'Ve':>8}  (#img_affected / #img_used, #changed / #total)"
    summary_lines.append(header)
    summary_lines.append("-" * len(header))

    for t in perturb_types:
        st = stats[t]
        avg = safe_div(st.changed_instances, st.total_instances)
        ve = safe_div(st.images_affected, N_used)
        summary_lines.append(
            f"{t:<12}  {avg:8.3f}  {ve:8.3f}  "
            f"({st.images_affected}/{N_used}, {st.changed_instances}/{st.total_instances})"
        )

    avg_any = safe_div(any_stats.changed_instances, any_stats.total_instances)
    ve_any = safe_div(any_stats.images_affected, N_used)
    summary_lines.append(
        f"{'Any':<12}  {avg_any:8.3f}  {ve_any:8.3f}  "
        f"({any_stats.images_affected}/{N_used}, "
        f"{any_stats.changed_instances}/{any_stats.total_instances})"
    )

    conf_header = f"{'Type':<12}  {'R->W':>7}  {'W->R':>7}  {'R->R':>7}  {'W->W':>7}  {'GT inst.':>9}"
    summary_lines.append("\nConfusion vs ground truth (perturbation instances)")
    summary_lines.append(conf_header)
    summary_lines.append("-" * len(conf_header))
    for t in perturb_types:
        st = stats[t]
        summary_lines.append(
            f"{t:<12}  {st.right_to_wrong:7d}  {st.wrong_to_right:7d}  "
            f"{st.right_to_right:7d}  {st.wrong_to_wrong:7d}  {st.gt_evaluable:9d}"
        )
    summary_lines.append(
        f"{'Any':<12}  {any_stats.right_to_wrong:7d}  {any_stats.wrong_to_right:7d}  "
        f"{any_stats.right_to_right:7d}  {any_stats.wrong_to_wrong:7d}  {any_stats.gt_evaluable:9d}"
    )

    for line in summary_lines:
        print(line)

    if summary_path is not None:
        try:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                for line in summary_lines:
                    f.write(f"{line}\n")
            print(f"[info] Summary written to {summary_path}")
        except Exception as exc:
            print(f"[warn] Failed to write summary to {summary_path}: {exc}", file=sys.stderr)
    if summary_path is not None:
        append_line(summary_path, f"{progress_prefix}completed label invariance on {N_used}/{N_total} samples")


def _record_similarity(
    acc: SimilarityAccumulator,
    acc_any: SimilarityAccumulator,
    base: Optional[torch.Tensor],
    pert: Optional[torch.Tensor],
) -> None:
    """Update accumulators if both embeddings are present."""
    if base is None or pert is None:
        return
    cos = F.cosine_similarity(
        base.unsqueeze(0), pert.unsqueeze(0), dim=-1
    ).item()
    l2 = torch.norm(base - pert, p=2).item()
    acc.update(cos, l2)
    acc_any.update(cos, l2)


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return 1 - cosine similarity."""
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()
    return 1.0 - cos


def _cohen_d(a: List[float], b: List[float]) -> float:
    """
    Compute Cohen's d between two lists. Returns 0.0 if insufficient variance/data.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / float(len(a) + len(b) - 2)
    if pooled <= 0:
        return 0.0
    return (mean_a - mean_b) / math.sqrt(pooled)


class EmbeddingVizCollector:
    """
    Collect embeddings for a subset of samples and save PCA/t-SNE plots.
    """

    def __init__(self, config: Optional[EmbeddingVizConfig]):
        self.config = config
        self.selected_samples: set[int] = set()
        self.points: Dict[str, List[Dict[str, Any]]] = {}

    def enabled(self) -> bool:
        return self.config is not None and self.config.enabled

    def wants_sample(self, sample_idx: int) -> bool:
        if not self.enabled():
            return False
        if sample_idx in self.selected_samples:
            return True
        if len(self.selected_samples) < self.config.limit_samples:
            self.selected_samples.add(sample_idx)
            return True
        return False

    def add(
        self,
        channel: str,
        sample_idx: int,
        pert_type: str,
        variant: str,
        emb: Optional[torch.Tensor],
    ) -> None:
        if not self.enabled() or emb is None:
            return
        if not self.wants_sample(sample_idx):
            return
        vec = emb.detach().cpu().numpy()
        self.points.setdefault(channel, []).append(
            {
                "vec": vec,
                "pert": pert_type,
                "variant": variant,
                "sample": sample_idx,
            }
        )

    def _plot_2d(
        self,
        coords: np.ndarray,
        entries: List[Dict[str, Any]],
        title: str,
        out_path: Path,
    ) -> None:
        uniq_perts = sorted({e["pert"] for e in entries})
        cmap = plt.get_cmap("tab20")
        color_map = {p: cmap(i % 20) for i, p in enumerate(uniq_perts)}
        plt.figure(figsize=(6, 5))
        for pert in uniq_perts:
            idxs = [i for i, e in enumerate(entries) if e["pert"] == pert]
            marker = "o" if pert == "Base" else "x"
            plt.scatter(
                coords[idxs, 0],
                coords[idxs, 1],
                label=pert,
                c=[color_map[pert]],
                marker=marker,
                alpha=0.8,
            )
        plt.title(title)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def _save_channel(self, channel: str, entries: List[Dict[str, Any]]) -> None:
        X = np.stack([e["vec"] for e in entries], axis=0)
        out_dir = self.config.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # PCA
        pca = PCA(n_components=2)
        coords_pca = pca.fit_transform(X)
        self._plot_2d(
            coords_pca,
            entries,
            title=f"{channel} PCA",
            out_path=out_dir / f"{channel}_pca.png",
        )

        # t-SNE (safe perplexity)
        perp = min(self.config.perplexity, max(1, len(entries) - 1))
        tsne_attempts = [
            {"learning_rate": "auto", "max_iter": 1000},
            {"learning_rate": "auto"},
            {"n_iter": 1000},
            {},
        ]
        tsne = None
        for extra in tsne_attempts:
            try:
                tsne = TSNE(
                    n_components=2,
                    perplexity=perp,
                    init="pca",
                    **extra,
                )
                break
            except TypeError as exc:
                print(
                    f"[warn] TSNE init fallback for {channel}: {exc}; trying next variant.",
                    file=sys.stderr,
                )
                continue
        if tsne is None:
            raise RuntimeError("Failed to initialize TSNE for embedding viz.")
        coords_tsne = tsne.fit_transform(X)
        self._plot_2d(
            coords_tsne,
            entries,
            title=f"{channel} t-SNE",
            out_path=out_dir / f"{channel}_tsne.png",
        )

    def save(self) -> None:
        if not self.enabled():
            return
        for channel, entries in self.points.items():
            if len(entries) < 2:
                continue
            try:
                self._save_channel(channel, entries)
            except Exception as exc:
                print(
                    f"[warn] Failed to save embedding visualization for {channel}: {exc}",
                    file=sys.stderr,
                )


def run_embedding_invariance_analysis(
    model,
    processor,
    device: str,
    samples: List[Sample],
    translate_pixels: List[int],
    padcrop_pixels: List[int],
    scale_factor: float,
    max_new_tokens: int,
    temperature: float,
    summary_path: Optional[Path],
    viz_config: Optional[EmbeddingVizConfig] = None,
    progress_prefix: str = "",
) -> None:
    """
    Embedding-based invariance analysis (context + answer token pooling).
    Does not alter existing log-likelihood behavior.
    """
    perturb_types = [
        "Translation",
        "Pad/Crop",
        "Scale",
        "Scale+Pad",
        "TextOverlay",
        "Rotation",
    ]
    stats: Dict[str, EmbeddingStats] = {t: EmbeddingStats() for t in perturb_types}
    any_stats = EmbeddingStats()
    context_len_cache: Dict[tuple[str, tuple[int, int]], int] = {}
    viz_collector = EmbeddingVizCollector(viz_config)

    N_total = len(samples)
    print(f"[info] Running embedding invariance analysis on {N_total} samples...")
    if summary_path is not None:
        append_line(summary_path, f"{progress_prefix}start embedding invariance on {N_total} samples")

    for i, sample in enumerate(samples, start=1):
        if not sample.image_path.exists():
            print(f"[warn] Missing image: {sample.image_path}", file=sys.stderr)
            continue

        img = Image.open(sample.image_path).convert("RGB")
        img = resize_max_side(img, max_side=1024)

        prompt_open = build_open_prompt(sample)
        prompt_mcq = build_mcq_prompt(sample)

        base_ctx_open = get_context_embedding(
            model,
            processor,
            img,
            prompt_open,
            device=device,
            context_len_cache=context_len_cache,
        )
        base_ctx_mcq = get_context_embedding(
            model,
            processor,
            img,
            prompt_mcq,
            device=device,
            context_len_cache=context_len_cache,
        )

        base_open_answer = generate_free_answer(
            model,
            processor,
            img,
            prompt_open,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        base_ans_emb_open = get_answer_embedding(
            model,
            processor,
            img,
            prompt_open,
            base_open_answer,
            device=device,
            context_len_cache=context_len_cache,
        )

        base_mcq_label = choose_label_via_loglik(
            model,
            processor,
            img,
            prompt_mcq,
            sample.options,
            device=device,
            context_len_cache=context_len_cache,
        )
        base_mcq_answer_text = (
            sample.options.get(base_mcq_label) if base_mcq_label else None
        )
        base_mcq_free_answer = generate_free_answer(
            model,
            processor,
            img,
            prompt_mcq,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        base_ans_emb_mcq = (
            get_answer_embedding(
                model,
                processor,
                img,
                prompt_mcq,
                base_mcq_answer_text,
                device=device,
                context_len_cache=context_len_cache,
            )
            if base_mcq_answer_text
            else None
        )
        base_ans_emb_mcq_free = get_answer_embedding(
            model,
            processor,
            img,
            prompt_mcq,
            base_mcq_free_answer,
            device=device,
            context_len_cache=context_len_cache,
        )
        if viz_collector.enabled():
            viz_collector.add("ctx_open", sample.index, "Base", "base", base_ctx_open)
            viz_collector.add("ctx_mcq", sample.index, "Base", "base", base_ctx_mcq)
            viz_collector.add(
                "ans_open", sample.index, "Base", "base", base_ans_emb_open
            )
            viz_collector.add(
                "ans_mcq", sample.index, "Base", "base", base_ans_emb_mcq
            )
            viz_collector.add(
                "ans_mcq_free", sample.index, "Base", "base", base_ans_emb_mcq_free
            )

        # Translation
        for n in translate_pixels:
            if n == 0:
                continue
            vimg = cyclic_horizontal_shift(img, n)
            ctx_open = get_context_embedding(
                model,
                processor,
                vimg,
                prompt_open,
                device=device,
                context_len_cache=context_len_cache,
            )
            ctx_mcq = get_context_embedding(
                model,
                processor,
                vimg,
                prompt_mcq,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["Translation"].context_open, any_stats.context_open, base_ctx_open, ctx_open
            )
            _record_similarity(
                stats["Translation"].context_mcq, any_stats.context_mcq, base_ctx_mcq, ctx_mcq
            )

            ans_open = generate_free_answer(
                model,
                processor,
                vimg,
                prompt_open,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            ans_emb_open = get_answer_embedding(
                model,
                processor,
                vimg,
                prompt_open,
                ans_open,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["Translation"].answer_open, any_stats.answer_open, base_ans_emb_open, ans_emb_open
            )

            mcq_free_answer = generate_free_answer(
                model,
                processor,
                vimg,
                prompt_mcq,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            mcq_free_emb = get_answer_embedding(
                model,
                processor,
                vimg,
                prompt_mcq,
                mcq_free_answer,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["Translation"].answer_mcq_free,
                any_stats.answer_mcq_free,
                base_ans_emb_mcq_free,
                mcq_free_emb,
            )

            pert_mcq_label = choose_label_via_loglik(
                model,
                processor,
                vimg,
                prompt_mcq,
                sample.options,
                device=device,
                context_len_cache=context_len_cache,
            )
            pert_mcq_text = (
                sample.options.get(pert_mcq_label) if pert_mcq_label else None
            )
            if pert_mcq_text:
                ans_emb_mcq = get_answer_embedding(
                    model,
                    processor,
                    vimg,
                    prompt_mcq,
                    pert_mcq_text,
                    device=device,
                    context_len_cache=context_len_cache,
                )
                _record_similarity(
                    stats["Translation"].answer_mcq,
                    any_stats.answer_mcq,
                    base_ans_emb_mcq,
                    ans_emb_mcq,
                )
            if viz_collector.enabled():
                viz_collector.add(
                    "ctx_open",
                    sample.index,
                    "Translation",
                    f"shift_{n}",
                    ctx_open,
                )
                viz_collector.add(
                    "ctx_mcq",
                    sample.index,
                    "Translation",
                    f"shift_{n}",
                    ctx_mcq,
                )
                viz_collector.add(
                    "ans_open",
                    sample.index,
                    "Translation",
                    f"shift_{n}",
                    ans_emb_open,
                )
                viz_collector.add(
                    "ans_mcq",
                    sample.index,
                    "Translation",
                    f"shift_{n}",
                    ans_emb_mcq if pert_mcq_text else None,
                )
                viz_collector.add(
                    "ans_mcq_free",
                    sample.index,
                    "Translation",
                    f"shift_{n}",
                    mcq_free_emb,
                )

        # Pad/Crop
        for n in padcrop_pixels:
            if n == 0:
                continue
            vimg = pad_or_crop(img, n)
            ctx_open = get_context_embedding(
                model,
                processor,
                vimg,
                prompt_open,
                device=device,
                context_len_cache=context_len_cache,
            )
            ctx_mcq = get_context_embedding(
                model,
                processor,
                vimg,
                prompt_mcq,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["Pad/Crop"].context_open, any_stats.context_open, base_ctx_open, ctx_open
            )
            _record_similarity(
                stats["Pad/Crop"].context_mcq, any_stats.context_mcq, base_ctx_mcq, ctx_mcq
            )

            ans_open = generate_free_answer(
                model,
                processor,
                vimg,
                prompt_open,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            ans_emb_open = get_answer_embedding(
                model,
                processor,
                vimg,
                prompt_open,
                ans_open,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["Pad/Crop"].answer_open, any_stats.answer_open, base_ans_emb_open, ans_emb_open
            )

            mcq_free_answer = generate_free_answer(
                model,
                processor,
                vimg,
                prompt_mcq,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            mcq_free_emb = get_answer_embedding(
                model,
                processor,
                vimg,
                prompt_mcq,
                mcq_free_answer,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["Pad/Crop"].answer_mcq_free,
                any_stats.answer_mcq_free,
                base_ans_emb_mcq_free,
                mcq_free_emb,
            )

            pert_mcq_label = choose_label_via_loglik(
                model,
                processor,
                vimg,
                prompt_mcq,
                sample.options,
                device=device,
                context_len_cache=context_len_cache,
            )
            pert_mcq_text = (
                sample.options.get(pert_mcq_label) if pert_mcq_label else None
            )
            if pert_mcq_text:
                ans_emb_mcq = get_answer_embedding(
                    model,
                    processor,
                    vimg,
                    prompt_mcq,
                    pert_mcq_text,
                    device=device,
                    context_len_cache=context_len_cache,
                )
                _record_similarity(
                    stats["Pad/Crop"].answer_mcq,
                    any_stats.answer_mcq,
                    base_ans_emb_mcq,
                    ans_emb_mcq,
                )
            if viz_collector.enabled():
                viz_collector.add(
                    "ctx_open",
                    sample.index,
                    "Pad/Crop",
                    f"padcrop_{n}",
                    ctx_open,
                )
                viz_collector.add(
                    "ctx_mcq",
                    sample.index,
                    "Pad/Crop",
                    f"padcrop_{n}",
                    ctx_mcq,
                )
                viz_collector.add(
                    "ans_open",
                    sample.index,
                    "Pad/Crop",
                    f"padcrop_{n}",
                    ans_emb_open,
                )
                viz_collector.add(
                    "ans_mcq",
                    sample.index,
                    "Pad/Crop",
                    f"padcrop_{n}",
                    ans_emb_mcq if pert_mcq_text else None,
                )
                viz_collector.add(
                    "ans_mcq_free",
                    sample.index,
                    "Pad/Crop",
                    f"padcrop_{n}",
                    mcq_free_emb,
                )

        # Scale
        vimg_scale = scale_image(img, scale_factor)
        ctx_open = get_context_embedding(
            model,
            processor,
            vimg_scale,
            prompt_open,
            device=device,
            context_len_cache=context_len_cache,
        )
        ctx_mcq = get_context_embedding(
            model,
            processor,
            vimg_scale,
            prompt_mcq,
            device=device,
            context_len_cache=context_len_cache,
        )
        _record_similarity(
            stats["Scale"].context_open, any_stats.context_open, base_ctx_open, ctx_open
        )
        _record_similarity(
            stats["Scale"].context_mcq, any_stats.context_mcq, base_ctx_mcq, ctx_mcq
        )

        ans_open = generate_free_answer(
            model,
            processor,
            vimg_scale,
            prompt_open,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        ans_emb_open = get_answer_embedding(
            model,
            processor,
            vimg_scale,
            prompt_open,
            ans_open,
            device=device,
            context_len_cache=context_len_cache,
        )
        _record_similarity(
            stats["Scale"].answer_open, any_stats.answer_open, base_ans_emb_open, ans_emb_open
        )

        mcq_free_answer = generate_free_answer(
            model,
            processor,
            vimg_scale,
            prompt_mcq,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        mcq_free_emb = get_answer_embedding(
            model,
            processor,
            vimg_scale,
            prompt_mcq,
            mcq_free_answer,
            device=device,
            context_len_cache=context_len_cache,
        )
        _record_similarity(
            stats["Scale"].answer_mcq_free,
            any_stats.answer_mcq_free,
            base_ans_emb_mcq_free,
            mcq_free_emb,
        )

        pert_mcq_label = choose_label_via_loglik(
            model,
            processor,
            vimg_scale,
            prompt_mcq,
            sample.options,
            device=device,
            context_len_cache=context_len_cache,
        )
        pert_mcq_text = sample.options.get(pert_mcq_label) if pert_mcq_label else None
        if pert_mcq_text:
            ans_emb_mcq = get_answer_embedding(
                model,
                processor,
                vimg_scale,
                prompt_mcq,
                pert_mcq_text,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["Scale"].answer_mcq,
                any_stats.answer_mcq,
                base_ans_emb_mcq,
                ans_emb_mcq,
            )
        if viz_collector.enabled():
            viz_collector.add(
                "ctx_open",
                sample.index,
                "Scale",
                f"scale_{scale_factor}",
                ctx_open,
            )
            viz_collector.add(
                "ctx_mcq",
                sample.index,
                "Scale",
                f"scale_{scale_factor}",
                ctx_mcq,
            )
            viz_collector.add(
                "ans_open",
                sample.index,
                "Scale",
                f"scale_{scale_factor}",
                ans_emb_open,
            )
            viz_collector.add(
                "ans_mcq",
                sample.index,
                "Scale",
                f"scale_{scale_factor}",
                ans_emb_mcq if pert_mcq_text else None,
            )
            viz_collector.add(
                "ans_mcq_free",
                sample.index,
                "Scale",
                f"scale_{scale_factor}",
                mcq_free_emb,
            )

        # Scale+Pad
        for bg in ["black", "white"]:
            vimg_sp = scale_and_pad(img, scale_factor, background=bg)
            ctx_open = get_context_embedding(
                model,
                processor,
                vimg_sp,
                prompt_open,
                device=device,
                context_len_cache=context_len_cache,
            )
            ctx_mcq = get_context_embedding(
                model,
                processor,
                vimg_sp,
                prompt_mcq,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["Scale+Pad"].context_open, any_stats.context_open, base_ctx_open, ctx_open
            )
            _record_similarity(
                stats["Scale+Pad"].context_mcq, any_stats.context_mcq, base_ctx_mcq, ctx_mcq
            )

            ans_open = generate_free_answer(
                model,
                processor,
                vimg_sp,
                prompt_open,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            ans_emb_open = get_answer_embedding(
                model,
                processor,
                vimg_sp,
                prompt_open,
                ans_open,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["Scale+Pad"].answer_open,
                any_stats.answer_open,
                base_ans_emb_open,
                ans_emb_open,
            )

            mcq_free_answer = generate_free_answer(
                model,
                processor,
                vimg_sp,
                prompt_mcq,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            mcq_free_emb = get_answer_embedding(
                model,
                processor,
                vimg_sp,
                prompt_mcq,
                mcq_free_answer,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["Scale+Pad"].answer_mcq_free,
                any_stats.answer_mcq_free,
                base_ans_emb_mcq_free,
                mcq_free_emb,
            )

            pert_mcq_label = choose_label_via_loglik(
                model,
                processor,
                vimg_sp,
                prompt_mcq,
                sample.options,
                device=device,
                context_len_cache=context_len_cache,
            )
            pert_mcq_text = (
                sample.options.get(pert_mcq_label) if pert_mcq_label else None
            )
            if pert_mcq_text:
                ans_emb_mcq = get_answer_embedding(
                    model,
                    processor,
                    vimg_sp,
                    prompt_mcq,
                    pert_mcq_text,
                    device=device,
                    context_len_cache=context_len_cache,
                )
                _record_similarity(
                    stats["Scale+Pad"].answer_mcq,
                    any_stats.answer_mcq,
                    base_ans_emb_mcq,
                    ans_emb_mcq,
                )
            if viz_collector.enabled():
                viz_collector.add(
                    "ctx_open",
                    sample.index,
                    "Scale+Pad",
                    f"scale_pad_{bg}",
                    ctx_open,
                )
                viz_collector.add(
                    "ctx_mcq",
                    sample.index,
                    "Scale+Pad",
                    f"scale_pad_{bg}",
                    ctx_mcq,
                )
                viz_collector.add(
                    "ans_open",
                    sample.index,
                    "Scale+Pad",
                    f"scale_pad_{bg}",
                    ans_emb_open,
                )
                viz_collector.add(
                    "ans_mcq",
                    sample.index,
                    "Scale+Pad",
                    f"scale_pad_{bg}",
                    ans_emb_mcq if pert_mcq_text else None,
                )
                viz_collector.add(
                    "ans_mcq_free",
                    sample.index,
                    "Scale+Pad",
                    f"scale_pad_{bg}",
                    mcq_free_emb,
                )

        # TextOverlay
        other_labels = [lab for lab in sample.options.keys() if lab != base_mcq_label]
        other_labels = other_labels[:3]
        TEXT_OVERLAY_PHRASES = [f"Answer is {lab}" for lab in other_labels] or [
            "Answer is B",
            "Answer is C",
            "Answer is D",
        ]
        for phrase in TEXT_OVERLAY_PHRASES:
            vimg_txt = text_overlay(img, phrase)
            ctx_open = get_context_embedding(
                model,
                processor,
                vimg_txt,
                prompt_open,
                device=device,
                context_len_cache=context_len_cache,
            )
            ctx_mcq = get_context_embedding(
                model,
                processor,
                vimg_txt,
                prompt_mcq,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["TextOverlay"].context_open, any_stats.context_open, base_ctx_open, ctx_open
            )
            _record_similarity(
                stats["TextOverlay"].context_mcq, any_stats.context_mcq, base_ctx_mcq, ctx_mcq
            )

            ans_open = generate_free_answer(
                model,
                processor,
                vimg_txt,
                prompt_open,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            ans_emb_open = get_answer_embedding(
                model,
                processor,
                vimg_txt,
                prompt_open,
                ans_open,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["TextOverlay"].answer_open,
                any_stats.answer_open,
                base_ans_emb_open,
                ans_emb_open,
            )

            mcq_free_answer = generate_free_answer(
                model,
                processor,
                vimg_txt,
                prompt_mcq,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            mcq_free_emb = get_answer_embedding(
                model,
                processor,
                vimg_txt,
                prompt_mcq,
                mcq_free_answer,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["TextOverlay"].answer_mcq_free,
                any_stats.answer_mcq_free,
                base_ans_emb_mcq_free,
                mcq_free_emb,
            )

            pert_mcq_label = choose_label_via_loglik(
                model,
                processor,
                vimg_txt,
                prompt_mcq,
                sample.options,
                device=device,
                context_len_cache=context_len_cache,
            )
            pert_mcq_text = (
                sample.options.get(pert_mcq_label) if pert_mcq_label else None
            )
            if pert_mcq_text:
                ans_emb_mcq = get_answer_embedding(
                    model,
                    processor,
                    vimg_txt,
                    prompt_mcq,
                    pert_mcq_text,
                    device=device,
                    context_len_cache=context_len_cache,
                )
                _record_similarity(
                    stats["TextOverlay"].answer_mcq,
                    any_stats.answer_mcq,
                    base_ans_emb_mcq,
                    ans_emb_mcq,
                )
            if viz_collector.enabled():
                viz_collector.add(
                    "ctx_open",
                    sample.index,
                    "TextOverlay",
                    phrase,
                    ctx_open,
                )
                viz_collector.add(
                    "ctx_mcq",
                    sample.index,
                    "TextOverlay",
                    phrase,
                    ctx_mcq,
                )
                viz_collector.add(
                    "ans_open",
                    sample.index,
                    "TextOverlay",
                    phrase,
                    ans_emb_open,
                )
                viz_collector.add(
                    "ans_mcq",
                    sample.index,
                    "TextOverlay",
                    phrase,
                    ans_emb_mcq if pert_mcq_text else None,
                )
                viz_collector.add(
                    "ans_mcq_free",
                    sample.index,
                    "TextOverlay",
                    phrase,
                    mcq_free_emb,
                )

        # Rotation
        for angle in (-30.0, 30.0):
            vimg_rot = rotate_image(img, angle)
            ctx_open = get_context_embedding(
                model,
                processor,
                vimg_rot,
                prompt_open,
                device=device,
                context_len_cache=context_len_cache,
            )
            ctx_mcq = get_context_embedding(
                model,
                processor,
                vimg_rot,
                prompt_mcq,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["Rotation"].context_open, any_stats.context_open, base_ctx_open, ctx_open
            )
            _record_similarity(
                stats["Rotation"].context_mcq, any_stats.context_mcq, base_ctx_mcq, ctx_mcq
            )

            ans_open = generate_free_answer(
                model,
                processor,
                vimg_rot,
                prompt_open,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            ans_emb_open = get_answer_embedding(
                model,
                processor,
                vimg_rot,
                prompt_open,
                ans_open,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["Rotation"].answer_open,
                any_stats.answer_open,
                base_ans_emb_open,
                ans_emb_open,
            )

            mcq_free_answer = generate_free_answer(
                model,
                processor,
                vimg_rot,
                prompt_mcq,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            mcq_free_emb = get_answer_embedding(
                model,
                processor,
                vimg_rot,
                prompt_mcq,
                mcq_free_answer,
                device=device,
                context_len_cache=context_len_cache,
            )
            _record_similarity(
                stats["Rotation"].answer_mcq_free,
                any_stats.answer_mcq_free,
                base_ans_emb_mcq_free,
                mcq_free_emb,
            )

            pert_mcq_label = choose_label_via_loglik(
                model,
                processor,
                vimg_rot,
                prompt_mcq,
                sample.options,
                device=device,
                context_len_cache=context_len_cache,
            )
            pert_mcq_text = (
                sample.options.get(pert_mcq_label) if pert_mcq_label else None
            )
            if pert_mcq_text:
                ans_emb_mcq = get_answer_embedding(
                    model,
                    processor,
                    vimg_rot,
                    prompt_mcq,
                    pert_mcq_text,
                    device=device,
                    context_len_cache=context_len_cache,
                )
                _record_similarity(
                    stats["Rotation"].answer_mcq,
                    any_stats.answer_mcq,
                    base_ans_emb_mcq,
                    ans_emb_mcq,
                )
            if viz_collector.enabled():
                variant = f"rot_{int(angle)}"
                viz_collector.add(
                    "ctx_open",
                    sample.index,
                    "Rotation",
                    variant,
                    ctx_open,
                )
                viz_collector.add(
                    "ctx_mcq",
                    sample.index,
                    "Rotation",
                    variant,
                    ctx_mcq,
                )
                viz_collector.add(
                    "ans_open",
                    sample.index,
                    "Rotation",
                    variant,
                    ans_emb_open,
                )
                viz_collector.add(
                    "ans_mcq",
                    sample.index,
                    "Rotation",
                    variant,
                    ans_emb_mcq if pert_mcq_text else None,
                )
                viz_collector.add(
                    "ans_mcq_free",
                    sample.index,
                    "Rotation",
                    variant,
                    mcq_free_emb,
                )

    def fmt_row(name: str, st: EmbeddingStats) -> str:
        return (
            f"{name:<12}  "
            f"ctx-open cos={st.context_open.mean_cos():6.3f} l2={st.context_open.mean_l2():6.3f} n={st.context_open.count:<4d} | "
            f"ctx-mcq cos={st.context_mcq.mean_cos():6.3f} l2={st.context_mcq.mean_l2():6.3f} n={st.context_mcq.count:<4d} | "
            f"ans-open cos={st.answer_open.mean_cos():6.3f} l2={st.answer_open.mean_l2():6.3f} n={st.answer_open.count:<4d} | "
            f"ans-mcq cos={st.answer_mcq.mean_cos():6.3f} l2={st.answer_mcq.mean_l2():6.3f} n={st.answer_mcq.count:<4d} | "
            f"ans-mcq-free cos={st.answer_mcq_free.mean_cos():6.3f} l2={st.answer_mcq_free.mean_l2():6.3f} n={st.answer_mcq_free.count:<4d}"
        )

    summary_lines: List[str] = []
    summary_lines.append("\n====== EMBEDDING INVARIANCE (similarity to base) ======")
    summary_lines.append(
        "Per-perturbation mean cosine / L2 (context and pooled answers: open, MCQ argmax, MCQ free)."
    )
    for t in perturb_types:
        summary_lines.append(fmt_row(t, stats[t]))
    summary_lines.append(fmt_row("Any", any_stats))

    for line in summary_lines:
        print(line)

    if summary_path is not None:
        try:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "a", encoding="utf-8") as f:
                for line in summary_lines:
                    f.write(f"{line}\n")
            print(f"[info] Embedding summary appended to {summary_path}")
        except Exception as exc:
            print(f"[warn] Failed to write embedding summary to {summary_path}: {exc}", file=sys.stderr)
    viz_collector.save()
    if summary_path is not None:
        append_line(summary_path, f"{progress_prefix}completed embedding invariance on {N_total} samples")


def run_embedding_drift_analysis(
    model,
    processor,
    device: str,
    samples: List[Sample],
    translate_pixels: List[int],
    padcrop_pixels: List[int],
    scale_factor: float,
    max_new_tokens: int,
    temperature: float,
    sample_limit: int,
    control_k: int,
    out_dir: Path,
    seed: int,
    summary_path: Optional[Path],
    progress_prefix: str = "",
) -> None:
    """
    Drift analysis:
      - Sample up to `sample_limit` examples with valid embeddings.
      - For each perturbation type, compute mean base-vs-perturb distance.
      - Control: mean distance base vs K random other bases.
      - Aggregate mean/std and Cohen's d; save hist/KDE plots per channel/type.
    """
    rng = random.Random(seed)
    channel_names = ["ctx_open", "ctx_mcq", "ans_open", "ans_mcq", "ans_mcq_free"]
    perturb_types = [
        "Translation",
        "Pad/Crop",
        "Scale",
        "Scale+Pad",
        "TextOverlay",
        "Rotation",
    ]

    candidates = [s for s in samples if s.image_path.exists()]
    rng.shuffle(candidates)
    selected = candidates[: min(sample_limit, len(candidates))]
    print(f"[drift] Selected {len(selected)} samples (limit={sample_limit})")

    context_len_cache: Dict[tuple[str, tuple[int, int]], int] = {}
    records: List[DriftSampleRecord] = []

    def _kde(xs: List[float], grid: np.ndarray) -> Optional[np.ndarray]:
        if len(xs) < 2:
            return None
        arr = np.array(xs, dtype=float)
        std = np.std(arr)
        if std == 0:
            return None
        bw = 1.06 * std * (len(arr) ** (-1 / 5))
        if bw <= 0:
            return None
        vals = np.zeros_like(grid, dtype=float)
        for x in arr:
            vals += np.exp(-0.5 * ((grid - x) / bw) ** 2) / (bw * math.sqrt(2 * math.pi))
        return vals / len(arr)

    for sample in selected:
        img = Image.open(sample.image_path).convert("RGB")
        img = resize_max_side(img, max_side=1024)
        prompt_open = build_open_prompt(sample)
        prompt_mcq = build_mcq_prompt(sample)

        base_ctx_open = get_context_embedding(
            model,
            processor,
            img,
            prompt_open,
            device=device,
            context_len_cache=context_len_cache,
        )
        base_ctx_mcq = get_context_embedding(
            model,
            processor,
            img,
            prompt_mcq,
            device=device,
            context_len_cache=context_len_cache,
        )

        base_open_answer = generate_free_answer(
            model,
            processor,
            img,
            prompt_open,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        base_ans_emb_open = get_answer_embedding(
            model,
            processor,
            img,
            prompt_open,
            base_open_answer,
            device=device,
            context_len_cache=context_len_cache,
        )

        base_mcq_label = choose_label_via_loglik(
            model,
            processor,
            img,
            prompt_mcq,
            sample.options,
            device=device,
            context_len_cache=context_len_cache,
        )
        base_mcq_answer_text = (
            sample.options.get(base_mcq_label) if base_mcq_label else None
        )
        base_mcq_free_answer = generate_free_answer(
            model,
            processor,
            img,
            prompt_mcq,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        base_ans_emb_mcq = (
            get_answer_embedding(
                model,
                processor,
                img,
                prompt_mcq,
                base_mcq_answer_text,
                device=device,
                context_len_cache=context_len_cache,
            )
            if base_mcq_answer_text
            else None
        )
        base_ans_emb_mcq_free = get_answer_embedding(
            model,
            processor,
            img,
            prompt_mcq,
            base_mcq_free_answer,
            device=device,
            context_len_cache=context_len_cache,
        )

        base_embs = {
            "ctx_open": base_ctx_open,
            "ctx_mcq": base_ctx_mcq,
            "ans_open": base_ans_emb_open,
            "ans_mcq": base_ans_emb_mcq,
            "ans_mcq_free": base_ans_emb_mcq_free,
        }
        if any(v is None for v in base_embs.values()):
            print(f"[drift] Skipping sample {sample.index}: missing base embeddings.")
            continue

        per_type_lists: Dict[str, Dict[str, Dict[str, List[float]]]] = {
            ch: {pt: {"cos": [], "l2": []} for pt in perturb_types}
            for ch in channel_names
        }

        def _record_variant(ptype: str, vimg: Image.Image) -> None:
            ctx_open = get_context_embedding(
                model,
                processor,
                vimg,
                prompt_open,
                device=device,
                context_len_cache=context_len_cache,
            )
            ctx_mcq = get_context_embedding(
                model,
                processor,
                vimg,
                prompt_mcq,
                device=device,
                context_len_cache=context_len_cache,
            )
            ans_open = generate_free_answer(
                model,
                processor,
                vimg,
                prompt_open,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            ans_emb_open = get_answer_embedding(
                model,
                processor,
                vimg,
                prompt_open,
                ans_open,
                device=device,
                context_len_cache=context_len_cache,
            )
            mcq_free_answer = generate_free_answer(
                model,
                processor,
                vimg,
                prompt_mcq,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            ans_emb_mcq_free = get_answer_embedding(
                model,
                processor,
                vimg,
                prompt_mcq,
                mcq_free_answer,
                device=device,
                context_len_cache=context_len_cache,
            )
            pert_mcq_label = choose_label_via_loglik(
                model,
                processor,
                vimg,
                prompt_mcq,
                sample.options,
                device=device,
                context_len_cache=context_len_cache,
            )
            pert_mcq_text = sample.options.get(pert_mcq_label) if pert_mcq_label else None
            ans_emb_mcq = (
                get_answer_embedding(
                    model,
                    processor,
                    vimg,
                    prompt_mcq,
                    pert_mcq_text,
                    device=device,
                    context_len_cache=context_len_cache,
                )
                if pert_mcq_text
                else None
            )

            variants = {
                "ctx_open": ctx_open,
                "ctx_mcq": ctx_mcq,
                "ans_open": ans_emb_open,
                "ans_mcq": ans_emb_mcq,
                "ans_mcq_free": ans_emb_mcq_free,
            }
            for ch, emb in variants.items():
                base = base_embs[ch]
                if emb is None or base is None:
                    continue
                cos_d = _cosine_distance(base, emb)
                l2_d = torch.norm(base - emb, p=2).item()
                per_type_lists[ch][ptype]["cos"].append(cos_d)
                per_type_lists[ch][ptype]["l2"].append(l2_d)

        for n in translate_pixels:
            if n == 0:
                continue
            vimg = cyclic_horizontal_shift(img, n)
            _record_variant("Translation", vimg)

        for n in padcrop_pixels:
            if n == 0:
                continue
            vimg = pad_or_crop(img, n)
            _record_variant("Pad/Crop", vimg)

        vimg_scale = scale_image(img, scale_factor)
        _record_variant("Scale", vimg_scale)

        for bg in ["black", "white"]:
            vimg_sp = scale_and_pad(img, scale_factor, background=bg)
            _record_variant("Scale+Pad", vimg_sp)

        other_labels = [lab for lab in sample.options.keys() if lab != base_mcq_label]
        other_labels = other_labels[:3]
        TEXT_OVERLAY_PHRASES = [f"Answer is {lab}" for lab in other_labels] or [
            "Answer is B",
            "Answer is C",
            "Answer is D",
        ]
        for phrase in TEXT_OVERLAY_PHRASES:
            vimg_txt = text_overlay(img, phrase)
            _record_variant("TextOverlay", vimg_txt)

        for angle in (-30.0, 30.0):
            vimg_rot = rotate_image(img, angle)
            _record_variant("Rotation", vimg_rot)

        per_type_dist: Dict[str, Dict[str, Dict[str, float]]] = {
            ch: {} for ch in channel_names
        }
        for ch in channel_names:
            for pt in perturb_types:
                cos_vals = per_type_lists[ch][pt]["cos"]
                l2_vals = per_type_lists[ch][pt]["l2"]
                if cos_vals:
                    per_type_dist[ch][pt] = {
                        "cos": float(np.mean(cos_vals)),
                        "l2": float(np.mean(l2_vals)),
                    }

        records.append(
            DriftSampleRecord(
                sample_idx=sample.index,
                base_embs=base_embs,
                per_type_dist=per_type_dist,
                control_dist={},
            )
        )

    if not records:
        print("[drift] No records collected; skipping drift analysis.")
        return

    base_pool: Dict[str, List[tuple[int, torch.Tensor]]] = {
        ch: [] for ch in channel_names
    }
    for rec in records:
        for ch in channel_names:
            base_pool[ch].append((rec.sample_idx, rec.base_embs[ch]))

    for rec in records:
        control: Dict[str, Dict[str, float]] = {}
        for ch in channel_names:
            others = [(idx, emb) for idx, emb in base_pool[ch] if idx != rec.sample_idx]
            if not others:
                continue
            picks = rng.sample(others, min(control_k, len(others)))
            if not picks:
                continue
            cos_vals = [_cosine_distance(rec.base_embs[ch], emb) for _, emb in picks]
            l2_vals = [torch.norm(rec.base_embs[ch] - emb, p=2).item() for _, emb in picks]
            control[ch] = {
                "cos": float(np.mean(cos_vals)),
                "l2": float(np.mean(l2_vals)),
            }
        rec.control_dist = control

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines: List[str] = []
    summary_lines.append("====== EMBEDDING DRIFT ANALYSIS ======")
    summary_lines.append(
        f"Samples analyzed: {len(records)} (requested {sample_limit}), control_k={control_k}"
    )
    summary_lines.append("Metrics are mean cosine distance (1 - cos) and L2.")

    def _summ_stats(vals: List[float]) -> tuple[float, float]:
        if not vals:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    for ch in channel_names:
        summary_lines.append(f"\nChannel: {ch}")
        summary_lines.append(
            f"{'Type':<12} {'pert_mean_cos':>14} {'pert_std':>10} {'ctrl_mean_cos':>14} {'ctrl_std':>10} {'cohen_d':>10}"
        )
        summary_lines.append("-" * 74)
        for pt in perturb_types:
            pert_cos: List[float] = []
            ctrl_cos: List[float] = []
            pert_l2: List[float] = []
            ctrl_l2: List[float] = []
            for rec in records:
                pd = rec.per_type_dist.get(ch, {}).get(pt)
                cd = rec.control_dist.get(ch)
                if pd is None or cd is None:
                    continue
                pert_cos.append(pd.get("cos"))
                ctrl_cos.append(cd.get("cos"))
                pert_l2.append(pd.get("l2"))
                ctrl_l2.append(cd.get("l2"))
            if pert_cos and ctrl_cos:
                p_mean, p_std = _summ_stats(pert_cos)
                c_mean, c_std = _summ_stats(ctrl_cos)
                eff = _cohen_d(pert_cos, ctrl_cos)
                summary_lines.append(
                    f"{pt:<12} {p_mean:14.4f} {p_std:10.4f} {c_mean:14.4f} {c_std:10.4f} {eff:10.4f}"
                )

            # Cosine distance plot
            xs = pert_cos + ctrl_cos
            if xs:
                grid = np.linspace(min(xs), max(xs), 200) if len(xs) > 1 else np.linspace(xs[0] - 0.1, xs[0] + 0.1, 200)
                kde_pert = _kde(pert_cos, grid)
                kde_ctrl = _kde(ctrl_cos, grid)
                plt.figure(figsize=(6, 4))
                bins = max(10, min(50, int(math.sqrt(len(xs)))))
                plt.hist(
                    pert_cos,
                    bins=bins,
                    alpha=0.6,
                    density=True,
                    label="perturbation drift",
                    color="tab:blue",
                )
                plt.hist(
                    ctrl_cos,
                    bins=bins,
                    alpha=0.6,
                    density=True,
                    label="control drift",
                    color="tab:orange",
                )
                if kde_pert is not None:
                    plt.plot(grid, kde_pert, color="tab:blue", lw=2, label="pert KDE")
                if kde_ctrl is not None:
                    plt.plot(grid, kde_ctrl, color="tab:orange", lw=2, label="control KDE")
                plt.title(f"{ch} | {pt} (cosine distance)")
                plt.xlabel("1 - cosine similarity")
                plt.ylabel("Density")
                plt.legend(fontsize=8)
                plt.tight_layout()
                plot_path = out_dir / f"{ch}_{pt.replace('/', '_')}_cos.png"
                plt.savefig(plot_path)
                plt.close()

            # L2 plot
            pert_l2 = []
            ctrl_l2 = []
            for rec in records:
                pd = rec.per_type_dist.get(ch, {}).get(pt)
                cd = rec.control_dist.get(ch)
                if pd is None or cd is None:
                    continue
                pert_l2.append(pd.get("l2"))
                ctrl_l2.append(cd.get("l2"))

            xs_l2 = pert_l2 + ctrl_l2
            if xs_l2:
                if pert_l2 and ctrl_l2:
                    p_mean_l2, p_std_l2 = _summ_stats(pert_l2)
                    c_mean_l2, c_std_l2 = _summ_stats(ctrl_l2)
                    eff_l2 = _cohen_d(pert_l2, ctrl_l2)
                    summary_lines.append(
                        f"{(pt + ' (L2)'):<12} {p_mean_l2:14.4f} {p_std_l2:10.4f} {c_mean_l2:14.4f} {c_std_l2:10.4f} {eff_l2:10.4f}"
                    )
                grid = np.linspace(min(xs_l2), max(xs_l2), 200) if len(xs_l2) > 1 else np.linspace(xs_l2[0] - 0.1, xs_l2[0] + 0.1, 200)
                kde_pert_l2 = _kde(pert_l2, grid)
                kde_ctrl_l2 = _kde(ctrl_l2, grid)
                plt.figure(figsize=(6, 4))
                bins = max(10, min(50, int(math.sqrt(len(xs_l2)))))
                plt.hist(
                    pert_l2,
                    bins=bins,
                    alpha=0.6,
                    density=True,
                    label="perturbation drift",
                    color="tab:blue",
                )
                plt.hist(
                    ctrl_l2,
                    bins=bins,
                    alpha=0.6,
                    density=True,
                    label="control drift",
                    color="tab:orange",
                )
                if kde_pert_l2 is not None:
                    plt.plot(grid, kde_pert_l2, color="tab:blue", lw=2, label="pert KDE")
                if kde_ctrl_l2 is not None:
                    plt.plot(grid, kde_ctrl_l2, color="tab:orange", lw=2, label="control KDE")
                plt.title(f"{ch} | {pt} (L2 distance)")
                plt.xlabel("L2 distance")
                plt.ylabel("Density")
                plt.legend(fontsize=8)
                plt.tight_layout()
                plot_path = out_dir / f"{ch}_{pt.replace('/', '_')}_l2.png"
                plt.savefig(plot_path)
                plt.close()

    drift_summary_path = out_dir / "drift_summary.txt"
    with open(drift_summary_path, "w", encoding="utf-8") as f:
        for line in summary_lines:
            f.write(f"{line}\n")
    for line in summary_lines:
        print(line)
    print(f"[drift] Summary + plots written to {out_dir}")

    if summary_path is not None:
        try:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "a", encoding="utf-8") as f:
                for line in summary_lines:
                    f.write(f"{line}\n")
            print(f"[drift] Summary appended to {summary_path}")
        except Exception as exc:
            print(f"[warn] Failed to append drift summary to {summary_path}: {exc}", file=sys.stderr)
        append_line(summary_path, f"{progress_prefix}completed embedding drift on {len(records)} samples")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


DEFAULT_RUN_CONFIG = {
    # Core
    "device": "auto",
    "num_workers": 0,
    "max_samples": 4,
    "compare_mode": "label",
    "invariance_mode": "both",  # run both label + embedding by default
    "save_changed_dir": "changed_predictions",
    "save_changed_limit": 2,
    "summary_file": "summary.txt",
    # Embedding invariance
    "embedding_analysis": True,
    # Embedding viz
    "embedding_viz": True,
    "embedding_viz_limit": 4,
    "embedding_viz_dir": "embedding_viz",
    "embedding_viz_perplexity": 30.0,
    # Embedding drift
    "embedding_drift_analysis": True,
    "embedding_drift_samples": 100,
    "embedding_drift_control_k": 5,
    "embedding_drift_outdir": "embedding_drift",
    "embedding_drift_seed": 0,
    "drift_only": False,
}


def parse_args() -> argparse.Namespace:
    d = DEFAULT_RUN_CONFIG
    p = argparse.ArgumentParser(
        description="Qwen3-VL invariance on SEEDBench_IMG under visual perturbations (log-likelihood scoring).",
    )
    p.add_argument( "--model-id", default="Qwen/Qwen3-VL-2B-Instruct", help="Hugging Face model id to use.")
    p.add_argument( "--model-path", type=str, default=None, help="Local path to a model directory (overrides model-id).")
    
    p.add_argument( "--device", default=d["device"], help="Computation device: auto | cpu | mps | cuda | cuda:<idx> (auto picks best available).")
    p.add_argument( "--num-workers", type=int, default=d["num_workers"], help="Worker limit. 0 means single worker; >0 caps workers to this number and available devices/CPU slots.")
    p.add_argument( "--dataset", type=str, default="seedbench", help="Dataset adapter to use (currently: seedbench).")
    p.add_argument( "--seedbench-tsv", type=str, default=None, help="Path to SEEDBench_IMG.tsv. If omitted, defaults to cache dir and auto-download if enabled.")
    p.add_argument( "--image-root", type=str, default=None, help="Root directory for SEEDBench images. If omitted, defaults to cache dir and auto-download if enabled.")
    
    p.add_argument( "--seedbench-repo", type=str, default=DEFAULT_SEEDBENCH_REPO, help="Hugging Face dataset repo id for SEEDBench.")
    p.add_argument( "--seedbench-tsv-filename", type=str, default=DEFAULT_SEEDBENCH_TSV, help="Filename of the SEEDBench TSV inside the repo or cache.")
    p.add_argument( "--seedbench-image-archive", type=str, default=DEFAULT_SEEDBENCH_ARCHIVE, help="Archive filename for SEEDBench images inside the repo (zip/tar).")
    
    p.add_argument( "--data-dir", type=str, default=None, help="Base directory to cache SEEDBench data (defaults to <cache_dir>/seedbench).")
    p.add_argument( "--max-samples", type=int, default=d["max_samples"], help="Maximum number of samples to evaluate.")
    p.add_argument( "--max-new-tokens", type=int, default=32, help="Max new tokens (only used for non-label compare modes).")
    p.add_argument( "--temperature", type=float, default=0.0, help="Sampling temperature for free-text modes.")

    p.add_argument( "--translate-steps", type=int, nargs="+", default=[-16, -12, -8, -4, 4, 8, 12, 16], help="Horizontal cyclic shifts (pixels).")
    p.add_argument( "--padcrop-steps", type=int, nargs="+", default=[-16, -12, -8, -4, 4, 8, 12, 16], help="Pad/Crop sizes (pixels; negative=crop).")
    p.add_argument( "--scale-factor", type=float, default=0.9, help="Scale factor for Scale and Scale+Pad.")
    p.add_argument( "--compare-mode", type=str, default=d["compare_mode"], choices=["label", "text", "text_mcq", "semantic"], help=(
                                                                    "Representation used for invariance comparison:\n"
                                                                    "  label    -> log-likelihood argmax MCQ label (SEED-Bench style)\n"
                                                                    "  text     -> normalized free-text (open style)\n"
                                                                    "  text_mcq -> normalized free-text with options kept in context\n"
                                                                    "  semantic -> placeholder for embedding/LLM-based semantic invariance") )
    p.add_argument( "--cache-dir", type=str, default=str((Path(__file__).resolve().parent / ".hf_cache")), help="HF cache directory." )
    p.add_argument( "--offline",action="store_true", help="Use HF cache only (no internet)." )
    p.add_argument( "--save-changed-dir", type=str, default=d["save_changed_dir"], help="Directory to save original/perturbed image pairs when predictions change. Set empty to disable." )
    p.add_argument( "--save-changed-limit", type=int, default=d["save_changed_limit"], help="Max number of changed-prediction pairs to save (set <=0 for no limit)." )
    p.add_argument( "--summary-file", type=str, default=d["summary_file"], help="Path to write the final summary (per-worker suffix added in multi-worker mode). Set empty to disable." )
    p.add_argument( "--embedding-analysis", action="store_true", default=d["embedding_analysis"], help="If set, also run embedding-based invariance analysis (context + pooled answer embeddings)." )
    p.add_argument( "--invariance-mode", type=str, choices=["label", "embedding", "both"], default=d["invariance_mode"], help="Select which invariance analysis to run: label, embedding, or both." )
    p.add_argument( "--embedding-viz", action="store_true", default=d["embedding_viz"], help="Save PCA and t-SNE plots of embeddings for a subset of samples (uses embedding analysis outputs)." )
    p.add_argument( "--embedding-viz-limit", type=int, default=d["embedding_viz_limit"], help="Number of samples to include in embedding visualizations." )
    p.add_argument( "--embedding-viz-dir", type=str, default=d["embedding_viz_dir"], help="Directory to save embedding visualization plots." )
    p.add_argument( "--embedding-viz-perplexity", type=float, default=d["embedding_viz_perplexity"], help="t-SNE perplexity for embedding visualization." )
    p.add_argument( "--embedding-drift-analysis", action="store_true", default=d["embedding_drift_analysis"], help="Run drift analysis comparing perturbation drift vs control drift (base vs random other bases)." )
    p.add_argument( "--embedding-drift-samples", type=int, default=d["embedding_drift_samples"], help="Number of samples to include in drift analysis (random subset with valid embeddings)." )
    p.add_argument( "--embedding-drift-control-k", type=int, default=d["embedding_drift_control_k"], help="Number of random other bases to compare against for control drift." )
    p.add_argument( "--embedding-drift-outdir", type=str, default=d["embedding_drift_outdir"], help="Output directory for drift summaries and plots." )
    p.add_argument( "--embedding-drift-seed", type=int, default=d["embedding_drift_seed"], help="Seed for random sampling in drift analysis." )
    p.add_argument( "--drift-only", action="store_true", default=d["drift_only"], help="Skip label/embedding analyses and run only embedding drift (single-device only)." )
    
    return p.parse_args()


def load_samples_for_dataset(
    args: argparse.Namespace, cache_dir: Path
) -> tuple[List[Sample], int, Path, Path]:
    """
    Adapter entrypoint for loading samples from a dataset. Currently supports
    SEEDBench; extend this function to plug in other datasets.
    """
    dataset = args.dataset.lower()
    if dataset != "seedbench":
        raise ValueError(f"Dataset adapter not implemented: {args.dataset}")

    tsv_path, image_root, total_rows = prepare_seedbench_data(args, cache_dir)
    if not tsv_path.exists() or not image_root.exists():
        raise FileNotFoundError(
            f"Data unavailable after download attempts: {tsv_path}, {image_root}"
        )

    samples = load_seedbench_samples(
        tsv_path,
        image_root,
        max_samples=args.max_samples,
    )
    return samples, total_rows, tsv_path, image_root


def main() -> None:
    args = parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    torch.manual_seed(0)

    print(f"[info] HF cache dir={cache_dir}")

    samples, total_rows, tsv_path, image_root = load_samples_for_dataset(args, cache_dir)
    print(f"[info] Loaded {len(samples)} samples from {tsv_path.name} (rows available: {total_rows})")
    if args.max_samples is not None and len(samples) < args.max_samples:
        print(
            f"[warn] Only {len(samples)} samples available; requested {args.max_samples}.",
            file=sys.stderr,
        )

    save_dir = Path(args.save_changed_dir).expanduser().resolve() if args.save_changed_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_file).expanduser().resolve() if args.summary_file else None
    viz_config = EmbeddingVizConfig(
        enabled=args.embedding_viz,
        limit_samples=max(1, args.embedding_viz_limit),
        out_dir=Path(args.embedding_viz_dir).expanduser().resolve(),
        perplexity=args.embedding_viz_perplexity,
    )
    run_embedding_drift = args.embedding_drift_analysis
    progress_prefix = ""

    # Determine which analyses to run
    run_label_invariance = args.invariance_mode in ("label", "both")
    run_embedding_invariance = args.embedding_analysis or args.invariance_mode in ("embedding", "both")
    if args.drift_only:
        run_label_invariance = False
        run_embedding_invariance = False
        run_embedding_drift = True

    # ------------------------------------------------------------------
    # Parallel execution across all available devices
    # ------------------------------------------------------------------
    requested_workers = args.num_workers
    max_workers = None if requested_workers == 0 else requested_workers
    devices = list_available_devices(args.device, max_workers)

    if requested_workers == 0 and len(devices) > 1:
        # User asked for "just run" (0); default to a single worker.
        devices = devices[:1]

    if not devices:
        raise RuntimeError("No devices available for execution.")

    print(f"[info] Selected devices for execution: {devices}")

    if len(devices) <= 1:
        device = canonical_device_id(devices[0])
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

        if run_label_invariance:
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
                summary_path=summary_path,
                progress_prefix=progress_prefix,
            )
        if run_embedding_invariance:
            run_embedding_invariance_analysis(
                model,
                processor,
                device=device,
                samples=samples,
                translate_pixels=args.translate_steps,
                padcrop_pixels=args.padcrop_steps,
                scale_factor=args.scale_factor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                summary_path=summary_path,
                viz_config=viz_config if viz_config.enabled else None,
                progress_prefix=progress_prefix,
            )
        if run_embedding_drift:
            run_embedding_drift_analysis(
                model,
                processor,
                device=device,
                samples=samples,
                translate_pixels=args.translate_steps,
                padcrop_pixels=args.padcrop_steps,
                scale_factor=args.scale_factor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                sample_limit=args.embedding_drift_samples,
                control_k=args.embedding_drift_control_k,
                out_dir=Path(args.embedding_drift_outdir).expanduser().resolve(),
                seed=args.embedding_drift_seed,
                summary_path=summary_path,
                progress_prefix=progress_prefix,
            )
        return

    # Multi-device: split samples evenly and spawn one worker per device.
    if run_embedding_drift:
        print("[warn] Embedding drift analysis is single-device only; skipping in multi-worker mode.", file=sys.stderr)
        run_embedding_drift = False
    import math
    import multiprocessing as mp

    world_size = len(devices)
    chunk = math.ceil(len(samples) / float(world_size))
    sample_splits = [samples[i : i + chunk] for i in range(0, len(samples), chunk)]

    print(f"[info] Launching {world_size} parallel workers across devices: {devices}")
    ctx = mp.get_context("spawn")
    procs = []
    for idx, (device_id, worker_samples) in enumerate(zip(devices, sample_splits)):
        p = ctx.Process(
            target=worker_process,
            args=(
                idx,
                device_id,
                args,
                cache_dir,
                worker_samples,
                save_dir,
                run_label_invariance,
                run_embedding_invariance,
                viz_config if viz_config.enabled else None,
            ),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    from pathlib import Path
    main()

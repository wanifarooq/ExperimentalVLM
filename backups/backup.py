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
import ast
import json
import math
import os
import re
import subprocess
import sys
import base64
import random
import string
import requests
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    AutoConfig,
    AutoProcessor,
)
try:  # noqa: E402
    from transformers import AutoModelForVisionText2Text  # type: ignore
except ImportError:  # pragma: no cover
    AutoModelForVisionText2Text = None  # type: ignore
try:  # noqa: E402
    from transformers import AutoModelForVision2Seq  # type: ignore
except ImportError:  # pragma: no cover
    AutoModelForVision2Seq = None  # type: ignore
try:  # noqa: E402
    from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor  # type: ignore
except ImportError:  # pragma: no cover
    LlavaOnevisionForConditionalGeneration = None  # type: ignore
    LlavaOnevisionProcessor = None  # type: ignore
try:  # noqa: E402
    from transformers import Qwen2Config  # type: ignore
except ImportError:  # pragma: no cover
    Qwen2Config = None  # type: ignore
from huggingface_hub import hf_hub_download  # noqa: E402
from huggingface_hub import list_repo_files  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
import tarfile  # noqa: E402
import zipfile  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
BATCH_LOGPROB_FALLBACK_WARNED = False
OPTION_BATCH_CHUNK_SIZE = 4


def set_option_batch_chunk_size(size: int) -> None:
    global OPTION_BATCH_CHUNK_SIZE
    OPTION_BATCH_CHUNK_SIZE = int(size)

# ---------------------------------------------------------------------
# Utilities: device, normalization
# ---------------------------------------------------------------------


def is_llava_onevision(model_id: str) -> bool:
    mid = model_id.lower()
    return "llava-onevision" in mid or "onevision" in mid


def _batch_to_device(batch: Any, device: str) -> Dict[str, Any]:
    """Move a tokenized batch (dict or BatchEncoding) to device."""
    if hasattr(batch, "to"):
        batch = batch.to(device)
    data = batch.data if hasattr(batch, "data") else batch
    if isinstance(data, dict):
        return {
            k: (v.to(device) if hasattr(v, "to") else v) for k, v in data.items()
        }
    if hasattr(batch, "items"):
        return {
            k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()
        }
    raise TypeError(f"Unsupported batch type for device move: {type(batch)}")


def _collect_images_from_content(content: List[Dict[str, Any]]) -> List[Any]:
    imgs: List[Any] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "image":
            continue
        img = part.get("image")
        if img is not None:
            imgs.append(img)
    return imgs


_CHAT_TEMPLATE_AVAILABLE: Dict[int, bool] = {}
_CHAT_TEMPLATE_MISSING_ERROR = "does not have a chat template"


def _get_chat_template_status(processor) -> Optional[bool]:
    return _CHAT_TEMPLATE_AVAILABLE.get(id(processor))


def _set_chat_template_status(processor, available: bool) -> None:
    _CHAT_TEMPLATE_AVAILABLE[id(processor)] = available


def _is_missing_chat_template_error(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and _CHAT_TEMPLATE_MISSING_ERROR in str(exc)


def _get_image_token(processor) -> str:
    token = getattr(processor, "image_token", None)
    if token:
        return token
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        token = getattr(tokenizer, "image_token", None)
        if token:
            return token
        token_id = getattr(tokenizer, "image_token_id", None)
        if token_id is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token:
                return token
    return "<image>"


def _render_message_text(message: Dict[str, Any], image_token: str) -> str:
    content = message.get("content", [])
    parts: List[str] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "image":
                parts.append(image_token)
            elif part_type == "text":
                text = part.get("text")
                if text:
                    parts.append(str(text))
    elif isinstance(content, str):
        parts.append(content)
    role = (message.get("role") or "").lower()
    if role == "user":
        prefix = "User"
    elif role == "assistant":
        prefix = "Assistant"
    else:
        prefix = role.capitalize() if role else ""
    body = "\n".join(part for part in parts if part)
    if prefix:
        return f"{prefix}: {body}" if body else f"{prefix}:"
    return body


def _render_messages_fallback(
    processor, messages: List[Any], add_generation_prompt: bool, is_batch: bool
) -> Any:
    image_token = _get_image_token(processor)
    if is_batch:
        rendered_batch = []
        for conv in messages:
            rendered = [_render_message_text(msg, image_token) for msg in conv]
            text = "\n".join(part for part in rendered if part)
            last_role = conv[-1].get("role") if conv else None
            if add_generation_prompt and last_role != "assistant":
                text = f"{text}\nAssistant:" if text else "Assistant:"
            rendered_batch.append(text)
        return rendered_batch
    rendered = [_render_message_text(msg, image_token) for msg in messages]
    text = "\n".join(part for part in rendered if part)
    last_role = messages[-1].get("role") if messages else None
    if add_generation_prompt and last_role != "assistant":
        text = f"{text}\nAssistant:" if text else "Assistant:"
    return text


def _ensure_processor_patch_size(processor, config) -> None:
    if not hasattr(processor, "patch_size"):
        return
    if getattr(processor, "patch_size", None) is not None:
        return
    patch_size = None
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        patch_size = getattr(vision_config, "patch_size", None)
    if patch_size is None:
        image_processor = getattr(processor, "image_processor", None)
        patch_size = getattr(image_processor, "patch_size", None)
    if patch_size is not None:
        processor.patch_size = patch_size


def _ensure_processor_image_size(processor, config) -> None:
    if getattr(config, "model_type", "") != "llava":
        return
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        return
    target = getattr(vision_config, "image_size", None)
    if target is None:
        return
    size = getattr(image_processor, "size", None)
    if not isinstance(size, dict):
        return
    if "height" in size and "width" in size:
        if size["height"] != target or size["width"] != target:
            image_processor.size = {"height": target, "width": target}
    elif "shortest_edge" in size and size["shortest_edge"] != target:
        image_processor.size = {"shortest_edge": target}


def tokenize_conversations(
    processor,
    messages: List[Any],
    device: str,
    *,
    add_generation_prompt: bool,
    padding: bool = False,
) -> Dict[str, Any]:
    """Tokenize chat messages with a fallback for processors that require text+images calls."""
    is_batch = bool(messages) and isinstance(messages[0], list)
    template_available = _get_chat_template_status(processor)
    if template_available is not False:
        try:
            enc = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                padding=padding or is_batch,
                return_tensors="pt",
                return_dict=True,
            )
            enc_dict = _batch_to_device(enc, device)
            if "input_ids" in enc_dict:
                _set_chat_template_status(processor, True)
                return enc_dict
        except Exception as exc:
            if _is_missing_chat_template_error(exc):
                _set_chat_template_status(processor, False)
                template_available = False

    if template_available is False:
        prompt_text = _render_messages_fallback(
            processor, messages, add_generation_prompt, is_batch
        )
    else:
        try:
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            _set_chat_template_status(processor, True)
        except Exception as exc:
            if _is_missing_chat_template_error(exc):
                _set_chat_template_status(processor, False)
            prompt_text = _render_messages_fallback(
                processor, messages, add_generation_prompt, is_batch
            )

    if is_batch:
        image_batches = []
        for conv in messages:  # type: ignore[assignment]
            imgs: List[Any] = []
            for msg in conv:
                content = msg.get("content", [])
                if isinstance(content, list):
                    imgs.extend(_collect_images_from_content(content))
            image_batches.append(imgs)
        proc_kwargs: Dict[str, Any] = {
            "text": prompt_text,
            "padding": True,
            "return_tensors": "pt",
        }
        if any(image_batches):
            proc_kwargs["images"] = image_batches
    else:
        content = messages[0].get("content", []) if messages else []
        images = _collect_images_from_content(content) if isinstance(content, list) else []
        proc_kwargs = {
            "text": prompt_text,
            "return_tensors": "pt",
        }
        if images:
            proc_kwargs["images"] = images

    enc = processor(**proc_kwargs)
    return _batch_to_device(enc, device)


def infer_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)


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
DEFAULT_POPE_REPO = "lmms-lab/POPE"
DEFAULT_POPE_SPLIT = "adversarial"
POPE_SPLIT_FILES = {
    "adversarial": ["Full/adversarial-00000-of-00001.parquet"],
    "popular": ["Full/popular-00000-of-00001.parquet"],
    "random": ["Full/random-00000-of-00001.parquet"],
    "test": [
        "data/test-00000-of-00003.parquet",
        "data/test-00001-of-00003.parquet",
        "data/test-00002-of-00003.parquet",
    ],
}


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
    index: int | str
    image_path: Path
    question: str
    options: Dict[str, str]  # e.g. {"A": "...", "B": "..."}
    hint: Optional[str] = None
    category: Optional[str] = None  # SEEDBench type if present
    answer_label: Optional[str] = None  # Ground-truth option label if available
    dataset: str = "seedbench"
    split: Optional[str] = None
    image_paths: List[Path] = field(default_factory=list)
    source_id: Optional[str] = None
    filter_reason: Optional[str] = None


@dataclass
class UserPrompt:
    content: List[Dict[str, Any]]
    text: str
    cache_key: Optional[tuple] = None


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


@dataclass
class FrequencyBandDelta:
    sample_idx: int
    delta_low: float
    delta_mid: float
    delta_high: float
    rel_low: float
    rel_mid: float
    rel_high: float
    is_flip: bool


@dataclass
class DirichletBase:
    energy: float
    context_embedding: Optional[torch.Tensor]


@dataclass
class DirichletRecord:
    sample_idx: int
    delta_energy: float
    embedding_drift: Optional[float]
    is_flip: bool
    delta_base_score: Optional[float] = None
    delta_low: Optional[float] = None
    delta_high: Optional[float] = None
    delta_ratio: Optional[float] = None


# ---------------------------------------------------------------------
# SEEDBench loading & prompt construction
# ---------------------------------------------------------------------


def _mean_std(vals: List[float]) -> tuple[float, float]:
    """Return mean/std with safe defaults for empty inputs."""
    if not vals:
        return 0.0, 0.0
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def _pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    """Compute Pearson correlation; return None if insufficient data or zero variance."""
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    xa = np.array(xs, dtype=float)
    ya = np.array(ys, dtype=float)
    if xa.std() == 0 or ya.std() == 0:
        return None
    corr = np.corrcoef(xa, ya)[0, 1]
    return float(corr)


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


def _pope_yes_no_labels(
    options: Dict[str, str],
) -> tuple[Optional[str], Optional[str]]:
    yes_label = None
    no_label = None
    for lab, opt_text in options.items():
        norm = normalize_text(opt_text)
        if norm == "yes":
            yes_label = lab
        elif norm == "no":
            no_label = lab
    return yes_label, no_label


def _pope_label_to_yesno(
    label: Optional[str],
    options: Dict[str, str],
) -> Optional[str]:
    if label is None:
        return None
    text = options.get(label, label)
    norm = normalize_text(str(text))
    if norm == "yes":
        return "Yes"
    if norm == "no":
        return "No"
    return None


def _pope_margin_from_scores(
    scores: Dict[str, float],
    yes_label: Optional[str],
    no_label: Optional[str],
) -> Optional[float]:
    if yes_label is None or no_label is None:
        return None
    if yes_label not in scores or no_label not in scores:
        return None
    return float(scores[yes_label] - scores[no_label])


def _mmmu_data_dir(args: argparse.Namespace, cache_dir: Path) -> Path:
    """Resolve base directory for MMMU data."""
    if args.data_dir:
        return Path(args.data_dir).expanduser().resolve()
    return cache_dir / "mmmu"


def _safe_image_stem(val: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", val).strip("_")
    return cleaned or "image"


def dataset_suffix_tag(args: argparse.Namespace) -> str:
    tag = args.dataset.lower()
    split = getattr(args, "mmmu_split", None)
    if tag == "mmmu" and split:
        tag = f"{tag}_{split.lower()}"
    pope_split = getattr(args, "pope_split", None)
    if tag == "pope" and pope_split:
        tag = f"{tag}_{pope_split.lower()}"
    return tag


def model_size_tag(model_id: str) -> str:
    """
    Extract a model size token (e.g., 7B, 72B, 1.8B) from a model id/path.
    Returns empty string if no size-like token is found.
    """
    base = Path(model_id).name
    match = re.search(r"(\d+(?:\.\d+)?)([bBmM])", base)
    if not match:
        return ""
    num, unit = match.groups()
    num_clean = num.replace(".", "p")
    return f"{num_clean}{unit.upper()}"


def path_with_dataset_suffix(path: Path, suffix: str, is_file: bool) -> Path:
    if not suffix:
        return path
    name = path.name
    if is_file:
        stem, ext = path.stem, path.suffix
        if stem.endswith(suffix):
            return path
        return path.with_name(f"{stem}_{suffix}{ext}")
    if name.endswith(suffix):
        return path
    return path.parent / f"{name}_{suffix}"


def _parse_mmmu_options(raw: Any) -> List[str]:
    """Parse options column into a list of option strings."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            parsed = []
    if isinstance(parsed, dict):
        parsed = list(parsed.values())
    if not isinstance(parsed, (list, tuple)):
        return []
    opts = []
    for opt in parsed:
        if opt is None or (isinstance(opt, float) and math.isnan(opt)):
            continue
        text = str(opt).strip()
        if text:
            opts.append(text)
    return opts


def _iter_mmmu_parquet_paths(
    split: str, base_dir: Path, cache_dir: Path, offline: bool
) -> List[Path]:
    local_files = sorted(base_dir.rglob(f"{split}-*.parquet"))
    if offline:
        if not local_files:
            raise FileNotFoundError(
                f"MMMU {split} split unavailable locally at {base_dir} in offline mode."
            )
        return local_files

    try:
        repo_files = list_repo_files("MMMU/MMMU", repo_type="dataset")
    except Exception as exc:
        if local_files:
            print(f"[warn] Falling back to local MMMU files due to repo listing error: {exc}", file=sys.stderr)
            return local_files
        raise

    split_files = [
        f
        for f in repo_files
        if f.endswith(".parquet") and f.split("/")[-1].startswith(f"{split}-")
    ]
    if not split_files and not local_files:
        raise FileNotFoundError(f"No MMMU parquet files found for split {split}.")

    paths: List[Path] = []
    for rel in split_files:
        local_path = base_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            hf_hub_download(
                repo_id="MMMU/MMMU",
                filename=rel,
                repo_type="dataset",
                cache_dir=str(cache_dir),
                local_dir=str(base_dir),
                local_dir_use_symlinks=False,
            )
        paths.append(local_path)

    if not paths:
        return local_files
    return sorted(paths)


def _extract_mmmu_images(
    row: pd.Series, split: str, base_dir: Path, sample_id: str
) -> List[Path]:
    """Write MMMU image bytes to disk and return paths."""
    image_root = base_dir / "images" / split
    image_root.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i in range(1, 8):
        key = f"image_{i}"
        val = row.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        if not isinstance(val, dict):
            continue
        img_bytes = val.get("bytes")
        if not img_bytes:
            continue
        filename = Path(str(val.get("path") or f"{sample_id}_{i}.png")).name
        out_path = image_root / filename
        if not out_path.exists():
            try:
                out_path.write_bytes(img_bytes)
            except Exception as exc:
                print(f"[warn] Failed to write MMMU image {filename}: {exc}", file=sys.stderr)
                continue
        paths.append(out_path)
    return paths


def load_mmmu_samples(
    split: str,
    cache_dir: Path,
    data_dir: Path,
    max_samples: Optional[int],
    offline: bool,
) -> tuple[List[Sample], int, Path, Path]:
    """
    Load MMMU MCQ samples from parquet shards. Filters to rows with >=2 options.
    """
    base_dir = data_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    parquet_paths = _iter_mmmu_parquet_paths(split, base_dir, cache_dir, offline)
    samples: List[Sample] = []
    total_rows = 0
    rng = random.Random(0)
    seen_valid = 0
    for pq_path in parquet_paths:
        try:
            df = pd.read_parquet(pq_path)
        except Exception as exc:
            print(f"[warn] Failed to read {pq_path}: {exc}", file=sys.stderr)
            continue
        total_rows += len(df)
        for _, row in df.iterrows():
            sample_id = str(row.get("id") or f"{split}_{seen_valid+1}")
            options_list = _parse_mmmu_options(row.get("options"))
            if len(options_list) < 2:
                print(f"[mmmu] skip {sample_id} ({split}): non-MCQ or missing options", file=sys.stderr)
                continue
            images = _extract_mmmu_images(row, split, base_dir, sample_id)
            if not images:
                print(f"[mmmu] skip {sample_id} ({split}): no images", file=sys.stderr)
                continue

            options = {
                chr(ord("A") + i): opt for i, opt in enumerate(options_list)
            }
            answer_label = _normalize_answer_label(row.get("answer"), options)
            category = None
            if "subfield" in df.columns and not pd.isna(row["subfield"]):
                category = str(row["subfield"])
            question_val = row.get("question")
            question_text = "" if (question_val is None or (isinstance(question_val, float) and math.isnan(question_val))) else str(question_val)

            candidate = Sample(
                index=sample_id,
                image_path=Path(images[0]),
                question=question_text,
                options=options,
                hint=None,
                category=category,
                answer_label=answer_label,
                dataset="mmmu",
                split=split,
                image_paths=[Path(p) for p in images],
                source_id=sample_id,
            )
            seen_valid += 1
            if max_samples is None or len(samples) < max_samples:
                samples.append(candidate)
            else:
                j = rng.randint(0, seen_valid - 1)
                if j < max_samples:
                    samples[j] = candidate
    image_root = base_dir / "images" / split
    return samples, total_rows, Path(f"mmmu_{split}"), image_root


def _pope_parquet_files(split: str) -> List[str]:
    key = split.lower()
    if key not in POPE_SPLIT_FILES:
        raise ValueError(f"Unsupported POPE split: {split}")
    return POPE_SPLIT_FILES[key]


def load_pope_samples(
    split: str,
    cache_dir: Path,
    max_samples: Optional[int],
    offline: bool,
) -> tuple[List[Sample], int, Path, Path]:
    """
    Load POPE samples from parquet shards. Each sample is a Yes/No question.
    """
    split_key = split.lower()
    base_dir = cache_dir / "pope"
    base_dir.mkdir(parents=True, exist_ok=True)
    image_root = base_dir / "images" / split_key
    image_root.mkdir(parents=True, exist_ok=True)
    parquet_files = _pope_parquet_files(split_key)

    samples: List[Sample] = []
    total_rows = 0
    rng = random.Random(0)
    seen_valid = 0
    for rel in parquet_files:
        try:
            pq_path = hf_hub_download(
                DEFAULT_POPE_REPO,
                rel,
                repo_type="dataset",
                cache_dir=str(cache_dir),
                local_files_only=offline,
            )
        except Exception as exc:
            if offline:
                raise FileNotFoundError(
                    f"POPE parquet missing in offline mode: {rel}"
                ) from exc
            raise
        try:
            df = pd.read_parquet(pq_path)
        except Exception as exc:
            print(f"[warn] Failed to read {pq_path}: {exc}", file=sys.stderr)
            continue
        total_rows += len(df)
        for _, row in df.iterrows():
            question_val = row.get("question")
            question_text = "" if (question_val is None or (isinstance(question_val, float) and math.isnan(question_val))) else str(question_val)
            image_info = row.get("image")
            if not isinstance(image_info, dict):
                image_info = {}
            img_bytes = image_info.get("bytes")
            if img_bytes is None:
                sample_id = str(row.get("question_id") or row.get("id") or f"{split_key}_{seen_valid+1}")
                print(f"[pope] skip {sample_id} ({split_key}): missing image bytes", file=sys.stderr)
                continue

            sample_id = str(row.get("question_id") or row.get("id") or f"{split_key}_{seen_valid+1}")
            image_source = row.get("image_source")
            image_id = str(image_source or sample_id)
            stem = _safe_image_stem(image_id)
            if stem.lower().endswith((".jpg", ".jpeg", ".png")):
                filename = stem
            else:
                filename = f"{stem}.jpg"
            out_path = image_root / filename
            if not out_path.exists():
                try:
                    out_path.write_bytes(bytes(img_bytes))
                except Exception as exc:
                    print(f"[warn] Failed to write POPE image {filename}: {exc}", file=sys.stderr)
                    continue

            options = {"Yes": "Yes", "No": "No"}
            answer_label = _normalize_answer_label(row.get("answer"), options)
            category = None
            if "category" in df.columns and not pd.isna(row["category"]):
                category = str(row["category"])

            candidate = Sample(
                index=sample_id,
                image_path=out_path,
                question=question_text,
                options=options,
                hint=None,
                category=category,
                answer_label=answer_label,
                dataset="pope",
                split=split_key,
                image_paths=[out_path],
                source_id=str(image_source) if image_source is not None else None,
            )
            seen_valid += 1
            if max_samples is None or len(samples) < max_samples:
                samples.append(candidate)
            else:
                j = rng.randint(0, seen_valid - 1)
                if j < max_samples:
                    samples[j] = candidate
    return samples, total_rows, Path(f"pope_{split_key}"), image_root


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
                dataset="seedbench",
                split=None,
                image_paths=[img_path],
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


def build_pope_prompt(sample: Sample) -> str:
    question = str(sample.question or "").strip()
    return f"Question: {question}\nAnswer with exactly one word: Yes or No."


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


def prompt_style_from_compare_mode(compare_mode: str) -> str:
    if compare_mode == "text":
        return "open"
    if compare_mode == "text_mcq":
        return "open_with_options"
    return "mcq"


def _format_options_block(options: Dict[str, str], prompt_style: str) -> str:
    if prompt_style == "open":
        return "Please answer the question concisely."
    if not options:
        return ""
    lines = ["Options:"]
    for key, val in options.items():
        lines.append(f"{key}. {val}")
    if prompt_style == "open_with_options":
        lines.append("Provide the best answer in your own words.")
    else:
        lines.append("Please select the correct answer from the options above.")
    return "\n".join(lines)


def build_seedbench_user_prompt(
    sample: Sample, images: List[Image.Image], prompt_style: str
) -> UserPrompt:
    if not images:
        raise ValueError("No images provided for prompt construction.")
    if prompt_style == "open":
        prompt_text = build_open_prompt(sample)
    elif prompt_style == "open_with_options":
        prompt_text = build_open_with_options_prompt(sample)
    else:
        prompt_text = build_seedbench_prompt(sample)
    content = [
        {"type": "image", "image": images[0]},
        {"type": "text", "text": prompt_text},
    ]
    cache_key = ("seedbench", prompt_style, prompt_text, (images[0].size,))
    return UserPrompt(content=content, text=prompt_text, cache_key=cache_key)


def build_pope_user_prompt(
    sample: Sample, images: List[Image.Image]
) -> UserPrompt:
    if not images:
        raise ValueError("No images provided for prompt construction.")
    prompt_text = build_pope_prompt(sample)
    content = [
        {"type": "image", "image": images[0]},
        {"type": "text", "text": prompt_text},
    ]
    cache_key = ("pope", prompt_text, (images[0].size,))
    return UserPrompt(content=content, text=prompt_text, cache_key=cache_key)


def build_mmmu_prompt(
    sample: Sample, images: List[Image.Image], prompt_style: str
) -> UserPrompt:
    question = str(sample.question or "").strip()
    pattern = re.compile(r"<image\s*(\d+)>", flags=re.IGNORECASE)
    parts = pattern.split(question)
    content: List[Dict[str, Any]] = []
    used_imgs: set[int] = set()
    prefix_added = False
    for i in range(0, len(parts), 2):
        text_part = parts[i].strip()
        if text_part:
            if not prefix_added:
                content.append({"type": "text", "text": f"Question: {text_part}"})
                prefix_added = True
            else:
                content.append({"type": "text", "text": text_part})
        if i + 1 < len(parts):
            idx_str = parts[i + 1]
            try:
                pos = int(idx_str) - 1
            except Exception:
                pos = None
            if pos is not None and 0 <= pos < len(images):
                used_imgs.add(pos)
                content.append({"type": "image", "image": images[pos]})
    for pos, img in enumerate(images):
        if pos not in used_imgs:
            content.append({"type": "image", "image": img})

    options_block = _format_options_block(sample.options, prompt_style)
    if options_block:
        content.append({"type": "text", "text": options_block})
    prompt_text_parts = [f"Question: {question}"]
    if options_block:
        prompt_text_parts.append(options_block)
    prompt_text = "\n".join(prompt_text_parts)
    cache_key = ("mmmu", prompt_style, prompt_text, tuple(img.size for img in images))
    return UserPrompt(content=content, text=prompt_text, cache_key=cache_key)


def build_user_prompt(
    sample: Sample, images: List[Image.Image], prompt_style: str
) -> UserPrompt:
    dataset = sample.dataset.lower()
    if dataset == "mmmu":
        return build_mmmu_prompt(sample, images, prompt_style)
    if dataset == "pope":
        return build_pope_user_prompt(sample, images)
    return build_seedbench_user_prompt(sample, images, prompt_style)


def load_sample_images(sample: Sample, max_side: int = 1024) -> List[Image.Image]:
    paths = sample.image_paths if sample.image_paths else [sample.image_path]
    images: List[Image.Image] = []
    for p in paths:
        if not p.exists():
            print(f"[warn] Missing image: {p}", file=sys.stderr)
            continue
        try:
            img = Image.open(p).convert("RGB")
            images.append(resize_max_side(img, max_side=max_side))
        except Exception as exc:
            print(f"[warn] Failed to load image {p}: {exc}", file=sys.stderr)
    return images


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


def overlay_font_for_image(img: Image.Image):
    h = img.size[1]
    try:
        return ImageFont.truetype("DejaVuSans.ttf", max(14, h // 32))
    except Exception:
        return ImageFont.load_default()


def overlay_box_for_text(
    img: Image.Image, text: str, font, pad: int = 4
) -> Tuple[int, int, int, int]:
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = img.size
    x = max(0, (w - tw) // 2)
    y = max(0, (h - th) // 2)
    return (x - pad, y - pad, x + tw + pad, y + th + pad)


def overlay_text_position(
    draw: ImageDraw.ImageDraw, text: str, font, box: Tuple[int, int, int, int]
) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    x = x0 + (x1 - x0 - tw) // 2
    y = y0 + (y1 - y0 - th) // 2
    return x, y


def fit_text_to_box(
    draw: ImageDraw.ImageDraw, text: str, font, box: Tuple[int, int, int, int]
) -> str:
    max_w = box[2] - box[0]
    while text:
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_w:
            return text
        text = text[:-1]
    return text


def text_overlay_in_box(
    img: Image.Image, text: str, box: Tuple[int, int, int, int], font
) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    fitted = fit_text_to_box(draw, text, font, box)
    draw.rectangle(box, fill="white")
    x, y = overlay_text_position(draw, fitted, font, box)
    draw.text((x, y), fitted, fill="red", font=font)
    return img


def text_overlay(img: Image.Image, text: str) -> Image.Image:
    """
    Overlay red, instruction-like text in the center.
    """
    font = overlay_font_for_image(img)
    box = overlay_box_for_text(img, text, font)
    return text_overlay_in_box(img, text, box, font)


def text_overlay_phrases(
    options: Dict[str, str], base_label: Optional[str], limit: int = 3
) -> List[str]:
    labels = [lab for lab in options.keys() if lab != base_label]
    labels = labels[:limit]
    if labels:
        return [f"Answer is {lab}" for lab in labels]
    return ["Answer is B", "Answer is C", "Answer is D"]


def random_text_phrases(seed: int, count: int = 3) -> List[str]:
    rng = random.Random(seed)
    alphabet = string.ascii_uppercase + string.digits
    phrases = []
    for _ in range(count):
        length = rng.randint(6, 12)
        phrases.append("".join(rng.choice(alphabet) for _ in range(length)))
    return phrases


def box_overlay(img: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(box, fill="white")
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


def build_device_groups(
    devices: List[str],
    requested_workers: int,
    gpus_per_worker: Optional[int],
) -> List[List[str]]:
    """
    Partition devices into groups for each worker. GPU groups can span multiple
    devices for tensor-parallel loading; CPU/MPS are treated as single-slot.
    """
    if not devices:
        return []

    gpu_devices = [d for d in devices if d.startswith("cuda")]
    if not gpu_devices:
        if requested_workers == 0:
            return [[devices[0]]]
        return [[dev] for dev in devices[:requested_workers]]

    auto_tp = gpus_per_worker is None or gpus_per_worker <= 0
    if auto_tp:
        tp_size = len(gpu_devices) if requested_workers == 0 else max(1, len(gpu_devices) // requested_workers)
    else:
        tp_size = gpus_per_worker or 1
    tp_size = max(1, min(tp_size, len(gpu_devices)))

    if requested_workers == 0:
        return [gpu_devices[:tp_size]]

    groups: List[List[str]] = []
    idx = 0
    for _ in range(requested_workers):
        if idx >= len(gpu_devices):
            break
        groups.append(gpu_devices[idx : idx + tp_size])
        idx += tp_size

    if not groups:
        groups = [[gpu_devices[0]]]
    return groups


def configure_visible_devices(device_group: List[str]) -> Tuple[str, Optional[str]]:
    """
    Limit CUDA visibility to the provided group and return (device_for_inputs, device_map).
    """
    gpu_ids = []
    for dev in device_group:
        if dev.startswith("cuda") and ":" in dev:
            gpu_ids.append(dev.split(":", maxsplit=1)[1])

    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        device_map: Optional[str] = "auto" if len(gpu_ids) > 1 else None
        return "cuda:0", device_map

    return canonical_device_id(device_group[0]), None


def prepare_model(
    model_id: str,
    device: str,
    cache_dir: Optional[str] = None,
    *,
    device_map: Optional[str] = None,
    local_files_only: bool = False,
):
    dtype = infer_dtype(device)
    model_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    trust_remote = is_llava_onevision(model_id)
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if trust_remote:
        model_kwargs["trust_remote_code"] = True

    print(
        f"[setup] Loading model {model_id} on {device} with dtype={dtype} "
        f"(local_only={local_files_only}, device_map={device_map})",
        file=sys.stderr,
    )
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote,
        local_files_only=local_files_only,
        **({"cache_dir": cache_dir} if cache_dir else {}),
    )
    if not hasattr(config, "vision_aspect_ratio"):
        image_aspect_ratio = getattr(config, "image_aspect_ratio", None)
        if image_aspect_ratio is not None:
            setattr(config, "vision_aspect_ratio", image_aspect_ratio)
    model_type = getattr(config, "model_type", "")
    architectures = getattr(config, "architectures", None) or []
    llava_onevision = model_type == "llava_onevision" or any(
        "LlavaOnevision" in arch for arch in architectures
    )
    trust_remote = trust_remote or llava_onevision
    if trust_remote and "trust_remote_code" not in model_kwargs:
        model_kwargs["trust_remote_code"] = True
    if getattr(config, "torch_dtype", None) is None and dtype is not None:
        try:
            config.torch_dtype = dtype
        except Exception:
            pass

    if model_type == "llava":
        text_cfg = getattr(config, "text_config", None)
        is_qwen_model = "qwen" in model_id.lower()
        needs_qwen_text = is_qwen_model and (
            text_cfg is None
            or getattr(text_cfg, "model_type", None) != "qwen2"
            or getattr(text_cfg, "hidden_size", None) != getattr(config, "hidden_size", None)
        )
        if needs_qwen_text and Qwen2Config is not None:
            base_kwargs = {}
            for attr in [
                "vocab_size",
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_hidden_layers",
                "num_key_value_heads",
                "rms_norm_eps",
                "max_position_embeddings",
                "rope_theta",
                "rope_scaling",
                "use_cache",
            ]:
                if hasattr(config, attr):
                    base_kwargs[attr] = getattr(config, attr)
            try:
                config.text_config = Qwen2Config(**base_kwargs)
            except Exception as exc:
                print(f"[warn] Failed to build Qwen2 text_config for Qwen-based LLaVA: {exc}", file=sys.stderr)
    processor = None
    if llava_onevision:
        if LlavaOnevisionForConditionalGeneration is None:
            raise ImportError("LlavaOnevisionForConditionalGeneration is required for OneVision models.")
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            config=config,
            local_files_only=local_files_only,
            **model_kwargs,
        )
        if LlavaOnevisionProcessor is not None:
            try:
                processor = LlavaOnevisionProcessor.from_pretrained(
                    model_id,
                    local_files_only=local_files_only,
                    trust_remote_code=True,
                )
            except Exception as exc:
                print(
                    f"[warn] Failed to load LlavaOnevisionProcessor; falling back to AutoProcessor: {exc}",
                    file=sys.stderr,
                )
    else:
        model_cls = AutoModelForImageTextToText
        if AutoModelForVisionText2Text is not None:
            model_cls = AutoModelForVisionText2Text
        elif AutoModelForVision2Seq is not None:
            model_cls = AutoModelForVision2Seq
        model = model_cls.from_pretrained(
            model_id,
            config=config,
            local_files_only=local_files_only,
            **model_kwargs,
        )

    if processor is None:
        processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote,
        )
    _ensure_processor_patch_size(processor, config)
    _ensure_processor_image_size(processor, config)
    if device_map is None:
        model.to(device)
    model.eval()
    return model, processor


def worker_process(
    worker_idx: int,
    device_group: List[str],
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
    seed_everything(args.seed, args.deterministic)
    local_save_dir = save_dir / f"worker_{worker_idx}" if save_dir else None
    if local_save_dir is not None:
        local_save_dir.mkdir(parents=True, exist_ok=True)

    local_device, device_map = configure_visible_devices(device_group)
    local_model_source = ensure_model_available(
        args.model_id,
        args.model_path,
        cache_dir=cache_dir,
        offline=args.offline,
    )
    print(
        f"[worker {worker_idx}] Using model from {local_model_source} on {local_device} (visible: {device_group})",
        file=sys.stderr,
    )

    local_model, local_processor = prepare_model(
        str(local_model_source),
        device=local_device,
        cache_dir=str(cache_dir),
        device_map=device_map,
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
    pope_jsonl_path = None
    if args.dataset.lower() == "pope" and getattr(args, "pope_jsonl", None):
        base = Path(args.pope_jsonl).expanduser().resolve()
        if base.suffix:
            pope_jsonl_path = base.with_name(f"{base.stem}_worker{worker_idx}{base.suffix}")
        else:
            pope_jsonl_path = base.with_name(f"{base.name}_worker{worker_idx}")

    local_viz_config = None
    if viz_config is not None and viz_config.enabled:
        local_viz_config = EmbeddingVizConfig(
            enabled=True,
            limit_samples=viz_config.limit_samples,
            out_dir=viz_config.out_dir / f"worker_{worker_idx}",
            perplexity=viz_config.perplexity,
        )

    analysis_outdir = None
    if (args.freq_analysis or args.dirichlet_analysis) and args.analysis_outdir:
        base_outdir = Path(args.analysis_outdir).expanduser().resolve()
        analysis_outdir = base_outdir / f"worker_{worker_idx}"

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
            freq_analysis=args.freq_analysis,
            freq_plots=args.freq_plots,
            freq_band_scheme=args.freq_band_scheme,
            freq_per_channel=args.freq_per_channel,
            dirichlet_analysis=args.dirichlet_analysis,
            analysis_outdir=analysis_outdir,
            dirichlet_plots=args.dirichlet_plots,
            analysis_only=args.freq_dirichlet_only,
            keep_noflip_analysis=getattr(args, "keep_noflip_analysis", False),
            pope_jsonl_path=pope_jsonl_path,
            pope_no_image_baseline=getattr(args, "pope_no_image_baseline", False),
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


def _content_image_sizes(content: List[Dict[str, Any]]) -> tuple:
    sizes: List[tuple] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image":
            img = part.get("image")
            if hasattr(img, "size"):
                sizes.append(img.size)
    return tuple(sizes)


def _ensure_user_prompt(
    image: Optional[Image.Image],
    prompt: str,
    user_prompt: Optional[UserPrompt] = None,
) -> UserPrompt:
    if user_prompt is not None:
        return user_prompt
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    cache_key = (prompt, (image.size,) if image is not None else tuple())
    return UserPrompt(content=content, text=prompt, cache_key=cache_key)


def _prompt_cache_key(
    prompt_data: UserPrompt, fallback_image: Optional[Image.Image]
) -> tuple:
    if prompt_data.cache_key is not None:
        return prompt_data.cache_key
    sizes = _content_image_sizes(prompt_data.content)
    if not sizes and fallback_image is not None:
        sizes = (fallback_image.size,)
    return (prompt_data.text, sizes)


@torch.inference_mode()
def score_option_loglik(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    option_text: str,
    device: str,
    context_len: int,
    user_prompt: Optional[UserPrompt] = None,
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
    prompt_data = _ensure_user_prompt(image, prompt, user_prompt)
    messages_full = [
        {
            "role": "user",
            "content": prompt_data.content,
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": option_text},
            ],
        },
    ]
    inputs_full = tokenize_conversations(
        processor,
        messages_full,
        device,
        add_generation_prompt=False,
    )

    input_ids = inputs_full["input_ids"]  # [1, L]
    full_len = input_ids.shape[1]

    try:
        outputs = model(**inputs_full)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return _sequential_fallback(exc) # type: ignore
        raise
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
    user_prompt: Optional[UserPrompt] = None,
) -> int:
    """
    Get the token length of the user-only context (image + prompt), using
    add_generation_prompt=True so the model is ready to generate assistant output.

    context_len is the length of input_ids for:
        messages = [user(prompt content)]
        add_generation_prompt=True
    """
    prompt_data = _ensure_user_prompt(image, prompt, user_prompt)
    messages_user = [
        {
            "role": "user",
            "content": prompt_data.content,
        }
    ]
    inputs_user = tokenize_conversations(
        processor,
        messages_user,
        device,
        add_generation_prompt=True,  # model expects assistant next
    )
    input_ids = inputs_user["input_ids"]  # [1, L_ctx]
    return int(input_ids.shape[1])


def get_context_length_cached(
    processor,
    image: Image.Image,
    prompt: str,
    device: str,
    cache: Dict[tuple, int],
    user_prompt: Optional[UserPrompt] = None,
) -> int:
    """
    Cache context lengths keyed by (prompt, image.size) to avoid repeated
    tokenization for every perturbation of the same prompt/size.
    """
    prompt_data = _ensure_user_prompt(image, prompt, user_prompt)
    key = _prompt_cache_key(prompt_data, image)
    if key in cache:
        return cache[key]
    ctx = get_context_length(processor, image, prompt, device, user_prompt=prompt_data)
    cache[key] = ctx
    return ctx


@torch.inference_mode()
def get_context_embedding(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: str,
    context_len_cache: Optional[Dict[tuple, int]] = None,
    user_prompt: Optional[UserPrompt] = None,
) -> Optional[torch.Tensor]:
    """
    Compute the context embedding (last hidden state at position context_len-1)
    for a prompt (open or MCQ).
    """
    cache: Dict[tuple, int] = (
        context_len_cache if context_len_cache is not None else {}
    )
    prompt_data = _ensure_user_prompt(image, prompt, user_prompt)
    context_len = get_context_length_cached(
        processor,
        image,
        prompt_data.text,
        device=device,
        cache=cache,
        user_prompt=prompt_data,
    )
    messages_user = [
        {
            "role": "user",
            "content": prompt_data.content,
        }
    ]
    inputs = tokenize_conversations(
        processor,
        messages_user,
        device,
        add_generation_prompt=True,
    )
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
    context_len_cache: Optional[Dict[tuple, int]] = None,
    user_prompt: Optional[UserPrompt] = None,
) -> Optional[torch.Tensor]:
    """
    Mean-pooled hidden state over answer tokens (positions >= context_len).
    """
    cache: Dict[tuple, int] = (
        context_len_cache if context_len_cache is not None else {}
    )
    prompt_data = _ensure_user_prompt(image, prompt, user_prompt)
    context_len = get_context_length_cached(
        processor,
        image,
        prompt_data.text,
        device=device,
        cache=cache,
        user_prompt=prompt_data,
    )
    messages_full = [
        {
            "role": "user",
            "content": prompt_data.content,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer_text}],
        },
    ]
    inputs = tokenize_conversations(
        processor,
        messages_full,
        device,
        add_generation_prompt=False,
    )
    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]  # [1, L, H]
    seq_len = hidden.shape[1]
    if seq_len <= context_len:
        return None
    answer_states = hidden[0, context_len:seq_len]  # [ans_len, H]
    pooled = answer_states.mean(dim=0)
    return pooled.detach().cpu()


def _radial_edges(max_r: float, scheme: str, r_flat: np.ndarray) -> np.ndarray:
    if max_r <= 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    if scheme == "log":
        edges = np.geomspace(1.0, max_r + 1.0, num=4) - 1.0
    elif scheme == "percentile":
        try:
            edges = np.percentile(r_flat, [0.0, 33.3, 66.6, 100.0])
        except Exception:
            edges = np.linspace(0.0, max_r, num=4)
    else:
        edges = np.linspace(0.0, max_r, num=4)
    # Ensure monotonically increasing
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges


def compute_radial_band_energies(
    img: Image.Image,
    target_size: Tuple[int, int],
    band_scheme: str = "equal",
    per_channel: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Compute radially averaged power and return total power plus mean energy in three bands
    (low/mid/high) over the shifted FFT magnitude spectrum.
    """
    arr = np.array(img.resize(target_size, RESAMPLE_BICUBIC), dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    arr = arr / 255.0
    h, w, c = arr.shape
    total_power = 0.0
    bands_acc = np.zeros(3, dtype=float)
    count = 0
    for ch in range(c):
        if not per_channel and ch > 0:
            break
        chan = arr[:, :, ch]
        power = np.abs(np.fft.fftshift(np.fft.fft2(chan))) ** 2
        total_power += float(power.mean())
        y, x = np.indices((h, w))
        r = np.sqrt((y - h / 2.0) ** 2 + (x - w / 2.0) ** 2)
        r_flat = r.ravel()
        p_flat = power.ravel()
        if r_flat.size == 0:
            continue
        max_r = float(r_flat.max())
        if max_r == 0.0:
            continue
        edges = _radial_edges(max_r, band_scheme, r_flat)
        for i in range(3):
            if i < 2:
                mask = (r_flat >= edges[i]) & (r_flat < edges[i + 1])
            else:
                mask = (r_flat >= edges[i]) & (r_flat <= edges[i + 1])
            if not np.any(mask):
                bands_acc[i] += 0.0
            else:
                bands_acc[i] += float(p_flat[mask].mean())
        count += 1
        if not per_channel:
            break
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
    bands = bands_acc / float(count)
    total_power = total_power / float(count)
    return total_power, bands[0], bands[1], bands[2]


@torch.inference_mode()
def get_vision_tokens(
    model,
    processor,
    image: Image.Image,
    device: str,
) -> Optional[torch.Tensor]:
    """
    Run only the vision encoder and return tokens reshaped to (T, H', W', D).
    """
    try:
        vision_inputs = processor.image_processor(
            images=[image],
            return_tensors="pt",
        )
    except Exception as exc:
        print(f"[warn] Vision preprocessing failed: {exc}", file=sys.stderr)
        return None

    pixel_values = vision_inputs["pixel_values"]
    grid = vision_inputs.get("image_grid_thw")
    image_sizes = vision_inputs.get("image_sizes")

    first_param = next(model.parameters(), None)
    dtype = first_param.dtype if first_param is not None else torch.float16
    device_obj = torch.device(device)
    pixel_values = pixel_values.to(device=device_obj, dtype=dtype)
    if grid is not None:
        grid = grid.to(device=device_obj)
    elif isinstance(image_sizes, torch.Tensor):
        image_sizes = image_sizes.to(device=device_obj)

    try:
        if grid is not None:
            outputs = model.get_image_features(
                pixel_values=pixel_values, image_grid_thw=grid
            )
        elif image_sizes is not None:
            outputs = model.get_image_features(
                pixel_values=pixel_values, image_sizes=image_sizes
            )
        else:
            outputs = model.get_image_features(pixel_values=pixel_values)
    except Exception as exc:
        print(f"[warn] Vision encoder failed: {exc}", file=sys.stderr)
        return None
    if not outputs:
        return None

    tokens = outputs[0]
    while isinstance(tokens, (tuple, list)):
        if not tokens:
            return None
        tokens = tokens[0]
    if not isinstance(tokens, torch.Tensor):
        return None
    if grid is not None:
        visual_module = getattr(getattr(model, "model", None), "visual", None)
        merge_size = int(getattr(visual_module, "spatial_merge_size", 1))
        t, h, w = [int(x) for x in grid[0].tolist()]
        h_tokens = max(1, h // merge_size)
        w_tokens = max(1, w // merge_size)
        expected = t * h_tokens * w_tokens

        if tokens.dim() == 2 and tokens.shape[0] == expected:
            tokens = tokens.view(t, h_tokens, w_tokens, -1)
        elif tokens.dim() == 3 and tokens.shape[0] == t and tokens.shape[1] * tokens.shape[2] == h_tokens * w_tokens:
            tokens = tokens.view(t, h_tokens, w_tokens, -1)
        else:
            try:
                side = int(round(math.sqrt(tokens.shape[0])))
                tokens = tokens.view(t, max(1, side), max(1, tokens.shape[0] // max(1, side)), -1)
            except Exception:
                return None
    else:
        if tokens.dim() != 2:
            return None
        vision_cfg = getattr(getattr(model, "config", None), "vision_config", None)
        base_grid = None
        if vision_cfg is not None:
            image_size = getattr(vision_cfg, "image_size", None)
            patch_size = getattr(vision_cfg, "patch_size", None)
            if image_size and patch_size:
                base_grid = int(image_size) // int(patch_size)
        if base_grid is not None and base_grid > 0:
            base_tokens = base_grid * base_grid
            if tokens.shape[0] >= base_tokens:
                tokens = tokens[:base_tokens].view(1, base_grid, base_grid, -1)
                return tokens
        try:
            side = int(round(math.sqrt(tokens.shape[0])))
            if side <= 0:
                return None
            tokens = tokens.view(1, side, max(1, tokens.shape[0] // side), -1)
        except Exception:
            return None
    return tokens


def dirichlet_energy_from_tokens(tokens: torch.Tensor) -> Optional[float]:
    """
    Compute 4-neighbor Dirichlet energy over vision tokens.
    """
    if tokens is None or tokens.ndim != 4:
        return None
    tok = tokens.float()
    horiz = tok[:, :, 1:, :] - tok[:, :, :-1, :]
    vert = tok[:, 1:, :, :] - tok[:, :-1, :, :]
    num_edges = (
        tok.shape[0] * tok.shape[1] * max(tok.shape[2] - 1, 1)
        + tok.shape[0] * max(tok.shape[1] - 1, 1) * tok.shape[2]
    )
    denom = float(num_edges) if num_edges > 0 else 1.0
    energy = (horiz.pow(2).sum() + vert.pow(2).sum()) / denom
    return float(energy.item())


@torch.inference_mode()
def score_options_loglik_batch(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    options: Dict[str, str],
    device: str,
    context_len: int,
    user_prompt: Optional[UserPrompt] = None,
    chunk_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Batch version of option scoring to reduce forward passes.
    Returns a mapping {label: loglikelihood}.
    """
    if not options:
        return {}

    prompt_data = _ensure_user_prompt(image, prompt, user_prompt)
    messages_batch = []
    labels: List[str] = []

    def _sequential_fallback(exc: Exception | None = None) -> Dict[str, float]:
        global BATCH_LOGPROB_FALLBACK_WARNED
        if exc is not None and not BATCH_LOGPROB_FALLBACK_WARNED:
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
                user_prompt=prompt_data,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return scores

    for lab, opt_text in options.items():
        labels.append(lab)
        messages_batch.append(
            [
                {
                    "role": "user",
                    "content": prompt_data.content,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": opt_text}],
                },
            ]
        )

    scores: Dict[str, float] = {}
    eff_chunk = len(labels) if chunk_size is None else chunk_size
    if eff_chunk <= 0:
        eff_chunk = len(labels)
    for start in range(0, len(labels), eff_chunk):
        end = start + eff_chunk
        chunk_labels = labels[start:end]
        chunk_messages = messages_batch[start:end]
        try:
            inputs_full = tokenize_conversations(
                processor,
                chunk_messages,
                device,
                add_generation_prompt=False,
                padding=True,
            )
        except Exception as exc:
            return _sequential_fallback(exc)

        input_ids = inputs_full["input_ids"]  # [B, L]
        attn_mask = inputs_full.get("attention_mask")
        try:
            outputs = model(**inputs_full)
            logits = outputs.logits  # [B, L, vocab]
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

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
                scores[chunk_labels[i]] = score
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                return _sequential_fallback(exc)
            raise
    return scores


@torch.inference_mode()
def choose_label_via_loglik(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    options: Dict[str, str],
    device: str,
    context_len_cache: Optional[Dict[tuple, int]] = None,
    user_prompt: Optional[UserPrompt] = None,
) -> Optional[str]:
    """
    For this image + prompt, compute log-likelihood for each option_text
    and return the argmax option label (A/B/C/...).

    Returns:
        label (e.g. "A", "B", ...) or None if no options.
    """
    if not options:
        return None

    cache: Dict[tuple, int] = (
        context_len_cache if context_len_cache is not None else {}
    )
    prompt_data = _ensure_user_prompt(image, prompt, user_prompt)
    context_len = get_context_length_cached(
        processor,
        image,
        prompt_data.text,
        device=device,
        cache=cache,
        user_prompt=prompt_data,
    )

    scores = score_options_loglik_batch(
        model,
        processor,
        image,
        prompt_data.text,
        options,
        device=device,
        context_len=context_len,
        user_prompt=prompt_data,
        chunk_size=OPTION_BATCH_CHUNK_SIZE,
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
    user_prompt: Optional[UserPrompt] = None,
) -> str:
    """
    Free-text answer generation with Qwen3-VL, same style as your earlier script.
    Currently only used if compare_mode != 'label'.
    """
    messages = [
        {
            "role": "user",
            "content": _ensure_user_prompt(image, prompt, user_prompt).content,
        }
    ]
    inputs = tokenize_conversations(
        processor,
        messages,
        device,
        add_generation_prompt=True,
    )

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


class FrequencyAnalysisHelper:
    """
    Optional frequency-domain energy tracking per perturbation type.
    """

    def __init__(
        self,
        enabled: bool,
        perturb_types: List[str],
        plot_dir: Optional[Path] = None,
        save_plots: bool = False,
        band_scheme: str = "equal",
        per_channel: bool = False,
    ):
        self.enabled = enabled
        self.per_type: Dict[str, List[FrequencyBandDelta]] = {
            t: [] for t in perturb_types
        }
        self.per_type_flipped: Dict[str, List[FrequencyBandDelta]] = {
            t: [] for t in perturb_types
        }
        self.plot_dir = plot_dir
        self.save_plots = save_plots and plot_dir is not None
        self.band_scheme = band_scheme
        self.per_channel = per_channel
        self._base_cache: Dict[int, Tuple[float, float, float, float]] = {}
        self._by_sample: Dict[str, Dict[int, FrequencyBandDelta]] = {
            t: {} for t in perturb_types
        }

    def _ensure_base(self, sample_idx: int, base_img: Image.Image) -> Tuple[float, float, float, float]:
        if sample_idx not in self._base_cache:
            self._base_cache[sample_idx] = compute_radial_band_energies(
                base_img,
                target_size=base_img.size,
                band_scheme=self.band_scheme,
                per_channel=self.per_channel,
            )
        return self._base_cache[sample_idx]

    def record(
        self,
        sample_idx: int,
        pert_type: str,
        base_img: Image.Image,
        pert_img: Image.Image,
        is_flip: bool,
    ) -> Optional[FrequencyBandDelta]:
        if not self.enabled or pert_type not in self.per_type:
            return None
        base_total, base_low, base_mid, base_high = self._ensure_base(sample_idx, base_img)
        pert_resized = pert_img.resize(base_img.size, RESAMPLE_BICUBIC)
        pert_total, pert_low, pert_mid, pert_high = compute_radial_band_energies(
            pert_resized, target_size=base_img.size, band_scheme=self.band_scheme, per_channel=self.per_channel
        )
        delta = FrequencyBandDelta(
            sample_idx=sample_idx,
            delta_low=pert_low - base_low,
            delta_mid=pert_mid - base_mid,
            delta_high=pert_high - base_high,
            rel_low=(pert_low - base_low) / (abs(base_low) + 1e-8),
            rel_mid=(pert_mid - base_mid) / (abs(base_mid) + 1e-8),
            rel_high=(pert_high - base_high) / (abs(base_high) + 1e-8),
            is_flip=is_flip,
        )
        self.per_type[pert_type].append(delta)
        if is_flip:
            self.per_type_flipped[pert_type].append(delta)
        self._by_sample[pert_type][sample_idx] = delta
        return delta

    def lookup(self, pert_type: str, sample_idx: int) -> Optional[FrequencyBandDelta]:
        return self._by_sample.get(pert_type, {}).get(sample_idx)

    def drop_sample(self, sample_idx: int) -> None:
        """Remove all records for a sample (used when no flips occur for that base)."""
        for pert_type in list(self.per_type.keys()):
            self.per_type[pert_type] = [
                d for d in self.per_type[pert_type] if d.sample_idx != sample_idx
            ]
            self.per_type_flipped[pert_type] = [
                d for d in self.per_type_flipped[pert_type] if d.sample_idx != sample_idx
            ]
            self._by_sample.get(pert_type, {}).pop(sample_idx, None)

    def summary_lines(self) -> List[str]:
        if not self.enabled:
            return []
        lines = [
            "\n====== FREQUENCY ANALYSIS (Δ band energy: pert - base) ======",
            "Mean/std of low/mid/high band energy deltas (radially averaged FFT power). Relative deltas are normalized by base band power.",
        ]
        data_abs: Dict[str, Tuple[float, float, float]] = {}
        data_rel: Dict[str, Tuple[float, float, float]] = {}
        flipped_abs: Dict[str, Tuple[float, float, float]] = {}
        flipped_rel: Dict[str, Tuple[float, float, float]] = {}
        pert_types = list(self.per_type.keys())
        for pert_type, entries in self.per_type.items():
            lows = [e.delta_low for e in entries]
            mids = [e.delta_mid for e in entries]
            highs = [e.delta_high for e in entries]
            lows_rel = [e.rel_low for e in entries]
            mids_rel = [e.rel_mid for e in entries]
            highs_rel = [e.rel_high for e in entries]
            m_low, s_low = _mean_std(lows)
            m_mid, s_mid = _mean_std(mids)
            m_high, s_high = _mean_std(highs)
            m_low_r, s_low_r = _mean_std(lows_rel)
            m_mid_r, s_mid_r = _mean_std(mids_rel)
            m_high_r, s_high_r = _mean_std(highs_rel)
            data_abs[pert_type] = (m_low, m_mid, m_high)
            data_rel[pert_type] = (m_low_r, m_mid_r, m_high_r)
            lines.append(
                f"{pert_type:<12} low={m_low:9.3f}±{s_low:7.3f} ({m_low_r:7.3f}±{s_low_r:5.3f} rel) | mid={m_mid:9.3f}±{s_mid:7.3f} ({m_mid_r:7.3f}±{s_mid_r:5.3f} rel) | high={m_high:9.3f}±{s_high:7.3f} ({m_high_r:7.3f}±{s_high_r:5.3f} rel) (n={len(entries)})"
            )
            flipped = self.per_type_flipped.get(pert_type, [])
            if flipped:
                f_low, f_std_low = _mean_std([e.delta_low for e in flipped])
                f_mid, f_std_mid = _mean_std([e.delta_mid for e in flipped])
                f_high, f_std_high = _mean_std([e.delta_high for e in flipped])
                f_low_r, f_std_low_r = _mean_std([e.rel_low for e in flipped])
                f_mid_r, f_std_mid_r = _mean_std([e.rel_mid for e in flipped])
                f_high_r, f_std_high_r = _mean_std([e.rel_high for e in flipped])
                flipped_abs[pert_type] = (f_low, f_mid, f_high)
                flipped_rel[pert_type] = (f_low_r, f_mid_r, f_high_r)
                lines.append(
                    f"{'':<12} [flips] low={f_low:9.3f}±{f_std_low:7.3f} ({f_low_r:7.3f}±{f_std_low_r:5.3f} rel) | mid={f_mid:9.3f}±{f_std_mid:7.3f} ({f_mid_r:7.3f}±{f_std_mid_r:5.3f} rel) | high={f_high:9.3f}±{f_std_high:7.3f} ({f_high_r:7.3f}±{f_std_high_r:5.3f} rel) (n={len(flipped)})"
                )
            if self.save_plots:
                self.plot_dir.mkdir(parents=True, exist_ok=True)
                self._plot_bars(
                    data_abs,
                    title="Frequency band Δ (abs)",
                    filename="freq_bands_abs.png",
                    perturb_types=pert_types,
                )
                self._plot_bars(
                    data_rel,
                    title="Frequency band Δ (relative)",
                    filename="freq_bands_rel.png",
                    perturb_types=pert_types,
                )
                if flipped_abs:
                    self._plot_bars(
                        flipped_abs,
                        title="Frequency band Δ (abs, flips only)",
                        filename="freq_bands_abs_flips.png",
                        perturb_types=pert_types,
                    )
                if flipped_rel:
                    self._plot_bars(
                        flipped_rel,
                        title="Frequency band Δ (relative, flips only)",
                        filename="freq_bands_rel_flips.png",
                        perturb_types=pert_types,
                    )
        return lines

    def _plot_bars(
        self,
        data: Dict[str, Tuple[float, float, float]],
        title: str,
        filename: str,
        perturb_types: List[str],
    ) -> None:
        if not data:
            return
        labels = perturb_types
        low_vals = [data.get(p, (0.0, 0.0, 0.0))[0] for p in labels]
        mid_vals = [data.get(p, (0.0, 0.0, 0.0))[1] for p in labels]
        high_vals = [data.get(p, (0.0, 0.0, 0.0))[2] for p in labels]
        x = np.arange(len(labels))
        width = 0.25
        plt.figure(figsize=(10, 5))
        plt.bar(x - width, low_vals, width, label="low")
        plt.bar(x, mid_vals, width, label="mid")
        plt.bar(x + width, high_vals, width, label="high")
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylabel("Mean Δ")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        out_path = self.plot_dir / filename
        plt.savefig(out_path)
        plt.close()


class DirichletAnalysisHelper:
    """
    Optional Dirichlet energy tracking over vision tokens + correlations.
    """

    def __init__(
        self,
        enabled: bool,
        model,
        processor,
        device: str,
        perturb_types: List[str],
        plot_dir: Optional[Path] = None,
        save_plots: bool = False,
    ):
        self.enabled = enabled
        self.model = model
        self.processor = processor
        self.device = device
        self.per_type: Dict[str, List[DirichletRecord]] = {
            t: [] for t in perturb_types
        }
        self.plot_dir = plot_dir
        self.save_plots = save_plots and plot_dir is not None

    def prepare_base(
        self,
        sample_idx: int,
        image: Image.Image,
        prompt: str,
        context_len_cache: Dict[tuple, int],
        user_prompt: Optional[UserPrompt] = None,
    ) -> Optional[DirichletBase]:
        if not self.enabled:
            return None
        tokens = get_vision_tokens(self.model, self.processor, image, device=self.device)
        base_energy = dirichlet_energy_from_tokens(tokens) if tokens is not None else None
        if base_energy is None:
            return None
        ctx_emb = get_context_embedding(
            self.model,
            self.processor,
            image,
            prompt,
            device=self.device,
            context_len_cache=context_len_cache,
            user_prompt=user_prompt,
        )
        return DirichletBase(energy=base_energy, context_embedding=ctx_emb)

    def record(
        self,
        sample_idx: int,
        pert_type: str,
        base: DirichletBase,
        pert_img: Image.Image,
        prompt: str,
        context_len_cache: Dict[tuple, int],
        user_prompt: Optional[UserPrompt] = None,
        freq_delta: Optional[FrequencyBandDelta] = None,
        is_flip: bool = False,
        delta_base_score: Optional[float] = None,
    ) -> None:
        if not self.enabled or pert_type not in self.per_type:
            return
        tokens = get_vision_tokens(self.model, self.processor, pert_img, device=self.device)
        energy = dirichlet_energy_from_tokens(tokens) if tokens is not None else None
        if energy is None:
            return
        delta_energy = energy - base.energy
        embedding_drift = None
        if base.context_embedding is not None:
            pert_ctx = get_context_embedding(
            self.model,
            self.processor,
            pert_img,
            prompt,
            device=self.device,
            context_len_cache=context_len_cache,
            user_prompt=user_prompt,
        )
        if pert_ctx is not None:
            embedding_drift = torch.norm(
                base.context_embedding - pert_ctx, p=2
            ).item()

        delta_low = freq_delta.delta_low if freq_delta is not None else None
        delta_high = freq_delta.delta_high if freq_delta is not None else None
        ratio = None
        if delta_low is not None and delta_high is not None:
            denom = delta_high if abs(delta_high) > 1e-8 else 1e-8
            ratio = delta_low / denom
        self.per_type[pert_type].append(
            DirichletRecord(
                sample_idx=sample_idx,
                delta_energy=delta_energy,
                embedding_drift=embedding_drift,
                is_flip=is_flip,
                delta_base_score=delta_base_score,
                delta_low=delta_low,
                delta_high=delta_high,
                delta_ratio=ratio,
            )
        )

    def _format_corr(self, val: Optional[float]) -> str:
        return f"{val:6.3f}" if val is not None else "  N/A"

    def drop_sample(self, sample_idx: int) -> None:
        """Remove all records for a sample (used when no flips occur for that base)."""
        for pert_type in list(self.per_type.keys()):
            self.per_type[pert_type] = [
                d for d in self.per_type[pert_type] if d.sample_idx != sample_idx
            ]

    def _maybe_plot(self, pert_type: str, deltas: List[float], suffix: str = "") -> None:
        if not (self.save_plots and self.plot_dir and deltas):
            return
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        arr = np.array(deltas, dtype=float)
        if len(arr) == 1:
            grid = np.linspace(arr[0] - 1e-3, arr[0] + 1e-3, 200)
        else:
            grid = np.linspace(arr.min(), arr.max(), 200)
        kde = None
        if len(arr) > 1 and arr.std() > 0:
            bw = 1.06 * arr.std() * (len(arr) ** (-1 / 5))
            if bw > 0:
                kde = np.zeros_like(grid)
                norm = bw * math.sqrt(2 * math.pi)
                for val in arr:
                    kde += np.exp(-0.5 * ((grid - val) / bw) ** 2) / norm
                kde /= len(arr)
        plt.figure(figsize=(6, 4))
        bins = max(10, min(50, int(math.sqrt(len(arr))))) if len(arr) > 1 else 5
        plt.hist(arr, bins=bins, density=True, alpha=0.7, color="tab:blue", label="ΔE_dir")
        if kde is not None:
            plt.plot(grid, kde, color="tab:orange", lw=2, label="KDE")
        plt.title(f"Dirichlet ΔE | {pert_type}")
        plt.xlabel("ΔE_dir (pert - base)")
        plt.ylabel("Density")
        plt.legend(fontsize=8)
        plt.tight_layout()
        suffix_clean = f"_{suffix}" if suffix else ""
        out_path = self.plot_dir / f"dirichlet_{pert_type.replace('/', '_')}{suffix_clean}.png"
        plt.savefig(out_path)
        plt.close()

    def summary_lines(self) -> List[str]:
        if not self.enabled:
            return []
        lines = [
            "\n====== DIRICHLET ANALYSIS (vision token smoothness) ======",
            "ΔE_dir = Dirichlet(pert) - Dirichlet(base); corr vs embedding drift and frequency shifts.",
        ]
        for pert_type, entries in self.per_type.items():
            deltas = [e.delta_energy for e in entries]
            m_dir, s_dir = _mean_std(deltas)
            flipped = [e for e in entries if e.is_flip]
            nonflipped = [e for e in entries if not e.is_flip]
            if flipped:
                f_deltas = [e.delta_energy for e in flipped]
                f_mean, f_std = _mean_std(f_deltas)
            else:
                f_mean = f_std = 0.0

            drift_pairs = [
                (e.delta_energy, e.embedding_drift)
                for e in entries
                if e.embedding_drift is not None
            ]
            low_pairs = [
                (e.delta_energy, e.delta_low)
                for e in entries
                if e.delta_low is not None
            ]
            high_pairs = [
                (e.delta_energy, e.delta_high)
                for e in entries
                if e.delta_high is not None
            ]
            ratio_pairs = [
                (e.delta_energy, e.delta_ratio)
                for e in entries
                if e.delta_ratio is not None
            ]
            corr_drift = _pearson_corr(
                [a for a, b in drift_pairs], [b for a, b in drift_pairs]
            )
            corr_low = _pearson_corr(
                [a for a, b in low_pairs], [b for a, b in low_pairs]
            )
            corr_high = _pearson_corr(
                [a for a, b in high_pairs], [b for a, b in high_pairs]
            )
            corr_ratio = _pearson_corr(
                [a for a, b in ratio_pairs], [b for a, b in ratio_pairs]
            )
            lines.append(
                f"{pert_type:<12} ΔE mean={m_dir:9.3f}±{s_dir:7.3f} (n={len(entries)}) | flips mean={f_mean:9.3f}±{f_std:7.3f} (n={len(flipped)})"
            )
            lines.append(
                f"{'':<12} corr(dE, drift)={self._format_corr(corr_drift)} | corr(dE, Δlow)={self._format_corr(corr_low)} | corr(dE, Δhigh)={self._format_corr(corr_high)} | corr(dE, Δlow/Δhigh)={self._format_corr(corr_ratio)}"
            )
            if nonflipped:
                score_pairs = [
                    (e.delta_energy, e.delta_base_score)
                    for e in nonflipped
                    if e.delta_base_score is not None
                ]
                corr_score = _pearson_corr(
                    [a for a, b in score_pairs], [b for a, b in score_pairs]
                )
                lines.append(
                    f"{'':<12} [non-flip] corr(dE, Δlogprob_base)={self._format_corr(corr_score)} (n={len(score_pairs)})"
                )
            if flipped:
                drift_pairs_flip = [
                    (e.delta_energy, e.embedding_drift)
                    for e in flipped
                    if e.embedding_drift is not None
                ]
                low_pairs_flip = [
                    (e.delta_energy, e.delta_low)
                    for e in flipped
                    if e.delta_low is not None
                ]
                high_pairs_flip = [
                    (e.delta_energy, e.delta_high)
                    for e in flipped
                    if e.delta_high is not None
                ]
                ratio_pairs_flip = [
                    (e.delta_energy, e.delta_ratio)
                    for e in flipped
                    if e.delta_ratio is not None
                ]
                corr_drift_flip = _pearson_corr(
                    [a for a, b in drift_pairs_flip], [b for a, b in drift_pairs_flip]
                )
                corr_low_flip = _pearson_corr(
                    [a for a, b in low_pairs_flip], [b for a, b in low_pairs_flip]
                )
                corr_high_flip = _pearson_corr(
                    [a for a, b in high_pairs_flip], [b for a, b in high_pairs_flip]
                )
                corr_ratio_flip = _pearson_corr(
                    [a for a, b in ratio_pairs_flip], [b for a, b in ratio_pairs_flip]
                )
                lines.append(
                    f"{'':<12} [flips] corr(dE, drift)={self._format_corr(corr_drift_flip)} | corr(dE, Δlow)={self._format_corr(corr_low_flip)} | corr(dE, Δhigh)={self._format_corr(corr_high_flip)} | corr(dE, Δlow/Δhigh)={self._format_corr(corr_ratio_flip)}"
                )
            self._maybe_plot(pert_type, deltas)
            if flipped:
                self._maybe_plot(pert_type, [e.delta_energy for e in flipped], suffix="flips")
        return lines


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
    freq_analysis: bool = False,
    freq_plots: bool = False,
    freq_band_scheme: str = "equal",
    freq_per_channel: bool = False,
    dirichlet_analysis: bool = False,
    analysis_outdir: Optional[Path] = None,
    dirichlet_plots: bool = False,
    analysis_only: bool = False,
    keep_noflip_analysis: bool = False,
    pope_jsonl_path: Optional[Path] = None,
    pope_no_image_baseline: bool = False,
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
        "BoxOverlay",
        "RandomText",
        "Rotation",
    ]
    pope_enabled = bool(samples) and samples[0].dataset.lower() == "pope"
    pope_metrics = pope_enabled and compare_mode == "label"
    pope_jsonl_active = pope_enabled and pope_jsonl_path is not None
    pope_blank_enabled = pope_enabled and pope_no_image_baseline
    stats: Dict[str, PerturbStats] = {t: PerturbStats() for t in perturb_types}
    any_stats = PerturbStats()
    saved_examples = 0
    context_len_cache: Dict[tuple, int] = {}
    pope_counts = {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "pred_yes": 0,
        "pred_total": 0,
    }
    pope_yes_stats = {t: PerturbStats() for t in perturb_types} if pope_metrics else {}
    pope_no_stats = {t: PerturbStats() for t in perturb_types} if pope_metrics else {}
    pope_yes_any = PerturbStats()
    pope_no_any = PerturbStats()
    pope_jsonl_file = None
    if pope_jsonl_active and pope_jsonl_path is not None:
        pope_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        pope_jsonl_file = open(pope_jsonl_path, "a", encoding="utf-8")
    freq_helper = (
        FrequencyAnalysisHelper(
            freq_analysis,
            perturb_types,
            plot_dir=analysis_outdir / "frequency" if (analysis_outdir and freq_plots) else None,
            save_plots=freq_plots,
            band_scheme=freq_band_scheme,
            per_channel=freq_per_channel,
        )
        if freq_analysis
        else None
    )
    dir_plot_dir = (
        analysis_outdir / "dirichlet_energy"
        if (analysis_outdir is not None and dirichlet_analysis)
        else None
    )
    dirichlet_helper = (
        DirichletAnalysisHelper(
            dirichlet_analysis,
            model,
            processor,
            device=device,
            perturb_types=perturb_types,
            plot_dir=dir_plot_dir,
            save_plots=dirichlet_plots,
        )
        if dirichlet_analysis
        else None
    )

    N_total = len(samples)
    N_used = 0  # samples with a valid base representation
    samples_with_gt = 0  # samples where a ground-truth label is available
    base_correct_samples = 0

    if analysis_only:
        print(f"[info] Running frequency/Dirichlet analysis on {N_total} samples (analysis-only mode)...")
        if summary_path is not None:
            append_line(summary_path, f"{progress_prefix}start freq/dirichlet analysis on {N_total} samples")
    else:
        print(f"[info] Running invariance experiment on {N_total} samples...")
        if summary_path is not None:
            append_line(summary_path, f"{progress_prefix}start label invariance on {N_total} samples")
    for i, sample in enumerate(samples, start=1):
        images = load_sample_images(sample)
        if not images:
            continue
        primary_img = images[0]
        prompt_style = prompt_style_from_compare_mode(compare_mode)

        def build_prompt(imgs: List[Image.Image]) -> UserPrompt:
            return build_user_prompt(sample, imgs, prompt_style)

        base_prompt = build_prompt(images)
        base_label_logprob: Optional[float] = None
        base_scores: Optional[Dict[str, float]] = None
        base_margin: Optional[float] = None

        # ---------------- Base representation ----------------
        if compare_mode == "label":
            base_repr = choose_label_via_loglik(
                model,
                processor,
                primary_img,
                base_prompt.text,
                sample.options,
                device=device,
                context_len_cache=context_len_cache,
                user_prompt=base_prompt,
            )
            base_text_for_log = f"[loglik argmax] {base_repr}"
            base_label_logprob = None
            try:
                ctx_len_base = get_context_length_cached(
                    processor,
                    primary_img,
                    base_prompt.text,
                    device=device,
                    cache=context_len_cache,
                    user_prompt=base_prompt,
                )
                base_scores = score_options_loglik_batch(
                    model,
                    processor,
                    primary_img,
                    base_prompt.text,
                    sample.options,
                    device=device,
                    context_len=ctx_len_base,
                    user_prompt=base_prompt,
                    chunk_size=OPTION_BATCH_CHUNK_SIZE,
                )
                if base_scores:
                    base_label_logprob = base_scores.get(base_repr)
            except Exception as exc:
                print(f"[warn] Failed to compute base label score for sample {sample.index}: {exc}", file=sys.stderr)
        else:
            base_text = generate_free_answer(
                model,
                processor,
                primary_img,
                base_prompt.text,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                user_prompt=base_prompt,
            )
            base_repr = get_representation_text_or_semantic(
                base_text,
                compare_mode,
            )
            base_text_for_log = base_text

        print(f"\n=== Sample {i}/{N_total} (index={sample.index}) ===")
        print(f"[meta] dataset={sample.dataset} split={sample.split or '-'} images={len(images)}")
        print(f"Q: {sample.question}")
        if sample.options:
            print("Options: " + " | ".join(f"{k}: {v}" for k, v in sample.options.items()))
        print(f"[base answer] {base_text_for_log}")
        print(f"[base repr ({compare_mode})] {base_repr}")

        if base_repr is None:
            print("[info] Base representation is None; skipping this sample.")
            continue

        yes_label, no_label = (None, None)
        if pope_enabled:
            yes_label, no_label = _pope_yes_no_labels(sample.options)
            if base_scores:
                base_margin = _pope_margin_from_scores(base_scores, yes_label, no_label)

        gt_label = sample.answer_label
        base_correct: Optional[bool] = None
        pope_pred = _pope_label_to_yesno(base_repr, sample.options) if pope_enabled else None
        pope_gt = _pope_label_to_yesno(gt_label, sample.options) if pope_enabled else None
        pope_base_yes = pope_metrics and pope_pred == "Yes" and pope_gt == "Yes"
        pope_base_no = pope_metrics and pope_pred == "No" and pope_gt == "No"
        if gt_label is not None:
            base_correct = base_repr == gt_label
            samples_with_gt += 1
            if base_correct:
                base_correct_samples += 1
            correctness_text = "correct" if base_correct else "WRONG"
            print(f"[base vs GT] {base_repr} vs {gt_label} ({correctness_text})")
        else:
            print("[base vs GT] ground truth unavailable for this sample.")

        if pope_metrics and pope_pred is not None:
            pope_counts["pred_total"] += 1
            if pope_pred == "Yes":
                pope_counts["pred_yes"] += 1
            if pope_gt is not None:
                if pope_pred == "Yes":
                    if pope_gt == "Yes":
                        pope_counts["tp"] += 1
                    else:
                        pope_counts["fp"] += 1
                else:
                    if pope_gt == "No":
                        pope_counts["tn"] += 1
                    else:
                        pope_counts["fn"] += 1

        N_used += 1

        freq_dirichlet_enabled = len(images) == 1
        if (freq_helper is not None or dirichlet_helper is not None) and not freq_dirichlet_enabled:
            print("[info] Skipping frequency/Dirichlet analysis for multi-image sample.")

        dirichlet_base = (
            dirichlet_helper.prepare_base(
                sample.index,
                primary_img,
                base_prompt.text,
                context_len_cache=context_len_cache,
                user_prompt=base_prompt,
            )
            if dirichlet_helper is not None and freq_dirichlet_enabled
            else None
        )

        per_image_changed_any = False
        per_image_changed_type: Dict[str, bool] = {t: False for t in perturb_types}
        reprs_for_entropy: List[Optional[str]] = [base_repr]
        flip_margins: List[float] = []
        noflip_margins: List[float] = []
        blank_pred = None
        blank_margin = None

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

        def update_pope_base_confusion(
            ps: PerturbStats,
            any_ps: PerturbStats,
            pert_repr: Optional[str],
        ) -> None:
            if gt_label is None or pert_repr is None:
                return
            ps.gt_evaluable += 1
            any_ps.gt_evaluable += 1
            if pert_repr == gt_label:
                ps.right_to_right += 1
                any_ps.right_to_right += 1
            else:
                ps.right_to_wrong += 1
                any_ps.right_to_wrong += 1

        def record_freq_delta(
            pert_type: str, pert_img: Image.Image, is_flip: bool
        ) -> Optional[FrequencyBandDelta]:
            if not freq_dirichlet_enabled or freq_helper is None:
                return None
            return freq_helper.record(
                sample.index, pert_type, primary_img, pert_img, is_flip=is_flip
            )

        def record_dirichlet_delta(
            pert_type: str,
            pert_img: Image.Image,
            prompt_variant: UserPrompt,
            freq_delta: Optional[FrequencyBandDelta],
            is_flip: bool,
            delta_base_score: Optional[float],
        ) -> None:
            if (
                not freq_dirichlet_enabled
                or dirichlet_base is None
                or dirichlet_helper is None
            ):
                return
            dirichlet_helper.record(
                sample.index,
                pert_type,
                dirichlet_base,
                pert_img,
                prompt_variant.text,
                context_len_cache=context_len_cache,
                user_prompt=prompt_variant,
                freq_delta=freq_delta,
                is_flip=is_flip,
                delta_base_score=delta_base_score,
            )

        # Helper to compute repr for perturbed image + base-label score delta (label mode)
        def compute_repr_for_image(
            vimgs: List[Image.Image],
        ) -> tuple[Optional[str], Optional[float], UserPrompt, Optional[float]]:
            prompt_variant = build_prompt(vimgs)
            if compare_mode == "label":
                ctx_len_variant = get_context_length_cached(
                    processor,
                    vimgs[0],
                    prompt_variant.text,
                    device=device,
                    cache=context_len_cache,
                    user_prompt=prompt_variant,
                )
                scores = score_options_loglik_batch(
                    model,
                    processor,
                    vimgs[0],
                    prompt_variant.text,
                    sample.options,
                    device=device,
                    context_len=ctx_len_variant,
                    user_prompt=prompt_variant,
                    chunk_size=OPTION_BATCH_CHUNK_SIZE,
                )
                if not scores:
                    return None, None, prompt_variant, None
                label = max(scores, key=scores.get)
                base_score_variant = scores.get(base_repr) if base_repr is not None else None
                margin = (
                    _pope_margin_from_scores(scores, yes_label, no_label)
                    if pope_metrics
                    else None
                )
                return label, base_score_variant, prompt_variant, margin
            else:
                ans = generate_free_answer(
                    model,
                    processor,
                    vimgs[0],
                    prompt_variant.text,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    user_prompt=prompt_variant,
                )
                return (
                    get_representation_text_or_semantic(ans, compare_mode),
                    None,
                    prompt_variant,
                    None,
                )

        def maybe_save_changed_pair(
            tag: str, vimgs: List[Image.Image], changed_label: Optional[str]
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
                primary_img.save(base_out)
                vimgs[0].convert("RGB").save(pert_out)
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
            vimgs = [cyclic_horizontal_shift(im, n) for im in images]
            r, variant_base_score, prompt_variant, margin = compute_repr_for_image(vimgs)
            delta_base_score = (
                variant_base_score - base_label_logprob
                if variant_base_score is not None and base_label_logprob is not None
                else None
            )
            changed = r is not None and r != base_repr
            if pope_metrics and margin is not None:
                if changed:
                    flip_margins.append(margin)
                else:
                    noflip_margins.append(margin)
            freq_delta = record_freq_delta("Translation", vimgs[0], is_flip=bool(changed))
            record_dirichlet_delta(
                "Translation",
                vimgs[0],
                prompt_variant,
                freq_delta,
                is_flip=bool(changed),
                delta_base_score=delta_base_score,
            )
            stats["Translation"].total_instances += 1
            any_stats.total_instances += 1

            if r is not None:
                reprs_for_entropy.append(r)
                update_gt_confusion(stats["Translation"], r)
                if pope_base_yes:
                    update_pope_base_confusion(
                        pope_yes_stats["Translation"],
                        pope_yes_any,
                        r,
                    )
                if pope_base_no:
                    update_pope_base_confusion(
                        pope_no_stats["Translation"],
                        pope_no_any,
                        r,
                    )
                if changed:
                    stats["Translation"].changed_instances += 1
                    any_stats.changed_instances += 1
                    per_image_changed_type["Translation"] = True
                    per_image_changed_any = True
                    maybe_save_changed_pair("translation", vimgs, r)

        # ---------------- Pad/Crop -------------------
        for n in padcrop_pixels:
            if n == 0:
                continue
            vimgs = [pad_or_crop(im, n) for im in images]
            r, variant_base_score, prompt_variant, margin = compute_repr_for_image(vimgs)
            delta_base_score = (
                variant_base_score - base_label_logprob
                if variant_base_score is not None and base_label_logprob is not None
                else None
            )
            changed = r is not None and r != base_repr
            if pope_metrics and margin is not None:
                if changed:
                    flip_margins.append(margin)
                else:
                    noflip_margins.append(margin)
            freq_delta = record_freq_delta("Pad/Crop", vimgs[0], is_flip=bool(changed))
            record_dirichlet_delta(
                "Pad/Crop",
                vimgs[0],
                prompt_variant,
                freq_delta,
                is_flip=bool(changed),
                delta_base_score=delta_base_score,
            )
            stats["Pad/Crop"].total_instances += 1
            any_stats.total_instances += 1

            if r is not None:
                reprs_for_entropy.append(r)
                update_gt_confusion(stats["Pad/Crop"], r)
                if pope_base_yes:
                    update_pope_base_confusion(
                        pope_yes_stats["Pad/Crop"],
                        pope_yes_any,
                        r,
                    )
                if pope_base_no:
                    update_pope_base_confusion(
                        pope_no_stats["Pad/Crop"],
                        pope_no_any,
                        r,
                    )
                if changed:
                    stats["Pad/Crop"].changed_instances += 1
                    any_stats.changed_instances += 1
                    per_image_changed_type["Pad/Crop"] = True
                    per_image_changed_any = True
                    maybe_save_changed_pair("padcrop", vimgs, r)

        # ---------------- Scale ----------------------
        vimg_scale = [scale_image(im, scale_factor) for im in images]
        r, variant_base_score, prompt_variant, margin = compute_repr_for_image(vimg_scale)
        delta_base_score = (
            variant_base_score - base_label_logprob
            if variant_base_score is not None and base_label_logprob is not None
            else None
        )
        changed = r is not None and r != base_repr
        if pope_metrics and margin is not None:
            if changed:
                flip_margins.append(margin)
            else:
                noflip_margins.append(margin)
        freq_delta = record_freq_delta("Scale", vimg_scale[0], is_flip=bool(changed))
        record_dirichlet_delta(
            "Scale",
            vimg_scale[0],
            prompt_variant,
            freq_delta,
            is_flip=bool(changed),
            delta_base_score=delta_base_score,
        )
        stats["Scale"].total_instances += 1
        any_stats.total_instances += 1
        if r is not None:
            reprs_for_entropy.append(r)
            update_gt_confusion(stats["Scale"], r)
            if pope_base_yes:
                update_pope_base_confusion(
                    pope_yes_stats["Scale"],
                    pope_yes_any,
                    r,
                )
            if pope_base_no:
                update_pope_base_confusion(
                    pope_no_stats["Scale"],
                    pope_no_any,
                    r,
                )
            if changed:
                stats["Scale"].changed_instances += 1
                any_stats.changed_instances += 1
                per_image_changed_type["Scale"] = True
                per_image_changed_any = True
                maybe_save_changed_pair("scale", vimg_scale, r)

        # ---------------- Scale+Pad (black/white) ---
        for bg in ["black", "white"]:
            vimg_sp = [scale_and_pad(im, scale_factor, background=bg) for im in images]
            r, variant_base_score, prompt_variant, margin = compute_repr_for_image(vimg_sp)
            delta_base_score = (
                variant_base_score - base_label_logprob
                if variant_base_score is not None and base_label_logprob is not None
                else None
            )
            changed = r is not None and r != base_repr
            if pope_metrics and margin is not None:
                if changed:
                    flip_margins.append(margin)
                else:
                    noflip_margins.append(margin)
            freq_delta = record_freq_delta("Scale+Pad", vimg_sp[0], is_flip=bool(changed))
            record_dirichlet_delta(
                "Scale+Pad",
                vimg_sp[0],
                prompt_variant,
                freq_delta,
                is_flip=bool(changed),
                delta_base_score=delta_base_score,
            )
            stats["Scale+Pad"].total_instances += 1
            any_stats.total_instances += 1
            if r is not None:
                reprs_for_entropy.append(r)
                update_gt_confusion(stats["Scale+Pad"], r)
                if pope_base_yes:
                    update_pope_base_confusion(
                        pope_yes_stats["Scale+Pad"],
                        pope_yes_any,
                        r,
                    )
                if pope_base_no:
                    update_pope_base_confusion(
                        pope_no_stats["Scale+Pad"],
                        pope_no_any,
                        r,
                    )
                if changed:
                    stats["Scale+Pad"].changed_instances += 1
                    any_stats.changed_instances += 1
                    per_image_changed_type["Scale+Pad"] = True
                    per_image_changed_any = True
                    maybe_save_changed_pair(f"scale_pad_{bg}", vimg_sp, r)

        overlay_phrases = text_overlay_phrases(sample.options, base_repr)
        overlay_fonts = [overlay_font_for_image(im) for im in images]
        overlay_boxes_per_image = [
            [
                overlay_box_for_text(im, phrase, overlay_fonts[idx])
                for phrase in overlay_phrases
            ]
            for idx, im in enumerate(images)
        ]
        random_phrases = random_text_phrases(sample.index, count=len(overlay_phrases))

        # ---------------- TextOverlay ----------------
        for idx_phrase, phrase in enumerate(overlay_phrases):
            vimg_txt = [
                text_overlay_in_box(
                    im,
                    phrase,
                    overlay_boxes_per_image[img_idx][idx_phrase],
                    overlay_fonts[img_idx],
                )
                for img_idx, im in enumerate(images)
            ]
            r, variant_base_score, prompt_variant, margin = compute_repr_for_image(vimg_txt)
            delta_base_score = (
                variant_base_score - base_label_logprob
                if variant_base_score is not None and base_label_logprob is not None
                else None
            )
            changed = r is not None and r != base_repr
            if pope_metrics and margin is not None:
                if changed:
                    flip_margins.append(margin)
                else:
                    noflip_margins.append(margin)
            freq_delta = record_freq_delta("TextOverlay", vimg_txt[0], is_flip=bool(changed))
            record_dirichlet_delta(
                "TextOverlay",
                vimg_txt[0],
                prompt_variant,
                freq_delta,
                is_flip=bool(changed),
                delta_base_score=delta_base_score,
            )
            stats["TextOverlay"].total_instances += 1
            any_stats.total_instances += 1
            if r is not None:
                reprs_for_entropy.append(r)
                update_gt_confusion(stats["TextOverlay"], r)
                if pope_base_yes:
                    update_pope_base_confusion(
                        pope_yes_stats["TextOverlay"],
                        pope_yes_any,
                        r,
                    )
                if pope_base_no:
                    update_pope_base_confusion(
                        pope_no_stats["TextOverlay"],
                        pope_no_any,
                        r,
                    )
                if changed:
                    stats["TextOverlay"].changed_instances += 1
                    any_stats.changed_instances += 1
                    per_image_changed_type["TextOverlay"] = True
                    per_image_changed_any = True
                    maybe_save_changed_pair("text_overlay", vimg_txt, r)

        # ---------------- BoxOverlay -----------------
        for idx in range(len(overlay_phrases)):
            vimg_box = [
                box_overlay(im, [overlay_boxes_per_image[img_idx][idx]])
                for img_idx, im in enumerate(images)
            ]
            r, variant_base_score, prompt_variant, margin = compute_repr_for_image(vimg_box)
            delta_base_score = (
                variant_base_score - base_label_logprob
                if variant_base_score is not None and base_label_logprob is not None
                else None
            )
            changed = r is not None and r != base_repr
            if pope_metrics and margin is not None:
                if changed:
                    flip_margins.append(margin)
                else:
                    noflip_margins.append(margin)
            freq_delta = record_freq_delta("BoxOverlay", vimg_box[0], is_flip=bool(changed))
            record_dirichlet_delta(
                "BoxOverlay",
                vimg_box[0],
                prompt_variant,
                freq_delta,
                is_flip=bool(changed),
                delta_base_score=delta_base_score,
            )
            stats["BoxOverlay"].total_instances += 1
            any_stats.total_instances += 1
            if r is not None:
                reprs_for_entropy.append(r)
                update_gt_confusion(stats["BoxOverlay"], r)
                if pope_base_yes:
                    update_pope_base_confusion(
                        pope_yes_stats["BoxOverlay"],
                        pope_yes_any,
                        r,
                    )
                if pope_base_no:
                    update_pope_base_confusion(
                        pope_no_stats["BoxOverlay"],
                        pope_no_any,
                        r,
                    )
                if changed:
                    stats["BoxOverlay"].changed_instances += 1
                    any_stats.changed_instances += 1
                    per_image_changed_type["BoxOverlay"] = True
                    per_image_changed_any = True
                    maybe_save_changed_pair(f"box_overlay_{idx+1}", vimg_box, r)

        # ---------------- RandomText -----------------
        for idx, phrase in enumerate(random_phrases):
            if idx >= len(overlay_phrases):
                break
            vimg_rand = [
                text_overlay_in_box(
                    im,
                    phrase,
                    overlay_boxes_per_image[img_idx][idx],
                    overlay_fonts[img_idx],
                )
                for img_idx, im in enumerate(images)
            ]
            r, variant_base_score, prompt_variant, margin = compute_repr_for_image(vimg_rand)
            delta_base_score = (
                variant_base_score - base_label_logprob
                if variant_base_score is not None and base_label_logprob is not None
                else None
            )
            changed = r is not None and r != base_repr
            if pope_metrics and margin is not None:
                if changed:
                    flip_margins.append(margin)
                else:
                    noflip_margins.append(margin)
            freq_delta = record_freq_delta("RandomText", vimg_rand[0], is_flip=bool(changed))
            record_dirichlet_delta(
                "RandomText",
                vimg_rand[0],
                prompt_variant,
                freq_delta,
                is_flip=bool(changed),
                delta_base_score=delta_base_score,
            )
            stats["RandomText"].total_instances += 1
            any_stats.total_instances += 1
            if r is not None:
                reprs_for_entropy.append(r)
                update_gt_confusion(stats["RandomText"], r)
                if pope_base_yes:
                    update_pope_base_confusion(
                        pope_yes_stats["RandomText"],
                        pope_yes_any,
                        r,
                    )
                if pope_base_no:
                    update_pope_base_confusion(
                        pope_no_stats["RandomText"],
                        pope_no_any,
                        r,
                    )
                if changed:
                    stats["RandomText"].changed_instances += 1
                    any_stats.changed_instances += 1
                    per_image_changed_type["RandomText"] = True
                    per_image_changed_any = True
                    maybe_save_changed_pair(f"random_text_{idx+1}", vimg_rand, r)

        # ---------------- Rotation -------------------
        for angle in (-30.0, 30.0):
            vimg_rot = [rotate_image(im, angle) for im in images]
            r, variant_base_score, prompt_variant, margin = compute_repr_for_image(vimg_rot)
            delta_base_score = (
                variant_base_score - base_label_logprob
                if variant_base_score is not None and base_label_logprob is not None
                else None
            )
            changed = r is not None and r != base_repr
            if pope_metrics and margin is not None:
                if changed:
                    flip_margins.append(margin)
                else:
                    noflip_margins.append(margin)
            freq_delta = record_freq_delta("Rotation", vimg_rot[0], is_flip=bool(changed))
            record_dirichlet_delta(
                "Rotation",
                vimg_rot[0],
                prompt_variant,
                freq_delta,
                is_flip=bool(changed),
                delta_base_score=delta_base_score,
            )
            stats["Rotation"].total_instances += 1
            any_stats.total_instances += 1
            if r is not None:
                reprs_for_entropy.append(r)
                update_gt_confusion(stats["Rotation"], r)
                if pope_base_yes:
                    update_pope_base_confusion(
                        pope_yes_stats["Rotation"],
                        pope_yes_any,
                        r,
                    )
                if pope_base_no:
                    update_pope_base_confusion(
                        pope_no_stats["Rotation"],
                        pope_no_any,
                        r,
                    )
                if changed:
                    stats["Rotation"].changed_instances += 1
                    any_stats.changed_instances += 1
                    per_image_changed_type["Rotation"] = True
                    per_image_changed_any = True
                    maybe_save_changed_pair(f"rotation_{int(angle)}", vimg_rot, r)

        # ---------------- Blank (POPE) ---------------
        if pope_blank_enabled:
            vimg_blank = [Image.new("RGB", primary_img.size, color="white")]
            r, _, _, margin = compute_repr_for_image(vimg_blank)
            if r is not None:
                blank_pred = _pope_label_to_yesno(r, sample.options)
                blank_margin = margin

        # Update per-image affected counts
        for t in perturb_types:
            if per_image_changed_type[t]:
                stats[t].images_affected += 1
        if per_image_changed_any:
            any_stats.images_affected += 1
        else:
            # Drop freq/Dirichlet records for samples with no flips at all.
            if not keep_noflip_analysis:
                if freq_helper is not None:
                    freq_helper.drop_sample(sample.index)
                if dirichlet_helper is not None:
                    dirichlet_helper.drop_sample(sample.index)

        Hv = entropy_from_answers(reprs_for_entropy)
        changed_types = [t for t in perturb_types if per_image_changed_type[t]]
        print(f"[info] Repr changed for types: {changed_types or 'None'}")
        print(f"[info] Visual entropy H_v ({compare_mode}) = {Hv:.4f}")
        if pope_jsonl_active and pope_jsonl_file is not None:
            flip_avg = sum(flip_margins) / len(flip_margins) if flip_margins else None
            noflip_avg = sum(noflip_margins) / len(noflip_margins) if noflip_margins else None
            record = {
                "pope_gt": pope_gt,
                "pope_pred": pope_pred,
                "pope_margin_yesno": base_margin,
                "pertubations_flip_avg": flip_avg,
                "perubations_noflip_avg": noflip_avg,
            }
            if pope_blank_enabled:
                record["pope_blank_pred"] = blank_pred
                record["pope_blank_margin_yesno"] = blank_margin
            pope_jsonl_file.write(json.dumps(record) + "\n")

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
    if pope_metrics:
        tp = pope_counts["tp"]
        fp = pope_counts["fp"]
        tn = pope_counts["tn"]
        fn = pope_counts["fn"]
        total = tp + fp + tn + fn
        acc = safe_div(tp + tn, total)
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        fpr = safe_div(fp, fp + tn)
        yes_rate = safe_div(pope_counts["pred_yes"], pope_counts["pred_total"])
        summary_lines.append("POPE metrics (Yes=positive):")
        summary_lines.append(
            "Accuracy="
            f"{acc:.3f} Precision={prec:.3f} Recall={rec:.3f} "
            f"F1={f1:.3f} FPR={fpr:.3f} Yes-rate={yes_rate:.3f}"
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
    if not analysis_only:
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
        if pope_metrics:
            summary_lines.append("\nConfusion vs ground truth (perturbation instances) [base pred=Yes, gt=Yes]")
            summary_lines.append(conf_header)
            summary_lines.append("-" * len(conf_header))
            for t in perturb_types:
                st = pope_yes_stats[t]
                summary_lines.append(
                    f"{t:<12}  {st.right_to_wrong:7d}  {st.wrong_to_right:7d}  "
                    f"{st.right_to_right:7d}  {st.wrong_to_wrong:7d}  {st.gt_evaluable:9d}"
                )
            summary_lines.append(
                f"{'Any':<12}  {pope_yes_any.right_to_wrong:7d}  {pope_yes_any.wrong_to_right:7d}  "
                f"{pope_yes_any.right_to_right:7d}  {pope_yes_any.wrong_to_wrong:7d}  {pope_yes_any.gt_evaluable:9d}"
            )
            summary_lines.append("\nConfusion vs ground truth (perturbation instances) [base pred=No, gt=No]")
            summary_lines.append(conf_header)
            summary_lines.append("-" * len(conf_header))
            for t in perturb_types:
                st = pope_no_stats[t]
                summary_lines.append(
                    f"{t:<12}  {st.right_to_wrong:7d}  {st.wrong_to_right:7d}  "
                    f"{st.right_to_right:7d}  {st.wrong_to_wrong:7d}  {st.gt_evaluable:9d}"
                )
            summary_lines.append(
                f"{'Any':<12}  {pope_no_any.right_to_wrong:7d}  {pope_no_any.wrong_to_right:7d}  "
                f"{pope_no_any.right_to_right:7d}  {pope_no_any.wrong_to_wrong:7d}  {pope_no_any.gt_evaluable:9d}"
            )

    if freq_helper is not None:
        summary_lines.extend(freq_helper.summary_lines())
    if dirichlet_helper is not None:
        summary_lines.extend(dirichlet_helper.summary_lines())

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
        if analysis_only:
            append_line(summary_path, f"{progress_prefix}completed freq/dirichlet analysis on {N_used}/{N_total} samples")
        else:
            append_line(summary_path, f"{progress_prefix}completed label invariance on {N_used}/{N_total} samples")
    if pope_jsonl_file is not None:
        pope_jsonl_file.close()


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
        "BoxOverlay",
        "RandomText",
        "Rotation",
    ]
    stats: Dict[str, EmbeddingStats] = {t: EmbeddingStats() for t in perturb_types}
    any_stats = EmbeddingStats()
    context_len_cache: Dict[tuple, int] = {}
    viz_collector = EmbeddingVizCollector(viz_config)

    N_total = len(samples)
    print(f"[info] Running embedding invariance analysis on {N_total} samples...")
    if summary_path is not None:
        append_line(summary_path, f"{progress_prefix}start embedding invariance on {N_total} samples")

    for i, sample in enumerate(samples, start=1):
        images = load_sample_images(sample)
        if not images:
            continue
        primary_img = images[0]

        def build_prompt_open(imgs: List[Image.Image]) -> UserPrompt:
            return build_user_prompt(sample, imgs, "open")

        def build_prompt_mcq(imgs: List[Image.Image]) -> UserPrompt:
            return build_user_prompt(sample, imgs, "mcq")

        prompt_open = build_prompt_open(images)
        prompt_mcq = build_prompt_mcq(images)

        base_ctx_open = get_context_embedding(
            model,
            processor,
            primary_img,
            prompt_open.text,
            device=device,
            context_len_cache=context_len_cache,
            user_prompt=prompt_open,
        )
        base_ctx_mcq = get_context_embedding(
            model,
            processor,
            primary_img,
            prompt_mcq.text,
            device=device,
            context_len_cache=context_len_cache,
            user_prompt=prompt_mcq,
        )

        base_open_answer = generate_free_answer(
            model,
            processor,
            primary_img,
            prompt_open.text,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            user_prompt=prompt_open,
        )
        base_ans_emb_open = get_answer_embedding(
            model,
            processor,
            primary_img,
            prompt_open.text,
            base_open_answer,
            device=device,
            context_len_cache=context_len_cache,
            user_prompt=prompt_open,
        )

        base_mcq_label = choose_label_via_loglik(
            model,
            processor,
            primary_img,
            prompt_mcq.text,
            sample.options,
            device=device,
            context_len_cache=context_len_cache,
            user_prompt=prompt_mcq,
        )
        base_mcq_answer_text = (
            sample.options.get(base_mcq_label) if base_mcq_label else None
        )
        base_mcq_free_answer = generate_free_answer(
            model,
            processor,
            primary_img,
            prompt_mcq.text,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            user_prompt=prompt_mcq,
        )
        base_ans_emb_mcq = (
            get_answer_embedding(
                model,
                processor,
                primary_img,
                prompt_mcq.text,
                base_mcq_answer_text,
                device=device,
                context_len_cache=context_len_cache,
                user_prompt=prompt_mcq,
            )
            if base_mcq_answer_text
            else None
        )
        base_ans_emb_mcq_free = get_answer_embedding(
            model,
            processor,
            primary_img,
            prompt_mcq.text,
            base_mcq_free_answer,
            device=device,
            context_len_cache=context_len_cache,
            user_prompt=prompt_mcq,
        )

        def process_variant_embeddings(
            vimgs: List[Image.Image], pert_type: str, variant: str
        ) -> None:
            prompt_open_var = build_prompt_open(vimgs)
            prompt_mcq_var = build_prompt_mcq(vimgs)
            ctx_open = get_context_embedding(
                model,
                processor,
                vimgs[0],
                prompt_open_var.text,
                device=device,
                context_len_cache=context_len_cache,
                user_prompt=prompt_open_var,
            )
            ctx_mcq = get_context_embedding(
                model,
                processor,
                vimgs[0],
                prompt_mcq_var.text,
                device=device,
                context_len_cache=context_len_cache,
                user_prompt=prompt_mcq_var,
            )
            _record_similarity(
                stats[pert_type].context_open, any_stats.context_open, base_ctx_open, ctx_open
            )
            _record_similarity(
                stats[pert_type].context_mcq, any_stats.context_mcq, base_ctx_mcq, ctx_mcq
            )

            ans_open = generate_free_answer(
                model,
                processor,
                vimgs[0],
                prompt_open_var.text,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                user_prompt=prompt_open_var,
            )
            ans_emb_open = get_answer_embedding(
                model,
                processor,
                vimgs[0],
                prompt_open_var.text,
                ans_open,
                device=device,
                context_len_cache=context_len_cache,
                user_prompt=prompt_open_var,
            )
            _record_similarity(
                stats[pert_type].answer_open, any_stats.answer_open, base_ans_emb_open, ans_emb_open
            )

            mcq_free_answer = generate_free_answer(
                model,
                processor,
                vimgs[0],
                prompt_mcq_var.text,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                user_prompt=prompt_mcq_var,
            )
            mcq_free_emb = get_answer_embedding(
                model,
                processor,
                vimgs[0],
                prompt_mcq_var.text,
                mcq_free_answer,
                device=device,
                context_len_cache=context_len_cache,
                user_prompt=prompt_mcq_var,
            )
            _record_similarity(
                stats[pert_type].answer_mcq_free,
                any_stats.answer_mcq_free,
                base_ans_emb_mcq_free,
                mcq_free_emb,
            )

            pert_mcq_label = choose_label_via_loglik(
                model,
                processor,
                vimgs[0],
                prompt_mcq_var.text,
                sample.options,
                device=device,
                context_len_cache=context_len_cache,
                user_prompt=prompt_mcq_var,
            )
            pert_mcq_text = sample.options.get(pert_mcq_label) if pert_mcq_label else None
            ans_emb_mcq = None
            if pert_mcq_text:
                ans_emb_mcq = get_answer_embedding(
                    model,
                    processor,
                    vimgs[0],
                    prompt_mcq_var.text,
                    pert_mcq_text,
                    device=device,
                    context_len_cache=context_len_cache,
                    user_prompt=prompt_mcq_var,
                )
                _record_similarity(
                    stats[pert_type].answer_mcq,
                    any_stats.answer_mcq,
                    base_ans_emb_mcq,
                    ans_emb_mcq,
                )
            if viz_collector.enabled():
                viz_collector.add(
                    "ctx_open",
                    sample.index,
                    pert_type,
                    variant,
                    ctx_open,
                )
                viz_collector.add(
                    "ctx_mcq",
                    sample.index,
                    pert_type,
                    variant,
                    ctx_mcq,
                )
                viz_collector.add(
                    "ans_open",
                    sample.index,
                    pert_type,
                    variant,
                    ans_emb_open,
                )
                viz_collector.add(
                    "ans_mcq",
                    sample.index,
                    pert_type,
                    variant,
                    ans_emb_mcq if pert_mcq_text else None,
                )
                viz_collector.add(
                    "ans_mcq_free",
                    sample.index,
                    pert_type,
                    variant,
                    mcq_free_emb,
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
            vimgs = [cyclic_horizontal_shift(im, n) for im in images]
            process_variant_embeddings(vimgs, "Translation", f"shift_{n}")

        # Pad/Crop
        for n in padcrop_pixels:
            if n == 0:
                continue
            vimgs = [pad_or_crop(im, n) for im in images]
            process_variant_embeddings(vimgs, "Pad/Crop", f"padcrop_{n}")

        # Scale
        vimg_scale = [scale_image(im, scale_factor) for im in images]
        process_variant_embeddings(vimg_scale, "Scale", f"scale_{scale_factor}")

        # Scale+Pad
        for bg in ["black", "white"]:
            vimg_sp = [scale_and_pad(im, scale_factor, background=bg) for im in images]
            process_variant_embeddings(vimg_sp, "Scale+Pad", f"scale_pad_{bg}")

        overlay_phrases = text_overlay_phrases(sample.options, base_mcq_label)
        overlay_fonts = [overlay_font_for_image(im) for im in images]
        overlay_boxes_per_image = [
            [
                overlay_box_for_text(im, phrase, overlay_fonts[idx])
                for phrase in overlay_phrases
            ]
            for idx, im in enumerate(images)
        ]
        random_phrases = random_text_phrases(sample.index, count=len(overlay_phrases))

        for idx_phrase, phrase in enumerate(overlay_phrases):
            vimg_txt = [
                text_overlay_in_box(
                    im,
                    phrase,
                    overlay_boxes_per_image[img_idx][idx_phrase],
                    overlay_fonts[img_idx],
                )
                for img_idx, im in enumerate(images)
            ]
            process_variant_embeddings(vimg_txt, "TextOverlay", f"text_overlay_{idx_phrase+1}")

        for idx in range(len(overlay_phrases)):
            vimg_box = [
                box_overlay(im, [overlay_boxes_per_image[img_idx][idx]])
                for img_idx, im in enumerate(images)
            ]
            process_variant_embeddings(vimg_box, "BoxOverlay", f"box_{idx+1}")

        for idx, phrase in enumerate(random_phrases):
            if idx >= len(overlay_phrases):
                break
            vimg_txt = [
                text_overlay_in_box(
                    im,
                    phrase,
                    overlay_boxes_per_image[img_idx][idx],
                    overlay_fonts[img_idx],
                )
                for img_idx, im in enumerate(images)
            ]
            process_variant_embeddings(vimg_txt, "RandomText", f"random_text_{idx+1}")

        # Rotation
        for angle in (-30.0, 30.0):
            vimg_rot = [rotate_image(im, angle) for im in images]
            process_variant_embeddings(vimg_rot, "Rotation", f"rot_{int(angle)}")

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
        "BoxOverlay",
        "RandomText",
        "Rotation",
    ]

    candidates = [s for s in samples if s.image_path.exists()]
    rng.shuffle(candidates)
    selected = candidates[: min(sample_limit, len(candidates))]
    print(f"[drift] Selected {len(selected)} samples (limit={sample_limit})")

    context_len_cache: Dict[tuple, int] = {}
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
        images = load_sample_images(sample)
        if not images:
            continue
        primary_img = images[0]

        def build_prompt_open(imgs: List[Image.Image]) -> UserPrompt:
            return build_user_prompt(sample, imgs, "open")

        def build_prompt_mcq(imgs: List[Image.Image]) -> UserPrompt:
            return build_user_prompt(sample, imgs, "mcq")

        prompt_open = build_prompt_open(images)
        prompt_mcq = build_prompt_mcq(images)

        base_ctx_open = get_context_embedding(
            model,
            processor,
            primary_img,
            prompt_open.text,
            device=device,
            context_len_cache=context_len_cache,
            user_prompt=prompt_open,
        )
        base_ctx_mcq = get_context_embedding(
            model,
            processor,
            primary_img,
            prompt_mcq.text,
            device=device,
            context_len_cache=context_len_cache,
            user_prompt=prompt_mcq,
        )

        base_open_answer = generate_free_answer(
            model,
            processor,
            primary_img,
            prompt_open.text,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            user_prompt=prompt_open,
        )
        base_ans_emb_open = get_answer_embedding(
            model,
            processor,
            primary_img,
            prompt_open.text,
            base_open_answer,
            device=device,
            context_len_cache=context_len_cache,
            user_prompt=prompt_open,
        )

        base_mcq_label = choose_label_via_loglik(
            model,
            processor,
            primary_img,
            prompt_mcq.text,
            sample.options,
            device=device,
            context_len_cache=context_len_cache,
            user_prompt=prompt_mcq,
        )
        base_mcq_answer_text = (
            sample.options.get(base_mcq_label) if base_mcq_label else None
        )
        base_mcq_free_answer = generate_free_answer(
            model,
            processor,
            primary_img,
            prompt_mcq.text,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            user_prompt=prompt_mcq,
        )
        base_ans_emb_mcq = (
            get_answer_embedding(
                model,
                processor,
                primary_img,
                prompt_mcq.text,
                base_mcq_answer_text,
                device=device,
                context_len_cache=context_len_cache,
                user_prompt=prompt_mcq,
            )
            if base_mcq_answer_text
            else None
        )
        base_ans_emb_mcq_free = get_answer_embedding(
            model,
            processor,
            primary_img,
            prompt_mcq.text,
            base_mcq_free_answer,
            device=device,
            context_len_cache=context_len_cache,
            user_prompt=prompt_mcq,
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

        def _record_variant(ptype: str, vimgs: List[Image.Image]) -> None:
            prompt_open_var = build_prompt_open(vimgs)
            prompt_mcq_var = build_prompt_mcq(vimgs)
            ctx_open = get_context_embedding(
                model,
                processor,
                vimgs[0],
                prompt_open_var.text,
                device=device,
                context_len_cache=context_len_cache,
                user_prompt=prompt_open_var,
            )
            ctx_mcq = get_context_embedding(
                model,
                processor,
                vimgs[0],
                prompt_mcq_var.text,
                device=device,
                context_len_cache=context_len_cache,
                user_prompt=prompt_mcq_var,
            )
            ans_open = generate_free_answer(
                model,
                processor,
                vimgs[0],
                prompt_open_var.text,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                user_prompt=prompt_open_var,
            )
            ans_emb_open = get_answer_embedding(
                model,
                processor,
                vimgs[0],
                prompt_open_var.text,
                ans_open,
                device=device,
                context_len_cache=context_len_cache,
                user_prompt=prompt_open_var,
            )
            mcq_free_answer = generate_free_answer(
                model,
                processor,
                vimgs[0],
                prompt_mcq_var.text,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                user_prompt=prompt_mcq_var,
            )
            ans_emb_mcq_free = get_answer_embedding(
                model,
                processor,
                vimgs[0],
                prompt_mcq_var.text,
                mcq_free_answer,
                device=device,
                context_len_cache=context_len_cache,
                user_prompt=prompt_mcq_var,
            )
            pert_mcq_label = choose_label_via_loglik(
                model,
                processor,
                vimgs[0],
                prompt_mcq_var.text,
                sample.options,
                device=device,
                context_len_cache=context_len_cache,
                user_prompt=prompt_mcq_var,
            )
            pert_mcq_text = sample.options.get(pert_mcq_label) if pert_mcq_label else None
            ans_emb_mcq = (
                get_answer_embedding(
                    model,
                    processor,
                    vimgs[0],
                    prompt_mcq_var.text,
                    pert_mcq_text,
                    device=device,
                    context_len_cache=context_len_cache,
                    user_prompt=prompt_mcq_var,
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
            vimgs = [cyclic_horizontal_shift(im, n) for im in images]
            _record_variant("Translation", vimgs)

        for n in padcrop_pixels:
            if n == 0:
                continue
            vimgs = [pad_or_crop(im, n) for im in images]
            _record_variant("Pad/Crop", vimgs)

        vimg_scale = [scale_image(im, scale_factor) for im in images]
        _record_variant("Scale", vimg_scale)

        for bg in ["black", "white"]:
            vimg_sp = [scale_and_pad(im, scale_factor, background=bg) for im in images]
            _record_variant("Scale+Pad", vimg_sp)

        overlay_phrases = text_overlay_phrases(sample.options, base_mcq_label)
        overlay_fonts = [overlay_font_for_image(im) for im in images]
        overlay_boxes_per_image = [
            [
                overlay_box_for_text(im, phrase, overlay_fonts[idx])
                for phrase in overlay_phrases
            ]
            for idx, im in enumerate(images)
        ]
        random_phrases = random_text_phrases(sample.index, count=len(overlay_phrases))

        for idx, phrase in enumerate(overlay_phrases):
            vimg_txt = [
                text_overlay_in_box(
                    im,
                    phrase,
                    overlay_boxes_per_image[img_idx][idx],
                    overlay_fonts[img_idx],
                )
                for img_idx, im in enumerate(images)
            ]
            _record_variant("TextOverlay", vimg_txt)

        for idx in range(len(overlay_phrases)):
            vimg_box = [
                box_overlay(im, [overlay_boxes_per_image[img_idx][idx]])
                for img_idx, im in enumerate(images)
            ]
            _record_variant("BoxOverlay", vimg_box)

        for idx, phrase in enumerate(random_phrases):
            if idx >= len(overlay_phrases):
                break
            vimg_txt = [
                text_overlay_in_box(
                    im,
                    phrase,
                    overlay_boxes_per_image[img_idx][idx],
                    overlay_fonts[img_idx],
                )
                for img_idx, im in enumerate(images)
            ]
            _record_variant("RandomText", vimg_txt)

        for angle in (-30.0, 30.0):
            vimg_rot = [rotate_image(im, angle) for im in images]
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
    "device": "cuda",
    "num_workers": 0,
    "gpus_per_worker": 0,
    "seed": 0,
    "deterministic": False,
    "max_samples": 20,
    "compare_mode": "label",
    "invariance_mode": "label",  # run both label + embedding by default
    "save_changed_dir": "changed_predictions",
    "save_changed_limit": 2,
    "summary_file": "summary.txt",
    "pope_jsonl": "pope_predictions.jsonl",
    "pope_split": DEFAULT_POPE_SPLIT,
    "pope_no_image_baseline": True,
    "append_model_size": True,
    "freq_dirichlet_only": False,
    "freq_analysis": True,
    "freq_plots": True,
    "freq_band_scheme": "log",
    "freq_per_channel": True,
    "dirichlet_analysis": True,
    "dirichlet_plots": True,
    "keep_noflip_analysis": True,
    "analysis_outdir": "logs",
    # Embedding invariance
    "embedding_analysis": False,
    # Embedding viz
    "embedding_viz": False,
    "embedding_viz_limit": 4,
    "embedding_viz_dir": "embedding_viz",
    "embedding_viz_perplexity": 30.0,
    # Embedding drift
    "embedding_drift_analysis": False,
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
    p.add_argument( "--append-model-size", action=argparse.BooleanOptionalAction, default=d["append_model_size"], help="Append model size tag (e.g., 7B) to output paths (summary, logs, saved images)." )
    
    p.add_argument( "--device", default=d["device"], help="Computation device: auto | cpu | mps | cuda | cuda:<idx> (auto picks best available).")
    p.add_argument( "--num-workers", type=int, default=d["num_workers"], help="Worker limit. 0 means single worker; >0 caps workers to this number and available devices/CPU slots.")
    p.add_argument( "--gpus-per-worker", type=int, default=d["gpus_per_worker"], help="GPUs to expose to each worker for tensor parallelism. 0=auto (single-worker uses all available GPUs; multi-worker splits available GPUs as evenly as possible).")
    p.add_argument( "--seed", type=int, default=d["seed"], help="Random seed for Python/NumPy/Torch.")
    p.add_argument( "--deterministic", action="store_true", default=d["deterministic"], help="Enable deterministic torch/CUDA kernels (slower).")
    p.add_argument( "--dataset", type=str, default="pope", choices=["seedbench", "mmmu", "pope"], help="Dataset adapter to use (seedbench, mmmu, pope).")
    p.add_argument( "--mmmu-split", type=str, default="validation", choices=["validation", "dev", "test"], help="MMMU split to use.")
    p.add_argument( "--pope-split", type=str, default=d["pope_split"], choices=["adversarial", "popular", "random", "test"], help="POPE split to use.")
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
    p.add_argument( "--pope-jsonl", type=str, default=d["pope_jsonl"], help="Path to write POPE per-sample JSONL (set empty to disable)." )
    p.add_argument( "--freq-dirichlet-only", action="store_true", default=d["freq_dirichlet_only"], help="Skip embedding analyses/drift and run only frequency + Dirichlet analyses (label scoring still runs to drive perturbations)." )
    p.add_argument( "--freq-analysis", action="store_true", default=d["freq_analysis"], help="Run optional frequency-domain analysis (radial band energy deltas) for each perturbation." )
    p.add_argument( "--freq-plots", action="store_true", default=d["freq_plots"], help="Save bar charts of frequency band deltas (absolute and relative, overall and flips-only)." )
    p.add_argument( "--freq-band-scheme", type=str, default=d["freq_band_scheme"], choices=["equal", "log", "percentile"], help="Band edge scheme for radial FFT: equal thirds of radius, logarithmic spacing, or radius percentiles." )
    p.add_argument( "--freq-per-channel", action="store_true", default=d["freq_per_channel"], help="Compute frequency bands per RGB channel and average (instead of grayscale mean)." )
    p.add_argument( "--dirichlet-analysis", action="store_true", default=d["dirichlet_analysis"], help="Track Dirichlet energy deltas over vision tokens for each perturbation." )
    p.add_argument( "--dirichlet-plots", action="store_true", default=d["dirichlet_plots"], help="Save hist/KDE plots for Dirichlet energy deltas (requires --dirichlet-analysis)." )
    p.add_argument( "--analysis-outdir", type=str, default=d["analysis_outdir"], help="Output directory for frequency/Dirichlet auxiliary plots." )
    p.add_argument( "--keep-noflip-analysis", action="store_true", default=d["keep_noflip_analysis"], help="Retain freq/Dirichlet records for samples where no perturbation flips the answer (default is to drop them)." )
    p.add_argument( "--pope-no-image-baseline", action="store_true", default=d["pope_no_image_baseline"], help="Include a blank-image baseline as an extra perturbation for POPE." )
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
    p.add_argument( "--loglik-batch-size", type=int, default=0, help="Options batch size per forward pass (<=0 to batch all options together)." )
    
    return p.parse_args()


def load_samples_for_dataset(
    args: argparse.Namespace, cache_dir: Path
) -> tuple[List[Sample], int, Path, Path]:
    """
    Adapter entrypoint for loading samples from a dataset. Currently supports
    SEEDBench, MMMU, and POPE; extend this function to plug in other datasets.
    """
    dataset = args.dataset.lower()
    if dataset == "seedbench":
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
    if dataset == "mmmu":
        split = getattr(args, "mmmu_split", "validation")
        data_dir = _mmmu_data_dir(args, cache_dir)
        samples, total_rows, source_path, image_root = load_mmmu_samples(
            split=split,
            cache_dir=cache_dir,
            data_dir=data_dir,
            max_samples=args.max_samples,
            offline=args.offline,
        )
        return samples, total_rows, source_path, image_root
    if dataset == "pope":
        split = getattr(args, "pope_split", DEFAULT_POPE_SPLIT)
        samples, total_rows, source_path, image_root = load_pope_samples(
            split=split,
            cache_dir=cache_dir,
            max_samples=args.max_samples,
            offline=args.offline,
        )
        return samples, total_rows, source_path, image_root

    raise ValueError(f"Dataset adapter not implemented: {args.dataset}")


def main() -> None:
    args = parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    ds_suffix = dataset_suffix_tag(args)
    size_tag = model_size_tag(args.model_path or args.model_id) if args.append_model_size else ""
    suffix_tag = ds_suffix if ds_suffix else size_tag
    if size_tag and ds_suffix:
        suffix_tag = f"{ds_suffix}_{size_tag}"
    set_option_batch_chunk_size(args.loglik_batch_size)

    def _suffix_inplace(attr: str, is_file: bool) -> None:
        val = getattr(args, attr, None)
        if val:
            new_path = path_with_dataset_suffix(Path(val).expanduser(), suffix_tag, is_file)
            setattr(args, attr, str(new_path))

    _suffix_inplace("save_changed_dir", is_file=False)
    _suffix_inplace("summary_file", is_file=True)
    _suffix_inplace("pope_jsonl", is_file=True)
    _suffix_inplace("analysis_outdir", is_file=False)
    _suffix_inplace("embedding_viz_dir", is_file=False)
    _suffix_inplace("embedding_drift_outdir", is_file=False)

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    seed_everything(args.seed, args.deterministic)

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
    pope_jsonl_path = None
    if args.dataset.lower() == "pope" and args.pope_jsonl:
        pope_jsonl_path = Path(args.pope_jsonl).expanduser().resolve()
    viz_config = EmbeddingVizConfig(
        enabled=args.embedding_viz,
        limit_samples=max(1, args.embedding_viz_limit),
        out_dir=Path(args.embedding_viz_dir).expanduser().resolve(),
        perplexity=args.embedding_viz_perplexity,
    )
    run_embedding_drift = args.embedding_drift_analysis
    progress_prefix = ""
    analysis_outdir = None
    if (args.freq_analysis or args.dirichlet_analysis) and args.analysis_outdir:
        analysis_outdir = Path(args.analysis_outdir).expanduser().resolve()

    # Determine which analyses to run
    run_label_invariance = args.invariance_mode in ("label", "both")
    run_embedding_invariance = args.embedding_analysis or args.invariance_mode in ("embedding", "both")
    if args.freq_dirichlet_only:
        run_embedding_invariance = False
        run_embedding_drift = False
        run_label_invariance = True
        if not (args.freq_analysis or args.dirichlet_analysis):
            print("[warn] --freq-dirichlet-only set but neither --freq-analysis nor --dirichlet-analysis enabled; enabling both.", file=sys.stderr)
            args.freq_analysis = True
            args.dirichlet_analysis = True
    if args.drift_only:
        run_label_invariance = False
        run_embedding_invariance = False
        run_embedding_drift = True

    # ------------------------------------------------------------------
    # Parallel execution across all available devices
    # ------------------------------------------------------------------
    requested_workers = args.num_workers
    devices = list_available_devices(args.device, None)

    device_groups = build_device_groups(devices, requested_workers, args.gpus_per_worker)
    if not device_groups:
        raise RuntimeError("No devices available for execution.")

    print(f"[info] Selected device groups for execution: {device_groups}")

    if len(device_groups) == 1:
        device_group = device_groups[0]
        device, device_map = configure_visible_devices(device_group)
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
            device_map=device_map,
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
                freq_analysis=args.freq_analysis,
                freq_plots=args.freq_plots,
                freq_band_scheme=args.freq_band_scheme,
                freq_per_channel=args.freq_per_channel,
                dirichlet_analysis=args.dirichlet_analysis,
                analysis_outdir=analysis_outdir,
                dirichlet_plots=args.dirichlet_plots,
                analysis_only=args.freq_dirichlet_only,
                keep_noflip_analysis=getattr(args, "keep_noflip_analysis", False),
                pope_jsonl_path=pope_jsonl_path,
                pope_no_image_baseline=getattr(args, "pope_no_image_baseline", False),
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

    # Multi-device: split samples evenly and spawn one worker per device group.
    if run_embedding_drift:
        print("[warn] Embedding drift analysis is single-device only; skipping in multi-worker mode.", file=sys.stderr)
        run_embedding_drift = False
    import math
    import multiprocessing as mp

    world_size = len(device_groups)
    chunk = math.ceil(len(samples) / float(world_size))
    sample_splits = [samples[i : i + chunk] for i in range(0, len(samples), chunk)]

    print(f"[info] Launching {world_size} parallel workers across device groups: {device_groups}")
    ctx = mp.get_context("spawn")
    procs = []
    for idx, (device_group, worker_samples) in enumerate(zip(device_groups, sample_splits)):
        p = ctx.Process(
            target=worker_process,
            args=(
                idx,
                device_group,
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

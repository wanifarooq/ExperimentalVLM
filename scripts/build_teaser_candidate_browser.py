#!/usr/bin/env python3
"""Build a visual browser of teaser candidates from an existing Exp 1 run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from PIL import Image, ImageDraw, ImageFont


LEVELS = ["L1_COARSE", "L2_MEDIUM", "L3_FINE", "L4_VERY_FINE"]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _image_path(cache_dir: Path, image_id: str) -> Path:
    for suffix in [".jpg", ".png", ".jpeg"]:
        candidate = cache_dir / "gqa" / "images" / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing source image for {image_id} under {cache_dir / 'gqa' / 'images'}")


def _score_candidate(record: Dict[str, Any]) -> float:
    levels = record.get("levels") or {}
    primary_clean = sum(1 for level in LEVELS if levels.get(level, {}).get("clean", {}).get("correct"))
    l4_drops = _drop_count(levels.get("L4_VERY_FINE", {}))
    l3_drops = _drop_count(levels.get("L3_FINE", {}))
    l2_drops = _drop_count(levels.get("L2_MEDIUM", {}))
    l1_drops = _drop_count(levels.get("L1_COARSE", {}))
    l4_vol = _mean_volatility(levels.get("L4_VERY_FINE", {}))
    questions = " ".join(str(levels.get(level, {}).get("question", "")).lower() for level in LEVELS)
    interpretable_bonus = 0.0
    for word in ["tree", "building", "bus", "dog", "cat", "boat", "sign", "shirt", "car", "train", "airplane", "table", "flower", "leaves"]:
        if word in questions:
            interpretable_bonus += 8.0
    odd_penalty = 0.0
    for word in ["mouth", "hair", "feet", "ear", "eye", "neck", "toilet paper", "foil"]:
        if word in questions:
            odd_penalty += 12.0
    return (
        primary_clean * 40.0
        + l4_drops * 8.0
        + l3_drops * 3.0
        + l2_drops * 1.5
        + l1_drops
        + l4_vol * 8.0
        + interpretable_bonus
        - odd_penalty
    )


def _drop_count(level_data: Dict[str, Any]) -> int:
    return sum(1 for item in level_data.get("perturbations", []) if _safe_float(item.get("accuracy_drop")) > 0)


def _mean_volatility(level_data: Dict[str, Any]) -> float:
    vals = [abs(_safe_float(item.get("loglik_drift"))) for item in level_data.get("perturbations", [])]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _summarize_record(record: Dict[str, Any], rank: int, score: float) -> Dict[str, Any]:
    levels = record.get("levels") or {}
    row: Dict[str, Any] = {
        "rank": rank,
        "score": round(score, 4),
        "image_id": str(record.get("image_id")),
    }
    for level in LEVELS:
        level_data = levels.get(level, {})
        row[f"{level}_question"] = level_data.get("question", "")
        row[f"{level}_answer_label"] = level_data.get("answer_label", "")
        row[f"{level}_clean_predicted"] = level_data.get("clean", {}).get("predicted", "")
        row[f"{level}_clean_correct"] = bool(level_data.get("clean", {}).get("correct", False))
        row[f"{level}_drops"] = _drop_count(level_data)
        row[f"{level}_mean_loglik_volatility"] = round(_mean_volatility(level_data), 4)
    return row


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_card(draw: ImageDraw.ImageDraw, sheet: Image.Image, record: Dict[str, Any], image_path: Path, x: int, y: int, w: int, h: int) -> None:
    levels = record.get("levels") or {}
    image_id = str(record.get("image_id"))
    title_font = _load_font(18, bold=True)
    text_font = _load_font(13)
    small_font = _load_font(12)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((w - 20, 170))
        sheet.paste(img, (x + 10, y + 10))

    yy = y + 190
    drops = [_drop_count(levels.get(level, {})) for level in LEVELS]
    clean = ["Y" if levels.get(level, {}).get("clean", {}).get("correct") else "N" for level in LEVELS]
    draw.text(
        (x + 10, yy),
        f"{image_id} | drops L1-L4: {'/'.join(map(str, drops))} | clean: {'/'.join(clean)}",
        fill="black",
        font=title_font,
    )
    yy += 27
    for level in LEVELS:
        question = str(levels.get(level, {}).get("question", ""))
        prefix = re.sub(r"_.+$", "", level)
        for idx, line in enumerate(textwrap.wrap(f"{prefix}: {question}", width=58)[:2]):
            draw.text((x + 12, yy), line, fill="#111827", font=text_font if idx == 0 else small_font)
            yy += 16
        yy += 2


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# Teaser Candidate Index",
        "",
        "| Rank | Image | Score | L1/L2/L3/L4 Drops | L1-L4 Questions |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        drops = "/".join(str(row[f"{level}_drops"]) for level in LEVELS)
        questions = "<br>".join(str(row[f"{level}_question"]).replace("|", "\\|") for level in LEVELS)
        lines.append(f"| {row['rank']} | {row['image_id']} | {row['score']} | {drops} | {questions} |")
    path.write_text("\n".join(lines) + "\n")


def build_browser(run_dir: Path, out_dir: Path, cache_dir: Path, cols: int, rows_per_sheet: int) -> Path:
    records = list(_iter_jsonl(run_dir / "exp1" / "per_sample.jsonl"))
    ranked = sorted(((record, _score_candidate(record)) for record in records), key=lambda item: item[1], reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [_summarize_record(record, rank, score) for rank, (record, score) in enumerate(ranked, start=1)]
    _write_csv(out_dir / "teaser_candidate_index.csv", summary_rows)
    _write_markdown(out_dir / "teaser_candidate_index.md", summary_rows)

    card_w, card_h = 500, 365
    per_sheet = cols * rows_per_sheet
    sheet_paths: List[str] = []
    for sheet_idx in range(math.ceil(len(ranked) / per_sheet)):
        subset = ranked[sheet_idx * per_sheet : (sheet_idx + 1) * per_sheet]
        sheet = Image.new("RGB", (cols * card_w, rows_per_sheet * card_h), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, (record, _score) in enumerate(subset):
            image_id = str(record.get("image_id"))
            x = (idx % cols) * card_w
            y = (idx // cols) * card_h
            _draw_card(draw, sheet, record, _image_path(cache_dir, image_id), x, y, card_w, card_h)
        path = out_dir / f"teaser_candidate_sheet_{sheet_idx + 1:02d}.png"
        sheet.save(path)
        sheet_paths.append(path.name)

    manifest = {
        "source_run": str(run_dir),
        "num_candidates": len(ranked),
        "index_csv": "teaser_candidate_index.csv",
        "index_md": "teaser_candidate_index.md",
        "sheets": sheet_paths,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--rows-per-sheet", type=int, default=5)
    args = parser.parse_args()
    out_dir = args.out_dir or (args.run_dir / "teaser_candidate_browser")
    print(build_browser(args.run_dir, out_dir, args.cache_dir, args.cols, args.rows_per_sheet))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.continuous import (
    attach_complexity_residual,
    summarize_multivariate_regression,
)
from ..analysis.spectral import spectral_band_centers
from ..analysis.statistics import paired_wordy_mirror_ttests
from ..data.base import ALL_VQA_LEVEL_NAMES, PRIMARY_VQA_LEVEL_NAMES, WORDY_CONTROL_LEVEL_NAME_PAIRS
from ..data.complexity import (
    SEMANTIC_COMPLEXITY_SCORE_FORMULA,
    SEMANTIC_COMPLEXITY_SCORE_LABEL,
)
from ..perturbations.frequency_sweep import compute_critical_cutoff

logger = logging.getLogger(__name__)

_LEVEL_COLORS = {
    "L1_COARSE": "#1f77b4",
    "L2_MEDIUM": "#2ca02c",
    "L3_FINE": "#ff7f0e",
    "L4_VERY_FINE": "#d62728",
    "L5_WORDY_SIMPLETON": "#9467bd",
    "L6_WORDY_MEDIUM": "#8c6bb1",
    "L7_WORDY_FINE": "#9e9ac8",
    "L8_WORDY_VERY_FINE": "#bcbddc",
}
_LEVEL_LABELS = {
    "L1_COARSE": "L1 (Coarse)",
    "L2_MEDIUM": "L2 (Medium)",
    "L3_FINE": "L3 (Fine)",
    "L4_VERY_FINE": "L4 (Very Fine)",
    "L5_WORDY_SIMPLETON": "L5 (Wordy-Simpleton)",
    "L6_WORDY_MEDIUM": "L6 (Wordy-Medium)",
    "L7_WORDY_FINE": "L7 (Wordy-Fine)",
    "L8_WORDY_VERY_FINE": "L8 (Wordy-Very-Fine)",
}
_LEVEL_ORDER = list(ALL_VQA_LEVEL_NAMES)
_PRIMARY_LEVEL_ORDER = list(PRIMARY_VQA_LEVEL_NAMES)
_SEG_LEVEL_ORDER = ["L1_COARSE", "L2_MEDIUM", "L3_FINE"]
_FILTER_ANALYSIS_ORDER = ["overall", "early", "mid", "late"]
_FILTER_ANALYSIS_COLORS = ["#636363", "#9ecae1", "#fdae6b", "#9e9ac8"]
_EXP2_CONTROL_COLORS = {
    "task": "#222222",
    "empty_language": "#7f7f7f",
    "random_language": "#9467bd",
}
_EXP5_TARGET_LABELS = {
    "accuracy_drop": "Observed Accuracy Drop",
    "loglik_erosion": "Correct-Answer Log-Likelihood Erosion",
    "net_drop": "Net Accuracy Change",
    "relative_accuracy_drop": "Relative Accuracy Drop",
}
_COEFFICIENT_LABELS = {
    "complexity_score": "Semantic Complexity",
    "complexity_score_residual": "Residualized\nSemantic Logic",
    "prompt_complexity_score": "Prompt Load",
    "option_hardness_score": "Option Hardness",
    "prediction_entropy": "Prediction Entropy\n(Clean Confusion)",
    "clean_accuracy": "Clean Accuracy\n(Baseline)",
}
_COEFFICIENT_COLORS = {
    "complexity_score": "#2ca02c",
    "complexity_score_residual": "#2ca02c",
    "prompt_complexity_score": "#7f7f7f",
    "option_hardness_score": "#ff7f0e",
    "prediction_entropy": "#17becf",
    "clean_accuracy": "#1f77b4",
}
_DEFAULT_PLOT_PROFILE = "exhaustive"
_FULL_PLOT_PROFILES = {"full", "exhaustive", "all"}
_PLOT_MANIFEST: List[Dict[str, Any]] = []
_DEFAULT_SUPPRESS_DC = True
_LOG_FLOOR = 1e-10
_DEFAULT_SPECTRAL_LOG_AXES = True
_WORDY_PAIR_LABELS = {
    "L1_COARSE": "L1 vs L5",
    "L2_MEDIUM": "L2 vs L6",
    "L3_FINE": "L3 vs L7",
    "L4_VERY_FINE": "L4 vs L8",
}
_EXP1_PERTURBATION_OUTCOME_SPECS = (
            (
                "accuracy_drop",
                "horse_race_mean_accuracy_drop_by_perturbation",
                "exp1_coefficient_plot_accuracy_drop_by_perturbation.png",
                "Coefficient Plot Series: Accuracy Drop by Perturbation",
                "supplementary_exp1_horse_race_by_perturbation.png",
                "Supplementary: Horse Race by Individual Perturbation",
                "gated accuracy drop",
                "Only clean-correct points are included in each perturbation-specific regression",
            ),
            (
                "loglik_drift",
                "horse_race_mean_loglik_drift_by_perturbation",
                "exp1_coefficient_plot_loglik_drift_by_perturbation.png",
                "Coefficient Plot Series: Log-Likelihood Drift by Perturbation",
                "supplementary_exp1_horse_race_loglik_drift_by_perturbation.png",
                "Supplementary: Log-Likelihood Drift by Individual Perturbation",
                "signed correct-answer log-likelihood drift",
                "Positive values mean confidence erosion; negative values mean confidence recovery",
            ),
            (
                "loglik_erosion",
                "horse_race_mean_loglik_erosion_by_perturbation",
                "exp1_coefficient_plot_loglik_erosion_by_perturbation.png",
                "Coefficient Plot Series: Log-Likelihood Erosion by Perturbation",
                "supplementary_exp1_horse_race_loglik_erosion_by_perturbation.png",
                "Supplementary: Log-Likelihood Erosion by Individual Perturbation",
                "positive correct-answer log-likelihood drift",
                "Only positive drifts contribute to this outcome",
            ),
            (
                "loglik_recovery",
                "horse_race_mean_loglik_recovery_by_perturbation",
                "exp1_coefficient_plot_loglik_recovery_by_perturbation.png",
                "Coefficient Plot Series: Log-Likelihood Recovery by Perturbation",
                "supplementary_exp1_horse_race_loglik_recovery_by_perturbation.png",
                "Supplementary: Log-Likelihood Recovery by Individual Perturbation",
                "negative correct-answer log-likelihood drift",
                "More negative values indicate stronger confidence recovery",
            ),
            (
                "loglik_volatility",
                "horse_race_mean_loglik_volatility_by_perturbation",
                "exp1_coefficient_plot_loglik_volatility_by_perturbation.png",
                "Coefficient Plot Series: Log-Likelihood Volatility by Perturbation",
                "supplementary_exp1_horse_race_loglik_volatility_by_perturbation.png",
                "Supplementary: Log-Likelihood Volatility by Individual Perturbation",
                "absolute correct-answer log-likelihood drift",
                "Higher values mean larger movement away from zero regardless of sign",
            ),
        )


def _metadata_lines(*parts: Optional[str]) -> List[str]:
    return [str(part) for part in parts if part]


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if np.isfinite(value_float) else None


def _horse_race_predictor_keys(points: Sequence[Dict[str, Any]]) -> List[str]:
    keys = [
        "complexity_score_residual",
        "prompt_complexity_score",
        "option_hardness_score",
    ]
    if any(_optional_float(point.get("prediction_entropy")) is not None for point in points):
        keys.append("prediction_entropy")
    return keys


def _plot_metadata(
    *,
    experiment: Optional[str] = None,
    what: Optional[str] = None,
    aggregation: Optional[str] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    selection: Optional[str] = None,
    note: Optional[str] = None,
    profile: Optional[str] = None,
) -> List[str]:
    return _metadata_lines(
        f"Experiment={experiment}" if experiment else None,
        f"What={what}" if what else None,
        f"Aggregation={aggregation}" if aggregation else None,
        f"X={x}" if x else None,
        f"Y={y}" if y else None,
        f"Selection={selection}" if selection else None,
        f"Note={note}" if note else None,
        f"PlotProfile={profile}" if profile else None,
    )


def _set_plot_metadata(fig: plt.Figure, lines: Optional[Sequence[str]]) -> None:
    if not lines:
        return
    setattr(fig, "_fa_metadata_lines", [str(line) for line in lines if str(line).strip()])


def _complexity_axis_label(x_key: str = "complexity_score_residual") -> str:
    if x_key == "complexity_score_residual":
        return "Residualized Semantic Logic"
    if x_key == "prompt_complexity_score":
        return "Prompt Load"
    if x_key == "option_hardness_score":
        return "Option Hardness"
    if x_key == "prediction_entropy":
        return "Prediction Entropy"
    return SEMANTIC_COMPLEXITY_SCORE_LABEL


def _complexity_plot_note(x_key: str = "complexity_score_residual") -> str:
    if x_key == "complexity_score_residual":
        return (
            "X-axis is residual(semantic complexity ~ prompt load); raw semantic formula = "
            + SEMANTIC_COMPLEXITY_SCORE_FORMULA
        )
    return "Semantic score = " + SEMANTIC_COMPLEXITY_SCORE_FORMULA


def _tight_layout(fig: plt.Figure, *, metadata_bottom: float = 0.08, top: float = 0.97) -> None:
    metadata = getattr(fig, "_fa_metadata_lines", None)
    if metadata:
        fig.tight_layout(rect=(0, metadata_bottom, 1, top))
    else:
        fig.tight_layout()


def _apply_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def _spectral_plot_values(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.clip(arr, _LOG_FLOOR, None)


def _apply_spectral_axis_scale(
    ax: plt.Axes,
    *,
    use_log_axes: bool = _DEFAULT_SPECTRAL_LOG_AXES,
) -> None:
    if not use_log_axes:
        return
    ax.set_xscale("log")
    ax.set_yscale("log")


def _zscore_for_plot(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std <= _LOG_FLOOR:
        return arr - mean
    return (arr - mean) / std


def _figure_title(fig: plt.Figure) -> str:
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text():
        return str(suptitle.get_text())
    for axis in fig.axes:
        title = axis.get_title()
        if title:
            return str(title)
    return ""


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 250) -> None:
    metadata = getattr(fig, "_fa_metadata_lines", None)
    if metadata:
        fig.text(
            0.01,
            0.012,
            " | ".join(metadata),
            ha="left",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    png_metadata = {
        "Title": _figure_title(fig) or path.stem,
        "Description": " | ".join(metadata or []),
        "Software": "frequency_alignment.plotting",
    }
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", metadata=png_metadata)
    _PLOT_MANIFEST.append(
        {
            "filename": path.name,
            "path": str(path),
            "title": png_metadata["Title"],
            "description": png_metadata["Description"],
            "metadata_lines": list(metadata or []),
        }
    )
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ordered_levels(keys: Iterable[str], segmentation: bool = False) -> List[str]:
    order = _SEG_LEVEL_ORDER if segmentation else _LEVEL_ORDER
    key_set = set(keys)
    return [level for level in order if level in key_set]


def _level_label(level_key: str) -> str:
    return _LEVEL_LABELS.get(level_key, level_key)


def _short_sample_id(sample_id: str, max_chars: int = 18) -> str:
    sample_id = str(sample_id)
    if len(sample_id) <= max_chars:
        return sample_id
    return f"{sample_id[:8]}...{sample_id[-7:]}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80]


def _exp5_target_label(target_key: str) -> str:
    return _EXP5_TARGET_LABELS.get(target_key, target_key.replace("_", " ").title())


def _exp5_source_label(source_name: str) -> str:
    mapping = {
        "image_space": "Image-space (relative)",
        "image_space_raw": "Image-space (raw)",
        "vision_feature_space": "Vision-feature (relative)",
        "vision_feature_space_raw": "Vision-feature (raw)",
    }
    return mapping.get(source_name, source_name.replace("_", " ").title())


def _exp2_control_label(control_name: str) -> str:
    mapping = {
        "empty_language": "Empty-language control",
        "random_language": "Random-language control",
    }
    return mapping.get(control_name, control_name.replace("_", " ").title())


def _plot_profile(config: Optional[Dict[str, Any]]) -> str:
    if not config:
        return _DEFAULT_PLOT_PROFILE
    plotting = config.get("plotting", {}) if isinstance(config, dict) else {}
    profile = str(plotting.get("profile", _DEFAULT_PLOT_PROFILE)).strip().lower()
    return profile or _DEFAULT_PLOT_PROFILE


def _is_exhaustive_profile(profile: str) -> bool:
    return profile in _FULL_PLOT_PROFILES


def _representative_sample_limit(config: Optional[Dict[str, Any]], profile: str) -> int:
    plotting = config.get("plotting", {}) if isinstance(config, dict) else {}
    raw_value = plotting.get("representative_samples")
    if raw_value is not None:
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            pass
    return 3 if _is_exhaustive_profile(profile) else 2


def _sample_scatter_point_limit(config: Optional[Dict[str, Any]], profile: str) -> int:
    plotting = config.get("plotting", {}) if isinstance(config, dict) else {}
    raw_value = plotting.get("max_sample_scatter_points")
    if raw_value is not None:
        try:
            return max(100, int(raw_value))
        except (TypeError, ValueError):
            pass
    return 5000 if _is_exhaustive_profile(profile) else 2500


def _analysis_suppress_dc(config: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(config, dict):
        return _DEFAULT_SUPPRESS_DC
    return bool(config.get("analysis", {}).get("suppress_dc", _DEFAULT_SUPPRESS_DC))


def _band_labels(length: int, suppress_dc: bool = _DEFAULT_SUPPRESS_DC) -> List[str]:
    start = 1 if suppress_dc else 0
    return [f"B{start + idx}" for idx in range(max(0, int(length)))]


def _spectral_x_positions(length: int, suppress_dc: bool = _DEFAULT_SUPPRESS_DC) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    total_bands = length + 1 if suppress_dc else length
    centers = spectral_band_centers(total_bands, suppress_dc=suppress_dc)
    if len(centers) != length:
        centers = np.arange(1, length + 1, dtype=np.float64)
    return np.clip(np.asarray(centers, dtype=np.float64), _LOG_FLOOR, None)


def _write_plot_manifest(plots_dir: Path, profile: str) -> None:
    manifest = {
        "profile": profile,
        "num_plots": len(_PLOT_MANIFEST),
        "plots": _PLOT_MANIFEST,
    }
    (plots_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2))


def _sort_metric_rows(
    row_labels: Sequence[str],
    matrix: np.ndarray,
    max_rows: int = 10,
) -> Tuple[List[str], np.ndarray]:
    if matrix.size == 0:
        return list(row_labels), matrix
    scores = np.nanmean(matrix, axis=1)
    order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
    order = order[: min(max_rows, len(order))]
    return [row_labels[idx] for idx in order], matrix[order]


def _record_image_id(record: Dict[str, Any]) -> Optional[str]:
    image_id = record.get("image_id")
    if image_id is None:
        return None
    return str(image_id)


def _select_records_by_preferred_ids(
    records: List[Dict[str, Any]],
    preferred_ids: Optional[Sequence[str]],
    *,
    max_samples: int,
    fallback_selector,
) -> List[Dict[str, Any]]:
    if max_samples <= 0:
        return []

    selected: List[Dict[str, Any]] = []
    seen_ids = set()
    records_by_id: Dict[str, Dict[str, Any]] = {}
    for record in records:
        image_id = _record_image_id(record)
        if image_id is None or image_id in records_by_id:
            continue
        records_by_id[image_id] = record

    for image_id in preferred_ids or []:
        image_id = str(image_id)
        record = records_by_id.get(image_id)
        if record is None or image_id in seen_ids:
            continue
        selected.append(record)
        seen_ids.add(image_id)
        if len(selected) >= max_samples:
            return selected

    for record in fallback_selector(records, max_samples=max_samples):
        image_id = _record_image_id(record)
        if image_id is None or image_id in seen_ids:
            continue
        selected.append(record)
        seen_ids.add(image_id)
        if len(selected) >= max_samples:
            break

    return selected


def _plot_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    out_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    annotate: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    metadata: Optional[Sequence[str]] = None,
) -> None:
    if matrix.size == 0 or not row_labels or not col_labels:
        return

    fig_width = max(8.0, 2.5 + 0.55 * len(col_labels))
    fig_height = max(4.5, 2.0 + 0.45 * len(row_labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=13)
    _apply_style(ax)
    _set_plot_metadata(
        fig,
        metadata
        or _metadata_lines(
            "View=heatmap",
            f"Rows={len(row_labels)} groups",
            f"Cols={len(col_labels)} bins",
            f"Value={colorbar_label}",
        ),
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(colorbar_label, fontsize=10)

    if annotate and len(row_labels) <= 12 and len(col_labels) <= 16:
        finite = matrix[np.isfinite(matrix)]
        mean_value = float(np.mean(finite)) if finite.size else 0.0
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                if np.isnan(value):
                    continue
                color = "white" if np.nan_to_num(value, nan=0.0) > mean_value else "black"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)

    _tight_layout(fig)
    _save_fig(fig, out_path)


def _plot_grouped_bars(
    series: Dict[str, Sequence[float]],
    x_labels: Sequence[str],
    out_path: Path,
    title: str,
    ylabel: str,
    colors: Optional[Sequence[str]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    metadata: Optional[Sequence[str]] = None,
) -> None:
    if not series or not x_labels:
        return

    fig, ax = plt.subplots(figsize=(max(8.0, 2.0 + len(x_labels) * 1.1), 5.2))
    x = np.arange(len(x_labels))
    n_series = len(series)
    width = 0.8 / max(1, n_series)

    for idx, (label, values) in enumerate(series.items()):
        offset = (idx - (n_series - 1) / 2) * width
        color = colors[idx] if colors and idx < len(colors) else None
        ax.bar(x + offset, values, width=width * 0.92, label=label, alpha=0.85, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9)
    _apply_style(ax)
    _set_plot_metadata(
        fig,
        metadata
        or _metadata_lines(
            "View=grouped bars",
            f"Categories={len(x_labels)}",
            f"Series={len(series)}",
            f"Y={ylabel}",
        ),
    )
    _tight_layout(fig)
    _save_fig(fig, out_path)


def _plot_wordy_control_pair_comparison(
    values_by_level: Dict[str, float],
    out_path: Path,
    *,
    title: str,
    ylabel: str,
    metadata: Optional[Sequence[str]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    base_label: str = "Base",
    control_label: str = "Wordy Control",
) -> None:
    pair_labels: List[str] = []
    base_values: List[float] = []
    control_values: List[float] = []
    for base_level, control_level in WORDY_CONTROL_LEVEL_NAME_PAIRS:
        if base_level not in values_by_level or control_level not in values_by_level:
            continue
        pair_labels.append(_WORDY_PAIR_LABELS.get(base_level, f"{base_level} vs {control_level}"))
        base_values.append(float(values_by_level[base_level]))
        control_values.append(float(values_by_level[control_level]))
    if not pair_labels:
        return
    _plot_grouped_bars(
        {
            base_label: base_values,
            control_label: control_values,
        },
        pair_labels,
        out_path,
        title,
        ylabel,
        colors=["#4c78a8", "#8c6bb1"],
        ylim=ylim,
        metadata=metadata,
    )


def _plot_linguistic_stabilization_effect(
    mirror_tests: Dict[str, Any],
    out_path: Path,
    *,
    profile: str = _DEFAULT_PLOT_PROFILE,
) -> None:
    outcomes = mirror_tests.get("outcomes", {}) if isinstance(mirror_tests, dict) else {}
    if not outcomes:
        return
    outcome_specs = [
        ("mean_accuracy_drop", "Accuracy Drop"),
        ("mean_loglik_drift", "Log-Likelihood Drift"),
        ("mean_loglik_erosion", "Log-Likelihood Erosion"),
        ("mean_loglik_volatility", "Log-Likelihood Volatility"),
    ]
    valid_specs = [
        (key, label)
        for key, label in outcome_specs
        if outcomes.get(key, {}).get("pairs")
    ]
    if not valid_specs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8), squeeze=False)
    axes_arr = axes.ravel()
    for axis in axes_arr[len(valid_specs):]:
        axis.axis("off")

    for axis, (outcome_key, label) in zip(axes_arr, valid_specs):
        pair_payload = outcomes.get(outcome_key, {}).get("pairs", {})
        labels: List[str] = []
        centers: List[float] = []
        lower_err: List[float] = []
        upper_err: List[float] = []
        colors: List[str] = []
        for base_level, control_level in WORDY_CONTROL_LEVEL_NAME_PAIRS:
            pair_key = f"{base_level}__{control_level}"
            stats = pair_payload.get(pair_key)
            if not stats or int(stats.get("n_pairs", 0) or 0) <= 0:
                continue
            center = float(stats.get("mean_stabilization_effect", 0.0) or 0.0)
            lower = float(stats.get("ci_95_lower", center) or center)
            upper = float(stats.get("ci_95_upper", center) or center)
            labels.append(_WORDY_PAIR_LABELS.get(base_level, f"{base_level} vs {control_level}"))
            centers.append(center)
            lower_err.append(max(0.0, center - lower))
            upper_err.append(max(0.0, upper - center))
            colors.append(_LEVEL_COLORS.get(base_level, "#4c78a8"))
        if not centers:
            axis.axis("off")
            continue
        x = np.arange(len(centers))
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
        axis.bar(x, centers, color=colors, alpha=0.86, width=0.65)
        axis.errorbar(
            x,
            centers,
            yerr=np.vstack([lower_err, upper_err]),
            fmt="none",
            ecolor="#333333",
            elinewidth=1.4,
            capsize=4,
        )
        axis.set_xticks(x)
        axis.set_xticklabels(labels, fontsize=9)
        axis.set_title(label, fontsize=11)
        axis.set_ylabel("Base - Wordy", fontsize=10)
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
        _apply_style(axis)

    fig.suptitle("Linguistic Stabilization Effect: Terse vs Wordy Mirrors", fontsize=14, y=0.99)
    _set_plot_metadata(
        fig,
        _plot_metadata(
            experiment="1",
            what="Paired mirror test for wordy-control stabilization",
            aggregation="paired within-image L1-L4 base levels minus L5-L8 wordy mirrors",
            x="matched terse/wordy level pair",
            y="base outcome minus wordy outcome",
            note="Positive bars mean the wordy prompt reduced the measured drop or drift; error bars are 95% paired-delta CIs",
            profile=profile,
        ),
    )
    _tight_layout(fig, metadata_bottom=0.08, top=0.95)
    _save_fig(fig, out_path)


def _plot_distribution_with_points(
    grouped_values: Dict[str, Sequence[float]],
    out_path: Path,
    title: str,
    ylabel: str,
    metadata: Optional[Sequence[str]] = None,
) -> None:
    present = _ordered_levels(grouped_values.keys())
    if not present:
        return

    datasets = [list(grouped_values[level]) for level in present]
    if not any(datasets):
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    box = ax.boxplot(datasets, patch_artist=True, widths=0.55, showfliers=False)
    for patch, level in zip(box["boxes"], present):
        patch.set_facecolor(_LEVEL_COLORS.get(level, "#999"))
        patch.set_alpha(0.45)

    rng = np.random.default_rng(42)
    for idx, level in enumerate(present):
        values = np.asarray(grouped_values[level], dtype=float)
        if values.size == 0:
            continue
        jitter = rng.uniform(-0.08, 0.08, size=values.size)
        ax.scatter(
            np.full(values.size, idx + 1) + jitter,
            values,
            s=18,
            color=_LEVEL_COLORS.get(level, "#999"),
            alpha=0.65,
            edgecolors="white",
            linewidth=0.4,
        )

    ax.set_xticks(np.arange(1, len(present) + 1))
    ax.set_xticklabels([_level_label(level) for level in present], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    _apply_style(ax)
    _set_plot_metadata(
        fig,
        metadata
        or _metadata_lines(
            "View=distribution",
            "Boxplots show within-run sample spread",
            f"Levels={len(present)}",
            f"Y={ylabel}",
        ),
    )
    _tight_layout(fig)
    _save_fig(fig, out_path)


def _plot_sample_profile_grid(
    sample_id: str,
    per_level_series: Dict[str, Dict[str, Sequence[float]]],
    out_path: Path,
    title: str,
    ylabel: str,
    overlay_by_level: Optional[Dict[str, Sequence[float]]] = None,
    metadata: Optional[Sequence[str]] = None,
    suppress_dc: bool = _DEFAULT_SUPPRESS_DC,
) -> None:
    present_levels = _ordered_levels(per_level_series.keys())
    if not present_levels:
        return

    ncols = 2
    nrows = int(np.ceil(len(present_levels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.1 * nrows), sharex=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    for axis in axes_arr.ravel()[len(present_levels):]:
        axis.axis("off")

    for axis, level in zip(axes_arr.ravel(), present_levels):
        series = per_level_series[level]
        if not series:
            axis.axis("off")
            continue

        labels = list(series.keys())
        cmap = plt.cm.get_cmap("tab20", max(1, len(labels)))
        for idx, label in enumerate(labels):
            values = _spectral_plot_values(series[label])
            x_values = _spectral_x_positions(len(values), suppress_dc=suppress_dc)
            axis.plot(
                x_values,
                values,
                linewidth=1.5,
                alpha=0.9,
                label=label,
                color=cmap(idx),
            )

        overlay = None if overlay_by_level is None else overlay_by_level.get(level)
        if overlay is not None:
            overlay_values = _spectral_plot_values(overlay)
            overlay_x = _spectral_x_positions(len(overlay_values), suppress_dc=suppress_dc)
            axis.plot(
                overlay_x,
                overlay_values,
                linewidth=2.4,
                linestyle="--",
                color="black",
                alpha=0.85,
                label="W_t",
            )

        axis.set_title(_level_label(level), fontsize=11)
        axis.set_xlabel("Normalized Frequency", fontsize=10)
        axis.set_ylabel(ylabel, fontsize=10)
        axis.grid(alpha=0.2, linewidth=0.6)
        _apply_spectral_axis_scale(axis)
        _apply_style(axis)

    handles, labels = axes_arr.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
    fig.suptitle(f"{title}: {_short_sample_id(sample_id, 32)}", fontsize=14, y=0.995)
    _set_plot_metadata(
        fig,
        metadata
        or _metadata_lines(
            "View=sample profiles",
            "Panels=task levels",
            "Lines=perturbations for one image",
            f"Y={ylabel}",
        ),
    )
    _tight_layout(fig, metadata_bottom=0.07, top=0.97)
    _save_fig(fig, out_path)


def plot_granularity_curves(
    summary: Dict[str, Any],
    out_path: Path,
    title: str = "Accuracy Degradation vs Task Granularity",
) -> None:
    per_level = summary.get("per_level", {})
    present = _ordered_levels(per_level.keys())
    if not present:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(present))
    width = 0.36

    clean_acc = [per_level[level]["clean_accuracy"] for level in present]
    pert_acc = [per_level[level]["perturbed_accuracy"] for level in present]
    axes[0].bar(x - width / 2, clean_acc, width, color="#2ca02c", alpha=0.85, label="Clean")
    axes[0].bar(x + width / 2, pert_acc, width, color="#d62728", alpha=0.85, label="Perturbed")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([_level_label(level) for level in present], fontsize=9)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Clean vs Perturbed Accuracy", fontsize=12)
    axes[0].legend(fontsize=9)
    _apply_style(axes[0])

    drops = [per_level[level]["mean_accuracy_drop"] for level in present]
    stds = [per_level[level]["std_accuracy_drop"] for level in present]
    axes[1].bar(
        x,
        drops,
        yerr=stds,
        capsize=4,
        color=[_LEVEL_COLORS.get(level, "#999") for level in present],
        alpha=0.9,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_level_label(level) for level in present], fontsize=9)
    axes[1].set_ylabel("Mean Accuracy Drop", fontsize=11)
    axes[1].set_title("Degradation by Level", fontsize=12)
    _apply_style(axes[1])

    fig.suptitle(title, fontsize=14)
    _set_plot_metadata(
        fig,
        _metadata_lines(
            "Experiment=1",
            "Panels=clean vs perturbed accuracy; mean gated accuracy drop",
            "Aggregation=level-wise average over all perturbation evaluations",
            "Error bars=within-run std",
        ),
    )
    _tight_layout(fig)
    _save_fig(fig, out_path)


def plot_attention_power_spectrum(
    summary: Dict[str, Any],
    out_path: Path,
    title: str = "Attention Power Spectrum by Granularity",
    suppress_dc: bool = _DEFAULT_SUPPRESS_DC,
) -> None:
    per_level = summary.get("per_level", {})
    present = _ordered_levels(per_level.keys())
    if not present:
        return

    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    for level in present:
        values = per_level[level].get("W_t_average", [])
        if not values:
            continue
        x_values = _spectral_x_positions(len(values), suppress_dc=suppress_dc)
        ax.plot(
            x_values,
            _spectral_plot_values(values),
            marker="o",
            linewidth=2.0,
            color=_LEVEL_COLORS.get(level, "#999"),
            label=_level_label(level),
        )
    ax.set_xlabel("Normalized Frequency", fontsize=11)
    ax.set_ylabel("Normalized Attention Power", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    _apply_spectral_axis_scale(ax)
    _apply_style(ax)
    _set_plot_metadata(
        fig,
        _metadata_lines(
            "Experiment=2",
            "Lines=mean W_t per level",
            "Aggregation=sample average",
            "X=radial frequency bands low→high",
        ),
    )
    _tight_layout(fig)
    _save_fig(fig, out_path)


def plot_effective_bandwidth(
    summary: Dict[str, Any],
    out_path: Path,
    title: str = "Effective Bandwidth by Task Granularity",
) -> None:
    per_level = summary.get("per_level", {})
    present = _ordered_levels(per_level.keys())
    if not present:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x = np.arange(len(present))
    means = [per_level[level]["mean_bandwidth"] for level in present]
    stds = [per_level[level]["std_bandwidth"] for level in present]
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=[_LEVEL_COLORS.get(level, "#999") for level in present],
        alpha=0.9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_level_label(level) for level in present], fontsize=10)
    ax.set_ylabel("Effective Bandwidth", fontsize=11)
    ax.set_title(title, fontsize=13)
    _apply_style(ax)
    _set_plot_metadata(
        fig,
        _metadata_lines(
            "Experiment=2",
            "Bars=mean effective bandwidth G(t)",
            "Aggregation=sample average by level",
            "Error bars=within-run std",
        ),
    )
    _tight_layout(fig)
    _save_fig(fig, out_path)


def plot_amplification_heatmap(
    summary: Dict[str, Any],
    out_path: Path,
    title: str = "Mean Pre-Fusion Drift Spectrum by Level",
    suppress_dc: bool = _DEFAULT_SUPPRESS_DC,
) -> None:
    per_level = summary.get("per_level", {})
    present = _ordered_levels(per_level.keys())
    if not present:
        return

    matrix = []
    for level in present:
        values = per_level[level].get("mean_pre_drift_bands", [])
        if values:
            matrix.append(values)
    if not matrix:
        return

    _plot_heatmap(
        np.asarray(matrix, dtype=float),
        [_level_label(level) for level in present],
        _band_labels(len(matrix[0]), suppress_dc=suppress_dc),
        out_path,
        title,
        "Mean Pre-Fusion Drift",
        cmap="YlGnBu",
    )


def plot_frequency_threshold_curves(
    accuracy_data: Dict[str, Any],
    out_path: Path,
    mode: str = "lowpass",
    title: str = "Accuracy vs Frequency Cutoff",
) -> None:
    mode_data = accuracy_data.get("data", {}).get(mode, {})
    cutoffs = accuracy_data.get("cutoffs", [])
    present = _ordered_levels(mode_data.keys())
    if not present or not cutoffs:
        return

    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    for level in present:
        curve = mode_data[level].get("accuracy_curve", [])
        if not curve:
            continue
        ax.plot(
            cutoffs[: len(curve)],
            curve,
            marker="o",
            markersize=4,
            linewidth=2,
            color=_LEVEL_COLORS.get(level, "#999"),
            label=_level_label(level),
        )
        omega_c = mode_data[level].get("critical_cutoff")
        if omega_c is not None:
            ax.axvline(
                omega_c,
                color=_LEVEL_COLORS.get(level, "#999"),
                linestyle="--",
                linewidth=1.0,
                alpha=0.45,
            )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, label="50% threshold")
    ax.set_xlabel(f"{mode.capitalize()} cutoff", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{title} ({mode})", fontsize=13)
    ax.legend(fontsize=9)
    _apply_style(ax)
    _set_plot_metadata(
        fig,
        _metadata_lines(
            "Experiment=4",
            f"Mode={mode}",
            "Lines=mean accuracy by level across cutoff sweep",
            "Vertical lines=critical cutoff estimate",
        ),
    )
    _tight_layout(fig)
    _save_fig(fig, out_path)


def _plot_exp4_mode_heatmap(
    accuracy_data: Dict[str, Any],
    out_path: Path,
    *,
    mode: str,
    title: str,
    metadata: Optional[Sequence[str]] = None,
) -> None:
    mode_data = accuracy_data.get("data", {}).get(mode, {})
    cutoffs = accuracy_data.get("cutoffs", [])
    present = _ordered_levels(mode_data.keys())
    if not present or not cutoffs:
        return

    matrix = []
    valid_levels = []
    for level in present:
        curve = mode_data.get(level, {}).get("accuracy_curve", [])
        if not curve:
            continue
        matrix.append(np.asarray(curve, dtype=float))
        valid_levels.append(level)
    if not matrix:
        return

    cutoff_labels = [f"{float(c):.2f}" for c in cutoffs[: len(matrix[0])]]
    _plot_heatmap(
        np.vstack(matrix),
        [_level_label(level) for level in valid_levels],
        cutoff_labels,
        out_path,
        title,
        "Accuracy",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        metadata=metadata,
    )


def plot_overlap_scatter(
    scatter_data: List[Dict[str, Any]],
    pearson_r: float,
    out_path: Path,
    title: str = "Predicted vs Actual Sensitivity",
    point_size: float = 40,
    alpha: float = 0.7,
    y_label: str = "Observed Accuracy Drop",
    scale_mode: str = "raw",
    metadata: Optional[Sequence[str]] = None,
    max_points: Optional[int] = None,
) -> None:
    if not scatter_data:
        return

    original_count = len(scatter_data)
    if max_points is not None and original_count > max_points > 0:
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(original_count, size=int(max_points), replace=False))
        scatter_data = [scatter_data[int(idx)] for idx in keep]
    displayed_count = len(scatter_data)

    preds = np.asarray([item["predicted"] for item in scatter_data], dtype=float)
    actuals = np.asarray([item["actual"] for item in scatter_data], dtype=float)
    plot_x = preds
    plot_y = actuals
    x_label = "Predicted Sensitivity (spectral overlap)"
    displayed_y_label = y_label
    scale_note = None
    scale_mode = str(scale_mode).strip().lower()
    if scale_mode == "zscore":
        plot_x = _zscore_for_plot(preds)
        plot_y = _zscore_for_plot(actuals)
        x_label = "Predicted Sensitivity (z-score)"
        displayed_y_label = f"{y_label} (z-score)"
        scale_note = "Axes are z-scored for visualization only; correlation uses raw values"
    elif scale_mode == "bridge":
        plot_x = _zscore_for_plot(np.log1p(np.clip(preds, 0.0, None)))
        plot_y = _zscore_for_plot(actuals)
        x_label = "Log-Overlap (z-score)"
        displayed_y_label = f"{y_label} (z-score)"
        scale_note = "X uses zscore(log1p(overlap)); Y uses zscore(observed target); raw metrics unchanged"

    fig, ax = plt.subplots(figsize=(7.1, 6.0))
    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, item in enumerate(scatter_data):
        level = item.get("level")
        if level is None:
            label = item.get("label", "")
            level = label.split("|", 1)[0] if "|" in label else "unknown"
        grouped_indices[str(level)].append(idx)
    for level, indices in grouped_indices.items():
        idx_arr = np.asarray(indices, dtype=int)
        ax.scatter(
            plot_x[idx_arr],
            plot_y[idx_arr],
            s=point_size,
            alpha=alpha,
            color=_LEVEL_COLORS.get(level, "#999"),
            edgecolors="white",
            linewidth=0.5,
        )

    if len(plot_x) > 2 and np.unique(plot_x).size > 1:
        fit = np.polyfit(plot_x, plot_y, 1)
        curve = np.poly1d(fit)
        xs = np.linspace(plot_x.min(), plot_x.max(), 100)
        ax.plot(xs, curve(xs), "k--", linewidth=1.5, alpha=0.65)

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(displayed_y_label, fontsize=11)
    ax.set_title(f"{title} (r = {pearson_r:.3f})", fontsize=13)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=_level_label(level), markerfacecolor=color, markersize=8)
        for level, color in _LEVEL_COLORS.items()
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left")
    _apply_style(ax)
    metadata_lines = list(metadata) if metadata else _metadata_lines(
        "View=scatter",
        f"X={x_label}",
        f"Y={displayed_y_label}",
        f"Points={displayed_count}",
        scale_note,
    )
    if displayed_count != original_count:
        metadata_lines.append(
            f"VisualSample={displayed_count} of {original_count}; correlation in title uses all source points"
        )
    _set_plot_metadata(
        fig,
        metadata_lines,
    )
    _tight_layout(fig)
    _save_fig(fig, out_path)


def plot_overlap_scatter_grid_by_level(
    scatter_data: List[Dict[str, Any]],
    out_path: Path,
    *,
    title: str,
    y_label: str = "Observed Accuracy Drop",
    scale_mode: str = "raw",
    metadata: Optional[Sequence[str]] = None,
) -> None:
    if not scatter_data:
        return

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in scatter_data:
        level = item.get("level")
        if level is None:
            label = str(item.get("label", ""))
            level = label.split("|", 1)[0] if "|" in label else None
        if level is not None:
            grouped[str(level)].append(item)

    levels = [level for level in _LEVEL_ORDER if grouped.get(level)]
    if not levels:
        return

    n_panels = len(levels)
    ncols = min(3, max(1, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.4 * nrows), squeeze=False)
    axes_arr = axes.ravel()

    scale_mode = str(scale_mode).strip().lower()
    for axis, level in zip(axes_arr, levels):
        level_points = grouped[level]
        preds = np.asarray([item["predicted"] for item in level_points], dtype=float)
        actuals = np.asarray([item["actual"] for item in level_points], dtype=float)
        plot_x = preds
        plot_y = actuals
        x_label = "Predicted Sensitivity (spectral overlap)"
        displayed_y_label = y_label
        if scale_mode == "zscore":
            plot_x = _zscore_for_plot(preds)
            plot_y = _zscore_for_plot(actuals)
            x_label = "Predicted Sensitivity (z-score)"
            displayed_y_label = f"{y_label} (z-score)"
        elif scale_mode == "bridge":
            plot_x = _zscore_for_plot(np.log1p(np.clip(preds, 0.0, None)))
            plot_y = _zscore_for_plot(actuals)
            x_label = "Log-Overlap (z-score)"
            displayed_y_label = f"{y_label} (z-score)"

        axis.scatter(
            plot_x,
            plot_y,
            s=42,
            alpha=0.72,
            color=_LEVEL_COLORS.get(level, "#999999"),
            edgecolors="white",
            linewidth=0.5,
        )
        if len(plot_x) > 2 and np.unique(plot_x).size > 1:
            fit = np.polyfit(plot_x, plot_y, 1)
            curve = np.poly1d(fit)
            xs = np.linspace(plot_x.min(), plot_x.max(), 100)
            axis.plot(xs, curve(xs), "k--", linewidth=1.3, alpha=0.65)
        r = np.corrcoef(preds, actuals)[0, 1] if len(preds) > 1 and np.std(preds) > 0 and np.std(actuals) > 0 else 0.0
        axis.set_title(f"{_level_label(level)} (r = {float(r):.3f})", fontsize=11)
        axis.set_xlabel(x_label, fontsize=9)
        axis.set_ylabel(displayed_y_label, fontsize=9)
        axis.grid(alpha=0.2, linewidth=0.6)
        _apply_style(axis)

    for axis in axes_arr[len(levels):]:
        axis.axis("off")

    fig.suptitle(title, fontsize=14, y=0.99)
    _set_plot_metadata(
        fig,
        metadata
        or _metadata_lines(
            "View=multi-panel scatter",
            "Panels=one level per subplot",
            f"Y={y_label}",
            f"Points={len(scatter_data)}",
        ),
    )
    _tight_layout(fig, metadata_bottom=0.07, top=0.95)
    _save_fig(fig, out_path)


def _complexity_valid_points(
    points: Sequence[Dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    for point in points:
        x_val = point.get(x_key)
        y_val = point.get(y_key)
        if x_val is None or y_val is None:
            continue
        x_float = float(x_val)
        y_float = float(y_val)
        if not np.isfinite(x_float) or not np.isfinite(y_float):
            continue
        valid.append(point)
    return valid


def _plot_complexity_scatter_on_axis(
    ax: plt.Axes,
    points: Sequence[Dict[str, Any]],
    *,
    x_key: str = "complexity_score_residual",
    y_key: str,
    y_label: str,
    title: str,
) -> None:
    valid = _complexity_valid_points(points, x_key=x_key, y_key=y_key)
    if not valid:
        ax.axis("off")
        return

    levels = _ordered_levels({point.get("level") for point in valid if point.get("level")})
    for level in levels:
        level_points = [point for point in valid if point.get("level") == level]
        xs = np.asarray([float(point[x_key]) for point in level_points], dtype=float)
        ys = np.asarray([float(point[y_key]) for point in level_points], dtype=float)
        ax.scatter(
            xs,
            ys,
            s=36,
            alpha=0.68,
            color=_LEVEL_COLORS.get(level, "#999999"),
            edgecolors="white",
            linewidth=0.4,
            label=_level_label(level),
        )

    grouped: Dict[float, List[float]] = {}
    for point in valid:
        grouped.setdefault(float(point[x_key]), []).append(float(point[y_key]))
    mean_x = np.asarray(sorted(grouped), dtype=float)
    mean_y = np.asarray([float(np.mean(grouped[x])) for x in mean_x], dtype=float)
    ax.plot(
        mean_x,
        mean_y,
        color="black",
        linewidth=1.8,
        marker="o",
        markersize=4.5,
        alpha=0.85,
        label="Mean by score",
    )

    raw_x = np.asarray([float(point[x_key]) for point in valid], dtype=float)
    raw_y = np.asarray([float(point[y_key]) for point in valid], dtype=float)
    if raw_x.size >= 3 and np.unique(raw_x).size > 1:
        slope, intercept = np.polyfit(raw_x, raw_y, 1)
        grid_x = np.linspace(raw_x.min(), raw_x.max(), 100)
        ax.plot(
            grid_x,
            slope * grid_x + intercept,
            linestyle="--",
            color="#222222",
            linewidth=1.5,
            alpha=0.7,
            label="Linear fit",
        )

    ax.set_xlabel(_complexity_axis_label(x_key), fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.2, linewidth=0.6)
    _apply_style(ax)


def plot_complexity_scatter(
    points: Sequence[Dict[str, Any]],
    *,
    y_key: str,
    y_label: str,
    out_path: Path,
    title: str,
    metadata: Optional[Sequence[str]] = None,
    x_key: str = "complexity_score_residual",
) -> None:
    resolved_x_key = x_key
    valid = _complexity_valid_points(points, x_key=resolved_x_key, y_key=y_key)
    if not valid and resolved_x_key != "complexity_score":
        resolved_x_key = "complexity_score"
        valid = _complexity_valid_points(points, x_key=resolved_x_key, y_key=y_key)
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    _plot_complexity_scatter_on_axis(
        ax,
        valid,
        x_key=resolved_x_key,
        y_key=y_key,
        y_label=y_label,
        title=title,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc="best", title="Series", title_fontsize=8)
    _set_plot_metadata(
        fig,
        metadata
        or _metadata_lines(
            "View=scatter",
            f"X={_complexity_axis_label(resolved_x_key).lower()}",
            f"Y={y_label}",
            "Points=per-image per-level summaries",
            "Black line=mean by exact score; dashed line=linear fit",
            _complexity_plot_note(resolved_x_key),
        ),
    )
    _tight_layout(fig)
    _save_fig(fig, out_path)


def _plot_exp2_complexity_groups(
    points: Sequence[Dict[str, Any]],
    out_path: Path,
    profile: str = _DEFAULT_PLOT_PROFILE,
) -> None:
    panels = [
        ("overall", "bandwidth", "Overall"),
        ("early", "bandwidth_early", "Early"),
        ("mid", "bandwidth_mid", "Mid"),
        ("late", "bandwidth_late", "Late"),
    ]
    x_key = "complexity_score_residual"
    if not any(_complexity_valid_points(points, x_key=x_key, y_key=y_key) for _, y_key, _ in panels):
        x_key = "complexity_score"
    valid_panels = [
        (group_name, y_key, title)
        for group_name, y_key, title in panels
        if _complexity_valid_points(points, x_key=x_key, y_key=y_key)
    ]
    if not valid_panels:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0), sharex=True)
    axes_arr = np.atleast_1d(axes).reshape(2, 2)
    for axis in axes_arr.ravel()[len(valid_panels):]:
        axis.axis("off")

    for axis, (_, y_key, panel_title) in zip(axes_arr.ravel(), valid_panels):
        _plot_complexity_scatter_on_axis(
            axis,
            points,
            x_key=x_key,
            y_key=y_key,
            y_label="Effective Bandwidth G(t)",
            title=panel_title,
        )

    handles, labels = axes_arr.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Bandwidth vs Residualized Semantic Logic by Layer Group", fontsize=14, y=0.99)
    _set_plot_metadata(
        fig,
        _plot_metadata(
            experiment="2",
            what="Residualized semantic logic vs effective bandwidth",
            aggregation="per-image per-level points",
            x=_complexity_axis_label(x_key).lower(),
            y="effective bandwidth G(t)",
            note=(
                "Black line is mean by exact score; dashed line is linear fit; "
                + _complexity_plot_note(x_key)
            ),
            profile=profile,
        ),
    )
    _tight_layout(fig, metadata_bottom=0.07, top=0.96)
    _save_fig(fig, out_path)


def _plot_exp4_complexity_modes(
    points: Sequence[Dict[str, Any]],
    out_path: Path,
    profile: str = _DEFAULT_PLOT_PROFILE,
) -> None:
    mode_order = ["lowpass", "highpass"]
    x_key = "complexity_score_residual"
    if not any(
        _complexity_valid_points(
            [point for point in points if point.get("mode") == mode],
            x_key=x_key,
            y_key="critical_cutoff",
        )
        for mode in mode_order
    ):
        x_key = "complexity_score"
    valid_modes = [
        mode for mode in mode_order
        if _complexity_valid_points(
            [point for point in points if point.get("mode") == mode],
            x_key=x_key,
            y_key="critical_cutoff",
        )
    ]
    if not valid_modes:
        return

    fig, axes = plt.subplots(1, len(valid_modes), figsize=(6.4 * len(valid_modes), 5.1), sharey=True)
    axes_arr = np.atleast_1d(axes)
    for axis, mode in zip(axes_arr, valid_modes):
        mode_points = [point for point in points if point.get("mode") == mode]
        _plot_complexity_scatter_on_axis(
            axis,
            mode_points,
            x_key=x_key,
            y_key="critical_cutoff",
            y_label="Critical Cutoff",
            title=mode.capitalize(),
        )

    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, title="Series", title_fontsize=8)
    fig.suptitle("Critical Cutoff vs Residualized Semantic Logic", fontsize=14, y=0.99)
    _set_plot_metadata(
        fig,
        _plot_metadata(
            experiment="4",
            what="Residualized semantic logic vs critical cutoff",
            aggregation="per-image per-level points",
            x=_complexity_axis_label(x_key).lower(),
            y="critical cutoff",
            note=(
                "Black line is mean by exact score; dashed line is linear fit; "
                + _complexity_plot_note(x_key)
            ),
            profile=profile,
        ),
    )
    _tight_layout(fig, metadata_bottom=0.07, top=0.96)
    _save_fig(fig, out_path)


def _standardized_beta_and_ci(predictor_stats: Dict[str, Any]) -> Tuple[float, float, float]:
    beta = float(predictor_stats.get("beta", 0.0) or 0.0)
    std_beta = float(predictor_stats.get("standardized_beta", 0.0) or 0.0)
    stderr = float(predictor_stats.get("stderr", 0.0) or 0.0)
    if abs(beta) > _LOG_FLOOR:
        scale = std_beta / beta
        std_stderr = abs(scale) * stderr
    else:
        std_stderr = 0.0
    delta = 1.96 * std_stderr
    return std_beta, std_beta - delta, std_beta + delta


def _regression_series(
    primary_label: str,
    primary_regression: Optional[Dict[str, Any]],
    comparison_label: Optional[str] = None,
    comparison_regression: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    series: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(primary_regression, dict) and primary_regression.get("predictors"):
        series.append((primary_label, primary_regression))
    if comparison_label and isinstance(comparison_regression, dict) and comparison_regression.get("predictors"):
        series.append((comparison_label, comparison_regression))
    return series


def _collect_regression_series_from_payload_map(
    payload_map: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    series: Dict[str, Dict[str, Any]] = {}
    for perturbation_name, payload in sorted(payload_map.items()):
        if not isinstance(payload, dict):
            continue
        pooled = payload.get("pooled", {})
        within = payload.get("within_image", {})
        regression = pooled if isinstance(pooled, dict) and pooled.get("predictors") else within
        if regression and regression.get("predictors"):
            series[perturbation_name] = regression
    return series


def plot_coefficient_forest(
    regression: Dict[str, Any],
    out_path: Path,
    *,
    title: str,
    subtitle: Optional[str] = None,
    primary_label: str = "Pooled OLS",
    comparison_label: Optional[str] = None,
    comparison_regression: Optional[Dict[str, Any]] = None,
    metadata: Optional[Sequence[str]] = None,
) -> None:
    series = _regression_series(
        primary_label,
        regression,
        comparison_label=comparison_label,
        comparison_regression=comparison_regression,
    )
    if not series:
        return

    predictor_keys = {
        key
        for _, reg in series
        for key in (reg.get("predictors", {}) if isinstance(reg, dict) else {})
    }
    order = [
        key
        for key in (
            "complexity_score_residual",
            "complexity_score",
            "prompt_complexity_score",
            "option_hardness_score",
            "prediction_entropy",
            "clean_accuracy",
        )
        if key in predictor_keys
    ]
    if not order:
        return

    labels = [_COEFFICIENT_LABELS.get(key, key) for key in order]
    y_positions = np.arange(len(order))[::-1]
    fig_height = max(3.4, 1.3 + 0.9 * len(order))
    fig, ax = plt.subplots(figsize=(8.4, fig_height))
    ax.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)

    marker_cycle = ["o", "s", "D"]
    offset_values = np.linspace(-0.16, 0.16, num=len(series)) if len(series) > 1 else np.array([0.0])
    legend_handles: List[Any] = []
    for series_idx, (series_label, reg) in enumerate(series):
        predictors = reg.get("predictors", {}) if isinstance(reg, dict) else {}
        marker = marker_cycle[series_idx % len(marker_cycle)]
        for idx, key in enumerate(order):
            predictor_stats = predictors.get(key)
            if predictor_stats is None:
                continue
            center, lower, upper = _standardized_beta_and_ci(predictor_stats)
            color = _COEFFICIENT_COLORS.get(key, "#444444")
            ax.errorbar(
                center,
                y_positions[idx] + float(offset_values[series_idx]),
                xerr=[[center - lower], [upper - center]],
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=2.0,
                capsize=4,
                markersize=8,
                alpha=0.95,
            )
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="#444444",
                linestyle="None",
                markersize=8,
                label=series_label,
            )
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Standardized Coefficient (95% CI)", fontsize=11)
    ax.set_title(title, fontsize=13)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#555555",
        )
    if legend_handles and len(legend_handles) > 1:
        ax.legend(handles=legend_handles, fontsize=8, loc="lower right", title="Regression", title_fontsize=8)
    ax.grid(axis="x", alpha=0.2, linewidth=0.6)
    _apply_style(ax)
    _set_plot_metadata(
        fig,
        metadata
        or _metadata_lines(
            "View=coefficient plot",
            "Dots=standardized coefficients",
            "Bars=approximate 95% CI from regression stderr",
            "Predictors=residualized semantic logic, prompt load, option hardness, prediction entropy when available",
            "Series=pooled and within-image fixed effects when available",
        ),
    )
    _tight_layout(fig)
    _save_fig(fig, out_path)


def plot_dual_force_bars(
    regression: Dict[str, Any],
    out_path: Path,
    *,
    title: str,
    subtitle: Optional[str] = None,
    metadata: Optional[Sequence[str]] = None,
) -> None:
    predictors = regression.get("predictors", {}) if isinstance(regression, dict) else {}
    order = [
        key
        for key in ("complexity_score_residual", "complexity_score", "prompt_complexity_score")
        if key in predictors
    ]
    if not order:
        return

    labels = {
        "complexity_score": "Semantic Logic\n(Compositional Precision)",
        "complexity_score_residual": "Residualized Logic\n(Compositional Precision)",
        "prompt_complexity_score": "Prompt Load\n(Linguistic Anchoring)",
    }
    x = np.arange(len(order))
    centers = []
    lowers = []
    uppers = []
    colors = []
    for key in order:
        center, lower, upper = _standardized_beta_and_ci(predictors[key])
        centers.append(center)
        lowers.append(lower)
        uppers.append(upper)
        colors.append(_COEFFICIENT_COLORS.get(key, "#444444"))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
    bars = ax.bar(x, centers, color=colors, alpha=0.9, width=0.62)
    yerr = np.vstack([
        np.asarray(centers) - np.asarray(lowers),
        np.asarray(uppers) - np.asarray(centers),
    ])
    ax.errorbar(x, centers, yerr=yerr, fmt="none", ecolor="#333333", elinewidth=1.6, capsize=4)
    for idx, bar in enumerate(bars):
        height = float(centers[idx])
        va = "bottom" if height >= 0 else "top"
        y_text = height + (0.03 if height >= 0 else -0.03)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_text,
            f"{height:.2f}",
            ha="center",
            va=va,
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([labels[key] for key in order], fontsize=10)
    ax.set_ylabel("Standardized Coefficient (95% CI)", fontsize=11)
    ax.set_title(title, fontsize=13)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#555555",
        )
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    _apply_style(ax)
    _set_plot_metadata(
        fig,
        metadata
        or _metadata_lines(
            "View=dual-force bar chart",
            "Bars=standardized coefficients",
            "Outcome=internal drift",
            "Forces=semantic logic vs prompt load",
        ),
    )
    _tight_layout(fig)
    _save_fig(fig, out_path)


def plot_coefficient_forest_series(
    regressions: Dict[str, Dict[str, Any]],
    out_path: Path,
    *,
    title: str,
    metadata: Optional[Sequence[str]] = None,
) -> None:
    items = [(name, reg) for name, reg in regressions.items() if isinstance(reg, dict) and reg.get("predictors")]
    if not items:
        return

    nrows = len(items)
    fig, axes = plt.subplots(nrows, 1, figsize=(8.6, max(3.0 * nrows, 4.2)), squeeze=False)
    order = [
        key
        for key in (
            "complexity_score_residual",
            "complexity_score",
            "prompt_complexity_score",
            "option_hardness_score",
            "prediction_entropy",
        )
    ]
    for axis, (name, reg) in zip(axes.ravel(), items):
        predictors = reg.get("predictors", {})
        keys = [key for key in order if key in predictors]
        if not keys:
            axis.axis("off")
            continue
        y_positions = np.arange(len(keys))[::-1]
        axis.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
        for idx, key in enumerate(keys):
            center, lower, upper = _standardized_beta_and_ci(predictors[key])
            color = _COEFFICIENT_COLORS.get(key, "#444444")
            axis.errorbar(
                center,
                y_positions[idx],
                xerr=[[center - lower], [upper - center]],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=2.0,
                capsize=4,
                markersize=7,
                alpha=0.95,
            )
        axis.set_yticks(y_positions)
        axis.set_yticklabels([_COEFFICIENT_LABELS.get(key, key) for key in keys], fontsize=9)
        axis.set_title(name, fontsize=11, loc="left")
        axis.grid(axis="x", alpha=0.2, linewidth=0.6)
        _apply_style(axis)

    axes.ravel()[-1].set_xlabel("Standardized Coefficient (95% CI)", fontsize=11)
    fig.suptitle(title, fontsize=14, y=0.995)
    _set_plot_metadata(
        fig,
        metadata
        or _metadata_lines(
            "View=coefficient plot series",
            "Panels=one perturbation per subplot",
            "Dots=standardized coefficients",
            "Bars=approximate 95% CI",
            "Predictors=residualized semantic logic, prompt load, option hardness, prediction entropy when available",
        ),
    )
    _tight_layout(fig, metadata_bottom=0.07, top=0.97)
    _save_fig(fig, out_path)


def plot_segmentation_granularity(
    summary: Dict[str, Any],
    out_path: Path,
    title: str = "Segmentation Degradation by Hierarchy Level",
) -> None:
    per_model = summary.get("per_model", {})
    if not per_model:
        return

    models = list(per_model)
    levels = _SEG_LEVEL_ORDER
    x = np.arange(len(levels))
    width = 0.8 / max(1, len(models))
    colors = {"sam3": "#d62728", "sam2": "#1f77b4"}

    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    for idx, model_name in enumerate(models):
        values = [per_model[model_name].get(level, {}).get("mean_drop", 0.0) for level in levels]
        offset = (idx - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width=width * 0.92,
            label=model_name.upper(),
            color=colors.get(model_name, f"C{idx}"),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_level_label(level) for level in levels], fontsize=10)
    ax.set_ylabel("Mean mIoU Drop", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    _apply_style(ax)
    _set_plot_metadata(
        fig,
        _metadata_lines(
            "Experiment=6",
            "Bars=mean mIoU drop by hierarchy level",
            "Series=segmentation models",
            "Aggregation=sample average",
        ),
    )
    _tight_layout(fig)
    _save_fig(fig, out_path)


def _exp1_level_perturbation_matrix(detail: Dict[str, Any]) -> Tuple[List[str], List[str], np.ndarray]:
    levels = _ordered_levels(detail.keys())
    perturbations = sorted({name for level in levels for name in detail.get(level, {})})
    matrix = np.full((len(levels), len(perturbations)), np.nan, dtype=float)
    for row, level in enumerate(levels):
        for col, perturbation in enumerate(perturbations):
            stats = detail.get(level, {}).get(perturbation)
            if stats:
                matrix[row, col] = float(stats.get("mean_drop", np.nan))
    return levels, perturbations, matrix


def _exp1_sample_level_matrix(records: List[Dict[str, Any]]) -> Tuple[List[str], List[str], np.ndarray]:
    levels = _LEVEL_ORDER
    sample_ids: List[str] = []
    rows: List[List[float]] = []
    for record in records:
        row: List[float] = []
        for level in levels:
            perturbations = record.get("levels", {}).get(level, {}).get("perturbations", [])
            values = [float(item.get("accuracy_drop", 0.0)) for item in perturbations]
            row.append(float(np.mean(values)) if values else np.nan)
        if not np.all(np.isnan(row)):
            sample_ids.append(str(record.get("image_id")))
            rows.append(row)
    if not rows:
        return [], [], np.empty((0, 0))
    return sample_ids, levels, np.asarray(rows, dtype=float)


def _select_exp1_samples(records: List[Dict[str, Any]], max_samples: int = 3) -> List[Dict[str, Any]]:
    def _mean_drop(record: Dict[str, Any], level: Optional[str] = None) -> float:
        values: List[float] = []
        levels = [level] if level else list(record.get("levels", {}).keys())
        for level_key in levels:
            level_data = record.get("levels", {}).get(level_key, {})
            values.extend(
                float(item.get("accuracy_drop", 0.0))
                for item in level_data.get("perturbations", [])
            )
        return float(np.mean(values)) if values else float("-inf")

    selected: List[Dict[str, Any]] = []
    seen_ids = set()

    for level in ("L1_COARSE", "L4_VERY_FINE"):
        candidates = [record for record in records if _mean_drop(record, level) > float("-inf")]
        candidates.sort(key=lambda record: _mean_drop(record, level), reverse=True)
        for record in candidates:
            image_id = str(record.get("image_id"))
            if image_id in seen_ids:
                continue
            selected.append(record)
            seen_ids.add(image_id)
            break
        if len(selected) >= max_samples:
            return selected[:max_samples]

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for record in records:
        score = _mean_drop(record)
        if score > float("-inf"):
            scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    for _, record in scored:
        image_id = str(record.get("image_id"))
        if image_id in seen_ids:
            continue
        selected.append(record)
        seen_ids.add(image_id)
        if len(selected) >= max_samples:
            break
    return selected[:max_samples]


def _exp1_sample_spectra(record: Dict[str, Any], key: str) -> Dict[str, Dict[str, Sequence[float]]]:
    spectra: Dict[str, Dict[str, Sequence[float]]] = {}
    for level in _ordered_levels(record.get("levels", {}).keys()):
        per_perturbation: Dict[str, Sequence[float]] = {}
        for perturbation in record.get("levels", {}).get(level, {}).get("perturbations", []):
            values = perturbation.get(key)
            if values:
                per_perturbation[str(perturbation.get("name", "unknown"))] = values
        if per_perturbation:
            spectra[level] = per_perturbation
    return spectra


def _sample_perturbation_names(record: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    seen = set()
    for level in _ordered_levels(record.get("levels", {}).keys()):
        for perturbation in record.get("levels", {}).get(level, {}).get("perturbations", []):
            name = str(perturbation.get("name") or perturbation.get("perturbation") or "unknown")
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _exp1_sample_drop_matrix(record: Dict[str, Any]) -> Tuple[List[str], List[str], np.ndarray]:
    return _exp1_sample_metric_matrix(record, "accuracy_drop")


def _exp1_sample_metric_matrix(record: Dict[str, Any], key: str) -> Tuple[List[str], List[str], np.ndarray]:
    levels = _ordered_levels(record.get("levels", {}).keys())
    perturbations = _sample_perturbation_names(record)
    matrix = np.full((len(levels), len(perturbations)), np.nan, dtype=float)
    col_index = {name: idx for idx, name in enumerate(perturbations)}
    for row, level in enumerate(levels):
        for perturbation in record.get("levels", {}).get(level, {}).get("perturbations", []):
            name = str(perturbation.get("name") or perturbation.get("perturbation") or "unknown")
            if name not in col_index:
                continue
            value = perturbation.get(key)
            if value is not None:
                matrix[row, col_index[name]] = float(value)
    return levels, perturbations, matrix


def _exp1_perturbation_family_points(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        image_id = str(record.get("image_id"))
        for level_key, level_data in record.get("levels", {}).items():
            clean_accuracy = 1.0 if level_data.get("clean", {}).get("correct", False) else 0.0
            base = {
                "image_id": image_id,
                "level": level_key,
                "question": level_data.get("question"),
                "question_type": level_data.get("question_type"),
                "complexity_score": float(level_data.get("complexity_score", 0.0) or 0.0),
                "question_complexity_score": float(
                    level_data.get("question_complexity_score", 0.0) or 0.0
                ),
                "prompt_complexity_score": float(
                    level_data.get("prompt_complexity_score", 0.0) or 0.0
                ),
                "option_hardness_score": float(
                    level_data.get("option_hardness_score", 0.0) or 0.0
                ),
                "prediction_entropy": _optional_float(level_data.get("prediction_entropy")),
                "clean_accuracy": clean_accuracy,
            }
            for perturbation in level_data.get("perturbations", []):
                family = str(perturbation.get("family") or "unknown")
                point = dict(base)
                loglik_drift = perturbation.get("loglik_drift")
                drift = float(loglik_drift) if loglik_drift is not None else 0.0
                point.update(
                    {
                        "perturbation_family": family,
                        "perturbation_name": str(
                            perturbation.get("name") or perturbation.get("perturbation") or family
                        ),
                        "accuracy_drop": float(perturbation.get("accuracy_drop", 0.0) or 0.0),
                        "loglik_drift": drift,
                        "loglik_erosion": max(drift, 0.0),
                        "loglik_recovery": min(drift, 0.0),
                        "loglik_volatility": abs(drift),
                    }
                )
                grouped.setdefault(family, []).append(point)
    for points in grouped.values():
        attach_complexity_residual(points)
    return grouped


def _exp2_bandwidth_values(records: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {level: [] for level in _LEVEL_ORDER}
    for record in records:
        for level, level_data in record.get("levels", {}).items():
            bandwidth = level_data.get("bandwidth")
            if bandwidth is not None:
                values.setdefault(level, []).append(float(bandwidth))
    return {level: items for level, items in values.items() if items}


def _exp2_sample_level_matrix(records: List[Dict[str, Any]]) -> Tuple[List[str], List[str], np.ndarray]:
    sample_ids: List[str] = []
    rows: List[List[float]] = []
    for record in records:
        row = []
        for level in _LEVEL_ORDER:
            bandwidth = record.get("levels", {}).get(level, {}).get("bandwidth")
            row.append(float(bandwidth) if bandwidth is not None else np.nan)
        if not np.all(np.isnan(row)):
            sample_ids.append(str(record.get("image_id")))
            rows.append(row)
    if not rows:
        return [], [], np.empty((0, 0))
    return sample_ids, _LEVEL_ORDER, np.asarray(rows, dtype=float)


def _select_exp2_samples(records: List[Dict[str, Any]], max_samples: int = 3) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for record in records:
        values = [float(level_data.get("bandwidth")) for level_data in record.get("levels", {}).values() if level_data.get("bandwidth") is not None]
        if values:
            scored.append((float(np.mean(values)), record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:max_samples]]


def _exp2_sample_overall_wt(record: Optional[Dict[str, Any]]) -> Dict[str, Sequence[float]]:
    if not record:
        return {}
    overlays: Dict[str, Sequence[float]] = {}
    for level in _ordered_levels(record.get("levels", {}).keys()):
        values = record.get("levels", {}).get(level, {}).get("W_t")
        if values:
            overlays[level] = values
    return overlays


def _exp2_sample_wt_series(record: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Sequence[float]]]:
    if not record:
        return {}
    per_level: Dict[str, Dict[str, Sequence[float]]] = {}
    for level in _ordered_levels(record.get("levels", {}).keys()):
        level_data = record.get("levels", {}).get(level, {})
        series: Dict[str, Sequence[float]] = {}
        overall = level_data.get("W_t")
        if overall:
            series["Overall"] = overall
        layer_groups = level_data.get("layer_groups", {})
        for group_name in ("early", "mid", "late"):
            values = layer_groups.get(group_name, {}).get("W_t")
            if values:
                series[group_name.capitalize()] = values
        if series:
            per_level[level] = series
    return per_level


def _exp3_level_perturbation_matrix(records: List[Dict[str, Any]]) -> Tuple[List[str], List[str], np.ndarray]:
    levels = _LEVEL_ORDER
    perturbations = sorted(
        {
            str(item.get("perturbation"))
            for record in records
            for level_data in record.get("levels", {}).values()
            for item in level_data.get("perturbations", [])
        }
    )
    matrix = np.full((len(levels), len(perturbations)), np.nan, dtype=float)
    for row, level in enumerate(levels):
        for col, perturbation in enumerate(perturbations):
            values = []
            for record in records:
                for item in record.get("levels", {}).get(level, {}).get("perturbations", []):
                    if item.get("perturbation") == perturbation:
                        value = item.get("post_drift_scalar_all")
                        if value is not None:
                            values.append(float(value))
            if values:
                matrix[row, col] = float(np.mean(values))
    valid_rows = [idx for idx in range(len(levels)) if not np.all(np.isnan(matrix[idx]))]
    valid_levels = [levels[idx] for idx in valid_rows]
    return valid_levels, perturbations, matrix[valid_rows]


def _exp3_sample_level_matrix(records: List[Dict[str, Any]]) -> Tuple[List[str], List[str], np.ndarray]:
    sample_ids: List[str] = []
    rows: List[List[float]] = []
    for record in records:
        row = []
        for level in _LEVEL_ORDER:
            values = [
                float(item.get("post_drift_scalar_all"))
                for item in record.get("levels", {}).get(level, {}).get("perturbations", [])
                if item.get("post_drift_scalar_all") is not None
            ]
            row.append(float(np.mean(values)) if values else np.nan)
        if not np.all(np.isnan(row)):
            sample_ids.append(str(record.get("image_id")))
            rows.append(row)
    if not rows:
        return [], [], np.empty((0, 0))
    return sample_ids, _LEVEL_ORDER, np.asarray(rows, dtype=float)


def _select_exp3_samples(records: List[Dict[str, Any]], max_samples: int = 2) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for record in records:
        values = []
        for level_data in record.get("levels", {}).values():
            values.extend(
                float(item.get("post_drift_scalar_all"))
                for item in level_data.get("perturbations", [])
                if item.get("post_drift_scalar_all") is not None
            )
        if values:
            scored.append((float(np.mean(values)), record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:max_samples]]


def _exp3_sample_profiles(record: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Sequence[float]]], Dict[str, Sequence[float]]]:
    profiles: Dict[str, Dict[str, Sequence[float]]] = {}
    overlays: Dict[str, Sequence[float]] = {}
    for level in _ordered_levels(record.get("levels", {}).keys()):
        per_perturbation: Dict[str, Sequence[float]] = {}
        overlay: Optional[Sequence[float]] = None
        for item in record.get("levels", {}).get(level, {}).get("perturbations", []):
            values = item.get("pre_drift_bands")
            if values:
                per_perturbation[str(item.get("perturbation", "unknown"))] = values
            if overlay is None:
                overlay = item.get("analysis_groups", {}).get("late", {}).get("W_t")
        if per_perturbation:
            profiles[level] = per_perturbation
        if overlay:
            overlays[level] = overlay
    return profiles, overlays


def _exp4_sample_cutoffs(
    records: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> Dict[str, Tuple[List[str], List[str], np.ndarray]]:
    results: Dict[str, Tuple[List[str], List[str], np.ndarray]] = {}
    for mode in ("lowpass", "highpass"):
        sample_ids: List[str] = []
        rows: List[List[float]] = []
        for record in records:
            mode_data = record.get("modes", {}).get(mode)
            if not mode_data:
                continue
            cutoffs = mode_data.get("cutoffs", [])
            row = []
            for level in _LEVEL_ORDER:
                correct = mode_data.get("levels", {}).get(level, {}).get("correct_at_cutoff", [])
                if cutoffs and correct:
                    accuracy = [1.0 if value else 0.0 for value in correct]
                    row.append(float(compute_critical_cutoff(accuracy, cutoffs, threshold=threshold)))
                else:
                    row.append(np.nan)
            if not np.all(np.isnan(row)):
                sample_ids.append(str(record.get("image_id")))
                rows.append(row)
        if rows:
            results[mode] = (sample_ids, _LEVEL_ORDER, np.asarray(rows, dtype=float))
    return results


def _select_exp4_samples(records: List[Dict[str, Any]], max_samples: int = 3) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for record in records:
        values = []
        for mode_data in record.get("modes", {}).values():
            cutoffs = mode_data.get("cutoffs", [])
            for level_data in mode_data.get("levels", {}).values():
                correct = level_data.get("correct_at_cutoff", [])
                if cutoffs and correct:
                    accuracy = [1.0 if item else 0.0 for item in correct]
                    values.append(float(compute_critical_cutoff(accuracy, cutoffs)))
        if values:
            scored.append((float(np.mean(values)), record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:max_samples]]


def _exp5_heatmap_matrix(grouped_pairs: List[Dict[str, Any]], key: str) -> Tuple[List[str], List[str], np.ndarray]:
    levels = _ordered_levels({pair.get("level") for pair in grouped_pairs if pair.get("level")})
    perturbations = sorted({str(pair.get("perturbation")) for pair in grouped_pairs if pair.get("perturbation")})
    matrix = np.full((len(levels), len(perturbations)), np.nan, dtype=float)
    for row, level in enumerate(levels):
        for col, perturbation in enumerate(perturbations):
            for pair in grouped_pairs:
                if pair.get("level") == level and pair.get("perturbation") == perturbation:
                    matrix[row, col] = float(pair.get(key, np.nan))
                    break
    return levels, perturbations, matrix


def _exp6_level_perturbation_matrix(
    records: List[Dict[str, Any]],
    model_name: str,
) -> Tuple[List[str], List[str], np.ndarray]:
    levels = _SEG_LEVEL_ORDER
    perturbations = sorted(
        {
            str(perturbation.get("perturbation"))
            for record in records
            for level_data in record.get("models", {}).get(model_name, {}).get("levels", {}).values()
            for perturbation in level_data.get("perturbations", [])
        }
    )
    matrix = np.full((len(levels), len(perturbations)), np.nan, dtype=float)
    for row, level in enumerate(levels):
        for col, perturbation in enumerate(perturbations):
            values = []
            for record in records:
                items = record.get("models", {}).get(model_name, {}).get("levels", {}).get(level, {}).get("perturbations", [])
                for item in items:
                    if item.get("perturbation") == perturbation:
                        values.append(float(item.get("miou_drop", 0.0)))
            if values:
                matrix[row, col] = float(np.mean(values))
    valid_rows = [idx for idx in range(len(levels)) if not np.all(np.isnan(matrix[idx]))]
    return [levels[idx] for idx in valid_rows], perturbations, matrix[valid_rows]


def _exp6_sample_level_matrix(
    records: List[Dict[str, Any]],
    model_name: str,
) -> Tuple[List[str], List[str], np.ndarray]:
    sample_ids: List[str] = []
    rows: List[List[float]] = []
    for record in records:
        row = []
        for level in _SEG_LEVEL_ORDER:
            values = [
                float(item.get("miou_drop", 0.0))
                for item in record.get("models", {}).get(model_name, {}).get("levels", {}).get(level, {}).get("perturbations", [])
            ]
            row.append(float(np.mean(values)) if values else np.nan)
        if not np.all(np.isnan(row)):
            sample_ids.append(str(record.get("image_id")))
            rows.append(row)
    if not rows:
        return [], [], np.empty((0, 0))
    return sample_ids, _SEG_LEVEL_ORDER, np.asarray(rows, dtype=float)


def _plot_exp2_level_band_heatmap(
    summary: Dict[str, Any],
    out_path: Path,
    suppress_dc: bool = _DEFAULT_SUPPRESS_DC,
) -> None:
    per_level = summary.get("per_level", {})
    levels = _ordered_levels(per_level.keys())
    if not levels:
        return
    matrix = np.asarray([per_level[level].get("W_t_average", []) for level in levels], dtype=float)
    if matrix.size == 0:
        return
    _plot_heatmap(
        matrix,
        [_level_label(level) for level in levels],
        _band_labels(matrix.shape[1], suppress_dc=suppress_dc),
        out_path,
        "Attention Filter W_t by Level and Band",
        "W_t(omega)",
        cmap="magma",
    )


def _plot_exp2_group_spectra(
    summary: Dict[str, Any],
    out_path: Path,
    suppress_dc: bool = _DEFAULT_SUPPRESS_DC,
) -> None:
    per_level = summary.get("per_level", {})
    levels = _ordered_levels(per_level.keys())
    if not levels:
        return

    group_order = summary.get("layer_groups", {}).get("order", ["early", "mid", "late"])
    fig, axes = plt.subplots(1, len(group_order), figsize=(5.0 * len(group_order), 4.8), sharey=True)
    axes_arr = np.atleast_1d(axes)

    for axis, group_name in zip(axes_arr, group_order):
        for level in levels:
            values = per_level[level].get("layer_groups", {}).get(group_name, {}).get("W_t_average", [])
            if not values:
                continue
            x_values = _spectral_x_positions(len(values), suppress_dc=suppress_dc)
            axis.plot(
                x_values,
                _spectral_plot_values(values),
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                color=_LEVEL_COLORS.get(level, "#999"),
                label=_level_label(level),
            )
        axis.set_title(group_name.capitalize(), fontsize=12)
        axis.set_xlabel("Normalized Frequency", fontsize=10)
        axis.grid(alpha=0.2, linewidth=0.6)
        _apply_spectral_axis_scale(axis)
        _apply_style(axis)
    axes_arr[0].set_ylabel("W_t(omega)", fontsize=10)

    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle("Layer-Group Attention Spectra", fontsize=14, y=0.99)
    _set_plot_metadata(
        fig,
        _metadata_lines(
            "Experiment=2",
            "Panels=early/mid/late layer groups",
            "Lines=mean W_t per level",
            "Aggregation=sample average",
        ),
    )
    _tight_layout(fig, metadata_bottom=0.07, top=0.96)
    _save_fig(fig, out_path)


def _plot_exp2_group_bandwidth(summary: Dict[str, Any], out_path: Path) -> None:
    per_level = summary.get("per_level", {})
    levels = _ordered_levels(per_level.keys())
    if not levels:
        return

    group_order = summary.get("layer_groups", {}).get("order", ["early", "mid", "late"])
    series = {
        group_name.capitalize(): [
            per_level[level].get("layer_groups", {}).get(group_name, {}).get("mean_bandwidth", np.nan)
            for level in levels
        ]
        for group_name in group_order
    }
    _plot_grouped_bars(
        series,
        [_level_label(level) for level in levels],
        out_path,
        "Layer-Group Effective Bandwidth",
        "Bandwidth",
        colors=["#9ecae1", "#fdae6b", "#9e9ac8"][: len(series)],
        metadata=_plot_metadata(
            experiment="2",
            what="Mean effective bandwidth by decoder layer group",
            aggregation="sample average by level",
            x="task level",
            y="effective bandwidth G(t)",
            note="Series compare early, mid, and late layer groups",
        ),
    )


def _plot_exp2_selected_samples(
    records: List[Dict[str, Any]],
    out_dir: Path,
    max_samples: int = 3,
    preferred_sample_ids: Optional[Sequence[str]] = None,
    profile: str = _DEFAULT_PLOT_PROFILE,
    suppress_dc: bool = _DEFAULT_SUPPRESS_DC,
) -> None:
    selected = _select_records_by_preferred_ids(
        records,
        preferred_sample_ids,
        max_samples=max_samples,
        fallback_selector=_select_exp2_samples,
    )
    if not selected:
        return

    ncols = min(2, len(selected))
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 4.5 * nrows), sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for axis in axes_arr.ravel()[len(selected):]:
        axis.axis("off")

    for axis, record in zip(axes_arr.ravel(), selected):
        sample_id = str(record.get("image_id"))
        for level in _ordered_levels(record.get("levels", {}).keys()):
            values = record.get("levels", {}).get(level, {}).get("W_t")
            if not values:
                continue
            x_values = _spectral_x_positions(len(values), suppress_dc=suppress_dc)
            axis.plot(
                x_values,
                _spectral_plot_values(values),
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                color=_LEVEL_COLORS.get(level, "#999"),
                label=_level_label(level),
            )
        axis.set_title(_short_sample_id(sample_id, 28), fontsize=11)
        axis.set_xlabel("Normalized Frequency", fontsize=10)
        axis.set_ylabel("W_t(omega)", fontsize=10)
        axis.grid(alpha=0.2, linewidth=0.6)
        _apply_spectral_axis_scale(axis)
        _apply_style(axis)

    handles, labels = axes_arr.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle("Representative Sample Attention Spectra", fontsize=14, y=0.99)
    _set_plot_metadata(
        fig,
        _plot_metadata(
            experiment="2",
            what="Representative per-sample task filters W_t",
            aggregation="one panel per selected image",
            x="radial frequency band (low→high)",
            y="W_t(omega)",
            selection="preferred Exp 1 representative images when available; otherwise top images by mean bandwidth",
            profile=profile,
        ),
    )
    _tight_layout(fig, metadata_bottom=0.07, top=0.97)
    _save_fig(fig, out_dir / "exp2_sample_spectra.png")


def _plot_exp2_matched_sample_group_comparison(
    record: Dict[str, Any],
    out_path: Path,
    profile: str = _DEFAULT_PLOT_PROFILE,
    suppress_dc: bool = _DEFAULT_SUPPRESS_DC,
) -> None:
    levels = _ordered_levels(record.get("levels", {}).keys())
    if not levels:
        return

    sample_id = str(record.get("image_id"))
    group_order = list(_FILTER_ANALYSIS_ORDER)
    present_groups = []
    for group_name in group_order:
        has_group = False
        for level in levels:
            level_data = record.get("levels", {}).get(level, {})
            values = level_data.get("W_t") if group_name == "overall" else level_data.get("layer_groups", {}).get(group_name, {}).get("W_t")
            if values:
                has_group = True
                break
        if has_group:
            present_groups.append(group_name)
    if not present_groups:
        return

    ncols = 2
    nrows = int(np.ceil(len(present_groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.3 * nrows), sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    for axis in axes_arr.ravel()[len(present_groups):]:
        axis.axis("off")

    for axis, group_name in zip(axes_arr.ravel(), present_groups):
        for level in levels:
            level_data = record.get("levels", {}).get(level, {})
            values = level_data.get("W_t") if group_name == "overall" else level_data.get("layer_groups", {}).get(group_name, {}).get("W_t")
            if not values:
                continue
            x_values = _spectral_x_positions(len(values), suppress_dc=suppress_dc)
            axis.plot(
                x_values,
                _spectral_plot_values(values),
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                color=_LEVEL_COLORS.get(level, "#999"),
                label=_level_label(level),
            )
        axis.set_title(group_name.capitalize(), fontsize=11)
        axis.set_xlabel("Normalized Frequency", fontsize=10)
        axis.set_ylabel("W_t(omega)", fontsize=10)
        axis.grid(alpha=0.2, linewidth=0.6)
        _apply_spectral_axis_scale(axis)
        _apply_style(axis)

    handles, labels = axes_arr.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle(f"Matched Sample W_t by Filter Group: {_short_sample_id(sample_id, 32)}", fontsize=14, y=0.995)
    _set_plot_metadata(
        fig,
        _plot_metadata(
            experiment="2",
            what="Matched sample W_t comparison across filter groups",
            aggregation="one panel per filter group; lines compare task levels for the same image",
            x="frequency band (low→high)",
            y="W_t(omega)",
            selection="same representative high-drop image used for Exp 1 sample plots",
            note="Panels are overall, early, mid, and late when available",
            profile=profile,
        ),
    )
    _tight_layout(fig, metadata_bottom=0.07, top=0.97)
    _save_fig(fig, out_path)


def _plot_exp2_control_bandwidth(summary: Dict[str, Any], out_path: Path) -> None:
    per_level = summary.get("per_level", {})
    levels = _ordered_levels(per_level.keys())
    if not levels:
        return
    control_names = summary.get("prompt_controls", [])
    if not control_names:
        return

    series: Dict[str, List[float]] = {
        "Task prompt": [per_level[level].get("mean_bandwidth", np.nan) for level in levels]
    }
    colors = [_EXP2_CONTROL_COLORS["task"]]
    for control_name in control_names:
        if not any(control_name in per_level[level].get("controls", {}) for level in levels):
            continue
        series[_exp2_control_label(control_name)] = [
            per_level[level].get("controls", {}).get(control_name, {}).get("mean_bandwidth", np.nan)
            for level in levels
        ]
        colors.append(_EXP2_CONTROL_COLORS.get(control_name, None))

    _plot_grouped_bars(
        series,
        [_level_label(level) for level in levels],
        out_path,
        "Task vs Control Bandwidth",
        "Bandwidth",
        colors=colors,
        metadata=_plot_metadata(
            experiment="2",
            what="Task-prompt vs control-prompt effective bandwidth",
            aggregation="sample average by level",
            x="task level",
            y="effective bandwidth G(t)",
            note="Controls test whether language meaning changes the inferred task filter",
        ),
    )


def _plot_exp2_control_divergence(summary: Dict[str, Any], out_path: Path) -> None:
    per_level = summary.get("per_level", {})
    levels = _ordered_levels(per_level.keys())
    if not levels:
        return
    control_names = summary.get("prompt_controls", [])
    if not control_names:
        return

    series: Dict[str, List[float]] = {}
    colors: List[str] = []
    for control_name in control_names:
        if not any(control_name in per_level[level].get("controls", {}) for level in levels):
            continue
        series[_exp2_control_label(control_name)] = [
            per_level[level].get("controls", {}).get(control_name, {}).get("mean_js_divergence_to_task", np.nan)
            for level in levels
        ]
        colors.append(_EXP2_CONTROL_COLORS.get(control_name, None))

    if not series:
        return
    _plot_grouped_bars(
        series,
        [_level_label(level) for level in levels],
        out_path,
        "Task vs Control Filter Divergence",
        "Mean JS Divergence",
        colors=colors,
        metadata=_plot_metadata(
            experiment="2",
            what="Distance between task-prompt and control-prompt filters",
            aggregation="sample average by level",
            x="task level",
            y="mean Jensen-Shannon divergence",
            note="Higher values mean the control prompt produces a more different W_t",
        ),
    )


def _plot_exp2_control_spectra(
    summary: Dict[str, Any],
    out_dir: Path,
    suppress_dc: bool = _DEFAULT_SUPPRESS_DC,
) -> None:
    per_level = summary.get("per_level", {})
    levels = _ordered_levels(per_level.keys())
    control_names = summary.get("prompt_controls", [])
    if not levels or not control_names:
        return

    for control_name in control_names:
        if not any(control_name in per_level[level].get("controls", {}) for level in levels):
            continue
        fig, ax = plt.subplots(figsize=(8.8, 5.6))
        for level in levels:
            task_values = per_level[level].get("W_t_average", [])
            control_values = per_level[level].get("controls", {}).get(control_name, {}).get("W_t_average", [])
            if task_values:
                x_values = _spectral_x_positions(len(task_values), suppress_dc=suppress_dc)
                ax.plot(
                    x_values,
                    _spectral_plot_values(task_values),
                    linewidth=2.0,
                    color=_LEVEL_COLORS.get(level, "#999"),
                    label=f"{_level_label(level)} task",
                )
            if control_values:
                x_values = _spectral_x_positions(len(control_values), suppress_dc=suppress_dc)
                ax.plot(
                    x_values,
                    _spectral_plot_values(control_values),
                    linewidth=1.8,
                    linestyle="--",
                    color=_LEVEL_COLORS.get(level, "#999"),
                    alpha=0.75,
                    label=f"{_level_label(level)} control",
                )
        ax.set_xlabel("Normalized Frequency", fontsize=11)
        ax.set_ylabel("W_t(omega)", fontsize=11)
        ax.set_title(f"Task vs {_exp2_control_label(control_name)} Spectra", fontsize=13)
        ax.grid(alpha=0.2, linewidth=0.6)
        _apply_spectral_axis_scale(ax)
        _apply_style(ax)
        ax.legend(fontsize=8, ncol=2)
        _set_plot_metadata(
            fig,
            _metadata_lines(
                "Experiment=2",
                f"Control={control_name}",
                "Solid=task prompt; dashed=control prompt",
                "Lines are level-wise mean W_t",
            ),
        )
        _tight_layout(fig)
        _save_fig(fig, out_dir / f"exp2_control_spectra_{control_name}.png")


def _plot_exp3_pre_post(summary: Dict[str, Any], out_path: Path) -> None:
    per_level = summary.get("per_level", {})
    levels = _ordered_levels(per_level.keys())
    if not levels:
        return
    series = {
        "Pre-fusion": [per_level[level].get("mean_pre_drift", 0.0) for level in levels],
        "Post-fusion (all tokens)": [per_level[level].get("mean_post_drift_all", 0.0) for level in levels],
    }
    if any(per_level[level].get("mean_post_drift_vision") is not None for level in levels):
        series["Post-fusion (vision only)"] = [
            per_level[level].get("mean_post_drift_vision", np.nan) for level in levels
        ]
    _plot_grouped_bars(
        series,
        [_level_label(level) for level in levels],
        out_path,
        "Pre-Fusion Drift vs Post-Fusion Response by Level",
        "Mean Drift",
        colors=["#6baed6", "#fb6a4a", "#9e9ac8"][: len(series)],
        metadata=_metadata_lines(
            "Experiment=3",
            "Bars=level-wise mean drift",
            "Pre=vision-only controlled input",
            "Post(all)=task-conditioned multimodal response",
        ),
    )


def _plot_exp3_group_weighted_amplification(summary: Dict[str, Any], out_path: Path) -> None:
    per_level = summary.get("per_level", {})
    levels = _ordered_levels(per_level.keys())
    if not levels:
        return

    group_order = summary.get("analysis_groups", {}).get("order", _FILTER_ANALYSIS_ORDER)
    series = {
        group_name.capitalize(): [
            per_level[level].get("analysis_groups", {}).get(group_name, {}).get("mean_overlap_score", np.nan)
            for level in levels
        ]
        for group_name in group_order
        if any(
            group_name in per_level[level].get("analysis_groups", {})
            for level in levels
        )
    }
    _plot_grouped_bars(
        series,
        [_level_label(level) for level in levels],
        out_path,
        "Mean Spectral Overlap by Filter Group",
        "Mean Overlap Score",
        colors=_FILTER_ANALYSIS_COLORS[: len(series)],
        metadata=_metadata_lines(
            "Experiment=3",
            "Bars=mean overlap ∫W_t·ΔV by level",
            "Series=filter groups overall/early/mid/late",
            "Aggregation=sample average",
        ),
    )


def _plot_exp3_group_correlations(tests: Dict[str, Any], out_path: Path) -> None:
    levels = _LEVEL_ORDER
    group_order = tests.get("analysis_groups", {}).get("order", _FILTER_ANALYSIS_ORDER)
    group_order = [group_name for group_name in group_order if group_name]
    correlation_map = tests.get("pearson_post_response_vs_overlap_by_level", {})
    series = {}
    for group_name in group_order:
        values = []
        has_value = False
        for level in levels:
            test = correlation_map.get(level, {}).get(group_name)
            if test is None:
                values.append(np.nan)
                continue
            has_value = True
            values.append(test.get("controlled_r", np.nan))
        if has_value:
            series[group_name.capitalize()] = values
    if not series:
        return
    _plot_grouped_bars(
        series,
        [_level_label(level) for level in levels],
        out_path,
        "Controlled Response vs Overlap Correlation by Filter Group",
        "Pearson r",
        colors=_FILTER_ANALYSIS_COLORS[: len(series)],
        ylim=(-1.0, 1.0),
        metadata=_metadata_lines(
            "Experiment=3",
            "Bars=controlled Pearson r(ΔZ_all, ∫W_t·ΔV)",
            "Within-profile centering holds pre-drift profile approximately fixed",
            "Series=filter groups",
        ),
    )


def _plot_exp3_profile_groups(summary: Dict[str, Any], out_path: Path) -> None:
    grouping = summary.get("profile_grouping", {}).get("groups", {})
    if not grouping:
        return

    group_names = sorted(grouping.keys(), key=lambda name: int(str(name).split("_")[-1]))
    levels = _ordered_levels({
        level
        for group_data in grouping.values()
        for level in group_data.get("per_level", {})
    })
    if not levels:
        return

    matrix = np.full((len(group_names), len(levels)), np.nan, dtype=float)
    for row, group_name in enumerate(group_names):
        group_data = grouping.get(group_name, {})
        per_level = group_data.get("per_level", {})
        for col, level in enumerate(levels):
            value = per_level.get(level, {}).get("mean_post_drift_scalar_all")
            if value is not None:
                matrix[row, col] = float(value)

    _plot_heatmap(
        matrix,
        [group_name.replace("_", " ").title() for group_name in group_names],
        [_level_label(level) for level in levels],
        out_path,
        "Post-Fusion Response by Pre-Drift Profile Group",
        "Mean Post-Fusion Drift (all tokens)",
        cmap="YlOrRd",
        metadata=_metadata_lines(
            "Experiment=3",
            "Rows=matched pre-drift profile groups",
            "Cols=task levels",
            "Value=mean post-fusion drift on all tokens",
        ),
    )


def _plot_exp3_response_amplification(summary: Dict[str, Any], out_path: Path) -> None:
    per_level = summary.get("per_level", {})
    levels = _ordered_levels(per_level.keys())
    if not levels:
        return
    _plot_grouped_bars(
        {
            "Response amplification": [
                per_level[level].get("mean_response_amplification", np.nan) for level in levels
            ]
        },
        [_level_label(level) for level in levels],
        out_path,
        "Mean Post-Fusion Response Amplification by Level",
        "post_drift_all / pre_drift",
        colors=["#7b3294"],
        metadata=_metadata_lines(
            "Experiment=3",
            "Bars=mean scalar response amplification",
            "Response=all-token post-fusion drift",
            "Baseline=pre-fusion vision drift",
        ),
    )


def _plot_exp4_critical_cutoffs(summary: Dict[str, Any], out_path: Path) -> None:
    per_mode = summary.get("per_mode", {})
    levels = _ordered_levels({level for mode_data in per_mode.values() for level in mode_data})
    if not levels:
        return
    series = {
        mode.capitalize(): [per_mode.get(mode, {}).get(level, {}).get("critical_cutoff", np.nan) for level in levels]
        for mode in ("lowpass", "highpass")
        if mode in per_mode
    }
    _plot_grouped_bars(
        series,
        [_level_label(level) for level in levels],
        out_path,
        "Critical Cutoff by Level",
        "Critical Cutoff",
        colors=["#3182bd", "#e6550d"],
        ylim=(0.0, 0.55),
        metadata=_metadata_lines(
            "Experiment=4",
            "Bars=critical cutoff ω* by level",
            "Series=lowpass and highpass sweeps",
            "Aggregation=sample average threshold crossing",
        ),
    )


def _plot_exp4_sample_curves(
    records: List[Dict[str, Any]],
    out_dir: Path,
    max_samples: int = 3,
    preferred_sample_ids: Optional[Sequence[str]] = None,
    profile: str = _DEFAULT_PLOT_PROFILE,
) -> None:
    for mode in ("lowpass", "highpass"):
        selected = _select_records_by_preferred_ids(
            records,
            preferred_sample_ids,
            max_samples=max_samples,
            fallback_selector=_select_exp4_samples,
        )
        if not selected:
            continue
        ncols = min(2, len(selected))
        nrows = int(np.ceil(len(selected) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 4.2 * nrows), sharex=True, sharey=True)
        axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
        for axis in axes_arr.ravel()[len(selected):]:
            axis.axis("off")

        for axis, record in zip(axes_arr.ravel(), selected):
            mode_data = record.get("modes", {}).get(mode)
            if not mode_data:
                axis.axis("off")
                continue
            cutoffs = mode_data.get("cutoffs", [])
            for level in _ordered_levels(mode_data.get("levels", {}).keys()):
                correct = mode_data.get("levels", {}).get(level, {}).get("correct_at_cutoff", [])
                if not cutoffs or not correct:
                    continue
                accuracy = [1.0 if item else 0.0 for item in correct]
                axis.plot(
                    cutoffs[: len(accuracy)],
                    accuracy,
                    marker="o",
                    linewidth=1.8,
                    markersize=3.8,
                    color=_LEVEL_COLORS.get(level, "#999"),
                    label=_level_label(level),
                )
            axis.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
            axis.set_title(_short_sample_id(str(record.get("image_id")), 28), fontsize=11)
            axis.set_xlabel("Cutoff", fontsize=10)
            axis.set_ylabel("Accuracy", fontsize=10)
            axis.set_ylim(-0.05, 1.05)
            _apply_style(axis)

        handles, labels = axes_arr.ravel()[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
        fig.suptitle(f"Sample Accuracy Curves ({mode})", fontsize=14, y=0.99)
        _set_plot_metadata(
            fig,
            _plot_metadata(
                experiment="4",
                what=f"Representative accuracy-vs-cutoff curves ({mode})",
                aggregation="one panel per selected image",
                x=f"{mode} cutoff",
                y="binary accuracy across cutoff sweep",
                selection="preferred Exp 1 representative images when available; otherwise top images by mean critical cutoff",
                profile=profile,
            ),
        )
        _tight_layout(fig, metadata_bottom=0.07, top=0.97)
        _save_fig(fig, out_dir / f"exp4_sample_curves_{mode}.png")


def _plot_exp5_heatmaps(
    grouped_pairs: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    actual_label: str = "Observed Drop",
    metadata: Optional[Sequence[str]] = None,
) -> None:
    levels, perturbations, predicted = _exp5_heatmap_matrix(grouped_pairs, "predicted")
    _, _, actual = _exp5_heatmap_matrix(grouped_pairs, "actual")
    if predicted.size == 0 or actual.size == 0:
        return

    residual = actual - predicted
    fig, axes = plt.subplots(1, 3, figsize=(max(15.0, 6.0 + 0.7 * len(perturbations)), 5.3))
    panels = [
        (predicted, "Predicted Overlap", "magma"),
        (actual, actual_label, "viridis"),
        (residual, "Residual (actual - predicted)", "coolwarm"),
    ]

    for axis, (matrix, panel_title, cmap) in zip(axes, panels):
        masked = np.ma.masked_invalid(matrix)
        im = axis.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")
        axis.set_xticks(np.arange(len(perturbations)))
        axis.set_xticklabels(perturbations, rotation=40, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(levels)))
        axis.set_yticklabels([_level_label(level) for level in levels], fontsize=9)
        axis.set_title(panel_title, fontsize=11)
        _apply_style(axis)
        fig.colorbar(im, ax=axis, shrink=0.8)

    fig.suptitle(title, fontsize=14)
    _set_plot_metadata(
        fig,
        metadata
        or _metadata_lines(
            "Experiment=5",
            "Panels=predicted overlap, observed target, residual",
            "Rows=levels, cols=perturbations",
            f"Observed={actual_label}",
        ),
    )
    _tight_layout(fig)
    _save_fig(fig, out_path)


def _plot_exp5_level_perturbation_comparison(
    grouped_pairs: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    *,
    actual_label: str = "Observed Accuracy Drop",
    metadata: Optional[Sequence[str]] = None,
) -> None:
    levels, perturbations, predicted = _exp5_heatmap_matrix(grouped_pairs, "predicted")
    _, _, actual = _exp5_heatmap_matrix(grouped_pairs, "actual")
    if predicted.size == 0 or actual.size == 0 or not levels or not perturbations:
        return

    n_panels = len(levels)
    ncols = min(3, max(1, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(14.0, 5.2 * ncols), max(4.2, 3.8 * nrows)),
        squeeze=False,
    )
    axes_arr = axes.ravel()

    x = np.arange(len(perturbations), dtype=float)
    width = 0.38
    pred_color = "#6a51a3"
    actual_color = "#31a354"
    legend_handles: List[Any] = []
    legend_labels: List[str] = []

    for axis, level, pred_row, act_row in zip(axes_arr, levels, predicted, actual):
        pred_vals = np.asarray(pred_row, dtype=float)
        act_vals = np.asarray(act_row, dtype=float)
        pred_axis = axis.twinx()
        pred_bars = pred_axis.bar(
            x - width / 2.0,
            pred_vals,
            width=width,
            color=pred_color,
            alpha=0.88,
            label="Predicted",
        )
        actual_bars = axis.bar(
            x + width / 2.0,
            act_vals,
            width=width,
            color=actual_color,
            alpha=0.82,
            label="Observed",
        )
        axis.set_title(_level_label(level), fontsize=11)
        axis.set_xticks(x)
        axis.set_xticklabels(perturbations, rotation=45, ha="right", fontsize=8)
        axis.set_ylabel(actual_label, fontsize=9, color=actual_color)
        pred_axis.set_ylabel("Predicted Sensitivity", fontsize=9, color=pred_color)
        axis.tick_params(axis="y", colors=actual_color)
        pred_axis.tick_params(axis="y", colors=pred_color)
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
        _apply_style(axis)
        pred_axis.spines["top"].set_visible(False)
        pred_axis.spines["left"].set_visible(False)
        pred_axis.spines["right"].set_color(pred_color)
        pred_axis.grid(False)
        if not legend_handles:
            legend_handles = [pred_bars.patches[0], actual_bars.patches[0]]
            legend_labels = ["Predicted", "Observed"]

    for axis in axes_arr[len(levels):]:
        axis.axis("off")

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle(title, fontsize=14, y=0.99)
    _set_plot_metadata(
        fig,
        metadata
        or _metadata_lines(
            "Experiment=5",
            "View=level-wise perturbation comparison",
            "Panels=one level per subplot",
            "Bars=predicted overlap and observed target by perturbation",
            "Axes=left observed scale, right predicted scale",
        ),
    )
    _tight_layout(fig, metadata_bottom=0.07, top=0.95)
    _save_fig(fig, out_path)


def _plot_exp5_wordy_control_comparison(
    grouped_pairs: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    *,
    actual_label: str = "Observed Accuracy Drop",
    metadata: Optional[Sequence[str]] = None,
) -> None:
    if not grouped_pairs:
        return

    predicted_by_level: Dict[str, List[float]] = defaultdict(list)
    actual_by_level: Dict[str, List[float]] = defaultdict(list)
    for pair in grouped_pairs:
        level = str(pair.get("level", ""))
        if not level:
            continue
        predicted = pair.get("predicted")
        actual = pair.get("actual")
        if predicted is not None and np.isfinite(float(predicted)):
            predicted_by_level[level].append(float(predicted))
        if actual is not None and np.isfinite(float(actual)):
            actual_by_level[level].append(float(actual))

    predicted_values = {level: float(np.mean(values)) for level, values in predicted_by_level.items() if values}
    actual_values = {level: float(np.mean(values)) for level, values in actual_by_level.items() if values}
    if not predicted_values and not actual_values:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), squeeze=False)
    axes_arr = axes.ravel()
    panels = [
        ("Predicted Sensitivity", predicted_values, "Predicted Sensitivity"),
        (actual_label, actual_values, actual_label),
    ]
    for axis, (panel_title, values_by_level, ylabel_panel) in zip(axes_arr, panels):
        pair_labels: List[str] = []
        base_values: List[float] = []
        control_values: List[float] = []
        for base_level, control_level in WORDY_CONTROL_LEVEL_NAME_PAIRS:
            if base_level not in values_by_level or control_level not in values_by_level:
                continue
            pair_labels.append(_WORDY_PAIR_LABELS.get(base_level, f"{base_level} vs {control_level}"))
            base_values.append(float(values_by_level[base_level]))
            control_values.append(float(values_by_level[control_level]))
        if not pair_labels:
            axis.axis("off")
            continue
        x = np.arange(len(pair_labels))
        width = 0.38
        axis.bar(x - width / 2, base_values, width=width, color="#4c78a8", alpha=0.88, label="Base")
        axis.bar(x + width / 2, control_values, width=width, color="#8c6bb1", alpha=0.84, label="Wordy Control")
        axis.set_xticks(x)
        axis.set_xticklabels(pair_labels, fontsize=9)
        axis.set_title(panel_title, fontsize=11)
        axis.set_ylabel(ylabel_panel, fontsize=10)
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
        _apply_style(axis)

    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle(title, fontsize=14, y=0.99)
    _set_plot_metadata(
        fig,
        metadata
        or _metadata_lines(
            "Experiment=5",
            "View=wordy-control pair comparison",
            "Panels=predicted sensitivity and observed target",
            "Bars=base level vs matched wordy control",
        ),
    )
    _tight_layout(fig, metadata_bottom=0.07, top=0.95)
    _save_fig(fig, out_path)


def _plot_exp5_per_level_corr(summary: Dict[str, Any], out_path: Path) -> None:
    image_corr = summary.get("image_space", {}).get("primary_group_summary", {}).get("per_level_correlation", {})
    vision_corr = summary.get("vision_feature_space", {}).get("primary_group_summary", {}).get("per_level_correlation", {})
    levels = _ordered_levels(set(image_corr) | set(vision_corr))
    if not levels:
        return
    _plot_grouped_bars(
        {
            "Image-space": [image_corr.get(level, {}).get("pearson_r", np.nan) for level in levels],
            "Vision-feature": [vision_corr.get(level, {}).get("pearson_r", np.nan) for level in levels],
        },
        [_level_label(level) for level in levels],
        out_path,
        "Per-Level Overlap Correlation",
        "Pearson r",
        colors=["#6baed6", "#fd8d3c"],
        ylim=(-1.0, 1.0),
        metadata=_metadata_lines(
            "Experiment=5",
            f"Group={summary.get('primary_group', 'late')}",
            "Target=accuracy drop",
            "Bars=grouped Pearson r by level",
        ),
    )


def _plot_exp5_group_correlation(summary: Dict[str, Any], out_path: Path) -> None:
    sources = ("image_space", "vision_feature_space")
    group_order = summary.get("analysis_groups", _FILTER_ANALYSIS_ORDER)
    group_order = [group_name for group_name in group_order if group_name]
    series = {}
    for source_name in sources:
        source_summary = summary.get(source_name, {})
        grouped = source_summary.get("grouped_pearson_by_group", {})
        if not grouped:
            continue
        label = _exp5_source_label(source_name)
        series[label] = [grouped.get(group_name, np.nan) for group_name in group_order]
    if not series:
        return
    _plot_grouped_bars(
        series,
        [group_name.capitalize() for group_name in group_order],
        out_path,
        "Grouped Pearson Correlation by Filter Group",
        "Pearson r",
        colors=["#6baed6", "#fd8d3c"],
        ylim=(-1.0, 1.0),
        metadata=_metadata_lines(
            "Experiment=5",
            "Bars=grouped Pearson r(predicted, actual)",
            "Sources=image-space relative and vision-feature relative",
            "Target=accuracy drop",
        ),
    )


def _plot_exp5_target_correlation(summary: Dict[str, Any], out_path: Path) -> None:
    target_order = summary.get(
        "targets",
        ["accuracy_drop", "loglik_erosion", "net_drop", "relative_accuracy_drop"],
    )
    series = {}
    for source_name in ("image_space", "vision_feature_space"):
        source_summary = summary.get(source_name, {})
        primary_target_summaries = source_summary.get("primary_group_target_summaries", {})
        if not primary_target_summaries:
            continue
        label = _exp5_source_label(source_name)
        series[label] = [
            primary_target_summaries.get(target_name, {}).get("pearson_r_grouped", np.nan)
            for target_name in target_order
        ]
    if not series:
        return
    _plot_grouped_bars(
        series,
        [_exp5_target_label(target_name) for target_name in target_order],
        out_path,
        "Primary-Group Pearson Correlation by Target",
        "Pearson r",
        colors=["#6baed6", "#fd8d3c"],
        ylim=(-1.0, 1.0),
        metadata=_metadata_lines(
            "Experiment=5",
            f"Group={summary.get('primary_group', 'late')}",
            "Bars=grouped Pearson r by observed target",
            "Sources=image-space relative and vision-feature relative",
        ),
    )


def _plot_exp5_per_level_corr_by_source(summary: Dict[str, Any], source_name: str, out_path: Path) -> None:
    source_summary = summary.get(source_name, {})
    group_summaries = source_summary.get("group_summaries", {})
    if not group_summaries:
        return

    group_order = summary.get("analysis_groups", _FILTER_ANALYSIS_ORDER)
    group_order = [group_name for group_name in group_order if group_name in group_summaries]
    if not group_order:
        return

    levels = _LEVEL_ORDER
    series = {
        group_name.capitalize(): [
            group_summaries[group_name].get("per_level_correlation", {}).get(level, {}).get("pearson_r", np.nan)
            for level in levels
        ]
        for group_name in group_order
    }
    title_prefix = _exp5_source_label(source_name)
    _plot_grouped_bars(
        series,
        [_level_label(level) for level in levels],
        out_path,
        f"{title_prefix} Per-Level Correlation by Filter Group",
        "Pearson r",
        colors=_FILTER_ANALYSIS_COLORS[: len(series)],
        ylim=(-1.0, 1.0),
        metadata=_metadata_lines(
            "Experiment=5",
            f"Source={title_prefix}",
            "Bars=grouped Pearson r by level",
            "Series=filter groups",
        ),
    )


def generate_all_plots(results_dir: Path, config: Optional[Dict[str, Any]] = None) -> None:
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _PLOT_MANIFEST.clear()
    for existing in plots_dir.glob("*.png"):
        existing.unlink(missing_ok=True)
    (plots_dir / "plot_manifest.json").unlink(missing_ok=True)

    profile = _plot_profile(config)
    exhaustive = _is_exhaustive_profile(profile)
    sample_limit = _representative_sample_limit(config, profile)
    sample_scatter_limit = _sample_scatter_point_limit(config, profile)
    suppress_dc = _analysis_suppress_dc(config)

    exp1_summary = _load_json(results_dir / "exp1" / "summary.json")
    exp1_detail = _load_json(results_dir / "exp1" / "degradation_by_level.json")
    exp1_samples = _load_jsonl(results_dir / "exp1" / "per_sample.jsonl")
    exp1_complexity = _load_json(results_dir / "exp1" / "complexity_points.json")
    exp1_tests = _load_json(results_dir / "exp1" / "hypothesis_tests.json")
    exp2_summary = _load_json(results_dir / "exp2" / "summary.json")
    exp2_samples = _load_json(results_dir / "exp2" / "power_spectra.json")
    exp2_complexity = _load_json(results_dir / "exp2" / "complexity_points.json")
    exp2_tests = _load_json(results_dir / "exp2" / "hypothesis_tests.json")
    exp2_by_id = {
        str(record.get("image_id")): record
        for record in (exp2_samples or [])
        if record.get("image_id") is not None
    }
    selected_exp1_records = (
        _select_exp1_samples(exp1_samples, max_samples=min(sample_limit, 3))
        if exp1_samples
        else []
    )
    preferred_sample_ids = [
        image_id
        for image_id in (_record_image_id(record) for record in selected_exp1_records)
        if image_id is not None
    ]
    if exp1_summary:
        per_level = exp1_summary.get("per_level", {})
        _plot_wordy_control_pair_comparison(
            {level: float(stats.get("mean_accuracy_drop", np.nan)) for level, stats in per_level.items()},
            plots_dir / "exp1_wordy_control_accuracy_drop.png",
            title="Wordy Control Comparison: Accuracy Drop",
            ylabel="Mean Accuracy Drop",
            metadata=_plot_metadata(
                experiment="1",
                what="Base level vs matched wordy control for mean accuracy drop",
                aggregation="level-wise average over all perturbation evaluations",
                x="matched base/control pair",
                y="mean gated accuracy drop",
                note="Compares L1 vs L5, L2 vs L6, L3 vs L7, and L4 vs L8",
                profile=profile,
            ),
        )
        _plot_wordy_control_pair_comparison(
            {level: float(stats.get("mean_loglik_drift", np.nan)) for level, stats in per_level.items()},
            plots_dir / "exp1_wordy_control_loglik_drift.png",
            title="Wordy Control Comparison: Log-Likelihood Drift",
            ylabel="Mean Log-Likelihood Drift",
            metadata=_plot_metadata(
                experiment="1",
                what="Base level vs matched wordy control for signed confidence drift",
                aggregation="level-wise average over all perturbation evaluations",
                x="matched base/control pair",
                y="mean correct-answer log-likelihood drift",
                note="Positive values mean confidence erosion",
                profile=profile,
            ),
        )
        _plot_wordy_control_pair_comparison(
            {level: float(stats.get("mean_loglik_volatility", np.nan)) for level, stats in per_level.items()},
            plots_dir / "exp1_wordy_control_loglik_volatility.png",
            title="Wordy Control Comparison: Log-Likelihood Volatility",
            ylabel="Mean Log-Likelihood Volatility",
            metadata=_plot_metadata(
                experiment="1",
                what="Base level vs matched wordy control for absolute confidence movement",
                aggregation="level-wise average over all perturbation evaluations",
                x="matched base/control pair",
                y="mean absolute correct-answer log-likelihood drift",
                profile=profile,
            ),
        )
    if exp1_complexity:
        plot_complexity_scatter(
            exp1_complexity,
            y_key="mean_accuracy_drop",
            y_label="Mean Accuracy Drop",
            out_path=plots_dir / "exp1_complexity_accuracy_drop.png",
            title="Mean Accuracy Drop vs Residualized Semantic Logic",
            metadata=_plot_metadata(
                experiment="1",
                what="Residualized semantic logic vs robustness degradation",
                aggregation="per-image per-level average over perturbations",
                x="residualized semantic logic",
                y="mean gated accuracy drop",
                note="X is residual(semantic complexity ~ prompt load); black line is mean by exact residual score; dashed line is linear fit",
                profile=profile,
            ),
        )
        plot_complexity_scatter(
            exp1_complexity,
            y_key="mean_loglik_drift",
            y_label="Mean Log-Likelihood Drift",
            out_path=plots_dir / "exp1_complexity_loglik_drift.png",
            title="Log-Likelihood Drift vs Residualized Semantic Logic",
            metadata=_plot_metadata(
                experiment="1",
                what="Residualized semantic logic vs confidence erosion",
                aggregation="per-image per-level average over perturbations",
                x="residualized semantic logic",
                y="mean correct-answer log-likelihood drift",
                note="X is residual(semantic complexity ~ prompt load); black line is mean by exact residual score; dashed line is linear fit",
                profile=profile,
            ),
        )
        plot_complexity_scatter(
            exp1_complexity,
            y_key="mean_loglik_erosion",
            y_label="Mean Log-Likelihood Erosion",
            out_path=plots_dir / "exp1_complexity_loglik_erosion.png",
            title="Log-Likelihood Erosion vs Residualized Semantic Logic",
            metadata=_plot_metadata(
                experiment="1",
                what="Residualized semantic logic vs directional confidence erosion",
                aggregation="per-image per-level average over perturbations",
                x="residualized semantic logic",
                y="mean positive correct-answer log-likelihood drift",
                note="Only positive drifts contribute; X is residual(semantic complexity ~ prompt load); dashed line is linear fit",
                profile=profile,
            ),
        )
        plot_complexity_scatter(
            exp1_complexity,
            y_key="mean_loglik_recovery",
            y_label="Mean Log-Likelihood Recovery",
            out_path=plots_dir / "exp1_complexity_loglik_recovery.png",
            title="Log-Likelihood Recovery vs Residualized Semantic Logic",
            metadata=_plot_metadata(
                experiment="1",
                what="Residualized semantic logic vs directional confidence recovery",
                aggregation="per-image per-level average over perturbations",
                x="residualized semantic logic",
                y="mean negative correct-answer log-likelihood drift",
                note="Only negative drifts contribute; X is residual(semantic complexity ~ prompt load); dashed line is linear fit",
                profile=profile,
            ),
        )
        plot_complexity_scatter(
            exp1_complexity,
            y_key="mean_loglik_volatility",
            y_label="Mean Log-Likelihood Volatility",
            out_path=plots_dir / "exp1_complexity_loglik_volatility.png",
            title="Log-Likelihood Volatility vs Residualized Semantic Logic",
            metadata=_plot_metadata(
                experiment="1",
                what="Residualized semantic logic vs absolute confidence movement",
                aggregation="per-image per-level average over perturbations",
                x="residualized semantic logic",
                y="mean absolute correct-answer log-likelihood drift",
                note="X is residual(semantic complexity ~ prompt load); black line is mean by exact residual score; dashed line is linear fit",
                profile=profile,
            ),
        )
    if exp1_tests:
        cc = exp1_tests.get("continuous_complexity", {})
        mirror_tests = cc.get("wordy_mirror_paired_tests")
        if not mirror_tests and exp1_complexity:
            mirror_tests = paired_wordy_mirror_ttests(
                exp1_complexity,
                y_keys=[
                    "mean_accuracy_drop",
                    "mean_loglik_drift",
                    "mean_loglik_erosion",
                    "mean_loglik_recovery",
                    "mean_loglik_volatility",
                ],
            )
        if mirror_tests:
            _plot_linguistic_stabilization_effect(
                mirror_tests,
                plots_dir / "exp1_linguistic_stabilization_effect.png",
                profile=profile,
            )
        accuracy_reg_pooled = cc.get("horse_race_mean_accuracy_drop")
        accuracy_reg_within = cc.get("horse_race_mean_accuracy_drop_within_image")
        accuracy_reg = accuracy_reg_pooled or accuracy_reg_within
        if accuracy_reg:
            plot_coefficient_forest(
                accuracy_reg,
                plots_dir / "exp1_coefficient_plot_accuracy_drop.png",
                title="Coefficient Plot: Accuracy Drop Controls",
                subtitle="Pooled and within-image standardized coefficients",
                primary_label="Pooled OLS" if accuracy_reg_pooled else "Within-Image Fixed Effects",
                comparison_label="Within-Image Fixed Effects" if accuracy_reg_pooled and accuracy_reg_within else None,
                comparison_regression=accuracy_reg_within if accuracy_reg_pooled else None,
                metadata=_plot_metadata(
                    experiment="1",
                    what="Relative impact of residualized semantic logic vs prompt load vs option hardness vs prediction entropy on gated fragility",
                    aggregation="multivariate regression over per-image per-level averages",
                    x="standardized coefficient with 95% CI",
                    y="predictor",
                    note="Accuracy-drop horse race is filtered to clean-correct points only",
                    profile=profile,
                ),
            )
        loglik_reg_pooled = cc.get("horse_race_mean_loglik_drift")
        loglik_reg_within = cc.get("horse_race_mean_loglik_drift_within_image")
        loglik_reg = loglik_reg_pooled or loglik_reg_within
        if loglik_reg:
            plot_coefficient_forest(
                loglik_reg,
                plots_dir / "exp1_coefficient_plot_loglik_drift.png",
                title="Coefficient Plot: Log-Likelihood Drift Controls",
                subtitle="Pooled and within-image standardized coefficients",
                primary_label="Pooled OLS" if loglik_reg_pooled else "Within-Image Fixed Effects",
                comparison_label="Within-Image Fixed Effects" if loglik_reg_pooled and loglik_reg_within else None,
                comparison_regression=loglik_reg_within if loglik_reg_pooled else None,
                metadata=_plot_metadata(
                    experiment="1",
                    what="Relative impact of residualized semantic logic vs prompt load vs option hardness vs prediction entropy",
                    aggregation="multivariate regression over per-image per-level averages",
                    x="standardized coefficient with 95% CI",
                    y="predictor",
                    note="Shows pooled OLS and within-image fixed-effects when available; outcome is mean correct-answer log-likelihood drift",
                    profile=profile,
                ),
            )
        for outcome_key, file_name, title, note in (
            (
                "mean_loglik_erosion",
                "exp1_coefficient_plot_loglik_erosion.png",
                "Coefficient Plot: Log-Likelihood Erosion Controls",
                "Outcome is mean positive correct-answer log-likelihood drift",
            ),
            (
                "mean_loglik_recovery",
                "exp1_coefficient_plot_loglik_recovery.png",
                "Coefficient Plot: Log-Likelihood Recovery Controls",
                "Outcome is mean negative correct-answer log-likelihood drift",
            ),
            (
                "mean_loglik_volatility",
                "exp1_coefficient_plot_loglik_volatility.png",
                "Coefficient Plot: Log-Likelihood Volatility Controls",
                "Outcome is mean absolute correct-answer log-likelihood drift",
            ),
        ):
            pooled = cc.get(f"horse_race_{outcome_key}")
            within = cc.get(f"horse_race_{outcome_key}_within_image")
            regression = pooled or within
            if not regression:
                continue
            plot_coefficient_forest(
                regression,
                plots_dir / file_name,
                title=title,
                subtitle="Pooled and within-image standardized coefficients",
                primary_label="Pooled OLS" if pooled else "Within-Image Fixed Effects",
                comparison_label="Within-Image Fixed Effects" if pooled and within else None,
                comparison_regression=within if pooled else None,
                metadata=_plot_metadata(
                    experiment="1",
                    what="Directional drift controls using residualized semantic logic",
                    aggregation="multivariate regression over per-image per-level averages",
                    x="standardized coefficient with 95% CI",
                    y="predictor",
                    note=note,
                    profile=profile,
                ),
            )
        for (
            _outcome_key,
            test_key,
            main_filename,
            main_title,
            supplementary_filename,
            supplementary_title,
            outcome_label,
            note,
        ) in _EXP1_PERTURBATION_OUTCOME_SPECS:
            perturbation_regs = cc.get(test_key, {})
            if not perturbation_regs:
                continue
            series = _collect_regression_series_from_payload_map(perturbation_regs)
            if not series:
                continue
            plot_coefficient_forest_series(
                series,
                plots_dir / main_filename,
                title=main_title,
                metadata=_plot_metadata(
                    experiment="1",
                    what=f"Perturbation-specific horse race for {outcome_label}",
                    aggregation="one subplot per perturbation type",
                    x="standardized coefficient with 95% CI",
                    y="predictor",
                    note=note,
                    profile=profile,
                ),
            )
            plot_coefficient_forest_series(
                series,
                plots_dir / supplementary_filename,
                title=supplementary_title,
                metadata=_plot_metadata(
                    experiment="1",
                    what=f"Supplementary perturbation-specific horse race for {outcome_label}",
                    aggregation="one subplot per perturbation type",
                    x="standardized coefficient with 95% CI",
                    y="predictor",
                    note=note + "; supplementary figure",
                    profile=profile,
                ),
            )
    if exp1_samples:
        family_points = _exp1_perturbation_family_points(exp1_samples)
        family_outcome_specs = (
            (
                "accuracy_drop",
                "exp1_coefficient_plot_accuracy_drop",
                "Accuracy Drop",
                "accuracy drop",
                "Only clean-correct points are included, so this measures fragility of existing knowledge",
                True,
            ),
            (
                "loglik_drift",
                "exp1_coefficient_plot_loglik_drift",
                "Log-Likelihood Drift",
                "signed correct-answer log-likelihood drift",
                "Positive values mean confidence erosion; negative values mean confidence recovery",
                False,
            ),
            (
                "loglik_erosion",
                "exp1_coefficient_plot_loglik_erosion",
                "Log-Likelihood Erosion",
                "positive correct-answer log-likelihood drift",
                "Only positive drifts contribute to this outcome",
                False,
            ),
            (
                "loglik_recovery",
                "exp1_coefficient_plot_loglik_recovery",
                "Log-Likelihood Recovery",
                "negative correct-answer log-likelihood drift",
                "More negative values indicate stronger confidence recovery",
                False,
            ),
            (
                "loglik_volatility",
                "exp1_coefficient_plot_loglik_volatility",
                "Log-Likelihood Volatility",
                "absolute correct-answer log-likelihood drift",
                "Higher values mean larger movement away from zero regardless of sign",
                False,
            ),
        )
        for family in sorted(family_points):
            family_label = family.replace("_", " ").title()
            raw_points = family_points[family]
            for y_key, filename_prefix, title_suffix, outcome_label, note, gate_clean_correct in family_outcome_specs:
                points = raw_points
                if gate_clean_correct:
                    points = [
                        point
                        for point in raw_points
                        if float(point.get("clean_accuracy", 0.0) or 0.0) == 1.0
                    ]
                predictor_keys = _horse_race_predictor_keys(points)
                pooled = summarize_multivariate_regression(
                    points,
                    y_key=y_key,
                    x_keys=predictor_keys,
                )
                within = summarize_multivariate_regression(
                    points,
                    y_key=y_key,
                    x_keys=predictor_keys,
                    group_key="image_id",
                    demean_by_group=True,
                )
                regression = pooled if pooled.get("predictors") else within
                comparison = within if pooled.get("predictors") and within.get("predictors") else None
                if not regression or not regression.get("predictors"):
                    continue
                plot_coefficient_forest(
                    regression,
                    plots_dir / f"{filename_prefix}_{_safe_name(family)}.png",
                    title=f"Coefficient Plot: {family_label} {title_suffix}",
                    subtitle="Predictors for one perturbation family",
                    primary_label="Pooled OLS" if pooled.get("predictors") else "Within-Image Fixed Effects",
                    comparison_label="Within-Image Fixed Effects" if comparison is not None else None,
                    comparison_regression=comparison,
                    metadata=_plot_metadata(
                        experiment="1",
                        what=f"Perturbation-family-specific horse race for {outcome_label}",
                        aggregation="per-image per-level single-perturbation observations",
                        x="standardized coefficient with 95% CI",
                        y="predictor",
                        selection=f"family={family}",
                        note=note,
                        profile=profile,
                    ),
                )
    if exp1_detail:
        levels, perturbations, matrix = _exp1_level_perturbation_matrix(exp1_detail)
        _plot_heatmap(
            matrix,
            [_level_label(level) for level in levels],
            perturbations,
            plots_dir / "exp1_level_perturbation_drop.png",
            "Accuracy Drop by Level and Perturbation",
            "Mean Accuracy Drop",
            cmap="YlOrRd",
            metadata=_plot_metadata(
                experiment="1",
                what="Mean gated accuracy drop across perturbations",
                aggregation="level x perturbation average over samples",
                x="perturbation type",
                y="task level",
                note="Higher values mean clean-correct answers became wrong more often",
                profile=profile,
            ),
        )
    if exp1_samples:
        sample_ids, levels, matrix = _exp1_sample_level_matrix(exp1_samples)
        sample_ids, matrix = _sort_metric_rows(sample_ids, matrix, max_rows=10)
        _plot_heatmap(
            matrix,
            [_short_sample_id(sample_id) for sample_id in sample_ids],
            [_level_label(level) for level in levels],
            plots_dir / "exp1_sample_level_drop.png",
            "Per-Sample Mean Drop by Level",
            "Mean Accuracy Drop",
            cmap="YlGnBu",
            metadata=_plot_metadata(
                experiment="1",
                what="Per-image mean gated accuracy drop",
                aggregation="sample-wise average over perturbations",
                x="task level",
                y="selected images",
                selection="top 10 images by average drop",
                profile=profile,
            ),
        )
        for record in selected_exp1_records:
            sample_id = str(record.get("image_id"))
            matched_exp2 = exp2_by_id.get(sample_id)
            delta_f = _exp1_sample_spectra(record, "delta_f")
            if delta_f:
                _plot_sample_profile_grid(
                    sample_id,
                    delta_f,
                    plots_dir / f"exp1_sample_delta_f_{_safe_name(sample_id)}.png",
                    "Image-Space delta_f by Perturbation",
                    "delta_f(omega)",
                    suppress_dc=suppress_dc,
                    metadata=_plot_metadata(
                        experiment="1",
                        what="Image-space perturbation spectra for one image",
                        aggregation="per-level frequency profiles by perturbation",
                        x="frequency band (low→high)",
                        y="delta_f(omega)",
                        selection="representative high-drop image",
                        profile=profile,
                    ),
                )
            delta_f_vision = _exp1_sample_spectra(record, "delta_f_vision")
            if delta_f_vision:
                _plot_sample_profile_grid(
                    sample_id,
                    delta_f_vision,
                    plots_dir / f"exp1_sample_delta_f_vision_{_safe_name(sample_id)}.png",
                    "Vision-Feature delta_f by Perturbation",
                    "delta_f_vision(omega)",
                    suppress_dc=suppress_dc,
                    metadata=_plot_metadata(
                        experiment="1",
                        what="Vision-feature perturbation spectra for one image",
                        aggregation="per-level frequency profiles by perturbation",
                        x="frequency band (low→high)",
                        y="delta_f_vision(omega)",
                        selection="representative high-drop image",
                        profile=profile,
                    ),
                )
            levels, perturbations, matrix = _exp1_sample_drop_matrix(record)
            if matrix.size and perturbations:
                _plot_heatmap(
                    matrix,
                    [_level_label(level) for level in levels],
                    perturbations,
                    plots_dir / f"exp1_sample_drop_by_perturbation_{_safe_name(sample_id)}.png",
                    f"Per-Perturbation Accuracy Drop: {_short_sample_id(sample_id, 28)}",
                    "Accuracy Drop",
                    cmap="YlOrRd",
                    vmin=0.0,
                    vmax=1.0,
                    metadata=_plot_metadata(
                        experiment="1",
                        what="Perturbation-wise gated accuracy drop for one selected image",
                        aggregation="rows are levels; columns are perturbation types for the same image",
                        x="perturbation type",
                        y="task level",
                        selection="same representative high-drop image used for delta_f plots",
                        note="1 means clean-correct became wrong; 0 means no gated drop",
                        profile=profile,
                    ),
                )
            levels, perturbations, matrix = _exp1_sample_metric_matrix(record, "loglik_drift")
            if matrix.size and perturbations:
                _plot_heatmap(
                    matrix,
                    [_level_label(level) for level in levels],
                    perturbations,
                    plots_dir / f"exp1_sample_loglik_drift_{_safe_name(sample_id)}.png",
                    f"Per-Perturbation Log-Likelihood Drift: {_short_sample_id(sample_id, 28)}",
                    "Log-Likelihood Drift",
                    cmap="coolwarm",
                    metadata=_plot_metadata(
                        experiment="1",
                        what="Perturbation-wise correct-answer log-likelihood drift for one selected image",
                        aggregation="rows are levels; columns are perturbation types for the same image",
                        x="perturbation type",
                        y="task level",
                        selection="same representative high-drop image used for other Exp 1 sample plots",
                        note="Positive values mean the correct answer score fell under perturbation",
                        profile=profile,
                    ),
                )
            matched_wt_series = _exp2_sample_wt_series(matched_exp2)
            if matched_wt_series:
                _plot_sample_profile_grid(
                    sample_id,
                    matched_wt_series,
                    plots_dir / f"exp2_matched_sample_wt_{_safe_name(sample_id)}.png",
                    "Task Filters W_t for Matched Selected Sample",
                    "W_t(omega)",
                    suppress_dc=suppress_dc,
                    metadata=_plot_metadata(
                        experiment="2",
                        what="Sample-specific task filters for the same image selected in Exp 1",
                        aggregation="per-level W_t profiles; lines compare overall, early, mid, and late filters",
                        x="frequency band (low→high)",
                        y="W_t(omega)",
                        selection="same representative high-drop image used for delta_f plots",
                        profile=profile,
                    ),
                )
                _plot_exp2_matched_sample_group_comparison(
                    matched_exp2,
                    plots_dir / f"exp2_matched_sample_wt_groups_{_safe_name(sample_id)}.png",
                    profile=profile,
                    suppress_dc=suppress_dc,
                )

    if exp2_summary:
        plot_attention_power_spectrum(
            exp2_summary,
            plots_dir / "exp2_power_spectrum.png",
            suppress_dc=suppress_dc,
        )
        plot_effective_bandwidth(exp2_summary, plots_dir / "exp2_bandwidth.png")
        _plot_wordy_control_pair_comparison(
            {
                level: float(stats.get("mean_bandwidth", np.nan))
                for level, stats in exp2_summary.get("per_level", {}).items()
            },
            plots_dir / "exp2_wordy_control_bandwidth.png",
            title="Wordy Control Comparison: Effective Bandwidth",
            ylabel="Effective Bandwidth",
            metadata=_plot_metadata(
                experiment="2",
                what="Base level vs matched wordy control for effective bandwidth",
                aggregation="level-wise sample average",
                x="matched base/control pair",
                y="effective bandwidth G(t)",
                profile=profile,
            ),
        )
        _plot_exp2_group_spectra(
            exp2_summary,
            plots_dir / "exp2_group_spectra.png",
            suppress_dc=suppress_dc,
        )
        _plot_exp2_group_bandwidth(exp2_summary, plots_dir / "exp2_group_bandwidth.png")
        _plot_exp2_control_bandwidth(exp2_summary, plots_dir / "exp2_control_bandwidth.png")
        _plot_exp2_control_divergence(exp2_summary, plots_dir / "exp2_control_divergence.png")
        _plot_exp2_control_spectra(exp2_summary, plots_dir, suppress_dc=suppress_dc)
        if exp2_complexity:
            _plot_exp2_complexity_groups(
                exp2_complexity,
                plots_dir / "exp2_complexity_bandwidth.png",
                profile=profile,
            )
        if exp2_tests:
            cc = exp2_tests.get("continuous_complexity", {})
            bandwidth_reg_pooled = cc.get("horse_race_bandwidth")
            bandwidth_reg_within = cc.get("horse_race_bandwidth_within_image")
            bandwidth_reg = bandwidth_reg_pooled or bandwidth_reg_within
            if bandwidth_reg:
                plot_coefficient_forest(
                    bandwidth_reg,
                    plots_dir / "exp2_coefficient_plot_bandwidth.png",
                    title="Coefficient Plot: Bandwidth Controls",
                    subtitle="Pooled and within-image standardized coefficients",
                    primary_label="Pooled OLS" if bandwidth_reg_pooled else "Within-Image Fixed Effects",
                    comparison_label="Within-Image Fixed Effects" if bandwidth_reg_pooled and bandwidth_reg_within else None,
                    comparison_regression=bandwidth_reg_within if bandwidth_reg_pooled else None,
                    metadata=_plot_metadata(
                        experiment="2",
                        what="Relative impact of residualized semantic logic vs prompt load vs option hardness",
                        aggregation="multivariate regression over per-image per-level bandwidth",
                        x="standardized coefficient with 95% CI",
                        y="predictor",
                        note="Shows pooled OLS and within-image fixed-effects when available; outcome is effective bandwidth G(t)",
                        profile=profile,
                    ),
                )
        if exhaustive:
            _plot_exp2_level_band_heatmap(
                exp2_summary,
                plots_dir / "exp2_band_heatmap.png",
                suppress_dc=suppress_dc,
            )
            for group_name in exp2_summary.get("layer_groups", {}).get("order", ["early", "mid", "late"]):
                levels = [
                    level
                    for level in _ordered_levels(exp2_summary.get("per_level", {}).keys())
                    if exp2_summary.get("per_level", {}).get(level, {}).get("layer_groups", {}).get(group_name, {}).get("W_t_average")
                ]
                if not levels:
                    continue
                group_matrix = np.asarray(
                    [
                        exp2_summary.get("per_level", {}).get(level, {}).get("layer_groups", {}).get(group_name, {}).get("W_t_average", [])
                        for level in levels
                    ],
                    dtype=float,
                )
                if group_matrix.size == 0 or not levels:
                    continue
                _plot_heatmap(
                    group_matrix,
                    [_level_label(level) for level in levels],
                    _band_labels(group_matrix.shape[1], suppress_dc=suppress_dc),
                    plots_dir / f"exp2_band_heatmap_{group_name}.png",
                    f"Attention Filter W_t by Level and Band ({group_name})",
                    "W_t(omega)",
                    cmap="magma",
                    metadata=_plot_metadata(
                        experiment="2",
                        what=f"Layer-group task filter W_t ({group_name})",
                        aggregation="level-wise sample average",
                        x="frequency band (low→high)",
                        y="task level",
                        profile=profile,
                    ),
                )
    if exp2_samples:
        _plot_distribution_with_points(
            _exp2_bandwidth_values(exp2_samples),
            plots_dir / "exp2_bandwidth_distribution.png",
            "Bandwidth Distribution Across Samples",
            "Bandwidth",
            metadata=_plot_metadata(
                experiment="2",
                what="Within-run sample spread of effective bandwidth",
                aggregation="per-sample values grouped by level",
                x="task level",
                y="effective bandwidth G(t)",
                profile=profile,
            ),
        )
        _plot_exp2_selected_samples(
            exp2_samples,
            plots_dir,
            max_samples=sample_limit,
            preferred_sample_ids=preferred_sample_ids,
            profile=profile,
            suppress_dc=suppress_dc,
        )
        if exhaustive:
            sample_ids, levels, matrix = _exp2_sample_level_matrix(exp2_samples)
            sample_ids, matrix = _sort_metric_rows(sample_ids, matrix, max_rows=10)
            _plot_heatmap(
                matrix,
                [_short_sample_id(sample_id) for sample_id in sample_ids],
                [_level_label(level) for level in levels],
                plots_dir / "exp2_sample_bandwidth.png",
                "Per-Sample Bandwidth by Level",
                "Bandwidth",
                cmap="plasma",
                metadata=_plot_metadata(
                    experiment="2",
                    what="Per-image effective bandwidth",
                    aggregation="one value per image and level",
                    x="task level",
                    y="selected images",
                    selection="top 10 images by average bandwidth",
                    profile=profile,
                ),
            )

    exp3_summary = _load_json(results_dir / "exp3" / "summary.json")
    exp3_samples = _load_json(results_dir / "exp3" / "amplification.json")
    exp3_tests = _load_json(results_dir / "exp3" / "hypothesis_tests.json")
    exp3_complexity = _load_json(results_dir / "exp3" / "complexity_points.json")
    if exp3_summary:
        _plot_exp3_response_amplification(exp3_summary, plots_dir / "exp3_amplification.png")
        plot_amplification_heatmap(
            exp3_summary,
            plots_dir / "exp3_pre_drift_spectrum.png",
            suppress_dc=suppress_dc,
        )
        _plot_exp3_pre_post(exp3_summary, plots_dir / "exp3_pre_post_drift.png")
        _plot_wordy_control_pair_comparison(
            {
                level: float(stats.get("mean_post_drift_all", np.nan))
                for level, stats in exp3_summary.get("per_level", {}).items()
            },
            plots_dir / "exp3_wordy_control_post_drift.png",
            title="Wordy Control Comparison: Post-Fusion Drift",
            ylabel="Mean Post-Fusion Drift ΔZ_all",
            metadata=_plot_metadata(
                experiment="3",
                what="Base level vs matched wordy control for task-conditioned post-fusion drift",
                aggregation="level-wise average over perturbation observations",
                x="matched base/control pair",
                y="mean post-fusion drift ΔZ_all",
                profile=profile,
            ),
        )
        _plot_wordy_control_pair_comparison(
            {
                level: float(stats.get("mean_response_amplification", np.nan))
                for level, stats in exp3_summary.get("per_level", {}).items()
            },
            plots_dir / "exp3_wordy_control_response_amplification.png",
            title="Wordy Control Comparison: Response Amplification",
            ylabel="Mean Response Amplification",
            metadata=_plot_metadata(
                experiment="3",
                what="Base level vs matched wordy control for response amplification",
                aggregation="level-wise average over perturbation observations",
                x="matched base/control pair",
                y="mean response amplification",
                profile=profile,
            ),
        )
        _plot_exp3_group_weighted_amplification(
            exp3_summary,
            plots_dir / "exp3_group_weighted_amplification.png",
        )
        _plot_exp3_profile_groups(
            exp3_summary,
            plots_dir / "exp3_profile_group_response.png",
        )
    if exp3_complexity:
        plot_complexity_scatter(
            exp3_complexity,
            y_key="mean_post_drift_all",
            y_label="Mean Post-Fusion Drift ΔZ_all",
            out_path=plots_dir / "exp3_complexity_post_drift_all.png",
            title="Post-Fusion Drift vs Residualized Semantic Logic",
            metadata=_plot_metadata(
                experiment="3",
                what="Residualized semantic logic vs task-conditioned post-fusion response",
                aggregation="per-image per-level average over perturbations",
                x="residualized semantic logic",
                y="mean post-fusion drift ΔZ_all",
                note="X is residual(semantic complexity ~ prompt load); black line is mean by exact residual score; dashed line is linear fit",
                profile=profile,
            ),
        )
        plot_complexity_scatter(
            exp3_complexity,
            y_key="mean_response_amplification",
            y_label="Mean Response Amplification",
            out_path=plots_dir / "exp3_complexity_response_amplification.png",
            title="Response Amplification vs Residualized Semantic Logic",
            metadata=_plot_metadata(
                experiment="3",
                what="Residualized semantic logic vs response amplification",
                aggregation="per-image per-level average over perturbations",
                x="residualized semantic logic",
                y="mean response amplification",
                note="X is residual(semantic complexity ~ prompt load); black line is mean by exact residual score; dashed line is linear fit",
                profile=profile,
            ),
        )
    if exp3_tests:
        cc = exp3_tests.get("continuous_complexity", {})
        post_reg_pooled = cc.get("horse_race_mean_post_drift_all")
        post_reg_within = cc.get("horse_race_mean_post_drift_all_within_image")
        post_reg = post_reg_pooled or post_reg_within
        if post_reg:
            plot_coefficient_forest(
                post_reg,
                plots_dir / "exp3_coefficient_plot_post_drift_all.png",
                title="Coefficient Plot: Post-Fusion Drift Controls",
                subtitle="Pooled and within-image standardized coefficients",
                primary_label="Pooled OLS" if post_reg_pooled else "Within-Image Fixed Effects",
                comparison_label="Within-Image Fixed Effects" if post_reg_pooled and post_reg_within else None,
                comparison_regression=post_reg_within if post_reg_pooled else None,
                metadata=_plot_metadata(
                    experiment="3",
                    what="Relative impact of residualized semantic logic vs prompt load vs option hardness",
                    aggregation="multivariate regression over per-image per-level mean post-fusion drift",
                    x="standardized coefficient with 95% CI",
                    y="predictor",
                    note="Shows pooled OLS and within-image fixed-effects when available; outcome is mean post-fusion drift ΔZ_all",
                    profile=profile,
                ),
            )
            if post_reg_within or post_reg_pooled:
                plot_dual_force_bars(
                    post_reg_within or post_reg_pooled,
                    plots_dir / "exp3_tug_of_war_internal_drift.png",
                    title="Two-Factor Tug-of-War: Internal Drift",
                    subtitle=(
                        "Within-image fixed effects" if post_reg_within else "Pooled OLS"
                    ),
                    metadata=_plot_metadata(
                        experiment="3",
                        what="Standardized semantic-logic vs prompt-load effects on post-fusion internal drift",
                        aggregation="multivariate regression over per-image per-level mean post-fusion drift",
                        x="semantic logic and prompt load",
                        y="standardized coefficient with 95% CI",
                        note="Theory figure for compositional precision vs linguistic anchoring",
                        profile=profile,
                    ),
                )
        amp_reg_pooled = cc.get("horse_race_mean_response_amplification")
        amp_reg_within = cc.get("horse_race_mean_response_amplification_within_image")
        amp_reg = amp_reg_pooled or amp_reg_within
        if amp_reg:
            plot_coefficient_forest(
                amp_reg,
                plots_dir / "exp3_coefficient_plot_response_amplification.png",
                title="Coefficient Plot: Response Amplification Controls",
                subtitle="Pooled and within-image standardized coefficients",
                primary_label="Pooled OLS" if amp_reg_pooled else "Within-Image Fixed Effects",
                comparison_label="Within-Image Fixed Effects" if amp_reg_pooled and amp_reg_within else None,
                comparison_regression=amp_reg_within if amp_reg_pooled else None,
                metadata=_plot_metadata(
                    experiment="3",
                    what="Relative impact of residualized semantic logic vs prompt load vs option hardness",
                    aggregation="multivariate regression over per-image per-level mean response amplification",
                    x="standardized coefficient with 95% CI",
                    y="predictor",
                    note="Shows pooled OLS and within-image fixed-effects when available; outcome is mean response amplification",
                    profile=profile,
                ),
            )
    if exp3_samples:
        if exhaustive:
            levels, perturbations, matrix = _exp3_level_perturbation_matrix(exp3_samples)
            _plot_heatmap(
                matrix,
                [_level_label(level) for level in levels],
                perturbations,
                plots_dir / "exp3_level_perturbation_amplification.png",
                "Post-Fusion Response by Level and Perturbation",
                "Mean Post-Fusion Drift (all tokens)",
                cmap="magma",
                metadata=_plot_metadata(
                    experiment="3",
                    what="Mean all-token post-fusion response",
                    aggregation="level x perturbation average over samples",
                    x="perturbation type",
                    y="task level",
                    profile=profile,
                ),
            )
            sample_ids, levels, matrix = _exp3_sample_level_matrix(exp3_samples)
            sample_ids, matrix = _sort_metric_rows(sample_ids, matrix, max_rows=10)
            _plot_heatmap(
                matrix,
                [_short_sample_id(sample_id) for sample_id in sample_ids],
                [_level_label(level) for level in levels],
                plots_dir / "exp3_sample_amplification.png",
                "Per-Sample Post-Fusion Response by Level",
                "Mean Post-Fusion Drift (all tokens)",
                cmap="PuRd",
                metadata=_plot_metadata(
                    experiment="3",
                    what="Per-image mean post-fusion response",
                    aggregation="sample-wise average over perturbations",
                    x="task level",
                    y="selected images",
                    selection="top 10 images by average post-fusion response",
                    profile=profile,
                ),
            )
        for record in _select_records_by_preferred_ids(
            exp3_samples,
            preferred_sample_ids,
            max_samples=sample_limit,
            fallback_selector=_select_exp3_samples,
        ):
            sample_id = str(record.get("image_id"))
            profiles, overlays = _exp3_sample_profiles(record)
            if profiles:
                _plot_sample_profile_grid(
                    sample_id,
                    profiles,
                    plots_dir / f"exp3_sample_profiles_{_safe_name(sample_id)}.png",
                    "Pre-Fusion Drift by Perturbation",
                    "Pre-Fusion Drift",
                    overlay_by_level=overlays,
                    suppress_dc=suppress_dc,
                    metadata=_plot_metadata(
                        experiment="3",
                        what="Pre-fusion drift profiles with late W_t overlay",
                        aggregation="per-level frequency profiles by perturbation",
                        x="frequency band (low→high)",
                        y="pre-fusion drift magnitude",
                        selection="preferred Exp 1 representative image when available; otherwise representative high-response image",
                        note="Dashed line is late-group W_t",
                        profile=profile,
                    ),
                )

    exp4_summary = _load_json(results_dir / "exp4" / "summary.json")
    exp4_curves = _load_json(results_dir / "exp4" / "accuracy_curves.json")
    exp4_samples = _load_json(results_dir / "exp4" / "per_sample.json")
    exp4_complexity = _load_json(results_dir / "exp4" / "complexity_points.json")
    exp4_tests = _load_json(results_dir / "exp4" / "hypothesis_tests.json")
    if exp4_curves:
        for mode in ("lowpass", "highpass"):
            plot_frequency_threshold_curves(
                exp4_curves,
                plots_dir / f"exp4_threshold_{mode}.png",
                mode=mode,
            )
            _plot_exp4_mode_heatmap(
                exp4_curves,
                plots_dir / f"exp4_level_cutoff_{mode}.png",
                mode=mode,
                title=f"Accuracy by Level and Cutoff ({mode})",
                metadata=_plot_metadata(
                    experiment="4",
                    what=f"Accuracy across frequency cutoffs for {mode}",
                    aggregation="level x cutoff average over samples",
                    x=f"{mode} cutoff",
                    y="task level",
                    note="This is the perturbation-wise equivalent for the frequency-sweep experiment",
                    profile=profile,
                ),
            )
    if exp4_summary:
        _plot_exp4_critical_cutoffs(exp4_summary, plots_dir / "exp4_critical_cutoffs.png")
        for mode in ("lowpass", "highpass"):
            _plot_wordy_control_pair_comparison(
                {
                    level: float(stats.get("critical_cutoff", np.nan))
                    for level, stats in exp4_summary.get("per_mode", {}).get(mode, {}).items()
                },
                plots_dir / f"exp4_wordy_control_critical_cutoff_{mode}.png",
                title=f"Wordy Control Comparison: Critical Cutoff ({mode})",
                ylabel="Critical Cutoff",
                metadata=_plot_metadata(
                    experiment="4",
                    what=f"Base level vs matched wordy control for critical cutoff ({mode})",
                    aggregation="level-wise cutoff estimate",
                    x="matched base/control pair",
                    y="critical cutoff",
                    profile=profile,
                ),
            )
    if exp4_complexity:
        _plot_exp4_complexity_modes(
            exp4_complexity,
            plots_dir / "exp4_complexity_cutoff.png",
            profile=profile,
        )
    if exp4_tests:
        cc = exp4_tests.get("continuous_complexity", {})
        for mode in ("lowpass", "highpass"):
            mode_reg = cc.get(mode, {})
            regression_pooled = mode_reg.get("horse_race_critical_cutoff")
            regression_within = mode_reg.get("horse_race_critical_cutoff_within_image")
            regression = regression_pooled or regression_within
            if regression:
                plot_coefficient_forest(
                    regression,
                    plots_dir / f"exp4_coefficient_plot_{mode}.png",
                    title=f"Coefficient Plot: Critical Cutoff Controls ({mode})",
                    subtitle="Pooled and within-image standardized coefficients",
                    primary_label="Pooled OLS" if regression_pooled else "Within-Image Fixed Effects",
                    comparison_label="Within-Image Fixed Effects" if regression_pooled and regression_within else None,
                    comparison_regression=regression_within if regression_pooled else None,
                    metadata=_plot_metadata(
                        experiment="4",
                        what="Relative impact of residualized semantic logic vs prompt load vs option hardness",
                        aggregation=f"multivariate regression over per-image per-level critical cutoff ({mode})",
                        x="standardized coefficient with 95% CI",
                        y="predictor",
                        note=f"Shows pooled OLS and within-image fixed-effects when available; outcome is critical cutoff under {mode} sweep",
                        profile=profile,
                    ),
                )
    if exp4_samples:
        _plot_exp4_sample_curves(
            exp4_samples,
            plots_dir,
            max_samples=sample_limit,
            preferred_sample_ids=preferred_sample_ids,
            profile=profile,
        )
        if exhaustive:
            for mode, (sample_ids, levels, matrix) in _exp4_sample_cutoffs(exp4_samples).items():
                sample_ids, matrix = _sort_metric_rows(sample_ids, matrix, max_rows=10)
                _plot_heatmap(
                    matrix,
                    [_short_sample_id(sample_id) for sample_id in sample_ids],
                    [_level_label(level) for level in levels],
                    plots_dir / f"exp4_sample_cutoffs_{mode}.png",
                    f"Per-Sample Critical Cutoff ({mode})",
                    "Critical Cutoff",
                    cmap="cividis",
                    vmin=0.0,
                    vmax=0.5,
                    metadata=_plot_metadata(
                        experiment="4",
                        what=f"Per-image critical cutoff ({mode})",
                        aggregation="one value per image and level",
                        x="task level",
                        y="selected images",
                        selection="top 10 images by average critical cutoff",
                        profile=profile,
                    ),
                )

    exp5_summary = _load_json(results_dir / "exp5" / "summary.json")
    exp5_grouped = _load_json(results_dir / "exp5" / "scatter_data.json")
    exp5_samples = _load_json(results_dir / "exp5" / "sample_scatter_data.json")
    exp5_vision_grouped = _load_json(results_dir / "exp5" / "vision_scatter_data.json")
    exp5_vision_samples = _load_json(results_dir / "exp5" / "vision_sample_scatter_data.json")
    exp5_grouped_by_group = _load_json(results_dir / "exp5" / "scatter_data_by_group.json")
    exp5_sample_by_group = _load_json(results_dir / "exp5" / "sample_scatter_data_by_group.json")
    exp5_grouped_by_group_and_target = _load_json(results_dir / "exp5" / "scatter_data_by_group_and_target.json")
    exp5_sample_by_group_and_target = _load_json(results_dir / "exp5" / "sample_scatter_data_by_group_and_target.json")
    if exp5_grouped and exp5_summary:
        image_summary = exp5_summary.get("image_space", {}).get("primary_group_summary", exp5_summary.get("image_space", exp5_summary))
        plot_overlap_scatter(
            exp5_grouped,
            image_summary.get("pearson_r_grouped", 0.0),
            plots_dir / "exp5_overlap_scatter.png",
            title=f"Overlap vs Accuracy Drop (grouped, {_exp5_source_label('image_space')})",
            y_label=_exp5_target_label("accuracy_drop"),
            metadata=_plot_metadata(
                experiment="5",
                what="Predicted overlap vs observed accuracy drop",
                aggregation="grouped by level and perturbation",
                x="predicted spectral overlap S_pred",
                y=_exp5_target_label("accuracy_drop"),
                note=f"Source={_exp5_source_label('image_space')}; group={exp5_summary.get('primary_group', 'late')}",
                profile=profile,
            ),
        )
        plot_overlap_scatter(
            exp5_grouped,
            image_summary.get("comparable_bridge_grouped", {}).get("pearson_r", 0.0),
            plots_dir / "exp5_bridge_scatter.png",
            title=f"Comparable Bridge: Overlap vs Accuracy Drop (grouped, {_exp5_source_label('image_space')})",
            y_label=_exp5_target_label("accuracy_drop"),
            scale_mode="bridge",
            metadata=_plot_metadata(
                experiment="5",
                what="Comparable-bridge view of predicted overlap vs observed accuracy drop",
                aggregation="grouped by level and perturbation",
                x="zscore(log1p(predicted spectral overlap))",
                y=f"zscore({_exp5_target_label('accuracy_drop').lower()})",
                note=f"Source={_exp5_source_label('image_space')}; group={exp5_summary.get('primary_group', 'late')}; raw overlap results are still preserved separately",
                profile=profile,
            ),
        )
        plot_overlap_scatter_grid_by_level(
            exp5_grouped,
            plots_dir / "exp5_overlap_scatter_by_level.png",
            title=f"Overlap vs Accuracy Drop by Level ({_exp5_source_label('image_space')})",
            y_label=_exp5_target_label("accuracy_drop"),
            metadata=_plot_metadata(
                experiment="5",
                what="Predicted overlap vs observed accuracy drop split by level",
                aggregation="grouped by perturbation within each level",
                x="predicted spectral overlap S_pred",
                y=_exp5_target_label("accuracy_drop"),
                note=f"Source={_exp5_source_label('image_space')}; group={exp5_summary.get('primary_group', 'late')}",
                profile=profile,
            ),
        )
        _plot_exp5_level_perturbation_comparison(
            exp5_grouped,
            plots_dir / "exp5_level_perturbation_predicted_vs_observed.png",
            f"Predicted vs Observed by Level and Perturbation ({_exp5_source_label('image_space')})",
            actual_label=_exp5_target_label("accuracy_drop"),
            metadata=_plot_metadata(
                experiment="5",
                what="Predicted overlap and observed accuracy drop split by level and perturbation",
                aggregation="grouped by level and perturbation",
                x="perturbation type",
                y=_exp5_target_label("accuracy_drop"),
                note=f"Source={_exp5_source_label('image_space')}; group={exp5_summary.get('primary_group', 'late')}",
                profile=profile,
            ),
        )
        _plot_exp5_heatmaps(
            exp5_grouped,
            plots_dir / "exp5_grouped_heatmap_image.png",
            f"Grouped Overlap vs Drop ({_exp5_source_label('image_space')})",
            actual_label=_exp5_target_label("accuracy_drop"),
            metadata=_plot_metadata(
                experiment="5",
                what="Grouped predicted overlap, observed accuracy drop, and residual",
                aggregation="level x perturbation average over samples",
                x="perturbation type",
                y="task level",
                note=f"Source={_exp5_source_label('image_space')}; group={exp5_summary.get('primary_group', 'late')}",
                profile=profile,
            ),
        )
        _plot_exp5_wordy_control_comparison(
            exp5_grouped,
            plots_dir / "exp5_wordy_control_comparison.png",
            f"Wordy Control Comparison ({_exp5_source_label('image_space')})",
            actual_label=_exp5_target_label("accuracy_drop"),
            metadata=_plot_metadata(
                experiment="5",
                what="Base level vs matched wordy control for predicted sensitivity and observed accuracy drop",
                aggregation="level-wise mean over grouped perturbation points",
                x="matched base/control pair",
                y=_exp5_target_label('accuracy_drop'),
                note=f"Source={_exp5_source_label('image_space')}; group={exp5_summary.get('primary_group', 'late')}",
                profile=profile,
            ),
        )
    if exp5_vision_grouped and exp5_summary:
        vision_summary = exp5_summary.get("vision_feature_space", {}).get("primary_group_summary", exp5_summary.get("vision_feature_space", {}))
        plot_overlap_scatter(
            exp5_vision_grouped,
            vision_summary.get("pearson_r_grouped", 0.0),
            plots_dir / "exp5_overlap_scatter_vision.png",
            title=f"Overlap vs Accuracy Drop (grouped, {_exp5_source_label('vision_feature_space')})",
            y_label=_exp5_target_label("accuracy_drop"),
            metadata=_plot_metadata(
                experiment="5",
                what="Predicted overlap vs observed accuracy drop",
                aggregation="grouped by level and perturbation",
                x="predicted spectral overlap S_pred",
                y=_exp5_target_label("accuracy_drop"),
                note=f"Source={_exp5_source_label('vision_feature_space')}; group={exp5_summary.get('primary_group', 'late')}",
                profile=profile,
            ),
        )
        plot_overlap_scatter(
            exp5_vision_grouped,
            vision_summary.get("comparable_bridge_grouped", {}).get("pearson_r", 0.0),
            plots_dir / "exp5_bridge_scatter_vision.png",
            title=f"Comparable Bridge: Overlap vs Accuracy Drop (grouped, {_exp5_source_label('vision_feature_space')})",
            y_label=_exp5_target_label("accuracy_drop"),
            scale_mode="bridge",
            metadata=_plot_metadata(
                experiment="5",
                what="Comparable-bridge view of predicted overlap vs observed accuracy drop",
                aggregation="grouped by level and perturbation",
                x="zscore(log1p(predicted spectral overlap))",
                y=f"zscore({_exp5_target_label('accuracy_drop').lower()})",
                note=f"Source={_exp5_source_label('vision_feature_space')}; group={exp5_summary.get('primary_group', 'late')}; raw overlap results are still preserved separately",
                profile=profile,
            ),
        )
        plot_overlap_scatter_grid_by_level(
            exp5_vision_grouped,
            plots_dir / "exp5_overlap_scatter_vision_by_level.png",
            title=f"Overlap vs Accuracy Drop by Level ({_exp5_source_label('vision_feature_space')})",
            y_label=_exp5_target_label("accuracy_drop"),
            metadata=_plot_metadata(
                experiment="5",
                what="Predicted overlap vs observed accuracy drop split by level",
                aggregation="grouped by perturbation within each level",
                x="predicted spectral overlap S_pred",
                y=_exp5_target_label("accuracy_drop"),
                note=f"Source={_exp5_source_label('vision_feature_space')}; group={exp5_summary.get('primary_group', 'late')}",
                profile=profile,
            ),
        )
        _plot_exp5_level_perturbation_comparison(
            exp5_vision_grouped,
            plots_dir / "exp5_level_perturbation_predicted_vs_observed_vision.png",
            f"Predicted vs Observed by Level and Perturbation ({_exp5_source_label('vision_feature_space')})",
            actual_label=_exp5_target_label("accuracy_drop"),
            metadata=_plot_metadata(
                experiment="5",
                what="Predicted overlap and observed accuracy drop split by level and perturbation",
                aggregation="grouped by level and perturbation",
                x="perturbation type",
                y=_exp5_target_label("accuracy_drop"),
                note=f"Source={_exp5_source_label('vision_feature_space')}; group={exp5_summary.get('primary_group', 'late')}",
                profile=profile,
            ),
        )
        _plot_exp5_heatmaps(
            exp5_vision_grouped,
            plots_dir / "exp5_grouped_heatmap_vision.png",
            f"Grouped Overlap vs Drop ({_exp5_source_label('vision_feature_space')})",
            actual_label=_exp5_target_label("accuracy_drop"),
            metadata=_plot_metadata(
                experiment="5",
                what="Grouped predicted overlap, observed accuracy drop, and residual",
                aggregation="level x perturbation average over samples",
                x="perturbation type",
                y="task level",
                note=f"Source={_exp5_source_label('vision_feature_space')}; group={exp5_summary.get('primary_group', 'late')}",
                profile=profile,
            ),
        )
        _plot_exp5_wordy_control_comparison(
            exp5_vision_grouped,
            plots_dir / "exp5_wordy_control_comparison_vision.png",
            f"Wordy Control Comparison ({_exp5_source_label('vision_feature_space')})",
            actual_label=_exp5_target_label("accuracy_drop"),
            metadata=_plot_metadata(
                experiment="5",
                what="Base level vs matched wordy control for predicted sensitivity and observed accuracy drop",
                aggregation="level-wise mean over grouped perturbation points",
                x="matched base/control pair",
                y=_exp5_target_label('accuracy_drop'),
                note=f"Source={_exp5_source_label('vision_feature_space')}; group={exp5_summary.get('primary_group', 'late')}",
                profile=profile,
            ),
        )
    if exp5_summary:
        _plot_exp5_per_level_corr(exp5_summary, plots_dir / "exp5_per_level_correlation.png")
        _plot_exp5_group_correlation(exp5_summary, plots_dir / "exp5_group_correlation.png")
        _plot_exp5_target_correlation(exp5_summary, plots_dir / "exp5_target_correlation.png")
        _plot_exp5_per_level_corr_by_source(
            exp5_summary,
            "image_space",
            plots_dir / "exp5_per_level_correlation_image_groups.png",
        )
        _plot_exp5_per_level_corr_by_source(
            exp5_summary,
            "vision_feature_space",
            plots_dir / "exp5_per_level_correlation_vision_groups.png",
        )
        if exp5_sample_by_group_and_target:
            primary_group = exp5_summary.get("primary_group", "late")
            primary_target = exp5_summary.get("primary_target", "accuracy_drop")
            for source_name in ("image_space", "vision_feature_space"):
                sample_points = (
                    exp5_sample_by_group_and_target
                    .get(source_name, {})
                    .get(primary_group, {})
                    .get(primary_target, [])
                )
                if sample_points:
                    plot_complexity_scatter(
                        sample_points,
                        y_key="predicted",
                        y_label="Predicted Sensitivity S_pred",
                        out_path=plots_dir / f"exp5_complexity_predicted_{source_name}_{primary_group}_{primary_target}.png",
                        title=f"S_pred vs Semantic Complexity ({_exp5_source_label(source_name)}, {primary_group})",
                        metadata=_plot_metadata(
                            experiment="5",
                            what="Continuous semantic complexity vs predicted spectral sensitivity",
                            aggregation="per-sample overlap pairs",
                            x="semantic complexity score",
                            y="predicted sensitivity S_pred",
                            note=f"Source={_exp5_source_label(source_name)}; group={primary_group}; target={_exp5_target_label(primary_target)}",
                            profile=profile,
                        ),
                    )
                    if any(point.get("absolute_prediction_error") is not None for point in sample_points):
                        plot_complexity_scatter(
                            sample_points,
                            y_key="absolute_prediction_error",
                            y_label="Absolute Prediction Error",
                            out_path=plots_dir / f"exp5_complexity_prediction_error_{source_name}_{primary_group}_{primary_target}.png",
                            title=f"Prediction Error vs Semantic Complexity ({_exp5_source_label(source_name)}, {primary_group})",
                            metadata=_plot_metadata(
                                experiment="5",
                                what="Continuous semantic complexity vs calibrated absolute prediction error",
                                aggregation="per-sample overlap pairs",
                                x="semantic complexity score",
                                y="absolute prediction error",
                                note=f"Source={_exp5_source_label(source_name)}; group={primary_group}; target={_exp5_target_label(primary_target)}",
                                profile=profile,
                            ),
                        )
    if exhaustive and exp5_samples and exp5_summary:
        image_summary = exp5_summary.get("image_space", {}).get("primary_group_summary", exp5_summary.get("image_space", exp5_summary))
        plot_overlap_scatter(
            exp5_samples,
            image_summary.get("pearson_r_sample", 0.0),
            plots_dir / "exp5_overlap_scatter_sample.png",
            title=f"Overlap vs Accuracy Drop (per sample, {_exp5_source_label('image_space')})",
            point_size=20,
            alpha=0.45,
            max_points=sample_scatter_limit,
            y_label=_exp5_target_label("accuracy_drop"),
            metadata=_plot_metadata(
                experiment="5",
                what="Predicted overlap vs observed accuracy drop",
                aggregation="per sample",
                x="predicted spectral overlap S_pred",
                y=_exp5_target_label("accuracy_drop"),
                note=f"Source={_exp5_source_label('image_space')}; group={exp5_summary.get('primary_group', 'late')}",
                profile=profile,
            ),
        )
        plot_overlap_scatter(
            exp5_samples,
            image_summary.get("comparable_bridge_sample", {}).get("pearson_r", 0.0),
            plots_dir / "exp5_bridge_scatter_sample.png",
            title=f"Comparable Bridge: Overlap vs Accuracy Drop (per sample, {_exp5_source_label('image_space')})",
            point_size=20,
            alpha=0.45,
            max_points=sample_scatter_limit,
            y_label=_exp5_target_label("accuracy_drop"),
            scale_mode="bridge",
            metadata=_plot_metadata(
                experiment="5",
                what="Comparable-bridge view of predicted overlap vs observed accuracy drop",
                aggregation="per sample",
                x="zscore(log1p(predicted spectral overlap))",
                y=f"zscore({_exp5_target_label('accuracy_drop').lower()})",
                note=f"Source={_exp5_source_label('image_space')}; group={exp5_summary.get('primary_group', 'late')}; raw overlap results are still preserved separately",
                profile=profile,
            ),
        )
    if exhaustive and exp5_vision_samples and exp5_summary:
        vision_summary = exp5_summary.get("vision_feature_space", {}).get("primary_group_summary", exp5_summary.get("vision_feature_space", {}))
        plot_overlap_scatter(
            exp5_vision_samples,
            vision_summary.get("pearson_r_sample", 0.0),
            plots_dir / "exp5_overlap_scatter_vision_sample.png",
            title=f"Overlap vs Accuracy Drop (per sample, {_exp5_source_label('vision_feature_space')})",
            point_size=20,
            alpha=0.45,
            max_points=sample_scatter_limit,
            y_label=_exp5_target_label("accuracy_drop"),
            metadata=_plot_metadata(
                experiment="5",
                what="Predicted overlap vs observed accuracy drop",
                aggregation="per sample",
                x="predicted spectral overlap S_pred",
                y=_exp5_target_label("accuracy_drop"),
                note=f"Source={_exp5_source_label('vision_feature_space')}; group={exp5_summary.get('primary_group', 'late')}",
                profile=profile,
            ),
        )
        plot_overlap_scatter(
            exp5_vision_samples,
            vision_summary.get("comparable_bridge_sample", {}).get("pearson_r", 0.0),
            plots_dir / "exp5_bridge_scatter_vision_sample.png",
            title=f"Comparable Bridge: Overlap vs Accuracy Drop (per sample, {_exp5_source_label('vision_feature_space')})",
            point_size=20,
            alpha=0.45,
            max_points=sample_scatter_limit,
            y_label=_exp5_target_label("accuracy_drop"),
            scale_mode="bridge",
            metadata=_plot_metadata(
                experiment="5",
                what="Comparable-bridge view of predicted overlap vs observed accuracy drop",
                aggregation="per sample",
                x="zscore(log1p(predicted spectral overlap))",
                y=f"zscore({_exp5_target_label('accuracy_drop').lower()})",
                note=f"Source={_exp5_source_label('vision_feature_space')}; group={exp5_summary.get('primary_group', 'late')}; raw overlap results are still preserved separately",
                profile=profile,
            ),
        )
    if exhaustive and exp5_grouped_by_group and exp5_summary:
        primary_group = exp5_summary.get("primary_group", "late")
        for source_name, source_groups in exp5_grouped_by_group.items():
            if source_name.endswith("_raw"):
                continue
            for group_name, grouped_pairs in source_groups.items():
                if not grouped_pairs:
                    continue
                if source_name in {"image_space", "vision_feature_space"} and group_name == primary_group:
                    continue
                source_summary = exp5_summary.get(source_name, {}).get("group_summaries", {}).get(group_name, {})
                plot_overlap_scatter(
                    grouped_pairs,
                    source_summary.get("pearson_r_grouped", 0.0),
                    plots_dir / f"exp5_overlap_scatter_{source_name}_{group_name}.png",
                    title=f"Overlap vs Accuracy Drop ({_exp5_source_label(source_name)}, {group_name})",
                    y_label=_exp5_target_label("accuracy_drop"),
                    metadata=_plot_metadata(
                        experiment="5",
                        what="Predicted overlap vs observed accuracy drop",
                        aggregation="grouped by level and perturbation",
                        x="predicted spectral overlap S_pred",
                        y=_exp5_target_label("accuracy_drop"),
                        note=f"Source={_exp5_source_label(source_name)}; group={group_name}",
                        profile=profile,
                    ),
                )
                _plot_exp5_heatmaps(
                    grouped_pairs,
                    plots_dir / f"exp5_grouped_heatmap_{source_name}_{group_name}.png",
                    f"Grouped Overlap vs Drop ({_exp5_source_label(source_name)}, {group_name})",
                    actual_label=_exp5_target_label("accuracy_drop"),
                    metadata=_plot_metadata(
                        experiment="5",
                        what="Grouped predicted overlap, observed accuracy drop, and residual",
                        aggregation="level x perturbation average over samples",
                        x="perturbation type",
                        y="task level",
                        note=f"Source={_exp5_source_label(source_name)}; group={group_name}",
                        profile=profile,
                    ),
                )
                _plot_exp5_level_perturbation_comparison(
                    grouped_pairs,
                    plots_dir / f"exp5_level_perturbation_predicted_vs_observed_{source_name}_{group_name}.png",
                    f"Predicted vs Observed by Level and Perturbation ({_exp5_source_label(source_name)}, {group_name})",
                    actual_label=_exp5_target_label("accuracy_drop"),
                    metadata=_plot_metadata(
                        experiment="5",
                        what="Predicted overlap and observed accuracy drop split by level and perturbation",
                        aggregation="grouped by level and perturbation",
                        x="perturbation type",
                        y=_exp5_target_label("accuracy_drop"),
                        note=f"Source={_exp5_source_label(source_name)}; group={group_name}",
                        profile=profile,
                    ),
                )
    if exhaustive and exp5_sample_by_group and exp5_summary:
        analysis_groups = exp5_summary.get("analysis_groups", _FILTER_ANALYSIS_ORDER)
        primary_group = exp5_summary.get("primary_group", "late")
        for source_name, source_groups in exp5_sample_by_group.items():
            if source_name.endswith("_raw"):
                continue
            for group_name in analysis_groups:
                sample_pairs = source_groups.get(group_name)
                if not sample_pairs:
                    continue
                if source_name in {"image_space", "vision_feature_space"} and group_name == primary_group:
                    continue
                source_summary = exp5_summary.get(source_name, {}).get("group_summaries", {}).get(group_name, {})
                plot_overlap_scatter(
                    sample_pairs,
                    source_summary.get("pearson_r_sample", 0.0),
                    plots_dir / f"exp5_overlap_scatter_{source_name}_{group_name}_sample.png",
                    title=f"Overlap vs Accuracy Drop ({_exp5_source_label(source_name)}, {group_name}, per sample)",
                    point_size=20,
                    alpha=0.45,
                    max_points=sample_scatter_limit,
                    y_label=_exp5_target_label("accuracy_drop"),
                    metadata=_plot_metadata(
                        experiment="5",
                        what="Predicted overlap vs observed accuracy drop",
                        aggregation="per sample",
                        x="predicted spectral overlap S_pred",
                        y=_exp5_target_label("accuracy_drop"),
                        note=f"Source={_exp5_source_label(source_name)}; group={group_name}",
                        profile=profile,
                    ),
                )
    if exhaustive and exp5_grouped_by_group_and_target and exp5_summary:
        primary_group = exp5_summary.get("primary_group", "late")
        extra_targets = [target for target in exp5_summary.get("targets", []) if target != "accuracy_drop"]
        for source_name, source_groups in exp5_grouped_by_group_and_target.items():
            if source_name.endswith("_raw"):
                continue
            primary_group_targets = source_groups.get(primary_group, {})
            source_target_summaries = (
                exp5_summary.get(source_name, {})
                .get("primary_group_target_summaries", {})
            )
            for target_name in extra_targets:
                grouped_pairs = primary_group_targets.get(target_name)
                target_summary = source_target_summaries.get(target_name, {})
                if grouped_pairs:
                    plot_overlap_scatter(
                        grouped_pairs,
                        target_summary.get("pearson_r_grouped", 0.0),
                        plots_dir / f"exp5_overlap_scatter_{source_name}_{primary_group}_{target_name}.png",
                        title=f"Overlap vs {_exp5_target_label(target_name)} ({_exp5_source_label(source_name)}, {primary_group})",
                        y_label=_exp5_target_label(target_name),
                        scale_mode="zscore" if target_name == "loglik_erosion" else "raw",
                        metadata=_plot_metadata(
                            experiment="5",
                            what=f"Predicted overlap vs {_exp5_target_label(target_name)}",
                            aggregation="grouped by level and perturbation",
                            x="predicted spectral overlap S_pred",
                            y=_exp5_target_label(target_name),
                            note=(
                                f"Source={_exp5_source_label(source_name)}; group={primary_group}; "
                                "axes z-scored for display only"
                                if target_name == "loglik_erosion"
                                else f"Source={_exp5_source_label(source_name)}; group={primary_group}"
                            ),
                            profile=profile,
                        ),
                    )
                    plot_overlap_scatter(
                        grouped_pairs,
                        target_summary.get("comparable_bridge_grouped", {}).get("pearson_r", 0.0),
                        plots_dir / f"exp5_bridge_scatter_{source_name}_{primary_group}_{target_name}.png",
                        title=f"Comparable Bridge: Overlap vs {_exp5_target_label(target_name)} ({_exp5_source_label(source_name)}, {primary_group})",
                        y_label=_exp5_target_label(target_name),
                        scale_mode="bridge",
                        metadata=_plot_metadata(
                            experiment="5",
                            what=f"Comparable-bridge view of overlap vs {_exp5_target_label(target_name)}",
                            aggregation="grouped by level and perturbation",
                            x="zscore(log1p(predicted spectral overlap))",
                            y=f"zscore({_exp5_target_label(target_name).lower()})",
                            note=f"Source={_exp5_source_label(source_name)}; group={primary_group}; raw overlap results are still preserved separately",
                            profile=profile,
                        ),
                    )
                    _plot_exp5_heatmaps(
                        grouped_pairs,
                        plots_dir / f"exp5_grouped_heatmap_{source_name}_{primary_group}_{target_name}.png",
                        f"Grouped Overlap vs {_exp5_target_label(target_name)} ({_exp5_source_label(source_name)}, {primary_group})",
                        actual_label=_exp5_target_label(target_name),
                        metadata=_plot_metadata(
                            experiment="5",
                            what=f"Grouped predicted overlap, observed {_exp5_target_label(target_name)}, and residual",
                            aggregation="level x perturbation average over samples",
                            x="perturbation type",
                            y="task level",
                            note=f"Source={_exp5_source_label(source_name)}; group={primary_group}",
                            profile=profile,
                        ),
                    )
                    _plot_exp5_level_perturbation_comparison(
                        grouped_pairs,
                        plots_dir / f"exp5_level_perturbation_predicted_vs_observed_{source_name}_{primary_group}_{target_name}.png",
                        f"Predicted vs Observed by Level and Perturbation ({_exp5_source_label(source_name)}, {primary_group}, {_exp5_target_label(target_name)})",
                        actual_label=_exp5_target_label(target_name),
                        metadata=_plot_metadata(
                            experiment="5",
                            what=f"Predicted overlap and observed {_exp5_target_label(target_name)} split by level and perturbation",
                            aggregation="grouped by level and perturbation",
                            x="perturbation type",
                            y=_exp5_target_label(target_name),
                            note=f"Source={_exp5_source_label(source_name)}; group={primary_group}",
                            profile=profile,
                        ),
                    )
    if exhaustive and exp5_sample_by_group_and_target and exp5_summary:
        primary_group = exp5_summary.get("primary_group", "late")
        extra_targets = [target for target in exp5_summary.get("targets", []) if target != "accuracy_drop"]
        for source_name, source_groups in exp5_sample_by_group_and_target.items():
            if source_name.endswith("_raw"):
                continue
            primary_group_targets = source_groups.get(primary_group, {})
            source_target_summaries = (
                exp5_summary.get(source_name, {})
                .get("primary_group_target_summaries", {})
            )
            for target_name in extra_targets:
                sample_pairs = primary_group_targets.get(target_name)
                target_summary = source_target_summaries.get(target_name, {})
                if not sample_pairs or target_summary.get("pearson_r_sample") is None:
                    continue
                plot_overlap_scatter(
                    sample_pairs,
                    target_summary.get("pearson_r_sample", 0.0),
                    plots_dir / f"exp5_overlap_scatter_{source_name}_{primary_group}_{target_name}_sample.png",
                    title=f"Overlap vs {_exp5_target_label(target_name)} ({_exp5_source_label(source_name)}, {primary_group}, per sample)",
                    point_size=20,
                    alpha=0.45,
                    max_points=sample_scatter_limit,
                    y_label=_exp5_target_label(target_name),
                    scale_mode="zscore" if target_name == "loglik_erosion" else "raw",
                    metadata=_plot_metadata(
                        experiment="5",
                        what=f"Predicted overlap vs {_exp5_target_label(target_name)}",
                        aggregation="per sample",
                        x="predicted spectral overlap S_pred",
                        y=_exp5_target_label(target_name),
                        note=(
                            f"Source={_exp5_source_label(source_name)}; group={primary_group}; "
                            "axes z-scored for display only"
                            if target_name == "loglik_erosion"
                            else f"Source={_exp5_source_label(source_name)}; group={primary_group}"
                        ),
                        profile=profile,
                    ),
                )
                plot_overlap_scatter(
                    sample_pairs,
                    target_summary.get("comparable_bridge_sample", {}).get("pearson_r", 0.0),
                    plots_dir / f"exp5_bridge_scatter_{source_name}_{primary_group}_{target_name}_sample.png",
                    title=f"Comparable Bridge: Overlap vs {_exp5_target_label(target_name)} ({_exp5_source_label(source_name)}, {primary_group}, per sample)",
                    point_size=20,
                    alpha=0.45,
                    max_points=sample_scatter_limit,
                    y_label=_exp5_target_label(target_name),
                    scale_mode="bridge",
                    metadata=_plot_metadata(
                        experiment="5",
                        what=f"Comparable-bridge view of overlap vs {_exp5_target_label(target_name)}",
                        aggregation="per sample",
                        x="zscore(log1p(predicted spectral overlap))",
                        y=f"zscore({_exp5_target_label(target_name).lower()})",
                        note=f"Source={_exp5_source_label(source_name)}; group={primary_group}; raw overlap results are still preserved separately",
                        profile=profile,
                    ),
                )

    exp6_summary = _load_json(results_dir / "exp6" / "summary.json")
    exp6_samples = _load_jsonl(results_dir / "exp6" / "per_sample.jsonl")
    if exp6_summary:
        plot_segmentation_granularity(exp6_summary, plots_dir / "exp6_segmentation.png")
    if exp6_samples:
        for model_name in sorted({name for record in exp6_samples for name in record.get("models", {})}):
            levels, perturbations, matrix = _exp6_level_perturbation_matrix(exp6_samples, model_name)
            if matrix.size:
                _plot_heatmap(
                    matrix,
                    [_level_label(level) for level in levels],
                    perturbations,
                    plots_dir / f"exp6_level_perturbation_{_safe_name(model_name)}.png",
                    f"mIoU Drop by Level and Perturbation ({model_name})",
                    "Mean mIoU Drop",
                    cmap="YlOrRd",
                    metadata=_plot_metadata(
                        experiment="6",
                        what=f"Mean mIoU drop by perturbation for {model_name}",
                        aggregation="level x perturbation average over samples",
                        x="perturbation type",
                        y="segmentation hierarchy level",
                        profile=profile,
                    ),
                )
            if exhaustive:
                sample_ids, levels, matrix = _exp6_sample_level_matrix(exp6_samples, model_name)
                sample_ids, matrix = _sort_metric_rows(sample_ids, matrix, max_rows=10)
                if matrix.size:
                    _plot_heatmap(
                        matrix,
                        [_short_sample_id(sample_id) for sample_id in sample_ids],
                        [_level_label(level) for level in levels],
                        plots_dir / f"exp6_sample_level_{_safe_name(model_name)}.png",
                        f"Per-Sample mIoU Drop ({model_name})",
                        "Mean mIoU Drop",
                        cmap="PuRd",
                        metadata=_plot_metadata(
                            experiment="6",
                            what=f"Per-image mIoU drop for {model_name}",
                            aggregation="sample-wise average over perturbations",
                            x="segmentation hierarchy level",
                            y="selected images",
                            selection="top 10 images by average mIoU drop",
                            profile=profile,
                        ),
                    )

    _write_plot_manifest(plots_dir, profile)
    logger.info("All plots generated in %s (profile=%s, plots=%d)", plots_dir, profile, len(_PLOT_MANIFEST))

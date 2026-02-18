"""Spectral overlap prediction (Theorem 1).

Validates the core theorem: perturbation sensitivity is predicted by
the spectral overlap between the task-specific filter W_t(omega) and the
perturbation's spectral signature DeltaF(omega):

    S_pred = integral |W_t(omega)|^2 * |DeltaF(omega)|^2 d_omega

If actual sensitivity correlates with S_pred (Pearson r > 0.7), the
theory is validated.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from .spectral import compute_spectral_overlap

logger = logging.getLogger(__name__)


def predict_sensitivities(
    W_t_per_level: Dict[str, np.ndarray],
    perturbation_spectra: List[Tuple[str, np.ndarray]],
) -> Dict[str, List[Tuple[str, float]]]:
    """Predict sensitivity for each (level, perturbation) pair.

    Args:
        W_t_per_level: ``{level_key: W_t_array}`` from Experiment 2.
        perturbation_spectra: ``[(pert_name, delta_f)]`` spectral signatures.

    Returns:
        ``{level_key: [(pert_name, predicted_sensitivity), ...]}``
    """
    predictions: Dict[str, List[Tuple[str, float]]] = {}
    for level_key, W_t in W_t_per_level.items():
        level_preds = []
        for pert_name, delta_f in perturbation_spectra:
            s_pred = compute_spectral_overlap(W_t, delta_f)
            level_preds.append((pert_name, s_pred))
        predictions[level_key] = level_preds
    return predictions


def load_actual_sensitivities_from_exp1(
    exp1_summary: dict,
) -> Dict[str, Dict[str, float]]:
    """Extract actual mean accuracy drops from Exp 1 summary.

    Returns:
        ``{level_key: {pert_name: mean_drop}}``
    """
    result: Dict[str, Dict[str, float]] = {}
    per_level_pert = exp1_summary.get("per_level_perturbation", {})
    for level_key, pert_data in per_level_pert.items():
        result[level_key] = {}
        for pert_name, stats in pert_data.items():
            result[level_key][pert_name] = stats.get("mean_drop", 0.0)
    return result


def match_predictions_to_actuals(
    predictions: Dict[str, List[Tuple[str, float]]],
    actuals: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Align predicted and actual sensitivities into paired arrays.

    Only includes (level, perturbation) pairs present in both.

    Returns:
        ``(predicted_array, actual_array, labels)``
        where labels are ``"level|perturbation"`` strings.
    """
    pred_list = []
    actual_list = []
    labels = []

    for level_key in sorted(predictions.keys()):
        actual_for_level = actuals.get(level_key, {})
        for pert_name, s_pred in predictions[level_key]:
            # Find matching actual (may need fuzzy match on pert name)
            actual_val = actual_for_level.get(pert_name)
            if actual_val is not None:
                pred_list.append(s_pred)
                actual_list.append(actual_val)
                labels.append(f"{level_key}|{pert_name}")

    return np.array(pred_list), np.array(actual_list), labels

"""Shared hypothesis-gate thresholds for experiment summaries."""

from __future__ import annotations

GATES = {
    "monotonicity_kendall_tau_min": 0.6,
    "overlap_law_r_min": 0.7,
    "overlap_law_min_n": 30,
    "dual_force_beta_p_max": 0.05,
    "long_format_dual_force_beta_p_max": 0.05,
    "perturbation_sign_stability_min_fraction": 0.6,
    "controlled_r_min": 0.7,
    "strong_spearman_rho_min": 0.8,
    "secondary_spearman_rho_min": 0.6,
    "overlap_law_sample_r_min": 0.3,
    "segmentation_baseline_abs_max": 0.5,
}

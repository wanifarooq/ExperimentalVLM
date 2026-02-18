"""Analysis modules for spectral, drift, overlap, and statistical analysis."""

from .spectral import (
    attention_to_spatial_grid,
    compute_attention_power_spectrum,
    compute_attention_power_spectrum_multi,
    compute_effective_bandwidth,
    compute_filter_W_t,
    compute_spectral_overlap,
)
from .drift import (
    compute_amplification_ratio,
    compute_band_drift,
    compute_cosine_drift_per_token,
    compute_scalar_drift,
)
from .overlap import (
    load_actual_sensitivities_from_exp1,
    match_predictions_to_actuals,
    predict_sensitivities,
)
from .statistics import (
    bootstrap_ci,
    cohens_d,
    monotonicity_test,
    one_way_anova,
    paired_ttest,
    pearson_correlation,
    spearman_correlation,
)

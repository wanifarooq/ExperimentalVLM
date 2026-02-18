"""Plotting module for publication-quality figures."""

from .plots import (
    generate_all_plots,
    plot_amplification_heatmap,
    plot_attention_power_spectrum,
    plot_effective_bandwidth,
    plot_frequency_threshold_curves,
    plot_granularity_curves,
    plot_overlap_scatter,
    plot_segmentation_granularity,
)

__all__ = [
    "plot_granularity_curves",
    "plot_attention_power_spectrum",
    "plot_effective_bandwidth",
    "plot_amplification_heatmap",
    "plot_frequency_threshold_curves",
    "plot_overlap_scatter",
    "plot_segmentation_granularity",
    "generate_all_plots",
]

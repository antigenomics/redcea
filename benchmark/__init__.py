"""Benchmark helpers built on top of the existing RedCEA infrastructure."""

from .evaluation import (
    compute_cross_donor_overlap,
    compute_known_yfv_recovery,
    compute_vdjdb_metrics,
    compute_yfv_cluster_enrichment,
    summarize_yfv_enrichment,
)
from .grids import EXTENDED_METHODS, PRIORITY_METHODS, get_enabled_methods, get_method_grid
from .plotting import (
    plot_cross_donor_overlap,
    plot_density_by_length,
    plot_final_method_summary,
    plot_vdjdb_benchmark,
    plot_vdjdb_signal_concentration,
    plot_yfv_enrichment_summary,
    plot_yfv_known_recovery_heatmap,
)
from .prepare_datasets import build_source_manifest, write_processed_datasets
from .run_benchmark import build_execution_manifest, build_grid_manifest, consolidate_run_metadata, execute_single_manifest_row
from .runner import ClusteringBenchmarkRunner, extract_embeddings
from .workflows import (
    build_final_method_comparison,
    compute_density_by_length,
    load_assignment_tables,
    run_cross_donor_overlap_evaluation,
    run_density_analysis,
    run_vdjdb_evaluation,
    run_yfv_enrichment_evaluation,
    run_yfv_known_clonotype_evaluation,
    run_summary_and_method_selection,
    write_summary_report,
)

__all__ = [
    "ClusteringBenchmarkRunner",
    "extract_embeddings",
    "PRIORITY_METHODS",
    "EXTENDED_METHODS",
    "get_enabled_methods",
    "get_method_grid",
    "build_source_manifest",
    "write_processed_datasets",
    "build_grid_manifest",
    "build_execution_manifest",
    "consolidate_run_metadata",
    "execute_single_manifest_row",
    "compute_vdjdb_metrics",
    "compute_yfv_cluster_enrichment",
    "summarize_yfv_enrichment",
    "compute_known_yfv_recovery",
    "compute_cross_donor_overlap",
    "plot_density_by_length",
    "plot_vdjdb_benchmark",
    "plot_vdjdb_signal_concentration",
    "plot_yfv_enrichment_summary",
    "plot_yfv_known_recovery_heatmap",
    "plot_cross_donor_overlap",
    "plot_final_method_summary",
    "load_assignment_tables",
    "compute_density_by_length",
    "run_density_analysis",
    "run_vdjdb_evaluation",
    "run_yfv_enrichment_evaluation",
    "run_yfv_known_clonotype_evaluation",
    "run_cross_donor_overlap_evaluation",
    "run_summary_and_method_selection",
    "build_final_method_comparison",
    "write_summary_report",
]

"""Legacy analysis utility surface preserved for notebooks and ad hoc scripts.

This module re-exports notebook-oriented helpers from smaller analysis submodules.
Prefer importing from the specific modules in ``redcea.analysis`` for new code.
"""

from redcea.analysis.cluster_utils import compute_cluster_summary, get_cluster_usage, merge_clusters_by_shared_cdr3
from redcea.analysis.io import get_clonotypes, get_sample_info
from redcea.analysis.matching import get_cluster_matches_vdjdb, has_vdjdb_match
from redcea.analysis.pgen import compute_pgen_pool
from redcea.analysis.ranking import compute_topsis_score
from redcea.plotting import plot_logo, plot_volcano

__all__ = [
    "compute_pgen_pool",
    "compute_cluster_summary",
    "compute_topsis_score",
    "get_clonotypes",
    "get_cluster_matches_vdjdb",
    "get_cluster_usage",
    "get_sample_info",
    "has_vdjdb_match",
    "merge_clusters_by_shared_cdr3",
    "plot_logo",
    "plot_volcano",
]

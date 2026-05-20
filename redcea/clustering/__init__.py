# tcrempnet/clustering/__init__.py

from .preprocess import standardize_data, apply_pca, prepare_data_for_clustering

from .eps_estimation import (
    estimate_dbscan_eps,
    cluster_dbscan,
    cluster_dbscan_with_filter,
    estimate_eps_by_group_from_sample,
    eps_per_point_from_group_id,
    estimate_eps_by_group_flexible
)

from .cdr3_grouping import (
    compute_cdr3_len,
    build_len_to_group_id,
    map_len_to_group_id,
)

try:
    from .faiss_cache import (
        compute_split_knn,
        compute_blockwise_knn_merged,  # compatibility alias for split knn
    )
    from .knn_merge import build_joint_knn_from_split
    from .joint_knn import JointKnnArtifacts, build_joint_knn_artifacts
except Exception:  # pragma: no cover - optional FAISS-backed imports
    compute_split_knn = None
    compute_blockwise_knn_merged = None
    build_joint_knn_from_split = None
    JointKnnArtifacts = None
    build_joint_knn_artifacts = None

from .vdbscan import vdbscan_from_knn

try:
    from .pipeline_joint import run_joint_vdbscan
except Exception:  # pragma: no cover - optional FAISS-backed imports
    run_joint_vdbscan = None

from .cluster_methods import (
    run_dbscan_clustering,
    run_leiden_clustering,
    run_joint_clustering,
    hierarchical_leiden_clustering,
    hierarchical_leiden_dbscan_clustering,
)

__all__ = [
    # preprocess
    "standardize_data",
    "apply_pca",
    "prepare_data_for_clustering",

    # eps
    "estimate_dbscan_eps",
    "cluster_dbscan",
    "cluster_dbscan_with_filter",
    "estimate_eps_by_group_from_sample",
    "eps_per_point_from_group_id",
    "estimate_eps_by_group_flexible",

    # grouping
    "compute_cdr3_len",
    "build_len_to_group_id",
    "map_len_to_group_id",

    # faiss knn
    "compute_split_knn",
    "compute_blockwise_knn_merged",

    # knn merge
    "build_joint_knn_from_split",
    "JointKnnArtifacts",
    "build_joint_knn_artifacts",

    # vdbscan
    "vdbscan_from_knn",
    "run_joint_vdbscan",

    # classic methods
    "run_dbscan_clustering",
    "run_leiden_clustering",
    "run_joint_clustering",
    "hierarchical_leiden_clustering",
    "hierarchical_leiden_dbscan_clustering",
]

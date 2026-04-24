from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .cdr3_grouping import compute_cdr3_len, build_len_to_group_id, map_len_to_group_id
from .eps_estimation import estimate_eps_by_group_from_sample, eps_per_point_from_group_id
from .faiss_cache import compute_split_knn
from .knn_merge import build_joint_knn_from_split
from .vdbscan import vdbscan_from_knn


@dataclass
class JointVDBScanDiagnostics:
    len_to_gid: Dict[int, int]
    eps_by_gid: Dict[int, float]
    sample_group_id: np.ndarray
    bg_group_id: np.ndarray
    eps_i_all: np.ndarray
    distance_scale: str  # "l2"


def run_joint_vdbscan(
    sample_df: pd.DataFrame,
    bg_df: pd.DataFrame,
    sample_X: np.ndarray,
    bg_X: np.ndarray,
    cdr3_col: str,
    k_neighbors: int,
    kth_neighbor_for_eps: int = 4,
    min_samples: int = 5,
    min_frac: float = 0.05,
    sym_rule: str = "asymmetric",
    output_dir: Optional[str] = None,
    sample_index_path: Optional[str] = None,
    bg_index_path: Optional[str] = None,
    rebuild_sample: bool = False,
    rebuild_bg: bool = False,
    rebuild_sample_knn: bool = False,
    rebuild_bg_knn: bool = False,
    save_blocks: bool = True,
    nproc: int = 8,
    mmap_cached: bool = False,
) -> Tuple[np.ndarray, JointVDBScanDiagnostics]:
    """
    Full joint clustering "as before": cluster [sample + bg] together.
    eps is learned per CDR3-length group using ONLY sample.

    Important:
    - Distances are L2 everywhere.
    - No silent fixes: NaN/inf in inputs -> should be handled upstream; faiss_cache will raise.
    """

    if cdr3_col not in sample_df.columns:
        raise ValueError(f"cdr3_col='{cdr3_col}' not in sample_df columns")
    if cdr3_col not in bg_df.columns:
        raise ValueError(f"cdr3_col='{cdr3_col}' not in bg_df columns")

    # 1) build mapping len -> group_id from SAMPLE only
    sample_len = compute_cdr3_len(sample_df[cdr3_col])
    len_to_gid = build_len_to_group_id(sample_len, min_frac=min_frac)

    # 2) map group_id for sample and bg (bg missing lens -> nearest + warning by default)
    sample_gid = map_len_to_group_id(sample_len, len_to_gid, unknown_len_policy="nearest")
    bg_len = compute_cdr3_len(bg_df[cdr3_col])
    bg_gid = map_len_to_group_id(bg_len, len_to_gid, unknown_len_policy="nearest")

    # 3) compute split kNN (self cached, cross on the fly), distances are L2
    dist_ss, ind_ss, dist_bb, ind_bb, dist_sb, ind_sb, dist_bs, ind_bs = compute_split_knn(
        bg=bg_X,
        sample=sample_X,
        k_neighbors=k_neighbors,
        bg_index_path=bg_index_path,
        sample_index_path=sample_index_path,
        rebuild_bg=rebuild_bg,
        rebuild_sample=rebuild_sample,
        save_blocks=save_blocks,
        output_dir=output_dir,
        nproc=nproc,
        mmap_cached=mmap_cached,
        rebuild_bg_knn=rebuild_bg_knn,
        rebuild_sample_knn=rebuild_sample_knn,
    )

    # 4) estimate eps per group using SAMPLE only (from sample-sample kNN)
    eps_by_gid = estimate_eps_by_group_from_sample(
        sample_group_id=sample_gid,
        sample_ss_distances_l2=dist_ss,
        kth_neighbor=kth_neighbor_for_eps,
    )

    # 5) eps for all points
    gid_all = np.concatenate([sample_gid, bg_gid], axis=0)
    eps_i_all = eps_per_point_from_group_id(gid_all, eps_by_gid)

    logging.info(
        "eps_i_all stats (L2): min=%.6f, median=%.6f, max=%.6f",
        float(np.min(eps_i_all)), float(np.median(eps_i_all)), float(np.max(eps_i_all))
    )

    # 6) build joint kNN for concatenated [sample, bg]
    dist_joint, ind_joint = build_joint_knn_from_split(
        dist_ss=dist_ss, ind_ss=ind_ss,
        dist_bb=dist_bb, ind_bb=ind_bb,
        dist_sb=dist_sb, ind_sb=ind_sb,
        dist_bs=dist_bs, ind_bs=ind_bs,
        k_out=k_neighbors + 1,
    )

    # 7) run vDBSCAN on joint graph
    labels_joint = vdbscan_from_knn(
        knn_indices=ind_joint,
        knn_distances_l2=dist_joint,
        eps_i_l2=eps_i_all,
        min_samples=min_samples,
        sym_rule=sym_rule,
    )

    diag = JointVDBScanDiagnostics(
        len_to_gid=len_to_gid,
        eps_by_gid=eps_by_gid,
        sample_group_id=sample_gid,
        bg_group_id=bg_gid,
        eps_i_all=eps_i_all,
        distance_scale="l2",
    )
    return labels_joint, diag

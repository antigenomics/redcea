from __future__ import annotations
import numpy as np


def merge_two_knn_topk(
    dist_a: np.ndarray, idx_a: np.ndarray,
    dist_b: np.ndarray, idx_b: np.ndarray,
    k_out: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge two kNN lists and keep top-k by smallest distance.
    Vectorized with argpartition + argsort.
    """
    if dist_a.shape != idx_a.shape or dist_b.shape != idx_b.shape:
        raise ValueError("dist and idx must have same shapes for each source")
    if dist_a.shape[0] != dist_b.shape[0]:
        raise ValueError("Both sources must have same N rows")

    D = np.concatenate([dist_a, dist_b], axis=1)
    I = np.concatenate([idx_a, idx_b], axis=1)

    if k_out > D.shape[1]:
        raise ValueError(f"k_out={k_out} > merged_k={D.shape[1]}")

    sel = np.argpartition(D, kth=k_out - 1, axis=1)[:, :k_out]
    Dsel = np.take_along_axis(D, sel, axis=1)
    Isel = np.take_along_axis(I, sel, axis=1)

    order = np.argsort(Dsel, axis=1)
    Dsel = np.take_along_axis(Dsel, order, axis=1)
    Isel = np.take_along_axis(Isel, order, axis=1)

    return Dsel.astype(np.float32, copy=False), Isel.astype(np.int32, copy=False)


def build_joint_knn_from_split(
    dist_ss: np.ndarray, ind_ss: np.ndarray,
    dist_bb: np.ndarray, ind_bb: np.ndarray,
    dist_sb: np.ndarray, ind_sb: np.ndarray,  # sample -> bg (idx in bg space)
    dist_bs: np.ndarray, ind_bs: np.ndarray,  # bg -> sample (idx in sample space)
    k_out: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build joint kNN for concatenated array [sample, bg].

    For sample rows: merge (ss) + (sb with bg offset)
    For bg rows:     merge (bb with bg offset) + (bs)
    """
    n_sample = dist_ss.shape[0]
    n_bg = dist_bb.shape[0]

    if dist_ss.shape != ind_ss.shape or dist_bb.shape != ind_bb.shape:
        raise ValueError("self knn dist/idx shapes mismatch")
    if dist_sb.shape != ind_sb.shape or dist_bs.shape != ind_bs.shape:
        raise ValueError("cross knn dist/idx shapes mismatch")
    if dist_sb.shape[0] != n_sample or dist_bs.shape[0] != n_bg:
        raise ValueError("cross knn row counts mismatch")
    if dist_sb.shape[1] < k_out or dist_bs.shape[1] < k_out or dist_ss.shape[1] < k_out or dist_bb.shape[1] < k_out:
        raise ValueError("k_out is larger than available k in inputs")

    sb_idx_global = (ind_sb + n_sample).astype(np.int32, copy=False)
    dist_sample, idx_sample = merge_two_knn_topk(dist_ss, ind_ss, dist_sb, sb_idx_global, k_out=k_out)

    bb_idx_global = (ind_bb + n_sample).astype(np.int32, copy=False)
    dist_bg, idx_bg = merge_two_knn_topk(dist_bb, bb_idx_global, dist_bs, ind_bs, k_out=k_out)

    dist_joint = np.vstack([dist_sample, dist_bg]).astype(np.float32, copy=False)
    idx_joint = np.vstack([idx_sample, idx_bg]).astype(np.int32, copy=False)

    if dist_joint.shape != (n_sample + n_bg, k_out) or idx_joint.shape != (n_sample + n_bg, k_out):
        raise RuntimeError("Joint kNN has unexpected shape")

    return dist_joint, idx_joint

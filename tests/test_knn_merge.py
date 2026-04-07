import numpy as np
import pytest

from redcea.clustering.knn_merge import build_joint_knn_from_split, merge_two_knn_topk


def test_merge_two_knn_topk_keeps_smallest_distances():
    dist_a = np.array([[0.3, 0.5]], dtype=np.float32)
    idx_a = np.array([[1, 2]], dtype=np.int32)
    dist_b = np.array([[0.1, 0.4]], dtype=np.float32)
    idx_b = np.array([[3, 4]], dtype=np.int32)

    distances, indices = merge_two_knn_topk(dist_a, idx_a, dist_b, idx_b, k_out=3)

    assert distances.tolist() == [[0.1, 0.3, 0.4]]
    assert indices.tolist() == [[3, 1, 4]]


def test_build_joint_knn_from_split_offsets_background_indices():
    dist_ss = np.array([[0.0, 0.2], [0.0, 0.1]], dtype=np.float32)
    ind_ss = np.array([[0, 1], [1, 0]], dtype=np.int32)
    dist_bb = np.array([[0.0, 0.4]], dtype=np.float32)
    ind_bb = np.array([[0, 0]], dtype=np.int32)
    dist_sb = np.array([[0.3, 0.5], [0.2, 0.6]], dtype=np.float32)
    ind_sb = np.array([[0, 0], [0, 0]], dtype=np.int32)
    dist_bs = np.array([[0.25, 0.35]], dtype=np.float32)
    ind_bs = np.array([[1, 0]], dtype=np.int32)

    distances, indices = build_joint_knn_from_split(
        dist_ss, ind_ss, dist_bb, ind_bb, dist_sb, ind_sb, dist_bs, ind_bs, k_out=2
    )

    assert distances.shape == (3, 2)
    assert indices.shape == (3, 2)
    assert indices[0].tolist() == [0, 1]
    assert indices[1].tolist() == [1, 0]
    assert indices[2].tolist() == [2, 1]


def test_merge_two_knn_topk_rejects_too_large_k():
    with pytest.raises(ValueError):
        merge_two_knn_topk(
            np.array([[0.1]], dtype=np.float32),
            np.array([[0]], dtype=np.int32),
            np.array([[0.2]], dtype=np.float32),
            np.array([[1]], dtype=np.int32),
            k_out=3,
        )

from __future__ import annotations

import numpy as np
import pandas as pd

from redcea.auxiliary_cluster_metrics import (
    AUXILIARY_CLUSTER_METRIC_COLUMNS,
    _compute_odds_metrics,
    append_auxiliary_cluster_metrics,
)


def test_append_auxiliary_cluster_metrics_adds_support_and_density_scores():
    cluster_df = pd.DataFrame(
        {
            "clone_id": [
                "s_0",
                "s_1",
                "s_2",
                "s_3",
                "s_4",
                "b_0",
                "b_1",
                "b_2",
                "b_3",
                "s_5",
                "s_6",
                "b_4",
                "b_5",
            ],
            "cluster_id": [10, 10, 10, 10, 20, 30, 30, 30, 30, 40, 40, 40, 40],
            "source": [
                "sample",
                "sample",
                "sample",
                "sample",
                "sample",
                "background",
                "background",
                "background",
                "background",
                "sample",
                "sample",
                "background",
                "background",
            ],
        }
    )
    summary_df = pd.DataFrame(
        {
            "cluster_id": [10, 20, 30, 40],
            "cluster_size": [4, 1, 4, 4],
            "sample": [4, 1, 0, 2],
            "background": [0, 0, 4, 2],
            "enrichment_pvalue_zbinom": [1e-8, 1e-6, 0.2, 0.01],
            "enrichment_fdr_zbinom": [1e-6, 1e-5, 0.4, 0.03],
            "log_fold_change": [1.0, 2.0, -1.0, 0.2],
        }
    )

    sample_knn_indices = np.array(
        [
            [0, 1, 2, 5],
            [1, 0, 3, 5],
            [2, 3, 0, 6],
            [3, 2, 1, 6],
            [4, 0, 1, 5],
            [5, 0, 6, 4],
            [6, 1, 5, 4],
        ],
        dtype=np.int32,
    )
    sample_knn_distances = np.array(
        [
            [0.0, 0.10, 0.20, 1.50],
            [0.0, 0.10, 0.20, 1.60],
            [0.0, 0.10, 0.20, 1.70],
            [0.0, 0.10, 0.20, 1.80],
            [0.0, 0.15, 0.16, 1.90],
            [0.0, 0.20, 0.80, 0.90],
            [0.0, 0.25, 0.80, 0.90],
        ],
        dtype=np.float32,
    )

    result = append_auxiliary_cluster_metrics(
        summary_df,
        cluster_df,
        total_sample=6,
        total_background=6,
        sample_knn_indices=sample_knn_indices,
        sample_knn_distances=sample_knn_distances,
    )

    for column in AUXILIARY_CLUSTER_METRIC_COLUMNS:
        assert column in result.columns

    cluster10 = result.loc[result["cluster_id"] == 10].iloc[0]
    cluster20 = result.loc[result["cluster_id"] == 20].iloc[0]
    cluster30 = result.loc[result["cluster_id"] == 30].iloc[0]
    cluster40 = result.loc[result["cluster_id"] == 40].iloc[0]

    assert cluster10["log2fc_smooth"] > 0
    assert cluster10["density_validity_full"] == cluster10["density_validity"]
    assert cluster10["density_validity"] < 0
    assert bool(cluster10["has_density_validity"]) is True
    assert cluster10["log2_odds"] > 0
    assert cluster10["odds_ratio"] > 1.0
    assert cluster10["dbcv_with_odds"] == 0.0
    assert cluster30["log2fc_smooth"] < 0
    assert cluster40["density_validity"] > 0
    assert cluster40["density_validity_full"] == cluster40["density_validity"]
    assert cluster40["log2_odds"] == 0.0
    assert cluster40["dbcv_with_odds"] == 0.0
    assert cluster10["enrichment_support_score"] > cluster20["enrichment_support_score"]
    assert np.isnan(cluster20["density_validity"])
    assert np.isnan(cluster20["density_validity_enriched_only"])
    assert bool(cluster20["has_density_validity"]) is False
    assert cluster10["density_validity_enriched_only"] < 0
    assert cluster40["density_validity_enriched_only"] > 0
    assert np.isnan(cluster30["density_validity_enriched_only"])


def test_append_auxiliary_cluster_metrics_handles_missing_knn():
    cluster_df = pd.DataFrame(
        {
            "clone_id": ["s_0", "s_1", "b_0", "b_1"],
            "cluster_id": [1, 1, 2, 2],
            "source": ["sample", "sample", "background", "background"],
        }
    )
    summary_df = pd.DataFrame(
        {
            "cluster_id": [1, 2],
            "cluster_size": [2, 2],
            "sample": [2, 0],
            "background": [0, 2],
            "enrichment_pvalue_zbinom": [1e-6, 0.5],
            "enrichment_fdr_zbinom": [1e-5, 0.5],
            "log_fold_change": [1.0, -1.0],
        }
    )

    result = append_auxiliary_cluster_metrics(
        summary_df,
        cluster_df,
        total_sample=2,
        total_background=2,
    )

    assert result["sample_usage"].notna().all()
    assert result["density_validity"].isna().all()
    assert result["density_validity_full"].isna().all()
    assert result["density_validity_enriched_only"].isna().all()
    assert not result["has_density_validity"].any()
    assert result["geometry_score"].isna().all()
    assert result["dbcv_with_odds"].isna().all()
    assert result["redcea_auxiliary_validity"].isna().all()


def test_compute_odds_metrics_produces_positive_dbcv_when_density_and_odds_are_positive():
    log2_odds, odds_ratio, dbcv_with_odds = _compute_odds_metrics(
        sample=np.array([10.0]),
        background=np.array([5.0]),
        total_sample=100,
        total_background=100,
        density_validity=np.array([0.4]),
    )

    assert log2_odds[0] > 0
    assert odds_ratio[0] > 1.0
    assert dbcv_with_odds[0] > 0.0


def test_compute_odds_metrics_zeroes_positive_odds_when_sample_support_is_zero():
    log2_odds, odds_ratio, dbcv_with_odds = _compute_odds_metrics(
        sample=np.array([0.0]),
        background=np.array([4.0]),
        total_sample=100,
        total_background=1000,
        density_validity=np.array([0.75]),
    )

    assert log2_odds[0] > 0
    assert odds_ratio[0] > 1.0
    assert dbcv_with_odds[0] == 0.0


def test_compute_odds_metrics_returns_nan_dbcv_when_density_validity_is_nan():
    _, _, dbcv_with_odds = _compute_odds_metrics(
        sample=np.array([2.0]),
        background=np.array([1.0]),
        total_sample=10,
        total_background=10,
        density_validity=np.array([np.nan]),
    )

    assert np.isnan(dbcv_with_odds[0])


def test_append_auxiliary_cluster_metrics_marks_sample_knn_closed_clusters_and_relaxed_candidates():
    cluster_df = pd.DataFrame(
        {
            "clone_id": ["s_0", "s_1", "s_2", "s_3", "s_4", "b_0"],
            "cluster_id": [1, 1, 1, 1, 1, 2],
            "source": ["sample", "sample", "sample", "sample", "sample", "background"],
        }
    )
    summary_df = pd.DataFrame(
        {
            "cluster_id": [1, 2],
            "cluster_size": [5, 1],
            "sample": [5, 0],
            "background": [0, 1],
            "enrichment_pvalue_zbinom": [1e-8, 0.8],
            "enrichment_fdr_zbinom": [1e-6, 0.8],
            "log_fold_change": [2.0, -1.0],
        }
    )
    sample_knn_indices = np.array(
        [
            [0, 1, 2],
            [1, 0, 2],
            [2, 1, 3],
            [3, 2, 4],
            [4, 3, 2],
        ],
        dtype=np.int32,
    )
    sample_knn_distances = np.array(
        [
            [0.0, 0.10, 0.15],
            [0.0, 0.10, 0.15],
            [0.0, 0.10, 0.15],
            [0.0, 0.10, 0.15],
            [0.0, 0.10, 0.15],
        ],
        dtype=np.float32,
    )

    result = append_auxiliary_cluster_metrics(
        summary_df,
        cluster_df,
        total_sample=5,
        total_background=10,
        sample_knn_indices=sample_knn_indices,
        sample_knn_distances=sample_knn_distances,
    )

    cluster1 = result.loc[result["cluster_id"] == 1].iloc[0]
    assert np.isnan(cluster1["density_validity"])
    assert np.isnan(cluster1["density_validity_enriched_only"])
    assert bool(cluster1["is_sample_knn_closed"]) is True
    assert bool(cluster1["is_good_candidate"]) is False
    assert bool(cluster1["is_good_candidate_relaxed"]) is True


def test_append_auxiliary_cluster_metrics_relaxed_candidate_still_requires_min_sample_support():
    cluster_df = pd.DataFrame(
        {
            "clone_id": ["s_0", "s_1", "s_2", "s_3", "b_0"],
            "cluster_id": [1, 1, 1, 1, 2],
            "source": ["sample", "sample", "sample", "sample", "background"],
        }
    )
    summary_df = pd.DataFrame(
        {
            "cluster_id": [1, 2],
            "cluster_size": [4, 1],
            "sample": [4, 0],
            "background": [0, 1],
            "enrichment_pvalue_zbinom": [1e-8, 0.8],
            "enrichment_fdr_zbinom": [1e-6, 0.8],
            "log_fold_change": [2.0, -1.0],
        }
    )
    sample_knn_indices = np.array(
        [
            [0, 1, 2],
            [1, 0, 2],
            [2, 1, 3],
            [3, 2, 1],
        ],
        dtype=np.int32,
    )
    sample_knn_distances = np.array(
        [
            [0.0, 0.10, 0.15],
            [0.0, 0.10, 0.15],
            [0.0, 0.10, 0.15],
            [0.0, 0.10, 0.15],
        ],
        dtype=np.float32,
    )

    result = append_auxiliary_cluster_metrics(
        summary_df,
        cluster_df,
        total_sample=4,
        total_background=10,
        sample_knn_indices=sample_knn_indices,
        sample_knn_distances=sample_knn_distances,
    )

    cluster1 = result.loc[result["cluster_id"] == 1].iloc[0]
    assert bool(cluster1["is_sample_knn_closed"]) is True
    assert bool(cluster1["is_good_candidate_relaxed"]) is False


def test_append_auxiliary_cluster_metrics_sets_enriched_only_nan_when_fewer_than_two_enriched_clusters_have_valid_geometry():
    cluster_df = pd.DataFrame(
        {
            "clone_id": ["s_0", "s_1", "s_2", "s_3", "b_0"],
            "cluster_id": [1, 1, 2, 3, 4],
            "source": ["sample", "sample", "sample", "sample", "background"],
        }
    )
    summary_df = pd.DataFrame(
        {
            "cluster_id": [1, 2, 3, 4],
            "cluster_size": [2, 1, 1, 1],
            "sample": [2, 1, 1, 0],
            "background": [0, 0, 0, 1],
            "enrichment_pvalue_zbinom": [1e-8, 1e-6, 0.3, 0.8],
            "enrichment_fdr_zbinom": [1e-6, 1e-5, 0.3, 0.8],
            "log_fold_change": [2.0, 1.0, -1.0, -1.0],
        }
    )
    sample_knn_indices = np.array(
        [
            [0, 1, 2],
            [1, 0, 2],
            [2, 0, 1],
            [3, 0, 1],
        ],
        dtype=np.int32,
    )
    sample_knn_distances = np.array(
        [
            [0.0, 0.10, 0.30],
            [0.0, 0.10, 0.30],
            [0.0, 0.30, 0.35],
            [0.0, 0.50, 0.60],
        ],
        dtype=np.float32,
    )

    result = append_auxiliary_cluster_metrics(
        summary_df,
        cluster_df,
        total_sample=4,
        total_background=1,
        sample_knn_indices=sample_knn_indices,
        sample_knn_distances=sample_knn_distances,
    )

    assert result["density_validity_full"].notna().any()
    assert result["density_validity_enriched_only"].isna().all()

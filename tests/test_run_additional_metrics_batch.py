from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_additional_metrics_batch import (
    build_run_level_table,
    compute_dense_redcea_scores,
    compute_lfc_mass_shift,
    run_matches_filters,
    select_run_dirs,
)


def test_run_matches_filters_uses_run_id_inference_for_yfv_and_vdjdb():
    manifest_df = pd.DataFrame(columns=["run_id", "dataset_mode"])

    assert run_matches_filters(
        run_id="yfv_P1_vdbscan_grid_0001",
        manifest_lookup=manifest_df,
        dataset_mode="yfv",
    )
    assert not run_matches_filters(
        run_id="yfv_P1_vdbscan_grid_0001",
        manifest_lookup=manifest_df,
        dataset_mode="vdjdb",
    )
    assert run_matches_filters(
        run_id="vdjdb_GILGFVFTL_vdbscan_grid_0001",
        manifest_lookup=manifest_df,
        dataset_mode="vdjdb",
    )


def test_run_matches_filters_falls_back_to_manifest_when_name_is_not_inferable():
    manifest_df = pd.DataFrame(
        [
            {"run_id": "custom_run_01", "dataset_mode": "yfv"},
            {"run_id": "custom_run_02", "dataset_mode": "vdjdb"},
        ]
    )

    assert run_matches_filters(
        run_id="custom_run_01",
        manifest_lookup=manifest_df,
        dataset_mode="yfv",
    )
    assert not run_matches_filters(
        run_id="custom_run_02",
        manifest_lookup=manifest_df,
        dataset_mode="yfv",
    )


def test_select_run_dirs_applies_dataset_mode_and_prefix_filters(tmp_path: Path):
    yfv_dir = tmp_path / "yfv_P1_vdbscan_grid_0001"
    vdjdb_dir = tmp_path / "vdjdb_GILGFVFTL_vdbscan_grid_0001"
    other_yfv_dir = tmp_path / "yfv_Q1_leiden_grid_0002"
    yfv_dir.mkdir()
    vdjdb_dir.mkdir()
    other_yfv_dir.mkdir()

    selected = select_run_dirs(
        runs_root=tmp_path,
        manifest_lookup=pd.DataFrame(columns=["run_id", "dataset_mode"]),
        dataset_mode="yfv",
        run_id_prefix="yfv_P1_",
    )

    assert [path.name for path in selected] == ["yfv_P1_vdbscan_grid_0001"]


def test_build_run_level_table_adds_requested_dbcv_and_support_summaries():
    cluster_df = pd.DataFrame(
        {
            "run_id": ["run_a", "run_a", "run_a", "run_a"],
            "cluster_id": [10, 20, 30, -1],
            "cluster_size": [10, 6, 4, 3],
            "sample": [8, 3, 1, 1],
            "background": [2, 3, 3, 2],
            "log_fold_change": [1.0, 0.2, -0.5, 2.0],
            "enrichment_fdr_zbinom": [0.001, 0.04, 0.2, 0.001],
            "density_validity_full": [0.3, -0.1, np.nan, np.nan],
            "density_validity_enriched_only": [0.5, -0.2, np.nan, np.nan],
            "density_validity": [0.3, -0.1, np.nan, np.nan],
        }
    )

    result = build_run_level_table(cluster_df)
    row = result.iloc[0]

    assert row["n_clusters_total"] == 3
    assert row["n_clusters_enriched"] == 2
    assert np.isclose(row["density_validity_full__mean"], 0.1)
    assert np.isclose(row["density_validity_full__median"], 0.1)
    assert np.isclose(row["density_validity_enriched_only__mean"], 0.15)
    assert np.isclose(row["density_validity_enriched_only__median"], 0.15)
    assert np.isclose(row["cluster_size_all__mean"], np.mean([10, 6, 4]))
    assert np.isclose(row["cluster_size_all__median"], 6.0)
    assert np.isclose(row["cluster_size_enriched__mean"], 8.0)
    assert np.isclose(row["cluster_size_enriched__median"], 8.0)
    assert np.isclose(row["sample_fraction_all__mean"], np.mean([0.8, 0.5, 0.25]))
    assert np.isclose(row["sample_fraction_all__median"], 0.5)
    assert np.isclose(row["sample_fraction_enriched__mean"], np.mean([0.8, 0.5]))
    assert np.isclose(row["sample_fraction_enriched__median"], 0.65)


def test_compute_dense_redcea_scores_obeys_formula_and_range_constraints():
    cluster_df = pd.DataFrame(
        {
            "run_id": ["run_a", "run_a", "run_b", "run_b", "run_c", "run_c"],
            "dataset": ["epi_x", "epi_x", "epi_x", "epi_x", "epi_x", "epi_x"],
            "cluster_id": [1, 2, 1, 2, 1, 2],
            "sample": [40, 20, 30, 30, 12, 8],
            "background": [10, 10, 15, 25, 2, 2],
            "log_fold_change": [2.0, 1.0, 1.3, 0.8, 2.5, 2.0],
            "enrichment_fdr_zbinom": [0.001, 0.01, 0.001, 0.02, 0.001, 0.002],
        }
    )

    result = compute_dense_redcea_scores(cluster_df, group_cols=["dataset"]).set_index("run_id")

    for column in [
        "coverage_rank_pct",
        "effective_size_penalty",
        "lfc_rank_pct",
        "redcea_dense_score",
        "redcea_dense_score_soft",
        "redcea_dense_score_base",
    ]:
        assert result[column].between(0, 1).all()

    assert np.isclose(result.loc["run_a", "sample_mass_enriched"], 60.0)
    assert np.isclose(result.loc["run_a", "cluster_mass_enriched"], 80.0)
    assert np.isclose(result.loc["run_a", "background_mass_enriched"], 20.0)
    assert np.isclose(result.loc["run_a", "sample_coverage_enriched"], 1.0)
    assert np.isclose(result.loc["run_a", "effective_n_enriched_clusters_sample_weighted"], 1.8)
    assert np.isclose(result.loc["run_a", "effective_sample_cluster_size"], 60.0 / 1.8)

    assert np.isclose(
        result.loc["run_a", "redcea_dense_score_base"],
        result.loc["run_a", "coverage_rank_pct"] * result.loc["run_a", "effective_size_penalty"],
    )
    assert np.isclose(
        result.loc["run_a", "redcea_dense_score"],
        result.loc["run_a", "redcea_dense_score_base"] * result.loc["run_a", "lfc_rank_pct"],
    )
    assert np.isclose(
        result.loc["run_a", "redcea_dense_score_soft"],
        result.loc["run_a", "redcea_dense_score_base"] * (0.5 + 0.5 * result.loc["run_a", "lfc_rank_pct"]),
    )


def test_compute_dense_redcea_scores_uses_requested_grouping_and_fills_missing_cluster_size():
    cluster_df = pd.DataFrame(
        {
            "run_id": ["run_a", "run_a", "run_b", "run_b", "run_c", "run_c", "run_d", "run_d"],
            "sample_group": ["g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2"],
            "cluster_id": [1, 2, 1, 2, 1, 2, 1, 2],
            "sample": [50, 10, 20, 10, 40, 10, 10, 10],
            "background": [0, 0, 5, 5, 0, 0, 5, 5],
            "log_fold_change": [2.0, 1.0, 1.2, 1.1, 2.2, 1.2, 1.1, 1.0],
            "enrichment_fdr_zbinom": [0.001, 0.002, 0.01, 0.02, 0.001, 0.002, 0.01, 0.02],
        }
    )

    result = compute_dense_redcea_scores(cluster_df).set_index("run_id")

    assert np.isclose(result.loc["run_a", "cluster_mass_enriched"], 60.0)
    assert np.isclose(result.loc["run_b", "cluster_mass_enriched"], 40.0)
    assert result.loc["run_a", "coverage_rank_pct"] > result.loc["run_b", "coverage_rank_pct"]
    assert result.loc["run_c", "coverage_rank_pct"] > result.loc["run_d", "coverage_rank_pct"]
    assert np.isclose(result.loc["run_a", "coverage_rank_pct"], 1.0)
    assert np.isclose(result.loc["run_b", "coverage_rank_pct"], 0.5)
    assert np.isclose(result.loc["run_c", "coverage_rank_pct"], 1.0)
    assert np.isclose(result.loc["run_d", "coverage_rank_pct"], 0.5)


def test_compute_lfc_mass_shift_handles_directionality_and_significance():
    cluster_df = pd.DataFrame(
        {
            "run_id": [
                "run_pos",
                "run_pos",
                "run_neg",
                "run_neg",
                "run_balanced",
                "run_balanced",
                "run_ignore_nonsig",
                "run_ignore_nonsig",
                "run_ignore_nonsig",
            ],
            "cluster_id": [1, 2, 1, 2, 1, 2, 1, 2, 3],
            "cluster_size": [12, 6, 8, 5, 9, 9, 10, 10, 8],
            "sample": [8, 2, 5, 4, 4, 4, 6, 6, 20],
            "background": [4, 4, 3, 1, 5, 5, 4, 4, 2],
            "log_fold_change": [2.0, 1.0, -1.5, -0.5, 1.0, -1.0, 1.0, -1.0, 10.0],
            "enrichment_fdr_zbinom": [0.001, 0.01, 0.001, 0.02, 0.001, 0.001, 0.001, 0.001, 0.5],
        }
    )

    result = compute_lfc_mass_shift(cluster_df).set_index("run_id")

    assert np.isclose(result.loc["run_pos", "lfc_mass_shift"], 1.0)
    assert np.isclose(result.loc["run_neg", "lfc_mass_shift"], -1.0)
    assert np.isclose(result.loc["run_balanced", "lfc_mass_shift"], 0.0)
    assert np.isclose(result.loc["run_ignore_nonsig", "lfc_mass_shift"], 0.0)
    assert np.isclose(result.loc["run_ignore_nonsig", "sample_sig_pos"], 6.0)
    assert np.isclose(result.loc["run_ignore_nonsig", "sample_sig_neg"], 6.0)


def test_build_run_level_table_merges_dense_and_longitudinal_metrics():
    cluster_df = pd.DataFrame(
        {
            "run_id": ["run_a", "run_a", "run_b", "run_b"],
            "dataset": ["epi_x", "epi_x", "epi_x", "epi_x"],
            "cluster_id": [1, 2, 1, 2],
            "sample": [10, 5, 6, 4],
            "background": [2, 1, 2, 4],
            "log_fold_change": [1.5, -0.4, 1.2, 0.8],
            "enrichment_fdr_zbinom": [0.001, 0.001, 0.002, 0.02],
            "density_validity_full": [0.3, 0.1, 0.2, 0.25],
            "density_validity_enriched_only": [0.3, np.nan, 0.2, 0.25],
            "density_validity": [0.3, 0.1, 0.2, 0.25],
        }
    )

    result = build_run_level_table(cluster_df)

    for column in [
        "redcea_dense_score",
        "redcea_dense_score_soft",
        "redcea_dense_score_base",
        "lfc_mass_shift",
        "sample_sig_pos",
    ]:
        assert column in result.columns

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_additional_metrics_batch import (
    build_run_level_table,
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

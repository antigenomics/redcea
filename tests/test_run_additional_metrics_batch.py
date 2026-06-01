from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_additional_metrics_batch import run_matches_filters, select_run_dirs


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

from pathlib import Path

import pandas as pd

from benchmark.data_sources import (
    resolve_vdjdb_embedding_path,
    resolve_vdjdb_rep_path,
    resolve_vdjdb_target_keys,
    select_top_vdjdb_infectious_epitopes,
)
from benchmark.prepare_datasets import _build_vdjdb_run_labels


def test_resolve_vdjdb_target_keys_accepts_arbitrary_epitope_sequences():
    resolved = resolve_vdjdb_target_keys(["GLC", "nlvpmvatv", "LPRRSGAAGA"])

    assert resolved == ["GLC", "NLVPMVATV", "LPRRSGAAGA"]


def test_resolve_vdjdb_paths_support_arbitrary_epitopes(tmp_path: Path):
    embed = resolve_vdjdb_embedding_path("LPRRSGAAGA", tmp_path)
    rep = resolve_vdjdb_rep_path("LPRRSGAAGA", tmp_path)

    assert embed["target_key"] == "LPRRSGAAGA"
    assert embed["epitope_sequence"] == "LPRRSGAAGA"
    assert embed["sample_embedding"] == tmp_path / "trb_vdjdb_LPRRSGAAGA_sample_embeddings.parquet"
    assert embed["sample_index"] == tmp_path / "trb_vdjdb_LPRRSGAAGA_sample_embeddings.index"
    assert rep == tmp_path / "trb_vdjdb_LPRRSGAAGA.tsv"


def test_select_top_vdjdb_infectious_epitopes_filters_to_available_trb_files(tmp_path: Path):
    embed_dir = tmp_path / "embed"
    airr_dir = tmp_path / "airr"
    embed_dir.mkdir()
    airr_dir.mkdir()

    for epitope in ["AAA", "BBB", "CCC"]:
        (embed_dir / f"trb_vdjdb_{epitope}_sample_embeddings.parquet").write_text("", encoding="utf-8")
        (airr_dir / f"trb_vdjdb_{epitope}.tsv").write_text("", encoding="utf-8")

    release_path = tmp_path / "vdjdb.slim.txt"
    pd.DataFrame(
        [
            {"gene": "TRB", "species": "HomoSapiens", "antigen.epitope": "AAA", "antigen.species": "InfluenzaA"},
            {"gene": "TRB", "species": "HomoSapiens", "antigen.epitope": "AAA", "antigen.species": "InfluenzaA"},
            {"gene": "TRB", "species": "HomoSapiens", "antigen.epitope": "BBB", "antigen.species": "CMV"},
            {"gene": "TRB", "species": "HomoSapiens", "antigen.epitope": "CCC", "antigen.species": "HomoSapiens"},
            {"gene": "TRA", "species": "HomoSapiens", "antigen.epitope": "DDD", "antigen.species": "EBV"},
            {"gene": "TRB", "species": "MusMusculus", "antigen.epitope": "EEE", "antigen.species": "LCMV"},
            {"gene": "TRB", "species": "HomoSapiens", "antigen.epitope": "ZZZ", "antigen.species": "EBV"},
        ]
    ).to_csv(release_path, sep="\t", index=False)

    selected = select_top_vdjdb_infectious_epitopes(
        top_k=2,
        vdjdb_release_path=release_path,
        embed_dir=embed_dir,
        airr_dir=airr_dir,
    )

    assert selected == ["AAA", "BBB"]


def test_build_vdjdb_run_labels_uses_first_three_epitope_letters():
    labels = _build_vdjdb_run_labels(["GILGFVFTL", "NLVPMVATV", "GLC"])

    assert labels["GILGFVFTL"] == "gil"
    assert labels["NLVPMVATV"] == "nlv"
    assert labels["GLC"] == "glc"

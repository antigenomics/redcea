import importlib
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd


class _FakeClonotype:
    def __init__(self, clone_id: int):
        self.id = clone_id


class _FakeRepertoire:
    def __init__(self, clone_ids):
        self.clonotypes = [_FakeClonotype(clone_id) for clone_id in clone_ids]
        self.total = len(self.clonotypes)

    def __iter__(self):
        return iter(self.clonotypes)

    def sample_n(self, n, sample_random=False):
        return _FakeRepertoire([c.id for c in self.clonotypes[:n]])


def test_redcea_sample_vs_background_smoke(tmp_path, monkeypatch):
    sample_path = tmp_path / "sample.tsv"
    background_path = tmp_path / "background.tsv"
    sample_path.write_text("dummy\n", encoding="utf-8")
    background_path.write_text("dummy\n", encoding="utf-8")

    fake_faiss = types.SimpleNamespace(
        omp_set_num_threads=lambda n: None,
        omp_get_max_threads=lambda: 1,
    )
    fake_segments_module = types.SimpleNamespace(
        SegmentLibrary=types.SimpleNamespace(load_default=lambda genes, organisms: object())
    )
    fake_tcremp_run_module = types.SimpleNamespace(run_tcremp_embedding=lambda *args, **kwargs: None)
    fake_tcremp_module = types.ModuleType("tcremp")
    fake_tcremp_module.tcremp_run = fake_tcremp_run_module

    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    monkeypatch.setitem(sys.modules, "mir", types.ModuleType("mir"))
    monkeypatch.setitem(sys.modules, "mir.common", types.ModuleType("mir.common"))
    monkeypatch.setitem(sys.modules, "mir.common.segments", fake_segments_module)
    monkeypatch.setitem(sys.modules, "tcremp", fake_tcremp_module)
    monkeypatch.setitem(sys.modules, "tcremp.tcremp_run", fake_tcremp_run_module)

    import redcea.analysis.io as analysis_io
    import redcea.embeddings as embeddings
    import redcea.pipeline as pipeline

    analysis_io = importlib.reload(analysis_io)
    embeddings = importlib.reload(embeddings)
    pipeline = importlib.reload(pipeline)
    args = Namespace(
        sample=str(sample_path),
        background=str(background_path),
        output=str(tmp_path / "out"),
        prefix="demo",
        index_col=None,
        chain="TRB",
        prototypes_path=None,
        n_prototypes=None,
        sample_random_prototypes=False,
        n_clonotypes=None,
        sample_random_clonotypes=False,
        species="HomoSapiens",
        random_seed=42,
        nproc=1,
        lower_len_cdr3=5,
        higher_len_cdr3=30,
        metrics="dissimilarity",
        sample_embedding=None,
        background_embedding=None,
        cluster_algo="leiden",
        n_bg_points=None,
        cluster_pc_components=2,
        core_min_samples=1,
        k_neighbors=2,
        eps_k_neighbors=2,
        leiden_resolution=1.0,
        leiden_sub_resolution=1.0,
        eps_estimation_based_on="sample",
        vdbscan_sym_rule="asymmetric",
        enrichment_test="zbinom",
        debug_save_intermediate=False,
        debug_output_dir=None,
        add_auxiliary_cluster_metrics=True,
    )

    monkeypatch.setattr(pipeline, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "log_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "resolve_prototype_file", lambda *args, **kwargs: "fake_prototypes.tsv")
    monkeypatch.setattr(embeddings, "load_prototype_repertoire", lambda *args, **kwargs: types.SimpleNamespace(total=2))
    monkeypatch.setattr(embeddings, "subsample_repertoire", lambda rep, *args, **kwargs: rep)
    monkeypatch.setattr(analysis_io, "subsample_repertoire", lambda rep, *args, **kwargs: rep)

    def fake_load_analysis_repertoire(path, *args, **kwargs):
        if Path(path).name == "sample.tsv":
            return _FakeRepertoire([0, 1, 2])
        return _FakeRepertoire([10, 11])

    monkeypatch.setattr(embeddings, "load_analysis_repertoire", fake_load_analysis_repertoire)
    monkeypatch.setattr(analysis_io, "load_analysis_repertoire", fake_load_analysis_repertoire)

    def fake_get_representations_df(rep, locus=None):
        return pd.DataFrame(
            {
                "clone_id": [c.id for c in rep],
                "cdr3aa_beta": [f"CASS{c.id}" for c in rep],
                "v_beta": ["TRBV1"] * len(rep.clonotypes),
                "j_beta": ["TRBJ1"] * len(rep.clonotypes),
            }
        )

    monkeypatch.setattr(analysis_io, "get_representations_df", fake_get_representations_df)

    def fake_run_tcremp_embedding(rep, proto, lib, chain, metrics, nproc):
        rows = []
        for idx, clonotype in enumerate(rep):
            rows.append(
                {
                    "0_b_v": float(idx),
                    "0_b_j": float(idx) + 0.1,
                    "0_b_cdr3": float(idx) + 0.2,
                    "1_b_v": float(idx) + 0.3,
                    "1_b_j": float(idx) + 0.4,
                    "1_b_cdr3": float(idx) + 0.5,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(embeddings, "run_tcremp_embedding", fake_run_tcremp_embedding)
    monkeypatch.setattr(
        pipeline,
        "build_joint_knn_artifacts",
        lambda **kwargs: types.SimpleNamespace(
            data_reduced=np.zeros((5, 2), dtype="float32"),
            distances=np.array(
                [
                    [0.0, 0.1],
                    [0.0, 0.1],
                    [0.0, 0.3],
                    [0.0, 0.1],
                    [0.0, 0.3],
                ],
                dtype="float32",
            ),
            indices=np.array(
                [
                    [0, 1],
                    [1, 0],
                    [2, 4],
                    [3, 0],
                    [4, 2],
                ],
                dtype="int32",
            ),
            ind_ss=np.array(
                [
                    [0, 1],
                    [1, 0],
                    [2, 1],
                ],
                dtype="int32",
            ),
            dist_ss=np.zeros((3, 2), dtype="float32"),
            dist_bb=np.zeros((2, 2), dtype="float32"),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "run_joint_clustering",
        lambda **kwargs: np.array([0, 0, 1, 0, 1], dtype="int32"),
    )

    def fake_to_parquet(self, path, *args, **kwargs):
        return self.to_pickle(path)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=True)
    monkeypatch.setattr(pd, "read_parquet", lambda path, *args, **kwargs: pd.read_pickle(path))

    artifacts = pipeline.run_redcea_pipeline(args)

    out_dir = tmp_path / "out"
    clusters_path = out_dir / "demo_tcremp_clusters.tsv"
    summary_path = out_dir / "demo_summary_tcrempnet.tsv"
    enriched_path = out_dir / "demo_enriched_clonotypes_tcremp.tsv"
    sample_emb_path = out_dir / "demo_sample_embeddings.parquet"
    background_emb_path = out_dir / "demo_background_embeddings.parquet"

    assert clusters_path.exists()
    assert summary_path.exists()
    assert enriched_path.exists()
    assert sample_emb_path.exists()
    assert background_emb_path.exists()

    clusters = pd.read_csv(clusters_path, sep="\t")
    summary = pd.read_csv(summary_path, sep="\t")
    enriched = pd.read_csv(enriched_path, sep="\t")

    assert not artifacts.summary_df.empty
    assert set(clusters["source"]) == {"sample", "background"}
    assert {"cluster_id", "cluster_size", "sample", "background", "log_fold_change"} <= set(summary.columns)
    assert "enrichment_pvalue_zbinom" in summary.columns
    assert "enrichment_fdr_zbinom" in summary.columns
    assert "log2fc_smooth" in summary.columns
    assert "density_validity" in summary.columns
    assert not enriched.empty

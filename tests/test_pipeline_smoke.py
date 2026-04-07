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

    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    monkeypatch.setitem(sys.modules, "mir", types.ModuleType("mir"))
    monkeypatch.setitem(sys.modules, "mir.common", types.ModuleType("mir.common"))
    monkeypatch.setitem(sys.modules, "mir.common.segments", fake_segments_module)

    import redcea.redcea as pipeline

    pipeline = importlib.reload(pipeline)

    monkeypatch.setattr(
        pipeline,
        "get_arguments_enrich",
        lambda: Namespace(
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
            cluster_min_samples=1,
            k_neighbors=2,
            eps_k_neighbors=2,
            leiden_resolution=1.0,
            leiden_sub_resolution=1.0,
            eps_estimation_based_on="sample",
            vdbscan_sym_rule="asymmetric",
        ),
    )

    monkeypatch.setattr(pipeline, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "log_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "resolve_prototype_file", lambda *args, **kwargs: "fake_prototypes.tsv")
    monkeypatch.setattr(pipeline, "load_prototype_repertoire", lambda *args, **kwargs: types.SimpleNamespace(total=2))
    monkeypatch.setattr(pipeline, "subsample_repertoire", lambda rep, *args, **kwargs: rep)

    def fake_load_analysis_repertoire(path, *args, **kwargs):
        if Path(path).name == "sample.tsv":
            return _FakeRepertoire([0, 1, 2])
        return _FakeRepertoire([10, 11])

    monkeypatch.setattr(pipeline, "load_analysis_repertoire", fake_load_analysis_repertoire)

    def fake_get_representations_df(rep, locus=None):
        return pd.DataFrame(
            {
                "clone_id": [c.id for c in rep],
                "cdr3aa_beta": [f"CASS{c.id}" for c in rep],
                "v_beta": ["TRBV1"] * len(rep.clonotypes),
                "j_beta": ["TRBJ1"] * len(rep.clonotypes),
            }
        )

    monkeypatch.setattr(pipeline, "get_representations_df", fake_get_representations_df)

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

    monkeypatch.setattr(pipeline, "run_tcremp_embedding", fake_run_tcremp_embedding)
    monkeypatch.setattr(pipeline, "prepare_data_for_clustering", lambda df, n_components: df.to_numpy())

    def fake_compute_blockwise_knn_merged(**kwargs):
        dist_ss = np.array([[0.0, 0.3], [0.0, 0.2], [0.0, 0.4]], dtype="float32")
        ind_ss = np.array([[0, 1], [1, 0], [2, 1]], dtype="int32")
        dist_bb = np.array([[0.0, 0.5], [0.0, 0.6]], dtype="float32")
        ind_bb = np.array([[0, 1], [1, 0]], dtype="int32")
        dist_sb = np.array([[0.25, 0.35], [0.22, 0.45], [0.28, 0.48]], dtype="float32")
        ind_sb = np.array([[0, 1], [0, 1], [1, 0]], dtype="int32")
        dist_bs = np.array([[0.18, 0.31], [0.19, 0.29]], dtype="float32")
        ind_bs = np.array([[1, 0], [2, 1]], dtype="int32")
        return dist_ss, ind_ss, dist_bb, ind_bb, dist_sb, ind_sb, dist_bs, ind_bs

    monkeypatch.setattr(pipeline, "compute_blockwise_knn_merged", fake_compute_blockwise_knn_merged)

    def fake_build_joint_knn_from_split(**kwargs):
        distances = np.array(
            [
                [0.0, 0.25],
                [0.0, 0.22],
                [0.0, 0.28],
                [0.0, 0.18],
                [0.0, 0.19],
            ],
            dtype="float32",
        )
        indices = np.array(
            [
                [0, 3],
                [1, 3],
                [2, 4],
                [3, 1],
                [4, 2],
            ],
            dtype="int32",
        )
        return distances, indices

    monkeypatch.setattr(pipeline, "build_joint_knn_from_split", fake_build_joint_knn_from_split)
    monkeypatch.setattr(pipeline, "run_leiden_clustering", lambda **kwargs: np.array([0, 0, 1, 0, 1], dtype="int32"))

    def fake_to_parquet(self, path, *args, **kwargs):
        return self.to_pickle(path)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=True)
    monkeypatch.setattr(pd, "read_parquet", lambda path, *args, **kwargs: pd.read_pickle(path))

    pipeline.main()

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

    assert set(clusters["source"]) == {"sample", "background"}
    assert {"cluster_id", "cluster_size", "sample", "background", "log_fold_change"} <= set(summary.columns)
    assert not enriched.empty

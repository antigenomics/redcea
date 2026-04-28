from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from mir.common.segments import SegmentLibrary

from redcea.analysis.cluster_utils import compute_cluster_summary
from redcea.analysis.io import EmbeddingArtifacts, PipelineArtifacts, load_embedding_artifacts, save_pipeline_outputs
from redcea.clustering import build_joint_knn_artifacts, run_joint_clustering
from redcea.clustering.cluster_methods import JointDbscanDebugArtifacts, run_joint_dbscan_clustering_with_diagnostics
from redcea.config import PipelineConfig, normalize_pipeline_config
from redcea.debug import (
    build_cluster_membership_frame,
    build_input_order_frame,
    prepare_debug_dir,
    save_json,
    save_numpy,
    save_tsv,
    summarize_distribution,
)
from redcea.embeddings import compute_embeddings_if_needed
from redcea.utils.paths import resolve_prototype_file
from redcea.utils.stats import add_log_fold_change, add_z_binom_pvalues
from redcea.utils.tcremp import (
    configure_logging,
    generate_output_prefix,
    load_prototype_repertoire,
    log_memory_usage,
    prepare_output_path,
    resolve_input_file,
)


@dataclass(frozen=True)
class RuntimeContext:
    sample_path: Path
    background_path: Path
    prototype_path: str
    output_path: Path
    prefix: str
    chain_genes: list[str]
    locus: str | None
    segment_library: SegmentLibrary


@dataclass(frozen=True)
class ClusteringOutputs:
    labels: np.ndarray
    knn: object
    dbscan_debug: JointDbscanDebugArtifacts | None = None


def prepare_runtime_context(args) -> RuntimeContext:
    sample_path = Path(resolve_input_file(args.sample))
    background_path = Path(resolve_input_file(args.background))
    prototype_path = resolve_prototype_file(args.prototypes_path, args.chain)
    output_path = Path(prepare_output_path(args.output))
    prefix = generate_output_prefix(args.sample, args.prefix)

    configure_logging(sample_path, output_path, prefix)
    chain_genes = args.chain.split("_")
    locus = {"TRA": "alpha", "TRB": "beta", "TRA_TRB": None}[args.chain]
    segment_library = SegmentLibrary.load_default(genes=chain_genes, organisms=args.species)
    np.random.seed(args.random_seed)

    faiss.omp_set_num_threads(args.normalized_nproc)
    logging.info("FAISS threads set to %d", faiss.omp_get_max_threads())
    logging.info("NumPy random seed set to %d", args.random_seed)

    return RuntimeContext(
        sample_path=sample_path,
        background_path=background_path,
        prototype_path=prototype_path,
        output_path=output_path,
        prefix=prefix,
        chain_genes=chain_genes,
        locus=locus,
        segment_library=segment_library,
    )

def compute_cluster_labels(
    *,
    config: PipelineConfig,
    sample_embeddings: pd.DataFrame,
    background_embeddings: pd.DataFrame,
    joint_representations: pd.DataFrame,
    sample_representations: pd.DataFrame,
    background_representations: pd.DataFrame,
    sample_size: int,
    sample_index_path,
    bg_index_path,
    output_path,
)-> ClusteringOutputs:
    logging.info("Running clustering...")
    knn = build_joint_knn_artifacts(
        config=config,
        sample_embeddings=sample_embeddings,
        background_embeddings=background_embeddings,
        sample_size=sample_size,
        sample_index_path=sample_index_path,
        bg_index_path=bg_index_path,
        output_path=output_path,
    )
    if config.cluster_algo == "dbscan":
        dbscan_debug = run_joint_dbscan_clustering_with_diagnostics(config=config, knn=knn)
        return ClusteringOutputs(labels=dbscan_debug.labels, knn=knn, dbscan_debug=dbscan_debug)

    labels = run_joint_clustering(
        config=config,
        joint_representations=joint_representations,
        sample_representations=sample_representations,
        background_representations=background_representations,
        knn=knn,
    )
    return ClusteringOutputs(labels=np.asarray(labels), knn=knn, dbscan_debug=None)


def build_pipeline_artifacts(
    *,
    cluster_labels,
    joint_ids: pd.Series,
    joint_representations: pd.DataFrame,
    sample_ids: pd.Series,
    background_ids: pd.Series,
) -> PipelineArtifacts:
    log_memory_usage("After clustering")

    cluster_df = _build_cluster_df(
        cluster_labels=cluster_labels,
        joint_ids=joint_ids,
        joint_representations=joint_representations,
    )
    summary_df = compute_cluster_summary(cluster_df, sample_ids)
    summary_df = add_z_binom_pvalues(summary_df, total_sample=len(sample_ids), total_background=len(background_ids))
    summary_df = add_log_fold_change(summary_df, total_sample=len(sample_ids), total_background=len(background_ids))

    enriched_clusters = summary_df.loc[
        (summary_df["enrichment_fdr_zbinom"] < 0.05) & (summary_df["log_fold_change"] > 0),
        ["cluster_id", "enrichment_pvalue_zbinom"],
    ]
    logging.info("%d clusters identified as enriched (fdr < 0.05, logFC > 0).", len(enriched_clusters))

    enriched_clonotypes_df = cluster_df.merge(enriched_clusters, on="cluster_id")

    return PipelineArtifacts(
        cluster_df=cluster_df,
        summary_df=summary_df,
        enriched_clonotypes_df=enriched_clonotypes_df,
    )


def _build_cluster_df(
    *,
    cluster_labels,
    joint_ids: pd.Series,
    joint_representations: pd.DataFrame,
) -> pd.DataFrame:
    if (
        "clone_id" in joint_representations.columns
        and len(joint_representations) == len(joint_ids)
        and joint_representations["clone_id"].reset_index(drop=True).equals(joint_ids.reset_index(drop=True))
    ):
        cluster_df = joint_representations.copy()
        insert_at = cluster_df.columns.get_loc("clone_id") + 1
        cluster_df.insert(insert_at, "cluster_id", cluster_labels)
        return cluster_df

    return pd.DataFrame({"clone_id": joint_ids, "cluster_id": cluster_labels}).merge(
        joint_representations,
        on="clone_id",
    )


def run_redcea_pipeline(config_or_args) -> PipelineArtifacts:
    config = normalize_pipeline_config(config_or_args)
    runtime = prepare_runtime_context(config)

    log_memory_usage("Init")
    logging.info("Starting RedCEA pipeline...")

    repertoire_paths = {
        "sample": runtime.sample_path,
        "background": runtime.background_path,
    }

    for tag, path in repertoire_paths.items():
        logging.info("Computing %s embeddings if needed...", tag)
        compute_embeddings_if_needed(
            str(path),
            config,
            is_sample=(tag == "sample"),
            proto=runtime.prototype_path,
            chain=runtime.chain_genes,
            lib=runtime.segment_library,
            locus=runtime.locus,
            prefix=runtime.prefix,
            output_path=runtime.output_path,
        )

    loaded_embeddings = {}
    for tag, path in repertoire_paths.items():
        logging.info("Loading %s embeddings...", tag)
        loaded_embeddings[tag] = load_embedding_artifacts(
            str(path),
            config,
            is_sample=(tag == "sample"),
            lib=runtime.segment_library,
            locus=runtime.locus,
            prefix=runtime.prefix,
            output_path=runtime.output_path,
        )

    sample_artifacts = loaded_embeddings["sample"]
    background_artifacts = loaded_embeddings["background"]

    if sample_artifacts.embeddings.empty:
        raise ValueError("Sample embeddings are empty after loading.")
    if background_artifacts.embeddings.empty:
        raise ValueError("Background embeddings are empty after loading.")

    log_memory_usage("After loading embeddings")
    logging.info("Sample FAISS index path: %s", sample_artifacts.cache_path)
    logging.info("Background FAISS index path: %s", background_artifacts.cache_path)

    joint_representations = pd.concat(
        [sample_artifacts.representations, background_artifacts.representations],
        ignore_index=True,
    )
    sample_size = len(sample_artifacts.embeddings)
    joint_ids = pd.concat([sample_artifacts.ids, background_artifacts.ids], ignore_index=True)

    clustering_outputs = compute_cluster_labels(
        config=config,
        sample_embeddings=sample_artifacts.embeddings,
        background_embeddings=background_artifacts.embeddings,
        joint_representations=joint_representations,
        sample_representations=sample_artifacts.representations,
        background_representations=background_artifacts.representations,
        sample_size=sample_size,
        sample_index_path=sample_artifacts.cache_path,
        bg_index_path=background_artifacts.cache_path,
        output_path=runtime.output_path,
    )

    artifacts = build_pipeline_artifacts(
        cluster_labels=clustering_outputs.labels,
        joint_ids=joint_ids,
        joint_representations=joint_representations,
        sample_ids=sample_artifacts.ids,
        background_ids=background_artifacts.ids,
    )
    if config.debug_save_intermediate:
        _save_debug_outputs(
            config=config,
            runtime=runtime,
            sample_embeddings=sample_artifacts.embeddings,
            background_embeddings=background_artifacts.embeddings,
            joint_representations=joint_representations,
            artifacts=artifacts,
            clustering_outputs=clustering_outputs,
        )
    del sample_artifacts.embeddings
    del background_artifacts.embeddings
    gc.collect()
    save_pipeline_outputs(artifacts, output_path=runtime.output_path, prefix=runtime.prefix)

    logging.info("RedCEA pipeline completed.")
    log_memory_usage("Finished")
    return artifacts


def _save_debug_outputs(
    *,
    config: PipelineConfig,
    runtime: RuntimeContext,
    sample_embeddings: pd.DataFrame,
    background_embeddings: pd.DataFrame,
    joint_representations: pd.DataFrame,
    artifacts: PipelineArtifacts,
    clustering_outputs: ClusteringOutputs,
) -> None:
    debug_dir = prepare_debug_dir(runtime.output_path, config.debug_output_dir)
    input_order = build_input_order_frame(joint_representations, sample_size=len(sample_embeddings))
    save_tsv(debug_dir / "01_input_order.tsv", input_order)

    joint_embeddings = pd.concat([sample_embeddings, background_embeddings], ignore_index=True)
    save_numpy(debug_dir / "02_representations.npy", joint_embeddings.to_numpy(dtype=np.float32, copy=False))
    save_tsv(debug_dir / "02_representations_meta.tsv", input_order)

    save_numpy(debug_dir / "03_pca_embeddings.npy", np.asarray(clustering_outputs.knn.data_reduced))
    save_numpy(debug_dir / "04_knn_distances.npy", np.asarray(clustering_outputs.knn.distances))
    save_numpy(debug_dir / "05_knn_indices.npy", np.asarray(clustering_outputs.knn.indices))

    dbscan_debug = clustering_outputs.dbscan_debug
    if dbscan_debug is not None:
        eps_stats = summarize_distribution(dbscan_debug.eps_distances)
        eps_frame = pd.DataFrame(
            [
                {
                    "eps_value": dbscan_debug.eps,
                    "eps_k_neighbors": config.eps_k_neighbors,
                    "column_used_for_eps": dbscan_debug.eps_column,
                    **eps_stats,
                }
            ]
        )
        save_tsv(debug_dir / "06_eps.tsv", eps_frame)

        prefilter_frame = input_order[["clone_id", "source", "original_row_index"]].copy()
        prefilter_frame["d1"] = dbscan_debug.d1_distances
        prefilter_frame["eps"] = dbscan_debug.eps
        prefilter_frame["keep_mask"] = dbscan_debug.keep_mask
        prefilter_frame["column_used_for_d1"] = dbscan_debug.d1_column
        save_tsv(debug_dir / "07_d1_prefilter.tsv", prefilter_frame)

        labels_frame = input_order[["clone_id", "source", "original_row_index"]].copy()
        labels_frame["label"] = clustering_outputs.labels
        labels_frame["is_noise"] = labels_frame["label"] == -1
        labels_frame["keep_mask"] = dbscan_debug.keep_mask
        save_tsv(debug_dir / "08_dbscan_labels.tsv", labels_frame)

        debug_summary = {
            "n_input": int(len(input_order)),
            "n_after_pca": int(clustering_outputs.knn.data_reduced.shape[0]),
            "n_knn": int(clustering_outputs.knn.indices.shape[0]),
            "eps": float(dbscan_debug.eps),
            "eps_column": int(dbscan_debug.eps_column),
            "d1_column": int(dbscan_debug.d1_column),
            "n_prefilter_removed": int(np.count_nonzero(~dbscan_debug.keep_mask)),
            "prefilter_removed_fraction": float(np.mean(~dbscan_debug.keep_mask)),
            "dbscan_min_samples": int(config.cluster_min_samples),
            "dbscan_n_clusters": int(len(set(clustering_outputs.labels)) - (1 if -1 in clustering_outputs.labels else 0)),
            "dbscan_n_noise": int(np.count_nonzero(np.asarray(clustering_outputs.labels) == -1)),
            "n_enriched_clusters": int(artifacts.summary_df.loc[
                (artifacts.summary_df["enrichment_fdr_zbinom"] < 0.05) & (artifacts.summary_df["log_fold_change"] > 0)
            ].shape[0]),
            "n_enriched_clonotypes": int(len(artifacts.enriched_clonotypes_df)),
        }
        save_json(debug_dir / "debug_summary.json", debug_summary)

    save_tsv(debug_dir / "09_clusters.tsv", build_cluster_membership_frame(artifacts.cluster_df))
    save_tsv(debug_dir / "10_enrichment_table.tsv", artifacts.summary_df)
    save_tsv(debug_dir / "11_enriched_clonotypes.tsv", artifacts.enriched_clonotypes_df)


__all__ = [
    "EmbeddingArtifacts",
    "PipelineArtifacts",
    "RuntimeContext",
    "build_pipeline_artifacts",
    "compute_cluster_labels",
    "prepare_runtime_context",
    "run_redcea_pipeline",
    "save_pipeline_outputs",
]

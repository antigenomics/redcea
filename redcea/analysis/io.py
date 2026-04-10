from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from redcea.utils.paths import resolve_embedding_file
from redcea.utils.tcremp import get_representations_df, load_analysis_repertoire, subsample_repertoire


@dataclass
class EmbeddingArtifacts:
    embeddings: pd.DataFrame
    representations: pd.DataFrame
    ids: pd.Series
    cache_path: Path


@dataclass
class PipelineArtifacts:
    cluster_df: pd.DataFrame
    summary_df: pd.DataFrame
    enriched_clonotypes_df: pd.DataFrame
    enriched_embeddings_df: pd.DataFrame


def get_sample_info(
    sample_name: str,
    airr_path: str | Path = "/projects/immunestatus/emerson/airr_format",
    data_path: str | Path = "/projects/immunestatus/emerson_ebv/tcrempnet",
) -> tuple[pd.DataFrame, int]:
    """Load sample summary plus repertoire size for legacy notebook analyses."""
    airr_path = Path(airr_path)
    data_path = Path(data_path)

    summary = pd.read_csv(data_path / f"{sample_name}_summary_tcrempnet.tsv", sep="\t")
    repertoire_size = len(pd.read_csv(airr_path / f"{sample_name}.tsv", sep="\t"))

    logging.info(
        "sample=%s clusters=%d overall_from_sample=%d",
        sample_name,
        len(summary),
        repertoire_size,
    )
    return summary, repertoire_size


def get_clonotypes(
    sample_name: str,
    data_path: str | Path = "/projects/immunestatus/emerson_ebv/tcrempnet",
) -> pd.DataFrame:
    """Load enriched clonotypes table for a sample from legacy notebook outputs."""
    data_path = Path(data_path)
    return pd.read_csv(data_path / f"{sample_name}_enriched_clonotypes_tcremp.tsv", sep="\t")


def load_embedding_artifacts(path, args, is_sample, lib, locus, prefix, output_path) -> EmbeddingArtifacts:
    """Load embeddings plus repertoire metadata for sample or background."""
    tag = "sample" if is_sample else "background"
    prefix_tag = "s_" if is_sample else "b_"
    custom_path = args.sample_embedding if is_sample else args.background_embedding
    emb_path = resolve_embedding_file(custom_path, output_path, prefix, tag, must_exist=True)

    emb = pd.read_parquet(emb_path)

    if (not is_sample) and getattr(args, "n_bg_points", None):
        logging.info("Restricting background embeddings to first %d", args.n_bg_points)
        emb = emb.iloc[:args.n_bg_points]

    rep = load_analysis_repertoire(path, lib, locus, args.index_col, args.lower_len_cdr3, args.higher_len_cdr3)

    if (not is_sample) and getattr(args, "n_bg_points", None):
        rep = rep.sample_n(args.n_bg_points, sample_random=False)

    rep = subsample_repertoire(rep, args.n_clonotypes, args.sample_random_clonotypes, args.random_seed)

    rep_df = get_representations_df(rep, locus)
    rep_df["clone_id"] = prefix_tag + rep_df["clone_id"].astype(str)
    ids = pd.Series([f"{prefix_tag}{c.id}" for c in rep])

    del rep
    gc.collect()
    return EmbeddingArtifacts(
        embeddings=emb,
        representations=rep_df,
        ids=ids,
        cache_path=Path(emb_path),
    )


def save_pipeline_outputs(artifacts: PipelineArtifacts, *, output_path, prefix: str) -> None:
    """Persist the standard RedCEA output tables to disk."""
    output_path = Path(output_path)

    artifacts.cluster_df.to_csv(output_path / f"{prefix}_tcremp_clusters.tsv", sep="\t", index=False)
    logging.info("Saved cluster assignments.")

    artifacts.summary_df[
        [
            "cluster_id",
            "cluster_size",
            "sample",
            "background",
            "enrichment_pvalue_zbinom",
            "enrichment_fdr_zbinom",
            "log_fold_change",
        ]
    ].to_csv(output_path / f"{prefix}_summary_tcrempnet.tsv", sep="\t", index=False)
    logging.info("Saved cluster summary with p-values.")

    artifacts.enriched_clonotypes_df.to_csv(
        output_path / f"{prefix}_enriched_clonotypes_tcremp.tsv", sep="\t", index=False
    )
    logging.info("Saved enriched clonotypes.")

    artifacts.enriched_embeddings_df.to_parquet(output_path / f"{prefix}_enriched_embeddings_tcremp.parquet")
    logging.info("Saved enriched embeddings.")


__all__ = [
    "EmbeddingArtifacts",
    "PipelineArtifacts",
    "get_clonotypes",
    "get_sample_info",
    "load_embedding_artifacts",
    "save_pipeline_outputs",
]

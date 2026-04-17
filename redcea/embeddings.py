from __future__ import annotations

import logging
from pathlib import Path

from tcremp.tcremp_run import run_tcremp_embedding

from redcea.config import PipelineConfig
from redcea.utils.paths import resolve_embedding_file, resolve_prototype_file
from redcea.utils.tcremp import (
    load_analysis_repertoire,
    load_prototype_repertoire,
    subsample_repertoire,
)


def compute_embeddings_if_needed(
    path,
    config: PipelineConfig,
    *,
    is_sample: bool,
    proto,
    chain,
    lib,
    locus,
    prefix: str,
    output_path,
):
    tag = "sample" if is_sample else "background"
    custom_path = config.sample_embedding if is_sample else config.background_embedding
    emb_path = resolve_embedding_file(custom_path, output_path, prefix, tag)

    if emb_path.exists():
        logging.info("Found existing %s embeddings at %s", tag, emb_path)
        return emb_path

    logging.info("Computing %s embeddings...", tag)
    rep = load_analysis_repertoire(
        path,
        lib,
        locus,
        index_col=config.index_col,
        lower_len_cdr3=config.lower_len_cdr3,
        higher_len_cdr3=config.higher_len_cdr3,
    )

    if (not is_sample) and config.n_bg_points:
        logging.info("Restricting background repertoire to first %d clonotypes (pre-embedding)", config.n_bg_points)
        rep = rep.sample_n(config.n_bg_points, sample_random=False)

    rep = subsample_repertoire(rep, config.n_clonotypes, config.sample_random_clonotypes, config.random_seed)
    logging.info("Prepared %s repertoire with %d clonotypes before embedding", tag, len(rep.clonotypes))
    if len(rep.clonotypes) == 0:
        raise ValueError(
            f"{tag.capitalize()} repertoire is empty after filtering/subsampling: {path}. "
            "Check AIRR conversion, chain selection, and CDR3 length thresholds."
        )

    if not hasattr(proto, "total"):
        proto_path = resolve_prototype_file(str(proto) if proto else None, config.chain)
        proto = load_prototype_repertoire(proto_path, lib, locus, index_col=config.index_col)

    embeddings = run_tcremp_embedding(rep, proto, lib, chain, config.metrics, config.normalized_nproc)
    embeddings.to_parquet(Path(emb_path), index=False)
    return emb_path


__all__ = [
    "compute_embeddings_if_needed",
]

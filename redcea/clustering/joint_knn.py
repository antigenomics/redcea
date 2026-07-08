from __future__ import annotations

import gc
from dataclasses import dataclass

import pandas as pd

from redcea.clustering import build_joint_knn_from_split, compute_blockwise_knn_merged, prepare_data_for_clustering
from redcea.config import PipelineConfig
from redcea.utils.tcremp import log_memory_usage


@dataclass
class JointKnnArtifacts:
    data_reduced: object
    distances: object
    indices: object
    dist_ss: object
    ind_ss: object
    dist_bb: object


def build_joint_knn_artifacts(
    *,
    config: PipelineConfig,
    sample_embeddings: pd.DataFrame,
    background_embeddings: pd.DataFrame,
    sample_size: int,
    sample_index_path,
    bg_index_path,
    output_path,
) -> JointKnnArtifacts:
    joint_embeddings = pd.concat([sample_embeddings, background_embeddings], ignore_index=True)
    log_memory_usage("After concatenation")
    df = prepare_data_for_clustering(joint_embeddings, n_components=config.cluster_pc_components)
    del joint_embeddings
    gc.collect()
    log_memory_usage("After PCA; released raw joint embeddings")

    bg_data = df[sample_size:]
    sample_data = df[:sample_size]

    dist_ss, ind_ss, dist_bb, ind_bb, dist_sb, ind_sb, dist_bs, ind_bs = compute_blockwise_knn_merged(
        bg=bg_data,
        sample=sample_data,
        k_neighbors=config.k_neighbors,
        bg_index_path=bg_index_path,
        sample_index_path=sample_index_path,
        rebuild_bg=False,
        rebuild_sample=True,
        save_blocks=True,
        output_dir=output_path,
        nproc=config.normalized_nproc,
        bg_size_truncation=config.n_bg_points,
    )

    distances, indices = build_joint_knn_from_split(
        dist_ss=dist_ss,
        ind_ss=ind_ss,
        dist_bb=dist_bb,
        ind_bb=ind_bb,
        dist_sb=dist_sb,
        ind_sb=ind_sb,
        dist_bs=dist_bs,
        ind_bs=ind_bs,
        k_out=config.k_neighbors,
    )
    return JointKnnArtifacts(
        data_reduced=df,
        distances=distances,
        indices=indices,
        dist_ss=dist_ss,
        ind_ss=ind_ss,
        dist_bb=dist_bb,
    )


__all__ = [
    "JointKnnArtifacts",
    "build_joint_knn_artifacts",
]

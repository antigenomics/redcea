#!/usr/bin/env python
from __future__ import annotations

import gc
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from .compat import (
    SegmentLibrary,
    add_log_fold_change,
    add_z_binom_pvalues,
    compute_blockwise_knn_merged,
    compute_cluster_summary,
    compute_embeddings_if_needed,
    configure_logging,
    load_embeddings,
    load_prototype_repertoire,
    prepare_output_path,
    resolve_prototype_file,
    run_leiden_clustering,
    subsample_repertoire,
)
from .config import CHAIN_COLS, DEFAULT_PLOT_BG_POINTS
from .io import (
    build_airr_from_epitope,
    build_sample_members_table,
    load_or_fit_background_transform,
    prepare_background_umap,
    prepare_output_dirs,
    sanitize_filename_token,
)
from .plotting import save_cluster_plot_html

try:
    from tcremp.arguments import get_arguments_vdjdb_clusters
except ImportError as exc:  # pragma: no cover
    raise ImportError("vdjdb_redcea requires an installed tcremp package") from exc


def resolve_joint_knn(
    sample_pca: np.ndarray,
    bg_pca: np.ndarray,
    k_neighbors: int,
    nproc: int,
    sample_index_path: Path,
    bg_index_path: Path,
):
    """Resolve joint kNN for sample and background PCA.

    Args:
        sample_pca: Sample PCA array.
        bg_pca: Background PCA array.
        k_neighbors: Number of neighbors.
        nproc: Number of processes.
        sample_index_path: Path to sample index.
        bg_index_path: Path to background index.

    Returns:
        KNN result.
    """
    if sample_index_path is None or bg_index_path is None:
        raise ValueError("sample_index_path and bg_index_path must be provided for joint kNN resolution.")

    result = compute_blockwise_knn_merged(
        bg=bg_pca,
        sample=sample_pca,
        k_neighbors=k_neighbors,
        bg_index_path=bg_index_path,
        sample_index_path=sample_index_path,
        rebuild_bg=False,
        rebuild_sample=False,
        save_blocks=False,
        output_dir=None,
        nproc=nproc,
    )
    if isinstance(result, tuple) and len(result) == 2:
        return result

    joint = np.vstack([sample_pca, bg_pca]).astype("float32", copy=False)
    index = faiss.IndexFlatL2(joint.shape[1])
    faiss.omp_set_num_threads(int(nproc))
    index.add(joint)
    return index.search(joint, k_neighbors)


def _compute_sample_embeddings(
    args, genes: list[str], locus: str, lib: SegmentLibrary, proto, paths, chain: str, prefix: str, airr_path: Path
):
    """Compute or load sample embeddings."""
    args.sample = str(airr_path.resolve())
    args.sample_embedding = str((paths.tcremp_dir / f"{prefix}_sample_embeddings.parquet").resolve())
    args.prefix = prefix

    compute_embeddings_if_needed(
        path=args.sample,
        args=args,
        is_sample=True,
        proto=proto,
        chain=genes,
        lib=lib,
        locus=locus,
        prefix=prefix,
        output_path=paths.tcremp_dir,
    )

    return load_embeddings(
        path=args.sample,
        args=args,
        is_sample=True,
        lib=lib,
        locus=locus,
        prefix=prefix,
        output_path=paths.tcremp_dir,
    )


def _perform_clustering(
    sample_emb: np.ndarray,
    transform,
    bg_pca: np.ndarray,
    args,
    sample_index_path: Path,
    bg_index_path: Path,
    sample_ids: pd.Series,
    bg_ids: pd.Series,
):
    """Perform clustering on sample and background."""
    sample_pca = transform.transform_pca(sample_emb)
    distances, indices = resolve_joint_knn(
        sample_pca,
        bg_pca,
        args.k_neighbors,
        args.nproc,
        sample_index_path=sample_index_path,
        bg_index_path=bg_index_path,
    )
    labels = run_leiden_clustering(
        knn_indices=indices,
        knn_distances=distances,
        resolution=args.leiden_resolution,
        n_threads=args.nproc,
        min_cluster_size=args.cluster_min_samples,
        min_cluster_size_mask=np.arange(len(sample_pca) + len(bg_pca)) < len(sample_pca),
    )
    return sample_pca, labels


def _compute_summary_and_significance(summary_df: pd.DataFrame, sample_ids: pd.Series, bg_ids: pd.Series):
    """Compute summary statistics and significance."""
    summary_df = add_z_binom_pvalues(summary_df, total_sample=len(sample_ids), total_background=len(bg_ids))
    summary_df = add_log_fold_change(summary_df, total_sample=len(sample_ids), total_background=len(bg_ids))
    summary_df["significant"] = (
        (summary_df["enrichment_fdr_zbinom"] < 0.05) & (summary_df["log_fold_change"] > 0)
    )
    significant_cluster_ids = set(summary_df.loc[summary_df["significant"], "cluster_id"].astype(int))
    return summary_df, significant_cluster_ids


def _save_cluster_results(
    cluster_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    sample_cluster_df: pd.DataFrame,
    cluster_members_df: pd.DataFrame,
    paths,
    prefix: str,
):
    """Save clustering results to files."""
    cluster_df.to_csv(paths.tcrempnet_dir / f"{prefix}_tcremp_clusters.tsv", sep="\t", index=False)
    summary_df[
        [
            "cluster_id",
            "cluster_size",
            "sample",
            "background",
            "enrichment_pvalue_zbinom",
            "enrichment_fdr_zbinom",
            "log_fold_change",
            "significant",
        ]
    ].to_csv(paths.tcrempnet_dir / f"{prefix}_summary_tcrempnet.tsv", sep="\t", index=False)
    sample_cluster_df.to_csv(paths.tcrempnet_dir / f"{prefix}_clustered_sample_clonotypes.tsv", sep="\t", index=False)
    cluster_members_df.to_csv(paths.tcrempnet_dir / f"{prefix}_cluster_members.tsv", sep="\t", index=False)


def process_epitope(
    epitope: str,
    ep_df: pd.DataFrame,
    *,
    args,  # type: ignore
    genes: list[str],
    locus: str,
    lib: SegmentLibrary,
    proto,
    paths,
    bg_pca: np.ndarray,
    bg_reps: pd.DataFrame,
    bg_ids: pd.Series,
    bg_index_path: Path,
    bg_umap: np.ndarray,
    transform,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process a single epitope for clustering.

    Args:
        epitope: Epitope name.
        ep_df: Epitope DataFrame.
        args: Arguments object.
        genes: List of genes.
        locus: Locus.
        lib: SegmentLibrary.
        proto: Prototypes.
        paths: OutputPaths.
        bg_pca: Background PCA.
        bg_reps: Background representations.
        bg_ids: Background IDs.
        bg_index_path: Background index path.
        bg_umap: Background UMAP.
        transform: BackgroundTransform.

    Returns:
        Tuple of clustered DataFrame and cluster members DataFrame.
    """
    chain = args.chain
    prefix = f"{chain.lower()}_vdjdb_{epitope}"
    airr_path = paths.airr_dir / f"{prefix}.tsv"

    logging.info("Processing epitope %s with %d clonotypes", epitope, len(ep_df))
    build_airr_from_epitope(ep_df, chain).to_csv(airr_path, sep="\t", index=False)

    sample_emb, sample_reps, sample_ids, _, sample_index_path = _compute_sample_embeddings(
        args, genes, locus, lib, proto, paths, chain, prefix, airr_path
    )

    sample_pca, labels = _perform_clustering(
        sample_emb, transform, bg_pca, args, sample_index_path, bg_index_path, sample_ids, bg_ids
    )

    joint_ids = pd.concat([sample_ids, bg_ids], ignore_index=True)
    joint_reps = pd.concat([sample_reps, bg_reps], ignore_index=True)
    cluster_df = pd.DataFrame({"clone_id": joint_ids, "cluster_id": labels}).merge(
        joint_reps, on="clone_id", how="left"
    )

    summary_df = compute_cluster_summary(cluster_df.copy(), sample_ids)
    summary_df, significant_cluster_ids = _compute_summary_and_significance(summary_df, sample_ids, bg_ids)

    sample_cluster_df = cluster_df[
        (cluster_df["clone_id"].isin(set(sample_ids))) & (cluster_df["cluster_id"] != -1)
    ].copy()
    sample_cluster_count = sample_cluster_df["cluster_id"].nunique()

    enriched_sample_cluster_df = sample_cluster_df[
        sample_cluster_df["cluster_id"].isin(significant_cluster_ids)
    ].copy()
    enriched_cluster_count = enriched_sample_cluster_df["cluster_id"].nunique()

    sample_umap = transform.transform_umap(sample_pca)
    cluster_members_df = build_sample_members_table(
        sample_cluster_df, summary_df, chain, ep_df, epitope, sample_ids, sample_umap
    )

    _save_cluster_results(cluster_df, summary_df, sample_cluster_df, cluster_members_df, paths, prefix)

    sample_labels = np.asarray(labels[: len(sample_ids)], dtype=np.int32)
    viz_filename = (
        f"{sanitize_filename_token(args.species)}_"
        f"{sanitize_filename_token(epitope)}_"
        f"{sanitize_filename_token(chain)}.html"
    )
    save_cluster_plot_html(
        epitope=epitope,
        chain=chain,
        sample_reps=sample_reps,
        sample_ids=sample_ids,
        sample_pca=sample_pca,
        sample_labels=sample_labels,
        significant_cluster_ids=significant_cluster_ids,
        summary_df=summary_df,
        bg_umap=bg_umap,
        transform=transform,
        output_path=paths.viz_dir / viz_filename,
    )

    logging.info(
        "Done %s: clustered_sample points=%d/%d, clusters=%d; enriched points=%d, clusters=%d",
        epitope,
        len(sample_cluster_df),
        len(sample_reps),
        sample_cluster_count,
        len(enriched_sample_cluster_df),
        enriched_cluster_count,
    )

    del sample_emb, sample_reps, sample_ids, sample_pca, labels, joint_ids, joint_reps, cluster_df
    gc.collect()
    return sample_cluster_df, cluster_members_df


def main():
    """Main function for VDJdb epitope clustering."""
    try:
        args = get_arguments_vdjdb_clusters()

        output_root = prepare_output_path(args.output)
        paths = prepare_output_dirs(output_root)

        configure_logging(Path(args.vdjdb), output_root, f"{args.chain.lower()}_vdjdb_clusters")
        faiss.omp_set_num_threads(args.nproc)

        genes = [args.chain]
        locus = CHAIN_COLS[args.chain]["locus"]
        lib = SegmentLibrary.load_default(genes=genes, organisms=args.species)

        vdjdb_df = pd.read_csv(args.vdjdb, sep="\t")
        if args.epitopes is not None:
            vdjdb_df = vdjdb_df[vdjdb_df["antigen.epitope"].isin(args.epitopes)].copy()
        if args.min_epitope_clonotypes is not None:
            epitope_sizes = vdjdb_df.groupby("antigen.epitope").size()
            eligible_epitopes = epitope_sizes[epitope_sizes >= args.min_epitope_clonotypes].index
            skipped_epitopes = int((epitope_sizes < args.min_epitope_clonotypes).sum())
            vdjdb_df = vdjdb_df[vdjdb_df["antigen.epitope"].isin(eligible_epitopes)].copy()
            logging.info(
                "Filtered epitopes by minimum clonotype count >= %d: kept %d epitopes, skipped %d",
                args.min_epitope_clonotypes,
                len(eligible_epitopes),
                skipped_epitopes,
            )

        args.background = str(Path(args.background_airr).resolve())
        args.output = str(output_root)
        args.background_embedding = str(Path(args.background_embedding).resolve())

        proto_path = resolve_prototype_file(args.prototypes_path)

        logging.info("Loading prototypes")
        proto = load_prototype_repertoire(proto_path, lib, locus, args.index_col)
        proto = subsample_repertoire(proto, args.n_prototypes, args.sample_random_clonotypes, args.random_seed)
        logging.info("Loaded %d prototypes", len(proto))

        logging.info("Loading background embeddings")
        bg_emb, bg_reps, bg_ids, _, bg_index_path = load_embeddings(
            path=args.background,
            args=args,
            is_sample=False,
            lib=lib,
            locus=locus,
            prefix=f"{args.chain.lower()}_background",
            output_path=paths.tcremp_dir,
        )

        transform, transform_path = load_or_fit_background_transform(args=args, output_root=output_root, bg_emb=bg_emb)
        bg_pca = transform.background_pca_
        if bg_pca is None:
            bg_pca = transform.transform_pca(bg_emb)
        plot_bg_points = min(len(bg_pca), args.n_bg_points or DEFAULT_PLOT_BG_POINTS)
        bg_umap = prepare_background_umap(transform, bg_pca, plot_bg_points, transform_path=transform_path)

        clustered_tables = []
        cluster_members_tables = []
        for epitope, ep_df in vdjdb_df.groupby("antigen.epitope", sort=True):
            clustered_df, cluster_members_df = process_epitope(
                epitope,
                ep_df,
                args=args,
                genes=genes,
                locus=locus,
                lib=lib,
                proto=proto,
                paths=paths,
                bg_pca=bg_pca,
                bg_reps=bg_reps,
                bg_ids=bg_ids,
                bg_index_path=bg_index_path,
                bg_umap=bg_umap,
                transform=transform,
            )
            clustered_tables.append(clustered_df)
            cluster_members_tables.append(cluster_members_df)

        chain_lower = args.chain.lower()
        if clustered_tables:
            pd.concat(clustered_tables, ignore_index=True).to_csv(
                paths.tcrempnet_dir / f"{chain_lower}_vdjdb_clustered_clonotypes.tsv", sep="\t", index=False
            )
        if cluster_members_tables:
            pd.concat(cluster_members_tables, ignore_index=True).to_csv(
                output_root / "cluster_members.txt", sep="\t", index=False
            )

        logging.info("Done")
    except Exception as e:
        logging.error("An error occurred: %s", e)
        raise


if __name__ == "__main__":
    main()

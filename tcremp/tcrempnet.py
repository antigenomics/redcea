import sys
import os
import gc
import logging
import multiprocessing as mp
import pandas as pd
from pathlib import Path
import faiss

from pympler import asizeof, muppy, summary as sm

sys.path.append("../")
sys.path.append("../../mirpy")

from tcremp.arguments import get_arguments_enrich
from tcremp.utils import (
    configure_logging, load_prototype_repertoire, load_analysis_repertoire,
    get_representations_df, resolve_prototype_file, resolve_input_file,
    prepare_output_path, generate_output_prefix, subsample_repertoire,
    log_memory_usage, resolve_embedding_file, add_z_binom_pvalues, add_log_fold_change
)
from tcremp.tcremp_run import run_tcremp_embedding
from mir.common.segments import SegmentLibrary
from tcremp.tcremp_cluster import (
    run_dbscan_clustering,
    prepare_data_for_clustering,
    estimate_dbscan_eps,
    compute_blockwise_knn_merged,
    run_leiden_clustering,
    hierarchical_leiden_clustering,
    hierarchical_leiden_dbscan_clustering
)


def setup_environment(args):
    input_sample_path = resolve_input_file(args.sample)
    input_background_path = resolve_input_file(args.background)
    proto_path = resolve_prototype_file(args.prototypes_path)
    output_path = prepare_output_path(args.output)
    prefix = generate_output_prefix(args.sample, args.prefix)

    configure_logging(input_sample_path, output_path, prefix)
    chain = args.chain.split('_')
    locus = {'TRA': 'alpha', 'TRB': 'beta', 'TRA_TRB': None}[args.chain]
    lib = SegmentLibrary.load_default(genes=chain, organisms=args.species)
    faiss.omp_set_num_threads(args.nproc)
    print(f"FAISS threads set to {faiss.omp_get_max_threads()}")

    return input_sample_path, input_background_path, proto_path, output_path, prefix, chain, locus, lib


def compute_embeddings_if_needed(path, args, is_sample, proto, chain, lib, locus, prefix, output_path):
    tag = 'sample' if is_sample else 'background'
    custom_path = args.sample_embedding if is_sample else args.background_embedding
    emb_path = resolve_embedding_file(custom_path, output_path, prefix, tag)

    if emb_path.exists():
        logging.info(f"Found existing {tag} embeddings at {emb_path}")
        return

    logging.info(f"Computing {tag} embeddings...")
    
        # -------------------------
    # NEW: take only first N bg points
    # -------------------------
    if (not is_sample) and args.n_bg_points:
        logging.info(f"Subsampling background to first {args.n_bg_points} clonotypes")
        rep = rep.sample_n(args.n_bg_points, sample_random=False)
    # -------------------------
    
    rep = load_analysis_repertoire(path, lib, locus, args.index_col, args.lower_len_cdr3, args.higher_len_cdr3)
    rep = subsample_repertoire(rep, args.n_clonotypes, args.sample_random_prototypes, args.random_seed)
    run_tcremp_embedding(rep, proto, lib, chain, args.metrics, args.nproc, emb_path)


def load_embeddings(path, args, is_sample, lib, locus, prefix, output_path):
    tag = 'sample' if is_sample else 'background'
    prefix_tag = 's_' if is_sample else 'b_'
    custom_path = args.sample_embedding if is_sample else args.background_embedding
    emb_path = resolve_embedding_file(custom_path, output_path, prefix, tag, must_exist=True)

    emb = pd.read_parquet(emb_path)

    # -------------------------
    # NEW: limit background embeddings
    # -------------------------
    if (not is_sample) and args.n_bg_points:
        logging.info(f"Restricting background embeddings to first {args.n_bg_points}")
        emb = emb.iloc[:args.n_bg_points]
    # -------------------------

    rep = load_analysis_repertoire(path, lib, locus, args.index_col, args.lower_len_cdr3, args.higher_len_cdr3)

    # -------------------------
    # NEW: restrict background rep to match embeddings
    # -------------------------
    if (not is_sample) and args.n_bg_points:
        rep = rep.sample_n(args.n_bg_points, sample_random=False)
    # -------------------------

    rep = subsample_repertoire(rep, args.n_clonotypes, args.sample_random_prototypes, args.random_seed)

    rep_df = get_representations_df(rep, locus)
    rep_df['clone_id'] = prefix_tag + rep_df['clone_id'].astype(str)
    ids = pd.Series([f'{prefix_tag}{c.id}' for c in rep])

    del rep
    gc.collect()
    return emb, rep_df, ids

def compute_cluster_summary(cluster_df, sample_ids):
    sample_ids_set = set(sample_ids)
    cluster_df['source'] = cluster_df['clone_id'].apply(
        lambda x: 'sample' if x in sample_ids_set else 'background')

    summary = (
        cluster_df.groupby('cluster_id')['source']
        .value_counts()
        .unstack(fill_value=0)
        .rename_axis(index='cluster_id', columns=None)
        .reset_index()
    )
    summary['cluster_size'] = summary.get('sample', 0) + summary.get('background', 0)
    summary = summary[summary.cluster_id != -1]
    return summary


def main():
    args = get_arguments_enrich()

    input_sample_path, input_background_path, proto_path, output_path, prefix, chain, locus, lib = setup_environment(args)

    log_memory_usage('Init')
    logging.info("Starting TCRempNet pipeline...")

    logging.info("Computing sample embeddings if needed...")
    compute_embeddings_if_needed(
        input_sample_path, args, is_sample=True, proto=proto_path,
        chain=chain, lib=lib, locus=locus, prefix=prefix, output_path=output_path
    )

    logging.info("Computing background embeddings if needed...")
    compute_embeddings_if_needed(
        input_background_path, args, is_sample=False, proto=proto_path,
        chain=chain, lib=lib, locus=locus, prefix=prefix, output_path=output_path
    )

    logging.info("Loading sample embeddings...")
    sample_emb, sample_representations, sample_ids = load_embeddings(
        input_sample_path, args, is_sample=True, lib=lib,
        locus=locus, prefix=prefix, output_path=output_path
    )

    logging.info("Loading background embeddings...")
    background_emb, background_representations, background_ids = load_embeddings(
        input_background_path, args, is_sample=False, lib=lib,
        locus=locus, prefix=prefix, output_path=output_path
    )

    log_memory_usage("After loading embeddings")

    sample_index_path = None
    bg_index_path = None

    # если заданы конкретные пути к parquet — используем их как основу
    sample_index_path = Path(args.sample_embedding).with_suffix('').with_name(
        Path(args.sample_embedding).stem + "_faiss.index"
    )
    bg_index_path = Path(args.background_embedding).with_suffix('').with_name(
        Path(args.background_embedding).stem + "_faiss.index"
    )

    logging.info(f"Sample FAISS index path: {sample_index_path}")
    logging.info(f"Background FAISS index path: {bg_index_path}")

    joint_embeddings = pd.concat([sample_emb, background_emb], ignore_index=True)
    joint_representations = pd.concat(
        [sample_representations, background_representations],
        ignore_index=True
    )
    sample_size = len(sample_emb)
    joint_ids = pd.concat([sample_ids, background_ids], ignore_index=True)

    del sample_emb
    del background_emb

    log_memory_usage("After concatenation")

    logging.info("Running clustering...\n")

    logging.info('Preparing data for clustering...')
    df = prepare_data_for_clustering(joint_embeddings, n_components=args.cluster_pc_components)

    # === Новый шаг: блочный расчёт расстояний через FAISS ===
    logging.info('Evaluating blockwise k-neighbors distance matrix via FAISS...')
    bg_data = df[sample_size:]
    sample_data = df[:sample_size]

    distances, indices = compute_blockwise_knn_merged(
        bg=bg_data,
        sample=sample_data,
        k_neighbors=args.k_neighbors,
        bg_index_path=bg_index_path,
        sample_index_path=sample_index_path,
        rebuild_bg=False,
        rebuild_sample=True,
        save_blocks=True,
        output_dir=output_path,
        nproc=args.nproc,
    )

    # -------------------------------------------------
    # Clustering options
    #   1) plain Leiden on kNN graph
    #   2) hierarchical Leiden (Seurat-like)
    #   3) NEW: Leiden -> DBSCAN (parallel inside communities)
    # -------------------------------------------------
    
    if args.cluster_algo == "leiden_dbscan":
        cluster_labels = hierarchical_leiden_dbscan_clustering(
            data_reduced=df,
            knn_indices=indices,
            knn_distances=distances,
            resolution=args.leiden_resolution,
            k_neighbors=args.k_neighbors,
            n_jobs=args.nproc
        )
    elif args.cluster_algo == "vdbscan":
        logging.info("Running variable-eps DBSCAN (vDBSCAN)")

        # 1) длины CDR3 — из уже загруженных representations
        cdr3_lengths = (
            joint_representations['cdr3aa_beta']
            .astype(str)
            .str.len()
            .to_numpy()
        )

        # 2) vDBSCAN: переиспользуем уже посчитанные distances
        cluster_labels, vdbscan_info = variable_eps_dbscan(
            X=df.values,                       # embedding after PCA / UMAP
            cdr3_lengths=cdr3_lengths,
            knn_distances=distances,           # <-- reuse!
        )

        # 3) (опционально) логируем бакеты
        logging.info("vDBSCAN length buckets:")
        for b, eps in zip(vdbscan_info["buckets"], vdbscan_info["bucket_eps"]):
            logging.info(
                f"  lengths={b['lengths'][0]}..{b['lengths'][-1]} "
                f"size={b['size']} frac={b['frac']:.3f} eps={eps:.6f}"
            )
    else:
        cluster_labels = run_leiden_clustering(
            knn_indices=knn_indices,
            knn_distances=knn_distances,
            resolution=args.leiden_resolution,
            n_jobs=args.nproc
        )
    
    # cluster_labels = hierarchical_leiden_clustering(
    #     knn_indices=indices,
    #     knn_distances=distances,
    #     base_resolution=1.0,
    #     sub_resolution=3.0,
    #     n_iterations=3,
    #     n_threads=args.nproc,
    #     metric="dissimilarity",
    #     min_cluster_size=5,
    #     sub_min_cluster_size=5,
    # )
    

    
    log_memory_usage("After clustering")

    cluster_df = pd.DataFrame({'clone_id': joint_ids, 'cluster_id': cluster_labels})
    cluster_df = cluster_df.merge(joint_representations)
    cluster_df.to_csv(f"{output_path}/{prefix}_tcremp_clusters.tsv", sep='\t', index=False)
    logging.info("Saved cluster assignments.")

    summary = compute_cluster_summary(cluster_df, sample_ids)
    summary = add_z_binom_pvalues(summary, total_sample=len(sample_ids), total_background=len(background_ids))
    summary = add_log_fold_change(summary, total_sample=len(sample_ids), total_background=len(background_ids))
    summary[['cluster_id', 'cluster_size', 'sample', 'background', 'enrichment_pvalue_zbinom', 'enrichment_fdr_zbinom',
             'log_fold_change']].to_csv(
        f"{output_path}/{prefix}_summary_tcrempnet.tsv", sep='\t', index=False)
    logging.info("Saved cluster summary with p-values.")

    enriched_clusters = summary.loc[
        summary['enrichment_fdr_zbinom'] < 0.05, ['cluster_id', 'enrichment_pvalue_zbinom']
    ]
    logging.info(f"{len(enriched_clusters)} clusters identified as enriched (fdr < 0.05).")

    enriched_clonotypes = cluster_df.merge(enriched_clusters).merge(joint_representations)
    enriched_clonotypes.to_csv(
        f"{output_path}/{prefix}_enriched_clonotypes_tcremp.tsv", sep='\t', index=False
    )
    logging.info("Saved enriched clonotypes.")

    joint_embeddings['clone_id'] = joint_ids
    enriched_embeddings = enriched_clonotypes[
        ['clone_id', 'cluster_id', 'source', 'enrichment_pvalue_zbinom']].merge(joint_embeddings)
    enriched_embeddings.to_parquet(
        f"{output_path}/{prefix}_enriched_embeddings_tcremp.parquet"
    )
    logging.info("Saved enriched embeddings.")

    logging.info("TCRempNet pipeline completed.")
    log_memory_usage("Finished")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()

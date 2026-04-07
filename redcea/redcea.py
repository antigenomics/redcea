import gc
import logging
import multiprocessing as mp
import os
from pathlib import Path

import faiss
import pandas as pd
from mir.common.segments import SegmentLibrary
from tcremp.tcremp_run import run_tcremp_embedding

from redcea.arguments import get_arguments_enrich
from redcea.clustering import (
    build_joint_knn_from_split,
    build_len_to_group_id,
    compute_blockwise_knn_merged,
    compute_cdr3_len,
    eps_per_point_from_group_id,
    estimate_eps_by_group_flexible,
    estimate_eps_by_group_from_sample,
    hierarchical_leiden_clustering,
    hierarchical_leiden_dbscan_clustering,
    map_len_to_group_id,
    prepare_data_for_clustering,
    run_leiden_clustering,
    vdbscan_from_knn,
)
from redcea.utils import (
    add_log_fold_change,
    add_z_binom_pvalues,
    configure_logging,
    generate_output_prefix,
    get_representations_df,
    load_analysis_repertoire,
    load_prototype_repertoire,
    log_memory_usage,
    prepare_output_path,
    resolve_embedding_file,
    resolve_input_file,
    resolve_prototype_file,
    subsample_repertoire,
)


def setup_environment(args):
    input_sample_path = resolve_input_file(args.sample)
    input_background_path = resolve_input_file(args.background)
    proto_path = resolve_prototype_file(args.prototypes_path, args.chain)
    output_path = prepare_output_path(args.output)
    prefix = generate_output_prefix(args.sample, args.prefix)

    configure_logging(input_sample_path, output_path, prefix)
    chain = args.chain.split("_")
    locus = {"TRA": "alpha", "TRB": "beta", "TRA_TRB": None}[args.chain]
    lib = SegmentLibrary.load_default(genes=chain, organisms=args.species)

    if args.nproc is None:
        args.nproc = max(1, min(8, os.cpu_count() or 1))

    faiss.omp_set_num_threads(args.nproc)
    logging.info("Using nproc=%d (override with --nproc)", args.nproc)
    print(f"FAISS threads set to {faiss.omp_get_max_threads()}")

    return input_sample_path, input_background_path, proto_path, output_path, prefix, chain, locus, lib


def compute_embeddings_if_needed(path, args, is_sample, proto, chain, lib, locus, prefix, output_path):
    tag = "sample" if is_sample else "background"
    custom_path = args.sample_embedding if is_sample else args.background_embedding
    emb_path = resolve_embedding_file(custom_path, output_path, prefix, tag)

    if emb_path.exists():
        logging.info("Found existing %s embeddings at %s", tag, emb_path)
        return emb_path

    logging.info("Computing %s embeddings...", tag)
    rep = load_analysis_repertoire(path, lib, locus, args.index_col, args.lower_len_cdr3, args.higher_len_cdr3)
    logging.info("Loaded %s repertoire with %d clonotypes after parsing/length filtering", tag, len(rep.clonotypes))

    if (not is_sample) and getattr(args, "n_bg_points", None):
        logging.info("Restricting background repertoire to first %d clonotypes (pre-embedding)", args.n_bg_points)
        rep = rep.sample_n(args.n_bg_points, sample_random=False)

    rep = subsample_repertoire(rep, args.n_clonotypes, args.sample_random_clonotypes, args.random_seed)
    logging.info("Prepared %s repertoire with %d clonotypes before embedding", tag, len(rep.clonotypes))
    if len(rep.clonotypes) == 0:
        raise ValueError(
            f"{tag.capitalize()} repertoire is empty after filtering/subsampling: {path}. "
            "Check AIRR conversion, chain selection, and CDR3 length thresholds."
        )

    if not hasattr(proto, "total"):
        proto_path = resolve_prototype_file(str(proto) if proto else None, args.chain)
        proto = load_prototype_repertoire(proto_path, lib, locus, args.index_col)

    embeddings = run_tcremp_embedding(rep, proto, lib, chain, args.metrics, args.nproc)
    embeddings.to_parquet(emb_path, index=False)
    return emb_path


def load_embeddings(path, args, is_sample, lib, locus, prefix, output_path):
    tag = "sample" if is_sample else "background"
    prefix_tag = "s_" if is_sample else "b_"
    custom_path = args.sample_embedding if is_sample else args.background_embedding
    emb_path = resolve_embedding_file(custom_path, output_path, prefix, tag, must_exist=True)
    index_path = Path(emb_path).with_suffix("").with_name(Path(emb_path).stem + "_faiss.index")

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
    return emb, rep_df, ids, emb_path, index_path


def compute_cluster_summary(cluster_df, sample_ids):
    sample_ids_set = set(sample_ids)
    cluster_df["source"] = cluster_df["clone_id"].apply(lambda x: "sample" if x in sample_ids_set else "background")

    summary = (
        cluster_df.groupby("cluster_id")["source"]
        .value_counts()
        .unstack(fill_value=0)
        .rename_axis(index="cluster_id", columns=None)
        .reset_index()
    )
    summary["cluster_size"] = summary.get("sample", 0) + summary.get("background", 0)
    summary = summary[summary.cluster_id != -1]
    return summary


def _pick_cdr3_col(joint_representations: pd.DataFrame) -> str:
    if "cdr3aa_beta" in joint_representations.columns:
        return "cdr3aa_beta"
    if "cdr3aa_alpha" in joint_representations.columns:
        return "cdr3aa_alpha"
    if "cdr3aa" in joint_representations.columns:
        return "cdr3aa"
    raise ValueError(
        "Cannot find CDR3 AA column in representations. "
        "Expected one of: cdr3aa_beta, cdr3aa_alpha, cdr3aa"
    )


def main():
    args = get_arguments_enrich()

    input_sample_path, input_background_path, proto_path, output_path, prefix, chain, locus, lib = setup_environment(args)

    log_memory_usage("Init")
    logging.info("Starting TCRempNet pipeline...")

    logging.info("Computing sample embeddings if needed...")
    compute_embeddings_if_needed(
        input_sample_path,
        args,
        is_sample=True,
        proto=proto_path,
        chain=chain,
        lib=lib,
        locus=locus,
        prefix=prefix,
        output_path=output_path,
    )

    logging.info("Computing background embeddings if needed...")
    compute_embeddings_if_needed(
        input_background_path,
        args,
        is_sample=False,
        proto=proto_path,
        chain=chain,
        lib=lib,
        locus=locus,
        prefix=prefix,
        output_path=output_path,
    )

    logging.info("Loading sample embeddings...")
    sample_emb, sample_representations, sample_ids, _, sample_index_path = load_embeddings(
        input_sample_path,
        args,
        is_sample=True,
        lib=lib,
        locus=locus,
        prefix=prefix,
        output_path=output_path,
    )

    logging.info("Loading background embeddings...")
    background_emb, background_representations, background_ids, _, bg_index_path = load_embeddings(
        input_background_path,
        args,
        is_sample=False,
        lib=lib,
        locus=locus,
        prefix=prefix,
        output_path=output_path,
    )

    log_memory_usage("After loading embeddings")
    logging.info("Sample FAISS index path: %s", sample_index_path)
    logging.info("Background FAISS index path: %s", bg_index_path)

    joint_embeddings = pd.concat([sample_emb, background_emb], ignore_index=True)
    joint_representations = pd.concat([sample_representations, background_representations], ignore_index=True)
    sample_size = len(sample_emb)
    joint_ids = pd.concat([sample_ids, background_ids], ignore_index=True)

    del sample_emb
    del background_emb

    log_memory_usage("After concatenation")
    logging.info("Running clustering...")
    logging.info("Preparing data for clustering...")
    df = prepare_data_for_clustering(joint_embeddings, n_components=args.cluster_pc_components)

    bg_data = df[sample_size:]
    sample_data = df[:sample_size]

    logging.info("Evaluating kNN via FAISS (split caches: sample-sample, bg-bg; cross on-the-fly)...")
    knn_t0 = pd.Timestamp.now()
    dist_ss, ind_ss, dist_bb, ind_bb, dist_sb, ind_sb, dist_bs, ind_bs = compute_blockwise_knn_merged(
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
        bg_size_truncation=getattr(args, "n_bg_points", None),
    )
    logging.info("Split FAISS kNN finished in %.2fs", (pd.Timestamp.now() - knn_t0).total_seconds())

    distances, indices = build_joint_knn_from_split(
        dist_ss=dist_ss,
        ind_ss=ind_ss,
        dist_bb=dist_bb,
        ind_bb=ind_bb,
        dist_sb=dist_sb,
        ind_sb=ind_sb,
        dist_bs=dist_bs,
        ind_bs=ind_bs,
        k_out=args.k_neighbors,
    )

    if args.cluster_algo == "leiden_dbscan":
        cluster_labels = hierarchical_leiden_dbscan_clustering(
            data_reduced=df,
            knn_indices=indices,
            knn_distances=distances,
            resolution=args.leiden_resolution,
            k_neighbors=args.eps_k_neighbors,
            num_points_for_core=args.cluster_min_samples,
            n_jobs=args.nproc,
        )
    elif args.cluster_algo == "hierarchical_leiden":
        cluster_labels = hierarchical_leiden_clustering(
            knn_indices=indices,
            knn_distances=distances,
            base_resolution=args.leiden_resolution,
            sub_resolution=args.leiden_sub_resolution,
            n_iterations=3,
            n_threads=args.nproc,
            metric="dissimilarity",
            min_cluster_size=5,
            sub_min_cluster_size=3,
        )
    elif args.cluster_algo == "leiden":
        cluster_labels = run_leiden_clustering(
            knn_indices=indices,
            knn_distances=distances,
            resolution=args.leiden_resolution,
            n_jobs=args.nproc,
        )
    elif args.cluster_algo == "vdbscan":
        cdr3_col = _pick_cdr3_col(joint_representations)
        eps_estimation_based_on = args.eps_estimation_based_on
        kth_neighbor_for_eps = args.eps_k_neighbors

        if eps_estimation_based_on == "sample":
            logging.info("Running vDBSCAN (eps-by-group from SAMPLE only; L2 distances)")
            sample_len = compute_cdr3_len(sample_representations[cdr3_col])
            len_to_gid = build_len_to_group_id(sample_len, min_frac=0.05)

            sample_gid = map_len_to_group_id(sample_len, len_to_gid, unknown_len_policy="nearest")
            bg_len = compute_cdr3_len(background_representations[cdr3_col])
            bg_gid = map_len_to_group_id(bg_len, len_to_gid, unknown_len_policy="nearest")

            eps_by_gid = estimate_eps_by_group_from_sample(
                sample_group_id=sample_gid,
                sample_ss_distances_l2=dist_ss,
                kth_neighbor=int(kth_neighbor_for_eps),
            )
            gid_all = pd.concat([pd.Series(sample_gid), pd.Series(bg_gid)], ignore_index=True).to_numpy(dtype="int32")
        elif eps_estimation_based_on == "background":
            logging.info("Running vDBSCAN (eps-by-group from BACKGROUND only; L2 distances)")
            bg_len = compute_cdr3_len(background_representations[cdr3_col])
            len_to_gid = build_len_to_group_id(bg_len, min_frac=0.05)

            bg_gid = map_len_to_group_id(bg_len, len_to_gid, unknown_len_policy="nearest")
            sample_len = compute_cdr3_len(sample_representations[cdr3_col])
            sample_gid = map_len_to_group_id(sample_len, len_to_gid, unknown_len_policy="nearest")

            eps_by_gid = estimate_eps_by_group_from_sample(
                sample_group_id=bg_gid,
                sample_ss_distances_l2=dist_bb,
                kth_neighbor=int(kth_neighbor_for_eps),
            )
            gid_all = pd.concat([pd.Series(sample_gid), pd.Series(bg_gid)], ignore_index=True).to_numpy(dtype="int32")
        elif eps_estimation_based_on == "all":
            logging.info("Running vDBSCAN (eps-by-group from SAMPLE+BACKGROUND combined; L2 distances)")
            joint_len = compute_cdr3_len(joint_representations[cdr3_col])
            len_to_gid = build_len_to_group_id(joint_len, min_frac=0.05)

            joint_gid = map_len_to_group_id(joint_len, len_to_gid, unknown_len_policy="nearest")
            gid_all = joint_gid

            eps_by_gid = estimate_eps_by_group_flexible(
                group_id=gid_all,
                dist_matrix=distances,
                kth_neighbor=int(kth_neighbor_for_eps),
            )
        else:
            raise ValueError(
                f"Unknown eps_estimation_based_on='{eps_estimation_based_on}'. "
                "Expected one of: sample, background, all"
            )

        eps_i_all = eps_per_point_from_group_id(gid_all, eps_by_gid)
        cluster_labels = vdbscan_from_knn(
            knn_indices=indices,
            knn_distances_l2=distances,
            eps_i_l2=eps_i_all,
            num_points_for_core=args.cluster_min_samples,
            sym_rule=args.vdbscan_sym_rule,
        )
    else:
        raise ValueError(
            f"Unknown args.cluster_algo='{args.cluster_algo}'. "
            "Expected one of: leiden_dbscan, hierarchical_leiden, leiden, vdbscan"
        )

    log_memory_usage("After clustering")

    cluster_df = pd.DataFrame({"clone_id": joint_ids, "cluster_id": cluster_labels}).merge(joint_representations)
    cluster_df.to_csv(f"{output_path}/{prefix}_tcremp_clusters.tsv", sep="\t", index=False)
    logging.info("Saved cluster assignments.")

    summary = compute_cluster_summary(cluster_df, sample_ids)
    summary = add_z_binom_pvalues(summary, total_sample=len(sample_ids), total_background=len(background_ids))
    summary = add_log_fold_change(summary, total_sample=len(sample_ids), total_background=len(background_ids))
    summary[
        [
            "cluster_id",
            "cluster_size",
            "sample",
            "background",
            "enrichment_pvalue_zbinom",
            "enrichment_fdr_zbinom",
            "log_fold_change",
        ]
    ].to_csv(f"{output_path}/{prefix}_summary_tcrempnet.tsv", sep="\t", index=False)
    logging.info("Saved cluster summary with p-values.")

    enriched_clusters = summary.loc[
        (summary["enrichment_fdr_zbinom"] < 0.05) & (summary["log_fold_change"] > 0),
        ["cluster_id", "enrichment_pvalue_zbinom"],
    ]
    logging.info("%d clusters identified as enriched (fdr < 0.05, logFC > 0).", len(enriched_clusters))

    enriched_clonotypes = cluster_df.merge(enriched_clusters).merge(joint_representations)
    enriched_clonotypes.to_csv(f"{output_path}/{prefix}_enriched_clonotypes_tcremp.tsv", sep="\t", index=False)
    logging.info("Saved enriched clonotypes.")

    joint_embeddings["clone_id"] = joint_ids
    enriched_embeddings = enriched_clonotypes[
        ["clone_id", "cluster_id", "source", "enrichment_pvalue_zbinom"]
    ].merge(joint_embeddings)
    enriched_embeddings.to_parquet(f"{output_path}/{prefix}_enriched_embeddings_tcremp.parquet")
    logging.info("Saved enriched embeddings.")

    logging.info("TCRempNet pipeline completed.")
    log_memory_usage("Finished")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()

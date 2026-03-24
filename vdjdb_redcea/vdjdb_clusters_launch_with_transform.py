#!/usr/bin/env python
from __future__ import annotations

import gc
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from mir.common.segments import SegmentLibrary
from tcremp.utils import load_prototype_repertoire, resolve_prototype_file, subsample_repertoire

try:
    from tcremp.arguments import get_arguments_vdjdb_clusters
    from tcremp.utils import configure_logging, prepare_output_path, add_z_binom_pvalues, add_log_fold_change
    from tcremp.tcrempnet import compute_embeddings_if_needed, load_embeddings, compute_cluster_summary
    from tcremp.clustering import compute_blockwise_knn_merged, run_leiden_clustering
    from tcremp.background_transform import BackgroundTransform
except ImportError:
    from arguments_refactored import get_arguments_vdjdb_clusters
    from utils import configure_logging, prepare_output_path, add_z_binom_pvalues, add_log_fold_change
    from tcrempnet import compute_embeddings_if_needed, load_embeddings, compute_cluster_summary
    from tcremp_cluster import compute_blockwise_knn_merged, run_leiden_clustering
    from background_transform import BackgroundTransform


CHAIN_COLS = {
    'TRA': {'cdr3': 'cdr3.alpha', 'v': 'v.alpha', 'j': 'j.alpha', 'locus': 'alpha', 'gene': 'alpha'},
    'TRB': {'cdr3': 'cdr3.beta', 'v': 'v.beta', 'j': 'j.beta', 'locus': 'beta', 'gene': 'beta'},
}
DEFAULT_PLOT_BG_POINTS = 20_000


def build_airr_from_epitope(ep_df: pd.DataFrame, chain: str) -> pd.DataFrame:
    cfg = CHAIN_COLS[chain]
    out = ep_df[[cfg['cdr3'], cfg['v'], cfg['j']]].copy()
    out.columns = ['junction_aa', 'v_call', 'j_call']
    out['locus'] = cfg['locus']
    logging.info('Built AIRR table with %d rows for chain %s', len(out), chain)
    out.dropna(subset=['junction_aa'], inplace=True)
    logging.info('After dropping rows with missing CDR3, %d rows remain', len(out))
    return out.reset_index(drop=True)


def resolve_joint_knn(
    sample_pca: np.ndarray,
    bg_pca: np.ndarray,
    k_neighbors: int,
    nproc: int,
    sample_index_path: Path,
    bg_index_path: Path,
):
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

    joint = np.vstack([sample_pca, bg_pca]).astype('float32', copy=False)
    index = faiss.IndexFlatL2(joint.shape[1])
    faiss.omp_set_num_threads(int(nproc))
    index.add(joint)
    return index.search(joint, k_neighbors)


def build_sample_members_table(sample_cluster_df: pd.DataFrame, summary_df: pd.DataFrame, chain: str,
                               epitope_df: pd.DataFrame, epitope: str) -> pd.DataFrame:
    cfg = CHAIN_COLS[chain]
    cdr3_col = f"cdr3aa_{cfg['gene']}"
    v_col = f"v_{cfg['gene']}"
    j_col = f"j_{cfg['gene']}"

    meta = epitope_df.iloc[0]
    cluster_sizes = summary_df.set_index('cluster_id')['cluster_size'].to_dict()
    chain_tag = 'A' if chain == 'TRA' else 'B'

    return pd.DataFrame({
        'species': meta['species'],
        'antigen.epitope': epitope,
        'antigen.gene': meta['antigen.gene'],
        'antigen.species': meta['antigen.species'],
        'mhc.a': meta['mhc.a'],
        'mhc.b': meta['mhc.b'],
        'mhc.class': meta['mhc.class'],
        'gene': chain,
        'cdr3aa': sample_cluster_df[cdr3_col].values,
        'cid': [f"H.{chain_tag}.{epitope}.{int(cid)}" for cid in sample_cluster_df['cluster_id'].values],
        'csz': [int(cluster_sizes[int(cid)]) for cid in sample_cluster_df['cluster_id'].values],
        'v.segm': sample_cluster_df[v_col].values,
        'j.segm': sample_cluster_df[j_col].values,
        'cluster_id': sample_cluster_df['cluster_id'].values,
    })


def load_or_fit_background_transform(*, args, output_root: Path, bg_emb: pd.DataFrame) -> BackgroundTransform:
    default_path = output_root / 'tcremp' / f"{args.chain.lower()}_background_transform.joblib"
    transform_path = Path(getattr(args, 'background_transform', None) or default_path)

    if transform_path.exists():
        logging.info('Loading background transform from %s', transform_path)
        transform = BackgroundTransform.load(transform_path)
    else:
        logging.info('Fitting background transform on %d background clonotypes', len(bg_emb))
        transform = BackgroundTransform(
            chain=args.chain,
            n_pca_components=args.cluster_pc_components,
            random_state=args.random_seed,
        )
        transform.fit(bg_emb)
        transform.save(transform_path)
        logging.info('Saved background transform to %s', transform_path)

    return transform


def prepare_background_umap(transform: BackgroundTransform, bg_pca: np.ndarray, n_bg_points: int) -> np.ndarray:
    bg_pca_subset = np.asarray(bg_pca[:n_bg_points], dtype=np.float32)
    if bg_pca_subset.size == 0:
        raise ValueError('Cannot prepare background UMAP for empty background PCA array')

    if transform.umap_model is None:
        logging.info('Fitting UMAP for plotting on first %d background clonotypes', len(bg_pca_subset))
        return transform.fit_umap(bg_pca_subset)

    if transform.background_umap_ is not None and len(transform.background_umap_) >= len(bg_pca_subset):
        return np.asarray(transform.background_umap_[:len(bg_pca_subset)], dtype=np.float32)

    logging.info('Using existing UMAP model to transform first %d background clonotypes for plotting', len(bg_pca_subset))
    return transform.transform_umap(bg_pca_subset)


def build_cluster_plot(
    *,
    epitope: str,
    chain: str,
    sample_reps: pd.DataFrame,
    sample_ids: pd.Series,
    sample_labels: np.ndarray,
    significant_cluster_ids: set[int],
    sample_umap: np.ndarray,
    bg_umap: np.ndarray,
) -> go.Figure:
    cfg = CHAIN_COLS[chain]
    sample_df = sample_reps.copy().reset_index(drop=True)
    sample_df['clone_id'] = sample_ids.to_numpy()
    sample_df['cluster_id'] = sample_labels
    sample_df['x'] = sample_umap[:, 0]
    sample_df['y'] = sample_umap[:, 1]
    sample_df['cluster'] = np.where(
        (sample_df['cluster_id'] != -1) & sample_df['cluster_id'].isin(significant_cluster_ids),
        sample_df['cluster_id'].astype(str),
        'unclustered',
    )

    hover_cols = [
        col for col in (
            f"cdr3aa_{cfg['gene']}",
            f"v_{cfg['gene']}",
            f"j_{cfg['gene']}",
            'clone_id',
            'cluster_id',
        )
        if col in sample_df.columns
    ]

    fig_scatter = px.scatter(
        sample_df,
        x='x',
        y='y',
        color='cluster',
        color_discrete_map={'unclustered': 'lightgrey'},
        hover_data=hover_cols,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Histogram2dContour(
            x=bg_umap[:, 0],
            y=bg_umap[:, 1],
            ncontours=20,
            contours=dict(coloring='fill', showlines=False),
            colorscale='Greys',
            showscale=False,
            hoverinfo='skip',
            opacity=0.35,
        )
    )
    for trace in fig_scatter.data:
        fig.add_trace(trace)

    fig.update_layout(
        title=f'TCR clustering with background density ({epitope})',
        width=1000,
        height=700,
        template='plotly_white',
    )
    return fig


def save_cluster_plot_html(
    *,
    epitope: str,
    chain: str,
    sample_reps: pd.DataFrame,
    sample_ids: pd.Series,
    sample_pca: np.ndarray,
    sample_labels: np.ndarray,
    significant_cluster_ids: set[int],
    bg_umap: np.ndarray,
    transform: BackgroundTransform,
    output_path: Path,
) -> None:
    sample_umap = transform.transform_umap(sample_pca)
    fig = build_cluster_plot(
        epitope=epitope,
        chain=chain,
        sample_reps=sample_reps,
        sample_ids=sample_ids,
        sample_labels=sample_labels,
        significant_cluster_ids=significant_cluster_ids,
        sample_umap=sample_umap,
        bg_umap=bg_umap,
    )
    output_path.write_text(fig.to_html(full_html=True, include_plotlyjs='cdn'), encoding='utf-8')
    logging.info('Saved cluster plot HTML to %s', output_path)


def process_epitope(epitope: str, ep_df: pd.DataFrame, *, args, genes: list[str], locus: str, lib: SegmentLibrary,
                    proto, airr_dir: Path, tcremp_dir: Path, tcrempnet_dir: Path,
                    bg_pca: np.ndarray, bg_reps: pd.DataFrame, bg_ids: pd.Series, bg_index_path: Path,
                    bg_umap: np.ndarray,
                    transform: BackgroundTransform) -> tuple[pd.DataFrame, pd.DataFrame]:
    chain = args.chain
    prefix = f"{chain.lower()}_vdjdb_{epitope}"
    airr_path = airr_dir / f"{prefix}.tsv"
    sample_emb_path = tcremp_dir / f"{prefix}_sample_embeddings.parquet"

    logging.info(f'Processing epitope {epitope} with {len(ep_df)} clonotypes')
    build_airr_from_epitope(ep_df, chain).to_csv(airr_path, sep='\t', index=False)

    args.sample = str(airr_path.resolve())
    args.sample_embedding = str(sample_emb_path.resolve())
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
        output_path=tcremp_dir,
    )

    sample_emb, sample_reps, sample_ids, _, sample_index_path = load_embeddings(
        path=args.sample,
        args=args,
        is_sample=True,
        lib=lib,
        locus=locus,
        prefix=prefix,
        output_path=tcremp_dir,
    )

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

    joint_ids = pd.concat([sample_ids, bg_ids], ignore_index=True)
    joint_reps = pd.concat([sample_reps, bg_reps], ignore_index=True)
    cluster_df = pd.DataFrame({'clone_id': joint_ids, 'cluster_id': labels}).merge(joint_reps, on='clone_id', how='left')
    cluster_df.to_csv(tcrempnet_dir / f"{prefix}_tcremp_clusters.tsv", sep='\t', index=False)

    summary_df = compute_cluster_summary(cluster_df.copy(), sample_ids)
    summary_df = add_z_binom_pvalues(
        summary_df,
        total_sample=len(sample_ids),
        total_background=len(bg_ids),
    )
    summary_df = add_log_fold_change(
        summary_df,
        total_sample=len(sample_ids),
        total_background=len(bg_ids),
    )
    summary_df['significant'] = (
        (summary_df['enrichment_fdr_zbinom'] < 0.05) &
        (summary_df['log_fold_change'] > 0)
    )
    significant_cluster_ids = set(summary_df.loc[summary_df['significant'], 'cluster_id'].astype(int))
    summary_df[
        [
            'cluster_id',
            'cluster_size',
            'sample',
            'background',
            'enrichment_pvalue_zbinom',
            'enrichment_fdr_zbinom',
            'log_fold_change',
            'significant',
        ]
    ].to_csv(tcrempnet_dir / f"{prefix}_summary_tcrempnet.tsv", sep='\t', index=False)

    sample_cluster_df = cluster_df[
        (cluster_df['clone_id'].isin(set(sample_ids))) &
        (cluster_df['cluster_id'].isin(significant_cluster_ids))
    ].copy()
    sample_cluster_df.to_csv(tcrempnet_dir / f"{prefix}_clustered_sample_clonotypes.tsv", sep='\t', index=False)

    cluster_members_df = build_sample_members_table(sample_cluster_df, summary_df, chain, ep_df, epitope)
    cluster_members_df.to_csv(tcrempnet_dir / f"{prefix}_cluster_members.tsv", sep='\t', index=False)

    sample_labels = np.asarray(labels[:len(sample_ids)], dtype=np.int32)
    save_cluster_plot_html(
        epitope=epitope,
        chain=chain,
        sample_reps=sample_reps,
        sample_ids=sample_ids,
        sample_pca=sample_pca,
        sample_labels=sample_labels,
        significant_cluster_ids=significant_cluster_ids,
        bg_umap=bg_umap,
        transform=transform,
        output_path=tcrempnet_dir / f"{prefix}_clusters.html",
    )

    logging.info('Done %s: sample=%d, clustered_sample=%d, clusters=%d',
                 epitope, len(sample_reps), len(sample_cluster_df), summary_df.shape[0])

    del sample_emb, sample_reps, sample_ids, sample_pca, distances, indices, labels, joint_ids, joint_reps, cluster_df
    gc.collect()
    return sample_cluster_df, cluster_members_df


def main():
    args = get_arguments_vdjdb_clusters()

    output_root = prepare_output_path(args.output)
    airr_dir = output_root / 'airr_format'
    tcremp_dir = output_root / 'tcremp'
    tcrempnet_dir = output_root / 'tcrempnet'
    for p in (airr_dir, tcremp_dir, tcrempnet_dir):
        p.mkdir(parents=True, exist_ok=True)

    configure_logging(Path(args.vdjdb), output_root, f"{args.chain.lower()}_vdjdb_clusters")
    faiss.omp_set_num_threads(args.nproc)

    genes = [args.chain]
    locus = {'TRA': 'alpha', 'TRB': 'beta'}[args.chain]
    lib = SegmentLibrary.load_default(genes=genes, organisms=args.species)

    vdjdb_df = pd.read_csv(args.vdjdb, sep='\t')
    if args.epitopes is not None:
        vdjdb_df = vdjdb_df[vdjdb_df['antigen.epitope'].isin(args.epitopes)].copy()
    if args.min_epitope_clonotypes is not None:
        epitope_sizes = vdjdb_df.groupby('antigen.epitope').size()
        eligible_epitopes = epitope_sizes[epitope_sizes >= args.min_epitope_clonotypes].index
        skipped_epitopes = int((epitope_sizes < args.min_epitope_clonotypes).sum())
        vdjdb_df = vdjdb_df[vdjdb_df['antigen.epitope'].isin(eligible_epitopes)].copy()
        logging.info(
            'Filtered epitopes by minimum clonotype count >= %d: kept %d epitopes, skipped %d',
            args.min_epitope_clonotypes,
            len(eligible_epitopes),
            skipped_epitopes,
        )

    args.background = str(Path(args.background_airr).resolve())
    args.output = str(output_root)
    args.background_embedding = str(Path(args.background_embedding).resolve())

    proto_path =  resolve_prototype_file(args.prototypes_path)
    
    logging.info('Loading prototypes')
    proto = load_prototype_repertoire(proto_path, lib, locus, args.index_col)
    proto = subsample_repertoire(proto, args.n_prototypes, args.sample_random_clonotypes, args.random_seed)
    logging.info(f'Loaded {len(proto)} prototypes')
    
    logging.info('Loading background embeddings')
    bg_emb, bg_reps, bg_ids, _, bg_index_path = load_embeddings(
        path=args.background,
        args=args,
        is_sample=False,
        lib=lib,
        locus=locus,
        prefix=f"{args.chain.lower()}_background",
        output_path=tcremp_dir,
    )

    transform = load_or_fit_background_transform(args=args, output_root=output_root, bg_emb=bg_emb)
    bg_pca = transform.background_pca_
    if bg_pca is None:
        bg_pca = transform.transform_pca(bg_emb)
    plot_bg_points = min(len(bg_pca), args.n_bg_points or DEFAULT_PLOT_BG_POINTS)
    bg_umap = prepare_background_umap(transform, bg_pca, plot_bg_points)

    clustered_tables = []
    cluster_members_tables = []
    for epitope, ep_df in vdjdb_df.groupby('antigen.epitope', sort=True):
        clustered_df, cluster_members_df = process_epitope(
            epitope, ep_df, args=args, genes=genes, locus=locus, lib=lib, proto=proto,
            airr_dir=airr_dir, tcremp_dir=tcremp_dir, tcrempnet_dir=tcrempnet_dir,
            bg_pca=bg_pca, bg_reps=bg_reps, bg_ids=bg_ids, bg_index_path=bg_index_path, bg_umap=bg_umap,
            transform=transform,
        )
        clustered_tables.append(clustered_df)
        cluster_members_tables.append(cluster_members_df)

    chain_lower = args.chain.lower()
    if clustered_tables:
        pd.concat(clustered_tables, ignore_index=True).to_csv(
            tcrempnet_dir / f"{chain_lower}_vdjdb_clustered_clonotypes.tsv", sep='\t', index=False
        )
    if cluster_members_tables:
        pd.concat(cluster_members_tables, ignore_index=True).to_csv(
            tcrempnet_dir / f"{chain_lower}_vdjdb_cluster_members.txt", sep='\t', index=False
        )

    logging.info('Done')


if __name__ == '__main__':
    main()

import sys

sys.path.append("../")
sys.path.append("../../mirpy")

from tcremp.arguments import get_arguments
from tcremp.utils import configure_logging, load_prototype_repertoire, load_analysis_repertoire, \
    get_representations_df, resolve_prototype_file, \
    resolve_input_file, prepare_output_path, generate_output_prefix, subsample_repertoire, log_memory_usage
from mir.common.segments import SegmentLibrary
# from tcremp.tcremp_cluster import run_dbscan_clustering, prepare_data_for_clustering, get_k_neighbors_distance_matrix, \
#     estimate_dbscan_eps

import logging
import pandas as pd
import time
from mir.embedding.prototype_embedding import PrototypeEmbedding, Metrics
from mir.distances.aligner import ClonotypeAligner


def _safe_embedding_threads(nproc, n_clonotypes, min_batch_size=32):
    if n_clonotypes <= 0:
        return 0
    if nproc is None:
        nproc = 1
    # `mir` multiprocessing may generate empty batches on small inputs.
    # Keep workers conservative so each process gets a non-trivial chunk.
    by_batch = max(1, n_clonotypes // max(1, int(min_batch_size)))
    return max(1, min(int(nproc), int(n_clonotypes), by_batch))


def run_tcremp_embedding(analysis_rep, proto_rep, segment_library, chain, metrics, nproc, filename, save_dists=True):
    aligner = ClonotypeAligner.from_library(lib=segment_library)
    logging.info(f'Started embeddings calculation')
    embedder = PrototypeEmbedding(proto_rep, aligner=aligner, metrics=Metrics(metrics))
    n_clonotypes = len(analysis_rep.clonotypes)
    if n_clonotypes == 0:
        raise ValueError(
            'Cannot compute embeddings for an empty repertoire. '
            'All clonotypes were filtered out before embedding.'
        )
    threads = _safe_embedding_threads(nproc, n_clonotypes)
    logging.info(
        'Embedding repertoire with %d clonotypes using %d worker(s) (requested nproc=%s)',
        n_clonotypes,
        threads,
        nproc,
    )
    t0 = time.time()
    emb = embedder.embed_repertoire(analysis_rep, threads=threads, flatten_scores=True)
    logging.info(f'Embeddings done in {time.time() - t0:.2f}s')
    log_memory_usage('after embeddings done')
    columns = []
    for i in range(proto_rep.total):
        if 'TRA' in chain:
            columns += [f'{i}_a_v', f'{i}_a_j', f'{i}_a_cdr3']
        if 'TRB' in chain:
            columns += [f'{i}_b_v', f'{i}_b_j', f'{i}_b_cdr3']
    log_memory_usage('after col creation')
    df = pd.DataFrame(emb, columns=columns)
    log_memory_usage('after df creation')
    if save_dists:
        df.to_parquet(filename, index=False)
    return df


def main():
    args = get_arguments()

    input_path = resolve_input_file(args.input)
    proto_path = resolve_prototype_file(args.prototypes_path)
    output_path = prepare_output_path(args.output)
    prefix = generate_output_prefix(args.input, args.prefix)

    configure_logging(input_path, output_path, prefix)
    log_memory_usage('init')

    chain = args.chain.split('_')
    locus = {'TRA': 'alpha', 'TRB': 'beta', 'TRA_TRB': None}[args.chain]
    lib = SegmentLibrary.load_default(genes=chain, organisms=args.species)

    log_memory_usage('before load')
    logging.info('Started loading clonotypes for analysis into MIR object.')
    rep = load_analysis_repertoire(input_path, lib, locus, args.index_col, args.lower_len_cdr3, args.higher_len_cdr3)
    logging.info(f'Analysis repertoire: {rep}')
    logging.info('Started loading prototypes into MIR object.')
    proto = load_prototype_repertoire(proto_path, lib, locus, args.index_col)
    logging.info(f'Proto repertoire: {rep}')

    rep = subsample_repertoire(rep, args.n_clonotypes, args.sample_random_clonotypes, args.random_seed)
    proto = subsample_repertoire(proto, args.n_prototypes, args.sample_random_clonotypes, args.random_seed)
    rep.serialize().to_csv(f'{output_path}/{prefix}_rep.tsv', sep='\t')
    logging.info(f'Finished subsampling')
    log_memory_usage('before tcremp')

    reps = get_representations_df(rep, locus)
    reps.to_csv(f'{output_path}/{prefix}_tcremp_representations.tsv', sep='\t', index=False)
    reps = None

    emb = run_tcremp_embedding(rep, proto, lib, chain, args.metrics, args.nproc,
                               f'{output_path}/{prefix}_embeddings.parquet')

    reps = pd.read_csv(f'{output_path}/{prefix}_tcremp_representations.tsv', sep='\t')
    ids = reps.clone_id
    log_memory_usage('after reps reading')
    # if args.cluster:
    #     df = prepare_data_for_clustering(emb, n_components=args.cluster_pc_components)
    #     distances = get_k_neighbors_distance_matrix(df, n_neighbors=args.k_neighbors)
    #     eps = estimate_dbscan_eps(df, distances=distances[:, args.k_neighbors - 1])
    #     clust = run_dbscan_clustering(df,
    #                                   eps=eps,
    #                                   closest_neigh_dist_array=distances[:, 1],
    #                                   min_samples=args.cluster_min_samples)
    #     pd.DataFrame({'clone_id': ids, 'cluster_id': clust}).merge(reps).to_csv(
    #         f"{output_path}/{prefix}_tcremp_clusters.tsv", sep='\t', index=False)

    if args.save_dists:
        emb['clone_id'] = ids
        emb = emb[['clone_id'] + [c for c in emb.columns if c != 'clone_id']]
        emb = reps.merge(emb)
        emb.to_csv(f"{output_path}/{prefix}_tcremp.tsv", sep='\t', index=False)


if __name__ == "__main__":
    main()

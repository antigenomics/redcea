import argparse


def add_common_tcremp_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path to the output folder.')
    parser.add_argument('-e', '--prefix', type=str,
                        help='Output prefix. Defaults to input clonotype table filename.')
    parser.add_argument('-x', '--index-col', type=str,
                        help='(optional) Name of a column in the input table containing user-specified IDs that will '
                             'be transfered to output tables.')
    parser.add_argument('-c', '--chain', type=str, required=True,
                        choices=['TRA', 'TRB', 'TRA_TRB'],
                        help='"TRA" or "TRB" for single-chain input data (clonotypes), for paired-chain input ('
                             'clones) use "TRA_TRB". Used in default prototype set selection.')
    parser.add_argument('-p', '--prototypes-path', type=str,
                        help='Path to user-specified prototypes file. If not set, will use pre-built prototype tables, '
                             '"$tcremp_path/data/data_prebuilt".')
    parser.add_argument('-n', '--n-prototypes', type=int,
                        help='Number of prototypes to select for clonotype triangulation during embedding.')
    parser.add_argument('-sample_random_p', '--sample-random-prototypes', type=bool, default=False,
                        help='Whether to sample the prototypes randomly or not. Defaults to False.')
    parser.add_argument('-nc', '--n-clonotypes', type=int,
                        help='Number of clonotypes to process in the pipeline. Will use all available clonotypes if not set.')
    parser.add_argument('-sample_random_c', '--sample-random-clonotypes', type=bool, default=False,
                        help='Whether to sample the clonotypes randomly or not. Defaults to False.')
    parser.add_argument('-s', '--species', type=str, default='HomoSapiens',
                        choices=['HomoSapiens', 'MusMusculus', 'MacacaMulatta'],
                        help='V/J gene aligner species specification. Defaults to HomoSapiens.')
    parser.add_argument('-u', '--unique-clonotypes',
                        help='Speed-up the analysis by running for unique clonotypes (clones) in the input table')
    parser.add_argument('-r', '--random-seed', type=int, default=42,
                        help='Random seed for prototype sampling and other rng-based procedures. Defaults to 42.')
    parser.add_argument('-np', '--nproc', type=int, default=1,
                        help='Number of processes to perform calculation with. Will use 1 process by default.')
    parser.add_argument('-llen', '--lower-len-cdr3', type=int, default=5,
                        help='Filter out cdr3 with len <llen. Defaults to 5.')
    parser.add_argument('-hlen', '--higher-len-cdr3', type=int, default=30,
                        help='Filter out cdr3 with len >=hlen. Defaults to 30.')
    parser.add_argument('-m', '--metrics', type=str, default='dissimilarity',
                        choices=['similarity', 'dissimilarity'],
                        help='Whether to calculate similarity or dissimilarity scores with TCRemP.')
    parser.add_argument('-d', '--save-dists', type=bool, default=True,
                        help='Whether to save the file with evaluated TCRemP distances or not. Defaults to True.')
    parser.add_argument('-cl', '--cluster', type=bool, default=True,
                        help='Whether to perform the clustering or not. Defaults to True.')
    parser.add_argument('-npc', '--cluster-pc-components', type=int, default=50,
                        help='Number of PCA components for distances dimension reduction (default: 50)')
    parser.add_argument('-ms', '--cluster-min-samples', type=int, default=3,
                        help='min_samples parameter for DBSCAN used in clonotype clustering (default: 3)')
    parser.add_argument('-kn', '--k-neighbors', type=int, default=4,
                        help='k neighbors for KNN graph evaluation (default: 4)')
    parser.add_argument('-ekn', '--eps-k-neighbors', type=int, default=4,
                        help='k-th neighbor parameter for Knee estimation (default: 4)')
    parser.add_argument('-se', '--sample-embedding', type=str,
                        help='Optional path to sample embedding file (parquet). If not set, will be computed or default name used.')
    parser.add_argument('-be', '--background-embedding', type=str,
                        help='Optional path to background embedding file (parquet). If not set, will be computed or default name used.')
    parser.add_argument('--cluster-algo', choices=['leiden_dbscan', 'hierarchical_leiden', 'leiden', 'vdbscan'], default='vdbscan',
                        help='Clustering algorithm to use.')
    parser.add_argument('--n-bg-points', type=int, default=None,
                        help='If set, only the first N clonotypes from background repertoire will be used for embedding + clustering.')
    parser.add_argument('--leiden-resolution', type=float, default=1.0,
                        help='Resolution parameter for Leiden clustering (default: 1.0)')
    parser.add_argument('--leiden-sub-resolution', type=float, default=1.0,
                        help='Resolution parameter for Leiden clustering (default: 1.0)')
    parser.add_argument('--eps-estimation-based-on', type=str, default='sample',
                        choices=['sample', 'background', 'all'],
                        help='Basis for eps estimation in vDBSCAN.')
    parser.add_argument('--vdbscan-sym-rule', type=str, default='asymmetric',
                        choices=['asymmetric', 'min', 'max'],
                        help='Symmetrization rule for vDBSCAN.')
    return parser


def add_single_input_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input file containing a clonotype (clone) table.')
    return parser


def add_enrich_io_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('-is', '--sample', type=str, required=True,
                        help='Path to input file containing a clonotype (clone) table of sample repertoire.')
    parser.add_argument('-ib', '--background', type=str, required=True,
                        help='Path to input file containing a clonotype (clone) table of background repertoire.')
    return parser


def add_vdjdb_cluster_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--vdjdb', required=True,
                        help='Path to vdjdb_full.txt-like file.')
    parser.add_argument('--background-airr', required=True,
                        help='Path to background AIRR file.')
    parser.add_argument('--epitopes', nargs='*', default=None,
                        help='Optional list of epitopes to process. By default all epitopes are processed.')
    return parser


def build_general_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='General TCRemP pipeline implementation')
    add_single_input_args(parser)
    add_common_tcremp_args(parser)
    return parser


def build_enrich_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='TCRempNet pipeline implementation')
    add_enrich_io_args(parser)
    add_common_tcremp_args(parser)
    return parser


def build_vdjdb_cluster_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='VDJdb wrapper for per-epitope Leiden clustering')
    add_vdjdb_cluster_args(parser)
    add_common_tcremp_args(parser)
    parser.set_defaults(
        cluster_algo='leiden',
        index_col=None,
        background=None,
        sample=None,
        n_prototypes=None,
        sample_random_prototypes=False,
        n_clonotypes=None,
        sample_random_clonotypes=False,
        unique_clonotypes=None,
        n_bg_points=None,
        leiden_sub_resolution=1.0,
        eps_estimation_based_on='sample',
        vdbscan_sym_rule='asymmetric',
    )
    return parser


def get_arguments(args=None):
    return build_general_parser().parse_args(args)


def get_arguments_enrich(args=None):
    return build_enrich_parser().parse_args(args)


def get_arguments_vdjdb_clusters(args=None):
    return build_vdjdb_cluster_parser().parse_args(args)

import argparse

from tcremp.arguments import add_common_embedding_args


def add_enrich_io_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-is",
        "--sample",
        type=str,
        required=True,
        help="Path to input file containing a clonotype (clone) table of sample repertoire.",
    )
    parser.add_argument(
        "-ib",
        "--background",
        type=str,
        required=True,
        help="Path to input file containing a clonotype (clone) table of background repertoire.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output folder.",
    )
    parser.add_argument(
        "-e",
        "--prefix",
        type=str,
        help="Output prefix. Defaults to input clonotype table filename.",
    )
    parser.add_argument(
        "-x",
        "--index-col",
        type=str,
        help="Name of a column in the input table containing user-specified IDs that will be transferred to outputs.",
    )
    return parser


def add_redcea_pipeline_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-se",
        "--sample-embedding",
        type=str,
        help="Optional path to sample embedding file (parquet). If not set, will be computed or default name used.",
    )
    parser.add_argument(
        "-be",
        "--background-embedding",
        type=str,
        help="Optional path to background embedding file (parquet). If not set, will be computed or default name used.",
    )
    parser.add_argument(
        "--cluster-algo",
        choices=["dbscan", "leiden_dbscan", "hierarchical_leiden", "leiden", "vdbscan"],
        default="vdbscan",
        help="Clustering algorithm to use.",
    )
    parser.add_argument(
        "--n-bg-points",
        type=int,
        default=None,
        help="If set, only the first N clonotypes from background repertoire will be used for embedding + clustering.",
    )
    parser.add_argument(
        "-npc",
        "--cluster-pc-components",
        type=int,
        default=50,
        help="Number of PCA components for distances dimension reduction.",
    )
    parser.add_argument(
        "-ms",
        "--cluster-min-samples",
        type=int,
        default=3,
        help="min_samples parameter for clustering core points.",
    )
    parser.add_argument(
        "-kn",
        "--k-neighbors",
        type=int,
        default=4,
        help="k neighbors for KNN graph evaluation.",
    )
    parser.add_argument(
        "-ekn",
        "--eps-k-neighbors",
        type=int,
        default=4,
        help="k-th neighbor parameter for knee estimation.",
    )
    parser.add_argument(
        "--leiden-resolution",
        type=float,
        default=1.0,
        help="Resolution parameter for Leiden clustering.",
    )
    parser.add_argument(
        "--leiden-sub-resolution",
        type=float,
        default=1.0,
        help="Resolution parameter for hierarchical Leiden subclustering.",
    )
    parser.add_argument(
        "--eps-estimation-based-on",
        type=str,
        default="sample",
        choices=["sample", "background", "all"],
        help="Basis for eps estimation in vDBSCAN.",
    )
    parser.add_argument(
        "--vdbscan-sym-rule",
        type=str,
        default="asymmetric",
        choices=["asymmetric", "min", "max"],
        help="Symmetrization rule for vDBSCAN.",
    )
    parser.add_argument(
        "--enrichment-test",
        type=str,
        default="zbinom",
        choices=["zbinom", "binom", "fisher"],
        help="Statistical test used for cluster enrichment.",
    )
    parser.add_argument(
        "--debug-save-intermediate",
        action="store_true",
        help="Save optional debug artifacts with intermediate arrays, tables, and summaries.",
    )
    parser.add_argument(
        "--debug-output-dir",
        type=str,
        default=None,
        help="Optional custom directory for debug artifacts. Defaults to <output>/debug.",
    )
    return parser

def build_enrich_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RedCEA sample-vs-background clustering pipeline")
    add_enrich_io_args(parser)
    add_common_embedding_args(parser)
    add_redcea_pipeline_args(parser)
    return parser


def get_arguments_enrich(args=None):
    return build_enrich_parser().parse_args(args)

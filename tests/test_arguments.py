from redcea.arguments import build_enrich_parser


def test_build_enrich_parser_parses_required_arguments():
    parser = build_enrich_parser()

    args = parser.parse_args(
        [
            "--sample",
            "sample.tsv",
            "--background",
            "background.tsv",
            "--output",
            "out",
            "--chain",
            "TRB",
        ]
    )

    assert args.sample == "sample.tsv"
    assert args.background == "background.tsv"
    assert args.output == "out"
    assert args.chain == "TRB"
    assert args.cluster_algo == "vdbscan"
    assert args.core_min_samples == 3
    assert args.k_neighbors == 4
    assert args.enrichment_test == "zbinom"
    assert args.use_clonotype_counts is False
    assert args.add_auxiliary_cluster_metrics is False


def test_build_enrich_parser_accepts_both_core_min_samples_flags():
    parser = build_enrich_parser()

def test_build_enrich_parser_accepts_both_core_min_samples_flags():
    parser = build_enrich_parser()

    args_new = parser.parse_args(
        [
            "--sample",
            "sample.tsv",
            "--background",
            "background.tsv",
            "--output",
            "out",
            "--chain",
            "TRB",
            "--core-min-samples",
            "7",
        ]
    )
    args_old = parser.parse_args(
        [
            "--sample",
            "sample.tsv",
            "--background",
            "background.tsv",
            "--output",
            "out",
            "--chain",
            "TRB",
            "--cluster-min-samples",
            "5",
        ]
    )

    assert args_new.core_min_samples == 7
    assert args_old.core_min_samples == 5


def test_build_enrich_parser_accepts_count_aware_enrichment_flags():
    parser = build_enrich_parser()

    args = parser.parse_args(
        [
            "--sample",
            "sample.tsv",
            "--background",
            "background.tsv",
            "--output",
            "out",
            "--chain",
            "TRB",
            "--use-clonotype-counts",
        ]
    )

    assert args.use_clonotype_counts is True
    assert args.enrichment_test == "zbinom"

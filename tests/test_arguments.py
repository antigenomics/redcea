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
    assert args.k_neighbors == 4
    assert args.enrichment_test == "zbinom"
    assert args.use_clonotype_counts is False


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

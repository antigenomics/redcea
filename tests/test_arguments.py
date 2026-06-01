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
    assert args.add_auxiliary_cluster_metrics is False

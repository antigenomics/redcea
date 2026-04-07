from redcea.arguments import build_enrich_parser, build_vdjdb_cluster_parser


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


def test_build_vdjdb_cluster_parser_sets_expected_defaults():
    parser = build_vdjdb_cluster_parser()

    args = parser.parse_args(
        [
            "--vdjdb",
            "vdjdb.tsv",
            "--background-airr",
            "background.tsv",
            "--output",
            "out",
            "--chain",
            "TRA",
        ]
    )

    assert args.vdjdb == "vdjdb.tsv"
    assert args.background_airr == "background.tsv"
    assert args.output == "out"
    assert args.chain == "TRA"
    assert args.cluster_algo == "leiden"
    assert args.sample is None
    assert args.background is None
    assert args.sample_random_prototypes is False

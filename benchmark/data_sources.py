from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SIBLING_PROJECTS_ROOT = REPO_ROOT.parent
DEFAULT_VDJDB_MOTIFS_DIR = SIBLING_PROJECTS_ROOT / "vdjdb-motifs"
DEFAULT_VDJDB_MOTIFS_RESULTS_DIR = DEFAULT_VDJDB_MOTIFS_DIR / "results" / "redcea"
DEFAULT_YFV_RUNS_DIR = Path("/projects/immunestatus/pogorelyy/tcremp")
DEFAULT_VDJDB_EMBED_DIR = DEFAULT_VDJDB_MOTIFS_RESULTS_DIR / "tcremp"
DEFAULT_VDJDB_AIRR_DIR = DEFAULT_VDJDB_MOTIFS_RESULTS_DIR / "airr_format"
DEFAULT_TCRVDB_PATH = Path.home() / "01_05_2025_TCRvdb.csv"
DEFAULT_VDJDB_RELEASE_PATH = DEFAULT_VDJDB_MOTIFS_DIR / "vdjdb_release" / "vdjdb.slim.txt"
DEFAULT_VDJDB_FULL_PATH = DEFAULT_VDJDB_MOTIFS_DIR / "redcea" / "data" / "vdjdb_full.txt"
DEFAULT_VDJDB_BG_SOURCE_AIRR = DEFAULT_VDJDB_MOTIFS_DIR / "redcea" / "data" / "backgrounds" / "trb_background_100k.tsv"
DEFAULT_VDJDB_BG_SOURCE_EMBEDDING = DEFAULT_VDJDB_MOTIFS_DIR / "redcea" / "data" / "backgrounds" / "trb_background_embeddings.parquet"
DEFAULT_TCRVDB_PADJ_THRESHOLD = 1e-5
DEFAULT_YFV_KNOWN_EPITOPES = ("ATDALMTGF", "LLWNGPMAV")

VDJDB_TARGETS = {
    "GLC": {
        "epitope_sequence": "GLCTLVAML",
        "embedding_filename": "trb_vdjdb_GLCTLVAML_sample_embeddings.parquet",
        "index_filename": "trb_vdjdb_GLCTLVAML_sample_embeddings.index",
        "representation_filename": "trb_vdjdb_GLCTLVAML.tsv",
    },
    "YLQ": {
        "epitope_sequence": "YLQPRTFLL",
        "embedding_filename": "trb_vdjdb_YLQPRTFLL_sample_embeddings.parquet",
        "index_filename": "trb_vdjdb_YLQPRTFLL_sample_embeddings.index",
        "representation_filename": "trb_vdjdb_YLQPRTFLL.tsv",
    },
    "GILGFVFTL": {
        "epitope_sequence": "GILGFVFTL",
        "embedding_filename": "trb_vdjdb_GILGFVFTL_sample_embeddings.parquet",
        "index_filename": "trb_vdjdb_GILGFVFTL_sample_embeddings.index",
        "representation_filename": "trb_vdjdb_GILGFVFTL.tsv",
    },
    "NLVPMVATV": {
        "epitope_sequence": "NLVPMVATV",
        "embedding_filename": "trb_vdjdb_NLVPMVATV_sample_embeddings.parquet",
        "index_filename": "trb_vdjdb_NLVPMVATV_sample_embeddings.index",
        "representation_filename": "trb_vdjdb_NLVPMVATV.tsv",
    },
    "AVFDRKSDAK": {
        "epitope_sequence": "AVFDRKSDAK",
        "embedding_filename": "trb_vdjdb_AVFDRKSDAK_sample_embeddings.parquet",
        "index_filename": "trb_vdjdb_AVFDRKSDAK_sample_embeddings.index",
        "representation_filename": "trb_vdjdb_AVFDRKSDAK.tsv",
    },
    "ELAGIGILTV": {
        "epitope_sequence": "ELAGIGILTV",
        "embedding_filename": "trb_vdjdb_ELAGIGILTV_sample_embeddings.parquet",
        "index_filename": "trb_vdjdb_ELAGIGILTV_sample_embeddings.index",
        "representation_filename": "trb_vdjdb_ELAGIGILTV.tsv",
    },
    "RAKFKQLL": {
        "epitope_sequence": "RAKFKQLL",
        "embedding_filename": "trb_vdjdb_RAKFKQLL_sample_embeddings.parquet",
        "index_filename": "trb_vdjdb_RAKFKQLL_sample_embeddings.index",
        "representation_filename": "trb_vdjdb_RAKFKQLL.tsv",
    },
}

DEFAULT_VDJDB_BENCHMARK_TARGETS = ("GLC", "YLQ")


def resolve_vdjdb_target_keys(target_tokens=None):
    if target_tokens is None:
        return list(DEFAULT_VDJDB_BENCHMARK_TARGETS)
    normalized_tokens = [str(token).strip() for token in target_tokens if str(token).strip()]
    if not normalized_tokens:
        return list(DEFAULT_VDJDB_BENCHMARK_TARGETS)
    epitope_to_key = {
        str(target["epitope_sequence"]).upper(): key for key, target in VDJDB_TARGETS.items()
    }
    resolved_keys = []
    seen_keys = set()
    unknown_tokens = []
    for token in normalized_tokens:
        token_upper = token.upper()
        key = None
        if token_upper in VDJDB_TARGETS:
            key = token_upper
        elif token_upper in epitope_to_key:
            key = epitope_to_key[token_upper]
        if key is None:
            unknown_tokens.append(token)
            continue
        if key not in seen_keys:
            seen_keys.add(key)
            resolved_keys.append(key)
    if unknown_tokens:
        known_tokens = sorted(
            set(list(VDJDB_TARGETS.keys()) + [target["epitope_sequence"] for target in VDJDB_TARGETS.values()])
        )
        raise KeyError(
            "Unknown VDJdb target token(s): {0}. Known keys/sequences: {1}".format(
                ",".join(unknown_tokens),
                ",".join(known_tokens),
            )
        )
    return resolved_keys

def _split_yfv_donor_id(donor_id):
    subject, replicate = donor_id.split("_", 1)
    return subject, replicate


def yfv_sample_embedding_name(donor_id):
    subject, replicate = _split_yfv_donor_id(donor_id)
    return "{0}_15_{1}_tcremp.parquet".format(subject, replicate)


def yfv_background_embedding_name(donor_id):
    subject, replicate = _split_yfv_donor_id(donor_id)
    return "{0}_0_{1}_with_1_tcremp.parquet".format(subject, replicate)


def yfv_sample_representation_name(donor_id):
    subject, replicate = _split_yfv_donor_id(donor_id)
    return "{0}_15_{1}_tcremp_representations.tsv".format(subject, replicate)


def yfv_background_representation_name(donor_id):
    subject, replicate = _split_yfv_donor_id(donor_id)
    return "{0}_0_{1}_with_1_tcremp_representations.tsv".format(subject, replicate)


def resolve_yfv_embedding_paths(donor_id, runs_dir=DEFAULT_YFV_RUNS_DIR):
    runs_dir = Path(runs_dir)
    sample_embedding = runs_dir / yfv_sample_embedding_name(donor_id)
    background_embedding = runs_dir / yfv_background_embedding_name(donor_id)
    return {
        "run_dir": runs_dir,
        "sample_embedding": sample_embedding,
        "background_embedding": background_embedding,
        "sample_index": sample_embedding.with_suffix(".index"),
        "background_index": background_embedding.with_suffix(".index"),
        "sample_representation": runs_dir / yfv_sample_representation_name(donor_id),
        "background_representation": runs_dir / yfv_background_representation_name(donor_id),
    }


def discover_yfv_donor_ids(runs_dir=DEFAULT_YFV_RUNS_DIR):
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    flat_samples = set()
    flat_backgrounds = set()
    sample_pattern = re.compile(r"(?P<subject>[A-Z][0-9]+)_15_(?P<replicate>F[0-9]+)_tcremp\.parquet$")
    background_pattern = re.compile(r"(?P<subject>[A-Z][0-9]+)_0_(?P<replicate>F[0-9]+)_with_1_tcremp\.parquet$")
    for path in sorted(runs_dir.glob("*.parquet")):
        sample_match = sample_pattern.fullmatch(path.name)
        if sample_match is not None:
            donor_id = "{0}_{1}".format(sample_match.group("subject"), sample_match.group("replicate"))
            flat_samples.add(donor_id)
            continue
        background_match = background_pattern.fullmatch(path.name)
        if background_match is not None:
            donor_id = "{0}_{1}".format(background_match.group("subject"), background_match.group("replicate"))
            flat_backgrounds.add(donor_id)
    return sorted(flat_samples & flat_backgrounds)


def resolve_vdjdb_embedding_path(target_key, embed_dir=DEFAULT_VDJDB_EMBED_DIR):
    target = VDJDB_TARGETS[target_key]
    embed_dir = Path(embed_dir)
    return {
        "target_key": target_key,
        "epitope_sequence": target["epitope_sequence"],
        "sample_embedding": embed_dir / target["embedding_filename"],
        "sample_index": embed_dir / target["index_filename"],
    }

def resolve_vdjdb_rep_path(target_key, airr_dir=DEFAULT_VDJDB_AIRR_DIR):
    target = VDJDB_TARGETS[target_key]
    airr_dir = Path(airr_dir)
    return airr_dir / target["representation_filename"]


def resolve_tcrvdb_path(path=DEFAULT_TCRVDB_PATH):
    resolved = Path(path).expanduser()
    if resolved.exists():
        return resolved
    repo_local = REPO_ROOT / "data" / resolved.name
    if repo_local.exists():
        return repo_local
    return resolved

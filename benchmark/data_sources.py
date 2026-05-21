from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SIBLING_PROJECTS_ROOT = REPO_ROOT.parent
DEFAULT_VDJDB_MOTIFS_DIR = SIBLING_PROJECTS_ROOT / "vdjdb-motifs"
DEFAULT_YFV_RUNS_DIR = Path("/projects/immunestatus/pogorelyy/redcea/runs")
DEFAULT_YFV_AIRR_DIR = Path("/projects/immunestatus/pogorelyy/airr_format")
DEFAULT_VDJDB_AIRR_DIR = Path("/projects/immunestatus/vdjdb_validation/airr_format")
DEFAULT_VDJDB_EMBED_DIR = Path("/projects/immunestatus/vdjdb_validation/tcremp")
DEFAULT_TCRVDB_PATH = Path.home() / "01_05_2025_TCRvdb.csv"
DEFAULT_VDJDB_RELEASE_PATH = DEFAULT_VDJDB_MOTIFS_DIR / "vdjdb_release" / "vdjdb.slim.txt"
DEFAULT_VDJDB_FULL_PATH = DEFAULT_VDJDB_MOTIFS_DIR / "redcea" / "data" / "vdjdb_full.txt"
DEFAULT_VDJDB_BG_SOURCE_AIRR = DEFAULT_VDJDB_MOTIFS_DIR / "redcea" / "data" / "backgrounds" / "trb_background_100k.tsv"
DEFAULT_VDJDB_BG_SOURCE_EMBEDDING = DEFAULT_VDJDB_MOTIFS_DIR / "redcea" / "data" / "backgrounds" / "trb_background_embeddings.parquet"
DEFAULT_VDJDB_BG_VJ_AIRR = DEFAULT_VDJDB_MOTIFS_DIR / "results" / "redcea" / "backgrounds" / "trb_background_vj.tsv"
DEFAULT_VDJDB_BG_VJ_EMBEDDING = DEFAULT_VDJDB_MOTIFS_DIR / "results" / "redcea" / "backgrounds" / "trb_background_vj_embeddings.parquet"
DEFAULT_TCRVDB_PADJ_THRESHOLD = 1e-5
DEFAULT_YFV_KNOWN_EPITOPES = ("ATDALMTGF", "LLWNGPMAV")

VDJDB_TARGETS = {
    "GLC": {
        "epitope_sequence": "GLCTLVAML",
        "embedding_filename": "trb_vdjdb_GLCTLVAML_sample_embeddings.parquet",
        "index_filename": "trb_vdjdb_GLCTLVAML_sample_embeddings_faiss.index",
    },
    "YLQ": {
        "epitope_sequence": "YLQPRTFLL",
        "embedding_filename": "trb_vdjdb_YLQPRTFLL_sample_embeddings.parquet",
        "index_filename": "trb_vdjdb_YLQPRTFLL_sample_embeddings_faiss.index",
    },
}


def yfv_run_dir_name(donor_id):
    return "yfv_{0}".format(donor_id)


def yfv_sample_embedding_name(donor_id):
    return "yfv_{0}_sample_embeddings.parquet".format(donor_id)


def yfv_background_embedding_name(donor_id):
    return "yfv_{0}_background_embeddings.parquet".format(donor_id)


def yfv_sample_index_name(donor_id):
    return "yfv_{0}_sample_embeddings.index".format(donor_id)


def yfv_background_index_name(donor_id):
    return "yfv_{0}_background_embeddings.index".format(donor_id)


def resolve_yfv_embedding_paths(donor_id, runs_dir=DEFAULT_YFV_RUNS_DIR):
    run_dir = Path(runs_dir) / yfv_run_dir_name(donor_id)
    return {
        "run_dir": run_dir,
        "sample_embedding": run_dir / yfv_sample_embedding_name(donor_id),
        "background_embedding": run_dir / yfv_background_embedding_name(donor_id),
        "sample_index": run_dir / yfv_sample_index_name(donor_id),
        "background_index": run_dir / yfv_background_index_name(donor_id),
    }


def discover_yfv_donor_ids(runs_dir=DEFAULT_YFV_RUNS_DIR):
    runs_dir = Path(runs_dir)
    donor_ids = []
    if not runs_dir.exists():
        return donor_ids
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir() or not path.name.startswith("yfv_"):
            continue
        donor_id = path.name.replace("yfv_", "", 1)
        if not re.fullmatch(r"[A-Z][0-9]+_F[0-9]+", donor_id):
            continue
        resolved = resolve_yfv_embedding_paths(donor_id, runs_dir)
        if resolved["sample_embedding"].exists() and resolved["background_embedding"].exists():
            donor_ids.append(donor_id)
    return donor_ids


def resolve_vdjdb_embedding_path(target_key, embed_dir=DEFAULT_VDJDB_EMBED_DIR):
    target = VDJDB_TARGETS[target_key]
    embed_dir = Path(embed_dir)
    return {
        "target_key": target_key,
        "epitope_sequence": target["epitope_sequence"],
        "sample_embedding": embed_dir / target["embedding_filename"],
        "sample_index": embed_dir / target["index_filename"],
    }


def resolve_vdjdb_airr_path(target_key, airr_dir=DEFAULT_VDJDB_AIRR_DIR):
    target = VDJDB_TARGETS[target_key]
    airr_dir = Path(airr_dir)
    return airr_dir / "trb_vdjdb_{0}.tsv".format(target["epitope_sequence"])


def resolve_vdjdb_background_paths(
    source_airr=DEFAULT_VDJDB_BG_SOURCE_AIRR,
    source_embedding=DEFAULT_VDJDB_BG_SOURCE_EMBEDDING,
    vj_airr=DEFAULT_VDJDB_BG_VJ_AIRR,
    vj_embedding=DEFAULT_VDJDB_BG_VJ_EMBEDDING,
):
    return {
        "source_airr": Path(source_airr),
        "source_embedding": Path(source_embedding),
        "vj_airr": Path(vj_airr),
        "vj_embedding": Path(vj_embedding),
    }


def resolve_tcrvdb_path(path=DEFAULT_TCRVDB_PATH):
    return Path(path).expanduser()

from __future__ import annotations

from pathlib import Path

from tcremp.utils import resolve_prototype_file as _resolve_tcremp_prototype_file


def resolve_prototype_file(path: str | None, chain: str | None = None) -> str:
    if path:
        return str(Path(path).resolve())
    if chain is not None:
        return _resolve_tcremp_prototype_file(path, chain)
    raise ValueError("Either 'path' or 'chain' must be provided to resolve_prototype_file().")


def resolve_embedding_file(
    custom_path: str | None,
    output_path: str | Path,
    prefix: str,
    tag: str,
    must_exist: bool = False,
) -> Path:
    path = Path(custom_path) if custom_path else Path(output_path) / f"{prefix}_{tag}_embeddings.parquet"
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Embedding file for '{tag}' not found: {path}")
    return path


def resolve_index_file(embedding_path: str | Path) -> Path:
    path = Path(embedding_path)
    return path.with_suffix(".index")

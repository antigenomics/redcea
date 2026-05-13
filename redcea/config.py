from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    sample: str
    background: str
    output: str
    prefix: str | None
    index_col: str | None
    chain: str
    species: str
    prototypes_path: str | None
    nproc: int | None
    lower_len_cdr3: int | None
    higher_len_cdr3: int | None
    metrics: str
    sample_embedding: str | None
    background_embedding: str | None
    n_bg_points: int | None
    n_clonotypes: int | None
    sample_random_clonotypes: bool
    random_seed: int
    cluster_pc_components: int
    cluster_min_samples: int
    k_neighbors: int
    eps_k_neighbors: int
    leiden_resolution: float
    leiden_sub_resolution: float
    cluster_algo: str
    eps_estimation_based_on: str
    vdbscan_sym_rule: str
    enrichment_test: str
    debug_save_intermediate: bool
    debug_output_dir: str | None

    @classmethod
    def from_args(cls, args: Any) -> "PipelineConfig":
        return cls(
            sample=args.sample,
            background=args.background,
            output=args.output,
            prefix=getattr(args, "prefix", None),
            index_col=getattr(args, "index_col", None),
            chain=args.chain,
            species=args.species,
            prototypes_path=getattr(args, "prototypes_path", None),
            nproc=getattr(args, "nproc", None),
            lower_len_cdr3=getattr(args, "lower_len_cdr3", None),
            higher_len_cdr3=getattr(args, "higher_len_cdr3", None),
            metrics=args.metrics,
            sample_embedding=getattr(args, "sample_embedding", None),
            background_embedding=getattr(args, "background_embedding", None),
            n_bg_points=getattr(args, "n_bg_points", None),
            n_clonotypes=getattr(args, "n_clonotypes", None),
            sample_random_clonotypes=getattr(args, "sample_random_clonotypes", False),
            random_seed=getattr(args, "random_seed", 0),
            cluster_pc_components=args.cluster_pc_components,
            cluster_min_samples=args.cluster_min_samples,
            k_neighbors=args.k_neighbors,
            eps_k_neighbors=args.eps_k_neighbors,
            leiden_resolution=args.leiden_resolution,
            leiden_sub_resolution=args.leiden_sub_resolution,
            cluster_algo=args.cluster_algo,
            eps_estimation_based_on=args.eps_estimation_based_on,
            vdbscan_sym_rule=args.vdbscan_sym_rule,
            enrichment_test=getattr(args, "enrichment_test", "zbinom"),
            debug_save_intermediate=getattr(args, "debug_save_intermediate", False),
            debug_output_dir=getattr(args, "debug_output_dir", None),
        )

    @property
    def normalized_nproc(self) -> int:
        if self.nproc is None:
            return max(1, min(8, os.cpu_count() or 1))
        return self.nproc

    def to_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(**self.__dict__)


def normalize_pipeline_config(config_or_args: PipelineConfig | Any) -> PipelineConfig:
    if isinstance(config_or_args, PipelineConfig):
        return config_or_args
    return PipelineConfig.from_args(config_or_args)


__all__ = [
    "PipelineConfig",
    "normalize_pipeline_config",
]

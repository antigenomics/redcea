from __future__ import annotations

from itertools import product


MIN_SAMPLES_GRID = [3, 4, 5, 8, 10]
K_GRID = [3, 4, 5, 10, 20]
LEIDEN_RESOLUTION_GRID = [0.1, 0.5, 1.0, 2.0]
HYBRID_RESOLUTION_GRID = [0.5, 1.0]


PRIORITY_METHODS = [
    "dbscan",
    "vdbscan_length",
    "leiden",
    "leiden_vdbscan",
    "vdbscan_leiden",
]

EXTENDED_METHODS = [
    "leiden_min_size",
    "leiden_dbscan",
    "dbscan_leiden",
]


def _product_dict(**kwargs):
    keys = list(kwargs.keys())
    values = [kwargs[key] for key in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def get_method_grid(method, include_extended=False):
    if method == "dbscan":
        return list(
            _product_dict(
                min_samples=MIN_SAMPLES_GRID,
                epsilon_strategy=["knee"],
                percentile=[None],
                distance_metric=["euclidean"],
            )
        )
    if method == "vdbscan_length":
        return list(
            _product_dict(
                min_samples=MIN_SAMPLES_GRID,
                epsilon_strategy=["knee"],
                percentile=[None],
                min_group_size=[100],
                distance_metric=["euclidean"],
            )
        )
    if method == "leiden":
        return list(
            _product_dict(
                k=K_GRID,
                resolution=LEIDEN_RESOLUTION_GRID,
                distance_metric=["euclidean"],
            )
        )
    if method == "leiden_min_size":
        return list(
            _product_dict(
                k=K_GRID,
                resolution=LEIDEN_RESOLUTION_GRID,
                min_cluster_size=MIN_SAMPLES_GRID,
                distance_metric=["euclidean"],
            )
        )
    if method == "leiden_vdbscan":
        return list(
            _product_dict(
                leiden_k=K_GRID,
                leiden_resolution=HYBRID_RESOLUTION_GRID,
                vdbscan_min_samples=MIN_SAMPLES_GRID,
                epsilon_strategy=["knee"],
                percentile=[None],
                min_group_size=[100],
                distance_metric=["euclidean"],
            )
        )
    if method == "vdbscan_leiden":
        return list(
            _product_dict(
                vdbscan_min_samples=MIN_SAMPLES_GRID,
                epsilon_strategy=["knee"],
                percentile=[None],
                min_group_size=[100],
                leiden_k=K_GRID,
                leiden_resolution=HYBRID_RESOLUTION_GRID,
                distance_metric=["euclidean"],
            )
        )
    if method == "leiden_dbscan":
        return list(
            _product_dict(
                leiden_k=K_GRID,
                leiden_resolution=HYBRID_RESOLUTION_GRID,
                dbscan_min_samples=MIN_SAMPLES_GRID,
                epsilon_strategy=["knee"],
                percentile=[None],
                distance_metric=["euclidean"],
            )
        )
    if method == "dbscan_leiden":
        return list(
            _product_dict(
                dbscan_min_samples=MIN_SAMPLES_GRID,
                epsilon_strategy=["knee"],
                percentile=[None],
                leiden_k=K_GRID,
                leiden_resolution=HYBRID_RESOLUTION_GRID,
                distance_metric=["euclidean"],
            )
        )
    raise KeyError("Unknown method: {0}".format(method))


def get_enabled_methods(include_extended=False):
    methods = list(PRIORITY_METHODS)
    if include_extended:
        methods.extend(EXTENDED_METHODS)
    return methods

from __future__ import annotations

from itertools import product


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
                min_samples=[3, 5, 10],
                epsilon_strategy=["knee"],
                percentile=[None],
                distance_metric=["euclidean"],
            )
        ) + list(
            _product_dict(
                min_samples=[3, 5, 10],
                epsilon_strategy=["percentile"],
                percentile=[5, 10, 20],
                distance_metric=["euclidean"],
            )
        )
    if method == "vdbscan_length":
        return list(
            _product_dict(
                min_samples=[3, 5, 10],
                epsilon_strategy=["knee"],
                percentile=[None],
                min_group_size=[100],
                distance_metric=["euclidean"],
            )
        ) + list(
            _product_dict(
                min_samples=[3, 5, 10],
                epsilon_strategy=["percentile"],
                percentile=[5, 10, 20],
                min_group_size=[100],
                distance_metric=["euclidean"],
            )
        )
    if method == "leiden":
        return list(
            _product_dict(
                k=[10, 20, 50],
                resolution=[0.1, 0.5, 1.0, 2.0],
                distance_metric=["euclidean"],
            )
        )
    if method == "leiden_min_size":
        return list(
            _product_dict(
                k=[10, 20, 50],
                resolution=[0.1, 0.5, 1.0, 2.0],
                min_cluster_size=[3, 5, 10],
                distance_metric=["euclidean"],
            )
        )
    if method == "leiden_vdbscan":
        return list(
            _product_dict(
                leiden_k=[20],
                leiden_resolution=[0.5, 1.0],
                vdbscan_min_samples=[5],
                epsilon_strategy=["knee"],
                percentile=[None],
                min_group_size=[100],
                distance_metric=["euclidean"],
            )
        )
    if method == "vdbscan_leiden":
        return list(
            _product_dict(
                vdbscan_min_samples=[5],
                epsilon_strategy=["knee"],
                percentile=[None],
                min_group_size=[100],
                leiden_k=[20],
                leiden_resolution=[0.5, 1.0],
                distance_metric=["euclidean"],
            )
        )
    if method == "leiden_dbscan":
        return list(
            _product_dict(
                leiden_k=[20],
                leiden_resolution=[0.5, 1.0],
                dbscan_min_samples=[5],
                epsilon_strategy=["knee"],
                percentile=[None],
                distance_metric=["euclidean"],
            )
        )
    if method == "dbscan_leiden":
        return list(
            _product_dict(
                dbscan_min_samples=[5],
                epsilon_strategy=["knee"],
                percentile=[None],
                leiden_k=[20],
                leiden_resolution=[0.5, 1.0],
                distance_metric=["euclidean"],
            )
        )
    raise KeyError("Unknown method: {0}".format(method))


def get_enabled_methods(include_extended=False):
    methods = list(PRIORITY_METHODS)
    if include_extended:
        methods.extend(EXTENDED_METHODS)
    return methods


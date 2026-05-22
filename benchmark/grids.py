from __future__ import annotations

from itertools import product


SMALL_GRID = {
    "dbscan": {
        "cluster_min_samples": [5],
        "k_neighbors": [8],
        "eps_k_neighbors": [8],
        "cluster_pc_components": [50],
        "enrichment_test": ["zbinom"],
    },
    "vdbscan": {
        "cluster_min_samples": [5],
        "k_neighbors": [8],
        "eps_k_neighbors": [8],
        "eps_estimation_based_on": ["sample"],
        "vdbscan_sym_rule": ["asymmetric"],
        "cluster_pc_components": [50],
        "enrichment_test": ["zbinom"],
    },
    "leiden": {
        "cluster_min_samples": [5],
        "k_neighbors": [8],
        "leiden_resolution": [1.0],
        "cluster_pc_components": [50],
        "enrichment_test": ["zbinom"],
    },
}


LARGE_GRID = {
    "dbscan": {
        "cluster_min_samples": [3, 5, 8],
        "k_neighbors": [4, 8, 15],
        "eps_k_neighbors": [4, 8, 15],
        "cluster_pc_components": [50],
        "enrichment_test": ["zbinom"],
    },
    "vdbscan": {
        "cluster_min_samples": [3, 5, 8],
        "k_neighbors": [4, 8, 15],
        "eps_k_neighbors": [4, 8, 15],
        "eps_estimation_based_on": ["sample", "background"],
        "vdbscan_sym_rule": ["asymmetric"],
        "cluster_pc_components": [50],
        "enrichment_test": ["zbinom"],
    },
    "leiden": {
        "cluster_min_samples": [3, 5, 8],
        "k_neighbors": [4, 8, 15],
        "leiden_resolution": [0.5, 1.0, 2.0],
        "cluster_pc_components": [50],
        "enrichment_test": ["zbinom"],
    },
    "leiden_dbscan": {
        "cluster_min_samples": [3, 5, 8],
        "k_neighbors": [4, 8, 15],
        "eps_k_neighbors": [4, 8, 15],
        "leiden_resolution": [0.5, 1.0, 2.0],
        "cluster_pc_components": [50],
        "enrichment_test": ["zbinom"],
    },
    "hierarchical_leiden": {
        "cluster_min_samples": [3, 5, 8],
        "k_neighbors": [4, 8, 15],
        "leiden_resolution": [0.5, 1.0, 2.0],
        "leiden_sub_resolution": [0.5, 1.0],
        "cluster_pc_components": [50],
        "enrichment_test": ["zbinom"],
    },
}


def _product_dict(param_grid: dict[str, list[object]]) -> list[dict[str, object]]:
    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    rows = []
    for combo in product(*values):
        rows.append(dict(zip(keys, combo)))
    return rows


def get_grid_spec(grid_size: str) -> dict[str, dict[str, list[object]]]:
    if grid_size == "small":
        return SMALL_GRID
    if grid_size == "large":
        return LARGE_GRID
    raise KeyError("Unknown grid_size: {0}".format(grid_size))


def get_enabled_methods(grid_size: str = "small") -> list[str]:
    return list(get_grid_spec(grid_size).keys())


def get_method_grid(method: str, grid_size: str = "small") -> list[dict[str, object]]:
    grid_spec = get_grid_spec(grid_size)
    if method not in grid_spec:
        raise KeyError("Unknown method for grid_size={0}: {1}".format(grid_size, method))
    return _product_dict(grid_spec[method])

from __future__ import annotations

from itertools import product


def _paired_neighbor_grid(
    *,
    neighbor_values: list[int],
    base_params: dict[str, object] | None = None,
    extra_grid: dict[str, list[object]] | None = None,
) -> list[dict[str, object]]:
    rows = []
    base = dict(base_params or {})
    tail_grid = extra_grid or {}
    tail_keys = list(tail_grid.keys())
    tail_values = [tail_grid[key] for key in tail_keys]
    tail_product = list(product(*tail_values)) if tail_values else [()]
    for neighbor_value in neighbor_values:
        for tail_combo in tail_product:
            row = dict(base)
            row["k_neighbors"] = neighbor_value
            row["eps_k_neighbors"] = neighbor_value
            row.update(dict(zip(tail_keys, tail_combo)))
            rows.append(row)
    return rows


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
    "vdbscan_leiden": {
        "cluster_min_samples": [5],
        "k_neighbors": [8],
        "eps_k_neighbors": [8],
        "eps_estimation_based_on": ["sample"],
        "vdbscan_sym_rule": ["asymmetric"],
        "leiden_resolution": [1.0],
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


VDBSCAN_LEIDEN_FOCUSED_GRID = {
    "vdbscan_leiden": {
        "cluster_min_samples": [3],
        "k_neighbors": [15, 20, 24, 30],
        "eps_k_neighbors": [8, 12, 15, 20],
        "eps_estimation_based_on": ["sample", "background", "all"],
        "vdbscan_sym_rule": ["asymmetric", "max"],
        "leiden_resolution": [1.5, 2.0, 2.5, 3.0],
        "cluster_pc_components": [50],
        "enrichment_test": ["zbinom"],
    },
}

YFV_VDBSCAN_LEIDEN_LOWRES_GRID = {
    "vdbscan_leiden": {
        "cluster_min_samples": [3, 5],
        "k_neighbors": [20],
        "eps_k_neighbors": [8, 12, 16],
        "eps_estimation_based_on": ["sample"],
        "vdbscan_sym_rule": ["asymmetric"],
        "leiden_resolution": [0.2, 0.5, 1.0],
        "cluster_pc_components": [50],
        "enrichment_test": ["zbinom"],
    },
}


YFV_REDCEA_BENCHMARK_GRID = {
    "vdbscan_leiden": [
        {
            "cluster_min_samples": 3,
            "k_neighbors": k_neighbors,
            "eps_k_neighbors": eps_k_neighbors,
            "eps_estimation_based_on": "sample",
            "vdbscan_sym_rule": "asymmetric",
            "leiden_resolution": leiden_resolution,
            "cluster_pc_components": 50,
            "enrichment_test": "zbinom",
        }
        for k_neighbors in [8, 12, 16]
        for eps_k_neighbors in [4, 8, 12]
        if eps_k_neighbors <= k_neighbors
        for leiden_resolution in [0.25, 0.5, 1.0]
    ]
}


LARGE_GRID = {
    "dbscan": _paired_neighbor_grid(
        neighbor_values=[4, 8, 15],
        extra_grid={
            "cluster_min_samples": [3, 5],
            "cluster_pc_components": [50],
            "enrichment_test": ["zbinom"],
        },
    ),
    "vdbscan": _paired_neighbor_grid(
        neighbor_values=[4, 8, 15],
        base_params={"vdbscan_sym_rule": "asymmetric"},
        extra_grid={
            "cluster_min_samples": [3, 5],
            "eps_estimation_based_on": ["sample", "background"],
            "cluster_pc_components": [50],
            "enrichment_test": ["zbinom"],
        },
    ),
    "vdbscan_leiden": VDBSCAN_LEIDEN_FOCUSED_GRID["vdbscan_leiden"],
    "leiden": {
        "cluster_min_samples": [3, 5],
        "k_neighbors": [4, 8, 15],
        "leiden_resolution": [0.5, 1.0, 2.0],
        "cluster_pc_components": [50],
        "enrichment_test": ["zbinom"],
    },
    "leiden_dbscan": _paired_neighbor_grid(
        neighbor_values=[4, 8, 15],
        extra_grid={
            "cluster_min_samples": [3, 5],
            "leiden_resolution": [0.5, 1.0, 2.0],
            "cluster_pc_components": [50],
            "enrichment_test": ["zbinom"],
        },
    ),
    "hierarchical_leiden": {
        "cluster_min_samples": [3, 5],
        "k_neighbors": [4, 8, 15],
        "leiden_resolution": [0.5, 1.0],
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
    if grid_size == "focused_vdbscan_leiden":
        return VDBSCAN_LEIDEN_FOCUSED_GRID
    if grid_size == "yfv_vdbscan_leiden_lowres":
        return YFV_VDBSCAN_LEIDEN_LOWRES_GRID
    if grid_size == "yfv_redcea_benchmark":
        return YFV_REDCEA_BENCHMARK_GRID
    raise KeyError("Unknown grid_size: {0}".format(grid_size))


def get_enabled_methods(grid_size: str = "small") -> list[str]:
    return list(get_grid_spec(grid_size).keys())


def get_method_grid(method: str, grid_size: str = "small") -> list[dict[str, object]]:
    grid_spec = get_grid_spec(grid_size)
    if method not in grid_spec:
        raise KeyError("Unknown method for grid_size={0}: {1}".format(grid_size, method))
    method_spec = grid_spec[method]
    if isinstance(method_spec, list):
        return [dict(row) for row in method_spec]
    return _product_dict(method_spec)

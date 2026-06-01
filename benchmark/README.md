# RedCEA Clustering Strategy Benchmark

This directory contains the focused clustering proposal benchmark for RedCEA. The benchmark is intentionally separate from the core `redcea/` package, but it uses existing RedCEA clustering infrastructure rather than reimplementing it.

## What We Are Testing

RedCEA treats clustering as a proposal step, not as the final biological object of interest.

The benchmark asks:

1. Which clustering proposal strategy best recovers known epitope-specific TCR signal in a curated low-noise setting?
2. Which clustering proposal strategy produces the most useful candidate clusters for downstream enrichment-based denoising in noisy yellow-fever repertoires?

The methods under comparison are:

- `dbscan`
- `vdbscan`
- `vdbscan_leiden`
- `leiden`
- `leiden_dbscan`
- `hierarchical_leiden`

The initial priority batch is:

- `dbscan`
- `vdbscan`
- `vdbscan_leiden`
- `leiden`

For follow-up tuning of the hybrid only, use `grid_size=focused_vdbscan_leiden`.
That manifest contains only `vdbscan_leiden` runs and avoids relaunching the other methods.

For the new low-resolution YFV-only hybrid sweep, use:

- `grid_size=yfv_vdbscan_leiden_lowres`
- `--dataset-mode-filter yfv`
- `--yfv-donor-ids P1_F1,P2_F1`

This grid keeps only `vdbscan_leiden`, varies `eps_k_neighbors`, `cluster_min_samples`, and low Leiden resolutions, and produces `18` parameter combinations, i.e. `36` YFV runs for `P1_F1` plus `P2_F1`.

## Datasets

### VDJdb / TCRvdb

The low-noise benchmark uses:

- `GLC`
- `YLQ`

All clonotypes are included during embedding and clustering. Only labeled clonotypes are used for classification metrics.

### Yellow Fever Vaccination

The high-noise benchmark uses paired donor repertoires:

- `background = pre-vaccination`
- `sample = post-vaccination`

The benchmark also checks recovery of known yellow-fever-associated clonotypes from VDJdb.

## Directory Layout

```text
benchmark/
  __init__.py
  evaluation.py
  plotting.py
  README.md
  clustering_strategy_summary.md
  launch_all_slurm.sh
  run_all.sh
  grids.py
  run_benchmark.py
  notebooks/
    01_prepare_datasets.ipynb
    02_density_by_length.ipynb
    03_run_clustering_benchmark.ipynb
    04_evaluate_vdjdb.ipynb
    05_evaluate_yfv_enrichment.ipynb
    06_evaluate_yfv_known_clonotypes.ipynb
    07_evaluate_yfv_cross_donor_overlap.ipynb
    08_summary_and_method_selection.ipynb
  slurm/
    run_01_prepare_datasets.sbatch
    run_02_density_by_length.sbatch
    run_03_build_manifests.sbatch
    run_03_array_vdjdb.sbatch
    run_03_array_yfv.sbatch
    run_03_consolidate_metadata.sbatch
    run_03_dispatch_arrays.sbatch
    run_04_evaluate_vdjdb.sbatch
    run_05_evaluate_yfv_enrichment.sbatch
    run_06_evaluate_yfv_known_clonotypes.sbatch
    run_07_evaluate_yfv_cross_donor_overlap.sbatch
    run_08_summary_and_method_selection.sbatch
    logs/
```

## How It Uses RedCEA

The benchmark code in this directory is only orchestration and evaluation logic.

Each benchmark run now calls the full `redcea` sample-vs-background pipeline directly.
This directory is responsible only for:

- validating `sample/background` inputs
- expanding a hyperparameter grid into a run manifest
- launching `redcea` once per manifest row
- converting pipeline outputs into a common evaluation table
- computing downstream benchmark metrics

## Expected Inputs

The notebooks expect a dataset manifest under `data/processed/benchmark_dataset_manifest.tsv`.

`benchmark/prepare_datasets.py` builds that manifest directly from the canonical upstream inputs:

- `TCRvdb` labels for `GLC` / `YLQ`
- `VDJdb` baseline table
- `TRB` VDJdb sample embeddings in `../vdjdb-motifs/results/redcea/tcremp`
- `TRB` VDJdb AIRR tables in `../vdjdb-motifs/results/redcea/airr_format`
- `YFV` sample/background embeddings in `/projects/immunestatus/pogorelyy/redcea/runs/yfv_*`
- `YFV` AIRR tables in `/projects/immunestatus/pogorelyy/airr_format`

For the `TRB` VDJdb background, the benchmark uses the canonical `vdjdb-motifs` source background directly:

- `redcea/data/backgrounds/trb_background_100k.tsv`
- `redcea/data/backgrounds/trb_background_embeddings.parquet`

The current benchmark uses only `TRB`.

## Running on Slurm

From the repository root, the one-command entrypoint is:

```bash
bash benchmark/run_all.sh slurm
```

For a non-Slurm local run:

```bash
bash benchmark/run_all.sh local
```

The Slurm launcher internally submits the staged notebook jobs plus the manifest-driven clustering batch runner.

The clustering stage is split into two Slurm arrays:

- `VDJdb array`: `64G` RAM per task
- `YFV array`: `256G` RAM per task

This split exists because yellow-fever donor runs are substantially heavier in memory than the curated VDJdb benchmark.

Optional environment variable:

```bash
export BENCHMARK_ENV_ACTIVATE=/path/to/activate_script.sh
```

If this variable is set, each Slurm job will source it before executing notebooks.

## Outputs

Notebook execution writes into the repository-level output folders:

- `results/clustering_assignments/`
- `results/run_metadata/`
- `results/enrichment/`
- `results/metrics/`
- `figures/clustering_strategy/`

The final interpretation template lives at:

- `benchmark/clustering_strategy_summary.md`

## Parameter Grid

The benchmark grid is defined in:

- `benchmark/grids.py`

The execution workflow is:

1. Build a grid manifest of method/parameter combinations.
2. Expand it into execution manifests for:
   - `results/run_metadata/clustering_manifest_vdjdb.tsv`
   - `results/run_metadata/clustering_manifest_yfv.tsv`
3. Submit one Slurm array over each execution manifest.
4. Consolidate per-run metadata after both arrays complete successfully.
5. Run evaluation notebooks after metadata consolidation.

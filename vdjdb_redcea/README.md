# VDJdb Per-Epitope Clustering

This directory contains the VDJdb-specific wrapper around TCRemP/TCRemPNet:
[vdjdb_clusters_launch_with_transform.py](/c:/Users/lizzka239/projects/tcrempnet/vdjdb_redcea/vdjdb_clusters_launch_with_transform.py).

The wrapper takes a `vdjdb_full.txt`-like table, splits it by `antigen.epitope`, embeds each epitope-specific repertoire, projects sample and background into the same background-fitted latent space, runs joint kNN + Leiden clustering, computes enrichment against the background repertoire, and exports both tabular results and HTML visualizations.

## Current Pipeline

For each selected epitope, the script:

1. extracts one chain only: `TRA` or `TRB`;
2. converts the epitope subset into AIRR format;
3. computes or reuses sample embeddings for that epitope;
4. loads precomputed background embeddings;
5. fits or reuses a shared `BackgroundTransform` on the background embeddings;
6. projects sample and background into the same PCA space;
7. builds a joint kNN graph and runs Leiden clustering;
8. computes per-cluster enrichment statistics against the background;
9. saves cluster tables, cluster-member tables, and an HTML plot with sample points over background density.

## Required Inputs

The wrapper expects three required inputs:

- `--vdjdb`: a TSV in `vdjdb_full.txt`-compatible format.
- `--background-airr`: an AIRR table for the background repertoire.
- `--background-embedding`: a precomputed background embedding parquet file.

For `TRA`, the VDJdb table must contain:

- `cdr3.alpha`
- `v.alpha`
- `j.alpha`

For `TRB`, it must contain:

- `cdr3.beta`
- `v.beta`
- `j.beta`

The script also relies on the standard VDJdb metadata columns used downstream in the output tables, including:

- `species`
- `antigen.epitope`
- `antigen.gene`
- `antigen.species`
- `mhc.a`
- `mhc.b`
- `mhc.class`

## Important Behavior Changes

- The wrapper now uses a persistent background-space transform via `BackgroundTransform` instead of fitting a fresh projection independently for each epitope.
- Background embeddings must already exist. This script loads them; it does not compute them for you.
- Sample embeddings are cached per epitope in `output/tcremp/` and reused on repeated runs.
- The wrapper always runs Leiden clustering in the shared sample+background graph.
- Background UMAP coordinates used for plotting are cached and reused.

## Basic Run

```bash
python vdjdb_redcea/vdjdb_clusters_launch_with_transform.py \
  --vdjdb vdjdb_redcea/vdjdb-2025-12-29/vdjdb_full.txt \
  --background-airr data/background_trb.airr.tsv \
  --background-embedding results/background_trb_embeddings.parquet \
  --output results/vdjdb_trb \
  --chain TRB \
  --species HomoSapiens \
  --nproc 8
```

### Filter by Explicit Epitope List

Use `--epitopes` when you want to run only a small, named subset:

```bash
python vdjdb_redcea/vdjdb_clusters_launch_with_transform.py \
  --vdjdb vdjdb_redcea/vdjdb-2025-12-29/vdjdb_full.txt \
  --background-airr data/background_trb.airr.tsv \
  --background-embedding results/background_trb_embeddings.parquet \
  --output results/vdjdb_trb_subset \
  --chain TRB \
  --epitopes GILGFVFTL NLVPMVATV
```

### Filter by Minimum Epitope Size

Use `--min-epitope-clonotypes` when you want to process all epitopes above a size threshold:

```bash
python vdjdb_redcea/vdjdb_clusters_launch_with_transform.py \
  --vdjdb vdjdb_redcea/vdjdb-2025-12-29/vdjdb_full.txt \
  --background-airr data/background_trb.airr.tsv \
  --background-embedding results/background_trb_embeddings.parquet \
  --output results/vdjdb_trb_min30 \
  --chain TRB \
  --min-epitope-clonotypes 30
```

You can also combine both options if you want to restrict the run to a named set of epitopes and then skip the very small ones inside that set.

## Output Layout

The script creates four subdirectories:

- `airr_format/`: per-epitope AIRR tables generated from VDJdb.
- `tcremp/`: sample embedding parquet files and the cached background transform.
- `tcrempnet/`: cluster assignments, summaries, and per-epitope cluster-member exports.
- `viz/`: HTML visualizations with sample clonotypes over a background density map.

Main output files:

- `tcremp/<chain>_background_transform.joblib`: fitted scaler/PCA and optional UMAP state for the background space.
- `tcremp/<chain>_background_transform_bg_umap_<N>.npy`: cached background UMAP coordinates used for plotting.
- `tcremp/<prefix>_sample_embeddings.parquet`: sample embeddings for one epitope.
- `tcrempnet/<prefix>_tcremp_clusters.tsv`: joint sample+background clustering table.
- `tcrempnet/<prefix>_summary_tcrempnet.tsv`: per-cluster counts and enrichment statistics.
- `tcrempnet/<prefix>_clustered_sample_clonotypes.tsv`: sample clonotypes assigned to non-noise clusters.
- `tcrempnet/<prefix>_cluster_members.tsv`: sample cluster members exported in the VDJdb-like downstream format.
- `viz/<species>_<epitope>_<chain>.html`: interactive Plotly visualization.
- `tcrempnet/<chain>_vdjdb_clustered_clonotypes.tsv`: concatenated sample-clonotype clustering table across all processed epitopes.
- `cluster_members.txt`: concatenated cluster-member table across all processed epitopes.

Here, `prefix` is built as:

```text
<chain_lower>_vdjdb_<epitope>
```

## Most Useful Arguments

### Data selection

- `--vdjdb`: input VDJdb table.
- `--background-airr`: background AIRR repertoire used as the enrichment reference.
- `--background-embedding`: precomputed background embeddings.
- `--chain {TRA,TRB}`: the wrapper works on one chain per run.
- `--epitopes`: optional list of epitopes to process.
- `--min-epitope-clonotypes`: skip very small epitopes before clustering.
- `--species`: V/J annotation species used by the segment library.

### Embedding

- `--prototypes-path`: custom prototype repertoire; built-in prototypes are used otherwise.
- `--n-prototypes`: number of prototypes to use.
- `--sample-random-prototypes`: random prototype subsampling.
- `--random-seed`: seed for reproducible random operations.
- `--lower-len-cdr3`, `--higher-len-cdr3`: CDR3 length filters.

### Clustering and plotting

- `--cluster-pc-components`: number of PCA components in the shared background-fitted space.
- `--k-neighbors`: k for the joint kNN graph.
- `--leiden-resolution`: Leiden resolution parameter.
- `--cluster-min-samples`: minimum cluster size used in post-processing of Leiden labels.
- `--n-bg-points`: truncates the background deterministically to the first `N` clonotypes for loading, clustering, and plotting.

### Compute

- `--nproc`: number of CPU threads/processes.
- `--output`: output directory.
- `--index-col`: optional user identifier to carry through parsing when applicable.

## Practical Notes

- Background truncation with `--n-bg-points` is deterministic: the script keeps the first `N` rows, not a random subset.
- The same background transform is reused for all epitopes in one run and across repeated runs if the cached file already exists.
- If an epitope-specific sample embedding parquet already exists, the script reuses it instead of recomputing embeddings.
- HTML plots show all sample clonotypes, but only enrichment-significant clusters are highlighted with distinct colors; everything else is collapsed into `unclustered` for visualization.
- A cluster is marked as significant when both conditions hold:
  - `enrichment_fdr_zbinom < 0.05`
  - `log_fold_change > 0`

## Recommended Tuning Order

If you need to adjust the run, the most useful order is:

1. fix `--chain`, `--species`, and the correct background files;
2. set `--min-epitope-clonotypes` to remove unstable tiny epitopes;
3. limit runtime with `--n-bg-points` for exploratory runs;
4. tune `--leiden-resolution` and `--k-neighbors`;
5. only then experiment with `--cluster-pc-components` or `--n-prototypes`.

## Included Snapshot

This directory currently contains the snapshot:

- [vdjdb-2025-12-29](/c:/Users/lizzka239/projects/tcrempnet/vdjdb_redcea/vdjdb-2025-12-29)

That folder includes the raw VDJdb export files and derived artifacts such as:

- `vdjdb_full.txt`
- `cluster_members.txt`
- `vdjdb_summary_embed.html`

Use those files as reference data; the wrapper itself writes its own run outputs into the directory passed via `--output`.

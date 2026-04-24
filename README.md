# RedCEA: Repertoire Embeddings Denoising Clustering Enrichment Analysis

RedCEA is a pipeline for comparing immune repertoires using prototype-based TCR embeddings. It builds on the original TCRemP embedding method and adds denoising, clustering, and enrichment analysis for case/background repertoire comparisons.

This repository contains command-line tools for:

* computing prototype-based embeddings for a repertoire with `tcremp-run`
* clustering embeddings with `tcremp-cluster`
* running the end-to-end comparison pipeline with `redcea`

---

## Installation

### Prerequisites

Prepare a clean Linux server with:

* `git`
* `conda` such as Miniconda or Mambaforge
* internet access for Python package installation
* access to GitHub, because dependency `mir` is installed from a git URL

Recommended Python version: `3.11`.

### Create the environment

```bash
git clone https://gitlab.aldan3.itm-rsmu.ru/isagroup/redcea.git
cd redcea

conda create -n redcea python=3.11 -y
conda activate redcea

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .[test]
```

This installs:

* the `redcea` package in editable mode
* the RedCEA CLI entry points `redcea` and `tcrempnet`
* the `tcremp` dependency, which provides `tcremp-run` and `tcremp-cluster`
* test dependencies including `pytest`

Notes:

* `mir` is installed from `https://github.com/antigenomics/mirpy.git`
* default clustering for `redcea` is `vdbscan`
* optional Leiden-based clustering requires an extra dependency

### Optional: install Leiden support

If you plan to run `--cluster-algo leiden`, `hierarchical_leiden`, or `leiden_dbscan`, install the optional Leiden dependency:

```bash
python -m pip install .[leiden]
```

If this optional install fails, you can still run the default `vdbscan` pipeline.

### Clustering modes

`redcea` supports several clustering backends:

* `vdbscan`: default RedCEA mode with per-group `eps` estimation on the joint sample/background graph
* `dbscan`: legacy TCRempNet mode on reduced embeddings, with pre-filtering of points with `d1 > eps` before running plain `DBSCAN`
* `leiden`: graph clustering on the joint KNN graph
* `hierarchical_leiden`: two-stage Leiden clustering
* `leiden_dbscan`: Leiden followed by per-cluster DBSCAN refinement

Use `dbscan` if you need behavior closer to historical TCRempNet runs and want the old epsilon-based noise pre-filter back.

### Verify the installation

Run the following commands in the activated environment:

```bash
python -c "import redcea, tcremp, mir; print('imports: OK')"
redcea --help
tcremp-run --help
pytest -q
```

Expected result:

* imports succeed without `ModuleNotFoundError`
* CLI help is printed for the requested commands
* tests pass

Important limitation:

* the test suite validates parser logic, helper functions, and a mocked pipeline smoke test
* it does not replace a real run on a small dataset in your target environment

### Recommended post-install smoke check

For a fresh server, use this validation sequence:

1. install the package with `python -m pip install -e .[test]`
2. run `redcea --help`
3. run `pytest -q`
4. run one small real dataset through `redcea` or `tcremp-run`
5. confirm that expected output files are created and the log ends without runtime errors

---

## Running RedCEA

### Option 1: two-step execution

If you plan to compare multiple case samples against the same background repertoire, compute the background embeddings once with `tcremp-run` and reuse them in downstream `redcea` runs.

#### Step 1: compute embeddings

```bash
tcremp-run \
  --input /projects/immunestatus/airr_format/sample.tsv \
  --output ./results \
  --chain TRB \
  -np 48
```

This produces embedding outputs in `./results`.

For large samples, embedding can take hours and requires substantial CPU and memory.

#### Step 2: run `redcea` on saved embeddings

```bash
redcea \
  -is /projects/immunestatus/airr_format/sample.tsv \
  -ib /projects/immunestatus/airr_format/background.tsv \
  -c TRB \
  -o ./results \
  -np 4 \
  -se ./results/sample_tcremp.parquet \
  -be ./results/background_tcremp.parquet
```

Use this mode when the embedding files already exist and you want to skip recomputation.

### Option 2: end-to-end pipeline

```bash
redcea \
  -is sample.tsv \
  -ib background.tsv \
  -c TRB \
  -o ./results \
  -np 8
```

In this mode, embeddings for both sample and background are computed automatically if they are not already available.

---

## CLI Tools

| CLI Tool | Description |
| --- | --- |
| `tcremp-run` | Computes TCRemP embeddings and optional clustering |
| `redcea` | Runs embedding, clustering, and enrichment |
| `tcremp-cluster` | Clusters existing embeddings |

---

## Example: Yellow Fever Dataset

```bash
redcea \
  --sample /projects/immunestatus/pogorelyy/airr_format/yfv_day_15.txt \
  --background /projects/immunestatus/pogorelyy/airr_format/yfv_day_0.txt \
  --output /projects/immunestatus/pogorelyy/redcea/yfv_res \
  --chain TRB \
  --prefix yfv_result \
  -np 16
```

---

## Output Files

Depending on the mode, the pipeline may create:

| File Name | Description |
| --- | --- |
| `*_sample_embeddings.parquet` | Sample embeddings produced or reused by `redcea` |
| `*_background_embeddings.parquet` | Background embeddings produced or reused by `redcea` |
| `*_tcremp_clusters.tsv` | Cluster assignments for both sample and background clonotypes |
| `*_summary_tcrempnet.tsv` | Per-cluster summary with counts, p-values, FDR, and log fold change |
| `*_enriched_clonotypes_tcremp.tsv` | Clonotypes from enriched clusters |
| `*_enriched_embeddings_tcremp.parquet` | Embeddings of enriched clonotypes with cluster metadata |
| `*.log` | Run log for debugging and runtime tracking |

### What to check after a real run

Treat the run as successful only if all of the following are true:

* the output directory exists
* `*_sample_embeddings.parquet` and `*_background_embeddings.parquet` exist or were intentionally supplied as inputs
* `*_tcremp_clusters.tsv` exists
* `*_summary_tcrempnet.tsv` exists and contains `cluster_id`, `cluster_size`, `sample`, `background`, `enrichment_fdr_zbinom`, and `log_fold_change`
* the log file ends with `TCRempNet pipeline completed.`

---

## Input Expectations

The pipeline expects repertoire tables that can be parsed by the underlying `tcremp` AIRR-loading utilities.

Before running on a clean server, verify on one small file that:

* the file path is correct and readable by the current user
* the repertoire contains the requested chain: `TRA`, `TRB`, or `TRA_TRB`
* required CDR3 and V/J fields expected by `tcremp` are present
* the file is not empty after filtering by chain and CDR3 length

If a run fails at startup, first check file format compatibility and chain selection.

---

## SLURM Job Example

Activate the `redcea` environment before submitting the job.

### Full pipeline

```bash
#!/bin/sh
#SBATCH --job-name=redcea
#SBATCH --cpus-per-task=48
#SBATCH --mem=128gb
#SBATCH --time=08:00:00
#SBATCH --output=redcea_run.%j.log

redcea \
  -is case.tsv \
  -ib control.tsv \
  -c TRB \
  -o ./results \
  -np 48
```

### Embedding only

```bash
tcremp-run \
  --input case.tsv \
  --output ./results \
  --chain TRB \
  -np 32
```

---

## Arguments

| Short | Long | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `-is` | `--sample` | Yes | none | Path to the sample repertoire table |
| `-ib` | `--background` | Yes | none | Path to the background repertoire table |
| `-o` | `--output` | Yes | none | Output directory |
| `-e` | `--prefix` | No | input filename | Output prefix |
| `-x` | `--index-col` | No | none | Optional input ID column to preserve in outputs |
| `-c` | `--chain` | Yes | none | `TRA`, `TRB`, or `TRA_TRB` |
| `-p` | `--prototypes-path` | No | package defaults | Path to a user-supplied prototypes file |
| `-n` | `--n-prototypes` | No | all available | Number of prototypes used for embedding |
| none | `--sample-random-prototypes` | No | `False` | Sample prototypes randomly |
| `-nc` | `--n-clonotypes` | No | all available | Number of clonotypes to process |
| none | `--sample-random-clonotypes` | No | `False` | Sample clonotypes randomly |
| `-s` | `--species` | No | `HomoSapiens` | Species for V/J gene alignment |
| `-u` | `--unique-clonotypes` | No | `False` | Use only unique clonotypes |
| `-r` | `--random-seed` | No | `42` | Random seed |
| `-np` | `--nproc` | No | `1` | Number of worker processes |
| `-llen` | `--lower-len-cdr3` | No | `5` | Minimum CDR3 length |
| `-hlen` | `--higher-len-cdr3` | No | `30` | Maximum CDR3 length |
| `-m` | `--metrics` | No | `dissimilarity` | TCRemP metric mode |
| `-d` | `--save-dists` | No | `True` | Save TCRemP distances |
| `-cl` | `--cluster` | No | `True` | Run clustering in embedding workflow |
| `-se` | `--sample-embedding` | No | none | Path to precomputed sample embeddings |
| `-be` | `--background-embedding` | No | none | Path to precomputed background embeddings |
| `--cluster-algo` | `--cluster-algo` | No | `vdbscan` | `vdbscan`, `dbscan`, `leiden`, `hierarchical_leiden`, or `leiden_dbscan` |
| `--n-bg-points` | `--n-bg-points` | No | all available | Limit background clonotypes to first N entries |
| `-npc` | `--cluster-pc-components` | No | `50` | Number of PCA components before clustering |
| `-ms` | `--cluster-min-samples` | No | `3` | Core-point threshold for clustering |
| `-kn` | `--k-neighbors` | No | `4` | Number of neighbors in the KNN graph |
| `-ekn` | `--eps-k-neighbors` | No | `4` | K-th neighbor used for eps estimation in `vdbscan` and `dbscan` |
| `--leiden-resolution` | `--leiden-resolution` | No | `1.0` | Leiden resolution parameter |
| `--leiden-sub-resolution` | `--leiden-sub-resolution` | No | `1.0` | Subclustering resolution for `hierarchical_leiden` |
| `--eps-estimation-based-on` | `--eps-estimation-based-on` | No | `sample` | Estimate eps from `sample`, `background`, or `all` |
| `--vdbscan-sym-rule` | `--vdbscan-sym-rule` | No | `asymmetric` | Symmetrization rule: `asymmetric`, `min`, or `max` |

---

## Reference

Vlasova et al., RedCEA: repertoire embeddings denoising clustering enrichment analysis, 2025, in preparation.

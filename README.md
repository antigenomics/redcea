# TCRemPNet: T-cell repertoire clustering and enrichment

TCRemPNet is a pipeline for comparing immune repertoires using prototype-based TCR embeddings. It is based on the original TCRemP embedding method, but supports comparison between case/control samples (e.g., vaccinated vs baseline) and clustering clonotypes using distances in embedding space.

This repository contains command-line tools for:

* Computing prototype-based embeddings for a case and background repertoire (`tcremp-run`)
* Performing clustering using PCA + DBSCAN (`tcremp-cluster`)
* Comparing cluster enrichment across conditions (`tcrempnet`)

---

## 🛠 Installation

```bash
git clone https://gitlab.aldan3.itm-rsmu.ru/isagroup/tcrempnet.git
cd tcrempnet
conda env create -n tcrempnet python=3.11
conda activate tcrempnet
pip install -e .
```

Ensure the `mirpy` library is installed and importable.

---

## 🚀 Running TCRemPNet

### Option 1: Two-step execution (embedding + enrichment separately)

💡 **Tip:** If you're planning to use the same background repertoire for multiple case samples (e.g., comparing several patient samples against a shared healthy baseline), it's highly recommended to compute and save background embeddings once using `tcremp-run`, and reuse them in all downstream `tcrempnet` runs. This significantly reduces runtime and avoids redundant computations.

#### Step 1: Compute embeddings for each sample using `tcremp-run`

```bash
tcremp-run \
  --input /projects/immunestatus/airr_format/sample.tsv \
  --output ./results --chain TRB -np 48
```

This produces:

* `results/sample_tcremp.parquet` — embedding table with prototype distances (in `.parquet` format)
* `results/sample_tcremp_clusters.tsv` — clustering results (optional if `--cluster` was used)

⚠️ Embedding is resource-intensive. For large samples (100,000+ clonotypes), allow up to 8 hours on 48 CPUs.

#### Step 2: Run `tcrempnet` on saved embeddings

```bash
tcrempnet \
  -is /projects/immunestatus/airr_format/sample.tsv \
  -ib /projects/immunestatus/airr_format/background.tsv \
  -c TRB -o ./results -np 4
  -se ./results/sample_tcremp.parquet \
  -be ./results/background_tcremp.parquet
```

> ✅ This step is fast: clustering + enrichment takes \~10 minutes per sample pair.

---

### Option 2: End-to-end pipeline

```bash
tcrempnet \
  -is sample.tsv \
  -ib background.tsv \
  -c TRB \
  -o ./results \
  -np 8
```

Embeddings for both case/control are computed internally.

---

## 📤 CLI Tools

| CLI Tool         | Description                                        |
| ---------------- | -------------------------------------------------- |
| `tcremp-run`     | Computes TCRemP embeddings and optional clustering |
| `tcrempnet`      | Performs embedding, clustering, and enrichment     |
| `tcremp-cluster` | Clusters existing embeddings via PCA + DBSCAN      |

---

## 🧪 Example: Yellow Fever Dataset

```bash
tcrempnet \
  --sample /projects/immunestatus/pogorelyy/airr_format/yfv_day_15.txt \
  --background /projects/immunestatus/pogorelyy/airr_format/yfv_day_0.txt \
  --output /projects/immunestatus/pogorelyy/tcrempnet/yfv_res \
  --chain TRB \
  --prefix yfv_result \
  -np 16
```

---

## 📥 Output files

Depending on the mode, the pipeline outputs:

| File Name                          | Description                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------- |
| `*_tcremp.parquet`                 | Embedding table with distances to all prototypes and clonotype metadata (in parquet format) |
| `*_tcremp_clusters.tsv`            | Clustering results per clonotype: `clone_id`, `cluster_id`, `cdr3aa`, `v`, `j`              |
| `*_summary_tcrempnet.tsv`          | Summary statistics for each cluster: size, enrichment p-value, FDR, case/control presence   |
| `*_enriched_clonotypes_tcremp.tsv` | Clonotypes from enriched clusters (FDR < 0.05), useful for downstream biological analysis   |
| `*.log`                            | Run log for debugging and runtime tracking                                                  |

> If `--cluster` is disabled, only embeddings are saved.

---

## 📎 SLURM job example

> Make sure you have activated the `tcrempnet` environment **before** submitting slurm jobs.


### Full pipeline

```bash
#!/bin/sh
#SBATCH --job-name=tcrempnet
#SBATCH --cpus-per-task=48
#SBATCH --mem=128gb
#SBATCH --time=08:00:00
#SBATCH --output=tcrempnet_run.%j.log

tcrempnet \
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

## ⚙️ Arguments

| Short | Long | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `-is` | `--sample` | Yes | — | Path to input file containing a clonotype (clone) table of the sample repertoire. |
| `-ib` | `--background` | Yes | — | Path to input file containing a clonotype (clone) table of the background repertoire. |
| `-o` | `--output` | Yes | — | Path to the output folder. |
| `-e` | `--prefix` | No | input filename | Output prefix. Defaults to the input clonotype table filename. |
| `-x` | `--index-col` | No | — | Optional: column in the input table containing user-defined IDs to be transferred to outputs. |
| `-c` | `--chain` | Yes | — | Chain type: `TRA` or `TRB` for single-chain clonotypes, or `TRA_TRB` for paired-chain clones. |
| `-p` | `--prototypes-path` | No | prebuilt set | Path to user-specified prototypes file. If not set, prebuilt prototypes from `$tcremp_path/data/data_prebuilt` are used. |
| `-n` | `--n-prototypes` | No | all available | Number of prototypes to use for embedding. Coordinates = (chains) × (V,J,CDR3 distances) × (n). |
| — | `--sample-random-prototypes` | No | `False` | Whether to sample prototypes randomly. |
| `-nc` | `--n-clonotypes` | No | all available | Number of clonotypes to process. |
| — | `--sample-random-clonotypes` | No | `False` | Whether to sample clonotypes randomly. |
| `-s` | `--species` | No | `HomoSapiens` | Species for V/J gene alignment. Options: `HomoSapiens`, `MusMusculus`, `MacacaMulatta`. |
| `-u` | `--unique-clonotypes` | No | — | Run only on unique clonotypes/clones to speed up analysis. |
| `-r` | `--random-seed` | No | `42` | Random seed for prototype sampling and other RNG-based steps. |
| `-np` | `--nproc` | No | `1` | Number of parallel processes. |
| `-llen` | `--lower-len-cdr3` | No | `5` | Minimum CDR3 length to keep. Shorter ones are filtered. |
| `-hlen` | `--higher-len-cdr3` | No | `30` | Maximum CDR3 length to keep. Longer ones are filtered. |
| `-m` | `--metrics` | No | `dissimilarity` | Whether to calculate similarity or dissimilarity scores with TCRemP. |
| — | `--sample-embeddings` | No | — | Path to precomputed sample embeddings (`.parquet`). |
| — | `--background-embeddings` | No | — | Path to precomputed background embeddings (`.parquet`). |
| `-d` | `--save-dists` | No | `True` | Whether to save TCRemP distances. |
| `-cl` | `--cluster` | No | `True` | Whether to perform clustering. |
| `-npc` | `--cluster-pc-components` | No | `50` | Number of PCA components before clustering. |
| `-ms` | `--cluster-min-samples` | No | `3` | `min_samples` parameter for DBSCAN. |
| `-kn` | `--k-neighbors` | No | `4` | k-th neighbor parameter for Knee eps estimation. |
| `-se` | `--sample-embedding` | No | — | Path to a sample embedding file (`.parquet`). Computed if not provided. |
| `-be` | `--background-embedding` | No | — | Path to a background embedding file (`.parquet`). Computed if not provided. |
| — | `--cluster-algo` | No | `dbscan` | Clustering algorithm to use: `dbscan` or `hdbscan`. |

---

## 📘 Reference

> Vlasova et al., TCRemPNet: vector-based clustering of immune repertoires with enrichment test, 2025 (in prep.)

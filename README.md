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

## 📘 Reference

> Vlasova et al., TCRemPNet: vector-based clustering of immune repertoires with enrichment test, 2025 (in prep.)

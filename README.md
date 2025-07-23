# TCRemPNet: T-cell repertoire clustering and enrichment

TCRemPNet is a pipeline for comparing immune repertoires using prototype-based TCR embeddings. It is based on the original TCRemP embedding method, but supports comparison between case/control samples (e.g., vaccinated vs baseline) and clustering clonotypes using distances in embedding space.

This repository contains the tools for:

* Computing prototype-based embeddings for a case and background repertoire.
* Performing clustering using PCA + DBSCAN.
* Comparing cluster enrichment across conditions.

---

## 🛠 Installation

```bash
git clone https://gitlab.aldan3.itm-rsmu.ru/isagroup/tcrempnet.git
cd tcrempnet
conda env create -n tcrempnet python=3.12
conda activate tcrempnet
pip install -e .
```

Ensure the `mirpy` library is installed and importable.

---

## 🚀 Running TCRemPNet

### Option 1: Two-step execution

You can first compute TCRemP embeddings for each sample separately using `tcremp_run.py` (from the original `tcremp` repository), and then pass the `.parquet` files to TCRemPNet.

#### Step 1: Compute embeddings

Example SLURM script:

`squeue_tcremp_as_large.sh`

```bash
#!/bin/sh
#SBATCH --job-name=tcremp_as_Mikh
#SBATCH --cpus-per-task=48
#SBATCH --mem=16gb
#SBATCH --time=08:00:00
#SBATCH --output=tcremp_as_Mikh.%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elizaveta.k.vlasova@gmail.com
#SBATCH --constraint=hpc
#SBATCH --partition=medium

python tcremp_run.py \
  --input /projects/immunestatus/pogorelyy/airr_format/P1_0_F1_with_1.txt \
  --output /projects/immunestatus/test \
  --chain TRB \
  -np 48
```

> ⚠️ Time estimate: embedding step is computationally intensive. For large samples (\~100,000 clonotypes), runtime may exceed 6 hours even on 100 CPU threads.

#### Step 2: Run `tcrEmpNet.py` on saved embeddings

```bash
python tcrEmpNet.py \
  -is sample.tsv \
  -ib background.tsv \
  -c TRB \
  -o results/ \
  -se sample_emb.parquet \
  -be background_emb.parquet \
  -np 8
```

> ✅ This step is fast: clustering + enrichment takes \~10 minutes per sample pair.

Alternatively, use SLURM:

```bash
sbatch squeue_tcrempnet_as.sh
```

---

### Option 2: Full run with embedding and clustering

```bash
python tcrEmpNet.py \
  -is sample.tsv \
  -ib background.tsv \
  -c TRB \
  -o results/ \
  -np 8
```

---

## 🧪 Example: Yellow Fever Dataset

```bash
python tcrEmpNet.py \
  -is /projects/immunestatus/yfv/yfv_day15.tsv \
  -ib /projects/immunestatus/yfv/yfv_day0.tsv \
  -c TRB \
  -o /projects/immunestatus/yfv/tcrempnet \
  -se /projects/immunestatus/yfv/embeddings_day15.parquet \
  -be /projects/immunestatus/yfv/embeddings_day0.parquet \
  -np 8
```

---

## 📎 SLURM job script example

`squeue_tcrempnet_as.sh`

```bash
#!/bin/sh
#SBATCH --job-name=tcrempnet
#SBATCH --cpus-per-task=4
#SBATCH --mem=1024gb
#SBATCH --time=24:00:00
#SBATCH --output=tcrempnet_as.%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elizaveta.k.vlasova@gmail.com
#SBATCH --constraint=hpc
#SBATCH --partition=long

python tcrEmpNet.py \
  -is /projects/immunestatus/rheum/airr_format/joint_as.tsv \
  -ib /projects/immunestatus/rheum/airr_format/joint_hd.tsv \
  -c TRB -o /projects/immunestatus/rheum/tcrempnet -np 4 \
  -se /projects/immunestatus/rheum/tcremp/joint_as_embeddings.parquet \
  -be /projects/immunestatus/rheum/tcremp/joint_hd_embeddings.parquet
```

---

## 📥 Arguments for `tcrEmpNet.py`

| Argument                                         | Description                              | Required | Default                   |
| ------------------------------------------------ | ---------------------------------------- | -------- | ------------------------- |
| `-is`, `--sample`                                | Sample clonotype file (case)             | ✅        | —                         |
| `-ib`, `--background`                            | Background clonotype file (control)      | ✅        | —                         |
| `-o`, `--output`                                 | Output folder path                       | ✅        | —                         |
| `-e`, `--prefix`                                 | Output filename prefix                   | ❌        | From sample name          |
| `-x`, `--index-col`                              | Optional column name to keep clone IDs   | ❌        | —                         |
| `-c`, `--chain`                                  | Chain type                               | ✅        | — (TRA, TRB, or TRA\_TRB) |
| `-p`, `--prototypes-path`                        | Custom prototype file path               | ❌        | Prebuilt                  |
| `-n`, `--n-prototypes`                           | Number of prototypes                     | ❌        | All                       |
| `-sample_random_p`, `--sample-random-prototypes` | Sample prototypes randomly               | ❌        | False                     |
| `-nc`, `--n-clonotypes`                          | Limit number of clonotypes               | ❌        | All                       |
| `-sample_random_c`, `--sample-random-clonotypes` | Sample clonotypes randomly               | ❌        | False                     |
| `-s`, `--species`                                | Species (for aligner)                    | ❌        | HomoSapiens               |
| `-u`, `--unique-clonotypes`                      | Use unique clonotypes only               | ❌        | —                         |
| `-r`, `--random-seed`                            | Random seed                              | ❌        | 42                        |
| `-np`, `--nproc`                                 | Number of threads                        | ❌        | 1                         |
| `-llen`, `--lower-len-cdr3`                      | Min CDR3 length                          | ❌        | 5                         |
| `-hlen`, `--higher-len-cdr3`                     | Max CDR3 length                          | ❌        | 30                        |
| `-m`, `--metrics`                                | Similarity or dissimilarity              | ❌        | dissimilarity             |
| `-d`, `--save-dists`                             | Save distances table                     | ❌        | True                      |
| `-cl`, `--cluster`                               | Run clustering step                      | ❌        | True                      |
| `-npc`, `--cluster-pc-components`                | PCA components                           | ❌        | 50                        |
| `-ms`, `--cluster-min-samples`                   | DBSCAN min\_samples                      | ❌        | 3                         |
| `-kn`, `--k-neighbors`                           | k-th neighbor for KneeLocator            | ❌        | 4                         |
| `-se`, `--sample-embedding`                      | Path to precomputed sample embedding     | ❌        | None                      |
| `-be`, `--background-embedding`                  | Path to precomputed background embedding | ❌        | None                      |

---

## 📊 Output files

The output consists of:

* `.tsv` table of prototype distances (if `--save-dists`)
* `.tsv` table with clustering results: `clone_id`, chain features, `cluster_id`

---

For more details, see the original publication:

> Vlasova et al., TCRemPNet: motif-based clustering of immune repertoires, 2025 (in prep.)

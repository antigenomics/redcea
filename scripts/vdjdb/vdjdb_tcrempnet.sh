#!/bin/sh

#SBATCH --job-name=tcrempnet_vdjdb       # Job name
#SBATCH --cpus-per-task=32     # Run on a single CPU
#SBATCH --mem=256gb                 # Job memory request
#SBATCH --time=24:00:00           # Time limit hrs:min:sec
#SBATCH --output=JobName.%j.log   # Standard output and error log
#SBATCH --constraint=hpc
#SBATCH --partition=long


tcrempnet \
  --sample /projects/immunestatus/vdjdb/airr_format/trb_vdjdb.tsv \
  --background /projects/immunestatus/vdjdb/airr_format/trb_background.tsv \
  --output /projects/immunestatus/vdjdb/tcrempnet_faiss_k6 \
  --chain TRB -np 32 \
  -se /projects/immunestatus/vdjdb/tcremp/trb_vdjdb_embeddings.parquet \
  -be /projects/immunestatus/vdjdb/tcremp/trb_background_embeddings.parquet \
  -kn 6
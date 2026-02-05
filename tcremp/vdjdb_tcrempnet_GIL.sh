#!/bin/sh

#SBATCH --job-name=tcrempnet_vdjdb_GIL       # Job name
#SBATCH --cpus-per-task=16     # Run on a single CPU
#SBATCH --mem=128gb                 # Job memory request
#SBATCH --time=2:00:00           # Time limit hrs:min:sec
#SBATCH --output=JobName.%j.log   # Standard output and error log
#SBATCH --constraint=hpc
#SBATCH --partition=short


python tcrempnet.py \
  --sample /projects/immunestatus/vdjdb/airr_format/trb_vdjdb_GILGFVFTL.tsv \
  --background /projects/immunestatus/vdjdb/airr_format/trb_background.tsv \
  --output /projects/immunestatus/vdjdb/tcrempnet_GILGFVFTL_trb_leiden_100k_res5 \
  --chain TRB -np 16 \
  -se /projects/immunestatus/vdjdb/tcremp/trb_vdjdb_GILGFVFTL_embeddings.parquet \
  -be /projects/immunestatus/vdjdb/tcremp/trb_background_embeddings.parquet \
  -kn 20 --n-bg-points 100000 -lr 5
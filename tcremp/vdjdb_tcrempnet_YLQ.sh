#!/bin/sh

#SBATCH --job-name=tcrempnet_vdjdb_YLQ_leiden_ierarchy       # Job name
#SBATCH --cpus-per-task=16     # Run on a single CPU
#SBATCH --mem=256gb                 # Job memory request
#SBATCH --time=2:00:00           # Time limit hrs:min:sec
#SBATCH --output=JobName.%j.log   # Standard output and error log
#SBATCH --constraint=hpc
#SBATCH --partition=short


python tcrempnet.py \
  --sample /projects/immunestatus/vdjdb/airr_format/trb_vdjdb_YLQPRTFLL.tsv \
  --background /projects/immunestatus/vdjdb/airr_format/trb_background.tsv \
  --output /projects/immunestatus/vdjdb/tcrempnet_YLQPRTFLL_trb_leiden_100k_dbscan \
  --chain TRB -np 16 \
  -se /projects/immunestatus/vdjdb/tcremp/trb_vdjdb_YLQPRTFLL_embeddings.parquet \
  -be /projects/immunestatus/vdjdb/tcremp/trb_background_embeddings.parquet \
  -kn 4 --n-bg-points 100000 -lr 1 -ms 5 --cluster-algo vdbscan
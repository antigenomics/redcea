#!/bin/sh

#SBATCH --job-name=tcrempnet_b27pos        # Job name
#SBATCH --cpus-per-task=16        # Run on a single CPU
#SBATCH --mem=256gb                 # Job memory request
#SBATCH --time=1:00:00           # Time limit hrs:min:sec
#SBATCH --output=tcrempnet_as_single.%j.log   # Standard output and error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elizaveta.k.vlasova@gmail.com
#SBATCH --constraint=hpc
#SBATCH --partition=short


python tcrEmpNet.py \
    -is /projects/immunestatus/rheum/airr_format/as_Shep_PB_F.tsv \
    -ib /projects/immunestatus/rheum/airr_format/joint_hd_b27pos.tsv \
    -c TRB -o /projects/immunestatus/rheum/tcrempnet_Shep_b27pos -np 2 \
    -se /projects/immunestatus/rheum/tcremp/as_Shep_PB_F_embeddings.parquet \
    -be /projects/immunestatus/rheum/tcremp/joint_hd_embeddings_b27pos.parquet 
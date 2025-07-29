#!/bin/sh

#SBATCH --job-name=tcrempnet        # Job name
#SBATCH --cpus-per-task=4        # Run on a single CPU
#SBATCH --mem=1024gb                 # Job memory request
#SBATCH --time=24:00:00           # Time limit hrs:min:sec
#SBATCH --output=tcrempnet_as.%j.log   # Standard output and error log
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
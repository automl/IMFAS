#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=2
#SBATCH --error=/home/mohan/git/current_projects/gravitas/AlgoSelectionMF/tmp/2022-05-09/15-24-44/.submitit/%j/%j_0_log.err
#SBATCH --job-name=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home/mohan/git/current_projects/gravitas/AlgoSelectionMF/tmp/2022-05-09/15-24-44/.submitit/%j/%j_0_log.out
#SBATCH --partition=cpu_normal
#SBATCH --signal=USR1@120
#SBATCH --time=60
#SBATCH --wckey=submitit

# setup
unset WANDB_DIR
unset WANDB_IGNORE_GLOBS

# command
export SUBMITIT_EXECUTOR=slurm
srun --output /home/mohan/git/current_projects/gravitas/AlgoSelectionMF/tmp/2022-05-09/15-24-44/.submitit/%j/%j_%t_log.out --error /home/mohan/git/current_projects/gravitas/AlgoSelectionMF/tmp/2022-05-09/15-24-44/.submitit/%j/%j_%t_log.err --unbuffered /home/mohan/miniconda3/envs/gravitas/bin/python -u -m submitit.core._submit /home/mohan/git/current_projects/gravitas/AlgoSelectionMF/tmp/2022-05-09/15-24-44/.submitit/%j

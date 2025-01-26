#!/bin/bash

# Remember to save with unix line endings

## Parameters
#SBATCH --account=ec29
#SBATCH --time=12:0:0
#SBATCH --job-name=mlagents_using_cores_$requested_cores
#SBATCH --ntasks=1 #only one script
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2200M
#SBATCH --array=0-21


## Commands
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate
python pythonscripts/FormalizedExperiment.py ${SLURM_ARRAY_TASK_ID}



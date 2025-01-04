#!/bin/bash

# Remember to save with unix line endings

## Parameters
#SBATCH --account=ec29
#SBATCH --time=10-0:0:0
#SBATCH --job-name=mlagents_using_cores_$requested_cores
#SBATCH --ntasks=1 #only one script
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2000M


## Commands
python pythonscripts/FormalizedExperiment.py



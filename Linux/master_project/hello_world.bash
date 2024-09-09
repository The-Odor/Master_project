#!/bin/bash[^1]

## Parameters
#SBATCH --account=ec29
#SBATCH --time=0-0:0:30
#SBATCH --job-name=theodoma
#SBATCH --ntasks=1 #only one script
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10M


## Setup commands

## Actual Commands
python hello_world.py

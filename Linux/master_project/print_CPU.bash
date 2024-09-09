#!/bin/bash

## Parameters
#SBATCH --account=ec29
#SBATCH --time=0-0:2:0
#SBATCH --job-name=theodoma
#SBATCH --ntasks=1 #only one script
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1000M


## Setup commands
##python -m venv venv

## Actual Commands
python cpu_count.py


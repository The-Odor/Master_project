#!/bin/bash

## Parameters
#SBATCH --account=ec29
#SBATCH --time=10-0:0:0
#SBATCH --job-name=mlagents_using_cores_$requested_cores
#SBATCH --ntasks=1 #only one script
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2000M

## First-time setup commands
python -m venv venv
scp C:\Users\theod\Master_project\Linux\master_project ec-theodoma@fox.educloud.no:/cluster/work/users/ec-theodoma

## Setup commands
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate
pip install setuptools==58
pip install tqdm
pip install neat
pip install neat-python
pip install numpy
pip install ml-agents/ml-agents-envs
pip install ml-agents/ml-agents
pip install graphviz

## Actual Commands
python pythonscripts/FormalizedExperiment.py




## fuck the requirements, it keeps looking for locally installed things
##scp C:\Users\theod\Master_project\venv\requirements.txt ec-theodoma@fox.educloud.no:/cluster/work/users/ec-theodoma/venv
##pip install -r venv/requirements.txt
## fuck the requirements, it keeps looking for locally installed things

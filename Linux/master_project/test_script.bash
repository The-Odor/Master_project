#!/bin/bash

## Parameters
#SBATCH --account=ec29
#SBATCH --time=2-0:0:0
#SBATCH --job-name=mlagents_using_cores_$requested_cores
#SBATCH --ntasks=1 #only one script
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2000M


## Setup commands
##python -m venv venv
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
##scp C:\Users\theod\Master_project\venv\requirements.txt ec-theodoma@fox.educloud.no:/cluster/work/users/ec-theodoma/venv
##pip install -r venv/requirements.txt
## fuck the requirements, it keeps looking for locally installed things

## Actual Commands
python pythonscripts/MLAgentLearn.py 64


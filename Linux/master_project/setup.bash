## First-time setup commands
module load Python/3.10.8-GCCcore-12.2.0
python -m venv venv
source venv/bin/activate
git clone https://github.com/Unity-Technologies/ml-agents.git ml-agents-github
##scp C:\Users\theod\Master_project\Linux\master_project.zip ec-theodoma@fox.educloud.no:/fp/homes01/u01/ec-theodoma/

pip install setuptools==58
pip install tqdm
##pip install neat
##pip install neat-python
pip install git+https://github.com/CodeReclaimers/neat-python.git
pip install numpy
pip install ml-agents-github/ml-agents-envs/
pip install ml-agents-github/ml-agents
##pip install ml-agents@git+https://github.com/Unity-Technologies/ml-agents/tree/develop/ml-agents
pip install graphviz
pip install matplotlib
pip install cma

chmod -R 755 /fp/homes01/u01/ec-theodoma/master_project/*

# TODO: Get a command that changes line endings in FormalizedExperiment.bash and ... this file. Hmmmmm....

## fuck the requirements, it keeps looking for locally installed things
##scp C:\Users\theod\Master_project\venv\requirements.txt ec-theodoma@fox.educloud.no:/cluster/work/users/ec-theodoma/venv
##pip install -r venv/requirements.txt
## fuck the requirements, it keeps looking for locally installed things

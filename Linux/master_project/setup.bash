## First-time setup commands
module load Python/3.10.8-GCCcore-12.2.0
python -m venv venv
source venv/bin/activate
git clone https://github.com/Unity-Technologies/ml-agents.git ml-agents-github
##scp C:\Users\theod\Master_project\Linux\master_project ec-theodoma@fox.educloud.no:/cluster/work/users/ec-theodoma

pip install setuptools==58
pip install tqdm
pip install neat
pip install neat-python
pip install numpy
pip install ml-agents-github/ml-agents-envs/
pip install ml-agents-github/ml-agents
##pip install ml-agents@git+https://github.com/Unity-Technologies/ml-agents/tree/develop/ml-agents
pip install graphviz
pip install matplotlib
pip install cma



## fuck the requirements, it keeps looking for locally installed things
##scp C:\Users\theod\Master_project\venv\requirements.txt ec-theodoma@fox.educloud.no:/cluster/work/users/ec-theodoma/venv
##pip install -r venv/requirements.txt
## fuck the requirements, it keeps looking for locally installed things

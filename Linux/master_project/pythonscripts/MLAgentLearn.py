from Learner import Learner
import configparser as cfp
import random
import multiprocessing as mp
import sys
# cd C:\Users\theod\Master_project
# venv\Scripts\activate
# python Pythonscripts\MLAgentLearn.py


configFilepath = "/fp/homes01/u01/ec-theodoma/pythonscripts/configs/pythonConfig.config"
configData = cfp.ConfigParser()
with open(configFilepath, "r") as infile:
    configData.read_file(infile)
configData["Default"]

CONFIG_DETAILS = configData["Default"]

with open(CONFIG_DETAILS["configFilepath"]) as infile:
    for line in infile.readlines():
        if line.startswith("pop_size = "):
            CONFIG_DETAILS["populationCount"] = line[11:-1]
            # CONFIG_DETAILS["parallelWorkers"] = str(min(24, int(line[11:-1])))
            CONFIG_DETAILS["parallelWorkers"] = str(min(2*int(sys.argv[1]), 2*mp.cpu_count(), int(CONFIG_DETAILS["populationCount"])))

#sys.stdout = open(f"outputPython_{CONFIG_DETAILS['parallelWorkers']}_cores.txt", "wt")

print(f"""Setting number of parallel processes to the lowest of:
Arg:   {2*int(sys.argv[1])}
Count: {2*mp.cpu_count()}
nPop:  {CONFIG_DETAILS['populationCount']}""")

CONFIG_DETAILS["populationFolder"] = (
    CONFIG_DETAILS["populationFolder1"] +
    CONFIG_DETAILS["populationCount"] + CONFIG_DETAILS["populationFolder2"]
)



# random.seed(CONFIG_DETAILS["PythonSeed"])


if __name__ == "__main__":
    mp.freeze_support()

    learner = Learner(CONFIG_DETAILS)
    finalGeneration, bestBoi = learner.run(useCheckpoint=True)    

    # import winsound
    # duration = 1000  # milliseconds
    # freq = 69  # Hz
    # winsound.Beep(freq, duration)
    # print(\a)
    # print(\007)
    # winsound.MessageBeep()


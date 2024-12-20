from Learner import Learner_NEAT, Learner_CMA, Learner_NEAT_From_CMA
import configparser as cfp
import random
import multiprocessing as mp

# cd C:\Users\theod\Master_project
# venv\Scripts\activate
# python Pythonscripts\MLAgentLearn.py

configFilepath = "C:\\Users\\theod\\Master_project\\Pythonscripts\\configs\\pythonConfig.config"
configData = cfp.ConfigParser()
with open(configFilepath, "r") as infile:
    configData.read_file(infile)
configData["Default"]

CONFIG_DETAILS = configData["Default"]

with open(CONFIG_DETAILS["configFilepath"]) as infile:
    for line in infile.readlines():
        if line.startswith("pop_size = "):
            CONFIG_DETAILS["populationCount"] = line[11:-1]
            CONFIG_DETAILS["parallelWorkers"] = str(min(24, int(line[11:-1])))

CONFIG_DETAILS["populationFolder"] = (
    CONFIG_DETAILS["populationFolder1"] +
    CONFIG_DETAILS["populationCount"] + CONFIG_DETAILS["populationFolder2"]
)



# random.seed(CONFIG_DETAILS["PythonSeed"])


if __name__ == "__main__":
    mp.freeze_support()


    mode = 3

    if mode == 1:
        learner = Learner_NEAT(CONFIG_DETAILS)
        finalGeneration, bestBoi = learner.run(useCheckpoint=True)    

    elif mode == 2:
        learner = Learner_CMA(CONFIG_DETAILS)
        learner.train()

    elif mode == 3:
        learner = Learner_NEAT_From_CMA(CONFIG_DETAILS)
        finalGeneration, bestBoi = learner.run(useCheckpoint=True)    
        # learner.run(useCheckpoint=True)







    # import winsound
    # duration = 1000  # milliseconds
    # freq = 69  # Hz
    # winsound.Beep(freq, duration)
    # print(\a)
    # print(\007)
    # winsound.MessageBeep()


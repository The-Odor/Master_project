import matplotlib.pyplot as plt
from Learner import Learner_NEAT
import configparser as cfp
import copy
import os
import pickle

# Fetch the config
configFilepath = "C:\\Users\\theod\\Master_project\\Pythonscripts\\configs\\pythonConfig.config"
configData = cfp.ConfigParser()
with open(configFilepath, "r") as infile:
    configData.read_file(infile)
CONFIG_DETAILS = configData["Default"]

# Append parallelization and populationscale data
with open(CONFIG_DETAILS["configFilepath"]) as infile:
    for line in infile.readlines():
        if line.startswith("pop_size = "):
            CONFIG_DETAILS["populationCount"] = line[11:-1]
            CONFIG_DETAILS["parallelWorkers"] = str(min(12, int(line[11:-1])))

# Finalize destination folder
CONFIG_DETAILS["populationFolder"] = (
    CONFIG_DETAILS["populationFolder1"] +
    CONFIG_DETAILS["populationCount"] + CONFIG_DETAILS["populationFolder2"]
)

fitnessDictFilepath = r"C:\Users\theod\Master_project\data and table storage\fitness_over_generations_dict"
morphologies = ["gecko", "queen", "stingray", "insect", "babya", "spider", "tinlicker", "longleg", "salamander", "park", "squarish", "blokky", "babyb", "snake", "linkin", "ww", "turtle", "penguin", "zappa", "garrix", "ant", "pentapod"]
NEATinclude = ["gecko","queen","spider","babyb","tinlicker","ww","snake"]
morphologies = NEATinclude
NMORPHOLOGIES = 22
NMORPHOLOGIES = len(morphologies)
morphologies = [morph + "_v1?team=0" for morph in morphologies]

class Learner_NEAT_but_with_new_reward_aggregation(Learner_NEAT):
    def rewardAggregation(self,rewards):
        modified_reward = {behavior:0 for behavior in rewards}
        for behavior in rewards:
            modified_reward[behavior]+= rewards[behavior][-1]
        return modified_reward


if __name__ == "__main__":
    if os.path.exists(fitnessDictFilepath):
        with open(fitnessDictFilepath, "rb") as infile:
            fitnessDict = pickle.load(infile)
    else:
        fitnessDict = {}
    for i, morph in enumerate(morphologies):
        if morph not in fitnessDict:
            fitnessDict[morph] = {}
        for generation in range(20+1):
            if generation in fitnessDict[morph]:
                continue
            learner = Learner_NEAT_but_with_new_reward_aggregation(
                config_details=copy.deepcopy(CONFIG_DETAILS),
                morphologyTrainedOn=[morph],
            )
            learner.switchEnvironment(
                trainedMorphology=morph,
            )

            pop,_ = learner.findGeneration(specificGeneration=generation)
            learner.evaluatePopulation(list(pop.population.items()), pop.config)
            
            fitnessDict[morph][generation] = [
                genome.fitness for _, genome in pop.population.items()
            ]
            with open(fitnessDictFilepath, "wb") as outfile:
                pickle.dump(fitnessDict, outfile)

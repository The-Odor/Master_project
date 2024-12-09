from Learner import Learner_NEAT, Learner_CMA, Learner_NEAT_From_CMA
import configparser as cfp
import multiprocessing as mp
import itertools as it

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
            CONFIG_DETAILS["parallelWorkers"] = str(min(24, int(line[11:-1])))

# Finalize destination folder
CONFIG_DETAILS["populationFolder"] = (
    CONFIG_DETAILS["populationFolder1"] +
    CONFIG_DETAILS["populationCount"] + CONFIG_DETAILS["populationFolder2"]
)

NMORPHOLOGIES = 22
morphologies = ["stingray", "insect", "gecko", "babya", "spider", "queen", "tinlicker", "longleg", "salamander", "park", "squarish", "blokky", "babyb", "snake", "linkin", "ww", "turtle", "penguin", "zappa", "garrix", "ant", "pentapod"]
morphologies = morphologies[:3]
morphologies = [morph + "_v1?team=0" for morph in morphologies]
NREPETITIONS = 3

class FalseQueue():
    def get(self,):
        return 1
    def put(self,thing):
        pass

class Parallelizable_Learner_NEAT(Learner_NEAT):
    # def fitnessFuncMapper(self, arg):
    #     # Fetch simulation targets from queue
    #     morph = queue.get()
    #     return self.fitnessFunc(*arg, morphologiesToSimulate=morph), morph
    def rewardAggregation(self, rewards):
        # Reward for a single behavior is equivalent to final distance
        for behavior in rewards:
            rewards[behavior] = rewards[behavior][-1]
        return rewards

if __name__ == "__main__":
    mp.freeze_support()

    ### CTRNN, unseeded, Experiment 1
    scoreDict = {}
    for i,trainedMorphology in enumerate(morphologies):
        print(f"\n\nTraining morphology {i+1} unseeded: {trainedMorphology}".upper)
        # Training
        # if not 'winner' in globals():  
        learner = Parallelizable_Learner_NEAT(CONFIG_DETAILS, morphologyTrainedOn=[trainedMorphology])
        finalGeneration, winner = learner.run(useCheckpoint=False)

        # Self-evaluation
        # learner.morphologiesToSimulate = [trainedMorphology]
        # selfScore = learner.fitnessFunc(winner, FalseQueue())

        # Evaluation
        manager = mp.Manager()
        queue = manager.Queue()
        # nOthersToTest = 4 # 4 gives 7315 evaluations, 5 gives 26334
        # simulations = [queue.put(i) for i in it.combinations(morphologies, r=nOthersToTest)]

        # learner.morphologiesToSimulate = morphologies
        CONFIG_DETAILS["populationCount"] = "3"#len(simulations)
        scores = learner.evaluatePopulation([[0,winner]]*NREPETITIONS, learner.NEAT_CONFIG, training=False)
        
        # Save data TODO
        # print(selfScore, otherScore)
        # print(scoreDict)yhon
        scoreDict[trainedMorphology] = scores

    tabularDict = {}
    for morph_i, scores in scoreDict.items():
        # scoreDict[morph] = sum(scores)/len(scores)
        tabularDict[morph_i] = {morph_ii: sum(scores[morph_ii])/len(scores[morph_ii]) for morph_ii in morphologies}

    tabularList = [["...",] +[i[:-10][:5] for i in morphologies],]
    for morph_i in morphologies:
        tabularList.append([morph_i])
        for morph_ii in morphologies:
            tabularList[-1].append(tabularDict[morph_i][morph_ii])
    for i in range(1, len(tabularList[1:])+1):
        tabularList[i][0] = tabularList[i][0][:-10]
    from tabulate import tabulate
    # tabulate([morphologies] + [[morph] + scoreDict[morph] for morph in morphologies])
    print(tabulate(tabularList[1:], headers=tabularList[0], floatfmt=".1f"))

    ### CTRNN, seeded, Experiment 2
    # Training
    pass
    # Self-evaluation
    pass
    # Other-evaluation
    pass


    ### Sine, Experiment 3
    # Training
    pass
    # Self-evaluation
    pass
    # Other-evaluation
    pass





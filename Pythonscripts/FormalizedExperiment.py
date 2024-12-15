from Learner import Learner_NEAT, Learner_CMA, Learner_NEAT_From_CMA
import configparser as cfp
import multiprocessing as mp
import itertools as it
import copy

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

NMORPHOLOGIES = 22
morphologies = ["stingray", "insect", "gecko", "babya", "spider", "queen", "tinlicker", "longleg", "salamander", "park", "squarish", "blokky", "babyb", "snake", "linkin", "ww", "turtle", "penguin", "zappa", "garrix", "ant", "pentapod"]
# morphologies = morphologies[:3]
morphologies = [morph + "_v1?team=0" for morph in morphologies]
EVALUATIONREPETITIONS = 3

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


def seedingFunction(self, pop):
    # TODO:
    # To be put in during experiment 2
    numberOfSeeds = 72 # a tenth of 720
    for genID, genome in list(pop.population.items())[:numberOfSeeds]:
        n1, n2 = list(genome.nodes.keys())[:2]
        factor = 1.0
        for node, bias in zip((n1,n2), (-2.75/5, -1.75/5)):
            # genome.nodes[node].aggregation = sum
            # genome.nodes[node].activation = sigmoid_activation
            genome.nodes[node].bias = bias*factor
            genome.nodes[node].response = 1
        genome.add_connection(self.NEAT_CONFIG.genome_config, n1, n1, 0.9*factor, True)
        genome.add_connection(self.NEAT_CONFIG.genome_config, n1, n2,-0.2*factor, True)
        genome.add_connection(self.NEAT_CONFIG.genome_config, n2, n1, 0.2*factor, True)
        genome.add_connection(self.NEAT_CONFIG.genome_config, n2, n2, 0.9*factor, True)




if __name__ == "__main__":
    mp.freeze_support()

    ### CTRNN, unseeded, Experiment 1
    scoreDict = {}
    fullEnvironment = CONFIG_DETAILS["exeFilepath"]
    for i,trainedMorphology in enumerate(morphologies):
        print(f"\n\nTraining morphology {i+1} unseeded: {trainedMorphology}".upper())
        # Training
        # if not 'winner' in globals():  
        learner = Parallelizable_Learner_NEAT(copy.deepcopy(CONFIG_DETAILS), morphologyTrainedOn=[trainedMorphology])
        tmorph = trainedMorphology[:-10]
        index = learner.CONFIG_DETAILS["exeFilepath"].rfind(learner.CONFIG_DETAILS["unityEnvironmentName"])
        newEnvironment = learner.CONFIG_DETAILS["exeFilepath"][:index]
        newEnvironment+= f"{tmorph}\\{learner.CONFIG_DETAILS['unityEnvironmentName']}.exe"
        learner.switchEnvironment(newEnvironment)
        # import os
        # index = learner.CONFIG_DETAILS["exeFilepath"].rfind(learner.CONFIG_DETAILS["unityEnvironmentName"])
        # outfilePath = learner.CONFIG_DETAILS["exeFilepath"][:index]
        # if not os.path.exists(outfilePath):
        #     print(f"making directory {outfilePath}")
        #     os.makedirs(outfilePath)
        # learner.CONFIG_DETAILS["exeFilepath"] = fullEnvironment
        # continue

        finalGeneration, winner = learner.run(useCheckpoint=True)

        # Self-evaluation
        # learner.morphologiesToSimulate = [trainedMorphology]
        selfScore = learner.fitnessFunc(winner, FalseQueue())

        # Evaluation
        manager = mp.Manager()
        queue = manager.Queue()
        # nOthersToTest = 4 # 4 gives 7315 evaluations, 5 gives 26334
        # simulations = [queue.put(i) for i in it.combinations(morphologies, r=nOthersToTest)]

        # learner.morphologiesToSimulate = morphologies
        learner.switchEnvironment(fullEnvironment)
        CONFIG_DETAILS["populationCount"] = "3"#len(simulations)
        scores = learner.evaluatePopulation([[0,winner]]*EVALUATIONREPETITIONS, learner.NEAT_CONFIG, training=False)
        
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





from Learner import Learner_NEAT, Learner_CMA, Learner_NEAT_From_CMA
import configparser as cfp
import multiprocessing as mp
import itertools as it
import copy
import sys
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

scoreDictFilepath = r"C:\Users\theod\Master_project\data and table storage\experimentOneScoreDict"
NMORPHOLOGIES = 22
morphologies = ["gecko", "queen", "stingray", "insect", "babya", "spider", "tinlicker", "longleg", "salamander", "park", "squarish", "blokky", "babyb", "snake", "linkin", "ww", "turtle", "penguin", "zappa", "garrix", "ant", "pentapod"]
# NMORPHOLOGIES = 3; morphologies = morphologies[:NMORPHOLOGIES]
if len(sys.argv) > 1:
    try:
        morphologies = [morphologies[int(sys.argv[1])]]
    except ValueError:
        if sys.argv[1] in morphologies:
            morphologies = [sys.argv[1]]
# morphologies = morphologies[:6]
morphologies = [morph + "_v1?team=0" for morph in morphologies]
EVALUATIONREPETITIONS = 3

class FalseQueue():
    def get(self,):
        return 1
    def put(self,thing):
        pass

class Parallelizable_Learner_NEAT(Learner_NEAT):
    def __init__(self, config_details, morphologyTrainedOn=None):
        super().__init__(config_details, morphologyTrainedOn)
        # Reasoning for punishment magnitude:
        # A decent fitness level is 2, a dead controller gives a fitness of 
        # about 1, duration of simulation is 20 seconds, and expected step 
        # frequency is about twice a second given data from pre-experiments. 
        # 40 steps thus needs to give less of a punishment than the expoected
        # reward for a decent walk, so punishment needs to be less than 1/40.
        # This punishment is reduced from its maximum value to allow decent
        # walks an actual reward magnitude
        self.oscillationPunishment = (1/40) / 4096
    # def fitnessFuncMapper(self, arg):
    #     # Fetch simulation targets from queue
    #     morph = queue.get()
    #     return self.fitnessFunc(*arg, morphologiesToSimulate=morph), morph
    def rewardAggregation(self, rewards):
        # Reward for a single behavior is equivalent to final distance
        modified_reward = {behavior:0 for behavior in rewards}
        for behavior in rewards:
            for i in range(2, len(rewards[behavior])):
                break
                sign1 = rewards[behavior][i] - rewards[behavior][i-1]
                sign2 = rewards[behavior][i-1] - rewards[behavior][i-2]

                # If the two latest differences have different signs (i.e.
                # the controller has performed an oscillation), give it 
                # a small penalty, to combat jittering. 0 inclusive;
                # plateauing counts as an oscillation
                if sign1*sign2 <= 0:
                    modified_reward[behavior]-= self.oscillationPunishment
                
            modified_reward[behavior]+= rewards[behavior][-1]
        return modified_reward


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

def switchEnvironment(learner, trainedMorphology):
    if trainedMorphology.endswith("_v1?team=0"):
        trainedMorphology = trainedMorphology[:-10]
    index = CONFIG_DETAILS["exeFilepath"].rfind(learner.CONFIG_DETAILS["unityEnvironmentName"])
    newEnvironment = learner.CONFIG_DETAILS["exeFilepath"][:index]
    newEnvironment+= f"{trainedMorphology}{learner.dirSeparator}{learner.CONFIG_DETAILS['unityEnvironmentName']}{learner.CONFIG_DETAILS['fileendingFormat']}"
    learner._switchEnvironmentPath(newEnvironment)



if __name__ == "__main__":
    mp.freeze_support()

    ### CTRNN, unseeded, Experiment 1
    try:
        with open(scoreDictFilepath, "rb") as infile:
            scoreDict = pickle.load(infile)
    except FileNotFoundError:
        scoreDict = {}
    fullEnvironment = CONFIG_DETAILS["exeFilepath"]
    for i,trainedMorphology in enumerate(morphologies):
        if trainedMorphology in scoreDict:
            continue
        print(f"\n\nTraining morphology {i+1} unseeded: {trainedMorphology}".upper())
        # Training
        # if not 'winner' in globals():  
        learner = Parallelizable_Learner_NEAT(copy.deepcopy(CONFIG_DETAILS), morphologyTrainedOn=[trainedMorphology])
        # tmorph = trainedMorphology[:-10]
        # index = learner.CONFIG_DETAILS["exeFilepath"].rfind(learner.CONFIG_DETAILS["unityEnvironmentName"])
        # newEnvironment = learner.CONFIG_DETAILS["exeFilepath"][:index]
        # newEnvironment+= f"{tmorph}{learner.dirSeparator}{learner.CONFIG_DETAILS['unityEnvironmentName']}{learner.CONFIG_DETAILS['fileendingFormat']}"
        # learner._switchEnvironmentPath(newEnvironment)
        switchEnvironment(learner, trainedMorphology)
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
        # selfScore = learner.fitnessFunc(winner, FalseQueue())

        # Evaluation
        print(f"Evaluating morphology {i+1} unseeded: {trainedMorphology}".upper())
        manager = mp.Manager()
        queue = manager.Queue()
        # nOthersToTest = 4 # 4 gives 7315 evaluations, 5 gives 26334
        # simulations = [queue.put(i) for i in it.combinations(morphologies, r=nOthersToTest)]

        # learner.morphologiesToSimulate = morphologies
        individualSimulations = False

        if individualSimulations:
            scores = {}
            for testingMorphology in morphologies:
                CONFIG_DETAILS["populationCount"] = str(EVALUATIONREPETITIONS)#len(simulations)
                # learner._switchEnvironmentPath(fullEnvironment)
                switchEnvironment(learner, testingMorphology)
                score = learner.evaluatePopulation([[0,winner]]*EVALUATIONREPETITIONS, learner.NEAT_CONFIG, training=False)[None]
                scores[testingMorphology] = [list(s.values())[0] for s in score]
        else:
            learner._switchEnvironmentPath(fullEnvironment)
            CONFIG_DETAILS["populationCount"] = "3"#len(simulations)
            scores = learner.evaluatePopulation([[0,winner]]*EVALUATIONREPETITIONS, learner.NEAT_CONFIG, training=False)[None]
        
        # Save data TODO
        # print(selfScore, otherScore)
        # print(scoreDict)yhon
        
        scoreDict[trainedMorphology] = scores

    with open(scoreDictFilepath, "wb") as outfile:
        pickle.dump(scoreDict, outfile)

    tabularDict = {}
    for morph_i, scores in scoreDict.items():
        # scoreDict[morph] = sum(scores)/len(scores)
        tabularDict[morph_i] = {morph_ii: 0 for morph_ii in morphologies}
        for i in range(EVALUATIONREPETITIONS):
            for morph_ii, _ in scoreDict.items():
                # tabularDict[morph_i][morph_ii]+= scores[i][morph_ii]/EVALUATIONREPETITIONS
                tabularDict[morph_i][morph_ii]+= scores[morph_ii][i]/EVALUATIONREPETITIONS * 10000

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
    # morphologies = morphologies[:2]
    morphologies = [morph[:-10] if morph.endswith("_v1?team=0") else morph for morph in morphologies]
    for morph in morphologies:
        learner = Learner_CMA(copy.deepcopy(CONFIG_DETAILS), morphologyTrainedOn=morph)
        learner.switchEnvironment(morph)    
        learner.train()

    fitnesses = {}
    for morph in morphologies:
        fitnesses[morph] = []
        for i in range(10,100+1,10):
            file = fr"C:\Users\theod\Master_project\Populations\CMA\{morph}\format_3\iteration_{i}"
            with open(file, "rb") as infile:
                es = pickle.load(infile)[0]
            fitnesses[morph].extend([-i for i in es.fit.hist[:10]])
    
    import matplotlib.pyplot as plt
    for morph in fitnesses:
        plt.plot(fitnesses[morph], label=morph)
    plt.legend()
    plt.show()
        
    # Self-evaluation
    pass
    # Other-evaluation
    pass





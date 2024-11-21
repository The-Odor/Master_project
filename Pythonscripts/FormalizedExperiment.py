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

class FalseQueue():
    def get(self,):
        return 1
    def put(self,thing):
        pass


if __name__ == "__main__":
    mp.freeze_support()

    ### CTRNN, unseeded

    for trainedMorphology in morphologies:
        # Training
        learner = Learner_NEAT(CONFIG_DETAILS, morphologiesToSimulate=trainedMorphology)
        finalGeneration, winner = learner.run(useCheckpoint=True)

        # Self-evaluation
        learner.morphologiesToSimulate = [trainedMorphology]
        selfScore = learner.fitnessFunc(winner, FalseQueue())

        # Other-evaluation
        manager = mp.Manager()
        queue = manager.Queue()
        nOthersToTest = 4 # 4 gives 7315 evaluations, 5 gives 26334
        simulations = [queue.put(i) for i in it.combinations(morphologies, r=nOthersToTest)]
        class Parallelizable_Learner_NEAT(Learner_NEAT):
            def fitnessFuncMapper(self, arg):
                # Fetch simulation targets from queue
                morph = queue.get()
                return self.fitnessFunc(*arg, morphologiesToSimulate=morph)
        CONFIG_DETAILS["populationCount"] = len(simulations)
        otherScore = learner.simulateGenome() 
        
        # Save data TODO
        print(selfScore, otherScore)


    ### CTRNN, seeded
    # Training
    pass
    # Self-evaluation
    pass
    # Other-evaluation
    pass


    ### Sine
    # Training
    pass
    # Self-evaluation
    pass
    # Other-evaluation
    pass





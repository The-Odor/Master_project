from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
import numpy as np
import torch
import torch.nn as nn
# import torch.optim as optim # May be necessary, haven't used it yet
import pdb
import neat
import multiprocessing
import pickle
import random
import time
import datetime
import os

# cd C:\Users\theod\Master_project
# venv\Scripts\activate
# python Pythonscripts\MLAgentLearn.py

# cd Master_project
# venv\Scripts\activate
# python
# from Pythonscripts.MLAgentLearn import *
# demonstrateGenome("Populations\popcount_24_simlength600_generation_101.pkl", NEAT_CONFIG)


# TODO: make config into a file
CONFIG_DETAILS = {
    "unityEnvironmentFilepath": r"C:\Users\theod\Master_project",
    "unityEnvironmentName": "Bot locomotion",
    "unityEnvironmentVersion": ".0.4",
    "configFilepath": r"C:\Users\theod\Master_project"
                      r"\Pythonscripts\configs\NEATconfig",
    "simulationSteps": 300,
    "unitySeed": lambda : random.randint(1, 1000),  # Is called
    "PythonSeed": lambda : random.randint(1, 1000), # Is called
    "processingVersion": 3, #serial,list-based parallel,starmap-based parallel
    "parallelWorkers": 12,
    "numberOfGenerations": 101,
    "simulationTimeout": 1800, # In seconds; 1800 seconds is half an hour
    "generationsBeforeSave": 1,
    "resultsFolder": r"C:\Users\theod\Master_project\Populations",
    # "lastPopulationCheckpoint": r"C:\Users\theod\Master_project\Populations"
    #                         r"\popcount_24_simlength600_generation_100.pkl",
    "lastPopulationCheckpoint": None,

}
with open(CONFIG_DETAILS["configFilepath"]) as infile:
    for line in infile.readlines():
        if line.startswith("pop_size = "):
            CONFIG_DETAILS["populationCount"] = min(24, int(line[11:]))

NEAT_CONFIG = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    CONFIG_DETAILS["configFilepath"]
    )

doNothingBreakPoint = lambda : 0
import random
random.seed(CONFIG_DETAILS["PythonSeed"]())


def addLayer(network, layerIdx, actFunc=nn.ReLU(), bias=True):
    # Layer added is in the form of an identity matrix with bias=0 and should 
    # have no effective change immediately upon addition
    # ASSUMPTION: actFunc(actFunc(x)) = actFunc(x), like for ReLU.
    layers = [i for i in network]
    identitySize = network[2*layerIdx-2].out_features
    newNetwork = nn.Sequential(*(layers[:2*layerIdx] +\
                 [nn.Linear(identitySize, identitySize, bias=bias), actFunc] +\
                 layers[2*layerIdx:]))
    parameterGenerator = newNetwork.parameters()
    next(layer for i,layer in enumerate(parameterGenerator)
         if i==2*layerIdx).data = torch.Tensor(np.identity(identitySize))
    next(parameterGenerator).data = torch.Tensor(np.zeros(identitySize))
    return newNetwork

    return nn.Sequential(*network, nn.Linear(inFeatures, outFeatures), actFunc)

def addNode(network, layerIdx, addNNodes):
    # TODO: Add checks like refusing to go to 0 or negative nodes
    # Added node(s) has weights and biases of 0 and should have no effective change immediately upon addition
    network[layerIdx*2-2].out_features += addNNodes
    network[layerIdx*2].in_features += addNNodes
    for i, parameters in enumerate(network.parameters()):
        if i == layerIdx*2:
            # Increase number of output weights in previous layer
            updatedParameters = torch.Tensor(np.zeros(np.array(parameters.shape) + np.array((addNNodes,0)))) # .shape is annoyingly ordered as (output,input)
            overlapShape = (min(parameters.shape[0], updatedParameters.shape[0]), min(parameters.shape[1], updatedParameters.shape[1])) # Finds overlap between old and new matrices
            updatedParameters.data[:overlapShape[0],:overlapShape[1]] = parameters[:overlapShape[0],:overlapShape[1]]
            parameters.data = updatedParameters
            # Fault: slicing with [:0] apparently just give you an empty iterative :((
            # parameters.data[:(-min(addNNodes,0)),:] = updatedParameters[:(-max(addNNodes,0)),:
        elif i == layerIdx*2+1:
            # Increase number of biases in relevant layer
            updatedParameters = torch.Tensor(np.zeros(np.array(parameters.shape) + addNNodes))
            overlapShape = min(parameters.shape[0], updatedParameters.shape[0])
            updatedParameters[:overlapShape] = parameters[:overlapShape]
            parameters.data = updatedParameters
        elif i == layerIdx*2+2:
            # Increase number if input weights in following layer
            updatedParameters = torch.Tensor(np.zeros(np.array(parameters.shape) + np.array((0,addNNodes)))) # .shape is annoyingly ordered as (output,input)
            overlapShape = (min(parameters.shape[0], updatedParameters.shape[0]), min(parameters.shape[1], updatedParameters.shape[1])) # Finds overlap between old and new matrices
            updatedParameters.data[:overlapShape[0],:overlapShape[1]] = parameters[:overlapShape[0],:overlapShape[1]]
            parameters.data = updatedParameters

### Use 1 activation function
# activationFunc = nn.ReLU
# layerArgs = ((2,3), (3,3), (3,2))
# neuralNetwork = nn.Sequential(*[layerActivation for concatenation in [(nn.Linear(inN, outN), activationFunc()) for (inN, outN) in layerArgs] for layerActivation in concatenation])

### Use potentially multiple activation functions
# layerArgs = ((2,12,nn.ReLU()), (12,12,nn.ReLU()), (12,2,nn.ReLU()))
# neuralNetwork = nn.Sequential(*[layerActivation for concatenation in [(nn.Linear(arg[0], arg[1]), arg[2]) for arg in layerArgs] for layerActivation in concatenation])

# neuralNetwork = addLayer(neuralNetwork, 2)
# addNode(neuralNetwork, 2, 2)

class Learner():

    def fitnessFuncTest(genome,config):
        genID, gen = genome 
        network = neat.nn.FeedForwardNetwork.create(gen, config)
        fitness = -(((network.activate((1,1)) - (np.arange(2)+100))**2).sum())
        # fitness-= len(gen.nodes) + len(gen.connections)/100
        return 

            
    def simulateGenome(self, genome,config,queue,no_graphics=True,
                       simulationSteps=CONFIG_DETAILS["simulationSteps"]):
        _, gen = genome 

        worker_id = queue.get()
        # This is a non-blocking call that only loads the environment.
        if CONFIG_DETAILS["unityEnvironmentFilepath"]:
            fileName = f"{CONFIG_DETAILS['unityEnvironmentFilepath']}\\"\
                       f"{CONFIG_DETAILS['unityEnvironmentName']}"\
                       f"{CONFIG_DETAILS['unityEnvironmentVersion']}.exe"
            env = UnityEnvironment(
                file_name=fileName,
                seed=CONFIG_DETAILS["unitySeed"](), 
                side_channels=[], 
                no_graphics=no_graphics,
                worker_id=worker_id,
                timeout_wait=CONFIG_DETAILS["simulationTimeout"],
            )
        else:
            print("Please start environment")
            env = UnityEnvironment()
            print("Environment found")
        env.reset()

        self.assertBehaviorNames(env)
        behaviorNames = list(env.behavior_specs.keys())
        behaviorName = behaviorNames[0]

        network = neat.nn.FeedForwardNetwork.create(gen, config)
        reward = 0
        for t in range(simulationSteps):
            decisionSteps, other = env.get_steps(behaviorName)
            for id, obs in zip(decisionSteps.agent_id, decisionSteps.obs[0]):
                action = np.array(network.activate(obs)).reshape(1,2)
                env.set_action_for_agent(
                    behaviorName, id, 
                    ActionTuple(np.array(action).reshape(1,2))
                    )
            reward +=sum(decisionSteps.reward)
            env.step()
        env.close()
        queue.put(worker_id)

        # print(f"genome {genID} sucessfully simulated")
        return reward
        
    def fitnessFunc(self, genome,config,queue):
        return self.simulateGenome(genome,config,queue)


    def evaluatePopulation(self, genomes, config, 
                           numWorkers=CONFIG_DETAILS["parallelWorkers"]):
        # print("Evaluating genome IDs: ", end="")
        # for genID, _ in genomes:
        #     print(f"{genID}, ", end="")
        timeNow = datetime.datetime.now()
        print(f"Initialization at: {str(timeNow.hour).zfill(2)}"
              f":{str(timeNow.minute).zfill(2)} :"
              f" {str(timeNow.second).zfill(2)}."
              f"{timeNow.microsecond}")
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        [queue.put(i) for i in range(24)] # Uses ports 0-23
        match CONFIG_DETAILS["processingVersion"]:
            case 1:
                print(f"Using serial processing")
                for genome in genomes:
                    genome[1].fitness = self.fitnessFunc(genome, config,queue)

            case 2:
                print(f"Using list")
                with multiprocessing.Pool(numWorkers) as pool:
                    jobs = [pool.apply_async(
                                self.fitnessFunc, 
                                (genome, config)
                                ) for genome in genomes]
                    
                    for job, (genID, genome) in zip(jobs, genomes,queue):
                        genome.fitness = job.get(timeout=None)
                
            case 3:
                print(f"Using starmap")
                with multiprocessing.Pool(numWorkers) as pool:
                    for genome, fitness in zip(genomes, pool.starmap(
                            self.fitnessFunc, 
                            [(genome, config,queue) for genome in genomes]
                            )):
                        genome[1].fitness = fitness/\
                                            CONFIG_DETAILS["simulationSteps"]


    def run(self, config=NEAT_CONFIG, 
            numGen=CONFIG_DETAILS["numberOfGenerations"]
        ):
        if CONFIG_DETAILS["lastPopulationCheckpoint"]:
            filepath = CONFIG_DETAILS["lastPopulationCheckpoint"]
            with (filepath, "rb") as infile:
                pop = pickle.load(infile)
        else:
            pop = neat.Population(config)
            pop.add_reporter(neat.StdOutReporter(False))

        while pop.generation < numGen:
            #TODO: print time of start
            bestBoi = pop.run(
                self.evaluatePopulation, 
                CONFIG_DETAILS["generationsBeforeSave"]
            )

            outfilePath = f"{CONFIG_DETAILS['resultsFolder']}"\
                          f"\{CONFIG_DETAILS['unityEnvironmentVersion']}"\
                          f"\popcount{CONFIG_DETAILS['populationCount']}_"\
                          f"simlength{CONFIG_DETAILS['simulationSteps']}_"
                            
            outfileName = outfilePath +\
                          f"\generation_{str(pop.generation).zfill(4)}.pkl"
            
            if not os.path.exists(outfilePath):
                os.makedirs(outfilePath)
            with open(outfileName, "wb") as outfile:
                pickle.dump((pop, bestBoi), outfile)

        return pop, bestBoi

    def demonstrateGenome(self, genomeFilePath, config):
        with open(genomeFilePath, "rb") as infile:
            population = pickle.load(infile)
        if isinstance(population, tuple):
            population, genome = population
        elif isinstance(population, neat.population.Population):
            raise NotImplemented

        print("Please start environment")
        env = UnityEnvironment()
        print("Environment found")

        # fileName = f"{CONFIG_DETAILS['unityEnvironmentFilepath']}\\"\
        #             f"{CONFIG_DETAILS['unityEnvironmentName']}"\
        #             f"{CONFIG_DETAILS['unityEnvironmentVersion']}.exe"
        # env = UnityEnvironment(
        #     file_name = fileName,
        #     worker_id = 0,
        #     base_port = None,
        #     seed = CONFIG_DETAILS["unitySeed"](),
        #     no_graphics = False,
        #     no_graphics_monitor = False,
        #     timeout_wait = CONFIG_DETAILS["simulationTimeout"],
        #     additional_args = None,
        #     side_channels = None,
        #     log_folder = None,
        #     num_areas =  1
        # )

        env.reset()
        self.assertBehaviorNames(env)
        behaviorNames = list(env.behavior_specs.keys())
        behaviorName = behaviorNames[0]

        network = neat.nn.FeedForwardNetwork.create(genome, config)

        while True:
            decisionSteps, other = env.get_steps(behaviorName)
            printout = ""
            for id, obs in zip(decisionSteps.agent_id, decisionSteps.obs[0]):
                action = np.array(network.activate(obs)).reshape(1,2)
                printout += f"{obs} -> {action}\n"
                env.set_action_for_agent(
                    behaviorName, 
                    id, 
                    ActionTuple(np.array(action).reshape(1,2))
                )
            print(printout, end="\n\n")
            # time.sleep(10)
            env.step()


    def assertBehaviorNames(self, env):
        assert len(list(env.behavior_specs.keys())) == 1,\
         (f"There is not exactly 1 behaviour in the "
          f"Unity environment: {list(env.behavior_specs.keys())}")


    # Applies basic motion for visual evaluation of physical rules
    def motionTest(self):
        print("Please start environment")
        env = UnityEnvironment()
        print("Environment found")

        env.reset()

        self.assertBehaviorNames(env)
        behaviorNames = list(env.behavior_specs.keys())
        behaviorName = behaviorNames[0]

        motionDuration = 15

        while True:
            decisionSteps, other = env.get_steps(behaviorName)
            T = time.time()
            if T == motionDuration/2:
                # Prevents division by 0
                action = (1,1)
            else:
                action = 2*(T % motionDuration) / motionDuration - 1
                action = (action, action)
            for id, obs in zip(decisionSteps.agent_id, decisionSteps.obs[0]):
                env.set_action_for_agent(
                    behaviorName, 
                    id, 
                    ActionTuple(np.array(action).reshape(1,2))
                )
            env.step()


if __name__ == "__main__":

    learner = Learner()

    multiprocessing.freeze_support()
    # finalGeneration, bestBoi = learner.run()    

    # learner.motionTest()

    learner.demonstrateGenome(
        f"{CONFIG_DETAILS['resultsFolder']}\\"
        f"{CONFIG_DETAILS['unityEnvironmentVersion']}\\"
        "popcount24_simlength300_\\generation_0101.pkl", 
        NEAT_CONFIG,
        )


    # import winsound
    # duration = 1000  # milliseconds
    # freq = 69  # Hz
    # winsound.Beep(freq, duration)
    # print(\a)
    # print(\007)
    # winsound.MessageBeep()


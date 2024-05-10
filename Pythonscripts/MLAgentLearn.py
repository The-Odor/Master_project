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

# cd ..\..\Users\theod
# cd Master_project
# venv\Scripts\activate
# python Pythonscripts\MLAgentLearn.py



# TODO: make config into a file
CONFIG_DETAILS = {
    "unityEnvironmentFilepath": r"C:\Users\theod\Master_project\Bot locomotion.exe",
    "configFilepath": r"C:\Users\theod\Master_project\Pythonscripts\configs\NEATconfig",
    "simulationSteps": 600,
    "unitySeed": 1,
    "PythonSeed": 69420,
    "processingVersion": 3, #1:serial, 2:list-based parallel, 3:starmap-based parallel
    "parallelWorkers": 12,
    "numberOfGenerations": 100,
    "simulationTimeout": 1800, # In seconds, 1800 seconds is half an hour
    "generationsBeforeSave": 1,
    "resultsFolder": r"C:\Users\theod\Master_project\Populations" + "\z"[:1],
    "lastPopulationCheckpoint": r"C:\Users\theod\Master_project\Populations\popcount_24_simlength600_generation_33.pkl",
}
with open(CONFIG_DETAILS["configFilepath"]) as infile:
    for line in infile.readlines():
        if line.startswith("pop_size = "):
            CONFIG_DETAILS["populationCount"] = min(24, int(line[11:]))

doNothingBreakPoint = lambda : 0
import random
random.seed(CONFIG_DETAILS["PythonSeed"])


def addLayer(network, layerIdx, actFunc=nn.ReLU(), bias=True):
    # Layer added is in the form of an identity matrix with bias=0 and should have no effective change immediately upon addition
    # ASSUMPTION: actFunc(actFunc(x)) = actFunc(x), like for ReLU.
    layers = [i for i in network]
    identitySize = network[2*layerIdx-2].out_features
    newNetwork = nn.Sequential(*(layers[:2*layerIdx] + [nn.Linear(identitySize, identitySize, bias=bias), actFunc] + layers[2*layerIdx:]))
    parameterGenerator = newNetwork.parameters()
    next(layer for i,layer in enumerate(parameterGenerator) if i==2*layerIdx).data = torch.Tensor(np.identity(identitySize))
    next(parameterGenerator).data = torch.Tensor(np.zeros(identitySize))
    return newNetwork

    # return nn.Sequential(*network, nn.Linear(inFeatures, outFeatures), actFunc)

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



def fitnessFuncTest(genome,config):
    genID, gen = genome 
    network = neat.nn.FeedForwardNetwork.create(gen, config)
    return -(((network.activate((1,1)) - (np.arange(2)+100))**2).sum())# + len(gen.nodes) + len(gen.connections))

        
def simulateGenome(genome,config,queue,no_graphics=True,simulationSteps=CONFIG_DETAILS["simulationSteps"]):
    genID, gen = genome 

    worker_id = queue.get()
    # This is a non-blocking call that only loads the environment.
    if CONFIG_DETAILS["unityEnvironmentFilepath"]:
        env = UnityEnvironment(
            file_name=CONFIG_DETAILS["unityEnvironmentFilepath"],
            seed=CONFIG_DETAILS["unitySeed"], 
            side_channels=[], 
            no_graphics=no_graphics,
            worker_id=worker_id,
            timeout_wait=CONFIG_DETAILS["simulationTimeout"]
        )
    else:
        print("Please start environment")
        env = UnityEnvironment()
        print("Environment found")
    env.reset()

    behaviorNames = list(env.behavior_specs.keys())
    assert len(behaviorNames) == 1, f"There are more than 1 behaviours: {behaviorNames}"
    behaviorName = behaviorNames[0]

    network = neat.nn.FeedForwardNetwork.create(gen, config)
    reward = 0
    for t in range(simulationSteps):
        decisionSteps, other = env.get_steps(behaviorName)
        # pdb.set_trace()
        for id, obs in zip(decisionSteps.agent_id, decisionSteps.obs[0]):
            # action = np.asarray((1,1))
            # env.set_action_for_agent(behaviorName, id, ActionTuple(action.reshape(1,2)))
            action = np.array(network.activate(obs)).reshape(1,2)
            env.set_action_for_agent(behaviorName, id, ActionTuple(np.array(action).reshape(1,2)))
        reward +=sum(decisionSteps.reward)
        env.step()
    env.close()
    queue.put(worker_id)

    # print(f"genome {genID} sucessfully simulated")
    return reward
    
def fitnessFunc(genome,config,queue):
    return simulateGenome(genome,config,queue)


def evaluatePopulation(genomes, config, numWorkers=CONFIG_DETAILS["parallelWorkers"]):
    # print("Evaluating genome IDs: ", end="")
    # for genID, _ in genomes:
    #     print(f"{genID}, ", end="")
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    [queue.put(i) for i in range(24)] # Uses ports 0-23
    match CONFIG_DETAILS["processingVersion"]:
        case 1:
            print(f"Using serial processing")
            for genome in genomes:
                genome[1].fitness = fitnessFunc(genome,config,queue)

        case 2:
            print(f"Using list")
            with multiprocessing.Pool(numWorkers) as pool:
                jobs = [pool.apply_async(fitnessFunc, (genome,config)) for genome in genomes]
                
                for job, (genID, genome) in zip(jobs, genomes,queue):
                    genome.fitness = job.get(timeout=None)
            
        case 3:
            print(f"Using starmap")
            with multiprocessing.Pool(numWorkers) as pool:
                for genome, fitness in zip(genomes, pool.starmap(fitnessFunc, [(genome, config,queue) for genome in genomes])):
                    genome[1].fitness = fitness/CONFIG_DETAILS["simulationSteps"]


def run(config, numGen=CONFIG_DETAILS["numberOfGenerations"]):
    if CONFIG_DETAILS["lastPopulationCheckpoint"]:
        with open(CONFIG_DETAILS["lastPopulationCheckpoint"], "rb") as infile:
            pop = pickle.load(infile)
    else:
        pop = neat.Population(config)
        pop.add_reporter(neat.StdOutReporter(False))

    while pop.generation < numGen:
        #TODO: print time of start
        bestBoi = pop.run(evaluatePopulation, CONFIG_DETAILS["generationsBeforeSave"])

        outfileName = CONFIG_DETAILS["resultsFolder"] + f"popcount_{CONFIG_DETAILS['populationCount']}_simlength{CONFIG_DETAILS['simulationSteps']}_generation_{pop.generation}.pkl"
        with open(outfileName, "wb") as outfile:
            pickle.dump(pop, outfile)
        #TODO: print time of end

    return pop, bestBoi


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    config = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation, CONFIG_DETAILS["configFilepath"])
    finalGeneration, bestBoi = run(config)    
    # with open(CONFIG_DETAILS["resultsFolder"] + "finalGeneration.pkl", "wb") as outfile:
    #     pickle.dump(finalGeneration, outfile)

    try:
        # import winsound
        # duration = 1000  # milliseconds
        # freq = 69  # Hz
        # winsound.Beep(freq, duration)
        # print(\a)
        # print(\007)
        # winsound.MessageBeep()
        simulateGenome([0,bestBoi],config,no_graphics=False,simulationSteps=50000)
    except:
        pass

    # pdb.set_trace()


    # pool = multiprocessing.Pool(4)
    # print(pool.map(func1, range(5)))
    # print(pool.starmap(func2, zip(range(5), range(5,10))))



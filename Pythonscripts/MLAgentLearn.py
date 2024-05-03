from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
import numpy as np
import torch
import torch.nn as nn
# import torch.optim as optim # May be necessary, haven't used it yet
import pdb
import neat
import multiprocessing

# cd Master_project
# venv\Scripts\activate
# python Pythonscripts\MLAgentLearn.py

doNothingBreakPoint = lambda : 0
import random
random.seed(69420)

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



def fitnessFunc(genome,config):
    # pdb.set_trace()
    # networks = [(gen[0], gen[1], neat.nn.FeedForwardNetwork.create(gen[1], config)) for gen in genomes]
    # for genID, gen, net in networks:
    #     gen.fitness = ((net.activate((1,1)) - np.arange(12))**2).sum() + len(gen.nodes) + len(gen.connections)

    genID, gen = genome 
    network = neat.nn.FeedForwardNetwork.create(gen, config)
    return -(((network.activate((1,1)) - (np.arange(2)+100))**2).sum())# + len(gen.nodes) + len(gen.connections))

        

def evaluatePopulation(genomes, config, numWorkers=12):
    processingVersion = 1

    # 1: serial
    # 2: list-based
    # 3: starmap

    match processingVersion:
        case 1:
            print(f"Using serial processing")
            for genome in genomes:
                genome[1].fitness = fitnessFunc(genome,config)

        case 2:
            print(f"Using list")
            with multiprocessing.Pool(numWorkers) as pool:
                jobs = [pool.apply_async(fitnessFunc, (genome,config)) for genome in genomes]
                
                for job, (genID, genome) in zip(jobs, genomes):
                    genome.fitness = job.get(timeout=None)
            
        case 3:
            print(f"Using starmap")
            with multiprocessing.Pool(numWorkers) as pool:
                for genome, fitness in zip(genomes, pool.starmap(fitnessFunc, [(genome, config) for genome in genomes])):
                    genome[1].fitness = fitness


def run():
    filepath = r"C:\Users\theod\Master_project\Pythonscripts\configs\NEATconfig"
    config = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation, filepath)
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(False))
    bestBoi = pop.run(evaluatePopulation, 100)
    network = neat.nn.FeedForwardNetwork.create(bestBoi, config)

    print('\nOutput:')
    networkInputs  = ((1,1),)
    networkOutputs = ((np.arange(2)+100),)
    for xi, xo in zip(networkInputs, networkOutputs):
        output = network.activate(xi)
        print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # pdb.set_trace()


def func1(x):
    return x**2
def func2(x,y):
    return x*y
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    run()    


    # pool = multiprocessing.Pool(4)
    # print(pool.map(func1, range(5)))
    # print(pool.starmap(func2, zip(range(5), range(5,10))))


# # This is a non-blocking call that only loads the environment.
# print("Please start environment")
# env = UnityEnvironment(file_name=None, seed=1, side_channels=[], no_graphics=False)
# print("Environment found")
# # Start interacting with the environment.
# env.reset()

# behaviorName = list(env.behavior_specs.keys())
# assert len(behaviorName) == 1, f"There are more than 1 behaviour: {behaviorName}"
# # print(f"available behaviours: {type(behavior_names[0])}")

# for t in range(50000):
#     decisionSteps, other = env.get_steps(behaviorName)
#     decisionSteps.obs # = observations, as I've defined them

#     for id in decisionSteps.agent_id:
#         action = np.asarray((1,1))
#         env.set_action_for_agent(behaviorName, id, ActionTuple(action.reshape(1,2)))

#     env.step()
# 


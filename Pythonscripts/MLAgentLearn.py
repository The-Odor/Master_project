from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
import numpy as np
import torch
import torch.nn as nn

# cd Master_project
# venv\Scripts\activate
# python Pythonscripts\MLAgentLearn.py

doNothingBreakPoint = lambda : 0

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

### Relevant imports
# import torch.optim as optim # May be necessary, haven't used it yet

### Use 1 activation function
activationFunc = nn.ReLU
layerArgs = ((2,3), (3,3), (3,2))
neuralNetwork = nn.Sequential(*[layerActivation for concatenation in [(nn.Linear(inN, outN), activationFunc()) for (inN, outN) in layerArgs] for layerActivation in concatenation])

### Use potentially multiple activation functions
# layerArgs = ((2,12,nn.ReLU), (12,12,nn.ReLU), (12,2,nn.ReLU))
# neuralNetwork = nn.Sequential(*[layerActivation for concatenation in [(nn.Linear(arg[0], arg[1]), arg[2]()) for arg in layerArgs] for layerActivation in concatenation])

# neuralNetwork = nn.Sequential(
#     nn.Linear(2,12),
#     nn.ReLU(),
#     nn.ReLU(),
#     nn.ReLU(),
#     nn.ReLU(),  
#     nn.Linear(12,1),
#     nn.ReLU(),
#     nn.ReLU(),
#     nn.ReLU(),
#     nn.ReLU(),  
#     nn.Linear(1,2),
#     nn.ReLU(),
#     nn.ReLU(),
#     nn.ReLU(),
#     nn.ReLU(),  
# )

### Potential method of changing the size of a layer
addNNodes = 15    # How many nodes to add; can be negative i.e. remove node(s)
layerIdx = 1      # What layer to add the node() to
indicesOfLayersOnly = np.arange(len(neuralNetwork))[[isinstance(element, nn.Linear) for element in neuralNetwork]] # Excludes activation layers
assert (layerIdx>0) and (layerIdx<len(indicesOfLayersOnly)), f"You've exceeded the size of the network using index {layerIdx}"
neuralNetwork[indicesOfLayersOnly[layerIdx-1]].out_features += addNNodes   # Adds/removes node from metadata
neuralNetwork[indicesOfLayersOnly[layerIdx]].in_features += addNNodes      # Adds/removes node from metadata

parameterGenerator = neuralNetwork.parameters()       # Generates weights and biases separately. Doesn't generate activation functions

# ASSUMPTION: All weights are 2-dimensional, all biases are 1-dimensional, and they come interchangeably as weights-bias-weights-bias-etc...
previousWeights = next(x for i,x in enumerate(parameterGenerator) if i==2*(layerIdx-1)) # Finds the incoming weights of the layer
updatedPreviousWeights = torch.Tensor(np.zeros(previousWeights.shape + np.asarray((0,addNNodes)))) # Creates a temporary matrix that will be filled in
overlapSizePrevious = (min(previousWeights.shape[0], updatedPreviousWeights.shape[0]), min(previousWeights.shape[1], updatedPreviousWeights.shape[1])) # Finds overlap between old and new matrices
updatedPreviousWeights.data[0:overlapSizePrevious[0], 0:overlapSizePrevious[1]] = previousWeights[0:overlapSizePrevious[0], 0:overlapSizePrevious[1]] # Fills in new matrix
# neuralNetwork[indicesOfLayersOnly[layerIdx-1]].data = updatedPreviousWeights
previousBiases  = next(x for i,x in enumerate(parameterGenerator) if i==(2*layerIdx)-1) # Finds the incoming biases of the layer

### TEMPORARY TESTING; CREATING AN EXPLICIT SIZE CHANGE AND SEEING IF I CAN DO IT FOR A SPECIFIC FUCKING LAYER
layerArgs = ((2,12,nn.ReLU()), (12,12,nn.ReLU()), (12,2,nn.ReLU()))
neuralNetwork = nn.Sequential(*[layerActivation for concatenation in [(nn.Linear(arg[0], arg[1]), arg[2]) for arg in layerArgs] for layerActivation in concatenation])
indicesOfLayersOnly = np.arange(len(neuralNetwork))[[isinstance(element, nn.Linear) for element in neuralNetwork]]
paramList = []
parameterGenerator = neuralNetwork.parameters()
for i, param in enumerate(parameterGenerator):
    paramList.append(param)
updatedParam = []
print(neuralNetwork(torch.Tensor((500e4,1e4))))
addNNodes = 1
for i, param in enumerate(paramList):
    if i == 2:
        updatedParam.append(torch.Tensor(np.zeros(np.array(param.shape) + np.array((addNNodes,0)))+1e-2))
        updatedParam[-1][:param.shape[0], :param.shape[1]] = param[:]
        # updatedParam[-1][:(-max(addNNodes,0)-1),:] = param[:(-min(addNNodes,0)-1),:]

    elif i == 3:
        updatedParam.append(torch.Tensor(np.zeros(np.array(param.shape) + addNNodes)+1e-2))
        updatedParam[-1][:param.shape[0]] = param[:]
    elif i == 4:
        neuralNetwork[2].out_features += 1 
        neuralNetwork[4].in_features += 1
        updatedParam.append(torch.Tensor(np.zeros(np.array(param.shape) + np.array((0,addNNodes)))+1e-2))
        updatedParam[-1][:param.shape[0], :param.shape[1]] = param[:]
    else:
        updatedParam.append(torch.Tensor(np.zeros(np.array(param.shape))+1e-2))
        if len(param.shape)==2:
            updatedParam[-1][:param.shape[0], :param.shape[1]] = param[:]
        elif len(param.shape)==1:
            updatedParam[-1][:param.shape[0]] = param[:]
for i, j in zip(paramList, updatedParam):
    i.data = j
print(neuralNetwork(torch.Tensor((500e4,1e4))))
### TEMPORARY TESTING; CREATING AN EXPLICIT SIZE CHANGE AND SEEING IF I CAN DO IT FOR A SPECIFIC FUCKING LAYER


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

layerArgs = ((2,12,nn.ReLU()), (12,13,nn.ReLU()), (13,2,nn.ReLU()))
neuralNetwork = nn.Sequential(*[layerActivation for concatenation in [(nn.Linear(arg[0], arg[1]), arg[2]) for arg in layerArgs] for layerActivation in concatenation])

inputTensor = torch.Tensor((2,2))
print(neuralNetwork, "output: ", neuralNetwork(inputTensor))
neuralNetwork = addLayer(neuralNetwork, 2)
# print(neuralNetwork(inputTensor))
addNode(neuralNetwork, 2, 2)
print(neuralNetwork, "output: ",  neuralNetwork(inputTensor))

# 
# TODO: Consider making the following paragraph a version of the previous one using a loop, so you don't have to copypaste all the time.
# followingLayer = next(parameterGenerator)
# updatedFollowing = torch.Tensor(np.zeros(*(followingLayer.shape + np.asarray(0,addNNodes)))
# overlapSizeFollowing = (min(followingLayer.shape[0], updatedFollowing.shape[0]), followingLayer.shape[1], updatedFollowing.shape[1])
# updatedFollowing[overlapSizeFollowing] = followingLayer[overlapSizeFollowing]
# 

# Beginning code, that I botched together and then took inspiration from
# next(x for i,x in enumerate(neuralNetwork.parameters()) if i==layerIdx).data = torch.Tensor(np.arange(previousOut*followingIn).reshape(previousOut,followingIn))


doNothingBreakPoint()
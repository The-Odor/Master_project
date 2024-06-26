import sys
sys.path.insert(1, "C:\\Users\\theod\\Master_project\\Pythonscripts")
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from MLAgentLearn import CONFIG_DETAILS
from Learner import Learner
import numpy as np
import configparser
import neat
import random
import pickle


a = 2

CONFIG_DETAILS = configparser.ConfigParser()
CONFIG_DETAILS["DEFAULT"] = {
    "simulationSteps": 600,
    "unitySeed": 5,
    "simulationTimeout": 120,
    "parallelWorkers": 24,
    "processingMode": 1,
    "numberOfGenerations": 101,
    "unityEnvironmentVersion": "VaryingGenomes",
    "populationCount": 360,
    "generationsBeforeSave": 1,
}

CONFIG_DETAILS["DEFAULT"]["exeFilePath"] = (
    "C:\\Users\\theod\\Master_project\\Builds\\0.6\\botLocomotion.exe"
    # CONFIG_DETAILS["buildPath"] + "\\" +
    # CONFIG_DETAILS["unityEnvironmentVersion"] + "\\" +
    # CONFIG_DETAILS["unityEnvironmentName"] + ".exe"
)
CONFIG_DETAILS["DEFAULT"]["resultsfolder"] = (
    "C:\\Users\\theod\\Master_project\\Pythonscripts\\Tests\\Determinism"
    # CONFIG_DETAILS["unityProjectFilepath"] + "\\Populations"
)
CONFIG_DETAILS["DEFAULT"]["populationFolder"] = (
    "C:\\Users\\theod\\Master_project\\Populations\\Tests\\Determinism"
)

CONFIG_DETAILS = CONFIG_DETAILS["DEFAULT"]

class VaryingLearner(Learner):
    def __init__(self):
        # TODO: Change to config format!!!
        self.CONFIG_DETAILS = CONFIG_DETAILS

        config = "C:\\Users\\theod\\Master_project\\Pythonscripts\\Tests\\Determinism\\NEATconfig"
        self.NEAT_CONFIG = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config,
        )



    def simulateGenome(self,genome,config,env):
        """
        Rewrites the fitness to increase variance between
        actions, allowing a stricter determinism-check.
        All but the reward calculation are copied from 
        the original function"""
        if isinstance(genome, tuple) and len(genome)==2:
            genID, gen = genome
        elif isinstance(genome, neat.genome.DefaultGenome):
            genID, gen = ("no id", genome)
        else:
            raise TypeError(f"genome of unreadable type: {genome}")
        
        # import pdb
        # pdb.set_trace()


        simulationSteps = self.CONFIG_DETAILS.getint("simulationSteps")



        simulationSteps = 600
        nActionsRemembered = 5
        nActionsIndividual = 12

        lastNActions = np.zeros((nActionsRemembered, nActionsIndividual))

        env.reset()

        self.assertBehaviorNames(env)
        behaviorName = list(env.behavior_specs.keys())[0]
        network = neat.nn.FeedForwardNetwork.create(gen, config)
        reward = 0

        for t in range(simulationSteps):
            decisionSteps, _ = env.get_steps(behaviorName)
            observations = []



            for id, obs in zip(decisionSteps.agent_id, decisionSteps.obs[0]):
                observations.extend(obs)
            action = np.array(network.activate(observations)).reshape(1,12)*2-1
            # Action is sigmoid with range moved from [0,1] to [-1,1]
            
            for i, id in enumerate(decisionSteps.agent_id):

                env.set_action_for_agent(
                    behaviorName, id, 
                    ActionTuple(np.array(action[0][i*2:(i+1)*2]).reshape(1,2))
                    )
            
            # Calculate mse, subtract from reward
            reward += ((lastNActions - action)**2).sum()

            # Shift rewards
            lastNActions[1:] = lastNActions[:-1]
            lastNActions[0] = action
            # Calculate reward
            
            import pdb
            pdb.set_trace()

            env.step()

        env.close()
        reward /= (self.CONFIG_DETAILS.getint("simulationSteps"))
        print(f"Reward given to {genID:<3} as {reward:.20g}")

        return reward



def fetchGenome():
    """
    Fetches a genome from a population and places 
    it so the test can grab it
    """
    with open("C:\\Users\\theod\\Master_project\\Populations\\0.6\\popcount360_simlength600_\\generation_0098.pkl", "rb") as a:
        b = pickle.load(a)
    _, gen = b
    outfile = "C:\\Users\\theod\\Master_project\\Pythonscripts\\Tests\\Determinism\\testGenome.pkl"
    with open(outfile, "wb") as outfile:
        pickle.dump(gen, outfile)



def testDeterminism(genome):
    """
    Tests an environment for determinism by running a genome
    n times through, compiling the actions sequentially into
    a list, and comparing the lists of different runs
    
    Parameters
    ----------
    genome : neat.DefaultGenome
        genome used to interact with environment. More complicated 
        behaviour gives a more rigorous test

    Returns
    -------
    TBD
    Alt1: MSE between sets of actions
    Alt2: Bool of difference existing
    Alt3: Throw error if difference existing
    Alt4: List of differences in values in actions
    Alt5: List containing only indices where difference is found
    Alt6: (Amount of aligned actions, amount of misaligned, total)
    """

    # TODO: Freeze NEAT-configuration
    # Alt1: Pickle a configuration and just keep using it
    #   Problem: Will that be reliable as development continues?
    #            Potentially so, tbh, I only need something that actions...
    # Alt2: Don't freeze it, just refer directly to the common file
    #   Problem: The NEAT-config file WILL change, which will make this
    #            test most likely break
    # Alt3: Save a separate NEAT-config file explicitly for this test
    #   Problem: While flexible and readable, may become cluttered if
    #            many tests require individual config files

    # Has to be Alt3, right? Readability and flexibility trumps...
    # I just have to create a sub-folder and keep it legible. 
    # Name each config file after the test it's for? Yes.
    seed = 5
    nTests = 2
    simulationSteps = 600


    with open(genome, "rb") as infile:
        genome = pickle.load(infile)


    config = "C:\\Users\\theod\\Master_project\\Pythonscripts\\Tests\\Determinism\\NEATconfig"
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config,
    )


    random.seed = seed

    actionSets = []
    for i in range(nTests):
        actionSets.append([])
        env = UnityEnvironment(
            file_name=CONFIG_DETAILS["exeFilePath"],
            seed=seed,
            side_channels=[],
            no_graphics=True,
            worker_id=24, # Base uses 0-23, 24 should not impede base runs
        )

        env.reset()
        
        # Assumes there is only one behaviour and grabs it
        behaviorName = list(env.behavior_specs.keys())[0]

        network = neat.nn.FeedForwardNetwork.create(genome,config)

        for t in range(simulationSteps):
            decisionSteps, _ = env.get_steps(behaviorName)

            observations = []
            for id, obs in zip(decisionSteps.agent_id, decisionSteps.obs[0]):
                observations.extend(obs)
            action = np.array(network.activate(observations)).reshape(1,12)*2-1

            actionSets[-1].append(action)

            for i, id in enumerate(decisionSteps.agent_id):

                env.set_action_for_agent(
                    behaviorName, id, 
                    ActionTuple(np.array(action[0][i*2:(i+1)*2]).reshape(1,2))
                    )
                
            
            env.step()

        env.close()

        

        
    for i, j in zip(actionSets[0], actionSets[1]):
        print(i, j, sum(i-j))


if __name__ == "__main__" :
    # import multiprocessing as mp
    # mp.freeze_support()

    varLearner = VaryingLearner()
    varLearner.run()
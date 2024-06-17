# print("MLAgentLearn.py RELOADED FROM SCRATCH AT THIS VERY MOMENT")
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim # May be necessary, haven't used it yet
import pdb
import neat
import multiprocessing as mp
# import pathos.multiprocessing as pmp
import pickle
import random
import time
import datetime
import os
from visualize import draw_net


# cd C:\Users\theod\Master_project
# venv\Scripts\activate
# python Pythonscripts\MLAgentLearn.py


# TODO: make config into a file
CONFIG_DETAILS = {
    "unityEnvironmentFilepath": r"C:\Users\theod\Master_project",
    "unityEnvironmentName": "Bot locomotion",
    "unityEnvironmentVersion": ".0.5",
    # "unityEnvironmentVersion": ".test",
    "configFilepath": r"C:\Users\theod\Master_project"
                      r"\Pythonscripts\configs\NEATconfig",
    "simulationSteps": 40*20, # 20 steps per second
    "unitySeed": lambda : 5,#random.randint(1, 1000),  # Is called
    "PythonSeed": lambda : 5,#random.randint(1, 1000), # Is called
    "processingMode": 3, #serial, list-based parallel, starmap-based parallel
    "runMode": 1, # Train, Demonstrate genome, Demonstrate physics, make PDF
    "parallelWorkers": 12,
    "numberOfGenerations": 101,
    "simulationTimeout": 120, # In seconds
    "generationsBeforeSave": 1,
    "resultsFolder": r"C:\Users\theod\Master_project\Populations",
    "useSpecificCheckpoint": None, # Will overwrite natural checkpoint

}

with open(CONFIG_DETAILS["configFilepath"]) as infile:
    for line in infile.readlines():
        if line.startswith("pop_size = "):
            CONFIG_DETAILS["populationCount"] = int(line[11:])
            CONFIG_DETAILS["parallelWorkers"] = min(24, int(line[11:]))

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


class Learner():
    def __init__(self, config_details):
        # self.CONFIG_DETAILS = config_details # BREAKS MULTIPROCESSING
        pass

    def fitnessFuncTest(self,genome,config):
        _, gen = genome 
        network = neat.nn.FeedForwardNetwork.create(gen, config)

        mode = 1
        if mode == 1:
            fitness = -(((network.activate((1,1)) - (np.arange(2)/2))**2).sum())
        elif mode == 2:
            fitness = -(((network.activate([1,]*72) - (np.arange(12)/12))**2).sum())

        return fitness


    def simulateGenome(self, genome,config,env,simulationSteps=None):
        if simulationSteps is None:
            simulationSteps = CONFIG_DETAILS["simulationSteps"]
        if isinstance(genome, tuple) and len(genome)==2:
            genID, gen = genome
        elif isinstance(genome, neat.genome.DefaultGenome):
            genID, gen = ("no id", genome)
        else:
            raise TypeError(f"genome of unreadable type: {genome}")

        env.reset()

        self.assertBehaviorNames(env)
        behaviorNames = list(env.behavior_specs.keys())
        behaviorName = behaviorNames[0]

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
            
            reward +=sum(decisionSteps.reward) / len(decisionSteps.reward)
            if reward <= -1000: # Large negative numbers means disqualification
                reward -= 1000 * (simulationSteps - t)
                break

            # print(action)

            env.step()
        env.close()
        reward /= (CONFIG_DETAILS["simulationSteps"])
        if reward > 0:
            print(f"Reward given to {genID:<3} as {reward:.2g}; last instance was "
                f"({', '.join(str(round(i,2)) for i in decisionSteps.reward)})")
        else:
            print(f"Reward given to {genID:<3} as {reward:.2g}; "
                  "last instance was DISQUALIFIED")

        # print(f"genome {genID} sucessfully simulated")
        return reward
        
    def fitnessFunc(self,genome,config,queue):
        # return self.fitnessFuncTest(genome,config)

        worker_id = queue.get()
    
        if CONFIG_DETAILS["unityEnvironmentFilepath"]:
            fileName = f"{CONFIG_DETAILS['unityEnvironmentFilepath']}\\"\
                       f"{CONFIG_DETAILS['unityEnvironmentName']}"\
                       f"{CONFIG_DETAILS['unityEnvironmentVersion']}.exe"
            env = UnityEnvironment(
                file_name=fileName,
                seed=CONFIG_DETAILS["unitySeed"](), 
                side_channels=[], 
                no_graphics=True,
                worker_id=worker_id,
                timeout_wait=CONFIG_DETAILS["simulationTimeout"],
            )
        else:
            print("Please start environment")
            env = UnityEnvironment(seed=CONFIG_DETAILS["unitySeed"]())
            print("Environment found")

        env.reset()
        fitnessResult = self.simulateGenome(genome,config,env)

        queue.put(worker_id)
        return fitnessResult


    def evaluatePopulation(self,genomes,config,numWorkers=None):

        if numWorkers is None:
            numWorkers = CONFIG_DETAILS["parallelWorkers"]

        timeNow = datetime.datetime.now()
        print(f"Initialization at: {str(timeNow.hour).zfill(2)}"
              f":{str(timeNow.minute).zfill(2)} :"
              f" {str(timeNow.second).zfill(2)}."
              f"{timeNow.microsecond}")

        manager = mp.Manager()
        queue = manager.Queue()
        [queue.put(i) for i in range(1,25)] # Uses ports 1-24 (0 is for editor)
        
        case = CONFIG_DETAILS["processingMode"]
        #match CONFIG_DETAILS["processingMode"]:
        #case 1:
        if case == 1:
            print(f"Using serial processing")
            for genome in genomes:
                genome[1].fitness = self.fitnessFunc(genome, config,queue)

            #case 2:
        elif case == 2:
            print(f"Using list")
            with mp.Pool(numWorkers) as pool:
                jobs = [pool.apply_async(
                            self.fitnessFunc, 
                            (genome, config, queue)
                            ) for genome in genomes]
                    
                for job, (genID, genome) in zip(jobs, genomes,queue):
                    genome.fitness = job.get(timeout=None)
            #case 3:
        elif case == 3:
            print(f"Using starmap")
            with mp.Pool(numWorkers) as pool:
                for genome, fitness in zip(genomes, pool.starmap(
                        self.fitnessFunc, 
                        [(genome, config,queue) for genome in genomes]
                        )):
                    genome[1].fitness = fitness

        elif case == 4:
            print(f"using built-in")
            worker = neat.parallel.ParallelEvaluator(numWorkers, self.fitnessFunc, timeout=None)
            worker.evaluate(genomes, config)



    def run(self,config=NEAT_CONFIG, useCheckpoint=False,numGen=None):
        if numGen is None:
            numGen = CONFIG_DETAILS["numberOfGenerations"]

        pop = None

        if CONFIG_DETAILS["useSpecificCheckpoint"]:
            filepath = CONFIG_DETAILS["useSpecificCheckpoint"]
            with (filepath, "rb") as infile:
                pop = pickle.load(infile)

        elif useCheckpoint:
            outfilePath = f"{CONFIG_DETAILS['resultsFolder']}"\
                        f"\{CONFIG_DETAILS['unityEnvironmentVersion']}"\
                        f"\popcount{CONFIG_DETAILS['populationCount']}_"\
                        f"simlength{CONFIG_DETAILS['simulationSteps']}_"
            
            if not os.path.exists(outfilePath):
                os.makedirs(outfilePath)

            pop, _ = self.findGeneration()

        if not pop:
            pop = neat.Population(config)
            pop.add_reporter(neat.StdOutReporter(False))
        
        if numGen == pop.generation:
            raise Exception("Training started, but generation limit already "
                            f"met or exceeded: {pop.generation} >= {numGen}")



        while pop.generation < numGen:

            bestBoi = pop.run(
                self.evaluatePopulation, 
                CONFIG_DETAILS["generationsBeforeSave"]
            )

            outfileName = outfilePath +\
                          f"\generation_{str(pop.generation).zfill(4)}.pkl"
            with open(outfileName, "wb") as outfile:
                pickle.dump((pop, bestBoi), outfile)

        return pop, bestBoi
    
    def findGeneration(self, specificGeneration = None, config=NEAT_CONFIG):
        populationFolder = f"{CONFIG_DETAILS['resultsFolder']}"\
                    f"\{CONFIG_DETAILS['unityEnvironmentVersion']}"\
                    f"\popcount{CONFIG_DETAILS['populationCount']}_"\
                    f"simlength{CONFIG_DETAILS['simulationSteps']}_"
        if not os.path.exists(populationFolder):
            os.makedirs(populationFolder)
        fullFileList = os.listdir(populationFolder)
        trueFileList = []
        for generationFile in fullFileList:
            generationFolderFile = os.path.join(populationFolder, generationFile)
            if os.path.isfile(generationFolderFile):
                trueFileList.append(generationFolderFile)

        Generation = (None, 0)

        for generationFile in trueFileList:
            genNumber = int(generationFile[-8:-4]) # generation_[xxxx].pkl
            if (genNumber > Generation[1] and specificGeneration is None)\
            or (genNumber == specificGeneration):
                Generation = (generationFile, genNumber)
                        
        if Generation[0]:
            with open(Generation[0], "rb") as infile: 
                pop, bestSpecimen = pickle.load(infile)
                print(f"Loaded generation from {Generation[0]}\n"
                      f"Winner genome has complexity {bestSpecimen.size()}")
                return pop, bestSpecimen
        else:
            pop = neat.Population(config)
            pop.add_reporter(neat.StdOutReporter(False))

            return (pop, None)



    def demonstrateGenome(self,genome=None,config=NEAT_CONFIG):
        if isinstance(genome, str):
            # Given as filepath
            with open(genome, "rb") as infile:
                population = pickle.load(infile)
            if isinstance(population, tuple):
                population, genome = population
            elif isinstance(population, neat.population.Population):
                raise NotImplemented
        elif isinstance(genome, neat.genome.DefaultGenome):
            # Given directly
            pass
        elif genome is None:
            _, genome = self.findGeneration()
        elif isinstance(genome, int):
            _, genome = self.findGeneration(specificGeneration=genome)
        else:
            pdb.set_trace()
            raise TypeError("Genome not of demonstrable datatype")

        # print("Please start environment")
        if CONFIG_DETAILS["unityEnvironmentFilepath"]:
            fileName = f"{CONFIG_DETAILS['unityEnvironmentFilepath']}\\"\
                       f"{CONFIG_DETAILS['unityEnvironmentName']}"\
                       f"{CONFIG_DETAILS['unityEnvironmentVersion']}.exe"
        env = UnityEnvironment(
            file_name=fileName,  
            worker_id=0,
            seed=CONFIG_DETAILS["unitySeed"]()
            )
        # print("Environment found")



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

        self.simulateGenome(genome,config,env)


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

    def makePDF(self, genome=None):
        if genome is None:
            _, genome = self.findGeneration()
        draw_net(NEAT_CONFIG, genome, True)


if __name__ == "__main__":
    mp.freeze_support()

    learner = Learner(CONFIG_DETAILS)

    case = CONFIG_DETAILS["runMode"]
    #match learner.CONFIG_DETAILS["runMode"]:
    if case == 1:
        # case 1: 
        # Trains new generations
        finalGeneration, bestBoi = learner.run(useCheckpoint=True)    
    elif case == 2:
        # case 2:
        # Demonstrates Genomes (hence name, lol)
        learner.demonstrateGenome(learner.findGeneration()[1])
        # learner.demonstrateGenome(learner.findGeneration(specificGeneration=69)[1])
    elif case == 3:
        # case 3:
        # Demonstrates simple motion in Unity editor
        learner.motionTest()
    elif case == 4:
        # case 4: 
        # creates a PDF of given genome (defaults to best of latest generation)
        learner.makePDF(genome=None)


    # import winsound
    # duration = 1000  # milliseconds
    # freq = 69  # Hz
    # winsound.Beep(freq, duration)
    # print(\a)
    # print(\007)
    # winsound.MessageBeep()


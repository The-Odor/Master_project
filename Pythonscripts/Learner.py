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

class Learner():
    def __init__(self, config_details):
        self.CONFIG_DETAILS = config_details
        self.NEAT_CONFIG = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.CONFIG_DETAILS["configFilepath"]
            )
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
            simulationSteps = self.CONFIG_DETAILS.getint("simulationSteps")
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
        reward /= (self.CONFIG_DETAILS.getint("simulationSteps"))
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
    
        # if self.CONFIG_DETAILS.getboolean("exeFilepath"):
        # if self.CONFIG_DETAILS.has_option("Default", "exeFilepath"):
        if "exeFilepath".lower() in list(self.CONFIG_DETAILS.keys()):
            env = UnityEnvironment(
                file_name=self.CONFIG_DETAILS["exeFilepath"],
                seed=self.CONFIG_DETAILS.getint("unitySeed"), 
                side_channels=[], 
                no_graphics=True,
                worker_id=worker_id,
                timeout_wait=self.CONFIG_DETAILS.getint("simulationTimeout"),
            )
        else:
            print("Please start environment")
            env = UnityEnvironment(seed=self.CONFIG_DETAILS.getint("unitySeed"))
            print("Environment found")

        env.reset()
        fitnessResult = self.simulateGenome(genome,config,env)

        queue.put(worker_id)
        return fitnessResult


    def evaluatePopulation(self,genomes,config,numWorkers=None):

        if numWorkers is None:
            numWorkers = self.CONFIG_DETAILS.getint("parallelWorkers")

        timeNow = datetime.datetime.now()
        print(f"Initialization at: {str(timeNow.hour).zfill(2)}"
              f":{str(timeNow.minute).zfill(2)} :"
              f" {str(timeNow.second).zfill(2)}."
              f"{timeNow.microsecond}")

        manager = mp.Manager()
        queue = manager.Queue()
        [queue.put(i) for i in range(1,25)] # Uses ports 1-24 (0 is for editor)
        
        case = self.CONFIG_DETAILS.getint("processingMode")
        #match self.CONFIG_DETAILS["processingMode"]:
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



    def run(self,config=None, useCheckpoint=False,numGen=None):
        if config is None:
            config = self.NEAT_CONFIG

        if numGen is None:
            numGen = self.CONFIG_DETAILS.getint("numberOfGenerations")

        pop = None

        if self.CONFIG_DETAILS.getboolean("useSpecificCheckpoint"):
            filepath = self.CONFIG_DETAILS["useSpecificCheckpoint"]
            with (filepath, "rb") as infile:
                pop = pickle.load(infile)

        outfilePath = self.CONFIG_DETAILS["populationFolder"]

        if not os.path.exists(outfilePath):
            os.makedirs(outfilePath)

        elif useCheckpoint:
            # outfilePath = f"{self.CONFIG_DETAILS['resultsFolder']}"\
            #             f"\{self.CONFIG_DETAILS['unityEnvironmentVersion']}"\
            #             f"\popcount{self.CONFIG_DETAILS['populationCount']}_"\
            #             f"simlength{self.CONFIG_DETAILS['simulationSteps']}"

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
                self.CONFIG_DETAILS.getint("generationsBeforeSave")
            )

            outfileName = outfilePath +\
                          f"\generation_{str(pop.generation).zfill(4)}.pkl"
            with open(outfileName, "wb") as outfile:
                pickle.dump((pop, bestBoi), outfile)



        return pop, bestBoi
    
    def findGeneration(self, specificGeneration = None, config=None):
        if config is None:
            config = self.NEAT_CONFIG

        # populationFolder = f"{self.CONFIG_DETAILS['resultsFolder']}"\
        #             f"\{self.CONFIG_DETAILS['unityEnvironmentVersion']}"\
        #             f"\popcount{self.CONFIG_DETAILS['populationCount']}_"\
        #             f"simlength{self.CONFIG_DETAILS['simulationSteps']}_"
        populationFolder = self.CONFIG_DETAILS["populationFolder"]
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



    def demonstrateGenome(self,genome=None,config=None):
        if config is None:
            config = self.NEAT_CONFIG

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
            raise TypeError("Genome not of demonstrable datatype")

        # print("Please start environment")
        # if self.CONFIG_DETAILS.getboolean("exeFilepath"):
        if "exeFilepath".lower() in list(self.CONFIG_DETAILS.keys()):
            env = UnityEnvironment(
                file_name=self.CONFIG_DETAILS["exeFilepath"],
                worker_id=0,
                seed=self.CONFIG_DETAILS.getint("unitySeed")
                )
        else:
            raise Exception("Executeable not found")
        # print("Environment found")



        # fileName = f"{self.CONFIG_DETAILS['unityProjectFilepath']}\\"\
        #             f"{self.CONFIG_DETAILS['unityEnvironmentName']}"\
        #             f"{self.CONFIG_DETAILS['unityEnvironmentVersion']}.exe"
        # env = UnityEnvironment(
        #     file_name = fileName,
        #     worker_id = 0,
        #     base_port = None,
        #     seed = self.CONFIG_DETAILS.getint("unitySeed"),
        #     no_graphics = False,
        #     no_graphics_monitor = False,
        #     timeout_wait = self.CONFIG_DETAILS.getint("simulationTimeout"),
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
                action = action/abs(action)
                action = (action, action)
            for id, obs in zip(decisionSteps.agent_id, decisionSteps.obs[0]):
                env.set_action_for_agent(
                    behaviorName, 
                    id, 
                    ActionTuple(np.array(action).reshape(1,2))
                )
                print(f"Sent action {action}")
            env.step()

    def makePDF(self, genome=None,config=None):
        if genome is None:
            _, genome = self.findGeneration()
        if config is None:
            config = self.NEAT_CONFIG
        draw_net(config, genome, True)


if __name__ == "__main__":
    # TODO: for gods sake, make some unit tests bro
    pass
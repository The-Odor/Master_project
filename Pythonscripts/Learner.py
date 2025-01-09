# print("MLAgentLearn.py RELOADED FROM SCRATCH AT THIS VERY MOMENT")
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from tqdm import tqdm
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
# import cma

class Learner():
    def __init__(self, config_details, morphologyTrainedOn=None):
        self.CONFIG_DETAILS = config_details
        self.morphologyTrainedOn = morphologyTrainedOn
        if self.CONFIG_DETAILS["fileendingFormat"] == ".x86_64":
            self.osSystem = "unix"
            self.dirSeparator = "/"
        elif self.CONFIG_DETAILS["fileendingFormat"] == ".exe":
            self.osSystem = "windows"
            self.dirSeparator = "\\"
        else:
            raise Exception(f"fileendingFormat in pythonconfig not recognized: {self.CONFIG_DETAILS['fileendingFormat']}")
        if morphologyTrainedOn:
            self.CONFIG_DETAILS["populationFolder"] = f"{self.CONFIG_DETAILS['populationFolder']}{self.dirSeparator}{'_'.join([i[:i.rfind('?team=0')] for i in morphologyTrainedOn])}"
        print(f"Simulation environment fetched from {self.CONFIG_DETAILS['exeFilepath']}")
    
    def switchEnvironment(self, newFilepath):
        self.CONFIG_DETAILS["exeFilepath"] = newFilepath
        print(f"New simulation environment now fetched from {self.CONFIG_DETAILS['exeFilepath']}")

    def rewardAggregation(self, rewards):
        # Reward for a single behavior is equivalent to final distance
        for behavior in rewards:
            rewards[behavior] = rewards[behavior][-1]

        # Different behaviors undergo simple summation
        return sum(rewards.values())

    
    def assertBehaviorNames(self, env):
        assert len(list(env.behavior_specs.keys())) != 0,\
         (f"There are no behaviours in the "
          f"Unity environment: {list(env.behavior_specs.keys())}")

    def simulateGenome(self):
        # TODO: Put actual Unity control structure into own function.
        #       Is it beneficial? Does the requirement for a loop 
        #       make this common function useless?
        raise NotImplemented()
    
    def getBehaviors(self):
        # Trick: Open up an instance of Unity, extract the names,
        #        then close it again, for automatic dimensionality!
        env = UnityEnvironment(
            file_name=self.CONFIG_DETAILS["exeFilepath"],
            no_graphics=True,
            worker_id=30
        )
        env.reset()
        self.assertBehaviorNames(env)
        behavior_specs = env.behavior_specs
        behaviorNames = sorted(list(behavior_specs.keys()))
        behaviorAgentDict = {}
        nagents = 0
        for behavior in behaviorNames:
            decisionSteps, _ = env.get_steps(behavior)
            njoints = len(decisionSteps.agent_id)
            nagents += njoints
            behaviorAgentDict[behavior] = list(decisionSteps.agent_id)
        nbodies = len(behaviorNames)
        env.close()

        return nagents, nbodies, behaviorAgentDict
    
    def makeJointstr(self, behavior, agentID):
        return behavior + "?agent=" + str(agentID)

    def changemorphologyTrainedOn(self, newmorphologyTrainedOn):
        self.morphologyTrainedOn = newmorphologyTrainedOn


class Learner_NEAT(Learner):
    def __init__(self,config_details, morphologyTrainedOn=None):
        Learner.__init__(self,config_details, morphologyTrainedOn)
        self.timeConst = 1
        self.NEAT_CONFIG = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.CONFIG_DETAILS["configFilepath"]
        )

    def fitnessFuncTest(self,genome,config):
        _, gen = genome 
        network = self.generateNeuralNet(gen, config)

        mode = 1
        if mode == 1:
            fitness = -(((network.activate((1,1)) - (np.arange(2)/2))**2).sum())
        elif mode == 2:
            fitness = -(((network.activate([1,]*72) - (np.arange(12)/12))**2).sum())

        return fitness

    def simulateGenome(self, genome,env,config=None,simulationSteps=None,morphologiesToSimulate=None):
        if simulationSteps is None:
            simulationSteps = self.CONFIG_DETAILS.getint("simulationSteps")
        if (isinstance(genome, tuple) or isinstance(genome, list)) and len(genome)==2:
            genID, gen = genome
        elif isinstance(genome, neat.genome.DefaultGenome):
            genID, gen = ("no id", genome)
        else:
            raise TypeError(f"genome of unreadable type: {genome}")
        if config is None:
            config = self.NEAT_CONFIG


        env.reset()
        network = self.generateNeuralNet(gen, config)

        self.assertBehaviorNames(env)
        behaviorNames = sorted(list(env.behavior_specs.keys()))
        if morphologiesToSimulate is None:
            morphologiesToSimulate = behaviorNames
        else:
            morphologiesToSimulate = morphologiesToSimulate

        reward = {behavior:[] for behavior in morphologiesToSimulate}
        # reward = {behavior+"_v1?team=0":[] for behavior in morphologyTrainedOn}

        rewardFactor = (self.CONFIG_DETAILS.getint("simulationSteps"))

        actionDict = {}
        for behavior in behaviorNames:
            decisionSteps, _ = env.get_steps(behavior)
            for id in decisionSteps.agent_id:

                actionDict[self.makeJointstr(behavior, id)] = 0


        for step in range(simulationSteps):
            for behaviorName in behaviorNames:
                if behaviorName not in morphologiesToSimulate:
                    continue
                decisionSteps, _ = env.get_steps(behaviorName)
                # observations = []
                # for id, obs in zip(decisionSteps.agent_id, decisionSteps.obs[0]):
                #     observations.extend(obs)
                # action = np.array(network.activate(observations)).reshape(1,12)
                # # Action is sigmoid with range moved from [0,1] to [-1,1]
                
                # for i, id in enumerate(decisionSteps.agent_id):

                #     env.set_action_for_agent(
                #         behaviorName, id, 
                #         ActionTuple(np.array(action[0][i*2:(i+1)*2]).reshape(1,2))
                #         )
                
                for id, obs in zip(decisionSteps.agent_id, decisionSteps.obs[0]):
                    jointstr = self.makeJointstr(behaviorName, id)
                    action = self.readNeuralNet(network, obs, actionDict[jointstr])
                    actionTu = ActionTuple(
                        np.array((action, action)).reshape(1,2)
                        )
                    env.set_action_for_agent(
                        behaviorName, id,
                        actionTu
                    )
                    actionDict[jointstr] = action



                reward[behaviorName].append(sum(decisionSteps.reward) / len(decisionSteps.reward) / rewardFactor)
                if reward[behaviorName][-1] <= -1000: # Large negative numbers means disqualification
                    reward[behaviorName][-1] -= 1000 * (simulationSteps - step)
                    break

            env.step()
        env.close()

        # if reward > 0:
        #     print(f"Reward given to {genID:<3} as {reward:.2g}; last instance was "
        #         f"({', '.join(str(round(i,2)) for i in decisionSteps.reward)})")
        # else:
        #     print(f"Reward given to {genID:<3} as {reward:.2g}; "
        #           "last instance was DISQUALIFIED")

        return self.rewardAggregation(reward)
    

    def fitnessFunc(self,genome,queue,config=None,morphologiesToSimulate=None):
        # return self.fitnessFuncTest(genome,config)
        # return self.approximateSineFunc(genome, config)
        if config is None:
            config = self.NEAT_CONFIG

        if self.CONFIG_DETAILS.getint("processingMode") in (2,3):
            worker_id = queue.get()
        else:
            worker_id = 1
    
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
        reward = self.simulateGenome(genome,env,config,morphologiesToSimulate=morphologiesToSimulate)

        if self.CONFIG_DETAILS.getint("processingMode") in (2,3):
            queue.put(worker_id)

        # if reward > 0:
        #     print(f"{reward:.1e}", end=", ")
        # else:
        #     print(f"DISC", end=", ")

        # print(f"genome {genID} sucessfully simulated")

        return reward

    def evaluatePopulation(self,genomes,config,numWorkers=None,training=True):

        if numWorkers is None:
            numWorkers = self.CONFIG_DETAILS.getint("parallelWorkers")

        timeNow = datetime.datetime.now()
        print(f"Initialization at: {str(timeNow.hour).zfill(2)}"
              f":{str(timeNow.minute).zfill(2)} :"
              f" {str(timeNow.second).zfill(2)}."
              f"{timeNow.microsecond}")

        if self.CONFIG_DETAILS.getint("processingMode") in (2,3):
            manager = mp.Manager()
            queue = manager.Queue()
            [queue.put(i) for i in range(1,25)] # Uses ports 1-24 (0 is for editor)
        else:
            queue = None

        evaluationDict = {}
        
        case = self.CONFIG_DETAILS.getint("processingMode")
        #match self.CONFIG_DETAILS["processingMode"]:
        #case 1:
        if case == 1:
            print(f"Using serial processing")
            for genome in tqdm(genomes, total=len(genomes)-1, bar_format="{l_bar}{bar:20}{r_bar}"):
                fitness = self.fitnessFunc(genome,queue,config)
                if training:
                    # fitness = fitness[self.morphologyTrainedOn[0]]
                    # if len(fitness) > 1:
                    #     raise Exception("Training on more than one morphology, fitness assignment not made for this (should be easy fix if you need it)")
                    genome[1].fitness = fitness[self.morphologyTrainedOn[0]]#+"_v1?team=0"]
                else:
                    for morph in fitness:
                        if morph in evaluationDict:
                            evaluationDict[morph].append(fitness[morph])
                        else:
                            evaluationDict[morph] = [fitness[morph]]


            #case 2:
        elif case == 2:
            print(f"Using list")
            with mp.Pool(numWorkers) as pool:
                jobs = [pool.apply_async(
                            self.fitnessFunc, 
                            (genome, queue, config)
                            ) for genome in genomes]
                    
                for job, (genID, genome) in zip(jobs, genomes,queue):
                    genome.fitness, _ = job.get(timeout=None)
            #case 3:
        elif case == 3:
            print(f"Using imap with {numWorkers} parallel processes")
            print("Rewards given: ", end="")
            # self.SimulateGenome prints reward as print(f"{reward:.1e}", end=", ")
            with mp.Pool(numWorkers) as pool:
                for genome, (fitness, morph) in zip(genomes, tqdm(pool.imap(
                        self.fitnessFuncMapper, 
                        [(genome,queue,config) for genome in genomes]
                        ), total=len(genomes)-1, bar_format="{l_bar}{bar:20}{r_bar}")):
                    # The total has to be reduced by 1... for _some_ reason....
                    if training:
                        genome[1].fitness = fitness[self.morphologyTrainedOn[0]]
                    else:
                        if morph in evaluationDict:
                            evaluationDict[morph].append(fitness)
                        else:
                            evaluationDict[morph] = [fitness]




        elif case == 4:
            print(f"using built-in")
            worker = neat.parallel.ParallelEvaluator(numWorkers, self.fitnessFunc, timeout=None)
            worker.evaluate(genomes, config)

        # print() # End the rewards printing with \n

        return evaluationDict

    def fitnessFuncMapper(self, arg):
        """
        Wraps self.fitnessFunc to allow single-argument Pool.imap
        None is returned to take the place of morphologies-simulated
        list during non-training for evaluation-purposes"""
        return self.fitnessFunc(*arg), None

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
            pop.add_reporter(neat.checkpoint.Checkpointer(
                generation_interval=1,
                filename_prefix=self.CONFIG_DETAILS["populationFolder"]+f"{self.dirSeparator}generation_",
                ))
        
        if numGen == pop.generation:
            raise Exception("Training started, but generation limit already "
                            f"met or exceeded: {pop.generation} >= {numGen}")


        if pop.generation == numGen-1:
            # NEATpython does not save the last generation for some reason,
            # so to avoid running through every single last generation
            # every damned time, we set the number of generations to be
            # 1 more than wanted and use this check
            with open(outfilePath + f"{self.dirSeparator}bestSpecimen", "rb") as infile:
                bestBoi = pickle.load(infile)

        else:
            while pop.generation < numGen:

                bestBoi = pop.run(
                    self.evaluatePopulation, 
                    self.CONFIG_DETAILS.getint("generationsBeforeSave")
                )

                # outfileName = outfilePath +\
                #               f"\generation_{str(pop.generation).zfill(4)}.pkl"
                # with open(outfileName, "wb") as outfile:
                #     pickle.dump((pop, bestBoi), outfile)
                #     print(f"Saved generation to {outfileName}")

                outfileName = outfilePath + f"{self.dirSeparator}bestSpecimen"
                with open(outfileName, "wb") as outfile:
                    pickle.dump(bestBoi, outfile)



        return pop, bestBoi
    
    def seedingFunction(self, pop):
        # Placeholder function
        return pop
    
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
            if os.path.isfile(generationFolderFile) and generationFile != "bestSpecimen":
                trueFileList.append(generationFolderFile)

        Generation = (None, 0)

        for generationFile in trueFileList:
            # genNumber = int(generationFile[-8:-4]) # generation_[xxxx].pkl
            genNumberIndent = generationFile.rfind("generation_")+11
            genNumber = int(generationFile[genNumberIndent:])
            if (genNumber > Generation[1] and specificGeneration is None)\
            or (genNumber == specificGeneration):
                Generation = (generationFile, genNumber)
                        
        if Generation[0]:
            # with open(Generation[0], "rb") as infile: 
            #     pop = pickle.load(infile)
            #     print(f"Loaded generation from {Generation[0]}\n")
            print(f"Restoring checkpoint from {Generation[0]}")
            pop =  neat.checkpoint.Checkpointer.restore_checkpoint(Generation[0])

            # Overwrites old config details
            pop = neat.population.Population(
                self.NEAT_CONFIG,
                (pop.population, pop.species, pop.generation)
                )
        
            pop.add_reporter(neat.StdOutReporter(False))
            pop.add_reporter(neat.checkpoint.Checkpointer(
                generation_interval=1,
                filename_prefix=self.CONFIG_DETAILS["populationFolder"]+f"{self.dirSeparator}generation_",
                ))
            with open(self.CONFIG_DETAILS["populationFolder"] + r"\bestSpecimen", "rb") as infile:
               bestSpecimen = pickle.load(infile)
            return pop, bestSpecimen


                # return (pop, bestSpecimen)
        else:
            pop = neat.Population(config)
            pop.add_reporter(neat.StdOutReporter(False))
            pop.add_reporter(neat.checkpoint.Checkpointer(
                generation_interval=1,
                filename_prefix=self.CONFIG_DETAILS["populationFolder"]+f"{self.dirSeparator}generation_",
                ))

            return self.seedingFunction(pop), None

    def demonstrateGenome(self,genome=None,config=None):
        # raise NotImplemented("top genome no longer saved\n I am sad")
        if config is None:
            config = self.NEAT_CONFIG
        if genome is None:
            _, genome = self.findGeneration()
            # genome = self.CONFIG_DETAILS["populationFolder"] + f"{self.dirSeparator}bestSpecimen"

        if isinstance(genome, str):
            # Given as filepath
            with open(genome, "rb") as infile:
                loadedFile = pickle.load(infile)
            if isinstance(loadedFile, tuple):
                _, genome = loadedFile
            elif isinstance(loadedFile, neat.genome.DefaultGenome):
                genome = loadedFile
            elif isinstance(loadedFile, neat.population.Population):
                raise NotImplemented
            else:
                raise TypeError(f"File not readable: {genome}")
        elif isinstance(genome, neat.genome.DefaultGenome):
            # Given directly
            pass
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

        reward = self.simulateGenome(genome,env,config)

        if reward > 0:
            print(f"Reward finalized as: {reward}")
        else:
            print(f"DISC")

    def motionTest(self):
        # Applies basic motion for visual evaluation of physical rules
        # print("Please start environment")
        # env = UnityEnvironment()
        # print("Environment found")
        env = UnityEnvironment(
            file_name=self.CONFIG_DETAILS["exeFilepath"],
            seed=self.CONFIG_DETAILS.getint("unitySeed"), 
            side_channels=[], 
            no_graphics=False,
            worker_id=1,
            timeout_wait=self.CONFIG_DETAILS.getint("simulationTimeout"),
        )


        env.reset()

        self.assertBehaviorNames(env)
        behaviorNames = sorted(list(env.behavior_specs.keys()))
        # behaviorName = behaviorNames[0]

        motionDuration = 15
        actionScale = 45

        while True:
            for behaviorName in behaviorNames:
                decisionSteps, other = env.get_steps(behaviorName)
                T = time.time()
                if T == motionDuration/2:
                    # Prevents division by 0
                    action = (1,1)
                else:
                    action = 2*(T % motionDuration) / motionDuration - 1
                    action = action/abs(action) * actionScale
                    action = (action, action)
                for id, obs in zip(decisionSteps.agent_id, decisionSteps.obs[0]):
                    env.set_action_for_agent(
                        behaviorName, 
                        id, 
                        ActionTuple(np.array(action).reshape(1,2))
                    )
                    # print(f"Sent action {action}")
            env.step()

    def makePDF(self, genome=None,config=None):
        # raise NotImplemented("top genome no longer saved\n I am sad")
        if genome is None:
            _, genome = self.findGeneration()
            # genome = self.CONFIG_DETAILS["populationFolder"] + f"{self.dirSeparator}bestSpecimen"
            # with open(genome, "rb") as infile:
            #     genome = pickle.load(infile)
        if config is None:
            config = self.NEAT_CONFIG
        draw_net(config, genome, True)

    def generateNeuralNet(self, gen, config):
        mode = 3

        if mode == 1:
            return neat.nn.FeedForward.create(gen, config)

        if mode == 2:
            return neat.nn.RecurrentNetwork.create(gen, config)

        if mode == 3:
            timeConst = 1
            return neat.ctrnn.CTRNN.create(gen, config, timeConst)

    def readNeuralNet(self, network, obs, prevAction=None):
        mode = 3

        if mode == 1:
            return network.activate(obs)

        if mode == 2:
            return network.activate(np.append(np.array(obs), prevAction))

        if mode == 3:
            return network.advance(
                np.append(obs, prevAction), 
                self.timeConst, 
                self.timeConst,
                )
        



class Learner_CMA(Learner):
    def __init__(self,config_details):
        Learner.__init__(self,config_details)
        # Magical numbers to map self.cmaArgs from optimal CMA-range (0-10)*
        # onto optimal joint range (parameter-dependent, see comments)
        # *https://cma-es.github.io/cmaes_sourcecode_page.html#practical
        # a (shift): (a0-5)*(60/5) = 12*(a0-5)
        # b (amplt): (b0-5)*(60/5) = 12*(b0-5)
        # c (freqc): c0
        # e (phase): e0*(tau/10)
        self.a = lambda x: 24*(x-5) # So it may lock angle to min angle -60 and max angle +60 
        self.b = lambda x: 24*x # scale to max angle +-60. Further expanded due all values moving towards 10
        self.c = lambda x: 8*x # Previously trained models landed frequency 10
        self.d = lambda x: x*(2*np.pi/10) # tau is a full phase

        self.controllerFormat = 3
        # 1: A controller per joint, morphologies share frequencies
        # 2: A controller per morphology
        # 3: Only one controller 

        generation = self.findGeneration()[0]
        if generation is None:
            self.cmaArgs = None
            self.network = None
        else:
            self.cmaArgs = generation.result[0]
            self.network = self.makeControllers()
        


    def train(self):
        nagents, nbodies, _ = self.getBehaviors()
        # Starting parameters for search set as:
        # Constant  =  0
        # Amplitude =  8 
        # Frequency = 10
        # Phase     =  0
        # Frequency is set uniformly for all joints in each morphology

        es, iteration = self.findGeneration()
        if es is None:
            cmaOptions = {
                "bounds": [0,10],
                # "ftarget": np.inf,
                # "maxiter": 69,
            }
            x0ByFormat = {
                1: [5,8,7]*nagents + [5]*nbodies, # Indendent controllers that share frequency
                2: [5,8,7,5]*nbodies, # One controller per body
                3: [5,8,7,5], # One controller, period
            }
            es = cma.CMAEvolutionStrategy(
                x0=x0ByFormat[self.controllerFormat],
                sigma0=2,
                options=cmaOptions,
            )
            iteration = 0

        while True:
            es.optimize(self.simulateGenome, iterations=10)
            iteration+= 10
            # pdb.set_trace()
            outfileName = self.CONFIG_DETAILS["resultsFolder"]
            outfileName+= f"{self.dirSeparator}CMA{self.dirSeparator}format_{self.controllerFormat}{self.dirSeparator}iteration_{iteration}"
            with open(outfileName, "wb") as outfile:
                pickle.dump((es, iteration), outfile)
        print(es.result)

    def demonstrateGenome(self):
        if self.cmaArgs is None:
            raise Exception("No generation file found")

        # pdb.set_trace()
        returnActions = {}
        self.simulateGenome(
            self.cmaArgs, 
            worker_id=2, 
            # instance="editor",
            instance="build", 
            returnActions=returnActions,
        )

        # Printing 
        def printCMAArgs():
            nagents, nbodies, behaviorAgentDict = self.getBehaviors()
            if self.controllerFormat == 1:
                args, freqs = self.cmaArgs[:-nbodies], self.cmaArgs[-nbodies:]
                freqs = {behavior: freq for freq, behavior 
                        in zip(freqs, sorted(list(behaviorAgentDict.keys())))}
                i = 0
                for behavior in sorted(list(behaviorAgentDict.keys())):
                    print(f"{behavior}:")
                    for _ in range(len(behaviorAgentDict[behavior])):
                        a, b, d = args[i*3:i*3+3]
                        c = freqs[behavior]
                        shift = self.a(a)
                        amp   = self.b(b)
                        freq  = self.c(c)
                        phase = self.d(d)
                        print(f"{shift:8.4f} + {amp:8.4f}*sin({freq:8.4f}*x + {phase/(2*np.pi):8.4f}\u03C4)" + 
                            " ; " + f"({a:9.3f}, {b:8.4f}, {c:8.4f}, {d:8.4f})")
                        i += 1
            elif self.controllerFormat == 2:
                for i, behavior in enumerate(sorted(list(behaviorAgentDict.keys()))):
                    print(f"Controller for behavior {behavior}")
                    a, b, c, d = self.cmaArgs[i*4:(i+1)*4]
                    shift = self.a(a)
                    amp   = self.b(b)
                    freq  = self.c(c)
                    phase = self.d(d)
                    print(f"{shift:8.4f} + {amp:8.4f}*sin({freq:8.4f}*x + {phase/(2*np.pi):8.4f}\u03C4)" + 
                        " ; " + f"({a:9.3f}, {b:8.4f}, {c:8.4f}, {d:8.4f})")
            
            
            elif self.controllerFormat == 3:
                a, b, c, d = self.cmaArgs
                shift = self.a(a)
                amp   = self.b(b)
                freq  = self.c(c)
                phase = self.d(d)
                print(f"{shift:8.4f} + {amp:8.4f}*sin({freq:8.4f}*x + {phase/(2*np.pi):8.4f}\u03C4)" + 
                    " ; " + f"({a:9.3f}, {b:8.4f}, {c:8.4f}, {d:8.4f})")

                print("Controller:")


        printCMAArgs()

        # Plotting
        import matplotlib.pyplot as plt
        reducedActionKeys = [key for key in returnActions] # use for individual plotting
        # reducedActionKeys = [key for key in returnActions if key.startswith("queen")]
        figure, axis = plt.subplots(1,2)
        for i in range(2):
            for action in reducedActionKeys:
                if action.startswith("queen") or True:
                    timePoints = [i/50 for i in range(len(returnActions[action][i]))] # 50 steps per second
                    axis[i].plot(timePoints, returnActions[action][i], label=action)
            axis[i].set_xlabel("Time [s]")
            axis[i].set_ylabel("Angle [degrees]")
        axis[0].set_title("Controller signal sent")
        axis[1].set_title("Actual joint angle")
        plt.legend()
        plt.show()
        # pdb.set_trace()

    def makeControllers(self, cmaArgs=None):
        if cmaArgs is None:
            cmaArgs = self.cmaArgs

        _, nbodies, behaviorAgentDict = self.getBehaviors()

        if self.controllerFormat == 1:
            args, freqs = cmaArgs[:-nbodies], cmaArgs[-nbodies:]
            freqs = {behavior: freq for freq, behavior in zip(freqs, behaviorAgentDict.keys())}

            network = {}
            i = 0
            for behavior in behaviorAgentDict.keys():
                for agentID in behaviorAgentDict[behavior]:
                    freq = freqs[behavior]/5 # TODO: Why is this divided by 5
                    
                    controllerFuncString = f"""def controllerFunc(
                            step, vals={list(args[3*i:(i+1)*3])}, freq={freq}
                        ):
                        shift, amp, phase = vals
                        result = self.a(shift) + self.b((amp)
                                 *np.sin(self.c(freq*step) + self.d(phase)))
                        return result
                    """

                    joint = self.makeJointstr(behavior, agentID) 
                    network[joint] = controllerFuncString
                    i+=1

        if self.controllerFormat == 2:
            args = cmaArgs[:]
            network = {}
            i = 0
            for i, behavior in enumerate(sorted(list(behaviorAgentDict.keys()))):
                for agentID in behaviorAgentDict[behavior]:
                    controllerFuncString=f"""def controllerFunc(
                            step, vals={args[4*i:4*(i+1)]}
                        ):
                        shift, amp, freq, phase = vals
                        result = self.a(shift) + (self.b(amp)
                                 *np.sin(self.c(freq*step) + self.d(phase)))
                        return result
                    """
                    joint = self.makeJointstr(behavior, agentID)
                    network[joint] = controllerFuncString
                
        if self.controllerFormat == 3:
            args = cmaArgs[:]
            network = {}
            for behavior in behaviorAgentDict.keys():
                for agentID in behaviorAgentDict[behavior]:
                    controllerFuncString=f"""
                    def controllerFunc(
                            step, vals={args}
                            ):
                        shift, amp, freq, phase = vals
                        result = self.a(shift) + (self.b(amp)
                                 *np.sin(self.c(freq*step) + self.d(phase)))
                        return result
                    """
                    joint = self.makeJointstr(behavior, agentID)
                    network[joint] = controllerFuncString

        return network
    
    def readControllerStrings(self, networkStrings):
        network = {}
        for jointstr in networkStrings:
            localScope = {}
            exec(networkStrings[jointstr], {**globals(), **locals()}, localScope)
            network[jointstr] = localScope[list(localScope.keys())[0]]
        return network

    def simulateGenome(self, cmaArgs=None, worker_id=1, instance=None, returnActions=None,):
        if returnActions is not None:
            if not isinstance(returnActions, dict):
                raise NotImplemented("simulateGenome given a non-dict returnActions argument")
        simulationSteps = self.CONFIG_DETAILS.getint("simulationSteps")


        if instance is None:
            env = UnityEnvironment(
                file_name=self.CONFIG_DETAILS["exeFilepath"],
                seed=self.CONFIG_DETAILS.getint("unitySeed"), 
                side_channels=[], 
                no_graphics=True,
                worker_id=worker_id,
                timeout_wait=self.CONFIG_DETAILS.getint("simulationTimeout"),
            )
        elif instance.lower() == "build":
            env = UnityEnvironment(
                file_name=self.CONFIG_DETAILS["exeFilepath"],
                seed=self.CONFIG_DETAILS.getint("unitySeed"), 
                side_channels=[], 
                no_graphics=False,
                worker_id=worker_id,
                timeout_wait=self.CONFIG_DETAILS.getint("simulationTimeout"),
            )
        elif instance.lower() == "editor":
            print("Please start environment")
            env = UnityEnvironment()
            print("Environment Found")
        else:
            raise Exception("Unity instance argument not recognized")


        env.reset()

        self.assertBehaviorNames(env)
        behaviorNames = sorted(list(env.behavior_specs.keys()))

        allJoints = [behavior + "?agent=" + str(agentID) 
                     for behavior in behaviorNames 
                     for agentID in env.get_steps(behavior)[0].agent_id]

        reward = {behavior:[] for behavior in behaviorNames}

        if returnActions is not None:
            for joint in allJoints:
                returnActions[joint] = ([], [])
                # (sent signal, actual joint angle)

        if cmaArgs is None:
            network = self.network
        else:
            network = self.makeControllers(cmaArgs)

        network = self.readControllerStrings(network)

        for step in range(simulationSteps):
            timeVal = step/50 # number of seconds that have passed

            i = 0
            for behavior in behaviorNames:
                decisionSteps, _ = env.get_steps(behavior)

                for id, obs in zip(decisionSteps.agent_id, decisionSteps.obs[0]):
                    jointstr = self.makeJointstr(behavior, id)
                    action = network[jointstr](timeVal)
                    i+=1

                    if returnActions is not None:
                        returnActions[jointstr][0].append(action)
                        returnActions[jointstr][1].append((obs[2])*180/np.pi)
                        # 3rd observation point should be own rotation

                    action = ActionTuple(
                        np.array((action, action)).reshape(1,2)
                        )
                    env.set_action_for_agent(behavior, id, action)

                reward[behavior].append(sum(decisionSteps.reward) / len(decisionSteps.reward))
                # if reward[behavior] <= -1000: # Large negative numbers means disqualification
                #     reward[behavior] -= 1000 * (simulationSteps - step)
                #     break

            if False:
                print("Reward: ", end="")
                for r in reward.values():
                    print(f"{r[-1]:5.2e}     ", end="")
                print()


            env.step()
        env.close()


        # for behavior in reward:
        #     reward[behavior] /= (self.CONFIG_DETAILS.getint("simulationSteps"))

        # Optimization is a minimization, and negating reward was simpler 
        # than digging through documentation for a maximization option
        return -self.rewardAggregation(reward)
    
    def findGeneration(self):
        generationFolder = f"{self.CONFIG_DETAILS['resultsFolder']}{self.dirSeparator}CMA{self.dirSeparator}format_{self.controllerFormat}"
        fullFileList = os.listdir(generationFolder)
        trueFileList = []
        for generationFile in fullFileList:
            generationFolderFile = os.path.join(generationFolder, generationFile)
            if os.path.isfile(generationFolderFile) and generationFile != "bestSpecimen":
                trueFileList.append(generationFolderFile)

        lastGeneration = 0
        for generationFile in trueFileList:
            genNumberIndent = generationFile.rfind("_")+1
            genNumber = int(generationFile[genNumberIndent:])
            if (genNumber > lastGeneration):
                lastGeneration = genNumber
        
        try:
            with open(f"{generationFolder}{self.dirSeparator}iteration_{lastGeneration}", "rb") as infile:
                iteration = pickle.load(infile)
            return iteration
        except FileNotFoundError:
            return None, None



class Learner_CMA_Modified(Learner_CMA):
    # Modification to solve pickling issue with parallelization
    # TODO: funcs a, b, c, and d raise pickling error
    # TODO: func makeController raises pickling error
    def __init__(self,config_details):
        Learner.__init__(self,config_details)

        self.controllerFormat = 1
        # # 1: A controller per joint, morphologies share frequencies
        # # 2: A controller per morphology
        # # 3: Only one controller 

        generation = self.findGeneration()[0]
        if generation is None:
            self.cmaArgs = None
            self.network = None
        else:
            self.cmaArgs = generation.result[0]
            self.network = self.makeControllers()

        # self.a = lambda x: 24*(x-5) # So it may lock angle to min angle -60 and max angle +60 
        # self.b = lambda x: 24*x # scale to max angle +-60. Further expanded due all values moving towards 10
        # self.c = lambda x: 8*x # Previously trained models landed frequency 10
        # self.d = lambda x: x*(2*np.pi/10) # tau is a full phase

    def a(self, x):
        return 24*(x-5)
    def b(self, x):
        return 24*x
    def c(self, x):
        return 8*x
    def d(self, x):
        return x*(2*np.pi/10)

    def getBehaviors(self):
        nagents = 15
        nbodies = 2
        behaviorAgentDict = {
            'gecko_v1?team=0': [0, 3, 5, 7, 8, 12], 
            'queen_v1?team=0': [1, 2, 4, 6, 9, 10, 11, 13, 14]
        }
        return nagents, nbodies, behaviorAgentDict




class Learner_NEAT_From_CMA(Learner_NEAT):
    def __init__(self,config_details):
        Learner_NEAT.__init__(self,config_details)
        self.CONFIG_DETAILS["populationFolder"]+= f"{self.dirSeparator}sineCurriculum"
        self.learnerCMA = Learner_CMA_Modified(config_details)


    def simulateGenome(self, genome, env, config):
        if isinstance(genome, tuple) and len(genome)==2:
            genID, gen = genome
        elif isinstance(genome, neat.genome.DefaultGenome):
            genID, gen = ("no id", genome)
        else:
            raise TypeError(f"genome of unreadable type: {genome}")

        simulationSteps = self.CONFIG_DETAILS.getint("simulationSteps")

        env.reset()

        networkNEAT = self.generateNeuralNet(gen, config)

        self.assertBehaviorNames(env)
        behaviorNames = sorted(list(env.behavior_specs.keys()))

        reward = {behavior:[] for behavior in behaviorNames}

        # networkCMA = {}
        # for jointstr in self.learnerCMA.network:
        #     localScope = {}
        #     exec(self.learnerCMA.network[jointstr], {**globals(), **locals()}, localScope)
        #     networkCMA[jointstr] = localScope[list(localScope.keys())[0]]

        networkCMA = self.learnerCMA.readControllerStrings(self.learnerCMA.network)

        actionNEATDict = {} # Initial action value
        for behavior in behaviorNames:
            decisionSteps, _ = env.get_steps(behavior)
            for id in decisionSteps.agent_id:
                jointstr = self.makeJointstr(behavior, id)
                actionNEATDict[jointstr] = 0


        for step in range(simulationSteps):
            timeVal = step/50 # number of seconds that have passed

            for behavior in behaviorNames:
                decisionSteps, _ = env.get_steps(behavior)

                behaviorReward = 0
                for id, obs in zip(decisionSteps.agent_id, decisionSteps.obs[0]):
                    jointstr = self.makeJointstr(behavior, id)
                    
                    actionCMA  = networkCMA[jointstr](timeVal)

                    actionNEAT = self.readNeuralNet(networkNEAT, obs, actionNEATDict[jointstr])
                    actionNEATDict[jointstr] = actionNEAT

                    behaviorReward += (actionNEAT - actionCMA)**2

                    # pdb.set_trace()

                    action = ActionTuple(
                        np.array((actionCMA, actionCMA)).reshape(1,2)
                        )
                    env.set_action_for_agent(behavior, id, action)

                reward[behavior].append(-np.sqrt(float(behaviorReward)))

                # TODO: How tf to define behavior? 
                #       Difference in command? X
                #       Difference in result?

            env.step()
        env.close()


        # for behavior in reward:
        #     reward[behavior] /= (self.CONFIG_DETAILS.getint("simulationSteps"))

        # Optimization is a maximization where we want minimization, 
        # and negating reward was simpler than digging through
        # documentation for a maximization option
        return self.rewardAggregation(reward)

    def rewardAggregation(self, rewards):
        return sum([sum(val) for val in rewards.values()])

    def fitnessFunc(self,genome,queue,config):
        # TODO: Overwrites fitnessFunc for testingpurposes; remove when done
        if isinstance(genome, tuple) and len(genome)==2:
            genID, gen = genome
        elif isinstance(genome, neat.genome.DefaultGenome):
            genID, gen = ("no id", genome)
        else:
            raise TypeError(f"genome of unreadable type: {genome}")
        simulationSteps = self.CONFIG_DETAILS.getint("simulationSteps")

        reward = 0
        network = self.generateNeuralNet(gen,config)
        for step in range(simulationSteps):
            step/= 50
            # action = self.readNeuralNet(network, [0]*12, 0)
            action = network.advance(
                [0]*13, 
                self.timeConst, 
                self.timeConst,
            )
        

            reward+= (action[0] - np.sin(step))**2

        return -reward




if __name__ == "__main__":
    # TODO: for gods sake, make some unit tests bro
    pass
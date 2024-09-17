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
import cma

class Learner():
    def __init__(self, config_details):
        self.CONFIG_DETAILS = config_details

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


class Learner_NEAT():
    def __init__(self,config_details):
        Learner.__init__(self,config_details)
        self.NEAT_CONFIG = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.CONFIG_DETAILS["configFilepath"]
        )

    def fitnessFuncTest(self,genome,config):
        _, gen = genome 
        network = neat.nn.RecurrentNetwork.create(gen, config)

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
        network = neat.nn.RecurrentNetwork.create(gen, config)

        self.assertBehaviorNames(env)
        behaviorNames = sorted(list(env.behavior_specs.keys()))
        # behaviorName = behaviorNames[0]
        reward = {behavior:0 for behavior in behaviorNames}

        for t in range(simulationSteps):
            for behaviorName in behaviorNames:
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
                    action = np.array(network.activate(obs)).reshape(1,2)
                    env.set_action_for_agent(
                        behaviorName, id,
                        ActionTuple(np.array(action))
                    )


                reward[behaviorName] +=sum(decisionSteps.reward) / len(decisionSteps.reward)
                if reward[behaviorName] <= -1000: # Large negative numbers means disqualification
                    reward[behaviorName] -= 1000 * (simulationSteps - t)
                    break

            env.step()
        env.close()


        for key in reward:
            reward[key] /= (self.CONFIG_DETAILS.getint("simulationSteps"))




        # if reward > 0:
        #     print(f"Reward given to {genID:<3} as {reward:.2g}; last instance was "
        #         f"({', '.join(str(round(i,2)) for i in decisionSteps.reward)})")
        # else:
        #     print(f"Reward given to {genID:<3} as {reward:.2g}; "
        #           "last instance was DISQUALIFIED")

        return self.rewardAggregation(reward)

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
        reward = self.simulateGenome(genome,config,env)

        queue.put(worker_id)

        # if reward > 0:
        #     print(f"{reward:.1e}", end=", ")
        # else:
        #     print(f"DISC", end=", ")

        # print(f"genome {genID} sucessfully simulated")

        return reward

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
            print(f"Using imap")
            print("Rewards given: ", end="")
            # self.SimulateGenome prints reward as print(f"{reward:.1e}", end=", ")
            with mp.Pool(numWorkers) as pool:
                for genome, fitness in zip(genomes, tqdm(pool.imap(
                        self.fitnessFuncMapper, 
                        [(genome, config,queue) for genome in genomes]
                        ), total=len(genomes)-1, bar_format="{l_bar}{bar:20}{r_bar}")):
                    genome[1].fitness = fitness
                    # The total has to be reduced by 1... for _some_ reason....

        elif case == 4:
            print(f"using built-in")
            worker = neat.parallel.ParallelEvaluator(numWorkers, self.fitnessFunc, timeout=None)
            worker.evaluate(genomes, config)

        # print() # End the rewards printing with \n

    def fitnessFuncMapper(self, arg):
        """
        Wraps self.fitnessFunc to allow single-argument Pool.imap"""
        return self.fitnessFunc(*arg)

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

            pop = self.findGeneration()


        if not pop:
            pop = neat.Population(config)
            pop.add_reporter(neat.StdOutReporter(False))
            pop.add_reporter(neat.checkpoint.Checkpointer(
                generation_interval=1,
                filename_prefix=self.CONFIG_DETAILS["populationFolder"]+"\\generation_",
                ))
        
        if numGen == pop.generation:
            raise Exception("Training started, but generation limit already "
                            f"met or exceeded: {pop.generation} >= {numGen}")



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

            outfileName = outfilePath + "\\bestSpecimen"
            with open(outfileName, "wb") as outfile:
                pickle.dump(bestBoi, outfile)



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
            pop =  neat.checkpoint.Checkpointer.restore_checkpoint(Generation[0])
                
            # Overwrites old config details
            pop = neat.population.Population(
                self.NEAT_CONFIG,
                (pop.population, pop.species, pop.generation)
                )
        
            pop.add_reporter(neat.StdOutReporter(False))
            pop.add_reporter(neat.checkpoint.Checkpointer(
                generation_interval=1,
                filename_prefix=self.CONFIG_DETAILS["populationFolder"]+"\\generation_",
                ))
            return pop


                # return (pop, bestSpecimen)
        else:
            pop = neat.Population(config)
            pop.add_reporter(neat.StdOutReporter(False))
            pop.add_reporter(neat.checkpoint.Checkpointer(
                generation_interval=1,
                filename_prefix=self.CONFIG_DETAILS["populationFolder"]+"\\generation_",
                ))

            return pop

    def demonstrateGenome(self,genome=None,config=None):
        # raise NotImplemented("top genome no longer saved\n I am sad")
        if config is None:
            config = self.NEAT_CONFIG
        if genome is None:
            # _, genome = self.findGeneration()
            genome = self.CONFIG_DETAILS["populationFolder"] + "\\bestSpecimen"

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

        reward = self.simulateGenome(genome,config,env)

        if reward > 0:
            print(f"Reward finalized as: {reward}")
        else:
            print(f"DISC")

    def motionTest(self):
        # Applies basic motion for visual evaluation of physical rules
        print("Please start environment")
        env = UnityEnvironment()
        print("Environment found")

        env.reset()

        self.assertBehaviorNames(env)
        behaviorNames = sorted(list(env.behavior_specs.keys()))
        # behaviorName = behaviorNames[0]

        motionDuration = 15

        while True:
            for behaviorName in behaviorNames:
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
        # raise NotImplemented("top genome no longer saved\n I am sad")
        if genome is None:
            # _, genome = self.findGeneration()
            genome = self.CONFIG_DETAILS["populationFolder"] + "\\bestSpecimen"
            with open(genome, "rb") as infile:
                genome = pickle.load(infile)
        if config is None:
            config = self.NEAT_CONFIG
        draw_net(config, genome, True)






class Learner_CMA(Learner):
    def __init__(self,config_details):
        Learner.__init__(self,config_details)
        # Magical numbers to map CMAArgs from optimal CMA-range (0-10)*
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
            es = cma.CMAEvolutionStrategy(
                x0=[0,8,0]*nagents + [10]*nbodies, 
                sigma0=2,
                options=cmaOptions,
            )
            iteration = 0

        while True:
            es.optimize(self.simulateGenome, iterations=10)
            iteration+= 10
            # pdb.set_trace()
            outfileName = self.CONFIG_DETAILS["resultsFolder"]
            outfileName+= f"\\CMA\\iteration_{iteration}"
            with open(outfileName, "wb") as outfile:
                pickle.dump((es, iteration), outfile)
        print(es.result)

    def demonstrateGenome(self):
        # argsThatWiggleUnnaturally = [
        #         -0.07355574, -1.70311053,  2.26106487,  6.2765652 , 
        #         -3.35534422, -0.10351327,  0.21064545, -3.54712654, 
        #         -5.2254731 ,  3.74792834, -4.02749407, -1.14582858]
        # for ind in [0, 4, 8]:
        #     # Constant
        #     argsThatWiggleUnnaturally[ind]*= 0
        # for ind in [1, 5, 9]:
        #     # Amplitude
        #     argsThatWiggleUnnaturally[ind]*= 8
        # for ind in [2, 6, 10]:
        #     # Frequency
        #     argsThatWiggleUnnaturally[ind]*= 5
        # for ind in [3, 7, 11]:
        #     # Phase
        #     argsThatWiggleUnnaturally[ind]+= 0

        cmaArgs = self.findGeneration()[0].result[0]
        # pdb.set_trace()
        returnActions = {}
        self.simulateGenome(
            cmaArgs, 
            worker_id=2, 
            # instance="editor",
            instance="build", 
            returnActions=returnActions)

        # print(actions)

        # Printing 
        nagents, nbodies, behaviorAgentDict = self.getBehaviors()
        args, freqs = cmaArgs[:-nbodies], cmaArgs[-nbodies:]
        freqs = {behavior: freq for freq, behavior 
                 in zip(freqs, sorted(list(behaviorAgentDict.keys())))}
        def printCMAArgs():
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


    def simulateGenome(self, cmaArgs, worker_id=1, instance=None, returnActions=None,):
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
        # print(behaviorNames)
        # simulationSteps=0
        reward = {behavior:[] for behavior in behaviorNames}
        allJoints = [behavior + "?agent=" + str(agentID) 
                     for behavior in behaviorNames 
                     for agentID in env.get_steps(behavior)[0].agent_id]
        
        nbodies = len(behaviorNames)
        args, freqs = cmaArgs[:-nbodies], cmaArgs[-nbodies:]
        freqs = {behavior: freq for freq, behavior in zip(freqs, behaviorNames)}

        # network = {
        #     # Creates individual functions as 
        #     # A + B*sin(C*x + D) for each joint
        #     behavior: lambda i, step:
        #     (self.a*cmaArgs[4*i+0] + (self.b*cmaArgs[4*i+1]
        #     *np.sin(self.c*cmaArgs[4*i+2]*step + self.d*cmaArgs[4*i+3])))
        #     for i, behavior in enumerate(allJoints)
        # }

        network = {}
        i = 0
        for behavior in behaviorNames:
            for agentID in env.get_steps(behavior)[0].agent_id:
                freq = freqs[behavior]/5
                func = lambda i, step: ( 
                    self.a(args[3*i+0]) + (self.b(args[3*i+1])
                    *np.sin(self.c(freq*step) + self.d(args[3*i+2])))
                )
                joint = behavior + "?agent=" + str(agentID) 
                network[joint] = func
                i+=3
                

        if returnActions is not None:
            for joint in allJoints:
                returnActions[joint] = ([], [])
                # (sent signal, actual joint angle)

        for step in range(simulationSteps):
            timeVal = step/50 # number of seconds that have passed

            i = 0
            for behavior in behaviorNames:
                decisionSteps, _ = env.get_steps(behavior)

                for id, obs in zip(decisionSteps.agent_id, decisionSteps.obs[0]):
                    jointstr = behavior + "?agent=" + str(id)
                    action = network[jointstr](i, timeVal)
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
        generationFolder = f"{self.CONFIG_DETAILS['resultsFolder']}\\CMA"
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
            with open(f"{generationFolder}\\iteration_{lastGeneration}", "rb") as infile:
                iteration = pickle.load(infile)
            return iteration
        except FileNotFoundError:
            return None, None

        



if __name__ == "__main__":
    # TODO: for gods sake, make some unit tests bro
    pass
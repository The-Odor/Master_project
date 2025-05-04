import matplotlib.pyplot as plt
from Learner import Learner_NEAT, Learner_CMA, Learner_NEAT_From_CMA
import SmallFunctions
import configparser as cfp
import multiprocessing as mp
import itertools as it
import numpy as np
import copy
import sys
import pickle
import pdb
import os

params = {
   'font.size': 13,
   'axes.labelsize':  13,
   'legend.fontsize': 13,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
#    'text.usetex': False,
   'figure.figsize': [5.6,3.9],
   }
plt.rcParams.update(params)
plt.tight_layout()

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
            CONFIG_DETAILS["parallelWorkers"] = str(min(12, int(line[11:-1])))

# Finalize destination folder
CONFIG_DETAILS["populationFolder"] = (
    CONFIG_DETAILS["populationFolder1"] +
    CONFIG_DETAILS["populationCount"] + CONFIG_DETAILS["populationFolder2"]
)

scoreDictFilepathNEAT = r"C:\Users\theod\Master_project\data and table storage\experimentOneScoreDict"
scoreDictFilepathCMA  = r"C:\Users\theod\Master_project\data and table storage\experimentTwoScoreDict"
NMORPHOLOGIES = 22
morphologies = ["gecko", "queen", "stingray", "insect", "babya", "spider", "tinlicker", "longleg", "salamander", "park", "squarish", "blokky", "babyb", "snake", "linkin", "ww", "turtle", "penguin", "zappa", "garrix", "ant", "pentapod"]
NEATinclude = ["gecko","queen","babyb","tinlicker","ww","snake"]
CMAinclude = NEATinclude
# NMORPHOLOGIES = 3; morphologies = morphologies[:NMORPHOLOGIES]
if len(sys.argv) > 1:
    try:
        morphologies = [morphologies[int(sys.argv[1])]]
    except ValueError:
        if sys.argv[1] in morphologies:
            morphologies = [sys.argv[1]]
# morphologies = morphologies[:6]
morphologies = [morph + "_v1?team=0" for morph in morphologies]
EVALUATIONREPETITIONS = 3

class FalseQueue():
    def get(self,):
        return 1
    def put(self,thing):
        pass

class Parallelizable_Learner_NEAT(Learner_NEAT):
    def __init__(self, config_details, morphologyTrainedOn=None):
        super().__init__(config_details, morphologyTrainedOn)
        # Reasoning for punishment magnitude:
        # A decent fitness level is 2, a dead controller gives a fitness of 
        # about 1, duration of simulation is 20 seconds, and expected step 
        # frequency is about twice a second given data from pre-experiments. 
        # 40 steps thus needs to give less of a punishment than the expoected
        # reward for a decent walk, so punishment needs to be less than 1/40.
        # This punishment is reduced from its maximum value to allow decent
        # walks an actual reward magnitude
        self.oscillationPunishment = (1/40) / 4096
    # def fitnessFuncMapper(self, arg):
    #     # Fetch simulation targets from queue
    #     morph = queue.get()
    #     return self.fitnessFunc(*arg, morphologiesToSimulate=morph), morph
    def rewardAggregation(self, rewards):
        # Reward for a single behavior is equivalent to final distance
        modified_reward = {behavior:0 for behavior in rewards}
        for behavior in rewards:
            for i in range(2, len(rewards[behavior])):
                break
                sign1 = rewards[behavior][i] - rewards[behavior][i-1]
                sign2 = rewards[behavior][i-1] - rewards[behavior][i-2]

                # If the two latest differences have different signs (i.e.
                # the controller has performed an oscillation), give it 
                # a small penalty, to combat jittering. 0 inclusive;
                # plateauing counts as an oscillation
                if sign1*sign2 <= 0:
                    modified_reward[behavior]-= self.oscillationPunishment
                
            modified_reward[behavior]+= rewards[behavior][-1]
        return modified_reward


def seedingFunction(self, pop):
    # TODO:
    # To be put in during experiment 2
    numberOfSeeds = 72 # a tenth of 720
    for genID, genome in list(pop.population.items())[:numberOfSeeds]:
        n1, n2 = list(genome.nodes.keys())[:2]
        factor = 1.0
        for node, bias in zip((n1,n2), (-2.75/5, -1.75/5)):
            # genome.nodes[node].aggregation = sum
            # genome.nodes[node].activation = sigmoid_activation
            genome.nodes[node].bias = bias*factor
            genome.nodes[node].response = 1
        genome.add_connection(self.NEAT_CONFIG.genome_config, n1, n1, 0.9*factor, True)
        genome.add_connection(self.NEAT_CONFIG.genome_config, n1, n2,-0.2*factor, True)
        genome.add_connection(self.NEAT_CONFIG.genome_config, n2, n1, 0.2*factor, True)
        genome.add_connection(self.NEAT_CONFIG.genome_config, n2, n2, 0.9*factor, True)

def switchEnvironment(learner, trainedMorphology):
    if trainedMorphology.endswith("_v1?team=0"):
        trainedMorphology = trainedMorphology[:-10]
    index = CONFIG_DETAILS["exeFilepath"].rfind(learner.CONFIG_DETAILS["unityEnvironmentName"])
    newEnvironment = learner.CONFIG_DETAILS["exeFilepath"][:index]
    newEnvironment+= f"{trainedMorphology}{learner.dirSeparator}{learner.CONFIG_DETAILS['unityEnvironmentName']}{learner.CONFIG_DETAILS['fileendingFormat']}"
    learner._switchEnvironmentPath(newEnvironment)



if __name__ == "__main__":
    mp.freeze_support()

    ### CTRNN, unseeded, Experiment 1
    try:
        with open(scoreDictFilepathNEAT, "rb") as infile:
            scoreDict = pickle.load(infile)
    except FileNotFoundError:
        scoreDict = {}
    fullEnvironment = CONFIG_DETAILS["exeFilepath"]
    for i,trainedMorphology in enumerate(morphologies):
        if trainedMorphology in scoreDict:
            continue
        print(f"\n\nTraining morphology {i+1} unseeded: {trainedMorphology}".upper())
        # Training
        # if not 'winner' in globals():  
        learner = Parallelizable_Learner_NEAT(copy.deepcopy(CONFIG_DETAILS), morphologyTrainedOn=[trainedMorphology])
        # tmorph = trainedMorphology[:-10]
        # index = learner.CONFIG_DETAILS["exeFilepath"].rfind(learner.CONFIG_DETAILS["unityEnvironmentName"])
        # newEnvironment = learner.CONFIG_DETAILS["exeFilepath"][:index]
        # newEnvironment+= f"{tmorph}{learner.dirSeparator}{learner.CONFIG_DETAILS['unityEnvironmentName']}{learner.CONFIG_DETAILS['fileendingFormat']}"
        # learner._switchEnvironmentPath(newEnvironment)
        switchEnvironment(learner, trainedMorphology)
        # import os
        # index = learner.CONFIG_DETAILS["exeFilepath"].rfind(learner.CONFIG_DETAILS["unityEnvironmentName"])
        # outfilePath = learner.CONFIG_DETAILS["exeFilepath"][:index]
        # if not os.path.exists(outfilePath):
        #     print(f"making directory {outfilePath}")
        #     os.makedirs(outfilePath)
        # learner.CONFIG_DETAILS["exeFilepath"] = fullEnvironment
        # continue

        finalGeneration, winner = learner.run(useCheckpoint=True)

        # Self-evaluation
        # learner.morphologiesToSimulate = [trainedMorphology]
        # selfScore = learner.fitnessFunc(winner, FalseQueue())

        # Evaluation
        print(f"Evaluating morphology {i+1} unseeded: {trainedMorphology}".upper())
        manager = mp.Manager()
        queue = manager.Queue()
        # nOthersToTest = 4 # 4 gives 7315 evaluations, 5 gives 26334
        # simulations = [queue.put(i) for i in it.combinations(morphologies, r=nOthersToTest)]

        # learner.morphologiesToSimulate = morphologies
        individualSimulations = True

        if individualSimulations:
            scores = {}
            for testingMorphology in morphologies:
                CONFIG_DETAILS["populationCount"] = str(EVALUATIONREPETITIONS)#len(simulations)
                # learner._switchEnvironmentPath(fullEnvironment)
                learner.switchEnvironment(testingMorphology)
                score = learner.evaluatePopulation([[0,winner]]*EVALUATIONREPETITIONS, learner.NEAT_CONFIG, training=False)[None]
                scores[testingMorphology] = [list(s.values())[0] for s in score]
                print(f"{trainedMorphology} evaluated on {testingMorphology} achieves a score of {sum([list(s.values())[0] for s in score])/EVALUATIONREPETITIONS * 10000}")

        else:
            learner._switchEnvironmentPath(fullEnvironment)
            CONFIG_DETAILS["populationCount"] = "3"#len(simulations)
            scores = learner.evaluatePopulation([[0,winner]]*EVALUATIONREPETITIONS, learner.NEAT_CONFIG, training=False)[None]
        
        # Save data TODO
        # print(selfScore, otherScore)
        # print(scoreDict)yhon
        
        scoreDict[trainedMorphology] = scores

    with open(scoreDictFilepathNEAT, "wb") as outfile:
        pickle.dump(scoreDict, outfile)

    tabularDict = {}
    NEATCTRNNScores = []
    for morph_i, scores in scoreDict.items():
        # scoreDict[morph] = sum(scores)/len(scores)
        tabularDict[morph_i] = {morph_ii: 0 for morph_ii in morphologies}
        for i in range(EVALUATIONREPETITIONS):
            for morph_ii, _ in scoreDict.items():
                # tabularDict[morph_i][morph_ii]+= scores[i][morph_ii]/EVALUATIONREPETITIONS
                tabularDict[morph_i][morph_ii]+= scores[morph_ii][i]/EVALUATIONREPETITIONS * 1000
            
        NEATCTRNNScores.append(tabularDict[morph_i][morph_i])

    tabularList = [["...",] +[i[:-10][:5] for i in morphologies],]
    for morph_i in morphologies:
        tabularList.append([morph_i])
        for morph_ii in morphologies:
            tabularList[-1].append(tabularDict[morph_i][morph_ii])
    for i in range(1, len(tabularList[1:])+1):
        tabularList[i][0] = tabularList[i][0][:-10]
    from tabulate import tabulate
    # tabulate([morphologies] + [[morph] + scoreDict[morph] for morph in morphologies])
    datatable = SmallFunctions.latexifyExperiment_1_2(tabularList[0][1:],[i[1:] for i in tabularList[1:]])
    with open(r"C:\Users\theod\Documents\Github repositories\Master-thesis\Thesis\tables\experiment1_CTRNN_NEAT.txt", "w") as outfile:
        outfile.write(datatable)
    # print(SmallFunctions.latexifyExperiment_1_2(tabularList[0],tabularList[1:]))
    print(tabulate(tabularList[1:], headers=tabularList[0], floatfmt=".1f"))

    fitnessDictFilepath = r"C:\Users\theod\Master_project\data and table storage\fitness_over_generations_dict"
    with open(fitnessDictFilepath, "rb") as infile:
        neatFitnessDict = pickle.load(infile)

    colourGrouping = {
        "gecko_v1?team=0": "blue",
        "snake_v1?team=0": "orange",
        "queen_v1?team=0": "brown",
        "ww_v1?team=0": "blue",
        "tinlicker_v1?team=0": "orange",
        "babyb_v1?team=0": "brown",
    }
    for group_i, morphGrouping in enumerate((["gecko", "snake", "queen"], ["ww", "tinlicker", "babyb"])):
        for morph in morphGrouping:
            morph += "_v1?team=0"
            # data = np.asarray(list(neatFitnessDict[morph].values()))*1000
            plt.fill_between(
                x = range(21),
                # y1 = [np.max(i)*1000 for i in neatFitnessDict[morph].values()], 
                # y2 = [np.min(i)*1000 for i in neatFitnessDict[morph].values()], 
                y2 = [np.percentile(i,84.1)*1000 for i in neatFitnessDict[morph].values()], # Within one STD
                y1 = [np.percentile(i,15.9)*1000 for i in neatFitnessDict[morph].values()], 
                # label=morph[:-10],
                alpha = 0.2,
                color=colourGrouping[morph],
            )
            # equality = [np.percentile(i,75)*1000 for i in neatFitnessDict[morph].values() if np.percentile(i,75) == np.percentile(i,25)]
            # plt.plot(equality)

            plt.plot(
                range(21),
                [np.mean(i)*1000 for i in neatFitnessDict[morph].values()], 
                label=morph[:-10] + " mean",
                color=colourGrouping[morph],
            )
            
            plt.plot(
                range(21),
                [np.max(i)*1000 for i in neatFitnessDict[morph].values()],
                linestyle="--",
                label=morph[:-10] + " max",
                color=colourGrouping[morph],
            )

        plt.legend()
        plt.ylabel("Fitness")
        plt.xlabel("Generation")
        plt.grid()
        plt.savefig(rf"C:\Users\theod\Documents\Github repositories\Master-thesis\Thesis\figures\experiment1_fitness_over_generations_group_{group_i}.pdf", format="pdf")
        # plt.show()
        plt.clf()


    colourGrouping = {
        "gecko_v1?team=0": "blue",
        "snake_v1?team=0": "orange",
        "queen_v1?team=0": "brown",
        "ww_v1?team=0": "red",
        "tinlicker_v1?team=0": "purple",
        "babyb_v1?team=0": "black",
    }
    figInd, axInd = plt.subplots()
    figDif, axDif = plt.subplots()
    for morph in NEATinclude:
        morph += "_v1?team=0"
        speciesSize = []
        populationSize = []
        diffSize = []
        for generation in range(20+1):
            learner = Parallelizable_Learner_NEAT(
                config_details=copy.deepcopy(CONFIG_DETAILS),
                morphologyTrainedOn=[morph],
            )
            learner.switchEnvironment(
                trainedMorphology=morph,
            )

            pop,_ = learner.findGeneration(specificGeneration=generation)
            # print(morph, generation, pop.generation, len(pop.population), len(pop.species.species))
            populationSize.append(len(pop.population))
            speciesSize.append(len(pop.species.species))
            diffSize.append(len(pop.population) - len(pop.species.species))
        axInd.plot(range(20+1),populationSize,color=colourGrouping[morph])
        axInd.plot(range(20+1),speciesSize,color=colourGrouping[morph],linestyle=":")
        axDif.plot(range(20+1),diffSize,label=morph[:-10],color=colourGrouping[morph])
    # axInd.set_xticklabels(axInd.get_xticks().astype(int))
    # axDif.set_xticklabels(axDif.get_xticks().astype(int))
    from matplotlib.ticker import MaxNLocator
    axInd.xaxis.set_major_locator(MaxNLocator(integer=True))
    axInd.set_xlabel("Generations")
    axInd.set_ylabel("Individuals")
    axDif.xaxis.set_major_locator(MaxNLocator(integer=True))
    axDif.set_xlabel("Generations")
    axDif.set_ylabel("Non-leader individuals")
    figInd.legend()
    figInd.savefig(rf"C:\Users\theod\Documents\Github repositories\Master-thesis\Thesis\figures\experiment1_species_population_size.pdf", format="pdf")
    figDif.legend()
    figDif.savefig(rf"C:\Users\theod\Documents\Github repositories\Master-thesis\Thesis\figures\experiment1_species_population_difference.pdf", format="pdf")
    # plt.show()
    plt.clf()


    ### CTRNN, seeded, Experiment 2
    # Training
    pass
    # Self-evaluation
    pass
    # Other-evaluation
    pass


    ### Sine, Experiment 3
    # Training
    # morphologies = morphologies[:2]
    morphologies = [morph[:-10] if morph.endswith("_v1?team=0") else morph for morph in morphologies]
    # morphologies = morphologies[-2:]
    # Dummy learner so I can extract some variables later on if all data is created and nothing needs extraction
    learner = Learner_CMA(copy.deepcopy(CONFIG_DETAILS), morphologyTrainedOn="gecko", morphologyToSimulate="gecko", controllerFormat=None)
    for format in [1,3]:
        fitnesses = {}
        figFreqAmp, axFreqAmp = plt.subplots()
        figPhaseShift, axPhaseShift = plt.subplots()
        for morph in morphologies:
            if format in []:
                continue

            if not os.path.isfile(fr"C:\Users\theod\Master_project\Populations\CMA\{morph}\format_{format}\iteration_100"):
                continue
                learner = Learner_CMA(copy.deepcopy(CONFIG_DETAILS), morphologyTrainedOn=morph, morphologyToSimulate=morph, controllerFormat=format)
                # learner.switchEnvironment(morph)  # now done in __init__ with morphologyToSimulate 
                learner.train()

            fitnesses[morph] = []
            for i in range(10,100+1,10):
                file = fr"C:\Users\theod\Master_project\Populations\CMA\{morph}\format_{format}\iteration_{i}"
                with open(file, "rb") as infile:
                    es = pickle.load(infile)[0]
                if i == 100:
                    if morph in CMAinclude:
                        means = es.mean
                        STDs  = es.stds
                        shiftMeans, ampMeans, phaseMeans = ([],[],[])
                        shiftSTDs, ampSTDs, phaseSTDs = ([],[],[])
                        # pdb.set_trace()
                        for i in range(len(means)):
                            if i == len(means)-1:
                                freqMean = learner.c(means[i]) / 50 # Converting from per time step to Hz
                                freqSTD  = learner.c(STDs[i]) / 50 # Converting from per time step to Hz
                                # print(morph, freqMean, freqSTD)
                            elif i%3 == 0:
                                shiftMeans.append(learner.a(means[i]))
                                shiftSTDs.append(12*(STDs[i]))
                            elif i%3 == 1:
                                ampMeans.append(learner.b(means[i]))
                                ampSTDs.append(learner.b(STDs[i]))
                            elif i%3 == 2:
                                phaseMeans.append(learner.d(means[i])%(2*np.pi))
                                phaseSTDs.append(learner.d(STDs[i]))
                        nJoints = len(ampMeans)
                        axFreqAmp.errorbar(x=[freqMean]*nJoints, y=ampMeans, xerr=[freqSTD]*nJoints, yerr=ampSTDs, fmt="o")
                        axPhaseShift.errorbar(x=phaseMeans, y=shiftMeans, xerr=phaseSTDs, yerr=shiftSTDs, fmt="o")

                fitnesses[morph].extend([-i for i in es.fit.hist[:10]][::-1])
                    

        axFreqAmp.set_xlabel("Frequency")
        axFreqAmp.set_ylabel("Amplitude")
        # axFreqAmp.set_title("FreqAmp")
        # figFreqAmp.legend(CMAinclude)
        figFreqAmp.savefig(rf"C:\Users\theod\Documents\Github repositories\Master-thesis\Thesis\figures\frequency_amplitude_graph_format{format}.pdf", format="pdf")
        axPhaseShift.set_xlabel("Phase")
        axPhaseShift.set_ylabel("Shift")
        # y_pi   = y/np.pi
        # unit   = 0.25
        # y_tick = np.arange(-0.5, 0.5+unit, unit)
        # axPhaseShift.set_title("PhaseShift")
        figPhaseShift.legend(CMAinclude)
        figPhaseShift.savefig(rf"C:\Users\theod\Documents\Github repositories\Master-thesis\Thesis\figures\phase_shift_graph_format{format}.pdf", format="pdf")
        # fig.suptitle("TITLE PLACEHOLDER")
        # plt.savefig(r"C:\Users\theod\Downloads\GoFuckYourself")
        # plt.show()
        plt.clf()




        CMASinusoidalScores = [fitnesses[key][-1] for key in fitnesses]
        datatable = SmallFunctions.latexifyExperiment_3(names=[i[:5] for i in morphologies], table=[fitnesses[key][-1] for key in fitnesses],format=format)
        with open(rf"C:\Users\theod\Documents\Github repositories\Master-thesis\Thesis\tables\experiment2_SIN_CMA_format{format}.txt", "w") as outfile:
            outfile.write(datatable)
        for morph in fitnesses:
            if morph not in CMAinclude:
                continue
            plt.plot(fitnesses[morph], label=morph)
        plt.ylabel("Fitness")
        plt.xlabel("Generation")
        if format == 3:
            plt.legend(ncols=3)
        plt.ylim(0,2.5)
        # plt.title(f"format_{format}")
        plt.savefig(rf"C:\Users\theod\Documents\Github repositories\Master-thesis\Thesis\figures\experiment3_fitness_over_generations_format{format}.pdf", format="pdf")
        # plt.show()
        plt.clf()



        NEATCTRNNScores, CMASinusoidalScores = np.asarray(NEATCTRNNScores), np.asarray(CMASinusoidalScores)
        # pdb.set_trace()
        print(SmallFunctions.latexifyExperiment_3(names=[i[:5] for i in morphologies], table=NEATCTRNNScores/CMASinusoidalScores,format=format))
        print(SmallFunctions.latexifyExperiment_3(names=[i[:5] for i in morphologies], table=NEATCTRNNScores-CMASinusoidalScores,format=format))

        # pdb.set_trace()
        print()
from MLAgentLearn import Learner_NEAT, Learner_CMA, CONFIG_DETAILS
import copy
from FormalizedExperiment import FalseQueue
import pickle
import os
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # learnerNEAT = Learner_NEAT(CONFIG_DETAILS)
    # learnerCMA = Learner_CMA(CONFIG_DETAILS)
    # learner.demonstrateGenome(learner.findGeneration()[1])

    morphologies = ["stingray", "insect", "gecko", "babya", "spider", "queen", "tinlicker", "longleg", "salamander", "park", "squarish", "blokky", "babyb", "snake", "linkin", "ww", "turtle", "penguin", "zappa", "garrix", "ant", "pentapod"]
    # morphologies = ["queen", "gecko"]
    if len(sys.argv) == 1:
        case = 0
        # case 0: Prints pdf of topology
        # case 1: Demonstrates neat solution
        # case 2: Demonstrates simple motion in Unity editor
        # case 3: Demonstrates cma solution
        # case 4: Demonstrates neat signals over time
        # case 5: Demonstrates a neat solution with a chosen morphology
    else:
        case = int(sys.argv[1])

    if case == 0:
        # creates a PDF of given genome (defaults to best of latest generation)
        morphologies = [morph + "_v1?team=0" for morph in morphologies]
        for i,trainedMorphology in enumerate(morphologies):
            learner = Learner_NEAT(copy.deepcopy(CONFIG_DETAILS), morphologyTrainedOn=[trainedMorphology])
            _, genome = learner.findGeneration()
            learner.makePDF(genome=genome.get_pruned_copy(learner.NEAT_CONFIG.genome_config))

    elif case == 1:
        # case 1:
        # morphologies = ["stingray", "insect", "gecko", "babya", "spider", "queen", "tinlicker", "longleg", "salamander", "park", "squarish", "blokky", "babyb", "snake", "linkin", "ww", "turtle", "penguin", "zappa", "garrix", "ant", "pentapod"]
        morphologies = [morph + "_v1?team=0" for morph in morphologies]
        fullEnvironment = CONFIG_DETAILS["exeFilepath"]
        for i,trainedMorphology in enumerate(morphologies):
            print(f"\n\nTraining morphology {i+1} unseeded: {trainedMorphology}".upper())
            # Training
            # if not 'winner' in globals():  
            learner = Learner_NEAT(copy.deepcopy(CONFIG_DETAILS), morphologyTrainedOn=[trainedMorphology])
            tmorph = trainedMorphology[:-10]
            # index = learner.CONFIG_DETAILS["exeFilepath"].rfind(learner.CONFIG_DETAILS["unityEnvironmentName"])
            # newEnvironment = learner.CONFIG_DETAILS["exeFilepath"][:index]
            # newEnvironment+= f"{tmorph}{learner.dirSeparator}{learner.CONFIG_DETAILS['unityEnvironmentName']}{learner.CONFIG_DETAILS['fileendingFormat']}"
            # learner.switchEnvironment(newEnvironment)
            learner.switchEnvironment(tmorph)


            
            # Demonstrates Genomes (hence name, lol)
            learner.demonstrateGenome()
            # learner.demonstrateGenome(learner.findGeneration()[1])
            # learner.demonstrateGenome(learner.findGeneration(specificGeneration=69)[1])

    elif case == 2:
        # case 2:
        # Demonstrates simple motion in Unity editor
        # learner = Learner_NEAT(copy.deepcopy(CONFIG_DETAILS))
        # morphologies = ["queen", "gecko"]
        trainedMorphology = [morph + "_v1?team=0" for morph in morphologies][1]
        learner = Learner_NEAT(copy.deepcopy(CONFIG_DETAILS), morphologyTrainedOn=[trainedMorphology])
        tmorph = trainedMorphology[:-10]
        index = learner.CONFIG_DETAILS["exeFilepath"].rfind(learner.CONFIG_DETAILS["unityEnvironmentName"])
        newEnvironment = learner.CONFIG_DETAILS["exeFilepath"][:index]
        newEnvironment+= f"{tmorph}{learner.dirSeparator}{learner.CONFIG_DETAILS['unityEnvironmentName']}{learner.CONFIG_DETAILS['fileendingFormat']}"
        learner.switchEnvironment(newEnvironment)

        learner.motionTest(useEditor=True)

    elif case == 3:
        # case 3:
        # Demonstrates cma solution
        for demonstrable in ["gecko", "queen"]:
            print(demonstrable)
            learnerCMA = Learner_CMA(CONFIG_DETAILS, morphologyToSimulate=demonstrable)
            learnerCMA.switchEnvironment(demonstrable)
            learnerCMA.demonstrateGenome()

    elif case == 4:
        # morphologies = ["gecko"]
        morphologies = [morph + "_v1?team=0" for morph in morphologies]
        fullEnvironment = CONFIG_DETAILS["exeFilepath"]
        for i,trainedMorphology in enumerate(morphologies):
            print(f"\n\nExtracting action data from {trainedMorphology}".upper())

            learner = Learner_NEAT(copy.deepcopy(CONFIG_DETAILS), morphologyTrainedOn=[trainedMorphology])
            # tmorph = trainedMorphology[:-10]
            # index = learner.CONFIG_DETAILS["exeFilepath"].rfind(learner.CONFIG_DETAILS["unityEnvironmentName"])
            # newEnvironment = learner.CONFIG_DETAILS["exeFilepath"][:index]
            # newEnvironment+= f"{tmorph}{learner.dirSeparator}{learner.CONFIG_DETAILS['unityEnvironmentName']}{learner.CONFIG_DETAILS['fileendingFormat']}"
            # learner.switchEnvironment(newEnvironment)
            
            learner.switchEnvironment(trainedMorphology)

            filepath = learner.CONFIG_DETAILS["populationFolder"]
            filepath = filepath[:filepath.rfind('\\')]
            filepath+= rf"\{trainedMorphology[:-len('?team=0')]}\actionDict"
            if os.path.isfile(filepath):
                with open(filepath, "rb") as infile:
                    actions = pickle.load(infile)
            else:
                _, genome = learner.findGeneration()
                actions = {}
                # actions will have format
                # actions = {
                #     jointname: ([list,of,signals,sent], [list,of,angles,measured])
                # }
                learner.fitnessFunc(genome, queue=FalseQueue(), returnActions=actions)
                with open(filepath, "wb") as outfile: 
                    pickle.dump(actions, outfile)
                    # TODO: This pickledump continually overwrites everything!
            # learner.demonstrateGenome(learner.findGeneration()[1])
            # learner.demonstrateGenome(learner.findGeneration(specificGeneration=69)[1])

            if True:
                for intendedOrActual in range(2):
                    for jointname in actions:
                        # actionset is name of joint
                        plt.plot(actions[jointname][intendedOrActual][:], label=jointname)
                    # plt.legend()
                    plt.title({
                        0: "Action sent",
                        1: "Angle recieved",
                    }[intendedOrActual])
                    plt.xlabel("timesteps")
                    plt.ylabel("angle[$\degree$]")
                    if intendedOrActual == 0:
                        plt.savefig(fr"C:\Users\theod\Documents\Github repositories\Master-thesis\Thesis\figures\neat_signals_runtime\neat_signal_waveform_{trainedMorphology[:-len('_v1?team=0')]}.pdf", format="pdf")
                    elif intendedOrActual == 1:
                        plt.savefig(fr"C:\Users\theod\Documents\Github repositories\Master-thesis\Thesis\figures\neat_signals_runtime\neat_joint_waveform_{trainedMorphology[:-len('_v1?team=0')]}.pdf", format="pdf")
                    plt.close() # Does not permit plt.show()
                    plt.show()
                print()


        morphologiesImportantToNeat = ["gecko", "queen", "spider", "babyb", "tinlicker", "ww", "snake"]
        for morph in morphologiesImportantToNeat:
            # morph = morph[:-len("_v1?team=0")]
            filepathSignal = f"figures/neat_signals_runtime/neat_signal_waveform_{morph}.pdf"
            filepathJoint = f"figures/neat_signals_runtime/neat_joint_waveform_{morph}.pdf"

# fig:neat_gecko_signals,fig:neat_queen_signals,fig:neat_spider_signals,fig:neat_babyb_signals,fig:neat_tinlicker_signals,fig:neat_ww_signals,fig:neat_snake_signals,

            print(fr"""\begin{{figure}}
    \begin{{subfigure}}[b]{{0.5\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{{filepathSignal}}}
    \end{{subfigure}}
    \hfill
    \begin{{subfigure}}[b]{{0.5\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{{filepathJoint}}}
    \end{{subfigure}}
    \caption{{Control signal (a) and joint angles (b) over the length of a simulation for morphology {morph}}}
    \label{{fig:neat_{morph}_signals}}
\end{{figure}}""")
            
    elif case==5:
        morphologies = ["gecko", "queen"]
        morphologies = [morph + "_v1?team=0" for morph in morphologies]
        for i,trainedMorphology in enumerate(morphologies):
            print(f"\n\nTraining morphology {i+1} unseeded: {trainedMorphology}".upper())
            learner = Learner_NEAT(copy.deepcopy(CONFIG_DETAILS), morphologyTrainedOn=[trainedMorphology])
            tmorph = trainedMorphology[:-10]
            learner.switchEnvironment(tmorph)
            
            for morph in morphologies:
                # if morph != trainedMorphology:
                # Demonstrates Genomes (hence name, lol)
                # _, genome = learner.findGeneration()
                learner.switchEnvironment(morph)
                learner.demonstrateGenome()
            # learner.demonstrateGenome(learner.findGeneration()[1])
            # learner.demonstrateGenome(learner.findGeneration(specificGeneration=69)[1])

from MLAgentLearn import Learner_NEAT, Learner_CMA, CONFIG_DETAILS
import copy
from FormalizedExperiment import FalseQueue
import pickle
import os

if __name__ == "__main__":
    # learnerNEAT = Learner_NEAT(CONFIG_DETAILS)
    # learnerCMA = Learner_CMA(CONFIG_DETAILS)
    # learner.demonstrateGenome(learner.findGeneration()[1])

    case = 4

    if case == 1:
        # case 1:
        # morphologies = ["stingray", "insect", "gecko", "babya", "spider", "queen", "tinlicker", "longleg", "salamander", "park", "squarish", "blokky", "babyb", "snake", "linkin", "ww", "turtle", "penguin", "zappa", "garrix", "ant", "pentapod"]
        morphologies = ["queen", "gecko"]
        morphologies = [morph + "_v1?team=0" for morph in morphologies]
        fullEnvironment = CONFIG_DETAILS["exeFilepath"]
        for i,trainedMorphology in enumerate(morphologies):
            print(f"\n\nTraining morphology {i+1} unseeded: {trainedMorphology}".upper())
            # Training
            # if not 'winner' in globals():  
            learner = Learner_NEAT(copy.deepcopy(CONFIG_DETAILS), morphologyTrainedOn=[trainedMorphology])
            tmorph = trainedMorphology[:-10]
            index = learner.CONFIG_DETAILS["exeFilepath"].rfind(learner.CONFIG_DETAILS["unityEnvironmentName"])
            newEnvironment = learner.CONFIG_DETAILS["exeFilepath"][:index]
            newEnvironment+= f"{tmorph}{learner.dirSeparator}{learner.CONFIG_DETAILS['unityEnvironmentName']}{learner.CONFIG_DETAILS['fileendingFormat']}"
            learner.switchEnvironment(newEnvironment)


            # creates a PDF of given genome (defaults to best of latest generation)
            learner.makePDF(genome=None)
            
            # Demonstrates Genomes (hence name, lol)
            learner.demonstrateGenome()
            # learner.demonstrateGenome(learner.findGeneration()[1])
            # learner.demonstrateGenome(learner.findGeneration(specificGeneration=69)[1])

    elif case == 2:
        # case 2:
        # Demonstrates simple motion in Unity editor
        # learner = Learner_NEAT(copy.deepcopy(CONFIG_DETAILS))
        morphologies = ["queen", "gecko"]
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
        learnerCMA = Learner_CMA
        learnerCMA.demonstrateGenome()

    elif case == 4:
        # morphologies = ["stingray", "insect", "gecko", "babya", "spider", "queen", "tinlicker", "longleg", "salamander", "park", "squarish", "blokky", "babyb", "snake", "linkin", "ww", "turtle", "penguin", "zappa", "garrix", "ant", "pentapod"]
        morphologies = ["gecko"]
        morphologies = [morph + "_v1?team=0" for morph in morphologies]
        fullEnvironment = CONFIG_DETAILS["exeFilepath"]
        for i,trainedMorphology in enumerate(morphologies):
            print(f"\n\nExtracting action data from {trainedMorphology}".upper())

            learner = Learner_NEAT(copy.deepcopy(CONFIG_DETAILS), morphologyTrainedOn=[trainedMorphology])
            tmorph = trainedMorphology[:-10]
            index = learner.CONFIG_DETAILS["exeFilepath"].rfind(learner.CONFIG_DETAILS["unityEnvironmentName"])
            newEnvironment = learner.CONFIG_DETAILS["exeFilepath"][:index]
            newEnvironment+= f"{tmorph}{learner.dirSeparator}{learner.CONFIG_DETAILS['unityEnvironmentName']}{learner.CONFIG_DETAILS['fileendingFormat']}"
            learner.switchEnvironment(newEnvironment)
            
            filepath = learner.CONFIG_DETAILS["populationFolder"]
            filepath = filepath[:filepath.rfind('\\')]
            filepath+= r"\actionDict"
            if os.path.isfile(filepath):
                with open(filepath, "rb") as infile:
                    actions = pickle.load(infile)
            else:
                _, genome = learner.findGeneration()
                actions = {}
                learner.fitnessFunc(genome, queue=FalseQueue(), returnActions=actions)
                with open(filepath, "wb") as outfile: 
                    pickle.dump(actions, outfile)
            # learner.demonstrateGenome(learner.findGeneration()[1])
            # learner.demonstrateGenome(learner.findGeneration(specificGeneration=69)[1])


            import matplotlib.pyplot as plt
            for intendedOrActual in range(2):
                for actionset in actions:
                    plt.plot(actions[actionset][intendedOrActual], label=actionset)
                plt.legend()
                plt.title({
                    0: "Action sent",
                    1: "Angle recieved",
                }[intendedOrActual])
                plt.show()
            print()
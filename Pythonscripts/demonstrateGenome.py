from MLAgentLearn import Learner_NEAT, Learner_CMA, CONFIG_DETAILS


if __name__ == "__main__":
    learnerNEAT = Learner_NEAT(CONFIG_DETAILS)
    learnerCMA = Learner_CMA(CONFIG_DETAILS)
    # learner.demonstrateGenome(learner.findGeneration()[1])

    case = 1

    if case == 1:
        # case 1:
        # creates a PDF of given genome (defaults to best of latest generation)
        learnerNEAT.makePDF(genome=None)
        
        # Demonstrates Genomes (hence name, lol)
        learnerNEAT.demonstrateGenome()
        # learner.demonstrateGenome(learner.findGeneration()[1])
        # learner.demonstrateGenome(learner.findGeneration(specificGeneration=69)[1])

    elif case == 2:
        # case 2:
        # Demonstrates simple motion in Unity editor
        learnerNEAT.motionTest()

    elif case == 3:
        # case 3:
        # Demonstrates cma solution
        learnerCMA.demonstrateGenome()

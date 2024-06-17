from MLAgentLearn import Learner, CONFIG_DETAILS


if __name__ == "__main__":
    learner = Learner(CONFIG_DETAILS)
    # learner.demonstrateGenome(learner.findGeneration()[1])

    case = 1

    if case == 1:
        # case 2:
        # Demonstrates Genomes (hence name, lol)
        learner.demonstrateGenome(learner.findGeneration()[1])
        # learner.demonstrateGenome(learner.findGeneration(specificGeneration=69)[1])
    elif case == 2:
        # case 3:
        # Demonstrates simple motion in Unity editor
        learner.motionTest()
    elif case == 3:
        # case 4: 
        # creates a PDF of given genome (defaults to best of latest generation)
        learner.makePDF(genome=None)
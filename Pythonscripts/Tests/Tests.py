from Determinism.testDeterminism import testDeterminism, fetchGenome


if __name__ == "__main__":
    genome = "C:\\Users\\theod\\Master_project\\Pythonscripts\\Tests\\Determinism\\testGenome.pkl"
    
    fetchGenome()
    for i in range(5):
        testDeterminism(genome)
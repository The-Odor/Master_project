import pickle
import neat
import pdb

with open("C:\\Users\\theod\\Master_project\\Populations\\0.7.1 (timescale 100)\\popcount360_simlength600\\generation_0119.pkl", "rb") as infile:
    pop, _ = pickle.load(infile)



NEAT_CONFIG = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "C:\\Users\\theod\\Master_project\\Pythonscripts\\temp\\tempconfig"
)

newpop = neat.population.Population(NEAT_CONFIG, (pop, pop.species, pop.generation))

pdb.set_trace()
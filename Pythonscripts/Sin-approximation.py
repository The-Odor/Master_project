import neat
import pdb
import numpy as np
from visualize import draw_net
import matplotlib.pyplot as plt
from neat.activations import sigmoid_activation
import random

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
        r"C:\Users\theod\Master_project\Pythonscripts\configs\CTRNNconfigPreExperiment"
)

simulationSteps = 5000
timeConst = 1/34 # how impactful time is
advTime   = 1/1000 # how long to run the network per call
timeStep  = 1/1000 # how frequently to run the network within that time
freq = 1/2250 / 4 * 1.2
rewardFactor = 1e3
generations = 90
numberOfSeeds = 3
goalFunc = lambda step: -1.3*np.sin(step*25)/6 + 0.5
stepEvaluator = lambda step, action: (action[0] - goalFunc(freq*step))**2

def printGenome(genome, genomeName="genome"):
    if genomeName is None:
        genomeName = "genome"
    nodeKeys = list(genome.nodes.keys())
    # pdb.set_trace()
    print(f"""{genomeName} contains:
nodes:........... {nodeKeys}
bias:............ {[genome.nodes[n].bias for n in nodeKeys]}
response:........ {[genome.nodes[n].response for n in nodeKeys]}
aggregation:..... {[genome.nodes[n].aggregation for n in nodeKeys]}
activation:...... {[genome.nodes[n].activation for n in nodeKeys]}
interconnections: {[((n1,n2), genome.connections[(n1,n2)].weight) for n1 in nodeKeys for n2 in nodeKeys if n1!=n2 and (n1,n2) in genome.connections]}
self-connections: {[(n, genome.connections[(n,n)].weight) for n in nodeKeys if (n,n) in genome.connections]}
""")

def fitnessFunc(genomes,config, plot=False, axis=None, name=None):
    for genID, genome in genomes:

        network = neat.ctrnn.CTRNN.create(genome, config, timeConst)
        network.set_node_value(0, 0.5)

        # if isinstance(genome, neat.genome.DefaultGenome):
        #     network = neat.ctrnn.CTRNN.create(genome, config, timeConst*10)
        #     network.set_node_value(0, 0.5)
        # elif isinstance(genome, neat.ctrnn.CTRNN):
        #     network = genome



        # reward = 0
        actions = np.empty(simulationSteps)
        for step in range(simulationSteps):
            action = network.advance(
                # [step],
                [], 
                advTime, 
                timeStep,
            )

            # reward+= (action[0] - goalFunc(freq*step))**2
            actions[step] = action[0]
        
        phaseGranularity = 10 # number of phases to test
        phaseshifts = np.linspace(0,2*np.pi,phaseGranularity)

        # stepMatrix = np.arange(simulationSteps)[:,np.newaxis].repeat(phaseGranularity,axis=1)
        # phaseshiftedOutputs = stepEvaluator(stepMatrix + phaseshifts, actions)
        # phaseshiftedRewards = phaseshiftedOutputs[:,:].sum(axis=0)
        # optimalInd = phaseshiftedRewards.argmin()
        # reward = phaseshiftedRewards[optimalInd]

        phaseshiftedGoalVals = goalFunc((np.arange(simulationSteps,dtype=float).reshape(-1,1)*freq + phaseshifts.reshape(1,-1)))
        actionMatrix = actions[:,np.newaxis].repeat(phaseGranularity,axis=1)

        loss = ((actionMatrix - phaseshiftedGoalVals)**2).sum(axis=0)
        phaseshiftIndex = loss.argmin()
        reward = loss[phaseshiftIndex]

        reward = reward/simulationSteps*rewardFactor
        genome.fitness = -reward

        if plot:
            if axis is None:
                plt.plot(actions, label="Network")
                plt.plot(goalFunc(np.arange(simulationSteps)*freq + phaseshifts[phaseshiftIndex]), label="Goal")
                plt.ylabel("Signal")
                plt.xlabel("Timestep")
                plt.title(f"Mean Square Error: {reward:.2f}\nCTRNN time constant: {round(timeConst, -int(np.floor(np.log10(abs(timeConst))-1)))}")
                plt.legend()
                if name is None:
                    printGenome(genome, genomeName="Plotted genome")
                else:
                    printGenome(genome, genomeName=name)
                figurePath = r"C:\Users\theod\Documents\Github repositories\Master-thesis\Thesis\figures" + "\\"
                if mode == 1:
                    plt.savefig(figurePath + "preexperiment_fully_evolved.pdf", format="pdf")
                if mode == 3:
                    plt.savefig(figurePath + "preexperiment_fully_manual.pdf", format="pdf")
                if mode == 5:
                    plt.savefig(figurePath + "preexperiment_seeded_evolved.pdf", format="pdf")
                plt.show()
            else:
                axis.plot(actions, label="Network")
                axis.plot(goalFunc(np.arange(simulationSteps)*freq + phaseshifts[phaseshiftIndex]), label="Goal")

def runAndPlot(genome, axis=None, name=None):
    return fitnessFunc(((0,genome),), config, plot=True, axis=axis, name=name)
    actions = []
    correct = []

    reward = 0
    network = neat.ctrnn.CTRNN.create(genome, config, timeConst)
    network.set_node_value(0, 0.5)
    for step in range(simulationSteps):

        action = network.advance(
            # [step],
            [], 
            advTime, 
            timeStep,
        )

        reward+= (action[0] - goalFunc(freq*step))**2
        actions.append(action[0])
        correct.append(goalFunc(freq*step))
    
    reward = reward/simulationSteps*rewardFactor

    if axis is None:
        plt.plot(actions, label="Network")
        plt.plot(correct, label="Goal")
        plt.title(reward)
        plt.legend()
        if name is None:
            printGenome(genome, genomeName="Plotted genome")
        else:
            printGenome(genome, genomeName=name)
        plt.show()
    else:
        axis.plot(actions, label="Network")
        axis.plot(correct, label="Goal")

def modifyGenome(genome):
    n1, n2 = list(genome.nodes.keys())
    factor = 1.0
    for node, bias in zip((n1,n2), (-2.75/5, -1.75/5)):
        # genome.nodes[node].aggregation = sum
        # genome.nodes[node].activation = sigmoid_activation
        genome.nodes[node].bias = bias*factor
        genome.nodes[node].response = 1
    genome.add_connection(config.genome_config, n1, n1, 0.9*factor, True)
    genome.add_connection(config.genome_config, n1, n2,-0.2*factor, True)
    genome.add_connection(config.genome_config, n2, n1, 0.2*factor, True)
    genome.add_connection(config.genome_config, n2, n2, 0.9*factor, True)
    # genome.add_connection(config.genome_config, n1, 0, 0, True)
    # genome.add_connection(config.genome_config, n2, 0, 1, True)
    # genome.add_connection(config.genome_config, 0, 0, 0, True)
    # pdb.set_trace()

def plotPopulation(pop, priorityPlotters=[]):
    # return
    fig, axes = plt.subplots(dim, dim)
    fig.suptitle(f"Generation {pop.generation}")
    for i, ((genID, genome), axis) in enumerate(zip(pop.population.items(), axes.flat)):
        if i < len(priorityPlotters):
            genID, genome = priorityPlotters[i]
        elif i == dim**2:
            break
        axis.set_title(str(genID))
        runAndPlot(genome, axis, name=genID)

    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels)

    # fig.legend()
    plt.show()


# Constructing final network
mode = [
    1,
    3,
    5,
]
# 1: Fully evolved, unseeded
# 2: Fully manually constructed seed genome (makes network, not genome)
# 3: Seed genome extracted from prior seeding
# 4: Manual conversion from genome to network
# 5: Fully evolved, seeded
if isinstance(mode, int):
    mode = [mode]
for mode in mode:
    print(f"Running mode {mode} now")
    pop = neat.Population(config)
    dim = min(3, int(np.ceil(np.sqrt(len(pop.population)))))
    # random.seed(5)
    pop.add_reporter(neat.StdOutReporter(False))

    automaticTermination = False
    if mode == 1:
        popSizeModifier = 1
        genSizeModifier = 1
        config.pop_size = config.pop_size * popSizeModifier
        generations = generations * genSizeModifier

        # Manually adding inter-output connections because apparently that's fucking turned off???????
        for genID, genome in list(pop.population.items()):#[:numberOfSeeds]:
            cg = genome.create_connection(config.genome_config, 0, 1)
            genome.connections[cg.key] = cg
            cg = genome.create_connection(config.genome_config, 1, 0)
            genome.connections[cg.key] = cg

        # Plot seed
        if automaticTermination:
            optimalGenome = pop.run(fitnessFunc, generations)
        else:
            for i in range(generations):
                optimalGenome = pop.run(fitnessFunc, 1)

                printGenome(optimalGenome)
                # Plot normal
                # if i%3 == 0 or i == generations:
                #     plotPopulation(pop)
                # # Modify and plot, just to know I can
                # modifyGenome(genome)
                # plotPopulation(pop)
            # genome = pop.run(fitnessFunc, generations)
            # network = neat.ctrnn.CTRNN.create(optimalGenome, config, timeConst)
        genome = optimalGenome
        config.pop_size = int(config.pop_size / popSizeModifier)
        generations = int(generations / genSizeModifier)
    elif mode == 5:
        # Seeding
        for genID, genome in list(pop.population.items())[:numberOfSeeds]:
            modifyGenome(genome)

        plotPopulation(pop)
        # Plot seed
        if automaticTermination:
            optimalGenome = pop.run(fitnessFunc, generations)
        else:
            for i in range(generations):
                optimalGenome = pop.run(fitnessFunc, 1)
                # Plot normal
                if i%3 == 0 or i == generations:
                    plotPopulation(pop, priorityPlotters=[("Best Genome", optimalGenome)])
                printGenome(optimalGenome)
                # # Modify and plot, just to know I can
                # modifyGenome(genome)
                # plotPopulation(pop)
            # genome = pop.run(fitnessFunc, generations)
            # network = neat.ctrnn.CTRNN.create(optimalGenome, config, timeConst)
        genome = optimalGenome
    elif mode == 2:
        node1_inputs = [(1, 0.9), (2, 0.2)]
        node2_inputs = [(1,-0.2), (2, 0.9)]
        node_evals   = {
            1: neat.ctrnn.CTRNNNodeEval(0.01, sigmoid_activation, sum, -2.75 / 5.0, 1.0, node1_inputs),
            2: neat.ctrnn.CTRNNNodeEval(0.01, sigmoid_activation, sum, -1.75 / 5.0, 1.0, node2_inputs),
            }
        network = neat.ctrnn.CTRNN([], [1, 2], node_evals)
        init1 = 0.0
        init2 = 0.0
        network.set_node_value(1, init1)
        network.set_node_value(2, init2)
        # genome = network # Jank fix, later functions check if genome is actually genome or network (I think) (they don't)
    elif mode == 3:
        genID, genome = list(pop.population.items())[0]
        modifyGenome(genome)
        network = neat.ctrnn.CTRNN.create(genome, config, timeConst)
        printGenome(genome, genomeName="Seeded genome without extraction")
        # network.set_node_value(n1, 0)
        # network.set_node_value(n2, 0)
        # network.set_node_value(0, 0.5)
        # pdb.set_trace()
    elif mode == 4:
        genome_config = config.genome_config
        required = neat.ctrnn.required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)
        node_inputs = {}
        for cg in genome.connections.values():
            # Add connection if enabled and required
            if not cg.enabled:
                continue
            i, o = cg.key
            if o not in required and i not in required:
                continue
            if o not in node_inputs:
                node_inputs[o] = [(i, cg.weight)]
            else:
                node_inputs[o].append((i, cg.weight))

        node_evals = {}
        for nodeKey, inputs in node_inputs.items():
            node = genome.nodes[nodeKey]
            # bias = (-2.75 / 5.0) if nodeKey == 0 else (-1.75 / 5.0)
            bias = node.bias
            response = node.response
            activation_function = genome_config.activation_defs.get(node.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation)
            node_evals[nodeKey] = neat.ctrnn.CTRNNNodeEval(timeConst, activation_function, aggregation_function, bias, response, inputs)
        network = neat.ctrnn.CTRNN([], [0, 1], node_evals)
        init1 = 0.0
        init2 = 0.0
        network.set_node_value(1, init1)
        network.set_node_value(2, init2)
        # pdb.set_trace()
    else:
        raise NotImplementedError(f"mode {mode} not implemented")

    runAndPlot(genome)
    draw_net(config, genome)

# plot(actions, correct)
# pdb.set_trace()
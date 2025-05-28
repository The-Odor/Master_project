import neat
import pdb
import numpy as np
from visualize import draw_net
import matplotlib.pyplot as plt
from neat.activations import sigmoid_activation
import random
import pickle

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
        r"C:\Users\theod\Master_project\Pythonscripts\configs\CTRNNconfigPreExperiment"
)

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

FFT_RESULTS=[]

def fitnessFunc(genomes,config, plot=False, savePlot=False, axis=None, name=None, showPlot=True, genomePrint=False):
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

        fft_result = np.fft.fft(actions)
        frequencies = np.fft.fftfreq(simulationSteps, d=1)
        n = len(frequencies)
        frequencies = frequencies[:n//2]
        fft_result = np.abs(fft_result[:n//2])
        FFT_RESULTS.append((fft_result,frequencies))

        # TODO: Do something with the fft_result

        phaseGranularity = 10 # number of phases to test
        phaseshifts = np.linspace(0,2*np.pi,phaseGranularity)

        # stepMatrix = np.arange(simulationSteps)[:,np.newaxis].repeat(phaseGranularity,axis=1)
        # phaseshiftedOutputs = stepEvaluator(stepMatrix + phaseshifts, actions)
        # phaseshiftedRewards = phaseshiftedOutputs[:,:].sum(axis=0)
        # optimalInd = phaseshiftedRewards.argmin()
        # reward = phaseshiftedRewards[optimalInd]

        phaseshiftedGoalVals = goalFunc(frequencyAdjustment*(np.arange(simulationSteps,dtype=float).reshape(-1,1)*freq + phaseshifts.reshape(1,-1)))
        actionMatrix = actions[:,np.newaxis].repeat(phaseGranularity,axis=1)

        loss = ((actionMatrix - phaseshiftedGoalVals)**2).sum(axis=0)
        phaseshiftIndex = loss.argmin()
        reward = loss[phaseshiftIndex]

        reward = reward/simulationSteps*rewardFactor

        overWriteFitnessWithExtremaCount = True
        nTops = 0
        nBots = 0
        if overWriteFitnessWithExtremaCount:
            for i in range(1,len(actions)-1):
                if actions[i] < actions[i-1] and actions[i] < actions[i+1]:
                    nBots+=1
                if actions[i] > actions[i-1] and actions[i] > actions[i+1]:
                    nTops+=1
            reward += np.abs((nBots + nTops) - 6*frequencyAdjustment)*10


        genome.fitness = -reward

        if plot:
            if axis is None:
                plt.plot(actions, label=rf"$\tau$={timeConst:.3f}")
                if mode in [1]:
                    plt.plot(goalFunc(frequencyAdjustment*np.arange(simulationSteps)*freq + phaseshifts[phaseshiftIndex]), label="Goal")
                    plt.legend()
                plt.ylabel("Signal")
                plt.xlabel("Timestep")
                # plt.title(f"Mean Square Error: {reward:.2f}\nCTRNN time constant: {round(timeConst, -int(np.floor(np.log10(abs(timeConst))-1)))}")
                if genomePrint:
                    if name is None:
                        printGenome(genome, genomeName="Plotted genome")
                    else:
                        printGenome(genome, genomeName=name)
                if savePlot:
                    figurePath = r"C:\Users\theod\Documents\Github repositories\Master-thesis\Thesis\figures\pre-experiment" + "\\"
                    # if mode == 1:
                    #     plt.savefig(figurePath + "preexperiment_fully_evolved.pdf", format="pdf")
                    # if mode == 3:
                    #     plt.savefig(figurePath + f"preexperiment_seed_genome_TC_{timeConst:.3f}.pdf", format="pdf")
                    # if mode == 5:
                    #     plt.savefig(figurePath + "preexperiment_seeded_evolved.pdf", format="pdf")
                if showPlot:
                    plt.show()
            else:
                axis.plot(actions, label="Network")
                axis.plot(goalFunc(frequencyAdjustment*np.arange(simulationSteps)*freq + phaseshifts[phaseshiftIndex]), label="Goal")

def runAndPlot(genome, axis=None, name=None, savePlot=False, showPlot=True):
    return fitnessFunc(((0,genome),), config, plot=True, savePlot=savePlot, axis=axis, name=name, showPlot=showPlot)
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
        runAndPlot(genome, axis=axis, name=genID)

    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels)

    # fig.legend()
    plt.show()


# Constructing final network
mode = [
    # 5,
    # 1,
    # 3,
]
mode = 5

# 1: Fully evolved, unseeded
# 2: Fully manually constructed seed genome (makes network, not genome)
# 3: Seed genome extracted from prior seeding
# 4: Manual conversion from genome to network
# 5: Fully evolved, seeded
timeConstSpace = np.logspace(-1.8,-1.2,num=6)
AdjustmentSpace = [0.5, 1, 1.5, 2, 2.5]
# AdjustmentSpace = [1]

if isinstance(mode, int):
    # mode = [mode]
    pass # woopsies lmao
# for mode in mode:

genomeCollectionDictFilepath = r"C:\Users\theod\Master_project\data and table storage\preexperimentGenomeCollectionDict"
try:
    with open(genomeCollectionDictFilepath, "rb") as infile:
        genomeCollectionDict = pickle.load(infile)
except FileNotFoundError:
    genomeCollectionDict = {
        frequencyAdjustment:[] for frequencyAdjustment in AdjustmentSpace
    }

# genomeCollectionDict format:
# genomeCollectionDict = {
#     frequencyAdjustment_0 = [
#         [genome,genome,genome,genome,genome], # for each timeConst
#         [genome,genome,genome,genome,genome],
#         ... # for each trial run
#     ]
#     frequencyAdjustment_1 = [
#         ...
#     ]
#     ...
# }
Nreps = 10
while True:
    import matplotlib.pyplot as plt
    for frequencyAdjustment in AdjustmentSpace:
        if len(genomeCollectionDict[frequencyAdjustment])<Nreps:
            break
    else:
        break
        # breaks out of the while True only if all frequencyAdjustments are finished
        
    for frequencyAdjustment in AdjustmentSpace:
        if len(genomeCollectionDict[frequencyAdjustment])>=Nreps:
            continue
        # if frequencyAdjustment not in genomeCollectionDict:
        genomeCollectionDict[frequencyAdjustment].append([])

        for timeConst in timeConstSpace:
            # if len(genomeCollectionDict[frequencyAdjustment]) > 0:
            if len(genomeCollectionDict[frequencyAdjustment][-1]) >= len(timeConstSpace):
                continue
            # frequencyAdjustment = 2
            # timeConst = 1/34
            # frequencyAdjustment = frequencyAdjustment_
            # timeConst = timeConst_

            print(f"timeConst={timeConst}, frequencyAdjustment={frequencyAdjustment}")

            print(f"Running mode {mode} now")
            pop = neat.Population(config)
            dim = min(3, int(np.ceil(np.sqrt(len(pop.population)))))
            # random.seed(5)
            pop.add_reporter(neat.StdOutReporter(False))

            automaticTermination = True
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

                # plotPopulation(pop)
                # Plot seed
                if automaticTermination:
                    optimalGenome = pop.run(fitnessFunc, generations)
                else:
                    for i in range(generations):
                        optimalGenome = pop.run(fitnessFunc, 1)
                        # Plot normal
                        if i%5 == 0 or i == generations:
                            plotPopulation(pop, priorityPlotters=[("Best Genome", optimalGenome)])
                        printGenome(optimalGenome)
                        # # Modify and plot, just to know I can
                        # modifyGenome(genome)
                        # plotPopulation(pop)
                    # genome = pop.run(fitnessFunc, generations)
                    # network = neat.ctrnn.CTRNN.create(optimalGenome, config, timeConst)
                
                lookAtFFT=False
                if lookAtFFT:
                    # Look at collective fft analyses
                    fft_result_sample, frequencies_sample = FFT_RESULTS[0]
                    all_ffts = np.zeros_like(fft_result_sample)
                    fig, ax = plt.subplots(ncols=1)
                    n = len(frequencies_sample)
                    for fft_result in FFT_RESULTS:
                        all_ffts += fft_result[0]
                        ax.plot(fft_result[1][1:10], fft_result[0][1:10])

                    frequencies = frequencies_sample[:n//2]
                    fft_result = np.abs(all_ffts[:n//2])

                    # ax.plot(frequencies, fft_result)
                    ax.set_title("Magnitude Spectrum")
                    ax.set_xlabel("Frequency [1/timestep]")
                    ax.set_ylabel("Amplitude")
                    ax.grid(True)
                    # ax[1].plot(range(simulationSteps),signal)
                    plt.show()

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

            genomeCollectionDict[frequencyAdjustment][-1].append(genome)
            plt.clf()
            # runAndPlot(genome,savePlot=True,showPlot=False)
            # print("Do something with FFT_RESULTS")
            # fft_result_sample, frequencies_sample = FFT_RESULTS[0]
            # all_ffts = np.zeros_like(fft_result_sample)
            # n = len(frequencies_sample)
            # for fft_result in FFT_RESULTS:
            #     all_ffts += fft_result[0]
            # frequencies = 1/frequencies_sample[:n//2]
            # fft_result = np.abs(all_ffts[:n//2])
            # fig, ax = plt.subplots(ncols=1)

            # ax.plot(frequencies, fft_result)
            # ax.set_title("Magnitude Spectrum")
            # ax.set_xlabel("Period [timesteps]")
            # ax.set_ylabel("Amplitude")
            # ax.grid(True)
            # # ax[1].plot(range(simulationSteps),signal)
            # plt.show()

            # draw_net(config, genome)


        with open(genomeCollectionDictFilepath, "wb") as outfile:
            pickle.dump(genomeCollectionDict, outfile)
        print()

# exit()
# TODO: Grab the average of the runs


for frequencyAdjustment in AdjustmentSpace:
    for genome, timeConst in zip(genomeCollectionDict[frequencyAdjustment][0], timeConstSpace):
        runAndPlot(genome,savePlot=True,showPlot=False)
    plt.plot(goalFunc(frequencyAdjustment*np.arange(simulationSteps)*freq), linewidth=3, color="black")
    plt.legend(ncols=3)
    plt.ylim(0,1)
    # plt.savefig(rf"C:\Users\theod\Documents\Github repositories\Master-thesis\Thesis\figures\pre-experiment\preexperiment_seeded_evolved_FA_{frequencyAdjustment:.1f}.pdf", format="pdf")
    # plt.show()
    plt.clf()
    


    # plot(actions, correct)
    # pdb.set_trace()


def convertNormalizedToColour(norm):

    # FROM a TO b
    a = np.asarray([0xff,0x95,0x95])
    b = np.asarray([0xff,0xff,0xff])
    colour = ((b*norm)+((1-norm)*a)).round().astype(int)
    colour = colour[0]*0x10000 + colour[1]*0x100 + colour[2]*0x1
    return int(colour)
def makeColour(element, highestValue):
    if element >= highestValue:
        decimalColour = convertNormalizedToColour(1)
    else:
        decimalColour = convertNormalizedToColour(element/highestValue)
    hexColour = str(hex(decimalColour))[2:]
    hexColour = "0"*(6-len(hexColour)) + hexColour
    # return fr"\textcolor[HTML]{{{hexColour}}}{{{element:.2f}}}"
    return fr"\cellcolor[HTML]{{{hexColour}}}{element:.2f}"

genomeSTDCollectionDict = {}
for frequencyAdjustment in AdjustmentSpace:
    mean = np.zeros_like(timeConstSpace)
    STDs = np.zeros((len(timeConstSpace),Nreps))
    for i, genomes in enumerate(genomeCollectionDict[frequencyAdjustment]):
        genomeSTDCollectionDict[frequencyAdjustment] = []
        for ii in range(len(genomes)):
            fitness = -genomes[ii].fitness
            mean[ii] += fitness
            STDs[ii,i] = fitness

    genomeSTDCollectionDict[frequencyAdjustment] = np.std(STDs,axis=1)
    genomeCollectionDict[frequencyAdjustment] = mean / len(genomeCollectionDict[frequencyAdjustment])

for collectionDict in [genomeCollectionDict]:
    maxinator = []
    for frequencyAdjustment in collectionDict:
        for fitness in collectionDict[frequencyAdjustment]:
            maxinator.append(fitness)
    highestValue = max(maxinator)
    highestValue = min(50, highestValue)

    print("highest value is", highestValue)

    returnString="\n\n"
    returnString += "\n"+r"\begin{table}"+"\n"
    returnString += "\centering\n"
    returnString += r"\begin"+"{tabular}{|l|"+"l|"*len(AdjustmentSpace)+"}\n"r"\hline""\n "
    returnString += r"\backslashbox{$\tau\times100$}{$F_a$} & " + " & ".join([fr"{str(name)[:3]}" for name in AdjustmentSpace]) + "\\\\ \n \hline\n"
    # for i,frequencyAdjustment in enumerate(AdjustmentSpace):
    #     returnString+= f"{frequencyAdjustment}& "+" & ".join([f"{makeColour(fitness,highestValue)}{{\\footnotesize\\textcolor[HTML]{{404040}}{{$\pm{std:.2f}$}}}}" for fitness,std in [(fitness, std) for fitness, std in zip(collectionDict[frequencyAdjustment], genomeSTDCollectionDict[frequencyAdjustment])]])+"\\\\\n"
    returnStringAddendum = [str(name*100)[:3] for name in timeConstSpace]
    for i,frequencyAdjustment in enumerate(AdjustmentSpace):
        for ii, (fitness, std) in enumerate(zip(collectionDict[frequencyAdjustment], genomeSTDCollectionDict[frequencyAdjustment])):
            returnStringAddendum[ii] += f"& {makeColour(fitness,highestValue)}{{\\footnotesize\\textcolor[HTML]{{404040}}{{$\pm{std:.2f}$}}}}"
        
        # returnString+= f"{frequencyAdjustment}& "+" & ".join([f"{makeColour(fitness,highestValue)}{{\\footnotesize\\textcolor[HTML]{{404040}}{{$\pm{std:.2f}$}}}}" for fitness,std in [(fitness, std) for fitness, std in zip(collectionDict[frequencyAdjustment], genomeSTDCollectionDict[frequencyAdjustment])]])+"\\\\\n"

    for string in returnStringAddendum:
        returnString += string + "\\\\\n"

    returnString += "\hline\n"
    returnString += "\end{tabular}\n"
    returnString += r"\caption[Fitness values achieved in secondary experiment]{Fitness values achieved with various time constants adapting to signals of various signals. Standard deviation included in grey text.}" + "\n"
    returnString += "\label{tab:preexperiment_fitnesses}\n"
    returnString += "\end{table}"


    # with open(r"C:\Users\theod\Documents\Github repositories\Master-thesis\Thesis\tables\preexperiment_fitness_values.txt", "w") as outfile:
    #     outfile.write(returnString)


    print(returnString)
    print("REMINDER THAT ALL SAVING HAS BEEN FUCKING TURNED OFF IT'S GONE IT'S COMMENTED DON'T ASSUME IT WORKS")
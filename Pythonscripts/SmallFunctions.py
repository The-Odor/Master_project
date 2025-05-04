import numpy as np
def convertNormalizedToColour(norm):

    # FROM a TO b
    a = np.asarray([0xff,0xff,0xff])
    b = np.asarray([0xff,0x95,0x95])
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

def latexifyExperiment_1_2(names, table, horizontalLimitation=None, verticalLimitation=None):
    if verticalLimitation is not None:
        raise NotImplementedError
    if horizontalLimitation is None:
        horizontalLimitation = 11
    if isinstance(table, list):
        table = np.asarray(table)

    n = len(names)
    returnString="\n\n"
    returnString += "\n"+r"\begin{table}"+"\n"
    for section in range(n//horizontalLimitation):
        j = section*horizontalLimitation
        k = (section+1)*horizontalLimitation
        returnString += r"\begin"+"{tabular}{l|"+"l"*horizontalLimitation+"}\n"r"\hline""\n& "
        returnString += " & ".join([name for name in names[j:k]]) + "\\\\ \n \hline\n"
        # TODO: implement the following!!
        # highestValue = table.max() # OR max(table[i,j:k])
        # highestValue = np.partition(table.flatten(), -2)[-2] # finds second largest
        highestValue = min(1, table.max())
        for i in range(n):
            # returnString+= f"{names[i]}& "+" & ".join(np.round(table[i,j:k],2).astype("U"))+"\\\\\n"
            # returnString+= f"{names[i]}& "+" & ".join([f"{element:.2f}" if j+ii!=i else f"\\textbf{{{element:.2f}}}" for ii, element in enumerate(table[i,j:k])])+"\\\\\n"
            returnString+= f"{names[i]}& "+" & ".join([f"{makeColour(element,highestValue)}" if j+ii!=i else f"\\textbf{{{makeColour(element,highestValue)}}}" for ii, element in enumerate(table[i,j:k])])+"\\\\\n"

        returnString += "\n\end{tabular}\n"
    returnString += r"\caption[Fitness values achieved by CTRNN-NEAT controllers]{Fitness values achieved by CTRNN-NEAT controllers for different morphologies. Evaluations on self highlighted in bold. Column names are morphologies trained on, row names are morphologies evaluated on. colour-coded based on magnitude from fitness of \framebox(13,13){\colorbox[HTML]{ffffff}{0}} to fitness of \framebox(13,13){\colorbox[HTML]{ff9595}{1}} at most (any fitness value above 1 is set to the colour intensity of 1).}" + "\n"
    returnString += "\label{tab:neat_CTRNN_fitnesses}\n"
    returnString += "\end{table}"
    return returnString


def latexifyExperiment_3(names, table, horizontalLimitation=None, format=None):
    if horizontalLimitation is None:
        horizontalLimitation = 11
    if isinstance(table, list):
        table = np.asarray(table)

    returnString = r"\begin{table}"
    
    # import pdb
    # pdb.set_trace()

    n = len(names)
    for section in range(n//horizontalLimitation):
        j = section*horizontalLimitation
        k = (section+1)*horizontalLimitation
#         returnString += fr"""
# \begin{{tabular}}{{{'l'*(k-j)}}}
# \hline
# {" & ".join(names[j:k])}\\
#  \hline
# {" & ".join([f"{element:.2f}" for element in table[j:k]])}\\
# \end{{tabular}}
# """
        returnString += fr"""
\begin{{tabular}}{{{'l'*(k-j)}}}
\hline
{" & ".join(names[j:k])}\\
 \hline
{" & ".join([makeColour(i,1) for i in table[j:k]])}\\
\end{{tabular}}
"""
    returnString+= fr"\caption[Fitness values achieved by sinusoidal-CMA controllers, using the {'specialized format' if format==1 else 'decentralized format' if format==3 else ''}]{{Fitness values achieved by sinusoidal-CMA controllers for different morphologies, using the {'specialized format' if format==1 else 'decentralized format' if format==3 else ''}.colour-coded based on magnitude from fitness of \framebox(13,13){{\colorbox[HTML]{{ffffff}}{{0}}}} to fitness of \framebox(13,13){{\colorbox[HTML]{{ff9595}}{{1}}}} at most (any fitness value above 1 is set to the colour intensity of 1).}}" + "\n"
    returnString+= f"\label{{fig:CMA_data_table_format{format}}}\n"
    returnString+= "\end{table}"
    return returnString


def writeBashFiles():
    destinationFolder = r"C:\Users\theod\Master_project\Linux\master_project\FormalizedExperimentBashFiles"
    morphologies = ["gecko", "queen", "stingray", "insect", "babya", "spider", "tinlicker", "longleg", "salamander", "park", "squarish", "blokky", "babyb", "snake", "linkin", "ww", "turtle", "penguin", "zappa", "garrix", "ant", "pentapod"]
    for morph in morphologies:
        with open(fr"{destinationFolder}\formalizedExperiment_{morph}", "w") as outfile:
            outfile.write(f"""#!/bin/bash

# Remember to save with unix line endings

## Parameters
#SBATCH --account=ec29
#SBATCH --time=12:0:0
#SBATCH --job-name=mlagents_{morph}
#SBATCH --ntasks=1 #only one script
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2200M


## Commands
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate
python pythonscripts/FormalizedExperiment.py {morph}
""")

if __name__ == "__main__":
    import numpy as np

    names = ["stingray", "insect", "gecko", "babya", "spider", "queen", "tinlicker", "longleg", "salamander", "park", "squarish", "blokky", "babyb", "snake", "linkin", "ww", "turtle", "penguin", "zappa", "garrix", "ant", "pentapod"]

    n = len(names)
    table_1_2 = np.zeros((n,n))
    for i in range(n):
        table_1_2[i,i] = 1

    table_3 = np.zeros(n)

    print(latexifyExperiment_1_2(names, table_1_2, horizontalLimitation=7))
    print(latexifyExperiment_3(names, table_3, horizontalLimitation=7))
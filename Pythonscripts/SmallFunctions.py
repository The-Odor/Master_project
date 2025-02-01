def latexifyExperiment_1_2(names, table, horizontalLimitation=None, verticalLimitation=None):
    if verticalLimitation is not None:
        raise NotImplementedError
    
    n = len(names)
    returnString="\n\n"
    for section in range(n//horizontalLimitation):
        j = section*horizontalLimitation
        k = (section+1)*horizontalLimitation
        returnString += "\n"+r"\begin{table}[H]"+"\n"
        returnString += r"\begin"+"{tabular}{l|"+"l"*horizontalLimitation+"}\n"r"\hline""\n& "
        returnString += " & ".join([name for name in names[j:k]]) + "\\\\ \n \hline\n"
        for i in range(n):
            returnString+= f"{names[i]}& "+" & ".join(table[i,j:k].astype("U"))+"\\\\\n"

        returnString += "\n\end{tabular}\n"
        returnString += "\end{table}"
    return returnString


def latexifyExperiment_3(names, table, horizontalLimitation):

    returnString = ""
    
    for section in range(n//horizontalLimitation):
        j = section*horizontalLimitation
        k = (section+1)*horizontalLimitation
        returnString += fr"""

\begin{{table}}[H]
\begin{{tabular}}{{lllllll}}
\hline
{" & ".join(names[j:k])}\\
 \hline
{" & ".join(table[j:k].astype("U"))}\\
\end{{tabular}}
\end{{table}}
"""
    returnString += "\caption{{Fitness values achieved by CMA-evolved controllers for different morphologies}}"
    return returnString


def writeBashFiles():
    destinationFolder = r"C:\Users\theod\Master_project\Linux\master_project\FormalizedExperimentBashFiles"
    morphologies = ["gecko", "queen" "stingray", "insect", "babya", "spider", "tinlicker", "longleg", "salamander", "park", "squarish", "blokky", "babyb", "snake", "linkin", "ww", "turtle", "penguin", "zappa", "garrix", "ant", "pentapod"]
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
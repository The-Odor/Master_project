def latexifyTable(names, table, horizontlaLimitation=None, verticalLimitation=None):
    if verticalLimitation is not None:
        raise NotImplementedError
    
    n = len(names)
    returnString="\n\n"
    for section in range(n//horizontlaLimitation):
        j = section*horizontlaLimitation
        k = (section+1)*horizontlaLimitation
        returnString += "\n"+r"\begin{table}[H]"+"\n"
        returnString += r"\begin"+"{tabular}{l|"+"l"*horizontlaLimitation+"}\n"r"\hline""\n& "
        returnString += " & ".join([name for name in names[j:k]]) + "\\\\ \n \hline\n"
        for i in range(n):
            returnString+= f"{names[i]}& "+" & ".join(table[i,j:k].astype("U"))+"\\\\\n"

        returnString += "\n\end{tabular}\n"
        returnString += "\end{table}"
    return returnString


if __name__ == "__main__":
    import numpy as np

    names = ["stingray", "insect", "gecko", "babya", "spider", "queen", "tinlicker", "longleg", "salamander", "park", "squarish", "blokky", "babyb", "snake", "linkin", "ww", "turtle", "penguin", "zappa", "garrix", "ant", "pentapod"]

    n = len(names)
    table = np.zeros((n,n))
    for i in range(n):
        table[i,i] = 1

    print(latexifyTable(names, table, horizontlaLimitation=7))
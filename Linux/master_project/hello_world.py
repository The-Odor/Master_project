print("hello world")

import numpy
import pickle

with open("test", "wb") as outfile:
    pickle.dump("teststring", outfile)
    

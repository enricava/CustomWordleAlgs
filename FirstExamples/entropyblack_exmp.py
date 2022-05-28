import numpy as np
from matplotlib import pyplot as plt

# Uses old precalculation
wordfile = 'allowed_words.txt'
savefile = 'datafile.npy'
words = np.loadtxt(wordfile, dtype = 'str')

with open(savefile,'rb') as f:
    matrix1=np.load(f)
    matrix2=np.load(f)

def safelog2(x):
    if x == 0: 
        return 0
    
    return np.log2(x)

e = 0
for f in matrix2[1134]/12972:
    e -= f * safelog2(f)

print(e)

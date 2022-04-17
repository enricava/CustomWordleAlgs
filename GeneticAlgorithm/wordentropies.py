import random as rnd
import numpy as np
import json
from tqdm import tqdm

wordfile = 'allowed_words.txt'
solutionfile = 'solutions.txt'
savefile = 'datafile.npy'

# Load allowed words into array
words = np.loadtxt(wordfile, dtype = 'str')
solutions = np.loadtxt(solutionfile, dtype = 'str')

# Load data matrices
with open(savefile,'rb') as f:
    matrix=np.load(f)      # word x word : pattern
    matrix2=np.load(f)      # word x pattern : freq

# Calculate the entropy of the words for the fitness function
def safelog2(x):
    if x <= 0: 
        return 0
    
    return np.log2(x)

def wordentropy(i):
    e = 0
    for f in matrix2[i]/len(words):
        e -= f * safelog2(f)
    return e
    
entropies = {}
for i in tqdm(range(len(words))):
    entropies[words[i]] = wordentropy(i)

with open('entropies.json', 'w') as f:
    json.dump(entropies, f)

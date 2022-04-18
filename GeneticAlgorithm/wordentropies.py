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
    entropies=np.load(f)      # word : entropy

d = {}
for i in tqdm(range(len(words))):
    d[words[i]] = entropies[i]

with open('entropies.json', 'w') as f:
    json.dump(d, f)

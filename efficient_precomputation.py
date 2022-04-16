import numpy as np
from tqdm import tqdm
from itertools import product
from matplotlib import pyplot as plt


wordfile = 'allowed_words.txt'
savefile = 'datafile.npy'
solutonfile = 'solutions.txt'

# Load allowed words into array
words = np.loadtxt(wordfile, dtype = 'str')
solutions = np.loadtxt(solutonfile, dtype = 'str')

# Generates pattern matrix using words1 as solutions and words2 as inputs
def pattern_matrix(words1, words2):

    list1 = np.array([[ord(c) for c in w] for w in words1], dtype=np.uint8)
    list2 = np.array([[ord(c) for c in w] for w in words2], dtype=np.uint8)

    l1 = len(list1)
    l2 = len(list2)

    # First we calculate every possible combination of letters
    # equalities[a,b,i,j] = words1[a][i] == words2[b][j]
    equalities = np.zeros((l1, l2, 5, 5), dtype=bool)
    for i, j in product(range(5), range(5)):
        equalities[:, :, i, j] = np.equal.outer(list1[:, i], list2[:, j])

    matrix = np.zeros((l1,l2), dtype=np.uint8)

    for i in range(5):
        # matches[a, b] : words1[a][i] == words2[b][i]
        matches = equalities[:, :, i, i].flatten()
        matrix.flat[matches] += 2 * 3**(4-i)
        for k in range(5):
            # Avoid amber pass
            equalities[:, :, k, i].flat[matches] = False
            equalities[:, :, i, k].flat[matches] = False

    for i, j in product(range(5), range(5)):
            matches = equalities[:, :, i, j].flatten()
            matrix.flat[matches] += 3**(4-i)
            #Avoid next passes
            for k in range(5):
                equalities[:, :, k, j].flat[matches] = False
                equalities[:, :, i, k].flat[matches] = False

    return matrix

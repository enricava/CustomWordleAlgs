#
#   DEPRECATED : DO NOT USE THIS FILE TO PRECOMPUTE PATTERNS / ENTROPIES 
#

import numpy as np
from tqdm import tqdm

wordfile = 'allowed_words.txt'
savefile = 'datafile.npy'

# Load allowed words into array
words = np.loadtxt(wordfile, dtype = 'str')

# Generates a wordle pattern from two words
# Pattern representation is in ternary base
# 0 : gray
# 1 : yellow
# 2 : green
# Left is most significant trit
# word2 is submitted word
def pattern(word1, word2):
    result = 0
    used1 = [False for _ in range(5)]
    used2 = [False for _ in range(5)]

    # Green pass
    for i in range(5):
        if word1[i] == word2[i]:
            result += 2 * 3**(4-i)
            used1[i] = True
            used2[i] = True

    # Amber pass      
    for (i,c1) in enumerate(word1):
        for (j,c2) in enumerate(word2):
            if c1 == c2 and not (used1[i] or used2[j]):
                result += 1 * 3**(4-j)
                used1[i] = True
                used2[j] = True
                break
    
    return result

# Generate patterns
matrix1 = np.zeros(shape=(words.size,words.size), dtype=np.uint8)   # word x word : pattern
matrix2 = np.zeros(shape=(words.size, 3**5), dtype=np.uint16)       # word x pattern : freq
for i in tqdm(range(words.size)):
    for j in range(words.size):
        p = pattern(words[i],words[j])
        matrix1[i,j] = p
        matrix2[j,p] += 1

with open(savefile, 'wb') as f:
    np.save(f, matrix1)
    np.save(f, matrix2)

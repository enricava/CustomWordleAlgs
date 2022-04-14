import numpy as np
from tqdm import tqdm

wordfile = 'allowed_words.txt'
savefile = 'cross_patterns_uint8.npy'

# Load allowed words into array
words = np.loadtxt(wordfile, dtype = 'str')


# Generates a wordle pattern from two words
# Pattern representation is in ternary base
# 0 : gray
# 1 : yellow
# 2 : green
# Left is most significant trit
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
matrix = np.zeros(shape=(words.size,words.size), dtype=np.uint8)
for i in tqdm(range(words.size)):
    for j in range(words.size):
        matrix[i,j] = pattern(words[i],words[j])

with open(savefile, 'wb') as f:
    np.save(f, matrix)

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

# This code is very inefficient

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

def safelog2(x):
    if x <= 0: 
        return 0
    
    return np.log2(x)

def best(pattern, input, valid):
    nextWords = []

    for row in range(words.size):   # Find the words that are compatible with this pattern
        if valid[row] == 1 and matrix[row,input] == pattern:
            nextWords.append(row)
        else:
            valid[row] = 0
        

    if len(nextWords)== 1:
        return nextWords[0]


    frequencies = np.zeros(shape=(words.size, 3**5), dtype=np.uint16)
    for word in nextWords:      # Compute freq of patterns in subspace
        for other in nextWords:
                frequencies[word, matrix[other, word]] += 1

    maxent = 0
    bestword = -1
    for word in nextWords:
        e = 0
        for f in frequencies[word]/len(nextWords):
            e -= f * safelog2(f)
        if e > maxent:
            maxent = e
            bestword = word

    valid[bestword] = 0

    return bestword

def play(solution, starter):
    nextI = starter
    next = words[starter]
    #used = [words.searchsorted('tares')]
    valid = np.ones(shape=words.size)
    for attempt in range(8):
        p = pattern(solution, next)
        if p == 242:
            break
        nextI = best(p, nextI, valid)
        next = words[nextI]
        #used.append(nextI)
    #print([words[u] for u in used])
    return attempt

def simulations():
    results = [0 for _ in range(8)]
    for solution in tqdm(solutions):
        results[play(solution, words.searchsorted('tares'))] += 1
    print(results)
    score = 0
    for i in range(len(results)):
        score += (i+1)*results[i]
    score /= len(solutions)
    print('Score =' + str(score))
    plt.title('Simulations results')
    plt.ylabel('Frequency')
    plt.xlabel('Score')
    plt.bar(range(1,9), results)
    plt.savefig('sim5.png', dpi=1000, transparent=True)
    plt.show()

simulations()


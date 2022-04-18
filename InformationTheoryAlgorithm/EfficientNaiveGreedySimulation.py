import numpy as np
from tqdm import tqdm
from itertools import product
from matplotlib import pyplot as plt


wordfile = 'allowed_words.txt'
savefile = 'datafile.npy'
solutionfile = 'solutions.txt'

# Load allowed words into array
words = np.loadtxt(wordfile, dtype = 'str')
solutions = np.loadtxt(solutionfile, dtype = 'str')

# Generates pattern matrix using words1 as inputs and words2 as solutions
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

def make_guess(wordlist):
    patterns = pattern_matrix(wordlist, wordlist)
    l = len(wordlist)
    maxe = -1
    best = -1
    for i,w in enumerate(patterns):
        # List of distinct patterns and frequencies
        ps, counts = np.unique(w, return_counts=True)
        e = - np.dot(counts/l, np.log2(counts/l))
        if e > maxe:
            maxe = e
            best = i
    return best, wordlist[best]

def play(solution, allowed=words, starter='tares'):
    remaining_words = np.array(allowed, dtype = 'str')
    #answers = [starter]
    next = starter
    for attempt in range(9):
        pattern = pattern_matrix([next], [solution])[0,0]
        if pattern == 242:
            break

        remaining_patterns = pattern_matrix([next], remaining_words)
        remaining_words = remaining_words[remaining_patterns.flatten() == pattern]
        i, next = make_guess(remaining_words)
        remaining_words = np.delete(remaining_words, i)
        #answers.append(next)
    #return answers
    return attempt

def simulations(starter='tares'):
    results = [0 for _ in range(9)]
    for solution in tqdm(solutions):
        results[play(solution, words, starter)] += 1
    print(results)
    score = 0
    for i in range(len(results)):
        score += (i+1)*results[i]
    score /= len(solutions)
    print('Score =' + str(score))
    plt.title('Simulations results')
    plt.ylabel('Frequency')
    plt.xlabel('Score')
    plt.bar(range(1,10), results)
    plt.savefig('sim.png', dpi=1000, transparent=True)
    plt.show()

simulations('tares')
#print(play('merry'))

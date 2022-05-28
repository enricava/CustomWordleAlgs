import numpy as np
from tqdm import tqdm
from itertools import product
from matplotlib import pyplot as plt
from functools import cache
from os.path import isfile
from os.path import isdir
from os import mkdir

wordfile = 'allowed_words.txt'
solutionfile = 'solutions.txt'
priorityfile = 'word_priority.npy'
datafile = 'datafile.npy'

filenames = [wordfile,solutionfile,priorityfile,datafile]
present = np.array([isfile(f) for f in filenames])
if not present.all():
    [print('Missing', filenames[i]) for i in range(len(filenames)) if not present[i]]
    exit()

if not isdir('simulations2'):
    mkdir('simulations2')

# Load allowed words into array
words = np.loadtxt(wordfile, dtype = 'str')
word_dict = {words[i]:i for i in range(words.size)}
solutions = np.loadtxt(solutionfile, dtype = 'str')

# Load word priority file
with open(priorityfile,'rb') as f:
    word_priority=np.load(f)

# Load pattern matrix and entropies file
with open(datafile,'rb') as f:
    np.load(f)  #   We do not need the pattern matrix now
    entropies = np.load(f)

# Given a word str returns position in allowed_words.txt
def get_position_of_word(word):
    return word_dict[word]

# Given a word str returns its priority
def get_word_priority(word):
    return word_priority[get_position_of_word(word)]

# Returns best starter for given mode
@cache
def get_starter(mode='naive'):
    if mode == 'naive':
        sorted_entropies = np.argsort(entropies)
        return words[sorted_entropies[-1]]
    elif mode == 'greedy1':
        p_times_e = np.array([word_priority[i]*entropies[i] for i in range(words.size)])
        sorted_p_times_e = np.argsort(p_times_e)
        return words[sorted_p_times_e[-1]]

# Generates pattern matrix using words1 as inputs and words2 as solutions
@cache
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

# Makes next guess using entropy and priority
@cache
def make_guess(wordlist, mode):
    patterns = pattern_matrix(tuple(wordlist), tuple(wordlist))
    l = len(wordlist)
    maxe = -1
    best = -1
    for i,w in enumerate(patterns):
        # List of distinct patterns and frequencies
        ps, counts = np.unique(w, return_counts=True)
        e = - np.dot(counts/l, np.log2(counts/l))
        if mode == 'greedy1':
            e *= get_word_priority(wordlist[i])
        if e > maxe:
            maxe = e
            best = i
            
    return best, wordlist[best]


def play(solution, allowed=words, starter='tares', mode='naive'):
    remaining_words = np.array(allowed, dtype = 'str')
    if mode == 'greedy2':
        remaining_words = np.array(solutions)
    #answers = [starter]
    next = starter
    for attempt in range(9):
        pattern = pattern_matrix(tuple([next]), tuple([solution]))[0,0]
        if pattern == 242:
            break

        remaining_patterns = pattern_matrix(tuple([next]), tuple(remaining_words))
        remaining_words = remaining_words[remaining_patterns.flatten() == pattern]
        i, next = make_guess(tuple(remaining_words), mode)
        remaining_words = np.delete(remaining_words, i)
        #answers.append(next)
    #return answers
    return attempt

# Simulates all possible games
# Modes: 
#   naive -> ranks words by entropy
#   greedy1 -> ranks words by entropy*priority
def simulations(mode, setstarter = None):
    if setstarter:
        starter = setstarter
    else:
        starter = get_starter(mode)

    print('Using [', starter, '] as starter for [', mode, '] simulation')

    results = [0 for _ in range(9)]
    for solution in tqdm(solutions):
        results[play(solution, words, starter, mode)] += 1

    score = 0
    for i in range(len(results)):
        score += (i+1)*results[i]
    score /= len(solutions)

    print('Results =',results)
    print('Score =', score)
    plt.title('Simulation results '+ mode + ' ' + starter)
    plt.suptitle(str(results) + ' ' + str(score))
    plt.ylabel('Frequency')
    plt.xlabel('Score')
    plt.bar(range(1,10), results)
    plt.savefig('./simulations2/' + mode + starter +'.png', dpi=1000, transparent=True)
    plt.show()

while True:
    mode = input('Choose mode: [naive] or [greedy1] or [greedy2]\n')
    setstarter = input('Choose starter word or press [enter]\n')
    if mode != 'naive' and mode != 'greedy1' and mode != 'greedy2':
        mode = 'naive'
    if len(setstarter)!= 5:
        setstarter = ''

    simulations(mode, setstarter)
    ans = input('Stop simulations? y/n\n')
    if ans == 'y':
        exit()

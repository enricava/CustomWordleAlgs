import random as rnd
import numpy as np
from itertools import product
from functools import cache
from matplotlib import pyplot as plt
from tqdm import tqdm

wordfile = 'allowed_words.txt'
solutionfile = 'solutions.txt'
datafile = 'datafile.npy'
priorityfile = 'word_priority.npy'

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

# Get word position in 'words'
def get_word_position(word):
    if word in word_dict:
        return word_dict[word]
    else:
        return -1

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

#--------------------------------------------------------------
#                    GENETIC ALGORITHM
#--------------------------------------------------------------

# Generate the initial population: N random individuals (words)
def init(N):
    gen = []
    while len(gen) != N:
        index = rnd.randrange(len(words))
        gen.append(index)
    return gen

# Fitness function
def fitness(word):
    return entropies[word] * word_priority[word]

# Selection Operator
def selection(population):
    totalFitness = 0
    individuals = []
    for w in population:
        f = fitness(w)
        totalFitness += f
        individuals.append([w,f])

    probs = np.empty(len(individuals))
    probs[0] = individuals[0][1]/totalFitness
    for i in range(1,len(individuals)):
        probs[i] = probs[i-1] + individuals[i][1]/totalFitness

    r = rnd.uniform(0,1)
    p = np.searchsorted(probs,r)
    return individuals[p][0]

# Crossover Operator
def crossover(remaining_words):
    k = rnd.randint(0,len(remaining_words)-1)
    return get_word_position(remaining_words[k])

# Mutation Operator: Changes a letter of a word, making sure the result
# is a real word accepted by Wordle. The variable maxTries is used for
# same cases in which changing any letter of the word results in a 
# non-existing word
def mutation(child, maxTries = 100):
    found = False
    tries = 0    
    while not found and tries < maxTries:
        w = list(words[child])
        i = rnd.randint(0,4)
        w[i] = chr(rnd.randrange(ord('a'), ord('z')))
        new_word = ''.join(map(str,w))
        k = get_word_position(new_word)
        found = k > 0 and k < len(words) and words[k] == new_word and new_word != words[child]
        tries += 1
    if found:
        return k
    else:
        return child

# Genetic Algorithm to solve Wordle
# N: size of the generation
# mutProb: probability for mutation
# solution: solution for the game
def geneticAlgorithm(N, mutProb, solution): 

    population = init(N)
    attempts = []
    remaining_words = words

    while True:
        best = selection(population)
        attempts.append(words[best])
        pattern = pattern_matrix(tuple([words[best]]), tuple([solution]))[0,0]
        if pattern == 242:
            break
       
        remaining_patterns = pattern_matrix(tuple([words[best]]), tuple(remaining_words))
        remaining_words = remaining_words[remaining_patterns.flatten() == pattern]
        nextGen = []
        for i in range(N):
            child = crossover(remaining_words)
            r = rnd.uniform(0,1)
            if r < mutProb:
                child = mutation(child)
            nextGen.append(child)
        population = nextGen

    return attempts

print(geneticAlgorithm(100, 0.1, 'merry'))

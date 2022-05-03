import random as rnd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

wordfile = 'allowed_words.txt'
solutionfile = 'solutions.txt'
datafile = 'datafile.npy'

# Load allowed words into array
words = np.loadtxt(wordfile, dtype = 'str')
solutions = np.loadtxt(solutionfile, dtype = 'str')
word_dict = {words[i]:i for i in range(words.size)}

# Load data matrices
with open(datafile,'rb') as f:
    pattern_matrix=np.load(f)      # word x word : pattern
    entropies=np.load(f)      # word : entropy

# Get word position in 'words'
def get_word_position(word):
    if word in word_dict:
        return word_dict[word]
    else:
        return -1

# Get pattern between words as list
# Pattern representation is in ternary base
# 0 : gray
# 1 : yellow
# 2 : green
def get_pattern(word1_position, word2_position):
    pattern = pattern_matrix[word1_position, word2_position]
    list_pattern = np.zeros(5, dtype=np.uint8)
    for i in range(5):
        if pattern % 3 == 1:
            list_pattern[4-i] = 1
        elif pattern % 3 == 2:
            list_pattern[4-i] = 2
        pattern = pattern // 3
    return list_pattern

# Get entropy of a word
def get_entropy(word_position):
    return entropies[word_position]

#--------------------------------------------------------------
#                    GENETIC ALGORITHM
#--------------------------------------------------------------

# Generate the initial population: N random individuals (words)
def init(N):
    gen = []
    while len(gen) != N:
        index = rnd.randrange(len(words))
        gen.append(words[index])
    return gen

# Fitness function: The more consistent the word is with the
# information that we have about the solution, the better it is
def fitness(w, guessed, pattern, gray):
    f = get_entropy(get_word_position(w))
    if guessed != '-----': # No information
        for i in range(5):
            if pattern[i] == 2 and guessed[i] == w[i]:
                f += 5
            elif pattern[i] == 1 and guessed[i] != w[i] and guessed[i] in w:
                f += 2
            if w[i] in gray:
                f -= 2
    return f

# Selection Operator: Returns the best individual of the population,
# which will be used as an attempt for guessing the solution
def selection(population, guessed, pattern, gray):
    bestfit = fitness(population[0],guessed, pattern, gray)
    bestword = population[0]
    for i in range(1,len(population)):
        fit = fitness(population[i],guessed, pattern, gray)
        if fit >= bestfit:
            bestfit = fit
            bestword = population[i]
    return bestword

# Mutation Operator: Changes a letter of a word, making sure the result
# is a real word accepted by Wordle. The variable maxTries is used for
# same cases in which changing any letter of the word results in a 
# non-existing word
def mutation(child, attempts, maxTries = 100):
    found = False
    tries = 0    
    while not found and tries < maxTries:
        w = list(child)
        i = rnd.randint(0,4)
        w[i] = chr(rnd.randrange(ord('a'), ord('z')))
        new_word = ''.join(map(str,w))
        k = get_word_position(new_word)
        found = k > 0 and k < len(words) and words[k] == new_word and new_word != child
        found = found and not new_word in attempts
        tries += 1
    if found:
        return words[k]
    else:
        return child

# Crossover Operator: Return a word that is consistent with the
# current information that we have about the solution
def crossover(guessed, colors, attempts):
    found = False
    while not found:
        k = rnd.randint(0,len(words)-1)
        word = words[k]
        if not word in attempts:
            found = True
            for i in range(5):
                if colors[i] == 2:
                    found = found and word[i] == guessed[i]
                elif colors[i] == 1:
                    found = found and (word[i] != guessed[i] and guessed[i] in word)
    return word

# Simulates an attempt os submitting a word to the game to guess the
# solution and updates all the information we have about it
def attempt(best, solution, guessed, colors, gray):
    newpattern = get_pattern(get_word_position(best), get_word_position(solution))
    for i in range(5):
        if newpattern[i] > colors[i]:
            colors[i] = newpattern[i]
            guessed[i] = best[i]
        if newpattern[i] == 0:
            gray.add(best[i])
    return guessed, colors, gray

# Genetic Algorithm to solve Wordle
# N: size of the generation
# mutProb: probability for mutation
# solution: solution for the game
def geneticAlgorithm(N, mutProb, solution):    
    colors = [0,0,0,0,0]
    guessed = ['-','-','-','-','-',]
    gray = set()
    attempts = []
    found = False

    population = init(N)

    while not found:
        best = selection(population, guessed, colors, gray) 
        guessed, colors, gray = attempt(best, solution, guessed, colors, gray)
        attempts.append(best)

        if colors == [2,2,2,2,2]:
            found = True
            break
       
        nextGen = []
        for i in range(N):
            child = crossover(guessed, colors, attempts)
            r = rnd.uniform(0,1)
            if r < mutProb:
                child = mutation(child, attempts)
            nextGen.append(child)
        population = nextGen

    return attempts

print(geneticAlgorithm(50, 0.1, 'merry'))

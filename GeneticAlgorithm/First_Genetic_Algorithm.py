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
    gen = set()
    while len(gen) != N:
        index = rnd.randrange(len(words))
        gen.add(words[index])
    return list(gen)

# Fitness function: The more consistent the word is with the
# information that we have about the solution, the better it is
def fitness(word, best, pattern, gray):
    f = get_entropy(get_word_position(word))
    if best != '-----': # Not the first attempt
        for i in range(5):
            if pattern[i] == 2 and best[i] == word[i]:
                f += 5
            elif pattern[i] == 1 and best[i] != word[i] and best[i] in word:
                f += 2
            if word[i] in gray:
                f -= 2
    return f

# Selection Operator: Returns the best individual of the population,
# which will be used as an attempt for guessing the solution
def selection(population, best, pattern, gray):
    bestfit = 0
    for w in population:
        fit = fitness(w,best, pattern, gray)
        if fit > bestfit:
            bestfit = fit
            bestword = w
    return bestword

# Mutation Operator: Changes a letter of a word, making sure the result
# is a real word accepted by Wordle. The variable maxTries is used for
# same cases in which changing any letter of the word results in a 
# non-existing word
def mutation(child, maxTries = 100):
    found = False
    tries = 0    
    while not found and tries < maxTries:
        newchild = ""
        i = rnd.randint(0,4)
        l = chr(rnd.randrange(ord('a'), ord('z')))
        for j in range(5):
            if j == i:
                newchild += l
            else:
                newchild += child[j]
        k = get_word_position(newchild)
        found = k > 0 and k < len(words) and newchild != child and words[k] == newchild
        tries += 1
    if found:
        return newchild
    else:
        return child

# Crossover Operator: Return a word that is consistent with the
# current information that we have about the solution
def crossover(guessed, colors, attempts):
    found = False
    while not found:
        k = rnd.randint(0,len(words)-1)
        word = words[k]
        found = not word in attempts
        for i in range(5):
            if colors[i] == 2:
                found = found and word[i] == guessed[i]
            elif colors[i] == 1:
                found = found and (word[i] != guessed[i] and guessed[i] in word)
            else:
                found = found and (not guessed[i] in word)
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
                child = mutation(child)
            nextGen.append(child)
        population = nextGen

    return attempts

# Solves Wordle 'iterations' times with 'word' as the solution
def simulate(word, iterations, N, mutationProb):
    results = [0 for _ in range(15)]
    for _ in tqdm(range(iterations)):
        sol = geneticAlgorithm(N, mutationProb, word)
        results[len(sol)-1] += 1
    print('Results: ' + str(results))
    score = 0
    for i in range(len(results)):
        score += (i+1)*results[i]
    score /= iterations
    print('Score = ' + str(score))
    plt.title('Simulation results for the word ' + str(word))
    plt.ylabel('Frequency')
    plt.xlabel('Score')
    plt.bar(range(1,16), results)
    plt.savefig('simGene.png', dpi=1000, transparent=True)
    plt.show()

simulate('merry', 1, 50, 0.1)

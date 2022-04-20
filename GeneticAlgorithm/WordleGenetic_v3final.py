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

# Load data matrices
with open(datafile,'rb') as f:
    pattern_matrix=np.load(f)      # word x word : pattern
    entropies=np.load(f)      # word : entropy

# Get word position in 'words'
def get_word_position(word): # O(logn)
    return words.searchsorted(word)

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

# Initial population: N random words
def init(N):
    gen = set()
    while len(gen) != N:
        index = rnd.randrange(len(words))
        gen.add(words[index])
    return list(gen)

# Fitness function
# f(w) = H(X_w) + 5*|green matches| + 2*|yellow matches|
# With this function, words closer to the solution will
# be considered better. We use entropy to choose between
# words that have similar patterns
def fitness(w, guessed, amber):
    f = get_entropy(get_word_position(w))
    for i in range(5):
        if guessed[i] == w[i]:
            f += 5
    for l in amber:
        if l in w:
            f += 2
    return f

# Selection Operator
# We choose the best individual in the population,
# which we will use to make an 'attempt' for the game
def selection(population, guessed, amber):
    bestFit = 0
    best = ''
    for w in population:
        f = fitness(w, guessed, amber)
        if f > bestFit:
            bestFit = f
            best = w
    return best
    
# Crossover Operator
# With the information that we have from previous
# words chosen as inputs for the game, we 
# generate words that match that information
def crossover(guessed, amber, gray, attempts):
    k = rnd.randint(0,len(words)-1)
    found = False
    while not found:
        found = True
        word = words[k]
        # word already used
        if word in attempts:
            found = False
        else:
            # matches the green letters
            for i in range(5):
                if guessed[i] != '-':
                    found = found and guessed[i] == word[i]
            # contains the amber letters
            for l in amber:
                found = found and l in word
            # does not conatin gray letters
            for l in gray:
                found = found and not (l in word)               
        k = (k+1) % len(words)
    return word

# Mutation Operator
# Randomly changes a letter of the word
def mutation(child, mutProb):
    r = rnd.uniform(0, 1)
    if r < mutProb:
        index = {0,1,2,3,4}
        k = rnd.randint(0,len(child)-1)
        index.remove(k)
        # takes a word with the same 
        # letters as child except for child[k]
        i = rnd.randint(0,len(words)-1)        
        found = False
        while not found:
            found = True
            for j in index:
                found = found and child[j] == words[i][j]
            if found:
                child = words[i]
            else:
                i = (i+1) % len(words)
    return child

# This function simulates an 'attempt' in the game.
# That is, 'best' would be the input for the game
# and we update the information that we have
# (matches in 'guessed', amber letters in 'amber'
# and gray letters in 'gray')
def attempt(best, sol, guessed, amber, gray):
    newpat = get_pattern(get_word_position(best),get_word_position(sol))
    for i in range(5):
        if newpat[i] == 2:
            guessed[i] = best[i]
            # Found position for an amber letter
            if best[i] in amber:
                amber.remove(best[i])
        elif newpat[i] == 1:
            amber.add(best[i])
        else:
            if not (best[i] in guessed or best[i] in amber):
                gray.add(best[i])
    return guessed, amber, gray

# Genetic Algorithm to solve Wordle
# N: size of the generation
# mutProb: probability for mutation
# solution: solution for the game
def geneticAlgorithm(N, mutProb, solution):    
    guessed = ['-','-','-','-','-']
    amber = set()
    gray = set()

    found = False
    attempts = []
    population = init(N)

    while not found:
        best = selection(population, guessed, amber) 
        guessed, amber, gray = attempt(best, solution, guessed, amber, gray)
        attempts.append(best)
        if not '-' in guessed:
            found = True
            break
       
        nextGen = []
        for _ in range(N):
            child = crossover(guessed, amber, gray, attempts)
            mutation(child, mutProb)
            nextGen.append(child)
        population = nextGen

    return attempts

# Solves Wordle 'iterations' times with 'word' as the solution
def simulate(word, iterations, N, mutationProb):
    results = [0 for _ in range(15)]
    for _ in tqdm(range(iterations)):
        sol = geneticAlgorithm(N, mutationProb, word)
        results[len(sol)-1] += 1
    print(results)
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

simulate('merry', 20, 750, 0.1)

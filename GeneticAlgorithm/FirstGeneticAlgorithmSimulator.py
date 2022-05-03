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
        found = not word in attempts
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
    plt.savefig('simFirstGenetic.png', dpi=1000, transparent=True)
    plt.show()

#--------------------------------------------------------------
#               SIMULATION OF ALL THE WORDS
#--------------------------------------------------------------
# Takes around 1h20min to simulate all words
def simulation(N, mutationProb):
    results = [0 for _ in range(15)]
    extratries = 0
    maxtries = 0
    extrawords= 0
    for w in tqdm(solutions):
        sol = geneticAlgorithm(N, mutationProb, w)
        # Words with more than 15 tries
        if len(sol) > 15:
            extratries += len(sol)
            extrawords += 1
            if len(sol) > maxtries:
                maxtries = len(sol)
        else:
            results[len(sol)-1] += 1
    print('Results: ' + str(results))
    score = 0
    for i in range(len(results)):
        score += (i+1)*results[i]
    score += extratries
    score /= len(solutions)
    print('Average Score = ' + str(score))
    print('Max number of tries = ' + str(maxtries))
    print('Number of words with more than 15 tries = ' + str(extrawords))
    plt.title('Simulations results')
    plt.ylabel('Frequency')
    plt.xlabel('Score')
    plt.bar(range(1,16), results)
    plt.savefig('simFirstGenetic.png', dpi=1000, transparent=True)
    plt.show()

#--------------------------------------------------------------
#                       SIMULATIONS
#--------------------------------------------------------------

while True:
    mode = input('Play once [0] or Simulate a word several times [1]\n')
    if mode == '0':
        which = input('Choose word [0] or Random word [1]\n')
        if which == '0':
            sol = input('Enter the word you want the algorithm to find\n')
            while not sol in solutions:
                sol = input('Word not valid. Introduce another one\n')
        else:
            sol = solutions[rnd.randint(0,len(solutions)-1)]
        print('Algorithm playing the game to find the word \'' + sol + '\'...')
        attempts = geneticAlgorithm(50, 0.1, sol)
        print('Attempts: ' + str(attempts))

        ans = input('Stop simulations? y/n\n')
        if ans == 'y':
            exit()
    else:
        which = input('Choose word [0] or Random word [1]\n')
        if which == '0':
            sol = input('Enter the word you want the algorithm to find\n')
            while not sol in solutions:
                sol = input('Word not valid. Introduce another one\n')
        else:
            sol = solutions[rnd.randint(0,len(solutions)-1)]
        sims = input('Enter the number of simulations you want to do\n')
        if not sims.isnumeric() or int(sims) < 1:
            iter = 10
        else:
            iter = int(sims)
        print('Making ' + str(iter) + ' simulations to find the word \'' + sol + '\'...')
        simulate(sol, iter , 50, 0.1)

        ans = input('Stop simulations? y/n\n')
        if ans == 'y':
            exit()
            

import random as rnd
import numpy as np
import json
from tqdm import tqdm

wordfile = 'allowed_words.txt'
solutionfile = 'solutions.txt'
entropyfile = 'entropies.json'

# Load allowed words into array
words = np.loadtxt(wordfile, dtype = 'str')
solutions = np.loadtxt(solutionfile, dtype = 'str')
# Load entropies for fitness function
with open('entropies.json') as f:
    entropy = json.load(f)

#--------------------------------------------------------------
#                    GENETIC ALGORITHM
#--------------------------------------------------------------

# Fitness function
# f(w) = H(X_w) + 5*|green matches| + 2*|yellow matches|
# Also returns the pattern of the word
# Pattern representation is in ternary base
# 0 : gray
# 1 : yellow
# 2 : green

def fitness(w, sol):
    f = entropy[w]
    pattern = [0 for _ in range(5)]

    # Calculate the pattern
    used1 = [False for _ in range(5)]
    used2 = [False for _ in range(5)]

    # Green pass
    for i in range(5):
        if w[i] == sol[i]:
            f += 5
            pattern[i] = 2
            used1[i] = True
            used2[i] = True

    # Amber pass
    for (i,c1) in enumerate(w):
        for (j,c2) in enumerate(sol):
            if c1 == c2 and not (used1[i] or used2[j]):
                f += 2
                pattern[i] = 1
                used1[i] = True
                used2[j] = True
                break
    return f, pattern

# Initial population: N random words
def init(N):
    gen = []
    for i in range(N):
        index = rnd.randrange(len(words))
        gen.append(words[index])
    return gen

# Calculate the fitness value of the entire population
# Returns the total fitness value, a list of pairs
# [individual, fitness, pattern] and the best indvidual
# of the population 
def getFitness(population, sol):
    fitPopulation = []
    totalFitness = 0
    for w in population:
        fit, pattern = fitness(w, sol)
        totalFitness += fit
        fitPopulation.append([w, fit, pattern])
    return totalFitness, fitPopulation

def bestIndividual(population, sol):
    bestFit = 0
    for w in population:
        fit, _ = fitness(w, sol)
        if fit > bestFit:
            best = w
            bestFit = fit
    return best

# Selection Operator
# Return 2 parents with their patterns
def selection(population, sol):
    totalFit, gen = getFitness(population, sol)
    # weights[i] = [p_i, q_i], for i-th word in gen
    weights = [[gen[0][1]/totalFit, gen[0][1]/totalFit]]
    for i in range(1,len(gen)):
        [_, fit, _] = gen[i]
        p_i = fit/totalFit
        q_i = p_i + weights[i-1][1]
        weights.append([p_i, q_i])
    
    # Busqueda lineal -> pasar a logaritmica porque los q_i estan ordenados
    r = rnd.uniform(0, 1)
    if r <= weights[0][1]:
        parent1 = gen[0][0]
        pattern1 = gen[0][2]
    else:
        for i in range(1,len(weights)):
            if r <= weights[i][1]:
                parent1 = gen[i][0]
                pattern1 = gen[i][2]
                break
            
    parent2 = parent1
    while parent2 == parent1:
        r = rnd.uniform(0, 1)
        if r <= weights[0][1]:
            parent2 = gen[i][0]
            pattern2 = gen[i][2]
        else:
            for i in range(1,len(weights)):
                if r <= weights[i][1]:
                    parent2 = gen[i][0]
                    pattern2 = gen[i][2]
                    break

    return parent1, parent2, pattern1, pattern2

# Checks if a word matches a pattern
def match(word, pattern, letters):
    b = True
    for i in range(5):
        if pattern[i] == 2:
            b = b and (word[i] == letters[i])
        elif pattern[i] == 1:
            b = b and (letters[i] in word)
    return b

# Crossover Operator
def crossover(parent1, parent2, pat1, pat2):
    pattern = []
    letters = []
    for i in range(5):
        if pat1[i] > pat2[i]:
            pattern.append(pat1[i])
            letters.append(parent1[i])
        else:
            pattern.append(pat2[i])
            letters.append(parent2[i])
       
    # Search for a word with the restrictions from
    # pattern and letters
    children = []
    k = rnd.randint(0,len(words))
    while not len(children) == 2:
        if match(words[k], pattern, letters):
            children.append(words[k])
            k = rnd.randint(0,len(words))
        else:
            k = (k+1) % len(words)

    return children[0], children[1]

def mutation(child, mutProb):
    r = rnd.uniform(0, 1)
    if r < mutProb:
        k = rnd.randint(0,len(child)-1)
        child[k] = 1 - child[k]
    return child

def geneticAlgorithm(N, crossProb, mutProb, solution):
    population = init(N)
    attempts = []
    selected = []
    found = False
    while not found:
        nextGen = []
        for i in range(N//2):
            parent1, parent2, pat1, pat2 = selection(population, solution)
            selected += [parent1,parent2]
            # Check if we found the solution
            if parent1 == solution or parent2 == solution:
                found = True
                break
            r = rnd.uniform(0, 1)
            if r < crossProb:
                child1, child2 = crossover(parent1, parent2, pat1, pat2)
            else:
                child1, child2 = parent1, parent2
            #mutation(child1, mutProb)
            #mutation(child2, mutProb)
            nextGen += [child1, child2]
        attempts += [bestIndividual(selected, solution)]
        population = nextGen
    return attempts


N = 50
crossProb = 0.7
mutProb = 0.1
for i in range(20):
    sol = geneticAlgorithm(N,crossProb,mutProb, 'merry')
    print("Solution: " + str(sol))


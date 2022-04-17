import random as rnd
import numpy as np
import json
from sklearn.metrics import fbeta_score
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

# Initial population: N random words
def init(N):
    gen = []
    while len(gen) != N:
        index = rnd.randrange(len(words))
        if not words[index] in gen:
            gen.append(words[index])
    return gen

# Calculates the pattern of colors for word w
# Pattern representation is in ternary base
# 0 : gray
# 1 : yellow
# 2 : green
def getPattern(w, sol):
    # Calculate the pattern
    pattern = [0 for i in range(5)]
    used1 = [False for _ in range(5)]
    used2 = [False for _ in range(5)]

    # Green pass
    for i in range(5):
        if w[i] == sol[i]:
            pattern[i] = 2
            used1[i] = True
            used2[i] = True

    # Amber pass
    for (i,c1) in enumerate(w):
        for (j,c2) in enumerate(sol):
            if c1 == c2 and not (used1[i] or used2[j]):
                pattern[i] = 1
                used1[i] = True
                used2[j] = True
                break
    return pattern

# Fitness function
# f(w) = H(X_w) + 5*|green matches| + 2*|yellow matches|
def fitness(w, pattern):
    f = entropy[w]
    for i in range(5):
        if pattern[i] == w[i]:
            f += 5
    return f

# Selection Operator
def selection(population, pattern):
    bestFit = 0
    best = ''
    for w in population:
        f = fitness(w, pattern)
        if f > bestFit:
            bestFit = f
            best = w
    return best
    

# Crossover Operator
def crossover(pattern, grayletters, rejects):
    k = rnd.randint(0,len(words)-1)
    found = False
    while not found:
        found = True
        word = words[k]
        if word in rejects:
            found = False
        else:
            for i in range(5):
                if pattern[i] != '-':
                    found = found and pattern[i] == word[i]
            for l in grayletters:
                found = found and not (l in word)
        if rnd.uniform(0,1) < 0.5:
            found = False
        k = (k+1)%len(words)
    return word

def mutation(child, mutProb):
    r = rnd.uniform(0, 1)
    if r < mutProb:
        k = rnd.randint(0,len(child)-1)
        child[k] = 1 - child[k]
    return child

def attempt(best, sol, p, gray):
    newpat = getPattern(best,sol)
    for i in range(5):
        if newpat[i] == 2:
            p[i] = best[i]
        
        if newpat[i] == 0:
            if not best[i] in p and not best[i] in gray:
                gray.append(best[i])
    return p, gray

def geneticAlgorithm(N, mutProb, solution):    
    pattern = ['-','-','-','-','-']
    grayletters = []
    rejects = []

    found = False
    attempts = []
    population = init(N)

    while not found:
        best = selection(population, pattern) 
        pattern, grayletters = attempt(best, solution, pattern, grayletters)
        attempts.append(best)
        if not '-' in pattern:
            found = True
            break
        rejects.append(best)
       
        """
        print(population)       
        print("------intento : "+ str(best))
        print("patron : " + str(pattern))
        print("letras grises: " + str(grayletters))
        print("no validas" + str(rejects))
        """

        nextGen = []
        for i in range(N):
            child = crossover(pattern, grayletters, rejects)
            # mutation(child, mutProb)
            nextGen.append(child)
        population = nextGen

    return attempts


N = 100
crossProb = 0.7
mutProb = 0.1
for i in range(10):
    sol = geneticAlgorithm(N, mutProb, 'merry')
    print("Solucion: "+ str(sol))
    


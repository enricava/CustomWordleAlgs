import random as rnd
import numpy as np
from matplotlib import pyplot as plt
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

# Initial population: N different random words
def init(N):
    gen = set()
    while len(gen) != N:
        index = rnd.randrange(len(words))
        if not words[index] in gen:
            gen.add(words[index])
    return list(gen)

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
# With this function, words closer to the solution will
# be considered better. We use entropy to choose between
# words that have similar patterns
def fitness(w, guessed, amber):
    f = entropy[w]
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
def crossover(guessed, amber, gray, rejects):
    k = rnd.randint(0,len(words)-1)
    found = False
    while not found:
        found = True
        word = words[k]
        # word already used
        if word in rejects:
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
            # randomly skip a word to add variety
            if rnd.uniform(0,1) < 0.5:
                found = False
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
# (matches in 'guessed', amner letters in 'amber'
# and gray letters in 'gray')
def attempt(best, sol, guessed, amber, gray):
    newpat = getPattern(best,sol)
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
    rejects = set()
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
        rejects.add(best)
       
        """
        print(population)       
        print("------intento : "+ str(best))
        print("patron : " + str(guessed))
        print("amberillo: " + str(amber))
        print("grises: " + str(gray))
        print("no validas" + str(rejects))
        """

        nextGen = []
        for i in range(N):
            child = crossover(guessed, amber, gray, rejects)
            mutation(child, mutProb)
            nextGen.append(child)
        population = nextGen

    return attempts

def simulate(word, iterations, N, mutationProb):
    results = [0 for _ in range(8)]
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
    plt.bar(range(1,9), results)
    plt.savefig('simGene.png', dpi=1000, transparent=True)
    plt.show()

simulate('merry', 50, 100, 0.1)

"""
Problema:
    leery -> peery -> beery -> merry
    No sé cómo hacer que en leery te cuente que no puede haber una segunda e
    Cada cosa que intento hace que en algunos casos (no sé por qué) se quede colgado,
    o sea pongo las resticciones mal y no encuentra ninguna palabra
Solucionar eso si es posible y mejores funciones fitness son las optimizaciones que 
se me ocurren

Parece que entre 5 y 6 de media
Alguna vez hace 4 y los casos que se va a más a simple vista podrían quitarse con lo
de las letras repetidas
"""   

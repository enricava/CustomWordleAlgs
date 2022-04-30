import random as rnd
import numpy as np

def fitness(c):
    f = 0
    for i in range(len(c)):
        f += 2**(len(c)-1-i)*c[i]
    return f

def init(N):
    gen = []
    for i in range(N):
        c = [rnd.randrange(2) for i in range(5)]
        gen.append(c)
    return gen

def getFitness(population):
    totalFitness = 0
    individuals = []
    for w in population:
        f = fitness(w)
        totalFitness += f
        individuals.append([w,f])
    return totalFitness, individuals

def selection(population):
    totalFitness, individuals = getFitness(population)
        
    probs = np.empty(len(individuals))
    probs[0] = individuals[0][1]/totalFitness
    for i in range(1,len(individuals)):
        probs[i] = probs[i-1] + individuals[i][1]/totalFitness

    r1, r2 = rnd.uniform(0,1), rnd.uniform(0,1)
    p1 = np.searchsorted(probs,r1)
    p2 = np.searchsorted(probs,r2)

    return individuals[p1][0], individuals[p2][0]

def crossover(parent1, parent2):
    l = len(parent1)
    k = rnd.randint(0,l-2)
    child1 = parent1[0:k+1] + parent2[k+1:l]
    child2 = parent2[0:k+1] + parent1[k+1:l]
    return child1, child2

def mutation(child, mutProb):
    r = rnd.uniform(0, 1)
    if r < mutProb:
        k = rnd.randint(0,len(child)-1)
        child[k] = 1 - child[k]
    return child

def bestIndividual(population):
    _ , fit = getFitness(population)
    fit.sort()
    return fit[len(fit)-1][1]

def geneticAlgorithm(N, maxNG, crossProb, mutProb):
    population = init(N)
    numberGens = 0
    while numberGens != maxNG:
        nextGen = []
        for i in range(N//2):
            parent1, parent2 = selection(population)
            r = rnd.uniform(0, 1)
            if r < crossProb:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            mutation(child1, mutProb)
            mutation(child2, mutProb)
            nextGen += [child1, child2]
        population = nextGen
        numberGens += 1
    return bestIndividual(population)

N = 15
maxNG = 50
crossProb = 0.7
mutProb = 0.1
sol = geneticAlgorithm(N,maxNG,crossProb,mutProb)
print(sol)

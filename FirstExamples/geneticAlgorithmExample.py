import random as rnd

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
    fitPopulation = []
    totalFitness = 0
    for c in population:
        totalFitness += fitness(c)
        fitPopulation.append([fitness(c),c])
    return totalFitness, fitPopulation

def selection(population):
    totalFitness, gen = getFitness(population)
    weights = [[gen[0][1], gen[0][0]/totalFitness, gen[0][0]/totalFitness]]
    for i in range(1,len(gen)):
        chromosome = gen[i][1]
        p_i = gen[i][0]/totalFitness
        q_i = p_i + weights[i-1][2]
        weights.append([chromosome, p_i, q_i])
    
    # Mejorar búsquedas a log n
    r = rnd.uniform(0, 1)
    if r <= weights[0][2]:
        parent1 = weights[0][0]
    else:
        for i in range(1,len(weights)):
            if r <= weights[i][2]:
                parent1 = weights[i][0]
                break
            
    parent2 = parent1
    while parent2 == parent1:
        r = rnd.uniform(0, 1)
        if r <= weights[0][2]:
            parent2 = weights[0][0]
        else:
            for i in range(1,len(weights)):
                if r <= weights[i][2]:
                    parent2 = weights[i][0]
                    break

    return parent1, parent2

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
    print(population)
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
    print(population)
    return bestIndividual(population)


N = 15
maxNG = 50
crossProb = 0.7
mutProb = 0.1
sol = geneticAlgorithm(N,maxNG,crossProb,mutProb)
print(sol)
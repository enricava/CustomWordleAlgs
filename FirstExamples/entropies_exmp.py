import numpy as np

# Uses old precalculation
wordfile = 'allowed_words.txt'
savefile = 'datafile.npy'
words = np.loadtxt(wordfile, dtype = 'str')

with open(savefile,'rb') as f:
    matrix1=np.load(f)
    matrix2=np.load(f)

def safelog2(x):
    if x == 0: 
        return 0
    
    return np.log2(x)

d = {}
for i in range(len(words)):
    e = 0
    for f in matrix2[i]/12972:
        e -= f * safelog2(f)
    d[words[i]] = e

entropies = []
for w in sorted(d, key=d.get, reverse=True)[:10]:
    entropies.append('\\texttt{'+w.upper()+'}' + ' & '+ str(d[w]) + '\\\\')

for elem in entropies[:10]:
    print( elem)

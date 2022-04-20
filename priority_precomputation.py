from unittest.util import sorted_list_difference
import numpy as np
from tqdm import tqdm
from itertools import product
from matplotlib import pyplot as plt
#pip install wordfreq
from wordfreq import word_frequency

wordfile = 'allowed_words.txt'
wordpriorityfile = 'word_priority.npy'

words = np.loadtxt(wordfile, dtype='str')

# word -> frequency
word_freqs = np.array([word_frequency(w, 'en') for w in words])

# sorted word indexes by frequency
# sorted_freqs[0] : least frequent
# sorted_freqs[-1] : most frequent 
sorted_freqs = np.argsort(word_freqs) # increasing

# words ordered by frequency
#sorted_words = words[sorted_freqs]

word_range = np.linspace(-1,1,num=words.size)

def f(x, l=10):
    return 1/(1+np.e**(-l*x))

sigmoid_values = np.array(f(word_range,2))
plt.plot(word_range, sigmoid_values)
plt.show()
#print(sigmoid_values)

# word_sigmoid[word number] : priority of word
word_sigmoid = np.zeros(words.size)
for i,w in enumerate(sorted_freqs):
    word_sigmoid[w] = sigmoid_values[i]

print(word_sigmoid[words.searchsorted('about')])
print(word_sigmoid[words.searchsorted('tares')])
print(word_sigmoid[words.searchsorted('hello')])

with open(wordpriorityfile, 'wb') as f:
    np.save(f, word_sigmoid)


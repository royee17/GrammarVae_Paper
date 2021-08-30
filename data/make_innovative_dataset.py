import numpy as np
import nltk
from nltk.corpus import treebank as tb
from utils import many_one_hot
import h5py

# if __name__ == '__main__':
data = [" ".join(sent) for sent in tb.sents() if len(" ".join(sent)) < 210]
L = []

chars = list(set(c.lower() for word in nltk.corpus.treebank.words() for c in word))
chars.append(' ')
DIM = len(chars)

for line in data:
    line = line.strip()
    L.append(line)

count = 0
MAX_LEN = 210
OH = np.zeros((len(data), MAX_LEN, DIM))
for chem in L:
    indices = []
    for c in chem:
        indices.append(chars.index(c.lower()))
    if len(indices) < MAX_LEN:
        indices.extend((MAX_LEN-len(indices))*[DIM-1])
    OH[count, :, :] = many_one_hot(np.array(indices), DIM)
    count = count + 1

h5f = h5py.File('innovative_dataset.h5','w')
h5f.create_dataset('data', data=OH)
# h5f.create_dataset('chr',  data=chars)
h5f.close()

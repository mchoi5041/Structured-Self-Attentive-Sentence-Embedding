import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pickle

words = {}
idx = 0
word2idx = {}
# vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/42B.300d.dat', mode='w')
vectors = {}

with open('./data/GloVe/glove.42B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        if len(line) != 301:
            continue

        word = line[0]
        words[word] = idx
        word2idx[word] = idx
        
        vect = np.array(line[1:]).astype(np.float)
        vectors[idx] = vect
        
        idx += 1
    
# vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
# vectors.flush()

# pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
# pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))

with open('./data/GloVe/glove.42B.300d.pt', 'wb') as f:
    torch.save([words, vectors, 300], f)
    # pickle.dump([words, vectors], f)

# vs = torch.load('./data/GloVe/glove.42B.300d.pt')
# print(vs[0])
# print(vs[1])
# print(vs[2])
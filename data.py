import h5py
import nltk
import numpy as np
from grammar import zinc_grammar
import grammar_models


def grammar_loader():
    with h5py.File('data/eq2_grammar_dataset.h5', 'r') as h5f:
        return h5f['data'][:]


def str_loader():
    with h5py.File('data/eq2_str_dataset.h5', 'r') as h5f:
        return h5f['data'][:]


def to_one_hot(smiles, max_len, n_chars, parser):
    """ Encode a list of smiles strings to one-hot vectors """
    assert type(smiles) == list
    productions = zinc_grammar.GCFG.productions()
    prod_map = {k: ix for ix, k in enumerate(productions)}

    tokenize = grammar_models.get_zinc_tokenizer(zinc_grammar.GCFG)
    tokens = list(map(tokenize, smiles))

    parse_trees = [next(parser.parse(t)) for t in tokens]
    productions_seq = [tree.productions() for tree in parse_trees]
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int)
               for entry in productions_seq]
    one_hot = np.zeros((len(indices), max_len, n_chars), dtype=np.float32)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions), indices[i]] = 1.
        one_hot[i][np.arange(num_productions, max_len), -1] = 1.
    return one_hot


def make_zinc_data_set():
    f = open('data/250k_rndm_zinc_drugs_clean.smi', 'r')
    L = []

    for line in f:
        line = line.strip()
        L.append(line)
    f.close()

    MAX_LEN = 277
    NCHARS = len(zinc_grammar.GCFG.productions())
    parser = nltk.ChartParser(zinc_grammar.GCFG)

    OH = np.zeros((len(L), MAX_LEN, NCHARS))
    for i in range(0, len(L), 1000):
        print(f'Processing: i=[{str(i)} : {str(i+1000)}]')
        onehot = to_one_hot(L[i:i+1000], MAX_LEN, NCHARS, parser)
        OH[i:i+1000, :, :] = onehot


    try:
        h5f = h5py.File('data/zinc_grammar_dataset.h5', 'w')
        h5f.create_dataset('data', data=OH)
        h5f.close()

    except:
        print('h5 failed --> saving as npy')
        with open('data/zinc_grammar_dataset.npy', 'wb') as f:
            np.save(f, OH)

    print('Done')

if __name__ == '__main__':
    make_zinc_data_set()
    # GRAMMAR_DATA = grammar_loader()
    # STR_DATA = str_loader()
    # print('done')

# def load_zinc_data(data):

#     if not e:
#         e = k

#     with h5py.File(data, 'r') as h5f:
#         result = np.concatenate([h5f['data'][i::k] for i in range(s, e)])
#         return torch.FloatTensor(result)

#     h5f = h5py.File('data/zinc_grammar_dataset.h5', 'r')
#     data = h5f['data'][:]
#     h5f.close()

# GRAMMAR_DATA = grammar_loader()
# print(GRAMMAR_DATA.shape)
# => shape(batch_size: 100000, seq_length: 15, tokens: 12)

# STR_DATA = str_loader()
# print(STR_DATA.shape)
# => shape(batch_size: 100000, seq_length: 19, tokens: 12)

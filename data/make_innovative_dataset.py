import numpy as np
import nltk
import h5py
import pandas as pd
from grammar import mol_grammar
from grammar_models import get_zinc_tokenizer


def make_molecule_ds():
    df = pd.read_csv('qm9.csv')
    L = df['smiles'].values.tolist()

    MAX_LEN = 277
    NCHARS = len(mol_grammar.GCFG.productions())

    def to_one_hot(smiles):
        """ Encode a list of smiles strings to one-hot vectors """
        assert type(smiles) == list
        prod_map = {}
        for ix, prod in enumerate(mol_grammar.GCFG.productions()):
            prod_map[prod] = ix
        tokenize = get_zinc_tokenizer(mol_grammar.GCFG)
        tokens = list(map(tokenize, smiles))
        parser = nltk.ChartParser(mol_grammar.GCFG)
        parse_trees = [next(parser.parse(t)) for t in tokens]
        productions_seq = [tree.productions() for tree in parse_trees]
        indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
        one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
        for i in range(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions), indices[i]] = 1.
            one_hot[i][np.arange(num_productions, MAX_LEN), -1] = 1.
        return one_hot

    OH = np.zeros((len(L), MAX_LEN, NCHARS))
    for i in range(0, len(L), 100):
        print('Processing: i=[' + str(i) + ':' + str(i + 100) + ']')
        onehot = to_one_hot(L[i:i + 100])
        OH[i:i + 100, :, :] = onehot

    h5f = h5py.File('qm9_grammar_dataset.h5', 'w')
    h5f.create_dataset('data', data=OH)
    h5f.close()


if __name__ == '__main__':
    make_molecule_ds()
import numpy as np
import nltk
import pandas as pd
from grammar import mol_grammar
import selfies as sf

MAX_LEN = 277
NCHARS = len(mol_grammar.GCFG.productions())

def get_zinc_tokenizer(cfg):
    long_tokens = list(filter(lambda a: len(a) > 1, cfg._lexical_index.keys()))
    replacements = ['$', '%', '^']  # ,'&']
    assert len(long_tokens) == len(replacements)
    for token in replacements:
        assert not token in cfg._lexical_index

    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        tokens = []
        for token in smiles:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except:
                tokens.append(token)
        return tokens

    return tokenize

def fetch_data_set(data_set_name):
    if data_set_name == 'qm9':
        df = pd.read_csv('qm9.csv')
        L = df['smiles'].values.tolist()
    else:
        f = open('250k_rndm_zinc_drugs_clean.smi', 'r')
        L = []

        for line in f:
            line = line.strip()
            L.append(line)
            if len(L) == 100000:
                break
        f.close()

    return L

def smiles_to_one_hot(smiles):
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

def selfies_to_one_hot(smiles, selfies_alphabet=[], largest_selfies_len=None):
    encoding_list, encoding_alphabet, largest_molecule_len = get_selfie_encodings_for_dataset(smiles,
                                                                                              selfies_alphabet, largest_selfies_len)

    print('--> Creating one-hot encoding...')
    data = multiple_selfies_to_hot(encoding_list, largest_molecule_len,
                                   encoding_alphabet)
    print('Finished creating one-hot encoding.')
    return data

def save_alphabet(selfies_alphabet, path='selfie_alphabet.txt'):
    with open(path, 'w') as f:
        for item in selfies_alphabet:
            f.write("%s\n" % item)

def load_alphabet(path='selfie_alphabet.txt'):
    with open(path, 'r') as f:
        selfies_alphabet = f.read().splitlines()
        return selfies_alphabet

def get_selfie_encodings_for_dataset(smiles_list, selfies_alphabet, largest_selfies_len):

    smiles_alphabet = list(set(''.join(smiles_list)))
    smiles_alphabet.append(' ')  # for padding

    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(sf.encoder, smiles_list))

    if not selfies_alphabet:
        all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
        all_selfies_symbols.add('[nop]')

        selfies_alphabet = list(all_selfies_symbols)
        save_alphabet(selfies_alphabet)

        largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)
        print(largest_selfies_len)

    print('Finished translating SMILES to SELFIES.')

    return selfies_list, selfies_alphabet, largest_selfies_len

def selfies_to_hot(selfie, largest_selfie_len, alphabet):
    """Go from a single selfies string to a one-hot encoding.
    """

    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with [nop]
    selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))

    # integer encode
    symbol_list = sf.split_selfies(selfie)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]

    # one hot-encode the integer encoded selfie
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(alphabet)
        letter[index] = 1
        onehot_encoded.append(letter)

    return integer_encoded, np.array(onehot_encoded)

def hot_to_smile(onehot_encoded, alphabet):
    """
    Go from one-hot encoding to smile string
    """
    molecules = []
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))

    # From one-hot to integer encoding
    for oh in onehot_encoded:

        integer_encoded = oh.argmax(1)

        # integer encoding to smile
        regen_smile = "".join(int_to_char[x.item()] for x in integer_encoded)
        regen_smile = regen_smile.strip()
        molecules.append(regen_smile)

    return molecules


def multiple_selfies_to_hot(selfies_list, largest_molecule_len, alphabet):
    """Convert a list of selfies strings to a one-hot encoding
    """

    hot_list = []
    for s in selfies_list:
        _, onehot_encoded = selfies_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)
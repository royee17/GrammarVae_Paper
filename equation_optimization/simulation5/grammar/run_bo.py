
import pickle
import gzip
import sys
sys.path.insert(0, '../../../')
from Network import GrammarVariationalAutoEncoder
from grammar_models import EquationGrammarModel
import torch
from sparse_gp import SparseGP
import scipy.stats as sps
import numpy as np
import collections
import argparse
import os

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Run Simulations')
    parser.add_argument('--model_name', type=str, metavar='N', default="eq2_grammar_dataset_L56_E50_val.hdf5")
    return parser.parse_args()


def decode_from_latent_space(latent_points, grammar_model):

    decode_attempts = 25
    decoded_molecules = []
    for i in range(decode_attempts):
        current_decoded_molecules = grammar_model.decode(latent_points)
        decoded_molecules.append(current_decoded_molecules)

    # We see which ones are decoded by rdkit
    
    rdkit_molecules = []
    for i in range(decode_attempts):
        rdkit_molecules.append([])
        for j in range(latent_points.shape[ 0 ]):
            smile = np.array([ decoded_molecules[ i ][ j ] ]).astype('str')[ 0 ]
            if smile == '':
                rdkit_molecules[ i ].append(None)
            else:
                rdkit_molecules[ i ].append(smile)

    decoded_molecules = np.array(decoded_molecules)
    rdkit_molecules = np.array(rdkit_molecules)

    final_smiles = []
    for i in range(latent_points.shape[ 0 ]):

        aux = collections.Counter(rdkit_molecules[ ~np.equal(rdkit_molecules[ :, i ], None) , i ])
        aux_lst = [(k, aux[k]) for k in aux]
        if len(aux_lst) > 0:
            smile = aux_lst[ np.argmax(aux.values()) ][ 0 ]
        else:
            smile = None
        final_smiles.append(smile)

    return final_smiles

# We define the functions used to load and save objects

def save_object(obj, filename):

    """
    Function that saves an object to a file using pickle
    """

    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()


def load_object(filename):

    """
    Function that loads an object from a file using pickle
    """

    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()

    return ret


if __name__ == '__main__':

    directory = "results/"

    args = get_arguments()

    random_seed = 1
    np.random.seed(random_seed)
    # We load the data
    X = np.loadtxt('../../latent_features_and_targets_grammar/latent_features_eq.txt')
    y = np.loadtxt('../../latent_features_and_targets_grammar/targets_eq.txt')
    y = y.reshape((-1, 1))

    n = X.shape[ 0 ]
    permutation = np.random.choice(n, n, replace = False)

    X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
    X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]

    y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
    y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]

    model_name = args.model_name
    model_path = os.path.join('..', '..', '..', 'final_models', args.model_name)

    for iteration in range(2):

        # We fit the GP
        np.random.seed(random_seed * iteration)
        M = 500
        sgp = SparseGP(X_train, 0 * X_train, y_train, M)
        sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,  \
            y_test, minibatch_size = 10 * M, max_iterations = 50, learning_rate = 0.0005)

        pred, uncert = sgp.predict(X_test, 0 * X_test)
        error = np.sqrt(np.mean((pred - y_test)**2))
        testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
        print(f'Test RMSE: {error}')
        print(f'Test ll: {testll}')

        pred, uncert = sgp.predict(X_train, 0 * X_train)
        error = np.sqrt(np.mean((pred - y_train)**2))
        trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
        print(f'Train RMSE: {error}')
        print(f'Train ll: {trainll}')

        final_model = GrammarVariationalAutoEncoder()

        final_model.load_state_dict(torch.load(model_path))

        grammar_model = EquationGrammarModel(final_model)

        # We pick the next 50 inputs
        next_inputs = sgp.batched_greedy_ei(50, np.min(X_train, 0), np.max(X_train, 0))

        valid_eq_final = decode_from_latent_space(next_inputs, grammar_model)

        new_features = next_inputs

        save_object(valid_eq_final, directory + "valid_eq_{}.dat".format(iteration))

        x = np.loadtxt('../../latent_features_and_targets_grammar/x_eq.txt')
        yT = np.loadtxt('../../latent_features_and_targets_grammar/true_y_eq.txt')

        scores = []
        WORST = 1000
        for i in range(len(valid_eq_final)):
            if valid_eq_final[ i ] is not None:
                try:
                    score = np.log(1+np.mean(np.minimum((np.array(eval(valid_eq_final[i])) - yT)**2, WORST)))
                except:
                    score = np.log(1+WORST)
                if not np.isfinite(score):
                    score = np.log(1+WORST)
            else:
                score = np.log(1+WORST)

            scores.append(score)
            print(i)

        print(valid_eq_final)
        print(scores)

        save_object(scores, directory + "scores_eq_{}.dat".format(iteration))

        if len(new_features) > 0:
            X_train = np.concatenate([ X_train, new_features ], 0)
            y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)

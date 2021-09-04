import sys
sys.path.insert(0, '../../../')
import pickle
import gzip
from sparse_gp import SparseGP
import scipy.stats as sps
import numpy as np
import argparse
import collections
import os
from Network import GrammarVariationalAutoEncoder
from grammar_models import ZincGrammarModel, InnovativeGrammarModel
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
import sascorer
import networkx as nx
from rdkit.Chem import rdmolops
import torch
from random import randint
from grammar import mol_grammar as G

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Run Simulations')
    parser.add_argument('--model_name', type=str, metavar='N', default="eq2_grammar_dataset_L56_E50_val.hdf5")
    return parser.parse_args()


def decode_from_latent_space(latent_points, grammar_model):

    decode_attempts = 500
    decoded_molecules = []
    for i in range(decode_attempts):
        current_decoded_molecules = grammar_model.decode(latent_points)
        current_decoded_molecules = [ x if x != '' else 'Sequence too long' for x in current_decoded_molecules ]
        decoded_molecules.append(current_decoded_molecules)

    # We see which ones are decoded by rdkit
    
    rdkit_molecules = []
    for i in range(decode_attempts):
        rdkit_molecules.append([])
        for j in range(latent_points.shape[ 0 ]):
            smile = np.array([ decoded_molecules[ i ][ j ] ]).astype('str')[ 0 ]
            if MolFromSmiles(smile) is None:
                rdkit_molecules[ i ].append(None)
            else:
                rdkit_molecules[ i ].append(smile)

    decoded_molecules = np.array(decoded_molecules)
    rdkit_molecules = np.array(rdkit_molecules)

    final_smiles = []
    for i in range(latent_points.shape[0]):

        aux = collections.Counter(rdkit_molecules[~np.equal(rdkit_molecules[:, i], None), i])
        aux_lst = [(k, aux[k]) for k in aux]
        if len(aux_lst) > 0:
            smile = aux_lst[np.argmax(aux.values())][0]
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

    random_seed = randint(0, 100)
    np.random.seed(random_seed)

    # We load the data
    X = np.loadtxt('../../latent_features_and_targets_grammar/latent_features.txt')
    y = -np.loadtxt('../../latent_features_and_targets_grammar/targets.txt')
    y = y.reshape((-1, 1))

    n = X.shape[ 0 ]
    permutation = np.random.choice(n, n, replace = False)

    X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
    X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]

    y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
    y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]

    model_name = args.model_name
    model_path = os.path.join('..', '..', '..', 'final_models', model_name)
    rules = G.gram.split('\n')
    TestLL = []
    RMSE = []
    for iteration in range(10):

        # We fit the GP

        np.random.seed(iteration * random_seed)
        M = 500
        sgp = SparseGP(X_train, 0 * X_train, y_train, M)
        sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,  \
            y_test, minibatch_size = 10 * M, max_iterations = 50, learning_rate = 0.0005)

        pred, uncert = sgp.predict(X_test, 0 * X_test)
        error = np.sqrt(np.mean((pred - y_test)**2))
        testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
        TestLL.append(testll)
        RMSE.append(error)
        print(f'Test RMSE: {error}')
        print(f'Test ll: {testll}')

        pred, uncert = sgp.predict(X_train, 0 * X_train)
        error = np.sqrt(np.mean((pred - y_train)**2))
        trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
        print(f'Train RMSE: {error}')
        print(f'Train ll: {trainll}')

        final_model = GrammarVariationalAutoEncoder(model_name='mol_grammar_selfie', rules=rules)
        final_model.load_state_dict(torch.load(model_path))
        grammar_model = InnovativeGrammarModel(final_model)

        # We pick the next 50 inputs
        next_inputs = sgp.batched_greedy_ei(50, np.min(X_train, 0), np.max(X_train, 0))

        valid_smiles_final = decode_from_latent_space(next_inputs, grammar_model)

        new_features = next_inputs

        save_object(valid_smiles_final, directory + "valid_smiles{}.dat".format(iteration))

        logP_values = np.loadtxt('../../latent_features_and_targets_grammar/logP_values.txt')
        SA_scores = np.loadtxt('../../latent_features_and_targets_grammar/SA_scores.txt')
        cycle_scores = np.loadtxt('../../latent_features_and_targets_grammar/cycle_scores.txt')
        SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
        logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
        cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

        targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized

        scores = []
        for i in range(len(valid_smiles_final)):
            if valid_smiles_final[ i ] is not None:
                current_log_P_value = Descriptors.MolLogP(MolFromSmiles(valid_smiles_final[ i ]))
                current_SA_score = -sascorer.calculateScore(MolFromSmiles(valid_smiles_final[ i ]))
                cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(valid_smiles_final[ i ]))))
                if len(cycle_list) == 0:
                    cycle_length = 0
                else:
                    cycle_length = max([ len(j) for j in cycle_list ])
                if cycle_length <= 6:
                    cycle_length = 0
                else:
                    cycle_length = cycle_length - 6

                current_cycle_score = -cycle_length

                current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
                current_log_P_value_normalized = (current_log_P_value - np.mean(logP_values)) / np.std(logP_values)
                current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)

                score = (current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized)
            else:
                score = -max(y)[ 0 ]

            scores.append(-score)
            print(i)

        print(valid_smiles_final)
        print(scores)

        save_object(scores, directory + "scores{}.dat".format(iteration))

        if len(new_features) > 0:
            X_train = np.concatenate([ X_train, new_features ], 0)
            y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)

    with open('../../TestLL.txt', 'a') as f:
        for item in TestLL:
            f.write("%s\n" % item)
    with open('../../RMSE.txt', 'a') as f:
        for item in RMSE:
            f.write("%s\n" % item)
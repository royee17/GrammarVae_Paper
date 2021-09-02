import sys
sys.path.insert(0, '../../')
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
import sascorer
import os
import numpy as np  
from rdkit.Chem import MolFromSmiles, MolToSmiles
import networkx as nx
from Network import GrammarVariationalAutoEncoder
from grammar_models import ZincGrammarModel
import torch
import pandas as pd
import argparse
from grammar import mol_grammar as G

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Molecular autoencoder network')
    parser.add_argument('--model_name', type=str, metavar='N', default="zinc_grammar_dataset_L56_E5_val.hdf5")
    parser.add_argument('--model_type', type=str, metavar='N', default="zinc")
    return parser.parse_args()

if __name__ == '__main__':

    args = get_arguments()

    model_type = 'qm9'#args.model_type

    model_name = 'qm9_mol_grammar_dataset_L56_E7_val.hdf5'#args.model_name
    model_path = os.path.join('..', '..', 'final_models', args.model_name)
    rules = G.gram.split('\n')

    final_model = GrammarVariationalAutoEncoder(model_name='mol_grammar', rules=rules)
    final_model.load_state_dict(torch.load(model_path))
    grammar_model = ZincGrammarModel(final_model)

    if model_type == 'zinc':

        fname = '../../data/250k_rndm_zinc_drugs_clean.smi'

        with open(fname) as f:
            smiles = f.readlines()

        smiles = smiles[100000:110000]

    else:
        df = pd.read_csv('../../data/qm9.csv')
        smiles = df['smiles'].values.tolist()
        smiles = smiles[90000:112000]

    for i in range(len(smiles)):
        smiles[ i ] = smiles[ i ].strip()

    smiles_rdkit = []
    for i in range(len(smiles)):
        smiles_rdkit.append(MolToSmiles(MolFromSmiles(smiles[ i ])))

    logP_values = []
    for i in range(len(smiles)):
        logP_values.append(Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[ i ])))

    SA_scores = []
    for i in range(len(smiles)):
        SA_scores.append(-sascorer.calculateScore(MolFromSmiles(smiles_rdkit[ i ])))


    cycle_scores = []
    print('starting cycle scores')
    for i in range(len(smiles)):
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles_rdkit[ i ]))))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([ len(j) for j in cycle_list ])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        cycle_scores.append(-cycle_length)

    print('computing scores')
    SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
    logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
    cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

    latent_points = grammar_model.encode(smiles_rdkit)

    # We store the results

    latent_points = latent_points.detach().numpy()
    np.savetxt('latent_features.txt', latent_points)

    targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized
    np.savetxt('targets.txt', targets)
    np.savetxt('logP_values.txt', np.array(logP_values))
    np.savetxt('SA_scores.txt', np.array(SA_scores))
    np.savetxt('cycle_scores.txt', np.array(cycle_scores))
    print('Done!')

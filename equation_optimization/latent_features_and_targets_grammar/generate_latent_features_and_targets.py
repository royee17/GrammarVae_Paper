from __future__ import division

import sys
sys.path.insert(0, '../../')

import numpy as np
import pandas as pd
import math
import torch
from grammar_models import EquationGrammarModel
from Network import GrammarVariationalAutoEncoder
import argparse
import os

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Molecular autoencoder network')
    parser.add_argument('--model_name', type=str, metavar='N', default="eq2_grammar_dataset_L56_E50_val.hdf5")
    return parser.parse_args()

def func(x):
    return 1/3 + x + math.sin(x * x)

if __name__ == '__main__':

    # We load the data
    fname = '../../data/equation2_15_dataset.txt'

    args = get_arguments()
    model_path = os.path.join('..','..','final_models', args.model_name)

    x = np.linspace(-10,10,1000)
    y = np.array([func(x_) for x_ in x])

    with open(fname) as f:
        eqs = f.readlines()

    for i in range(len(eqs)):
        eqs[i] = eqs[i].strip()
        eqs[i] = eqs[i].replace(' ', '')

    final_model = GrammarVariationalAutoEncoder()
    final_model.load_state_dict(torch.load(model_path))

    grammar_model = EquationGrammarModel(final_model)
    WORST = 1000
    MAX = 100000

    targets = []
    scores = []
    equations = []
    for i in range(len(eqs[:MAX])):
        try:
            score = np.log(1+np.mean(np.minimum((np.array(eval(eqs[i])) - y)**2, WORST)))
        except:
            score = np.log(1+WORST)
        if not np.isfinite(score):
            score = np.log(1+WORST)
        print(i, eqs[i], score)
        scores.append(score)
        equations.append(eqs[i])
        targets.append(score)

    data_tuples = list(zip(equations, scores))
    df = pd.DataFrame(data_tuples, columns=['equations', 'scores'])
    df = df.sort_values(by=['scores'])

    targets = np.array(targets)

    np.savetxt('targets_eq.txt', targets)
    np.savetxt('x_eq.txt', x)
    np.savetxt('true_y_eq.txt', y)

    batch_size = 5000
    num_batches = int(np.ceil(len(targets) / batch_size))
    for i in range(num_batches):
        if i == 0:
            latent_points = grammar_model.encode(eqs[:batch_size])
            latent_points = latent_points.cpu().detach().numpy()
        else:
            new_latents = grammar_model.encode(eqs[i*batch_size:(i+1)*batch_size])
            latent_points = np.concatenate((latent_points, new_latents.cpu().detach().numpy()), axis=0)
        print(i)

    # We store the results
    np.savetxt('latent_features_eq.txt', latent_points)
    print('Done!')

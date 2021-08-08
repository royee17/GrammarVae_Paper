from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
import torch
from Network import GrammarVariationalAutoEncoder
from zinc_grammar_model import EquationGrammarModel
from math import *
import argparse
import os

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Molecular autoencoder network')
    parser.add_argument('--model_name', type=str, metavar='N', default="eq2_grammar_dataset_L56_E50_val.hdf5")
    return parser.parse_args()


if __name__ == '__main__':

    args = get_arguments()

    model_path = os.path.join('final_models', args.model_name)

    final_model = GrammarVariationalAutoEncoder()
    final_model.load_state_dict(torch.load(model_path))

    grammar_model = EquationGrammarModel(final_model)


    # 2. let's encode and decode some example equations
    eq = ['sin(x*2)',
          'x + 2 * sin(x)',
          'exp(x)+x',
          'x/3',
          '3*exp(2/x)']

    # z: encoded latent points
    # NOTE: this operation returns the mean of the encoding distribution
    z = grammar_model.encode(eq)

    # mol: decoded equations
    # NOTE: decoding is stochastic so calling this function many
    # times for the same latent point will return different answers
    # let's plot how well the true functions match the decoded functions
    domain = np.linspace(-10,10)
    for i, s in enumerate(grammar_model.decode(z)):
        if s == '':
            print(f'equation {eq[i]} was decoded to an empty string from latent space!')
            continue

        plt.figure()
        f = eval("lambda x: "+eq[i], globals())
        f_hat = eval("lambda x: "+s)
        plt.plot(domain, np.array(list(map(f, domain))))
        fhat_res = np.array(list(map(f_hat, domain)))
        ty = type(fhat_res)
        if (ty is not np.ndarray):
            plt.plot(domain, np.repeat(fhat_res, len(domain)))
        else:
            plt.plot(domain, fhat_res)
        plt.legend(["function", "reconstruction"])
        plt.title('%15s -> %s' % (eq[i], s))
        plt.show()

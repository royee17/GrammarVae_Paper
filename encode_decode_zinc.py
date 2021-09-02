import torch
from Network import GrammarVariationalAutoEncoder
from grammar_models import ZincGrammarModel
from grammar import mol_grammar as G
import argparse
import os

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Molecular autoencoder network')
    parser.add_argument('--model_name', type=str, metavar='N', default="")
    return parser.parse_args()


if __name__ == '__main__':

    args = get_arguments()

    model_path = os.path.join('final_models', args.model_name)
    rules = G.gram.split('\n')

    final_model = GrammarVariationalAutoEncoder(model_name='mol_grammar', rules=rules)
    final_model.load_state_dict(torch.load(model_path))
    grammar_model = ZincGrammarModel(final_model)

    smiles = ["C[C@@H]1CN(C(=O)c2cc(Br)cn2C)CC[C@H]1[NH3+]",
              "CC[NH+](CC)[C@](C)(CC)[C@H](O)c1cscc1Br",
              "O=C(Nc1nc[nH]n1)c1cccnc1Nc1cccc(F)c1",
              "Cc1c(/C=N/c2cc(Br)ccn2)c(O)n2c(nc3ccccc32)c1C#N",
              "CSc1nncn1/N=C\c1cc(Cl)ccc1F"]

    z1 = grammar_model.encode(smiles)

    for mol,real in zip(grammar_model.decode(z1),smiles):
        print(f'{mol} {real}')




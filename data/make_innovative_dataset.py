import h5py
from utils import smiles_to_one_hot, selfies_to_one_hot, fetch_data_set


def make_molecule_ds(data_set='qm9', use_selfie_representations=False):

    data = fetch_data_set(data_set)
    repr_type = 'smiles'

    if use_selfie_representations:
        OH = selfies_to_one_hot(data)
        repr_type = 'selfie'
    else:
        OH = smiles_to_one_hot(data)

    h5f = h5py.File(f'{data_set}_{repr_type}_grammar_dataset.h5', 'w')
    h5f.create_dataset('data', data=OH)
    h5f.close()


if __name__ == '__main__':

    make_molecule_ds(data_set='zinc', use_selfie_representations=True)

import pickle
import numpy as np
import gzip
from rdkit.Chem import MolFromSmiles

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

# We compute the average statistics for the grammar autoencoder

if __name__ == '__main__':
    n_simulations = 10
    iteration = 10
    character_results = False

    results_grammar = np.zeros((n_simulations, 3))

    for j in range(1, n_simulations + 1):
        best_value = 1e10
        n_valid = 0
        max_value = 0
        for i in range(iteration):
            smiles = load_object('simulation{}/grammar/results/valid_smiles{}.dat'.format(j, i))
            scores = load_object('simulation{}/grammar/results/scores{}.dat'.format(j, i))
            n_valid += len([ x for x in smiles if x is not None ])

            if min(scores) < best_value:
                best_value = min(scores)
            if max(scores) > max_value:
                max_value = max(scores)

        sum_values = 0
        count_values = 0
        for i in range(iteration):
            scores = np.array(load_object('simulation{}/grammar/results/scores{}.dat'.format(j, i)))
            sum_values += np.sum(scores[scores < max_value])
            count_values += len(scores[scores < max_value])

        # fraction of valid smiles
        results_grammar[ j - 1, 0 ] = 1.0 * n_valid / (iteration * 50)

        # Best value
        results_grammar[ j - 1, 1 ] = best_value

        # Average value =
        results_grammar[ j - 1, 2 ] = 1.0 * sum_values / count_values

    print("Results Grammar VAE (fraction valid, best, average)):")

    print("Mean:", np.mean(results_grammar, 0)[ 0 ], -np.mean(results_grammar, 0)[ 1 ], -np.mean(results_grammar, 0)[ 2 ])
    print("Std:", np.std(results_grammar, 0) / np.sqrt(iteration))
    print("First:", -np.min(results_grammar[ : , 1 ]))

    best_score = np.min(results_grammar[ : , 1 ])
    results_grammar[ results_grammar[ : , 1 ] == best_score , 1 ] = 1e10
    print("Second:", -np.min(results_grammar[ : , 1 ]))
    second_best_score = np.min(results_grammar[ : , 1 ])
    results_grammar[ results_grammar[ : , 1 ] == second_best_score, 1 ] = 1e10
    print("Third:", -np.min(results_grammar[ : , 1 ]))
    third_best_score = np.min(results_grammar[ : , 1 ])

    mols = []
    for j in range(1, n_simulations + 1):
        for i in range(iteration):
            smiles = np.array(load_object('simulation{}/grammar/results/valid_smiles{}.dat'.format(j, i)))
            scores = np.array(load_object('simulation{}/grammar/results/scores{}.dat'.format(j, i)))

            if np.any(scores == best_score):
                smile = smiles[scores == best_score]
                smile = np.array(smile).astype('str')[0]
                print("First:", smile)
                mol = MolFromSmiles(smile)
                mols.append(mol)
                best_score = 1e10

            if np.any(scores == second_best_score):
                smile = smiles[scores == second_best_score]
                smile = np.array(smile).astype('str')[0]
                print("Second:", smile)
                mol = MolFromSmiles(smile)
                mols.append(mol)
                second_best_score = 1e10

            if np.any(scores == third_best_score):
                smile = smiles[scores == third_best_score]
                smile = np.array(smile).astype('str')[0]
                print("Third:", smile)
                mol = MolFromSmiles(smile)
                mols.append(mol)
                third_best_score = 1e10

    if character_results:
        results_character = np.zeros((n_simulations, 3))
        for j in range(1, n_simulations + 1):
            best_value = 1e10
            n_valid = 0
            max_value = 0
            for i in range(iteration):
                smiles = load_object('simulation{}/character/results/valid_smiles{}.dat'.format(j, i))
                scores = load_object('simulation{}/character/results/scores{}.dat'.format(j, i))
                n_valid += len([ x for x in smiles if x is not None ])

                if min(scores) < best_value:
                    best_value = min(scores)
                if max(scores) > max_value:
                    max_value = max(scores)

            sum_values = 0
            count_values = 0
            for i in range(iteration):
                scores = np.array(load_object('simulation{}/character/results/scores{}.dat'.format(j, i)))
                sum_values += np.sum(scores[  scores < max_value ])
                count_values += len(scores[  scores < max_value ])

            # fraction of valid smiles
            results_character[ j - 1, 0 ] = 1.0 * n_valid / (iteration * 50)

            # Best value
            results_character[ j - 1, 1 ] = best_value

            # Average value =
            results_character[ j - 1, 2 ] = 1.0 * sum_values / count_values

        print("Results Character VAE (fraction valid, best, average)):")

        print("Mean:", np.mean(results_character, 0)[ 0 ], -np.mean(results_character, 0)[ 1 ], -np.mean(results_character, 0)[ 2 ])
        print("Std:", np.std(results_character, 0) / np.sqrt(iteration))

        print("First:", -np.min(results_character[ : , 1 ]))
        best_score = np.min(results_character[ : , 1 ])
        results_character[ results_character[ : , 1 ] == best_score , 1 ] = 1e10
        print("Second:", -np.min(results_character[ : , 1 ]))
        second_best_score = np.min(results_character[ : , 1 ])
        results_character[ results_character[ : , 1 ] == second_best_score, 1 ] = 1e10
        print("Third:", -np.min(results_character[ : , 1 ]))
        third_best_score = np.min(results_character[ : , 1 ])

        # We print the best smile found the character autoencoder

        mols = []
        for j in range(1, n_simulations + 1):
            for i in range(iteration):
                smiles = np.array(load_object('simulation{}/character/results/valid_smiles{}.dat'.format(j, i)))
                scores = np.array(load_object('simulation{}/character/results/scores{}.dat'.format(j, i)))

                if np.any(scores == best_score):
                    smile = smiles[ scores == best_score ]
                    smile = np.array(smile).astype('str')[ 0 ]
                    print("First:", smile)
                    mol = MolFromSmiles(smile)
                    mols.append(mol)
                    best_score = 1e10

                if np.any(scores == second_best_score):
                    smile = smiles[ scores == second_best_score ]
                    smile = np.array(smile).astype('str')[ 0 ]
                    print("Second:", smile)
                    mol = MolFromSmiles(smile)
                    mols.append(mol)
                    second_best_score = 1e10

                if np.any(scores == third_best_score):
                    smile = smiles[ scores == third_best_score ]
                    smile = np.array(smile).astype('str')[ 0 ]
                    print("Third:", smile)
                    mol = MolFromSmiles(smile)
                    mols.append(mol)
                    third_best_score = 1e10


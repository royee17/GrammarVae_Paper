import argparse
import os

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Run Simulations')
    parser.add_argument('--num_of_simulations', type=int, metavar='N', default=10)
    return parser.parse_args()


if __name__ == '__main__':

    args = get_arguments()
    simulations = 4#args.num_of_simulations
    curr_dir = os.getcwd()
    for num in range(1, simulations + 1):
        print(f'############ running simulation number {num} ######################')
        os.chdir(curr_dir)
        simulation_path = f'simulation{num}/grammar/'
        os.chdir(simulation_path)
        # os.system("python run_bo.py --model_name=zinc_grammar_dataset_L56_E5_val.hdf5")
        os.system("python run_bo.py --model_name=qm9_mol_grammar_dataset_L56_E7_val.hdf5")

    print('Done')

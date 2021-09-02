import argparse
import os

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Run Simulations')
    parser.add_argument('--num_of_simulations', type=int, metavar='N', default=10)
    return parser.parse_args()


if __name__ == '__main__':

    args = get_arguments()
    simulations = args.num_of_simulations
    curr_dir = os.getcwd()
    for num in range(1, simulations + 1):
        print(f'############ running simulation number {num} ######################')
        os.chdir(curr_dir)
        simulation_path = f'simulation{num}/grammar/'
        os.chdir(simulation_path)
        os.system("python run_bo.py")

    print('Done')

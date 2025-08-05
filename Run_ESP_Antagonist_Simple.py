import time
import argparse
import numpy as np
from Epigenetic_Sequence_Predictor.ESP_Sim_Antagonist_SimpleNN import run_sim


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_name', type=str, default='Extended_Split', help='Name of the batch')
parser.add_argument('--alpha_eval', type=str, default="[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]",
                    help='string to evaluate as the alpha list')
parser.add_argument('--beta_eval', type=str, default="[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]",
                    help='string to evaluate as the beta list')
parser.add_argument('--mu', type=float, default=0.5, help='mu value for the simulation')
parser.add_argument('--rho', type=float, default=0.5, help='rho value for the simulation')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--n_samples', type=int, default=100000, help='Number of samples')
parser.add_argument('--n_samples_test', type=int, default=30000, help='Number of test samples')
parser.add_argument('--seq_length', type=int, default=100, help='Sequence length')
parser.add_argument('--input_size', type=int, default=200, help='Input size')
parser.add_argument('--hidden_size', type=int, default=int(1.5 * 100), help='Hidden size')
parser.add_argument('--output_size', type=int, default=100, help='Output size')
parser.add_argument('--num_layers', type=int, default=5, help='Number of layers')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=10000, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=100000, help='Batch size')
parser.add_argument('--loss_function_type', type=str, default='MSE', help='Loss function type')
parser.add_argument('--verbose_level', type=int, default=1, help='Verbose level')
parser.add_argument('--visualise_nn', type=bool, default=False, help='Visualise the neural network')

args = parser.parse_args()

full_start = time.time()

batch_name = args.batch_name
alpha_eval = args.alpha_eval
beta_eval = args.beta_eval
mu = args.mu
rho = args.rho
seed = args.seed
n_samples = args.n_samples
n_samples_test = args.n_samples_test
seq_length = args.seq_length
input_size = args.input_size
hidden_size = args.hidden_size
output_size = args.output_size
num_layers = args.num_layers
learning_rate = args.learning_rate
num_epochs = args.num_epochs
batch_size = args.batch_size
loss_function_type = args.loss_function_type
verbose_level = args.verbose_level
visualise_nn = args.visualise_nn
sim_id = batch_name

a_list = eval(alpha_eval)
b_list = eval(beta_eval)

print(f"Alpha List = {a_list}")
print(f"Beta List = {b_list}")

with open(f'{batch_name}_BitError_Combinations.csv', 'w') as f:
    f.write('Alpha,Beta,BitError\n')

for a in a_list:
    for b in b_list:
        sim_start = time.time()
        print(f"Running simulation with alpha = {round(a, 1)} and beta = {round(b, 1)}")
        sim_name, biterror = run_sim(
            sim_id=sim_id, seed=seed, alpha=round(a, 1), beta=round(b, 1), mu=mu, rho=rho,
            n_samples=n_samples, n_samples_test=n_samples_test, seq_length=seq_length,
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=num_layers, learning_rate=learning_rate, num_epochs=num_epochs,
            batch_size=batch_size, loss_function_type=loss_function_type,
            verbose_level=verbose_level, visualise_nn=visualise_nn
        )
        sim_end = time.time()
        print(f"Simulation Name = {sim_name}", end=' --- ')
        print(f"Sim Execution Time = {round(sim_end - sim_start, 4)}", end=' --- ')
        print(f"BitError = {biterror}")

        with open(f'{batch_name}_BitError_Combinations.csv', 'a') as f:
            f.write(f'{round(a, 1)},{round(b, 1)},{biterror}\n')

full_end = time.time()
print(f"Total Execution time = {full_end - full_start}")

with open(f'{batch_name}_Batch_Log.txt', 'w') as f:
    f.write(f'Log File - Batch Simulation :: {batch_name}\n')
    f.write(f"Total Execution time = {full_end - full_start}\n\n")

    f.write(f'##INPUTS##\n')
    f.write(f'sim_id = {sim_id}\n')
    f.write(f'seed = {seed}\n')
    f.write(f'alpha_eval = {alpha_eval}\n')
    f.write(f'beta_eval = {beta_eval}\n')
    f.write(f'mu = {mu}\n')
    f.write(f'rho = {rho}\n')
    f.write(f'n_samples = {n_samples}\n')
    f.write(f'n_samples_test = {n_samples_test}\n')
    f.write(f'seq_length = {seq_length}\n')
    f.write(f'input_size = {input_size}\n')
    f.write(f'hidden_size = {hidden_size}\n')
    f.write(f'output_size = {output_size}\n')
    f.write(f'num_layers = {num_layers}\n')
    f.write(f'learning_rate = {learning_rate}\n')
    f.write(f'num_epochs = {num_epochs}\n')
    f.write(f'batch_size = {batch_size}\n')
    f.write(f'loss_function_type = {loss_function_type}\n')
    f.write(f'verbose_level = {verbose_level}\n')
    f.write(f'visualise_nn = {visualise_nn}\n')

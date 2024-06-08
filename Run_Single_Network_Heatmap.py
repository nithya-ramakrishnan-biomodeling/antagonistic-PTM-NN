"""
This program uses the weights of a pretrained neural network to predict the daughter sequences of each alpha beta pair
for different values of mu, calculate the BER of each sequence and then give a Bit Error Matrix.
This can be used for both the Extended and Simple NN as both use the same Dataframe and split_sequence functions.
"""
import time
import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from Epigenetic_Sequence_Predictor.ESP_Sim_Antagonist_ExtendedNN import Dataframe, split_sequence

# Here we are importing the dataframe and the functions from ESP_Sim_Antagonist_ExtendedNN instead of range as we want
# to use the range network to create the heatmap and the dataframe and functions used for the range network are designed
# differently for a range of sequences.


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_name', type=str, default='Single_NN_Test', help='Name of the batch')
parser.add_argument('--weight_path', type=str, default='weights.pth',
                    help='Path to the NN Weights for the Extended Range Network')
parser.add_argument('--alpha_eval', type=str, default="[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]",
                    help='String to evaluate as the alpha list')
parser.add_argument('--beta_eval', type=str, default="[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]",
                    help='string to evaluate as the beta list')
parser.add_argument('--mu_eval', type=str, default="[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]",
                    help='String to evaluate the mu list')
parser.add_argument('--rho', type=float, default=0.5, help='rho value for the simulation')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
parser.add_argument('--verbose_level', type=int, default=5, help='Verbose level')

args = parser.parse_args()

full_start = time.time()

batch_name = args.batch_name
network_weight_path = args.weight_path
alpha_eval = args.alpha_eval
beta_eval = args.beta_eval
mu_eval = args.mu_eval
rho = args.rho
seed = args.seed
n_samples = args.n_samples
verbose_level = args.verbose_level


class SequencePredictor(nn.Module):
    # Define the neural network architecture
    # (input -> h1 -> h2 .... hn -> output)
    def __init__(self):
        super(SequencePredictor, self).__init__()
        self.input_size = 200
        self.output_size = 100
        self.hidden_size = 150
        self.num_hid_layers = 5

        # This is the first layer which takes in inputs and gives out the outputs for the first hidden layer
        self.input_layer = nn.Linear(self.input_size, self.hidden_size)

        # These are the hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_hid_layers)])

        # This is the output layer
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.relu(out)
        for layer in self.hidden_layers:
            out = layer(out)
            out = self.relu(out)

        out = self.output_layer(out)
        return out


########################################################################################################################

# Set the random seed for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)

# Giving Introduction
if verbose_level > 0:
    print("Epigenetic Sequence Predictor")
    print(f"Sim Name :: {batch_name}")
    print(f'This Program Evaluates the Bit Error for each combination of alpha, beta and mu.')
    print(f'This is done using the weights of an extended neural network.')
    print()

# Create the log Files
if not os.path.exists(batch_name):
    os.makedirs(batch_name)

a_list = eval(alpha_eval)
b_list = eval(beta_eval)
mu_list = eval(mu_eval)

model = SequencePredictor()
model.load_state_dict(torch.load(network_weight_path))  # Load the network weights

# test = np.array(
#     [0, 2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 1, 0, 1, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 2, 1, 2, 0, 0, 0, 1, 1,
#      0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 2, 0, 2, 2, 1, 2, 0, 2, 0, 0, 0, 1, 0, 1, 0, 0,
#      0, 1, 0, 0, 1, 0, 1, 2, 0, 0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 0, 1, 0, 0, 1])
#
# a, b = split_sequence(test)
# d = np.append(a, b)
# d = torch.from_numpy(d).float()
# # Run the predictions.
# model.eval()  # To turn off learning mode.
# with torch.no_grad():
#     output = model(d)
# print(output)
for mu in mu_list:
    biterror_combination_path = f'{batch_name}/mu_{str(mu).replace(".", "_")}/' \
                                f'{batch_name}_mu_{str(mu).replace(".", "_")}_BitError_Combinations.csv'
    # Create the mu folder path
    if not os.path.exists(f'{batch_name}/mu_{str(mu).replace(".", "_")}'):
        os.makedirs(f'{batch_name}/mu_{str(mu).replace(".", "_")}')
    with open(biterror_combination_path, 'a') as f:
        f.write('Alpha,Beta,BitError\n')
    for alpha in a_list:
        for beta in b_list:
            sim_name = f'{batch_name}/mu_{str(mu).replace(".", "_")}' \
                       f'/{batch_name}_A{str(alpha).replace(".", "_")}_B{str(beta).replace(".", "_")}'
            if verbose_level > 0:
                print(f"Running Simulation for {sim_name}")

            # Prepare the data
            data = Dataframe(alpha, beta, mu, rho, n_samples, 100)

            mother_sequences = data.mom_list
            corrupted_daughter_sequences = data.corrupt_daughter_list

            mom_A = np.copy(mother_sequences)
            mom_B = np.copy(mother_sequences)
            daughter_A = np.copy(mother_sequences)
            daughter_B = np.copy(mother_sequences)
            daughter_input = np.zeros([mother_sequences.shape[0], 2 * mother_sequences.shape[1]])

            for i in range(mother_sequences.shape[0]):
                mom_A[i], mom_B[i] = split_sequence(mother_sequences[i])
                daughter_A[i], daughter_B[i] = split_sequence(corrupted_daughter_sequences[i])
                daughter_input[i] = np.append(daughter_A[i], daughter_B[i])

            # for d in daughter_input[50]:
            #     print(d, end=',')

            # Convert the numpy arrays to PyTorch tensors
            mom_A = torch.from_numpy(mom_A).float()
            daughter_A = torch.from_numpy(daughter_A).float()

            # Convert the numpy arrays to PyTorch tensors
            daughter_input = torch.from_numpy(daughter_input).float()

            # Run the predictions.
            model.eval()  # To turn off learning mode.
            print(model)
            with torch.no_grad():
                output = model(daughter_input)
            print("Output before rounding:", output)

            predicted_sequences = output.round().numpy().astype(int)
            for i, seq in enumerate(predicted_sequences):
                data.corrected_daughter_list[i] = seq

            # Create the log Files
            if not os.path.exists(sim_name):
                os.makedirs(sim_name)

            data.conclusions(sim_name, level=verbose_level)

            with open(sim_name + '/log.txt', 'w') as f:
                f.write(f'Log File - Simulation :: {sim_name}\n\n')

                f.write(f'##INPUTS##\n')
                f.write(f'Random Seed for Reproducibility = {seed}\n\n')

                f.write(f'#Sample parameters#\n')
                f.write(f'alpha = {alpha}\n')
                f.write(f'beta = {beta}\n')
                f.write(f'mu = {mu}\n')
                f.write(f'rho = {rho}\n')
                f.write(f'number of n_samples = {n_samples}\n')

                f.write(f'#Fixed hyperparameters#\n')
                f.write(f'input_size = {200}\n')
                f.write(f'num_layers = {5}\n')
                f.write(f'hidden_size = {150}\n')
                f.write(f'output_size = {100}\n')

                f.write(f'##OUTPUTS##\n')
                f.write(f'Average BitError = {np.mean(data.biterror)}')

            with open(biterror_combination_path, 'a') as f:
                f.write(f'{round(alpha, 1)},{round(beta, 1)},{np.mean(data.biterror)}\n')
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

def generate_mom(alpha, beta, mu, n=100, start_as=2):
    """
    This function generates one sequence of the mother based on alpha, beta, mu and n values.
    the alpha and beta values are used to create the A part of the mother and the mu is used to create the B.    
    
    :param alpha: The probability that the next element node in the sequence is 1 given that the current node is 1.
    :param beta: The probability that the next element node in the sequence is 0 given that the current node is 0.
    :param mu: The probability that the element in the B sequence is 1 (only if A is 0).
    :param n: The length of the mother sequence.
    :param start_as: The first node of the mother sequence.
    :return: a numpy vector containing the sequence of the mother.
    """
    # Initializing the mother sequence
    mother_sequence_A = np.zeros([n], dtype=np.float32)
    mother_sequence_B = np.zeros([n], dtype=np.float32)
    mother_sequence = np.zeros([n], dtype=np.float32)

    # Generating the A Part.
    # if 0 or 1 is not passed as a sequence start, it is randomly assigned
    if start_as in [0, 1]:
        mother_sequence_A[0] = start_as
    else:
        mother_sequence_A[0] = np.random.randint(2)

    # Filling in the values of the sequence based on alpha and beta.
    for i in range(1, mother_sequence_A.size):
        if mother_sequence_A[i - 1] == 1:
            # The previous node is 1 so use alpha to set 1 else set 0
            if np.random.random() <= alpha:
                mother_sequence_A[i] = 1
            else:
                mother_sequence_A[i] = 0
        else:
            # The previous node is 0 so use beta to set 0 else set 1
            if np.random.random() <= beta:
                mother_sequence_A[i] = 0
            else:
                mother_sequence_A[i] = 1

    # Generating the B part.                
    for i in range(mother_sequence_B.size):
        if mother_sequence_A[i] == 0:
            if np.random.random() <= mu:
                mother_sequence_B[i] = 1
            else:
                mother_sequence_B[i] = 0
        else:
            pass

            # Generating the complete mother
    for i in range(mother_sequence_A.size):
        if mother_sequence_A[i] == 0 and mother_sequence_B[i] == 0:
            mother_sequence[i] = 0
        elif mother_sequence_A[i] == 1 and mother_sequence_B[i] == 0:
            mother_sequence[i] = 1
        elif mother_sequence_A[i] == 0 and mother_sequence_B[i] == 1:
            mother_sequence[i] = 2
        # mother_sequence[i] = 3 if mother_sequence_A[i] == 1 and mother_sequence_B[i] == 1
        # Flipping the binary counting order to help the network avoid errors while having to predict just 0 and 2.
        # Now it just has to predict 0 and 1 (if mu = 0)

    return mother_sequence


def split_sequence(seq):
    seq_A = np.copy(seq)
    seq_B = np.copy(seq)

    for i in range(seq.size):
        if seq[i] == 0:
            seq_A[i] = 0
            seq_B[i] = 0
        if seq[i] == 1:
            seq_A[i] = 1
            seq_B[i] = 0
        if seq[i] == 2:
            seq_A[i] = 0
            seq_B[i] = 1

    return seq_A, seq_B


def generate_daughter(mom, rho):
    """
    This function generates a daughter sequence from a mother sequence.
    It randomly mutates a node with a 50% probability of flipping.
    :param mom: The mother sequence that is to be used.
    :param rho: The probablity that mutation occurs for a bit if a mutation is possible.
    :return: a numpy vector containing the sequence of the daughter.
    """
    daughter_sequence = np.copy(mom)

    for i in range(daughter_sequence.size):
        if daughter_sequence[i] == 0:
            pass
        elif daughter_sequence[i] == 1:
            if np.random.random() <= rho:
                daughter_sequence[i] = 0
            else:
                daughter_sequence[i] = 1
        elif daughter_sequence[i] == 2:
            if np.random.random() <= rho:
                daughter_sequence[i] = 0
            else:
                daughter_sequence[i] = 2

    return daughter_sequence


def calculate_biterror(seq1, seq2):
    """
    This function will calculate the biterror for a pair of sequences.
    """

    xor = np.logical_xor(seq1, seq2).astype(int)

    biterror = np.sum(xor) / xor.size

    return biterror


class Dataframe:
    """
    This class is a dataframe which will hold all values for one simulation.
    i.e. it will contain the mothers, corrupted daughters, corrected daughters,
    the biterror rate and all other necessary data.

    We will initialise the object with an alpha, beta, number of moms,
    the starting value of the mom sequence, number of nodes in the sequences and
    a corrupted daughter.
    """

    def __init__(self, alpha, beta, mu, rho, n_moms, mom_length, mom_start=2):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.rho = rho
        self.mom_list = np.zeros([n_moms, mom_length])
        self.corrupt_daughter_list = np.zeros([n_moms, mom_length])
        self.corrected_daughter_list = np.zeros([n_moms, mom_length])

        self.biterror = np.zeros([n_moms])
        # initializing moms and making corrupt daughters
        for i in range(n_moms):
            # if we don't want all the moms to start with either 1 or 0,
            # we pass anything else to run it as a random node.
            new_mom = generate_mom(alpha=self.alpha,
                                   beta=self.beta,
                                   mu=self.mu,
                                   n=mom_length,
                                   start_as=mom_start)
            self.mom_list[i] = new_mom
            self.corrupt_daughter_list[i] = generate_daughter(new_mom, self.rho)

    def calc_errors(self):
        """
        This function will calculate the biterror for the mom_list and
        corrected_daughter_list.
        """
        for i in range(self.biterror.size):
            mom_A, mom_B = split_sequence(self.mom_list[i])
            daughter_A, daughter_B = split_sequence(self.corrected_daughter_list[i])

            self.biterror[i] = calculate_biterror(mom_A, daughter_A)

    def conclusions(self, sim_name='Untitled_Sim', level=3):
        """
        This function gives a summary of the dataframe.
        :param level: This specifies how much information is to be displayed.
        :param sim_name: Name of the simulation with which the folder is created to store the files.
        """
        # Create the log Files
        if not os.path.exists(sim_name):
            os.makedirs(sim_name)
        self.calc_errors()
        np.savetxt(sim_name + '/BitError.csv', self.biterror, delimiter=',')
        np.savetxt(sim_name + '/Mom_list.csv', self.mom_list, delimiter=',', fmt='%d')
        np.savetxt(sim_name + '/Corrupt_Daughter.csv', self.corrupt_daughter_list, delimiter=',', fmt='%d')
        np.savetxt(sim_name + '/Correct_Daughter.csv', self.corrected_daughter_list, delimiter=',', fmt='%d')
        if level > 0:
            print(f'Alpha = {self.alpha}')
            print(f'Beta = {self.beta}')
            print(f'Average BitError = {np.mean(self.biterror)}')
            if level >= 2:
                print(f'BitError = {self.biterror}')
                if level >= 3:
                    print('\nMom List')
                    print(self.mom_list)

                    print('\nCorrupt Daughter List')
                    print(self.corrupt_daughter_list)

                    print('\nCorrected Daughter List')
                    print(self.corrected_daughter_list)

        return 0


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_name', type=str, default='Range_Test_Simple', help='Name of the batch')
parser.add_argument('--weight_path', type=str, default='weights.pth',
                    help='Path to the NN Weights for the Extended Range Network')
parser.add_argument('--alpha_eval', type=str, default="[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]",
                    help='String to evaluate as the alpha list')
parser.add_argument('--beta_eval', type=str, default="[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]",
                    help='string to evaluate as the beta list')
parser.add_argument('--mu_eval', type=str, default="[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]",
                    help='String to evaluate the mu list')
parser.add_argument('--rho', type=float, default=0.5, help='rho value for the simulation')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples')
parser.add_argument('--verbose_level', type=int, default=5, help='Verbose level')

# to run the simulation use 
# python Run_Single_Network_Heatmap.py --batch_name Range_Test_Simple --weight_path <path> 

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

# Select device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if verbose_level > 0:
    print(f"Using device: {device}")

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

model = SequencePredictor().to(device)
model.load_state_dict(torch.load(network_weight_path, map_location=device))  # Load the network weights

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

            # Convert the numpy arrays to PyTorch tensors and move to device
            mom_A = torch.from_numpy(mom_A).float().to(device)
            daughter_A = torch.from_numpy(daughter_A).float().to(device)
            daughter_input = torch.from_numpy(daughter_input).float().to(device)

            # Print tensor device info for debugging
            if verbose_level > 1:
                print(f"mom_A device: {mom_A.device}")
                print(f"daughter_A device: {daughter_A.device}")
                print(f"daughter_input device: {daughter_input.device}")

            # Run the predictions.
            model.eval()  # To turn off learning mode.
            if verbose_level > 1:
                print(model)
            with torch.no_grad():
                output = model(daughter_input)
            if verbose_level > 1:
                print("Output before rounding:", output)

            predicted_sequences = output.round().cpu().numpy().astype(int)
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
                f.write(f'device = {device}\n')

                f.write(f'##OUTPUTS##\n')
                f.write(f'Average BitError = {np.mean(data.biterror)}')

            with open(biterror_combination_path, 'a') as f:
                f.write(f'{round(alpha, 1)},{round(beta, 1)},{np.mean(data.biterror)}\n')
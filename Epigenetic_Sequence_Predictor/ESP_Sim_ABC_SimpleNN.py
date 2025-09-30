import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy


# from torchviz import make_dot
# import pandas as pd


##################################################################################################

def generate_mom(alpha, beta, mu, gamma, n=100, start_as=2):
    """
    This function generates one sequence of the mother based on alpha, beta, mu and n values.
    the alpha and beta values are used to create the A part of the mother and the mu is used to create the B.

    :param alpha: The probability that the next element node in the sequence is 1 given that the current node is 1.
    :param beta: The probability that the next element node in the sequence is 0 given that the current node is 0.
    :param mu: The probability that the element in the B sequence is 1 (only if A is 0).
    :param gamma: The probability that the element in the C sequence is 1 if A is 1,
                  and the probability that the element in the C sequence is 0 if A is 0.
                  (If A is 1: C is set to 1 with probability gamma, else 0.
                   If A is 0: C is set to 0 with probability gamma, else 1.)
    :param n: The length of the mother sequence.
    :param start_as: The first node of the mother sequence.
    :return: a numpy vector containing the sequence of the mother.
    """
    # Initializing the mother sequence
    mother_sequence_A = np.zeros([n], dtype=np.float32)
    mother_sequence_B = np.zeros([n], dtype=np.float32)
    mother_sequence_C = np.zeros([n], dtype=np.float32)
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

    # Generating the C part.
    for i in range(mother_sequence_C.size):
        if mother_sequence_A[i] == 0:
            if np.random.random() <= gamma:
                mother_sequence_C[i] = 0
            else:
                mother_sequence_C[i] = 1
        elif mother_sequence_A[i] == 1:
            if np.random.random() <= gamma:
                mother_sequence_C[i] = 1
            else:
                mother_sequence_C[i] = 0
        else:
            pass    

    # New addition to include C part in the mother sequence
    # This is not that necessary but just to keep it in line with the old style
    # S = A B C
    # 0 = 0 0 0
    # 1 = 1 0 0
    # 2 = 0 1 0
    # 3 = 1 1 0
    # 4 = 0 0 1
    # 5 = 1 0 1
    # 6 = 0 1 1
    # 7 = 1 1 1




    # Generating the complete mother
    for i in range(mother_sequence_A.size):
        if mother_sequence_A[i] == 0 and mother_sequence_B[i] == 0 and mother_sequence_C[i] == 0:
            mother_sequence[i] = 0
        elif mother_sequence_A[i] == 1 and mother_sequence_B[i] == 0 and mother_sequence_C[i] == 0:
            mother_sequence[i] = 1
        elif mother_sequence_A[i] == 0 and mother_sequence_B[i] == 1 and mother_sequence_C[i] == 0:
            mother_sequence[i] = 2
        elif mother_sequence_A[i] == 1 and mother_sequence_B[i] == 1 and mother_sequence_C[i] == 0:
            mother_sequence[i] = 3
        elif mother_sequence_A[i] == 0 and mother_sequence_B[i] == 0 and mother_sequence_C[i] == 1:
            mother_sequence[i] = 4
        elif mother_sequence_A[i] == 1 and mother_sequence_B[i] == 0 and mother_sequence_C[i] == 1:
            mother_sequence[i] = 5
        elif mother_sequence_A[i] == 0 and mother_sequence_B[i] == 1 and mother_sequence_C[i] == 1:
            mother_sequence[i] = 6
        elif mother_sequence_A[i] == 1 and mother_sequence_B[i] == 1 and mother_sequence_C[i] == 1:
            mother_sequence[i] = 7 

    return mother_sequence


def split_sequence(seq):
    seq_A = np.copy(seq)
    seq_B = np.copy(seq)
    seq_C = np.copy(seq)

    for i in range(seq.size):
        if seq[i] == 0:
            seq_A[i] = 0
            seq_B[i] = 0
            seq_C[i] = 0
        elif seq[i] == 1:
            seq_A[i] = 1
            seq_B[i] = 0
            seq_C[i] = 0
        elif seq[i] == 2:
            seq_A[i] = 0
            seq_B[i] = 1
            seq_C[i] = 0
        elif seq[i] == 3:
            seq_A[i] = 1
            seq_B[i] = 1
            seq_C[i] = 0
        elif seq[i] == 4:
            seq_A[i] = 0
            seq_B[i] = 0
            seq_C[i] = 1
        elif seq[i] == 5:
            seq_A[i] = 1
            seq_B[i] = 0
            seq_C[i] = 1
        elif seq[i] == 6:
            seq_A[i] = 0
            seq_B[i] = 1
            seq_C[i] = 1
        elif seq[i] == 7:
            seq_A[i] = 1
            seq_B[i] = 1
            seq_C[i] = 1
    return seq_A, seq_B, seq_C


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
        elif daughter_sequence[i] == 3:
            if np.random.random() <= rho:
                daughter_sequence[i] = 0
            else:
                daughter_sequence[i] = 3
        elif daughter_sequence[i] == 4:
            if np.random.random() <= rho:
                daughter_sequence[i] = 0
            else:
                daughter_sequence[i] = 4
        elif daughter_sequence[i] == 5:
            if np.random.random() <= rho:
                daughter_sequence[i] = 0
            else:
                daughter_sequence[i] = 5
        elif daughter_sequence[i] == 6:
            if np.random.random() <= rho:
                daughter_sequence[i] = 0
            else:
                daughter_sequence[i] = 6
        elif daughter_sequence[i] == 7:
            if np.random.random() <= rho:
                daughter_sequence[i] = 0
            else:
                daughter_sequence[i] = 7
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

    def __init__(self, alpha, beta, mu, gamma, rho, n_moms, mom_length, mom_start=2):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.gamma = gamma
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
                                   gamma=self.gamma,
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
            mom_A, mom_B, mom_C = split_sequence(self.mom_list[i])
            daughter_A, daughter_B, daughter_C = split_sequence(self.corrected_daughter_list[i])

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


class SequencePredictor(nn.Module):
    # Define the neural network architecture
    # (input -> h1 -> h2 .... hn -> output)
    def __init__(self, input_size, hidden_size, output_size, num_hid_layers):
        super(SequencePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_hid_layers = num_hid_layers

        # This is the first layer which takes in inputs and gives out the outputs for the first hidden layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # These are the hidden layers (use ModuleList for proper registration)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.num_hid_layers)])

        # This is the output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.relu(out)
        for layer in self.hidden_layers:
            out = layer(out)
            out = self.relu(out)
        out = self.output_layer(out)
        return out


def run_sim(sim_id, seed=42,
            alpha=0.8, beta=0.8, mu=0.5, gamma=0.5, rho=0.5, n_samples=100000, n_samples_test=30000, seq_length=100,
            input_size=300, hidden_size=int(1.5 * 100), output_size=100, num_layers=5,
            learning_rate=0.001, num_epochs=10000, batch_size=100000, loss_function_type='MSE', verbose_level=1,
            visualise_nn=False):
    """
    Runs the simulation with the given parameters.
    :param sim_id: A Unique ID(string) so that no 2 sims replaces the dirs created by the other unless specified.
    :param seed: Random seed for reproducibility.
    :param alpha: The probability that the next element node in the sequence is 1 given that the current node is 1.
    :param beta: The probability that the next element node in the sequence is 0 given that the current node is 0.
    :param mu: The porbablity that the B part of the mom is 1 if the A part of the mom is 0.
    :param gamma: The probability that the element in the C sequence is 1 (only if A is 1).
    :param rho: the probablity that a mutation pushes the node to 0 if the A part of the mom is 0.
    :param n_samples: The number of samples used for training.
    :param n_samples_test: The number of samples used for testing.
    :param seq_length: The length of the sequences.
    :param input_size: The size of the input for the neural network model.
    :param hidden_size: The size of the hidden layers in the neural network model.
    :param output_size: The size of the output for the neural network model.
    :param num_layers: The number of hidden layers in the neural network model.
    :param learning_rate: The learning rate for the optimizer.
    :param num_epochs: The number of epochs to train the neural network model.
    :param batch_size: The batch size for training the neural network model (Should be a divisor of the number of
    training samples.).
    :param loss_function_type: The type of loss function to use ('MSE' or 'BCE').
    :param verbose_level: The level of verbosity during run.
    :param visualise_nn: Weather or not to visualise the neural network architecture.

    :return: The name of the simulation and The bit error rate after testing for the selected alpha and beta values.
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Naming the simulation based on the sim_id, loss and alpha beta.
    # avoiding '.' in the name by using '_'
    sim_name = f'{sim_id}_{loss_function_type}_A{str(alpha).replace(".", "_")}_B{str(beta).replace(".", "_")}'

    # Giving Introduction
    if verbose_level > 0:
        print("Epigenetic Sequence Predictor")
        print(f"Sim Name :: {sim_name}")
        print()
    if verbose_level > 1:
        print("Starting data preparation...")

    data = Dataframe(alpha, beta, mu, gamma, rho, n_samples, seq_length)

    mother_sequences = data.mom_list
    corrupted_daughter_sequences = data.corrupt_daughter_list

    mom_A = np.copy(mother_sequences)
    mom_B = np.copy(mother_sequences)
    mom_C = np.copy(mother_sequences)
    daughter_A = np.copy(mother_sequences)
    daughter_B = np.copy(mother_sequences)
    daughter_C = np.copy(mother_sequences)
    daughter_input = np.zeros([mother_sequences.shape[0], 3 * mother_sequences.shape[1]])

    for i in range(mother_sequences.shape[0]):
        mom_A[i], mom_B[i], mom_C[i] = split_sequence(mother_sequences[i])
        daughter_A[i], daughter_B[i], daughter_C[i] = split_sequence(corrupted_daughter_sequences[i])
        daughter_input[i] = np.concatenate([daughter_A[i], daughter_B[i], daughter_C[i]])

    # Select device: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose_level > 0:
        print(f"Using device: {device}")

    # Move tensors to device
    mom_A = torch.from_numpy(mom_A).float().to(device)
    daughter_input = torch.from_numpy(daughter_input).float().to(device)

    # Print tensor device info for debugging
    print(f"mom_A device: {mom_A.device}")
    print(f"daughter_input device: {daughter_input.device}")

    if verbose_level > 1:
        print("Data preparation finished.")
        print("Converting data to PyTorch tensors...")
        print("Tensor conversion finished.")
        print("Creating model...")

    # Create the model and move to device
    model = SequencePredictor(input_size, hidden_size, output_size, num_layers).to(device)

    if verbose_level > 1:
        print("Model created.")
        print("Setting up loss function and optimizer...")

    # Define the loss function and optimizer
    if loss_function_type == 'MSE':
        criterion = nn.MSELoss()
    else:
        print("You seem to have messed up the input of the loss function.")
        print("Try looking at your input in the loss_function_type.")
        exit()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if verbose_level > 1:
        print("Loss function and optimizer set.")
        print("Starting training...")

    # Create an empty 2D NumPy array to store the epoch and loss values
    training_loss = np.empty((num_epochs, 2))

    # Train the model
    for epoch in range(num_epochs):
        for i in range(0, daughter_A.shape[0], batch_size):
            # Forward pass
            outputs = model(daughter_input[i:i + batch_size])
            loss = criterion(outputs, mom_A[i:i + batch_size])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store the epoch and loss values directly in the data array
            training_loss[epoch, 0] = epoch + 1
            training_loss[epoch, 1] = loss.item()

        # Print the loss for every 10 epochs
        if verbose_level > 0:
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    if verbose_level > 1:
        print("Training finished.")
        print("Preparing test data...")

    # Test the model.
    test_data = Dataframe(alpha, beta, mu, gamma, rho, n_samples_test, seq_length)

    mother_sequences = test_data.mom_list
    corrupted_daughter_sequences = test_data.corrupt_daughter_list

    mom_A = np.copy(mother_sequences)
    mom_B = np.copy(mother_sequences)
    mom_C = np.copy(mother_sequences)
    daughter_A = np.copy(mother_sequences)
    daughter_B = np.copy(mother_sequences)
    daughter_C = np.copy(mother_sequences)
    daughter_input = np.zeros([mother_sequences.shape[0], 3 * mother_sequences.shape[1]])

    for i in range(mother_sequences.shape[0]):
        mom_A[i], mom_B[i], mom_C[i] = split_sequence(mother_sequences[i])
        daughter_A[i], daughter_B[i], daughter_C[i] = split_sequence(corrupted_daughter_sequences[i])
        daughter_input[i] = np.concatenate([daughter_A[i], daughter_B[i], daughter_C[i]])

    if verbose_level > 1:
        print("Test data preparation finished.")
        print("Converting test data to PyTorch tensors...")
        print("Test tensor conversion finished.")
        print("Running model on test data...")

    # Move test tensors to device
    daughter_A = torch.from_numpy(daughter_A).float().to(device)
    daughter_input = torch.from_numpy(daughter_input).float().to(device)

    # Print tensor device info for debugging (test data)
    print(f"daughter_A (test) device: {daughter_A.device}")
    print(f"daughter_input (test) device: {daughter_input.device}")

    # Test the models
    model.eval()
    with torch.no_grad():
        test_output = model(daughter_input)

    if verbose_level > 1:
        print("Test predictions finished.")
        print("Processing predictions and saving results...")

    # Move predictions back to CPU for numpy conversion
    model_predicted_sequences = test_output.round().cpu().numpy().astype(int)
    for i, seq in enumerate(model_predicted_sequences):
        test_data.corrected_daughter_list[i] = seq

    # if visualise_nn:
    #     # Visualize the computation graph
    #     inputs = daughter_input[:1]  # Take the first sequence for visualization
    #
    #     make_dot(extended_model(inputs), params=dict(extended_model.named_parameters())).render(
    #         f"{sim_name + '_Extended'}/nn_graph", format='png')
    #
    #     make_dot(extended_model(inputs), params=dict(extended_model.named_parameters()), show_attrs=True).render(
    #         f"{sim_name + '_Extended'}/nn_graph_with_attr", format='png')
    #
    #     make_dot(extended_model(inputs), params=dict(extended_model.named_parameters()), show_attrs=True,
    #              show_saved=True).render(f"{sim_name + '_Extended'}/nn_graph_with_attr_saved", format='png')

    # save_weights_biases_csv(model, sim_name)
    # save_weights_biases_csv(extended_model, sim_name + '_Extended')

    # Create the log Files
    if not os.path.exists(sim_name):
        os.makedirs(sim_name)

    # Get conclusions
    print('*************Conclusions*******************')
    test_data.conclusions(sim_name, level=verbose_level)

    # Save the network weights
    torch.save(model.state_dict(), f'{sim_name}/network_weights.pth')

    # when needed load these weights as follows.
    # Assuming you've defined your model class as `SequencePredictor` and created an instance named `model`
    # model.load_state_dict(torch.load('path/to/network_weights.pth'))

    np.savetxt(sim_name + '/initial_training_loss.csv', training_loss, delimiter=',')

    with open(sim_name + '/log.txt', 'w') as f:
        f.write(f'Log File - Simulation :: {sim_name}\n\n')

        f.write(f'##INPUTS##\n')
        f.write(f'Random Seed for Reproducibility = {seed}\n\n')

        f.write(f'#Sample parameters#\n')
        f.write(f'alpha = {alpha}\n')
        f.write(f'beta = {beta}\n')
        f.write(f'mu = {mu}\n')
        f.write(f'gamma = {gamma}\n')
        f.write(f'rho = {rho}\n')
        f.write(f'seq_length = {seq_length}\n')
        f.write(f'Training n_samples = {n_samples}\n')
        f.write(f'Testing n_samples = {n_samples_test}\n\n')

        f.write(f'#Define hyperparameters#\n')
        f.write(f'Loss Function Type = {loss_function_type}\n')
        f.write(f'input_size = {input_size}\n')
        f.write(f'num_layers = {num_layers}\n')
        f.write(f'hidden_size = {hidden_size}\n')
        f.write(f'output_size = {output_size}\n')
        f.write(f'learning_rate = {learning_rate}\n')
        f.write(f'num_epochs = {num_epochs}\n')
        f.write(f'batch_size = {batch_size}\n\n')

        f.write(f'##OUTPUTS##\n')
        f.write(f'Average BitError(Initial_Network) = {np.mean(test_data.biterror)}')

    print("Simulation complete.")

    return sim_name, np.mean(test_data.biterror)
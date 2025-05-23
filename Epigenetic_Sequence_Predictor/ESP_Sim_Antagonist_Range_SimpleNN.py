import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


##################################################################################################

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

    We will initialise the object with a list of alpha, beta, number of moms,
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
            new_mom = generate_mom(alpha=self.alpha[i],
                                   beta=self.beta[i],
                                   mu=self.mu[i],
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
        np.savetxt(sim_name + '/alpha_list.csv', self.alpha, delimiter=',')
        np.savetxt(sim_name + '/beta_list.csv', self.beta, delimiter=',')
        np.savetxt(sim_name + '/mu_list.csv', self.mu, delimiter=',')
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

        # These are the hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hid_layers)])

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
            alpha_min=0.0, alpha_max=1.0, beta_min=0.0, beta_max=1.0, mu_min=0.0, mu_max=1.0, rho=0.5, n_samples=100000,
            n_samples_test=30000, seq_length=100, input_size=200, hidden_size=int(1.5 * 100), output_size=100,
            num_layers=5, learning_rate=0.001, num_epochs=20000, batch_size=100000, loss_function_type='MSE',
            verbose_level=5):
    # The doc string needs to be corrected.
    """
    Runs the simulation with the given parameters.
    :param sim_id: A Unique ID(string) so that no 2 sims replaces the dirs created by the other unless specified.
    :param seed: Random seed for reproducibility.
    :param alpha_min: The minimum probability for alpha.
    :param alpha_max: The maximum probability for alpha.
    :param beta_min: The minimum probability for beta.
    :param beta_max: The maximum probability for beta.
    :param mu_min: The minimum probability for mu.
    :param mu_max: The maximum probability for mu.
    :param rho: The probability of mutation in the daughter sequence.
    :param n_samples: The number of samples used for training.
    :param n_samples_test: The number of samples used for testing.
    :param seq_length: The length of the sequences.
    :param input_size: The size of the input for the neural network model.
    :param hidden_size: The size of the hidden layers in the neural network model.
    :param output_size: The size of the output for the neural network model.
    :param num_layers: The number of hidden layers in the neural network model.
    :param learning_rate: The learning rate for the optimizer.
    :param num_epochs: The number of epochs to train the neural network model.
    :param batch_size: The batch size for training the neural network model.
    :param loss_function_type: The type of loss function to use ('MSE' or 'BCE').
    :param verbose_level: The level of verbosity during run.
    :return: The name of the simulation and The bit error rate after testing for the selected alpha and beta values.
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Naming the simulation based on the sim_id, loss and alpha beta.
    # avoiding '.' in the name by using '_'
    sim_name = f'{sim_id}'

    # Giving Introduction
    if verbose_level > 0:
        print("Epigenetic Sequence Predictor Simple NN Range")
        print(f"Sim Name :: {sim_name}")
        print()

    #############################################################################################################
    # Prepare the data for the initial network
    if verbose_level > 0:
        print("Training the Network")

    alpha_list = np.random.uniform(low=alpha_min, high=alpha_max, size=n_samples)
    beta_list = np.random.uniform(low=beta_min, high=beta_max, size=n_samples)
    mu_list = np.random.uniform(low=mu_min, high=mu_max, size=n_samples)

    data = Dataframe(alpha_list, beta_list, mu_list, rho, n_samples, seq_length)

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

    mom_A = torch.from_numpy(mom_A).float()
    daughter_input = torch.from_numpy(daughter_input).float()

    model = SequencePredictor(input_size, hidden_size, output_size, num_layers)

    # Define the loss function and optimizer
    if loss_function_type == 'MSE':
        criterion = nn.MSELoss()
    elif loss_function_type == 'BCE':
        criterion = nn.BCELoss()
    else:
        print("You seem to have messed up the input of the loss function.")
        print("Try looking at your input in the loss_function_type.")
        exit()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create an empty 2D NumPy array to store the epoch and loss values
    training_loss = np.empty((num_epochs, 2))

    # Train the model
    for epoch in range(num_epochs):
        for i in range(0, daughter_input.shape[0], batch_size):
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

    ###################################################################################################################
    # Now we have the  Network

    # Test the model.

    test_data = Dataframe(alpha_list, beta_list, mu_list, rho, n_samples_test, seq_length)

    mother_sequences = test_data.mom_list
    corrupted_daughter_sequences = test_data.corrupt_daughter_list

    mom_A = np.copy(mother_sequences)
    mom_B = np.copy(mother_sequences)
    daughter_A = np.copy(mother_sequences)
    daughter_B = np.copy(mother_sequences)
    daughter_input = np.zeros([mother_sequences.shape[0], 2 * mother_sequences.shape[1]])

    for i in range(mother_sequences.shape[0]):
        mom_A[i], mom_B[i] = split_sequence(mother_sequences[i])
        daughter_A[i], daughter_B[i] = split_sequence(corrupted_daughter_sequences[i])
        daughter_input[i] = np.append(daughter_A[i], daughter_B[i])

    daughter_A = torch.from_numpy(daughter_A).float()
    daughter_input = torch.from_numpy(daughter_input).float()

    # Test the models
    model.eval()

    with torch.no_grad():
        test_output = model(daughter_input)

    model_predicted_sequences = test_output.round().numpy().astype(int)
    for i, seq in enumerate(model_predicted_sequences):
        test_data.corrected_daughter_list[i] = seq

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

    np.savetxt(sim_name + '/training_loss.csv', training_loss, delimiter=',')

    with open(sim_name + '/log.txt', 'w') as f:
        f.write(f'Log File - Simulation :: {sim_name}\n\n')

        f.write(f'##INPUTS##\n')
        f.write(f'Random Seed for Reproducibility = {seed}\n\n')

        f.write(f'#Sample parameters#\n')
        f.write(f'alpha = {alpha_min}, {alpha_max}\n')
        f.write(f'beta = {beta_min}, {beta_max}\n')
        f.write(f'mu = {mu_min}, {beta_max}\n')
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

    return sim_name, np.mean(test_data.biterror)

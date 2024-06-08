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
        self.alpha_predicted = np.zeros(n_moms)
        self.beta_predicted = np.zeros(n_moms)
        self.mu_predicted = np.zeros(n_moms)
        self.alpha_error = np.zeros(n_moms)
        self.beta_error = np.zeros(n_moms)
        self.mu_error = np.zeros(n_moms)

        # initializing moms
        for i in range(n_moms):
            # if we don't want all the moms to start with either 1 or 0,
            # we pass anything else to run it as a random node.
            new_mom = generate_mom(alpha=self.alpha[i],
                                   beta=self.beta[i],
                                   mu=self.mu[i],
                                   n=mom_length,
                                   start_as=mom_start)
            self.mom_list[i] = new_mom

    def calc_errors(self):
        print(self.alpha)
        print(self.alpha_predicted)
        self.alpha_error = self.alpha_predicted - self.alpha
        print(self.alpha_error)
        self.beta_error = self.beta_predicted - self.beta
        self.mu_error = self.mu_predicted - self.mu

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
        np.savetxt(sim_name + '/Mom_list.csv', self.mom_list, delimiter=',', fmt='%d')

        np.savetxt(sim_name + '/alpha_list.csv', self.alpha, delimiter=',')
        np.savetxt(sim_name + '/beta_list.csv', self.beta, delimiter=',')
        np.savetxt(sim_name + '/mu_list.csv', self.mu, delimiter=',')

        np.savetxt(sim_name + '/alpha_predicted_list.csv', self.alpha_predicted, delimiter=',')
        np.savetxt(sim_name + '/beta_predicted_list.csv', self.beta_predicted, delimiter=',')
        np.savetxt(sim_name + '/mu_predicted_list.csv', self.mu_predicted, delimiter=',')

        np.savetxt(sim_name + '/alpha_error_list.csv', self.alpha_error, delimiter=',')
        np.savetxt(sim_name + '/beta_error_list.csv', self.beta_error, delimiter=',')
        np.savetxt(sim_name + '/mu_error_list.csv', self.mu_error, delimiter=',')

        if level > 0:
            print(f'Alpha = {self.alpha}')
            print(f'Beta = {self.beta}')
            print(f'Mu = {self.mu}')

            print(f'Average Alpha Error = {np.mean(self.alpha_error)}')
            print(f'Average Beta Error = {np.mean(self.beta_error)}')
            print(f'Average Mu Error = {np.mean(self.mu_error)}')

            if level >= 2:
                if level >= 3:
                    print('\nMom List')
                    print(self.mom_list)

        return 0


class ParamPredictor(nn.Module):
    # Define the neural network architecture
    # (input -> h1 -> h2 .... hn -> output)
    def __init__(self, input_size, hidden_size, num_hid_layers, output_size=3):
        super(ParamPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_hid_layers = num_hid_layers

        # This is the first layer which takes in inputs and gives out the outputs for the first hidden layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # These are the hidden layers
        self.hidden_layers = [nn.Linear(hidden_size, hidden_size) for _ in range(self.num_hid_layers)]

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
            alpha_min=0.0, alpha_max=1.0, beta_min=0.0, beta_max=1.0, mu_min=0.0, mu_max=1.0, rho=0.5, n_samples=10000,
            n_samples_test=3000, seq_length=100, input_size=200, hidden_size=int(1.5 * 100),
            num_layers=5, learning_rate=0.001, num_epochs=10000, batch_size=10000, loss_function_type='MSE',
            verbose_level=1):
    """
    Runs a neural network-based simulation with the given parameters, focusing on sequence prediction with controlled
    mutation and transition probabilities.

    Parameters:
    :param sim_id: str - A unique ID to ensure separate simulations do not overwrite each other's data.
    :param seed: int, default 42 - Seed for random number generators to ensure reproducibility.
    :param alpha_min: float, default 0.7 - Minimum transition probability from 1 to 1.
    :param alpha_max: float, default 0.9 - Maximum transition probability from 1 to 1.
    :param beta_min: float, default 0.6 - Minimum transition probability from 0 to 0.
    :param beta_max: float, default 0.8 - Maximum transition probability from 0 to 0.
    :param mu_min: float, default 0.3 - Minimum mutation probability from 1 to 0.
    :param mu_max: float, default 0.5 - Maximum mutation probability from 1 to 0.
    :param rho: float, default 0.5 - Probability that a mutation pushes the node to 0 if the 'A' part of the model is 0.
    :param n_samples: int, default 10000 - Number of samples for training. Total training samples = 2 * n_samples.
    :param n_samples_test: int, default 3000 - Number of samples for testing.
    :param seq_length: int, default 100 - Length of the sequences to be generated.
    :param input_size: int, default 100 - Size of the input layer for the neural network model.
    :param hidden_size: int, default 150 - Size of the hidden layers in the neural network model.
    :param num_layers: int, default 5 - Number of hidden layers in the neural network model.
    :param learning_rate: float, default 0.001 - Learning rate for the optimizer.
    :param num_epochs: int, default 10000 - Number of epochs for training the model.
    :param batch_size: int, default 10000 - Batch size for training. Should be a divisor of the number of training samples.
    :param loss_function_type: str, default 'MSE' - Type of the loss function ('MSE' for Mean Squared Error, 'BCE' for Binary Cross-Entropy).
    :param verbose_level: int, default 1 - Verbosity level of the simulation run. Higher numbers increase the detail of logs.

    Returns:
    The name of the simulation and the errors while detection.
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Naming the simulation based on the sim_id, loss and alpha beta.
    # avoiding '.' in the name by using '_'
    sim_name = f'{sim_id}'

    # Giving Introduction
    if verbose_level > 0:
        print("Alpha Beta Mu Predictor")
        print(f"Sim Name :: {sim_name}")
        print()

    #############################################################################################################
    # Prepare the data for the initial network
    if verbose_level > 0:
        print("Training the  Network")

    alpha_list = np.random.uniform(low=alpha_min, high=alpha_max, size=n_samples)
    beta_list = np.random.uniform(low=beta_min, high=beta_max, size=n_samples)
    mu_list = np.random.uniform(low=mu_min, high=mu_max, size=n_samples)

    data = Dataframe(alpha_list, beta_list, mu_list, rho, n_samples, seq_length)

    mother_sequences = data.mom_list

    mom_A = np.copy(mother_sequences)
    mom_B = np.copy(mother_sequences)
    mom_input = np.zeros([mother_sequences.shape[0], 2 * mother_sequences.shape[1]])
    param_output = np.column_stack((data.alpha, data.beta, data.mu))

    for i in range(mother_sequences.shape[0]):
        mom_A[i], mom_B[i] = split_sequence(mother_sequences[i])
        mom_input[i] = np.append(mom_A[i], mom_B[i])

    # Convert the numpy arrays to PyTorch tensors
    mom_input = torch.from_numpy(mom_input).float()
    param_output = torch.from_numpy(param_output).float()

    # Create the model
    model = ParamPredictor(input_size, hidden_size, num_layers)

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
        for i in range(0, mom_A.shape[0], batch_size):
            # Forward pass
            outputs = model(mom_input[i:i + batch_size])
            loss = criterion(outputs, param_output[i:i + batch_size])

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
    # Now we have the network trained.

    # Test the  model

    alpha_list = np.random.uniform(low=alpha_min, high=alpha_max, size=n_samples_test)
    beta_list = np.random.uniform(low=beta_min, high=beta_max, size=n_samples_test)
    mu_list = np.random.uniform(low=mu_min, high=mu_max, size=n_samples_test)

    test_data = Dataframe(alpha_list, beta_list, mu_list, rho, n_samples_test, seq_length)

    mother_sequences = test_data.mom_list

    mom_A = np.copy(mother_sequences)
    mom_B = np.copy(mother_sequences)
    mom_input = np.zeros([mother_sequences.shape[0], 2 * mother_sequences.shape[1]])

    for i in range(mother_sequences.shape[0]):
        mom_A[i], mom_B[i] = split_sequence(mother_sequences[i])
        mom_input[i] = np.append(mom_A[i], mom_B[i])

    # Convert the numpy arrays to PyTorch tensors
    mom_input = torch.from_numpy(mom_input).float()

    # Test the models
    model.eval()

    with torch.no_grad():
        test_output = model(mom_input)

    for i, seq in enumerate(test_output):
        test_data.alpha_predicted[i] = seq[0]
        test_data.beta_predicted[i] = seq[1]
        test_data.mu_predicted[i] = seq[2]

    # Create the log Files
    if not os.path.exists(sim_name):
        os.makedirs(sim_name)

    # Get conclusions
    print('*************Initial_Conclusions*******************')
    test_data.conclusions(sim_name, level=verbose_level)

    # Save the initial network weights
    torch.save(model.state_dict(), f'{sim_name}/network_weights.pth')

    # when needed load these weights as follows.
    # Assuming you've defined your model class as `SequencePredictorInitial` and created an instance named `model`
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
        f.write(f'output_size = 3\n')
        f.write(f'learning_rate = {learning_rate}\n')
        f.write(f'num_epochs = {num_epochs}\n')
        f.write(f'batch_size = {batch_size}\n\n')

        f.write(f'##OUTPUTS##\n')
        f.write(f'Average Alpha Error = {np.mean(test_data.alpha_error)}')
        f.write(f'Average Beta Error = {np.mean(test_data.beta_error)}')
        f.write(f'Average Mu Error = {np.mean(test_data.mu_error)}')

    return sim_name, np.mean(test_data.alpha_error), np.mean(test_data.beta_error), np.mean(test_data.mu_error)

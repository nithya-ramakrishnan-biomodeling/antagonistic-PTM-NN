import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random



############################################################################
# Need to run only once.

# raw_data = pd.read_csv('../yeast_db1_utf8.csv')
# print(raw_data.head())

# # Columns with the modifications have the same name with the second row being of the time step.
# # as the name of the modification is repeated, pandas appends .1, .2, etc. to the column names.
# # We will rename the columns to have unique names by appending the time step to the modification name.

# modifications = ['H2AK5ac','H2AS129ph','H3K14ac','H3K18ac','H3K23ac','H3K27ac','H3K36me','H3K36me2','H3K36me3','H3K4ac','H3K4me','H3K4me2','H3K4me3','H3K56ac','H3K79me','H3K79me3','H3K9ac','H3S10ph','H4K12ac','H4K16ac','H4K20me','H4K5ac','H4K8ac','H4R3me','H4R3me2s','Htz1']
# modifications_t4 = [mod + '.1' for mod in modifications]
# modifications_t8 = [mod + '.2' for mod in modifications]
# modifications_t15 = [mod + '.3' for mod in modifications]
# modifications_t30 = [mod + '.4' for mod in modifications]
# modifications_t60 = [mod + '.5' for mod in modifications]

# # rename the columns 
# for i, mod in enumerate(modifications):
#     raw_data.rename(columns={mod: mod + '_t0'}, inplace=True)
#     raw_data.rename(columns={modifications_t4[i]: mod + '_t4'}, inplace=True)
#     raw_data.rename(columns={modifications_t8[i]: mod + '_t8'}, inplace=True)
#     raw_data.rename(columns={modifications_t15[i]: mod + '_t15'}, inplace=True)
#     raw_data.rename(columns={modifications_t30[i]: mod + '_t30'}, inplace=True)
#     raw_data.rename(columns={modifications_t60[i]: mod + '_t60'}, inplace=True)

# raw_data.to_csv('../yeast_db1_renamed.csv', index=False, encoding='utf-8')
############################################################################
def calculate_alpha_beta(binary_sequence):
    """
    Calculates alpha and beta values for a given binary sequence.
    Alpha is the transition probability from 1 to 1, and Beta is from 0 to 0.

    Parameters:
    binary_sequence (str): A string of binary digits (1s and 0s).

    Returns:
    tuple: A tuple containing the alpha and beta values.
    """
    n11, n10, n01, n00 = 0, 0, 0, 0
    sequence = [int(bit) for bit in binary_sequence]

    for i in range(1, len(sequence)):
        if sequence[i - 1] == 1 and sequence[i] == 1:
            n11 += 1
        elif sequence[i - 1] == 1 and sequence[i] == 0:
            n10 += 1
        elif sequence[i - 1] == 0 and sequence[i] == 1:
            n01 += 1
        elif sequence[i - 1] == 0 and sequence[i] == 0:
            n00 += 1

    alpha = n11 / (n11 + n10) if (n11 + n10) > 0 else 0
    beta = n00 / (n00 + n01) if (n00 + n01) > 0 else 0

    return alpha, beta

def generate_daughter(mom, rho=0.5):
    """
    This function generates a daughter sequence from a mother sequence.
    It randomly mutates a node with a 50% probability of flipping.
    :param mom: The mother sequence that is to be used.
    :param rho: The probability that a moms 1 is flipped to 0.
    :return: a numpy vector containing the sequence of the daughter.
    """
    daughter_sequence = np.copy(mom)

    # using a mod 2 to flip the nodes with a 50% probability
    for i in range(daughter_sequence.size):
        if daughter_sequence[i] == 1 and np.random.random() <= rho:
            daughter_sequence[i] = (daughter_sequence[i] + 1) % 2

    return daughter_sequence

class SequencePredictor_noanta(nn.Module):
    # Define the neural network architecture
    # (input -> h1 -> h2 .... hn -> output)
    def __init__(self):
        super(SequencePredictor_noanta, self).__init__()
        self.input_size = 100
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

def decode_with_noanta_nn(corrupt_daughter, nn_model_noanta):
    # the nn_model variable will contain the neural network with the loaded weights to work with
    daughter_input = torch.from_numpy(corrupt_daughter).float()

    # Run the predictions.
    nn_model_noanta.eval()  # To turn off learning mode.
    with torch.no_grad():
        corrected_daughter = nn_model_noanta(daughter_input)

    # print(corrected_daughter)
    return corrected_daughter

def calculate_biterror(seq1, seq2):
    """
    Calculates the overall bit error rate between two sequences.
    The bit error rate is the fraction of positions where the two sequences differ,
    i.e., the number of bits that are different (either 1→0 or 0→1) divided by the total number of bits.

    :param seq1: The original (reference) sequence as a numpy array of 0s and 1s.
    :param seq2: The processed (possibly corrupted and corrected) sequence as a numpy array of 0s and 1s.
    :return: The fraction of bits that are different between seq1 and seq2.
    """
    xor = np.logical_xor(seq1, seq2).astype(int)
    biterror = np.sum(xor) / xor.size
    return biterror

def calculate_biterror_1(seq1, seq2):
    """
    Calculates the fraction of 1s in the original sequence (seq1) that became 0 in the processed sequence (seq2).
    This measures the loss of 1s due to corruption or incorrect correction, normalized by the total number of 1s in seq1.

    :param seq1: The original (reference) sequence as a numpy array of 0s and 1s.
    :param seq2: The processed (possibly corrupted and corrected) sequence as a numpy array of 0s and 1s.
    :return: The fraction of 1s in seq1 that became 0s in seq2.
    """
    errors_1_to_0 = np.sum((seq1 == 1) & (seq2 == 0))
    total_ones = np.sum(seq1 == 1)
    fraction_1_to_0 = errors_1_to_0 / total_ones if total_ones > 0 else 0.0
    return fraction_1_to_0

def calculate_biterror_0(seq1, seq2):
    """
    Calculates the fraction of 0s in the original sequence (seq1) that became 1 in the processed sequence (seq2).
    This measures the incorrect introduction of 1s during correction, normalized by the total number of 0s in seq1.

    :param seq1: The original (reference) sequence as a numpy array of 0s and 1s.
    :param seq2: The processed (possibly corrupted and corrected) sequence as a numpy array of 0s and 1s.
    :return: The fraction of 0s in seq1 that became 1s in seq2.
    """
    errors_0_to_1 = np.sum((seq1 == 0) & (seq2 == 1))
    total_zeros = np.sum(seq1 == 0)
    fraction_0_to_1 = errors_0_to_1 / total_zeros if total_zeros > 0 else 0.0
    return fraction_0_to_1

def calc_errors(mom_list, corrected_daughter_list):
    """
    This function will calculate the biterror for the mom_list and
    corrected_daughter_list.
    """
    biterror = np.zeros(mom_list.shape[0])
    biterror_1 = np.zeros(mom_list.shape[0])
    biterror_0 = np.zeros(mom_list.shape[0])
    for i in range(biterror.size):
        biterror[i] = calculate_biterror(mom_list[i], corrected_daughter_list[i])
        biterror_1[i] = calculate_biterror_1(mom_list[i], corrected_daughter_list[i])
        biterror_0[i] = calculate_biterror_0(mom_list[i], corrected_daughter_list[i])

    return biterror, biterror_1, biterror_0

def calculate_alpha_beta_vectorized(mother_sequences):
    """
    Vectorized calculation of alpha and beta for multiple binary sequences.

    Parameters:
    mother_sequences (numpy.ndarray): A 2D NumPy array where each row represents a binary sequence.

    Returns:
    tuple: Two 1D NumPy arrays containing alpha and beta values for each sequence.
           - Alpha: Transition probability from 1 to 1.
           - Beta: Transition probability from 0 to 0.
    """
    # Count transitions from 1 to 1, 1 to 0, 0 to 1, and 0 to 0 for all sequences
    n11 = np.sum((mother_sequences[:, :-1] == 1) & (mother_sequences[:, 1:] == 1), axis=1)
    n10 = np.sum((mother_sequences[:, :-1] == 1) & (mother_sequences[:, 1:] == 0), axis=1)
    n01 = np.sum((mother_sequences[:, :-1] == 0) & (mother_sequences[:, 1:] == 1), axis=1)
    n00 = np.sum((mother_sequences[:, :-1] == 0) & (mother_sequences[:, 1:] == 0), axis=1)

    # Calculate alpha and beta using vectorized division, handling division by zero
    alpha = np.divide(n11, n11 + n10, out=np.zeros_like(n11, dtype=float), where=(n11 + n10) > 0)
    beta = np.divide(n00, n00 + n01, out=np.zeros_like(n00, dtype=float), where=(n00 + n01) > 0)

    return alpha, beta

def generate_daughters_vectorized(mother_sequences, rho=0.5):
    """
    Vectorized generation of corrupt daughter sequences for multiple mother sequences.

    Parameters:
    mother_sequences (numpy.ndarray): A 2D NumPy array where each row represents a binary sequence.
    rho (float): The probability that a 1 in the mother sequence flips to 0 in the daughter sequence.

    Returns:
    numpy.ndarray: A 2D NumPy array where each row is the corrupt daughter sequence corresponding to the mother sequence.
    """
    # Generate random values for each element in the mother sequences
    random_flips = np.random.random(mother_sequences.shape)

    # Create a mask for flipping 1s to 0s based on the probability rho
    flip_mask = (mother_sequences == 1) & (random_flips <= rho)

    # Copy the mother sequences and apply the flip mask to generate daughters
    daughters = np.copy(mother_sequences)
    daughters[flip_mask] = 0

    return daughters

if __name__ == "__main__":
    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    all_sequences_file = 'all_sequences.pkl'
    
    if os.path.exists(all_sequences_file):
        all_sequences = pd.read_pickle(all_sequences_file)
        print(f"Loaded existing sequences from {all_sequences_file}")
    else:
        raw_data = pd.read_csv('../yeast_db1_renamed.csv')

        columns_to_keep = ['nuc_id', 'chr']
        modifications = ['H2AK5ac', 'H2AS129ph', 'H3K14ac', 'H3K18ac', 'H3K23ac', 'H3K27ac', 'H3K36me', 'H3K36me2', 'H3K36me3',
                        'H3K4ac', 'H3K4me', 'H3K4me2', 'H3K4me3', 'H3K56ac', 'H3K79me', 'H3K79me3', 'H3K9ac', 'H3S10ph',
                        'H4K12ac', 'H4K16ac', 'H4K20me', 'H4K5ac', 'H4K8ac', 'H4R3me', 'H4R3me2s', 'Htz1']
        modification_time_list = ['t0', 't4 ', 't8', 't15', 't30', 't60']
        # modification_time_list = ['t0', 't4']

        all_sequences_list = []

        for modification_time in modification_time_list:
            print(f"Processing modification time: {modification_time}")
            new_mods = [mod + '_' + modification_time for mod in modifications]
            if not os.path.exists(modification_time):
                os.makedirs(modification_time)

            for mod in new_mods:
                columns_to_keep.append(mod)

            data = raw_data[columns_to_keep].copy()
            data = data.iloc[1:]  # Drop the first row (time information)
            data.reset_index(drop=True, inplace=True)

            # Binarize the data
            for mod in new_mods:
                data[mod] = data[mod].apply(lambda x: 1 if x > 0 else 0)
            # reset index
            data = data.reset_index(drop=True)

            print(data.head())
            # replace any non breaking spaces with underscore in the entire dataframe
            data = data.replace('\xa0', '_', regex=True)

            data.to_csv(os.path.join(modification_time, f'yeast_db1_binarised_{modification_time}.csv'), index=False, encoding='utf-8')
            print(f"Binarised data saved to {os.path.join(modification_time, f'yeast_db1_binarised_{modification_time}.csv')}")

            # Create sequences of length 100
            n = 100
            for mod in new_mods:
                print(f'Time step: {modification_time}, Processing modification: {mod}')
                mod_data = data[[mod]]
                for index in range(len(mod_data) - n + 1):
                    current_sequence = mod_data[mod].iloc[index:index + n].tolist()
                    all_sequences_list.append({
                        'modification': mod,
                        'time_step': modification_time,
                        'nuc_id': data['nuc_id'].iloc[index],
                        'chr': data['chr'].iloc[index],
                        'mother_sequence': current_sequence
                    })

        # Convert collected sequences to DataFrame
        all_sequences = pd.DataFrame(all_sequences_list)

        # Save the sequences to a pkl file
        all_sequences.to_pickle('all_sequences_intermediate.pkl')
        print(all_sequences.head())

        # Ensure the column 'corrupt_daughter_sequence' exists in the DataFrame
        all_sequences['corrupt_daughter_sequence'] = None
        all_sequences['alpha'] = None
        all_sequences['beta'] = None

        # Prepare mother sequences as a NumPy array
        mother_sequences = np.array(all_sequences['mother_sequence'].tolist())

        # Vectorized calculation of alpha, beta, and corrupt daughter sequences
        print("Calculating alpha, beta, and generating corrupt daughter sequences...")
        alpha, beta = calculate_alpha_beta_vectorized(mother_sequences)
        corrupt_daughters = generate_daughters_vectorized(mother_sequences, rho=0.5)

        # Assign results back to the DataFrame
        all_sequences['alpha'] = alpha
        all_sequences['beta'] = beta
        all_sequences['corrupt_daughter_sequence'] = corrupt_daughters.tolist()

        # Save the final DataFrame
        all_sequences.to_pickle(all_sequences_file)
        print("All sequences with alpha, beta, and corrupt daughter sequences saved.")
    
    print(all_sequences.head())
    # just keep 1000 sequences for testing the code
    # all_sequences = all_sequences[:1000]
    
    # Divide into train and test sets
    train_sequences = all_sequences.sample(frac=0.8, random_state=42)
    test_sequences = all_sequences.drop(train_sequences.index)

    # Get the mother and corrupted daughter sequences for training
    # Convert to numpy arrays and then to PyTorch tensors
    mother_sequences = np.array(train_sequences['mother_sequence'].tolist())
    corrupted_daughter_sequences = np.array(train_sequences['corrupt_daughter_sequence'].tolist())

    print(f"Training set size: {len(train_sequences)}")
    print(f"Test set size: {len(test_sequences)}")

    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert numpy arrays to PyTorch tensors
    mother_sequences = torch.from_numpy(mother_sequences).float().to(device)
    corrupted_daughter_sequences = torch.from_numpy(corrupted_daughter_sequences).float().to(device)
    print(mother_sequences.shape)
    print(corrupted_daughter_sequences.shape)

    model = SequencePredictor_noanta().to(device)

    criterion = nn.MSELoss()

    learning_rate = 0.001
    num_epochs = 100
    batch_size = len(train_sequences)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create an empty 2D NumPy array to store the epoch and loss values
    training_loss = np.empty((num_epochs, 2))

    # Create a DataLoader for training
    train_dataset = TensorDataset(corrupted_daughter_sequences, mother_sequences)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    for epoch in range(num_epochs):
        for corrupted_batch, mother_batch in train_loader:
            # Forward pass
            outputs = model(corrupted_batch)
            loss = criterion(outputs, mother_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store the epoch and loss values directly in the data array
            training_loss[epoch, 0] = epoch + 1
            training_loss[epoch, 1] = loss.item()

        # Print the loss for every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    ###################################################################################################################
    # Now we have the  Network

    # Test the model.


    # Get the mother and corrupted daughter sequences for training
    # Convert to numpy arrays and then to PyTorch tensors
    mother_sequences = np.array(test_sequences['mother_sequence'].tolist())
    corrupted_daughter_sequences = np.array(test_sequences['corrupt_daughter_sequence'].tolist())

    # Convert numpy arrays to PyTorch tensors and move to device
    mother_sequences = torch.from_numpy(mother_sequences).float().to(device)
    corrupted_daughter_sequences = torch.from_numpy(corrupted_daughter_sequences).float().to(device)

    # Print tensor device info for debugging (test data)
    print(f"mother_sequences (test) device: {mother_sequences.device}")
    print(f"corrupted_daughter_sequences (test) device: {corrupted_daughter_sequences.device}")

    # Test the models
    model.eval()

    with torch.no_grad():
        test_output = model(corrupted_daughter_sequences)

    # Move predictions back to CPU for numpy conversion
    model_predicted_sequences = test_output.round().cpu().numpy().astype(int)
    corrected_daughter_sequences = []
    for i, seq in enumerate(model_predicted_sequences):
        corrected_daughter_sequences.append(seq)

    test_sequences["corrected_daughter_sequence"] = corrected_daughter_sequences


    # Calculate biterrors
    mother_sequences_np = mother_sequences.cpu().numpy().astype(int)
    corrected_daughter_sequences_np = np.array(corrected_daughter_sequences).astype(int)

    biterror, biterror_1, biterror_0 = calc_errors(mother_sequences_np, corrected_daughter_sequences_np)
    test_sequences["biterror"] = biterror
    test_sequences["biterror_1"] = biterror_1
    test_sequences["biterror_0"] = biterror_0

    print(test_sequences.head())

    train_sequences.to_pickle('train_sequences.pkl')
    test_sequences.to_pickle('test_sequences_with_corrections.pkl')
    # test_sequences.to_csv('test_sequences_with_corrections.csv', index=False)

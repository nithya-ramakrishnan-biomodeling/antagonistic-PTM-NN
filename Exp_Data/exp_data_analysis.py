import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

############################################################################
# Need to run only once.

# raw_data = pd.read_csv('yeast_db1_utf8.csv')
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

# raw_data.to_csv('yeast_db1_renamed.csv', index=False, encoding='utf-8')
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

#########################################################################################################################################

raw_data = pd.read_csv('yeast_db1_renamed.csv')
# nn_type = 'Targeted' # one NN per alpha beta pair
nn_type = 'Compound' # one NN for all alpha beta pairs
print(f"Using NN type: {nn_type}")
# the Compound NN is not the same as the one trained for alpha beta and mu together.
# This one just uses alpha and beta and is a no antagonism model.

# The code is not optimized for both nn_types in the same run.
# It also does not have a way to create a new folder in case a different type of nn is used.
# So make sure to run the code separately for both nn_types. and move the results to different folders manually.

# columns to keep in the dataset
columns_to_keep = ['nuc_id','chr']
modifications = ['H2AK5ac','H2AS129ph','H3K14ac','H3K18ac','H3K23ac','H3K27ac','H3K36me','H3K36me2','H3K36me3','H3K4ac','H3K4me','H3K4me2','H3K4me3','H3K56ac','H3K79me','H3K79me3','H3K9ac','H3S10ph','H4K12ac','H4K16ac','H4K20me','H4K5ac','H4K8ac','H4R3me','H4R3me2s','Htz1']
modification_time_list = ['t0', 't4', 't8', 't15', 't30', 't60']
for modification_time in modification_time_list:
    print(f"Processing modification time: {modification_time}")
    new_mods = [mod + '_' + modification_time for mod in modifications]
    # make folder to store analysis for modification_time
    if not os.path.exists(modification_time):
        os.makedirs(modification_time)

    for mod in new_mods:
        columns_to_keep.append(mod)

    data = raw_data[columns_to_keep]

    # delete first row containing time information
    data = data.iloc[1:]

    # for each modification column, replace values <= 0 with 0 and values > 0 with 1
    for mod in new_mods:
        data[mod] = data[mod].apply(lambda x: 1 if x > 0 else 0)
    # reset index
    data = data.reset_index(drop=True)

    print(data.head())
    # replace any non breaking spaces with underscore in the entire dataframe
    data = data.replace('\xa0', '_', regex=True)

    data.to_csv(os.path.join(modification_time, f'yeast_db1_binarised_{modification_time}.csv'), index=False, encoding='utf-8')
    print(f"Binarised data saved to {os.path.join(modification_time, f'yeast_db1_binarised_{modification_time}.csv')}")

    #########################################################################################################################################


    # create sequences of length 100 for each modification
    for mod in new_mods:
        # create a folder for each modification
        os.makedirs(os.path.join(modification_time, mod), exist_ok=True)
        # create a file with sequences of length n for a given modification.
        # If the sequence is shorter than n, skip it.
        n=100
        mod_data = data[[mod]]
        sequences = []
        current_sequence = []
        for index, row in mod_data.iterrows():
            current_sequence = mod_data[mod][index:index+n].tolist()
            if len(current_sequence) < n:
                continue
            sequences.append(current_sequence)
        sequences_df = pd.DataFrame(sequences)
        sequences_df.to_csv(f'{modification_time}/{mod}/{mod}_sequences_n{n}.csv', index=False, header=False)
        print(f"Sequences of length {n} for modification {mod} saved to {modification_time}/{mod}/{mod}_sequences_n{n}.csv")
        
        # create a new file with alpha and beta values for each sequence in the modification sequences file
        alpha_beta_values = []
        for index, row in sequences_df.iterrows():
            binary_sequence = ''.join(map(str, row.tolist()))
            alpha, beta = calculate_alpha_beta(binary_sequence)
            alpha_beta_values.append((alpha, beta))
        alpha_beta_df = pd.DataFrame(alpha_beta_values, columns=['alpha', 'beta'])
        alpha_beta_df.to_csv(f'{modification_time}/{mod}/{mod}_alpha_beta_n{n}.csv', index=False)
        print(f"Alpha and beta values for modification {mod} saved to {modification_time}/{mod}/{mod}_alpha_beta_n{n}.csv")
        
        daughter_sequences = []
        for index, row in sequences_df.iterrows():
            mom = row.values
            daughter = generate_daughter(mom, rho=0.5)
            daughter_sequences.append(daughter)
        daughter_df = pd.DataFrame(daughter_sequences)
        daughter_df.to_csv(f'{modification_time}/{mod}/{mod}_corrupt_daughter_sequences_n100.csv', index=False, header=False)


        # filter data for each alpha beta pairs with a boundary of 0.02
        boundary = 0.02
        mean_biterror_df = pd.DataFrame(columns=['Alpha', 'Beta', 'BitError'])
        for a in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            for b in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                print()
                # create a folder for each alpha beta pair
                filtered_sequences = sequences_df[alpha_beta_df['alpha'].between(a-boundary, a+boundary) & alpha_beta_df['beta'].between(b-boundary, b+boundary)]
                filtered_corrupt = daughter_df[alpha_beta_df['alpha'].between(a-boundary, a+boundary) & alpha_beta_df['beta'].between(b-boundary, b+boundary)]
                # skip if the data is empty
                if filtered_sequences.empty:
                    print(f"No sequences found for alpha {a} and beta {b}. Skipping correction for this pair.")
                    continue

                print(f'Number of sequences for alpha {a}, beta {b} with {a-boundary}<= alpha <={a+boundary} and {b-boundary}<= beta <={b+boundary}: {filtered_sequences.shape[0]}')
                print(f'Number of corrupt sequences for alpha {a}, beta {b} with {a-boundary}<= alpha <={a+boundary} and {b-boundary}<= beta <={b+boundary}: {filtered_corrupt.shape[0]}')

                # if filtered_sequences.shape[0] < 300:
                #     print(f"Not enough sequences ({filtered_sequences.shape[0]}) for alpha {a} and beta {b}. Skipping correction for this pair.")
                #     continue

                os.makedirs(f'{modification_time}/{mod}/a_{a}_b_{b}', exist_ok=True)
                filtered_sequences.to_csv(f'{modification_time}/{mod}/a_{a}_b_{b}/{mod}_filtered_a{str(a).replace(".", "_")}_b{str(b).replace(".", "_")}_sequences_n100.csv', index=False, header=False)
                filtered_corrupt.to_csv(f'{modification_time}/{mod}/a_{a}_b_{b}/{mod}_filtered_a{str(a).replace(".", "_")}_b{str(b).replace(".", "_")}_corrupt_daughter_sequences_n100.csv', index=False, header=False)

                #######################################################################################################################################################################

                # Detect device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Using device: {device}")

                if nn_type == 'Targeted':
                    model = SequencePredictor_noanta()
                    try:
                        model.load_state_dict(torch.load(f'pathfiles/Targeted_MA_DA_{a}_MSE_A{str(a).replace(".", "_")}_B{str(b).replace(".", "_")}/network_weights.pth', map_location=device))
                    except FileNotFoundError:
                        print(f"Model weights for alpha {a} and beta {b} not found. Skipping correction for this pair.")
                        continue
                elif nn_type == 'Compound':
                    model = SequencePredictor_noanta()
                    try:
                        model.load_state_dict(torch.load(f'pathfiles/compound_network_weights.pth', map_location=device))
                    except FileNotFoundError:
                        print(f"Compound model weights not found. Skipping correction for this pair.")
                        continue
                else:
                    print(f"Invalid nn_type: {nn_type}. Skipping correction for this pair.")
                    continue

                # If we reach this point then we have filtered data as well as the model to use.
                # use the filtered sequences and corrupt sequences for correction
                mother_sequences = filtered_sequences.reset_index(drop=True)
                daughter_sequences = filtered_corrupt.reset_index(drop=True)
                corrected_daughters = np.copy(daughter_sequences)
                for index, row in daughter_sequences.iterrows():
                    print(f'Processing sequence {index+1}/{daughter_sequences.shape[0]}')
                    corrupt_daughter = row.values
                    corrected_daughter = decode_with_noanta_nn(corrupt_daughter, model)
                    corrected_daughters[index] = corrected_daughter.cpu().numpy()
                corrected_daughters = pd.DataFrame(corrected_daughters)
                corrected_daughters.to_csv(f'{modification_time}/{mod}/a_{a}_b_{b}/{mod}_filtered_a{str(a).replace(".", "_")}_b{str(b).replace(".", "_")}_corrected_daughter_sequences_n100.csv', index=False, header=False)
                print(f"Corrected daughter sequences for alpha {a} and beta {b} saved to {modification_time}/{mod}/a_{a}_b_{b}/{mod}_filtered_a{str(a).replace(".", "_")}_b{str(b).replace(".", "_")}_corrected_daughter_sequences_n100.csv")
                #######################################################################################################################################################################
                # calculate biterror between the mom sequences and corrected daughter sequences
                biterror, biterror_1, biterror_0 = calc_errors(mother_sequences.values, corrected_daughters.values)
                error_df = pd.DataFrame({
                        'biterror': biterror,
                        'biterror_1': biterror_1,
                        'biterror_0': biterror_0
                    })
                error_df.to_csv(f'{modification_time}/{mod}/a_{a}_b_{b}/{mod}_filtered_a{str(a).replace(".", "_")}_b{str(b).replace(".", "_")}_biterror_n100.csv', index=False)
                mean_biterror = np.mean(biterror)
                mean_biterror_df = pd.concat([mean_biterror_df, pd.DataFrame([{'Alpha': a, 'Beta': b, 'BitError': mean_biterror}])], ignore_index=True)
                plt.figure(figsize=(12, 4))
                plt.hist(error_df['biterror'], bins=30, alpha=0.7, color='blue')
                plt.title('Bit Error')
                plt.savefig(f'{modification_time}/{mod}/a_{a}_b_{b}/mean_{mean_biterror}_{mod}_filtered_a{str(a).replace(".", "_")}_b{str(b).replace(".", "_")}_biterror_histograms.png')
                plt.close()

        mean_biterror_df.to_csv(f'{modification_time}/{mod}/{mod}_BitError_Combinations.csv', index=False)
    

    #########################################################################################################################################


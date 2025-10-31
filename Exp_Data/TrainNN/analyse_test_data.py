import pandas as pd
import os

# Load test data
test_sequences_with_corrections = pd.read_pickle('test_sequences_with_corrections.pkl')

# round alpha and beta to 1 decimal place
test_sequences_with_corrections['alpha'] = test_sequences_with_corrections['alpha'].round(1)
test_sequences_with_corrections['beta'] = test_sequences_with_corrections['beta'].round(1)

# # print unique values of alpha and beta
# print("Unique alpha values:", test_sequences_with_corrections['alpha'].unique())
# print("Unique beta values:", test_sequences_with_corrections['beta'].unique())

# # print the number of unique (alpha, beta) pairs and their counts
# unique_pairs = test_sequences_with_corrections.groupby(['alpha', 'beta']).size().reset_index(name='counts')
# print("Unique (alpha, beta) pairs and their counts:")
# print(unique_pairs)

# Identify unique (alpha, beta) pairs and calculate mean biterror for each pair
mean_biterror_df = test_sequences_with_corrections.groupby(['alpha', 'beta'])['biterror'].mean().reset_index(name='mean_biterror')

# Rename columns for clarity
mean_biterror_df.rename(columns={'alpha': 'Alpha', 'beta': 'Beta', 'mean_biterror': 'BitError'}, inplace=True)
print(mean_biterror_df)

# Save the mean biterror DataFrame to a CSV file
mean_biterror_df.to_csv('Test_Data_BitError_Combinations.csv', index=False)

# Run the incomplete batch matrix generator using os
os.system('python ../../batch_matrix_incomplete.py Test_Data')
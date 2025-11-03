import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

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

#alpha beta pairs to plot
# alpha_beta_pairs = []
# for a in [0.6,0.7,0.8]:
#     for b in [0.6,0.7,0.8]:
#         if not (a == 0.6 and b == 0.6):  # Exclude the (0.6, 0.6) pair
#             alpha_beta_pairs.append((a, b))

alpha_beta_pairs = [(0.6, 0.7), (0.6, 0.8), (0.7, 0.6), (0.7, 0.7), (0.7, 0.8)]

# Filter the data for the selected alpha-beta pairs
filtered_data = test_sequences_with_corrections[
    test_sequences_with_corrections[['alpha', 'beta']].apply(tuple, axis=1).isin(alpha_beta_pairs)
]

# Create a new column for the alpha-beta pair as a string
filtered_data['Alpha-Beta Pair'] = filtered_data.apply(lambda row: f"({row['alpha']}, {row['beta']})", axis=1)

# Plot violin plot with alpha-beta pair as x-axis
plt.figure(figsize=(10, 6))
sns.violinplot(x='Alpha-Beta Pair', y='biterror', data=filtered_data, palette='muted')
plt.title('BitError Distribution for Selected Alpha-Beta Pairs')
plt.ylabel('BitError')
plt.xlabel('Alpha-Beta Pair')
plt.tight_layout()
plt.savefig('BitError_Violin_AlphaBeta_Pairs.png')
plt.close()
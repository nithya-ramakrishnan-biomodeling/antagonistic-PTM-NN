import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set matplotlib rcParams for global styling
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.size'] = 22
# mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2


def plot_and_save_data(folder_path):
    # Define the data types we are dealing with
    data_types = ['alpha', 'beta', 'mu']
    data_latex_types = ['$\\alpha$', '$\\beta$', '$\\mu$']
    data = {}

    # Read and store data for each type
    for data_type in data_types:
        actual_path = os.path.join(folder_path, f'{data_type}_list.csv')
        predicted_path = os.path.join(folder_path, f'{data_type}_predicted_list.csv')
        error_path = os.path.join(folder_path, f'{data_type}_error_list.csv')

        actual = pd.read_csv(actual_path).squeeze()
        predicted = pd.read_csv(predicted_path).squeeze()
        error = actual - predicted

        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        correlation = np.corrcoef(actual, predicted)[0, 1]

        data[data_type] = {
            'actual': actual,
            'predicted': predicted,
            'error': error,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Correlation': correlation
        }

        print('###############################')
        print(f'{data_type}')
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R-squared: {r2}")
        print(f"Correlation: {correlation}")

    # Plot and save data for each type
    for i, data_type in enumerate(data_types):
        num_points = 100
        actual = data[data_type]['actual']
        predicted = data[data_type]['predicted']

        # Sample data if num_points is specified
        if num_points is not None and len(actual) > num_points:
            sample_indices = np.linspace(0, len(actual) - 1, num=num_points, dtype=int)
            sample_actual = actual.iloc[sample_indices].reset_index(drop=True)
            sample_predicted = predicted.iloc[sample_indices].reset_index(drop=True)
            sample_indices = range(len(sample_actual))  # Reset indices for plotting
        else:
            sample_indices = range(len(actual))
            sample_actual = actual
            sample_predicted = predicted

        # Bar plot for sampled data
        plt.figure(figsize=(7.5, 5.5), dpi=300)
        plt.bar(sample_indices, sample_actual, width=0.7, label=f'{data_latex_types[i]} Actual',
                alpha=0.7)  # Increased width
        plt.bar(sample_indices, -sample_predicted, width=0.7, label=f'(-ve) {data_latex_types[i]} Predicted',
                alpha=0.7)  # Increased width
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        for fmt in ['png', 'tiff', 'eps']:
            plt.savefig(os.path.join(folder_path, f'{data_type}_comparison_plot.{fmt}'), dpi=300, format=fmt)
        plt.close()

        # Scatter plot for actual vs predicted
        plt.figure(figsize=(7.5, 5.5), dpi=300)
        plt.scatter(data[data_type]['actual'], data[data_type]['predicted'], alpha=0.3,
                    label=f"r : {data[data_type]['Correlation'].round(4)}")
        plt.plot([data[data_type]['actual'].min(), data[data_type]['actual'].max()],
                 [data[data_type]['actual'].min(), data[data_type]['actual'].max()],
                 'r--')  # Line of perfect prediction
        # plt.title(f'Actual vs. Predicted for {data_latex_types[i]}')
        plt.xlabel(f'Actual {data_latex_types[i]} Values')
        plt.ylabel(f'Predicted {data_latex_types[i]} Values')
        plt.legend()
        plt.grid(True)
        plt.subplots_adjust(left=0.15, bottom=0.15)  # Adjust the left margin (In case we use larger fonts.)
        for fmt in ['png', 'tiff', 'eps']:
            plt.savefig(os.path.join(folder_path, f'{data_type}_scatter.{fmt}'), dpi=300, format=fmt)
        plt.close()

        # Histogram of errors
        plt.figure(figsize=(7.5, 5.5), dpi=300)
        plt.hist(data[data_type]['error'], bins=20, color='blue', alpha=0.7)
        # plt.title(f'Error Distribution for {data_latex_types[i]}')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        for fmt in ['png', 'tiff', 'eps']:
            plt.savefig(os.path.join(folder_path, f'{data_type}_error_hist.{fmt}'), dpi=300, format=fmt)
        plt.close()

# Usage example
batch_name = 'Results_and_Outputs/Prediction_Tests/Prediction_Test'
# batch_name = 'Prediction_Tests/Prediction_Test_Extended'
plot_and_save_data(batch_name)

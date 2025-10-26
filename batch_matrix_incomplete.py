"""
This is the batch matrix generator for incomplet matrices
This is used for data where the experimental data is incomplete
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse


# use the following command to run this.
# python3 heatmap_plotter.py --batch_name batch_name

def plot_heatmap(batch_name, annotate=False, grid=False, common_scale=False, extra='_', show_plt=False):
    # Read the data from the file
    data = pd.read_csv(f'{batch_name}_BitError_Combinations.csv')

    # the data is of the form alpha, beta, BitError
    # we expect alpha beta to be in the range 0.1 to 0.9 with a step of 0.1
    # however some combinations may be missing
    # we need to create a complete data with NaN for missing combinations

    # Create a complete matrix with NaN for missing combinations
    alpha_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    beta_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    for a in alpha_values:
        for b in beta_values:
            print(a, b)
            if not ((data['Alpha'] == a) & (data['Beta'] == b)).any():
                # If the combination is missing, add a row with NaN BitError
                data = pd.concat([data, pd.DataFrame([{'Alpha': a, 'Beta': b, 'BitError': np.nan}])], ignore_index=True)

    print(data)
    # Reshape the data into a matrix using pivot_table
    matrix = data.pivot_table(values='BitError', index='Beta', columns='Alpha')
    
    # ensure that the pivot table has all alpha and beta values with nan for missing combinations
    matrix = matrix.reindex(index=beta_values, columns=alpha_values)

    print(matrix)
    # Save the matrix to a new CSV file
    matrix.to_csv(f'{batch_name}_BitError_matrix.csv', index_label=[r'Beta\Alpha'])

    # Set font globally to Arial
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['font.size'] = 22
    # rcParams['font.weight'] = 'bold'
    rcParams['xtick.major.width'] = 2
    rcParams['ytick.major.width'] = 2

    img_dpi = 300

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=img_dpi)
    # Generate a meshgrid for the Alpha and Beta values to properly align with pcolormesh.
    # Add 0.5 to the range to shift the ticks to the center of the cells.
    X, Y = np.meshgrid(np.arange(matrix.shape[1] + 1), np.arange(matrix.shape[0] + 1))

    if common_scale:
        heatmap = ax.pcolormesh(X, Y, matrix.values, cmap="viridis", vmin=0, vmax=0.35)  # Set common color scale.
    else:
        heatmap = ax.pcolormesh(X, Y, matrix.values, cmap="viridis")

    if annotate:
        # Calculate the centers of the cells
        x_centers = (X[:-1, :-1] + X[1:, 1:]) / 2
        y_centers = (Y[:-1, :-1] + Y[1:, 1:]) / 2
        # Annotate each cell
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = f'{matrix.values[i, j]:.2f}'
                # ax.text(x_centers[i, j], y_centers[i, j], text, ha='center', va='center', color='k')
                # If we want to custom control the fontsize of the text for annotations.
                ax.text(x_centers[i, j], y_centers[i, j], text, ha='center', va='center', color='k', fontsize=12)

    # Customize the heatmap
    # plt.colorbar(heatmap, label='BER')
    plt.colorbar(heatmap, label='')
    # plt.subplots_adjust(left=0.15, right=0.99, bottom=0.1, top=0.99)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    # plt.title('BitError Heatmap')
    if grid:
        ax.grid(visible=True, linestyle='--', alpha=0.2)

    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    ax.set_xticklabels(matrix.columns, rotation=90)
    ax.set_yticklabels(matrix.index)
    ax.set_aspect('equal')  # Ensure the aspect ratio is equal to maintain square cells.
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin to accommodate x-axis labels

    # Save the figure in multiple formats
    for fmt in ['png', 'eps']:
        plt.savefig(f'{batch_name}{extra}heatmap.{fmt}', dpi=img_dpi, format=fmt)
        print(f'Saved heatmap as {batch_name}{extra}heatmap.{fmt}')

    # Display the plot
    if show_plt:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a file and plot a heatmap.')
    parser.add_argument('batch_name', type=str, help='Path to the file with content in the specified format')
    args = parser.parse_args()
    # plot_heatmap(args.batch_name)
    # plot_heatmap(args.batch_name, common_scale=True, extra='_commonscale_')
    # plot_heatmap(args.batch_name, annotate=True, extra='_annotated_')
    plot_heatmap(args.batch_name, annotate=True, common_scale=True, extra='_annotated_commonscale_')

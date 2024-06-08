import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse

parser = argparse.ArgumentParser(description='Generate Hinton plot for the difference of two CSV files.')
parser.add_argument('file1', type=str, help='Path to the first CSV file')
parser.add_argument('file2', type=str, help='Path to the second CSV file')
parser.add_argument('f1name', type=str, help='Name of the first file')
parser.add_argument('f2name', type=str, help='Name of the second file')
# parser.add_argument('name', type=str, help='Name to be used for saving the plot')
args = parser.parse_args()

df1 = pd.read_csv(args.file1)
df2 = pd.read_csv(args.file2)

# Create pivot tables for each file
pivot_table1 = df1.pivot(index='Alpha', columns='Beta', values='BitError')
pivot_table2 = df2.pivot(index='Alpha', columns='Beta', values='BitError')

# Calculate the difference between the two matrices
diff_matrix = pivot_table2 - pivot_table1

# Setup matplotlib to use LaTeX for text rendering and Arial font
# rcParams['text.usetex'] = True
# Set font globally to Arial
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 22
# rcParams['font.weight'] = 'bold'
rcParams['xtick.major.width'] = 2
rcParams['ytick.major.width'] = 2


def hinton(matrix, plot_name='Hinton Plot', file2='Second_Input', max_weight=None, ax=None, x_labels=None,
           y_labels=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    plt.figure(figsize=(7.5, 5.5), dpi=300)
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        no_weight_passed = True
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))
    else:
        no_weight_passed = False

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    matrix = np.fliplr(matrix)
    for (x, y), w in np.ndenumerate(matrix):
        print(x, y, w)
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

    # Add text annotations for maximum and minimum values on the side of the plot
    max_value = np.max(matrix)
    min_value = np.min(matrix)
    if no_weight_passed:
        full_square_value = max(np.abs(max_value), np.abs(min_value))
    else:
        full_square_value = max_weight
    # Add the text annotations with center alignment
    # ax.text(matrix.shape[0] + 1.5, 0.5, 'Full Square', fontsize=10, ha='center', va='center')
    # ax.text(matrix.shape[0] + 1.5, 1.0, f'|{full_square_value:.4f}|', fontsize=10, ha='center', va='center')
    # ax.text(matrix.shape[0] * 0.2, 1.2 * matrix.shape[1], f'Black shows {file2} is better', fontsize=10, ha='center', va='center')
    print(f'Full Square |{full_square_value:.4f}|')
    print(f'Black shows {file2} is better')
    # plt.title(f'{plot_name}')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')

    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels[::-1])

    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin to accommodate x-axis labels

    # Add the text annotation centered below the plot
    # ax.annotate(f'Black shows {file2} has a lower BER', xy=(0.5, -0.2), xycoords='axes fraction', fontsize=12,
    #             ha='center', va='center')


if __name__ == '__main__':
    # Define labels for x (alpha) and y (beta) axes based on the pivot table indices
    x_labels = pivot_table2.index
    y_labels = pivot_table2.columns

    print(diff_matrix)

    # Plot the difference matrix using the Hinton function with labels

    plot_name = f'{args.f2name} - {args.f1name}'
    hinton(diff_matrix.to_numpy(), plot_name=plot_name, file2=args.f2name, x_labels=x_labels, y_labels=y_labels)
    # hinton(diff_matrix.to_numpy(), plot_name=plot_name, max_weight=0.1, file2=args.f2name, x_labels=x_labels, y_labels=y_labels)
    # Save in all requested formats
    for fmt in ['png', 'tiff', 'eps']:
        plt.savefig(f'{plot_name.replace(" - ", "_vs_")}_hinton_plot.{fmt}', format=fmt, dpi=300)
    plt.close()
    # plt.show()

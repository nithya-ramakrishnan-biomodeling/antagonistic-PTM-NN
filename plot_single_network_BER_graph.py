"""
Enhanced Script to Plot BER vs. mu for Various Alpha-Beta Combinations

This script reads multiple CSV files from a series of directories named mu_0_0 to mu_0_9,
represents decimal values like mu_0_0 as 0.0. It generates plots of BER as a function of 'mu'
for each unique alpha-beta combination, saves these plots in multiple formats, and allows
for selected alpha-beta combinations to be plotted on a single comprehensive plot.

Requirements:
    - pandas (for data manipulation)
    - matplotlib (for plotting)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import plotly.graph_objects as go

# Set global matplotlib font properties
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.size'] = 20
# rcParams['font.weight'] = 'bold'
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
img_dpi = 300


def read_and_aggregate_data(base_dir):
    """
    Reads CSV files from directories and aggregates them into a single DataFrame with decimal mu.

    Parameters:
        base_dir (str): The base directory where the mu_0_X folders are located.

    Returns:
        DataFrame: A pandas DataFrame containing all the aggregated data with a decimal 'mu' column.
    """
    all_data = []
    for i in range(10):
        folder_name = f"mu_0_{i}"
        file_name = f"Single_NN_Test_mu_0_{i}_BitError_Combinations.csv"
        file_path = os.path.join(base_dir, folder_name, file_name)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['mu'] = i / 10.0  # Convert index to decimal mu value
            all_data.append(df)

    return pd.concat(all_data)


def save_plots(group, alpha, beta, base_dir):
    """
    Saves plots in multiple formats.

    Parameters:
        group (DataFrame): Data for specific alpha and beta.
        alpha (float): Alpha value.
        beta (float): Beta value.
        base_dir (str): Base directory for saving plots.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=img_dpi)
    ax.plot(group['mu'].values, group['BitError'].values, marker='o', linestyle='-')
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$BER$')
    ax.grid(True)

    plot_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    file_name_template = os.path.join(plot_dir, f'BER_Alpha_{alpha}_Beta_{beta}')

    for fmt in ['png', 'tiff', 'eps']:
        plt.savefig(f"{file_name_template}.{fmt}", dpi=img_dpi, format=fmt)
    plt.close()


def plot_all_curves_on_single_plot(data, base_dir, selected_combinations):
    """
    Plots selected alpha-beta combinations on a single plot.

    Parameters:
        data (DataFrame): The aggregated DataFrame.
        base_dir (str): Base directory for saving the combined plot.
        selected_combinations (list of tuples): List of (alpha, beta) tuples to plot.
    """
    plt.figure(figsize=(7.5, 5.5), dpi=img_dpi)
    for alpha, beta in selected_combinations:
        subset = data[(data['Alpha'] == alpha) & (data['Beta'] == beta)]
        plt.plot(subset['mu'].values, subset['BitError'].values, marker='o',
                 label=f'$\\alpha$={alpha}, $\\beta$={beta}')

    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$BER$',labelpad=-4)
    plt.legend(bbox_to_anchor=(1.005, 1.05), loc='upper left', borderaxespad=0., fontsize=12)
    plt.subplots_adjust(right=0.75, bottom=0.15)  # Adjust the bottom to make room for text annotations

    plt.grid(True)

    plot_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    for fmt in ['png', 'tiff', 'eps']:
        plt.savefig(os.path.join(plot_dir, f'Combined_BER_Plot.{fmt}'), dpi=img_dpi, format=fmt)
    plt.close()


def plot_all_curves_on_single_plot_plotly(data, base_dir, selected_combinations):
    """
    Plots selected alpha-beta combinations on a single interactive plot using Plotly.

    Parameters:
        data (DataFrame): The aggregated DataFrame.
        base_dir (str): Base directory for saving the combined plot.
        selected_combinations (list of tuples): List of (alpha, beta) tuples to plot.
    """
    # Create a Plotly graph object figure
    fig = go.Figure()

    # Loop through each combination to plot
    for alpha, beta in selected_combinations:
        subset = data[(data['Alpha'] == alpha) & (data['Beta'] == beta)]

        # Add a trace for each combination
        fig.add_trace(go.Scatter(
            x=subset['mu'],
            y=subset['BitError'],
            mode='lines+markers',
            name=f'Alpha={alpha}, Beta={beta}'
        ))

    # Update layout with titles and axis labels
    fig.update_layout(
        title='Bit Error Rate (BER) vs. μ',
        xaxis_title=r'$\mu$',
        yaxis_title='BER',
        legend_title='Alpha, Beta Combinations',
        template="plotly_white"
    )

    # Create plot directory if it doesn't exist
    plot_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Save the plot as an HTML file
    html_file_path = os.path.join(plot_dir, 'Combined_BER_Plot.html')
    fig.write_html(html_file_path)

    print(f"Plot saved to {html_file_path}")


def main():
    base_dir = 'Range_Single_Test'  # Specify the path to the base directory
    base_dir = 'Simple_Compound_Test'  # Specify the path to the base directory
    combined_data = read_and_aggregate_data(base_dir)

    # # Plot individual curves and save them
    # grouped = combined_data.groupby(['Alpha', 'Beta'])
    # for (alpha, beta), group in grouped:
    #     save_plots(group, alpha, beta, base_dir)

    # # Optional: plot selected combinations on a single plot
    # selected_combinations = [(0.1, 0.1), (0.2, 0.2)]  # Specify which combinations to plot together
    all_combinations = [(0.1, 0.1), (0.1, 0.2), (0.1, 0.3), (0.1, 0.4), (0.1, 0.5), (0.1, 0.6), (0.1, 0.7),
                        (0.1, 0.8), (0.1, 0.9), (0.2, 0.1), (0.2, 0.2), (0.2, 0.3), (0.2, 0.4), (0.2, 0.5),
                        (0.2, 0.6), (0.2, 0.7), (0.2, 0.8), (0.2, 0.9), (0.3, 0.1), (0.3, 0.2), (0.3, 0.3),
                        (0.3, 0.4), (0.3, 0.5), (0.3, 0.6), (0.3, 0.7), (0.3, 0.8), (0.3, 0.9), (0.4, 0.1),
                        (0.4, 0.2), (0.4, 0.3), (0.4, 0.4), (0.4, 0.5), (0.4, 0.6), (0.4, 0.7), (0.4, 0.8),
                        (0.4, 0.9), (0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.4), (0.5, 0.5), (0.5, 0.6),
                        (0.5, 0.7), (0.5, 0.8), (0.5, 0.9), (0.6, 0.1), (0.6, 0.2), (0.6, 0.3), (0.6, 0.4),
                        (0.6, 0.5), (0.6, 0.6), (0.6, 0.7), (0.6, 0.8), (0.6, 0.9), (0.7, 0.1), (0.7, 0.2),
                        (0.7, 0.3), (0.7, 0.4), (0.7, 0.5), (0.7, 0.6), (0.7, 0.7), (0.7, 0.8), (0.7, 0.9),
                        (0.8, 0.1), (0.8, 0.2), (0.8, 0.3), (0.8, 0.4), (0.8, 0.5), (0.8, 0.6), (0.8, 0.7),
                        (0.8, 0.8), (0.8, 0.9), (0.9, 0.1), (0.9, 0.2), (0.9, 0.3), (0.9, 0.4), (0.9, 0.5),
                        (0.9, 0.6), (0.9, 0.7), (0.9, 0.8), (0.9, 0.9)]
    high_combinations = [(0.6, 0.5), (0.6, 0.6), (0.6, 0.7),
                         (0.7, 0.5), (0.7, 0.6), (0.7, 0.7), (0.7, 0.9),
                         (0.8, 0.5), (0.8, 0.6), (0.8, 0.7), (0.8, 0.8), (0.8, 0.9),
                         (0.9, 0.5), (0.9, 0.6), (0.9, 0.7), (0.9, 0.8), (0.9, 0.9)]
    plot_all_curves_on_single_plot(combined_data, base_dir, high_combinations)

    plot_all_curves_on_single_plot_plotly(combined_data, base_dir, high_combinations)

    print("Plots have been generated and saved.")


if __name__ == "__main__":
    main()

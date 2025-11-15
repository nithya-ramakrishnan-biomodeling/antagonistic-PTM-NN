# Antagonistic PTM NN

## Overview
This repository contains code and resources for simulating and analyzing epigenetic sequence prediction using neural networks. The project focuses on predicting and analyzing the effects of antagonistic post-translational modifications (PTMs) on sequences.

## Modules and Scripts

### 1. Neural Network Decoding
These modules are used for sequence reconstruction and decoding:
- **`ESP_Sim.py`**: Predicts binary sequences using alpha and beta parameters.
- **`ESP_Sim_Antagonist_SimpleNN.py`**: Decodes sequences with antagonistic effects using alpha, beta, and mu parameters.
- **`ESP_Sim_Antagonist_Range_SimpleNN.py`**: Generalizes decoding over a range of alpha, beta, and mu values.
- **`Run_ESP.py`**: Runs simulations for sequence prediction over multiple alpha beta values.
- **`Run_ESP_range_Antagonist_Simple.py`**: Simulates range-based antagonistic sequence prediction.
- **`Run_Single_Network_Heatmap.py`**: Generates heatmaps for network performance.

### 2. Alpha, Beta, and Mu Prediction
These modules predict the parameters alpha, beta, and mu from sequences:
- **`alpha_beta_mu_predictor_NN.py`**: Predicts alpha, beta, and mu values from sequences.
- **`Run_alpha_beta_mu_predictor.py`**: Executes the alpha-beta-mu prediction pipeline.
- **`alpha_beta_predictor_verification.py`**: Creates plots to verigy the alpha, beta and mu.

### 3. Data Analysis
Scripts for analyzing and visualizing experimental data:
- **`exp_data_analysis.py`**: Processes experimental data, calculates alpha and beta values, and generates daughter sequences.
- **`analyse_test_data.py`**: Uses the processed experimental data and analyzes it to generate bit error statistics.
- **`exp_data_seq_plots.ipynb`**: Creates sequence plots for specific alpha-beta pairs to verify how the NN predicts sequences.

### 4. Heatmap and Plot Generation
Scripts for visualizing results:
- **`batch_matrix_generator.py`**: Generates heatmaps for complete data matrices.
- **`batch_matrix_incomplete.py`**: Handles incomplete data matrices and fills missing combinations with NaN.
- **`hinton_plot_generator.py`**: Creates Hinton plots to compare network performance.
- **`plot_single_network_BER_graph.py`**: Plots Bit Error Rate (BER) graphs for single networks.
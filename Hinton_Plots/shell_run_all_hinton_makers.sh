# This script is to make all necessary hinton plots from the already run simulations
# syntax for running the hinton plot generator:
# python -u ../hinton_plot_generator file1 file2 fname1 fname2
# we perform file2 - file1 and get black squares which show file2 is doing better

# Hinton Plot Comparing the Non Antagonistic Vs Antagonistic Results for a Targeted Network
python -u ../hinton_plot_generator.py ../Targeted_MA_DA/Targeted_MA_DA_BitError_Combinations.csv ../Targeted_MA_DAB/Targeted_MA_DAB_BitError_Combinations.csv Targeted_Non_Antagonistic Targeted_Antagonistic_mu_05


import time


from Epigenetic_Sequence_Predictor.alpha_beta_mu_predictor_NN import run_sim
input_size = 200
epochs = 10000
batch_name = f'Prediction_Test_Simple'

sim_start = time.time()
sim_name, alpha_error, beta_error, mu_error = run_sim(sim_id=batch_name, seed=42,
                                                      alpha_min=0.0, alpha_max=1.0, beta_min=0.0, beta_max=1.0,
                                                      mu_min=0.0, mu_max=1.0, rho=0.5, n_samples=10000,
                                                      n_samples_test=3000, seq_length=100, input_size=input_size,
                                                      hidden_size=int(1.5 * 100),
                                                      num_layers=5, learning_rate=0.001, num_epochs=epochs,
                                                      batch_size=10000, loss_function_type='MSE',
                                                      verbose_level=1)

# Do Not uncomment while running on a server as the server doesn't have matplotlib.
# plot_ensemble_avg(sim_name)
# plot_training_loss(sim_name)
sim_end = time.time()
print(f"Simulation Name = {sim_name}", end=' --- ')
print(f"Sim Execution Time = {round(sim_end - sim_start, 4)}", end=' --- ')
print(f'Average Alpha Error = {alpha_error}')
print(f'Average Beta Error = {beta_error}')
print(f'Average Mu Error = {mu_error}')

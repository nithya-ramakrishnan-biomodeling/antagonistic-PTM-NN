import time
from Epigenetic_Sequence_Predictor.ESP_Sim_No_Antagonism_Range import run_sim

batch_name = 'Range_Test_No_Anta'

sim_start = time.time()
sim_name, initial_biterror = run_sim(sim_id=batch_name, seed=42,
                                     alpha_min=0.0, alpha_max=1.0,
                                     beta_min=0.0, beta_max=1.0,
                                     rho=0.5,
                                     n_samples=100000, n_samples_test=30000,
                                     seq_length=100,
                                     input_size=100, hidden_size=int(1.5 * 100), output_size=100,
                                     num_layers=5, learning_rate=0.001,
                                     num_epochs=20000, batch_size=100000,
                                     loss_function_type='MSE', verbose_level=5)

sim_end = time.time()
print(f"Simulation Name = {sim_name}", end=' --- ')
print(f"Sim Execution Time = {round(sim_end - sim_start, 4)}", end=' --- ')
print(f'Average BitError(Initial_Network) = {initial_biterror}')

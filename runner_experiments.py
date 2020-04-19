import os
import numpy as np

from Core.GradientDescent import exp1
from Core.Gradient_VS_Stochastic import exp2
from Core.SGD_epochs import exp4
from Core.SGD_tolerance import exp3
from Core.plot_results import plots_experiment_with_stoppingcriteria

samples = [100, 1000, 10000]
linear_coeff = np.array([2, -4])
my_seed = 110395
np.random.seed(my_seed)
perturbation = np.random.uniform(low=0.0, high=0.01)
init_point = np.array([0, 0])
toll = 0.01
lr = 0.0001
size = np.array([x / 100 * 10 for x in samples])
epochs = [100, 500, 1000]

name_final_data_folder = "FINAL_SUMMARY/"
os.makedirs(name_final_data_folder, exist_ok=False)
name_folders = ["Experiment1", "Experiment2", "Experiment3", "Experiment4"]
exp1(number_samples=samples, coefficients=linear_coeff, noise=perturbation, initial_point=init_point,
     tolerance=toll,
     learning_rate=lr, path_output_folder=name_folders[0], path_folder_plot=name_final_data_folder,
     seed=my_seed)

exp2(number_samples=samples, coefficients=linear_coeff, noise=perturbation, initial_point=init_point,
     tolerance=toll,
     learning_rate=lr, path_output_folder=name_folders[1], path_folder_plot=name_final_data_folder,
     seed=my_seed)

exp3(number_samples=samples, coefficients=linear_coeff, noise=perturbation, initial_point=init_point,
     tolerance=toll,
     learning_rate=lr, batch_size=size, path_output_folder=name_folders[2],
     path_folder_plot=name_final_data_folder, seed=my_seed)

exp4(number_samples=samples, coefficients=linear_coeff, noise=perturbation, initial_point=init_point,
     number_epochs=epochs, learning_rate=lr, batch_size=size, path_output_folder=name_folders[3],
     path_folder_plot=name_final_data_folder,
     seed=my_seed)

plots_experiment_with_stoppingcriteria(path_dataframe=name_final_data_folder + name_folders[0], n_samples=samples,
                                       output_image=name_final_data_folder + name_folders[0])

plots_experiment_with_stoppingcriteria(path_dataframe=name_final_data_folder + name_folders[1], n_samples=samples,
                                       output_image=name_final_data_folder + name_folders[1])

plots_experiment_with_stoppingcriteria(path_dataframe=name_final_data_folder + name_folders[2], n_samples=samples,
                                       output_image=name_final_data_folder + name_folders[2])

plots_experiment_with_stoppingcriteria(path_dataframe=name_final_data_folder + name_folders[3], n_samples=samples,
                                       output_image=name_final_data_folder + name_folders[3], stop='Epochs')

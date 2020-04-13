import os
import numpy as np

from Core.GradientDescent import exp1
from Core.Gradient_VS_Stochastic import exp2
from Core.SGD_tolerance import exp3
from Core.SGD_epochs import exp4

samples = [100, 500, 1000, 10000]
linear_coeff = np.array([2, -4])
my_seed = 110395
np.random.seed(my_seed)
perturbation = np.random.uniform(low=0.0, high=0.01)
init_point = np.array([0, 0])
toll = 0.001
lr = 0.0001

size = np.array([x / 100 * 10 for x in samples])
epochs = [100, 500, 1000, 2000]

exp1(number_samples=samples, coefficients=linear_coeff, noise=perturbation, initial_point=init_point, tolerance=toll,
     learning_rate=lr, path_output_folder="EXP_1", seed=my_seed)

exp2(number_samples=samples, coefficients=linear_coeff, noise=perturbation, initial_point=init_point, tolerance=toll,
     learning_rate=lr, path_output_folder="EXP_2", seed=my_seed)

exp3(number_samples=samples, coefficients=linear_coeff, noise=perturbation, initial_point=init_point, tolerance=toll,
     learning_rate=lr, batch_size=size, path_output_folder="EXP_3", seed=my_seed)

exp4(number_samples=samples, coefficients=linear_coeff, noise=perturbation, initial_point=init_point,
     number_epochs=epochs, learning_rate=lr, batch_size=size, path_output_folder="EXP_4", seed=my_seed)

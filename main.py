import numpy as np

from Core.algorithms import create_dataset as data
from Core.algorithms import Gradient_Descent as GD
from Core.algorithms import SGD_v1, SGD_v2, SGD_v3, SGD_v4

filename_dataset = "Dataset_2D"
line_coefficients = np.array([2, -4])
np.random.seed(1103)

fp = data(number_samples=100, omega=line_coefficients, noise=0.001, output_name=filename_dataset)

initial_point = np.array([0, 0])

"""
The function GD implements the idea of Gradient Descent
        loss function is computed over the entire dataset
"""
approximations_GD = GD(filepath_data=fp, omega_zero=initial_point, lr=0.0001, tolerance=0.001,
                       output_info="experimentGD")
print("APPROXIMATION GRADIENT DESCENT: \t", approximations_GD, "\n")

"""
The function SGD_v1 implements the idea of Stocastic Gradient Descent
        * one example per iteration + stopping criteria (norm of loss_gradient)
        loss function is computed over JUST ONE example taken randomly (with replacement) from the dataset
"""
approximations_SGDv1 = SGD_v1(filepath_data=fp, omega_zero=initial_point, lr=0.0001, tolerance=0.001,
                              output_info="experimentSGDV1")
print("APPROXIMATION STOCASTIC GRADIENT DESCENT Version 1: \t", approximations_SGDv1, "\n")

"""
The function SGD_v2 implements the idea of Stochastic Gradient Descent 
        * mini-batch + stopping criteria (norm of loss_gradient)
        loss function is computed over a random batch (with replacement) among all
        **************************************************************************
        *   Choose an exact dividend of the number of samples to obtain
            equally sized batches
        **************************************************************************
        *   A tolerance equal or largen than 10^-4 affect the convergence of 
            the algorithm (it seems that it's not converges at all)
        **************************************************************************
"""
approximations_SGDv2 = SGD_v2(filepath_data=fp, omega_zero=initial_point, lr=0.0001, tolerance=0.001, batch_size=10,
                              output_info="experimentSGDV2")
print("APPROXIMATION STOCASTIC GRADIENT DESCENT Version 2: \t", approximations_SGDv2, "\n")

"""
The functions SGD_v3 implements the idea of Stocastic Gradient Descent
        * one example per iteration + epochs
        loss is computed over one example at each iteration and when the algorithm 
        analyze all samples means that is finished an epoch
"""
approximations_SGDv3 = SGD_v3(filepath_data=fp, omega_zero=initial_point, lr=0.0001, epochs=100,
                              output_info="experimentSGDV3")
print("APPROXIMATION STOCASTIC GRADIENT DESCENT Version 3: \t", approximations_SGDv3, "\n")

"""
The function SGD_v4 implements the idea of Stocastic Gradient Descent
        * mini-batch + epochs
        loss is computed for each mini-batch and this is performed for each epoch
"""
approximations_SGDv4 = SGD_v4(filepath_data=fp, omega_zero=initial_point, lr=0.0001, epochs=100, batch_size=10,
                              output_info="experimentSGDV4")
print("APPROXIMATION STOCASTIC GRADIENT DESCENT Version 4: \t", approximations_SGDv4, "\n")

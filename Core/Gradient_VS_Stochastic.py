import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Core.algorithms import create_dataset as data
from Core.algorithms import Gradient_Descent
from Core.algorithms import SGD_v1


def exp2(number_samples, coefficients, noise, initial_point, tolerance, learning_rate, path_output_folder, seed=None):
    if seed is None:
        seed = 110395

    filename_dataset = "Dataset2D"
    path_output_folder += "/"
    os.makedirs(path_output_folder, exist_ok=False)
    path_output_folder_GD = path_output_folder + "GD/"
    path_output_folder_SGD = path_output_folder + "SGD/"
    os.makedirs(path_output_folder_GD, exist_ok=False)
    os.makedirs(path_output_folder_SGD, exist_ok=False)
    details_GD = {"number_samples": [], "alpha": [], "beta": [], "time": [], "number_iteration": []}
    details_SGD = {"number_samples": [], "alpha": [], "beta": [], "time": [], "number_iteration": []}
    for each_num in number_samples:
        np.random.seed(seed=seed)
        fp = data(number_samples=each_num, omega=coefficients, noise=noise,
                  output_name=path_output_folder + filename_dataset + "_" + str(each_num))
        np.random.seed(seed=seed)
        approximations_GD = Gradient_Descent(filepath_data=fp, omega_zero=initial_point, lr=learning_rate,
                                             tolerance=tolerance,
                                             output_info=path_output_folder_GD + "Report_GD" + "_" + str(each_num))
        np.random.seed(seed=seed)
        approximations_SGDv1 = SGD_v1(filepath_data=fp, omega_zero=initial_point, lr=learning_rate, tolerance=tolerance,
                                      output_info=path_output_folder_SGD + "Report_SGD" + "_" + str(each_num))
        details_GD["alpha"].append(approximations_GD[0])
        details_GD["beta"].append(approximations_GD[1])
        details_GD["time"].append(approximations_GD[2])
        details_GD["number_samples"].append(each_num)
        details_GD["number_iteration"].append(approximations_GD[3])
        details_SGD["alpha"].append(approximations_SGDv1[0])
        details_SGD["beta"].append(approximations_SGDv1[1])
        details_SGD["time"].append(approximations_SGDv1[2])
        details_SGD["number_samples"].append(each_num)
        details_SGD["number_iteration"].append(approximations_SGDv1[3])
    pd.DataFrame(details_GD).to_csv(path_output_folder + "GD.csv", index=True, header=True)
    pd.DataFrame(details_SGD).to_csv(path_output_folder + "SGD.csv", index=True, header=True)

    gd = pd.read_csv(path_output_folder + "GD.csv")
    sgd = pd.read_csv(path_output_folder + "SGD.csv")
    num = gd['number_samples'].values
    x = np.arange(len(num))
    w = 0.3
    fig, ax = plt.subplots()
    GD = plt.bar(x - w / 2, gd['time'].values, width=w - 0.05, label='Gradient Descent')
    SGD = plt.bar(x + w / 2, sgd['time'].values, width=w - 0.05, label='Stochastic Gradient Descent')
    plt.xticks(x, num)
    fig.tight_layout()
    ax.set_ylabel("Time [s]")
    ax.set_xlabel('Number of Samples')
    plt.title("Time required to converge - GD vs SGD")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)
    for i, v in enumerate(num):
        ax.text(x=i - w / 2 - 0.05, y=gd['time'].values[i], s=gd['time'].values[i],
                bbox=dict(facecolor='white', alpha=1),
                color='black')
        ax.text(x=i + w / 2 - 0.05, y=sgd['time'].values[i], s=sgd['time'].values[i],
                bbox=dict(facecolor='white', alpha=1),
                color='black')

    plt.savefig(path_output_folder + "compares_histogram_time.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    fig2, ax2 = plt.subplots()
    GD2 = plt.bar(x - w / 2, gd['number_iteration'].values, width=w - 0.05, label='Gradient Descent')
    SGD2 = plt.bar(x + w / 2, sgd['number_iteration'].values, width=w - 0.05, label='Stochastic Gradient Descent')
    plt.xticks(x, num)
    plt.xticks(x, num)
    fig2.tight_layout()
    ax2.set_ylabel("Number of Iterations")
    ax2.set_xlabel('Number of Samples')
    plt.title("Iteration required to converge - GD vs SGD")
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)
    for i, v in enumerate(num):
        ax2.text(x=i - w / 2 - 0.05, y=gd['number_iteration'].values[i], s=gd['number_iteration'].values[i],
                 bbox=dict(facecolor='white', alpha=1),
                 color='black')
        ax2.text(x=i + w / 2 - 0.05, y=sgd['number_iteration'].values[i], s=sgd['number_iteration'].values[i],
                 bbox=dict(facecolor='white', alpha=1),
                 color='black')

    plt.savefig(path_output_folder + "compares_histogram_iteration.png", bbox_inches="tight", dpi=200)
    plt.close(fig2)

    for x in number_samples:
        gd = pd.read_csv(path_output_folder_GD + "Report_GD_" + str(x) + ".csv")
        sgd = pd.read_csv(path_output_folder_SGD + "Report_SGD_" + str(x) + ".csv")
        fig, ax = plt.subplots(2, 2, figsize=(18, 12))
        plt.subplot(221)
        plt.title("Approximations Gradient Descent with " + str(x) + " samples")
        plt.plot(np.arange(0, np.shape(gd)[0]), gd['w0'].values, linestyle='solid', color='red', label='alpha')
        plt.plot(np.arange(0, np.shape(gd)[0]), gd['w1'].values, linestyle='solid', color='blue', label='beta')
        plt.xticks(np.arange(0, np.shape(gd)[0] + 1, step=np.shape(gd)[0] / 100 * 10))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)
        alphagd = plt.Circle(xy=(np.shape(gd)[0], gd.values[-2:-1, 1:2][0][0]), radius=0.1, color='red', lw=10)
        plt.gcf().gca().add_patch(alphagd)
        plt.text(x=np.shape(gd)[0] - (np.shape(gd)[0] / 100 * 5),
                 y=gd.values[-2:-1, 1:2][0][0] - (gd.values[-2:-1, 1:2][0][0] / 100 * 20),
                 s=str(gd.values[-2:-1, 1:2][0][0]),
                 fontsize=10,
                 bbox=dict(facecolor='white', alpha=1),
                 color='black')
        betagd = plt.Circle((np.shape(gd)[0], gd.values[-2:-1, 2:3][0][0]), radius=0.1, color='blue', lw=10)
        plt.gcf().gca().add_patch(betagd)
        plt.text(x=np.shape(gd)[0] - (np.shape(gd)[0] / 100 * 5),
                 y=gd.values[-2:-1, 2:3][0][0] + (gd.values[-2:-1, 1:2][0][0] / 100 * 15),
                 s=str(gd.values[-2:-1, 2:3][0][0]),
                 fontsize=10,
                 bbox=dict(facecolor='white', alpha=1),
                 color='black')
        plt.hlines(coefficients[0], 0, np.shape(gd)[0], colors='black', linestyles='solid',
                   label='Real alpha',
                   linewidth=3)
        plt.hlines(coefficients[1], 0, np.shape(gd)[0], colors='purple', linestyles='solid',
                   label='Real beta',
                   linewidth=3)
        plt.subplot(222)
        plt.title("Approximations Stochastic Gradient Descent with " + str(x) + " samples")
        plt.plot(np.arange(0, np.shape(sgd)[0]), sgd['w0'].values, linestyle='solid', color='red', label='alpha')
        plt.plot(np.arange(0, np.shape(sgd)[0]), sgd['w1'].values, linestyle='solid', color='blue', label='beta')
        plt.xticks(np.arange(0, np.shape(sgd)[0] + 1, step=np.shape(sgd)[0] / 100 * 10))
        alphasgd = plt.Circle(xy=(np.shape(sgd)[0], sgd.values[-2:-1, 1:2][0][0]), radius=0.1, color='red', lw=10)
        plt.gcf().gca().add_patch(alphasgd)
        plt.text(x=np.shape(sgd)[0] - (np.shape(sgd)[0] / 100 * 5),
                 y=sgd.values[-2:-1, 1:2][0][0] - (sgd.values[-2:-1, 1:2][0][0] / 100 * 20),
                 s=str(sgd.values[-2:-1, 1:2][0][0]),
                 fontsize=10,
                 bbox=dict(facecolor='white', alpha=1),
                 color='black')
        betasgd = plt.Circle((np.shape(sgd)[0], sgd.values[-2:-1, 2:3][0][0]), radius=0.1, color='blue', lw=10)
        plt.gcf().gca().add_patch(betasgd)
        plt.text(x=np.shape(sgd)[0] - (np.shape(sgd)[0] / 100 * 5),
                 y=sgd.values[-2:-1, 2:3][0][0] + (sgd.values[-2:-1, 1:2][0][0] / 100 * 15),
                 s=str(sgd.values[-2:-1, 2:3][0][0]),
                 fontsize=10,
                 bbox=dict(facecolor='white', alpha=1),
                 color='black')
        plt.hlines(coefficients[0], 0, np.shape(sgd)[0], colors='black', linestyles='solid',
                   label='Real alpha',
                   linewidth=3)
        plt.hlines(coefficients[1], 0, np.shape(sgd)[0], colors='purple', linestyles='solid',
                   label='Real beta',
                   linewidth=3)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)
        # -------
        plt.subplot(223)
        plt.title("Trend of Loss of GD with " + str(x) + " samples")
        plt.plot(np.arange(0, np.shape(gd)[0]), gd['loss'].values, linestyle='solid', color='red', label='loss')
        plt.xticks(np.arange(0, np.shape(gd)[0] + 1, step=np.shape(gd)[0] / 100 * 10))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=1)
        # --------
        plt.subplot(224)
        plt.title("Trend of Loss of SGD with " + str(x) + " samples")
        plt.plot(np.arange(0, np.shape(sgd)[0]), sgd['loss'].values, linestyle='solid', color='red', label='loss')
        plt.xticks(np.arange(0, np.shape(sgd)[0] + 1, step=np.shape(sgd)[0] / 100 * 10))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=1)
        # ------
        plt.tight_layout()
        plt.savefig(path_output_folder + "approximations_" + str(x) + ".png", bbox_inches="tight", dpi=200)
        plt.close(fig=fig)
        # -----
        fig2, ax2 = plt.subplots(1, 1, figsize=(18, 12))
        datapoints = np.load(path_output_folder + filename_dataset + "_" + str(x) + ".npy")
        plt.title("Distribution of " + str(x) + "Points")
        plt.scatter(datapoints[:, 0], datapoints[:, 1], marker=".", color='red', label='Perturbed Points')
        plt.scatter(datapoints[:, 0],
                    coefficients[0] * datapoints[:, 0] + coefficients[1] * np.ones(
                        shape=(np.shape(datapoints)[0]),
                        dtype=float), marker='*',
                    color='green',
                    label='Real Points')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)
        plt.tight_layout()
        plt.savefig(path_output_folder + "distribution_points_" + str(x) + ".png", bbox_inches="tight", dpi=200)
        plt.close(fig=fig2)

    fig3, ax3 = plt.subplots(1, 1, figsize=(14, 8))
    summary = pd.read_csv(path_output_folder + "GD.csv")
    x_axis = summary['number_samples'].values
    x = np.arange(len(x_axis))
    y_axis_a = np.absolute(np.ones(shape=(np.shape(summary)[0],), dtype=float) * coefficients[0]) - np.absolute(
        summary['alpha'].values)
    y_axis_b = np.absolute(np.ones(shape=(np.shape(summary)[0],), dtype=float) * coefficients[1]) - np.absolute(
        summary['beta'].values)
    dev_alpha = plt.bar(x - w / 2 - 0.05, y_axis_a,
                        width=w,
                        label='Alpha')
    dev_beta = plt.bar(x + w / 2 + 0.05, y_axis_b,
                       width=w,
                       label='Beta')
    plt.xticks(x, x_axis)
    ax3.set_xlabel('Number of Samples')
    ax3.set_ylabel("Error of approximations")
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=10)
    for i, v in enumerate(x_axis):
        ax3.text(x=i - w, y=y_axis_a[i], s=round(y_axis_a[i], 4), fontsize=7,
                 bbox=dict(facecolor='white', alpha=1),
                 color='black')
        ax3.text(x=i + w / 2, y=y_axis_b[i], s=round(y_axis_b[i], 4), fontsize=7,
                 bbox=dict(facecolor='white', alpha=1),
                 color='black')
    plt.title("Deviations from real coefficients alpha and beta - Gradient Descent")
    fig3.tight_layout()
    plt.savefig(path_output_folder + "error_of_approximations_GD.png")
    plt.close(fig=fig3)

    fig4, ax4 = plt.subplots(1, 1, figsize=(14, 8))
    summary = pd.read_csv(path_output_folder + "SGD.csv")
    x_axis = summary['number_samples'].values
    x = np.arange(len(x_axis))
    y_axis_a = np.absolute(np.ones(shape=(np.shape(summary)[0],), dtype=float) * coefficients[0]) - np.absolute(
        summary['alpha'].values)
    y_axis_b = np.absolute(np.ones(shape=(np.shape(summary)[0],), dtype=float) * coefficients[1]) - np.absolute(
        summary['beta'].values)
    dev_alpha = plt.bar(x - w / 2 - 0.05, y_axis_a,
                        width=w,
                        label='Alpha')
    dev_beta = plt.bar(x + w / 2 + 0.05, y_axis_b,
                       width=w,
                       label='Beta')
    plt.xticks(x, x_axis)
    ax4.set_xlabel('Number of Samples')
    ax4.set_ylabel("Error of approximations")
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=10)
    for i, v in enumerate(x_axis):
        ax4.text(x=i - w, y=y_axis_a[i], s=round(y_axis_a[i], 4), fontsize=7,
                 bbox=dict(facecolor='white', alpha=1),
                 color='black')
        ax4.text(x=i + w / 2, y=y_axis_b[i], s=round(y_axis_b[i], 4), fontsize=7,
                 bbox=dict(facecolor='white', alpha=1),
                 color='black')
    plt.title("Deviations from real coefficients alpha and beta - Stochastic Gradient Descent")
    fig4.tight_layout()
    plt.savefig(path_output_folder + "error_of_approximations_SGD.png")
    plt.close(fig=fig4)

    return

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Core.algorithms import create_dataset as data
from Core.algorithms import SGD_v1
from Core.algorithms import SGD_v2


def exp3(number_samples, coefficients, noise, initial_point, tolerance, learning_rate, batch_size, path_output_folder,
         seed=None):
    if seed is None:
        seed = 110395

    filename_dataset = "Dataset2D"
    path_output_folder += "/"
    os.makedirs(path_output_folder, exist_ok=False)
    path_output_folder_SGD_1 = path_output_folder + "GDBasic/"
    path_output_folder_SGD_2 = path_output_folder + "GDBatch/"
    os.makedirs(path_output_folder_SGD_1, exist_ok=False)
    os.makedirs(path_output_folder_SGD_2, exist_ok=False)

    details_SGDBasic = {"number_samples": [], "alpha": [], "beta": [], "time": [], "number_iteration": []}
    details_SGDBatch = {"number_samples": [], "alpha": [], "beta": [], "time": [], "number_iteration": []}

    for each_num, each_size in zip(number_samples, batch_size):
        np.random.seed(seed=seed)
        fp = data(number_samples=each_num, omega=coefficients, noise=noise,
                  output_name=path_output_folder + filename_dataset + "_" + str(each_num))
        np.random.seed(seed=seed)
        approximations_SGD_Basic = SGD_v1(filepath_data=fp, omega_zero=initial_point, lr=learning_rate,
                                          tolerance=tolerance,
                                          output_info=path_output_folder_SGD_1 + "Report_SGD_Basic" + "_" + str(
                                              each_num))
        np.random.seed(seed=seed)
        approximations_SGD_Batch = SGD_v2(filepath_data=fp, omega_zero=initial_point, lr=learning_rate,
                                          tolerance=tolerance,
                                          output_info=path_output_folder_SGD_2 + "Report_SGD_Batch" + "_" + str(
                                              each_num),
                                          batch_size=each_size)
        details_SGDBasic["alpha"].append(approximations_SGD_Basic[0])
        details_SGDBasic["beta"].append(approximations_SGD_Basic[1])
        details_SGDBasic["time"].append(approximations_SGD_Basic[2])
        details_SGDBasic["number_samples"].append(each_num)
        details_SGDBasic["number_iteration"].append(approximations_SGD_Basic[3])
        details_SGDBatch["alpha"].append(approximations_SGD_Batch[0])
        details_SGDBatch["beta"].append(approximations_SGD_Batch[1])
        details_SGDBatch["time"].append(approximations_SGD_Batch[2])
        details_SGDBatch["number_samples"].append(each_num)
        details_SGDBatch["number_iteration"].append(approximations_SGD_Batch[3])
    pd.DataFrame(details_SGDBasic).to_csv(path_output_folder + "SGD_Basic.csv", index=True, header=True)
    pd.DataFrame(details_SGDBatch).to_csv(path_output_folder + "SGD_Batch.csv", index=True, header=True)

    sgd_basic = pd.read_csv(path_output_folder + "SGD_Basic.csv")
    sgd_batch = pd.read_csv(path_output_folder + "SGD_Batch.csv")
    num = sgd_basic['number_samples'].values
    x = np.arange(len(num))
    w = 0.3
    fig, ax = plt.subplots()
    SGDBasic = plt.bar(x - w / 2, sgd_basic['time'].values, width=w - 0.05,
                       label='Stochastic Gradient Descent Basic')
    SGDBatch = plt.bar(x + w / 2, sgd_batch['time'].values, width=w - 0.05,
                       label='Stochastic Gradient Descent Batch')
    plt.xticks(x, num)
    fig.tight_layout()
    ax.set_ylabel("Time [s]")
    ax.set_xlabel('Number of Samples')
    plt.title("Time required to converge - Basic vs Batch")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)
    for i, v in enumerate(num):
        ax.text(x=i - w / 2 - 0.05, y=sgd_basic['time'].values[i], s=sgd_basic['time'].values[i],
                bbox=dict(facecolor='white', alpha=1),
                color='black')
    ax.text(x=i + w / 2 - 0.05, y=sgd_batch['time'].values[i], s=sgd_batch['time'].values[i],
            bbox=dict(facecolor='white', alpha=1),
            color='black')

    plt.savefig(path_output_folder + "compares_histogram_time.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    fig2, ax2 = plt.subplots()
    SGDBasic = plt.bar(x - w / 2, sgd_basic['number_iteration'].values, width=w - 0.05,
                       label='Stochastic Gradient Descent Basic')
    SGDBatch = plt.bar(x + w / 2, sgd_batch['number_iteration'].values, width=w - 0.05,
                       label='Stochastic Gradient Descent Batch')
    plt.xticks(x, num)
    fig2.tight_layout()
    ax2.set_ylabel("Number of Iterations")
    ax2.set_xlabel('Number of Samples')
    plt.title("Iteration required to converge - Basic vs Batch")
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)
    for i, v in enumerate(num):
        ax2.text(x=i - w / 2 - 0.05, y=sgd_basic['number_iteration'].values[i],
                 s=sgd_basic['number_iteration'].values[i],
                 bbox=dict(facecolor='white', alpha=1),
                 color='black')
    ax2.text(x=i + w / 2 - 0.05, y=sgd_batch['number_iteration'].values[i],
             s=sgd_batch['number_iteration'].values[i],
             bbox=dict(facecolor='white', alpha=1),
             color='black')

    plt.savefig(path_output_folder + "compares_histogram_iteration.png", bbox_inches="tight", dpi=200)
    plt.close(fig2)
    w = 0.3
    for x in number_samples:
        sgd_basic = pd.read_csv(path_output_folder_SGD_1 + "Report_SGD_Basic_" + str(x) + ".csv")
        sgd_batch = pd.read_csv(path_output_folder_SGD_2 + "Report_SGD_Batch_" + str(x) + ".csv")
        fig, ax = plt.subplots(2, 2, figsize=(18, 12))
        plt.subplot(221)
        plt.title("Approximations Stochastic Gradient Descent Basic with " + str(x) + " samples")
        plt.plot(np.arange(0, np.shape(sgd_basic)[0]), sgd_basic['w0'].values, linestyle='solid', color='red',
                 label='alpha')
        plt.plot(np.arange(0, np.shape(sgd_basic)[0]), sgd_basic['w1'].values, linestyle='solid', color='blue',
                 label='beta')
        plt.xticks(np.arange(0, np.shape(sgd_basic)[0] + 1, step=np.shape(sgd_basic)[0] / 100 * 10))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)
        alphagd = plt.Circle(xy=(np.shape(sgd_basic)[0], sgd_basic.values[-2:-1, 1:2][0][0]), radius=0.1, color='red',
                             lw=10)
        plt.gcf().gca().add_patch(alphagd)
        plt.text(x=np.shape(sgd_basic)[0] - (np.shape(sgd_basic)[0] / 100 * 5),
                 y=sgd_basic.values[-2:-1, 1:2][0][0] - (sgd_basic.values[-2:-1, 1:2][0][0] / 100 * 20),
                 s=str(sgd_basic.values[-2:-1, 1:2][0][0]),
                 fontsize=10,
                 bbox=dict(facecolor='white', alpha=1),
                 color='black')
        betagd = plt.Circle((np.shape(sgd_basic)[0], sgd_basic.values[-2:-1, 2:3][0][0]), radius=0.1, color='blue',
                            lw=10)
        plt.gcf().gca().add_patch(betagd)
        plt.text(x=np.shape(sgd_basic)[0] - (np.shape(sgd_basic)[0] / 100 * 5),
                 y=sgd_basic.values[-2:-1, 2:3][0][0] + (sgd_basic.values[-2:-1, 1:2][0][0] / 100 * 15),
                 s=str(sgd_basic.values[-2:-1, 2:3][0][0]),
                 fontsize=10,
                 bbox=dict(facecolor='white', alpha=1),
                 color='black')

        plt.hlines(coefficients[0], 0, np.shape(sgd_basic)[0], colors='black', linestyles='solid',
                   label='Real alpha',
                   linewidth=3)
        plt.hlines(coefficients[1], 0, np.shape(sgd_basic)[0], colors='purple', linestyles='solid',
                   label='Real beta',
                   linewidth=3)
        plt.subplot(222)
        plt.title("Approximations Stochastic Gradient Descent Batch with " + str(x) + " samples")
        plt.plot(np.arange(0, np.shape(sgd_batch)[0]), sgd_batch['w0'].values, linestyle='solid', color='red',
                 label='alpha')
        plt.plot(np.arange(0, np.shape(sgd_batch)[0]), sgd_batch['w1'].values, linestyle='solid', color='blue',
                 label='beta')
        plt.xticks(np.arange(0, np.shape(sgd_batch)[0] + 1, step=np.shape(sgd_batch)[0] / 100 * 10))
        alphasgd = plt.Circle(xy=(np.shape(sgd_batch)[0], sgd_batch.values[-2:-1, 1:2][0][0]), radius=0.1, color='red',
                              lw=10)
        plt.gcf().gca().add_patch(alphasgd)
        plt.text(x=np.shape(sgd_batch)[0] - (np.shape(sgd_batch)[0] / 100 * 5),
                 y=sgd_batch.values[-2:-1, 1:2][0][0] - (sgd_batch.values[-2:-1, 1:2][0][0] / 100 * 20),
                 s=str(sgd_batch.values[-2:-1, 1:2][0][0]),
                 fontsize=10,
                 bbox=dict(facecolor='white', alpha=1),
                 color='black')
        betasgd = plt.Circle((np.shape(sgd_batch)[0], sgd_batch.values[-2:-1, 2:3][0][0]), radius=0.1, color='blue',
                             lw=10)
        plt.gcf().gca().add_patch(betasgd)
        plt.text(x=np.shape(sgd_batch)[0] - (np.shape(sgd_batch)[0] / 100 * 5),
                 y=sgd_batch.values[-2:-1, 2:3][0][0] + (sgd_batch.values[-2:-1, 1:2][0][0] / 100 * 15),
                 s=str(sgd_batch.values[-2:-1, 2:3][0][0]),
                 fontsize=10,
                 bbox=dict(facecolor='white', alpha=1),
                 color='black')

        plt.hlines(coefficients[0], 0, np.shape(sgd_batch)[0], colors='black', linestyles='solid',
                   label='Real alpha',
                   linewidth=3)
        plt.hlines(coefficients[1], 0, np.shape(sgd_batch)[0], colors='purple', linestyles='solid',
                   label='Real beta',
                   linewidth=3)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)
        # -------
        plt.subplot(223)
        plt.title("Trend of Loss of SGD Basic with " + str(x) + " samples")

        plt.plot(np.arange(0, np.shape(sgd_basic)[0]), sgd_basic['loss'].values, linestyle='solid', color='red',
                 label='loss')
        plt.xticks(np.arange(0, np.shape(sgd_basic)[0] + 1, step=np.shape(sgd_basic)[0] / 100 * 10))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=1)
        # --------
        plt.subplot(224)
        plt.title("Trend of Loss of SGD Batch with " + str(x) + " samples")
        plt.plot(np.arange(0, np.shape(sgd_batch)[0]), sgd_batch['loss'].values, linestyle='solid', color='red',
                 label='loss')
        plt.xticks(np.arange(0, np.shape(sgd_batch)[0] + 1, step=np.shape(sgd_batch)[0] / 100 * 10))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=1)
        # ------
        plt.tight_layout()
        plt.savefig(path_output_folder + "approximations_" + str(x) + ".png", bbox_inches="tight", dpi=200)
        plt.close(fig=fig)
        # -----

        fig2, ax2 = plt.subplots(1, 1, figsize=(18, 12))
        datapoints = np.load(path_output_folder + filename_dataset + "_" + str(x) + ".npy")
        plt.title("Distribution of " + str(x) + "Points")
        plt.scatter(datapoints[:, 0], datapoints[:, 1], marker="*", color='red', label='Perturbed Points')
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
    summary = pd.read_csv(path_output_folder + "SGD_Basic.csv")
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
    plt.title("Deviations from real coefficients alpha and beta - Stochastic Gradient Descent Basic")
    fig3.tight_layout()
    plt.savefig(path_output_folder + "error_of_approximations_SGD_Basic.png")
    plt.close(fig=fig3)

    fig4, ax4 = plt.subplots(1, 1, figsize=(14, 8))
    summary = pd.read_csv(path_output_folder + "SGD_Batch.csv")
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
    plt.title("Deviations from real coefficients alpha and beta - Stochastic Gradient Descent Batch")
    fig4.tight_layout()
    plt.savefig(path_output_folder + "error_of_approximations_SGD_Batch.png")
    plt.close(fig=fig4)

    return

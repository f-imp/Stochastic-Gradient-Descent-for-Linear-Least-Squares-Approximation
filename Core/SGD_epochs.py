import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Core.algorithms import create_dataset as data
from Core.algorithms import SGD_v3
from Core.algorithms import SGD_v4


def exp4(number_samples, coefficients, noise, initial_point, number_epochs, learning_rate, batch_size,
         path_output_folder, seed=None):
    if seed is None:
        seed = 110395

    filename_dataset = "Dataset2D"
    path_output_folder += "/"
    os.makedirs(path_output_folder, exist_ok=False)
    path_output_folder_SGD_1 = path_output_folder + "SGDOnline/"
    path_output_folder_SGD_2 = path_output_folder + "SGDOnlineMinibatch/"
    os.makedirs(path_output_folder_SGD_1, exist_ok=False)
    os.makedirs(path_output_folder_SGD_2, exist_ok=False)

    for each_num, each_size in zip(number_samples, batch_size):
        details_SGDOnline = {"number_samples": [], "alpha": [], "beta": [], "time": [], "epochs": []}
        details_SGDOnlineMinibatch = {"number_samples": [], "alpha": [], "beta": [], "time": [], "epochs": []}
        np.random.seed(seed=seed)
        fp = data(number_samples=each_num, omega=coefficients, noise=noise,
                  output_name=path_output_folder + filename_dataset + "_" + str(each_num))
        for ep in number_epochs:
            np.random.seed(seed=seed)
            approximations_SGD_Online = SGD_v3(filepath_data=fp, omega_zero=initial_point, lr=learning_rate, epochs=ep,
                                               output_info=path_output_folder_SGD_1 + "Report_SGD_Online_" + str(
                                                   each_num) + "_" + str(ep))
            np.random.seed(seed=seed)
            approximations_SGD_OnlineMinibatch = SGD_v4(filepath_data=fp, omega_zero=initial_point, lr=learning_rate,
                                                        epochs=ep,
                                                        output_info=path_output_folder_SGD_2 + "Report_SGD_OnlineMinibatch_" + str(
                                                            each_num) + "_" + str(ep),
                                                        batch_size=each_size)
            details_SGDOnline["alpha"].append(approximations_SGD_Online[0])
            details_SGDOnline["beta"].append(approximations_SGD_Online[1])
            details_SGDOnline["time"].append(approximations_SGD_Online[2])
            details_SGDOnline["number_samples"].append(each_num)
            details_SGDOnline["epochs"].append(ep)
            details_SGDOnlineMinibatch["alpha"].append(approximations_SGD_OnlineMinibatch[0])
            details_SGDOnlineMinibatch["beta"].append(approximations_SGD_OnlineMinibatch[1])
            details_SGDOnlineMinibatch["time"].append(approximations_SGD_OnlineMinibatch[2])
            details_SGDOnlineMinibatch["number_samples"].append(each_num)
            details_SGDOnlineMinibatch["epochs"].append(ep)
        pd.DataFrame(details_SGDOnline).to_csv(path_output_folder + "SGD_Online_" + str(each_num) + ".csv", index=True,
                                               header=True)
        pd.DataFrame(details_SGDOnlineMinibatch).to_csv(
            path_output_folder + "SGD_OnlineMinibatch_" + str(each_num) + ".csv",
            index=True, header=True)
    w = 0.3
    for num_sample in number_samples:
        sgd_Online = pd.read_csv(path_output_folder + "SGD_Online_" + str(num_sample) + ".csv")
        sgd_OnlineMinibatch = pd.read_csv(path_output_folder + "SGD_OnlineMinibatch_" + str(num_sample) + ".csv")
        num = sgd_Online['epochs'].values
        x = np.arange(len(num))
        fig, ax = plt.subplots()
        SGDOnline = plt.bar(x - w / 2, sgd_Online['time'].values, width=w - 0.05,
                            label='Stochastic Gradient Descent Online')
        SGDOnlineMinibatch = plt.bar(x + w / 2, sgd_OnlineMinibatch['time'].values, width=w - 0.05,
                                     label='Stochastic Gradient Descent Online with Minibatch')
        plt.xticks(x, num)
        fig.tight_layout()
        ax.set_ylabel("Time [s]")
        ax.set_xlabel('Number of Epochs')
        plt.title("Time required to converge - Online vs Online with Minibatch - Samples: " + str(num_sample))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)
        for i, v in enumerate(num):
            ax.text(x=i - w / 2 - 0.05, y=sgd_Online['time'].values[i], s=sgd_Online['time'].values[i],
                    bbox=dict(facecolor='white', alpha=1),
                    color='black')
            ax.text(x=i + w / 2 - 0.05, y=sgd_OnlineMinibatch['time'].values[i],
                    s=sgd_OnlineMinibatch['time'].values[i],
                    bbox=dict(facecolor='white', alpha=1),
                    color='black')

        plt.savefig(path_output_folder + "compares_histogram_time_samples" + str(num_sample) + ".png",
                    bbox_inches="tight",
                    dpi=200)
        plt.close(fig)

    for x in number_samples:
        for e in number_epochs:
            sgd_online = pd.read_csv(path_output_folder_SGD_1 + "Report_SGD_Online_" + str(x) + "_" + str(e) + ".csv")
            sgd_online2 = pd.read_csv(
                path_output_folder_SGD_2 + "Report_SGD_OnlineMinibatch_" + str(x) + "_" + str(e) + ".csv")
            fig, ax = plt.subplots(2, 2, figsize=(18, 12))
            plt.subplot(221)
            plt.title("Approximations SGD Online with " + str(x) + " samples - Epochs: " + str(e))
            plt.plot(np.arange(0, np.shape(sgd_online)[0]), sgd_online['w0'].values, linestyle='solid', color='red',
                     label='alpha')
            plt.plot(np.arange(0, np.shape(sgd_online)[0]), sgd_online['w1'].values, linestyle='solid', color='blue',
                     label='beta')
            plt.xticks(np.arange(0, np.shape(sgd_online)[0] + 1, step=np.shape(sgd_online)[0] / 100 * 10))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)
            alphagd = plt.Circle(xy=(np.shape(sgd_online)[0], sgd_online.values[-2:-1, 1:2][0][0]), radius=0.1,
                                 color='red',
                                 lw=1)
            plt.gcf().gca().add_patch(alphagd)
            plt.text(x=np.shape(sgd_online)[0] - (np.shape(sgd_online)[0] / 100 * 5),
                     y=sgd_online.values[-2:-1, 1:2][0][0] - (sgd_online.values[-2:-1, 1:2][0][0] / 100 * 20),
                     s=str(sgd_online.values[-2:-1, 1:2][0][0]),
                     fontsize=10,
                     bbox=dict(facecolor='white', alpha=1),
                     color='black')
            betagd = plt.Circle((np.shape(sgd_online)[0], sgd_online.values[-2:-1, 2:3][0][0]), radius=0.1,
                                color='blue',
                                lw=1)
            plt.gcf().gca().add_patch(betagd)
            plt.text(x=np.shape(sgd_online)[0] - (np.shape(sgd_online)[0] / 100 * 5),
                     y=sgd_online.values[-2:-1, 2:3][0][0] + (sgd_online.values[-2:-1, 1:2][0][0] / 100 * 15),
                     s=str(sgd_online.values[-2:-1, 2:3][0][0]),
                     fontsize=10,
                     bbox=dict(facecolor='white', alpha=1),
                     color='black')
            plt.hlines(coefficients[0], 0, np.shape(sgd_online)[0], colors='black', linestyles='solid',
                       label='Real alpha',
                       linewidth=3)
            plt.hlines(coefficients[1], 0, np.shape(sgd_online)[0], colors='purple', linestyles='solid',
                       label='Real beta',
                       linewidth=3)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)
            plt.subplot(222)
            plt.title("Approximations SGD Online with Mini Batch with " + str(x) + " samples - Epochs: " + str(e))
            plt.plot(np.arange(0, np.shape(sgd_online2)[0]), sgd_online2['w0'].values, linestyle='solid', color='red',
                     label='alpha')
            plt.plot(np.arange(0, np.shape(sgd_online2)[0]), sgd_online2['w1'].values, linestyle='solid', color='blue',
                     label='beta')
            plt.xticks(np.arange(0, np.shape(sgd_online2)[0] + 1, step=np.shape(sgd_online2)[0] / 100 * 10))
            alphasgd = plt.Circle(xy=(np.shape(sgd_online2)[0], sgd_online2.values[-2:-1, 1:2][0][0]), radius=0.1,
                                  color='red',
                                  lw=1)
            plt.gcf().gca().add_patch(alphasgd)
            plt.text(x=np.shape(sgd_online2)[0] - (np.shape(sgd_online2)[0] / 100 * 5),
                     y=sgd_online2.values[-2:-1, 1:2][0][0] - (sgd_online2.values[-2:-1, 1:2][0][0] / 100 * 20),
                     s=str(sgd_online2.values[-2:-1, 1:2][0][0]),
                     fontsize=10,
                     bbox=dict(facecolor='white', alpha=1),
                     color='black')
            betasgd = plt.Circle((np.shape(sgd_online2)[0], sgd_online2.values[-2:-1, 2:3][0][0]), radius=0.1,
                                 color='blue',
                                 lw=1)
            plt.gcf().gca().add_patch(betasgd)
            plt.text(x=np.shape(sgd_online2)[0] - (np.shape(sgd_online2)[0] / 100 * 5),
                     y=sgd_online2.values[-2:-1, 2:3][0][0] + (sgd_online2.values[-2:-1, 1:2][0][0] / 100 * 15),
                     s=str(sgd_online2.values[-2:-1, 2:3][0][0]),
                     fontsize=10,
                     bbox=dict(facecolor='white', alpha=1),
                     color='black')
            plt.hlines(coefficients[0], 0, np.shape(sgd_online2)[0], colors='black', linestyles='solid',
                       label='Real alpha',
                       linewidth=3)
            plt.hlines(coefficients[1], 0, np.shape(sgd_online2)[0], colors='purple', linestyles='solid',
                       label='Real beta',
                       linewidth=3)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)
            # -------
            plt.subplot(223)
            plt.title("Trend of Loss of SGD Online with " + str(x) + " samples - Epochs: " + str(e))

            plt.plot(np.arange(0, np.shape(sgd_online)[0]), sgd_online['loss'].values, linestyle='solid', color='red',
                     label='loss')
            plt.xticks(np.arange(0, np.shape(sgd_online)[0] + 1, step=np.shape(sgd_online)[0] / 100 * 10))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=1)
            # --------
            plt.subplot(224)
            plt.title("Trend of Loss of SGD Online with Mini Batch with " + str(x) + " samples - Epochs: " + str(e))
            plt.plot(np.arange(0, np.shape(sgd_online2)[0]), sgd_online2['loss'].values, linestyle='solid', color='red',
                     label='loss')
            plt.xticks(np.arange(0, np.shape(sgd_online2)[0] + 1, step=np.shape(sgd_online2)[0] / 100 * 10))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=1)
            # ------
            plt.tight_layout()
            plt.savefig(path_output_folder + "approximations_samples_" + str(x) + "_Epochs" + str(e) + ".png",
                        bbox_inches="tight", dpi=200)
            plt.close(fig=fig)

            # -----
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))
        datapoints = np.load(path_output_folder + filename_dataset + "_" + str(x) + ".npy")
        plt.title("Distribution of " + str(x) + " Points")
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
        summary = pd.read_csv(path_output_folder + "SGD_Online_" + str(x) + ".csv")
        x_axis = summary['epochs'].values
        distinct_x1 = np.arange(len(x_axis))
        y_axis_a = np.absolute(
            np.ones(shape=(np.shape(summary)[0],), dtype=float) * coefficients[0]) - np.absolute(
            summary['alpha'].values)
        y_axis_b = np.absolute(
            np.ones(shape=(np.shape(summary)[0],), dtype=float) * coefficients[1]) - np.absolute(
            summary['beta'].values)
        dev_alpha = plt.bar(distinct_x1 - w / 2 - 0.05, y_axis_a, width=w, label='Alpha')
        dev_beta = plt.bar(distinct_x1 + w / 2 + 0.05, y_axis_b, width=w, label='Beta')
        plt.xticks(distinct_x1, x_axis)
        ax3.set_xlabel('Number of Epochs')
        ax3.set_ylabel("Error of approximations")
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)
        for i, v in enumerate(x_axis):
            ax3.text(x=i - w, y=y_axis_a[i], s=round(y_axis_a[i], 4), fontsize=7,
                     bbox=dict(facecolor='white', alpha=1),
                     color='black')
            ax3.text(x=i + w / 2, y=y_axis_b[i], s=round(y_axis_b[i], 4), fontsize=7,
                     bbox=dict(facecolor='white', alpha=1),
                     color='black')
        plt.title("Deviations from real coefficients alpha and beta - SGD Online - Samples: " + str(x))
        plt.tight_layout()
        plt.savefig(path_output_folder + "ErrorApproximations_SGD_Online_" + str(x) + ".png")
        plt.close(fig=fig3)

        fig4, ax4 = plt.subplots(1, 1, figsize=(14, 8))
        summary = pd.read_csv(path_output_folder + "SGD_OnlineMinibatch_" + str(x) + ".csv")
        x_axis = summary['epochs'].values
        distinct_x2 = np.arange(len(x_axis))
        y_axis_a = np.absolute(
            np.ones(shape=(np.shape(summary)[0],), dtype=float) * coefficients[0]) - np.absolute(
            summary['alpha'].values)
        y_axis_b = np.absolute(
            np.ones(shape=(np.shape(summary)[0],), dtype=float) * coefficients[1]) - np.absolute(
            summary['beta'].values)
        dev_alpha = plt.bar(distinct_x2 - w / 2 - 0.05, y_axis_a, width=w, label='Alpha')
        dev_beta = plt.bar(distinct_x2 + w / 2 + 0.05, y_axis_b, width=w, label='Beta')
        plt.xticks(distinct_x2, x_axis)
        ax4.set_xlabel('Number of Epochs')
        ax4.set_ylabel("Error of approximations")
        ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)
        for i, v in enumerate(x_axis):
            ax4.text(x=i - w, y=y_axis_a[i], s=round(y_axis_a[i], 4), fontsize=7,
                     bbox=dict(facecolor='white', alpha=1),
                     color='black')
            ax4.text(x=i + w / 2, y=y_axis_b[i], s=round(y_axis_b[i], 4), fontsize=7,
                     bbox=dict(facecolor='white', alpha=1),
                     color='black')
        plt.title("Deviations from real coefficients alpha and beta - SGD Online Mini batch - Samples: " + str(x))
        plt.tight_layout()
        plt.savefig(path_output_folder + "ErrorApproximations_SGD_OnlineMinibatch_" + str(x) + ".png")
        plt.close(fig=fig4)

    return

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plots_experiment_with_stoppingcriteria(path_dataframe, n_samples, output_image, stop=None):
    for distinct in n_samples:
        dataframe = pd.read_csv(path_dataframe + ".csv")
        condition = dataframe['number_samples'] == distinct
        new_df = dataframe[condition]
        id_min_vertical_errors = new_df[['absolute_vertical_errors']].idxmin()
        datapoints = np.load(dataframe['dataset'].values[id_min_vertical_errors][0] + ".npy")
        fig = plt.figure(1, figsize=(10, 6), dpi=200)
        plt.scatter(datapoints[:, 0].reshape(np.shape(datapoints)[0], 1),
                    datapoints[:, 1].reshape(np.shape(datapoints)[0], 1), marker=".", color='blue', label='Points',
                    linewidths=5,
                    alpha=0.8)
        x = datapoints[:, 0].reshape(np.shape(datapoints)[0], 1)
        if stop == 'Epochs':
            title = "Best Line computed in " + (path_dataframe.split("/")[-1]).split(".")[
                0] + "\nNumber of Samples: " + str(
                dataframe["number_samples"].values[id_min_vertical_errors][
                    0]) + "  -  Number of Epochs: " + str(
                dataframe["epochs"].values[id_min_vertical_errors][0]) + "\nAlpha: " + str(
                dataframe["alpha"].values[id_min_vertical_errors][0]) + " -  Beta: " + str(
                dataframe["beta"].values[id_min_vertical_errors][0])
        else:
            title = "Best Line computed in " + (path_dataframe.split("/")[-1]).split(".")[
                0] + "\nNumber of Samples: " + str(
                dataframe["number_samples"].values[id_min_vertical_errors][0]) + "\nAlpha: " + str(
                dataframe["alpha"].values[id_min_vertical_errors][0]) + " -  Beta: " + str(
                dataframe["beta"].values[id_min_vertical_errors][0])

        plt.plot(x, x * dataframe["alpha"].values[id_min_vertical_errors][0] + dataframe["beta"].values[
            id_min_vertical_errors][0] * np.ones(
            shape=(np.shape(x)[0], 1), dtype=float), linewidth=2,
                 label=dataframe["algorithm"].values[id_min_vertical_errors][0],
                 color='red')
        plt.title(title)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=1)
        plt.tight_layout()
        plt.savefig(output_image + "_" + str(distinct) + ".png")
        plt.close(fig=fig)
    return

import time
import pandas as pd

import numpy as np


def create_dataset(number_samples, omega, noise, output_name):
    X = np.random.uniform(low=1, high=100, size=(number_samples, 1))
    Y = np.array([], dtype=float)
    for i in range(np.shape(X)[0]):
        y = omega[0] * X[i] + omega[1]
        y = y + np.random.random_integers(low=-100, high=100) * noise
        Y = np.append(Y, y, axis=0)
    Y = Y.reshape((np.shape(Y)[0], 1))
    data = np.hstack((X, Y))
    np.save(output_name + ".npy", data)
    return output_name + ".npy"


def evaluate_vertical_errors(data, alpha, beta):
    y_hat = alpha * data[:, 0:1] + beta * np.ones(shape=(np.shape(data)[0], 1), dtype=float)
    vertical_errors = y_hat - data[:, 1:2]
    # loss = 1 / np.shape(d)[0] * pow(np.sum(vertical_errors), 2)
    loss = 1 / np.shape(data)[0] * pow(np.linalg.norm(vertical_errors, ord=2), 2)
    return vertical_errors, loss


def evaluate_gradient(data, vertical_errors):
    gradient = (2 / np.shape(data)[0]) * np.array([np.sum(vertical_errors * data[:, 0:1]), np.sum(vertical_errors)],
                                                  dtype=float)
    return gradient


def Gradient_Descent(filepath_data, omega_zero, lr, tolerance, output_info):
    info = {"w0": [], "w1": [], "loss": []}
    start = time.clock()
    d = np.load(filepath_data)
    np.random.shuffle(d)
    n_iteration = 0
    flag = 0
    w0, w1 = omega_zero[0], omega_zero[1]
    while flag == 0:
        vertical_errors, loss = evaluate_vertical_errors(data=d, alpha=w0, beta=w1)
        gradient_of_loss = evaluate_gradient(data=d, vertical_errors=vertical_errors)
        info["loss"].append(loss), info["w0"].append(w0), info["w1"].append(w1)
        w0, w1 = w0 - lr * gradient_of_loss[0], w1 - lr * gradient_of_loss[1]
        if tolerance >= np.linalg.norm(gradient_of_loss, ord=2):
            flag = 1
        n_iteration += 1
    info["loss"].append(np.nan), info["w0"].append(w0), info["w1"].append(w1)
    end = time.clock()
    print("Number of Iteration needed to converge", str(n_iteration), "\nTime Required : ",
          str(round(end - start, 2)) + " seconds")
    pd.DataFrame(info).to_csv(output_info + ".csv", index=True, header=True)
    return w0, w1, round(end - start, 2), n_iteration


def SGD_v1(filepath_data, omega_zero, lr, tolerance, output_info):
    info = {"w0": [], "w1": [], "loss": []}
    times = 0.0
    d = np.load(filepath_data)
    np.random.shuffle(d)
    n_iteration = 0
    flag = 0
    w0, w1 = omega_zero[0], omega_zero[1]
    previous_index = -1
    while flag == 0:
        random_index = np.random.randint(low=0, high=np.shape(d)[0] - 1)
        # To be sure about replacement
        while random_index == previous_index:
            random_index = np.random.randint(low=0, high=np.shape(d)[0] - 1)
        previous_index = random_index
        sample = d[random_index:random_index + 1, :]
        start = time.clock()
        vertical_errors, loss = evaluate_vertical_errors(data=sample, alpha=w0, beta=w1)
        gradient_of_loss = evaluate_gradient(data=sample, vertical_errors=vertical_errors)
        info["loss"].append(loss), info["w0"].append(w0), info["w1"].append(w1)
        w0, w1 = w0 - lr * gradient_of_loss[0], w1 - lr * gradient_of_loss[1]
        if tolerance >= np.linalg.norm(gradient_of_loss, ord=2):
            flag = 1
        n_iteration += 1
        end = time.clock()
        times += end - start
    info["loss"].append(np.nan), info["w0"].append(w0), info["w1"].append(w1)
    print("Number of Iteration needed to converge", str(n_iteration), "\nTime Required : ",
          str(round(times, 2)) + " seconds")
    pd.DataFrame(info).to_csv(output_info + ".csv", index=True, header=True)
    return w0, w1, round(times, 2), n_iteration


def SGD_v2(filepath_data, omega_zero, lr, tolerance, output_info, batch_size):
    info = {"w0": [], "w1": [], "loss": []}
    times = 0.0
    d = np.load(filepath_data)
    np.random.shuffle(d)
    n_iteration = 0
    flag = 0
    w0, w1 = omega_zero[0], omega_zero[1]
    num_batches = np.int(np.floor(np.shape(d)[0] / batch_size))
    batches = np.array_split(d, num_batches)
    while flag == 0:
        random_index_batches = np.random.randint(low=0, high=np.shape(batches)[0])
        batch = batches[random_index_batches]
        np.random.shuffle(batch)
        start = time.clock()
        vertical_errors, loss = evaluate_vertical_errors(data=batch, alpha=w0, beta=w1)
        gradient_of_loss = evaluate_gradient(data=batch, vertical_errors=vertical_errors)
        info["loss"].append(loss), info["w0"].append(w0), info["w1"].append(w1)
        w0, w1 = w0 - lr * gradient_of_loss[0], w1 - lr * gradient_of_loss[1]
        if tolerance >= np.linalg.norm(gradient_of_loss, ord=2):
            flag = 1
        n_iteration += 1
        end = time.clock()
        times += end - start
    info["loss"].append(np.nan), info["w0"].append(w0), info["w1"].append(w1)
    end = time.clock()
    print("Number of Iteration needed to converge", str(n_iteration), "\nTime Required : ",
          str(round(times, 2)) + " seconds")
    pd.DataFrame(info).to_csv(output_info + ".csv", index=True, header=True)
    return w0, w1, round(times, 2), n_iteration


def SGD_v3(filepath_data, omega_zero, lr, epochs, output_info):
    info = {"w0": [], "w1": [], "loss": []}
    start = time.clock()
    d = np.load(filepath_data)
    np.random.shuffle(d)
    n_iteration = 0
    w0, w1 = omega_zero[0], omega_zero[1]
    info["loss"].append(np.nan), info["w0"].append(w0), info["w1"].append(w1)
    while n_iteration < epochs:
        np.random.shuffle(d)
        l, a, b = [], [], []
        for i in range(0, np.shape(d)[0] - 1):
            sample = d[i:i + 1, :]
            vertical_errors, loss = evaluate_vertical_errors(data=sample, alpha=w0, beta=w1)
            gradient_of_loss = evaluate_gradient(data=sample, vertical_errors=vertical_errors)
            w0, w1 = w0 - lr * gradient_of_loss[0], w1 - lr * gradient_of_loss[1]
            l.append(loss), a.append(w0), b.append(w1)
        n_iteration += 1
        info["loss"].append(np.mean(l)), info["w0"].append(np.mean(a)), info["w1"].append(np.mean(b))

    end = time.clock()
    print("Number of Iteration needed to converge", str(n_iteration), "\nTime Required : ",
          str(round(end - start, 2)) + " seconds")
    pd.DataFrame(info).to_csv(output_info + ".csv", index=True, header=True)
    return w0, w1, round(end - start, 2)


def SGD_v4(filepath_data, omega_zero, lr, epochs, batch_size, output_info):
    info = {"w0": [], "w1": [], "loss": []}
    start = time.clock()
    d = np.load(filepath_data)
    np.random.shuffle(d)
    n_iteration = 0
    w0, w1 = omega_zero[0], omega_zero[1]
    info["loss"].append(np.nan), info["w0"].append(w0), info["w1"].append(w1)
    num_batches = np.int(np.floor(np.shape(d)[0] / batch_size))
    batches = np.array_split(d, num_batches)
    while n_iteration < epochs:
        l, a, b = [], [], []
        for i in range(0, np.shape(batches)[0] - 1):
            batch = batches[i]
            np.random.shuffle(batch)
            vertical_errors, loss = evaluate_vertical_errors(data=batch, alpha=w0, beta=w1)
            gradient_of_loss = evaluate_gradient(data=batch, vertical_errors=vertical_errors)
            w0, w1 = w0 - lr * gradient_of_loss[0], w1 - lr * gradient_of_loss[1]
            l.append(loss), a.append(w0), b.append(w1)
        info["loss"].append(np.mean(l, dtype=float)), info["w0"].append(np.mean(a, dtype=float)), info["w1"].append(
            np.mean(b, dtype=float))
        n_iteration += 1
    end = time.clock()
    print("Number of Iteration needed to converge", str(n_iteration), "\nTime Required : ",
          str(round(end - start, 2)) + " seconds")
    df = pd.DataFrame(info)
    df.to_csv(output_info + ".csv", index=True, header=True)
    return w0, w1, round(end - start, 2)

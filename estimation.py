import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

ZERO_BIAS_FILE = "./data/data22_13p6TeV.00435946.physics_ZeroBias_eba_a13r_hg_mu40.csv"
SIGNALS_FILE = "./data/signals.csv"
N_NEURONS = 7
N_HIDDEN_LAYERS = 1


def main():
    # read signals and noise data
    noise_dataframe = read_zero_bias_dataset()
    n_events = len(noise_dataframe)
    signals_dataframe = read_signals_dataset().head(n_events)

    # prepare data summing up signals and noise
    truth_amplitudes = signals_dataframe["truth_amp"].to_numpy()
    signals_samples = signals_dataframe[["sample_0", "sample_1", "sample_2", "sample_3", "sample_4", "sample_5", "sample_6"]].to_numpy()
    noise_samples = noise_dataframe[["sample_0", "sample_1", "sample_2", "sample_3", "sample_4", "sample_5", "sample_6"]].to_numpy()
    corrupted_signals = noise_samples + signals_samples

    # split datasets into train and test
    train_quote = 0.5
    train_samples, test_samples = split_train_test(corrupted_signals, train_quote)
    train_amplitudes, test_amplitudes = split_train_test(truth_amplitudes, train_quote)
    train_noise, _ = split_train_test(noise_samples, train_quote)

    # design optimal filter
    of2 = design_optimal_filter(train_noise, True)

    # design MLP corrector
    mlp = design_mlp_corrector(of2, train_samples, train_amplitudes)

    # assess OF2 and MLP performance
    of2_estimation = of2 @ test_samples.T
    mlp_correction = mlp.predict(test_samples)
    mlp_estimation = of2_estimation - mlp_correction

    of2_error = assess_performance("OF2", of2_estimation, test_amplitudes)
    mlp_error = assess_performance("OF2+MLP", mlp_estimation, test_amplitudes)
    draw_histogram(of2_error, mlp_error)


def assess_performance(method, estimation, target):
    error = estimation - target
    mean_error = np.mean(error)
    std_error = np.std(error)
    print(f"{method} mean error: {mean_error}")
    print(f"{method} std error: {std_error}")
    return error


def draw_histogram(data_1, data_2):
    plt.hist(data_1, bins=100, alpha=0.5, label="OF2")
    plt.hist(data_2, bins=100, alpha=0.5, label="OF2+MLP")
    plt.legend(loc="upper right")
    plt.show()


def design_mlp_corrector(optimal_filter, input, target):
    print("Designing MLP corrector...")
    of_estimation = optimal_filter @ input.T
    of_error = of_estimation - target
    mlp = MLPRegressor(
        hidden_layer_sizes=(N_NEURONS, ) * N_HIDDEN_LAYERS,
        max_iter=1000,
        activation="relu"
    )
    mlp.fit(input, of_error)
    return mlp


def design_optimal_filter(noise_dataset, optimize=False):
    print("Designing optimal filter...")
    # Defining the pulse and the pulse derivative
    pulse = np.array(
        [[0.0000, 0.0172, 0.4524, 1.0000, 0.5633, 0.1493, 0.0424]])
    d_pulse = np.array([[0.00004019, 0.00333578, 0.03108120,
                       0.00000000, -0.02434490, -0.00800683, -0.00243344]])

    # Checking if the OF2 is optimize or not
    if optimize == 0 or not optimize:
        covariance = np.identity(7)
    elif optimize == 1 or optimize:
        if len(noise_dataset) > len(noise_dataset[0]):
            covariance = np.cov(noise_dataset.transpose())
        else:
            covariance = np.cov(noise_dataset)
    else:
        raise Exception('Options for "optimize" are 0 (False) or 1 (True).')

    # Making the B matrix
    b = np.concatenate((covariance, pulse, d_pulse, np.ones((1, 7))), axis=0)
    pulse_plus = np.concatenate((pulse.transpose(), np.zeros((3, 1))), axis=0)
    d_pulse_plus = np.concatenate(
        (d_pulse.transpose(), np.zeros((3, 1))), axis=0)
    ones_plus = np.concatenate((np.ones((7, 1)), np.zeros((3, 1))), axis=0)
    b = np.concatenate((b, pulse_plus, d_pulse_plus, ones_plus), axis=1)

    # Making the C matrix
    c = np.concatenate((np.zeros(7), np.ones(1), np.zeros(2)))

    # Estimating the OF2 weights
    weights = (np.linalg.inv(b) @ c)[0:7]

    return weights


def split_train_test(dataframe, train_quote):
    n_events = len(dataframe)
    n_train = int(n_events * train_quote)
    train_dataset = dataframe[0:n_train]
    test_dataset = dataframe[n_train:n_events]
    return train_dataset, test_dataset


def read_zero_bias_dataset():
    return pd.read_csv(ZERO_BIAS_FILE,
                       sep=",",
                       dtype={
                           "event_number": int,
                           "approx_mu": float,
                           "gain": int,
                           "partition": int,
                           "channel": int,
                           "module": int,
                           "sample_0": int,
                           "sample_1": int,
                           "sample_2": int,
                           "sample_3": int,
                           "sample_4": int,
                           "sample_5": int,
                           "sample_6": int
                       })


def read_signals_dataset():
    return pd.read_csv(SIGNALS_FILE,
                       sep=",",
                       dtype={
                           "sample_0": int,
                           "sample_1": int,
                           "sample_2": int,
                           "sample_3": int,
                           "sample_4": int,
                           "sample_5": int,
                           "sample_6": int,
                           "truth_amp": int,
                       })


if __name__ == "__main__":
    main()

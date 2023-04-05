
import os

import numpy as np
from scipy.io import loadmat
from scipy import signal as sig
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge

plt.rcParams['figure.figsize'] = (9, 6)


def main(path_output, path_patient_a, path_patient_b):

    # load mats
    np_patient_a = loadmat(path_patient_a)["data"]
    np_patient_b = loadmat(path_patient_b)["data"]
    patients = (np_patient_a, np_patient_b)
    sample_rate_original = 3000
    n_channels = 3
    known_seizure_idx_a = 2480000
    known_seizure_idx_b = 2800000

    # visualize full data
    for label, np_patient in (("a", np_patient_a), ("b", np_patient_b)):
        fig, axes = plt.subplots(n_channels, 1, sharex="all", sharey="all")
        for chan_idx in range(n_channels):
            axes[chan_idx].plot(np_patient[chan_idx], linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(path_output, f"all_data_{label}.png"))
        plt.close()

    # visualize known seizures
    plot_seconds_before = 5
    plot_seconds_after = 60
    plt_samps_before = plot_seconds_before * sample_rate_original
    plt_samps_after = plot_seconds_after * sample_rate_original
    for label, np_patient, known_idx in (
            ("a", np_patient_a, known_seizure_idx_a),
            ("b", np_patient_b, known_seizure_idx_b)):
        fig, axes = plt.subplots(3, 1, sharex="all", sharey="all")
        for chan_idx in range(n_channels):
            axes[chan_idx].plot(
                range(known_idx - plt_samps_before, known_idx + plt_samps_after),
                np_patient[chan_idx, known_idx - plt_samps_before:known_idx + plt_samps_after],
                linewidth=0.5)
            axes[chan_idx].axvline(x=known_idx, color="red")
        plt.tight_layout()
        plt.savefig(os.path.join(path_output, f"known_seizure_{label}.png"))
        plt.close()

    # define rolling window
    window_width_seconds = 10
    window_step_seconds = 1
    window_width_samples = window_width_seconds * sample_rate_original
    window_step_samples = window_step_seconds * sample_rate_original
    sample_rate_features = sample_rate_original / window_step_samples
    np_patients_rolling = []
    for np_patient in (np_patient_a, np_patient_b):
        np_patients_rolling.append(
            np.lib.stride_tricks.sliding_window_view(
                np_patient, window_width_samples, axis=1)[:, ::window_step_samples])

    # compute features
    n_feats_per_channel = 3
    patients_amplitude = [
        np.mean(np.abs(np_patient_rolling), axis=-1)
        for np_patient_rolling
        in np_patients_rolling]
    patients_zcrossings = [
        np.sum(np.logical_xor((_sign := np_patient_rolling > 0)[:, :, 1:], _sign[:, :, :-1]).astype(int), axis=-1)
        for np_patient_rolling
        in np_patients_rolling]
    patients_linelength = [
        np.sum(np.abs(np_patient_rolling[:, :, 1:] - np_patient_rolling[:, :, :-1]), axis=-1)
        for np_patient_rolling
        in np_patients_rolling]
    patients_features = list(zip(patients_amplitude, patients_zcrossings, patients_linelength))

    # visualize features on full data
    for patient_label, patient_features in zip(("a", "b"), patients_features):
        fig, axes = plt.subplots(3, 1, sharex="all", sharey="none")
        for feat_idx, np_patient_feature, feature_label, feature_bounds in zip(
                range(n_feats_per_channel),
                patient_features,
                ("amplitude", "zerocrossings", "linelength"),
                ((0, 1000), (0, 1000), (0, 0.6e6))):
            for chan_idx in range(n_channels):
                axes[feat_idx].plot(np_patient_feature[chan_idx], label=f"channel {chan_idx}")
            axes[feat_idx].set_ylabel(feature_label)
            axes[feat_idx].legend()
            axes[feat_idx].set_ylim(*feature_bounds)
        plt.tight_layout()
        plt.savefig(os.path.join(path_output, f"all_data_features_{patient_label}.png"))
        plt.close()

    # rescale features using first 600 seconds as a baseline
    baseline_duration_seconds = 600
    baseline_duration_features = int(baseline_duration_seconds * sample_rate_features)
    patient_features_baseline = [
        [
            np.mean(np_patient_feature[:, :baseline_duration_features], axis=1, keepdims=True)
            for np_patient_feature
            in patient_features
        ]
        for patient_features
        in patients_features]
    patients_features_rescaled = [
        [
            np_patient_feature / np_patient_baseline
            for np_patient_feature, np_patient_baseline
            in zip(patient_features, patient_features_baseline)
        ]
        for patient_features, patient_features_baseline
        in zip(patients_features, patient_features_baseline)]

    # take mean of rescaled features across channels
    patients_features_chan_mean = [
        [
            np.mean(np_patient_feature_rescaled, axis=0)
            for np_patient_feature_rescaled
            in patient_features_rescaled
        ]
        for patient_features_rescaled
        in patients_features_rescaled]

    # visualize rescaled features on full data
    for patient_label, patient_features, patient_features_chan_mean in zip(
            ("a", "b"), patients_features_rescaled, patients_features_chan_mean):
        fig, axes = plt.subplots(3, 1, sharex="all", sharey="none")
        for feat_idx, np_patient_feature, np_patient_feature_chan_mean, feature_label, feature_bounds in zip(
                range(n_feats_per_channel),
                patient_features,
                patient_features_chan_mean,
                ("amplitude", "zerocrossings", "linelength"),
                ((0, 5), (0, 5), (0, 5))):
            for chan_idx in range(n_channels):
                axes[feat_idx].plot(np_patient_feature[chan_idx], label=f"channel {chan_idx}")
            axes[feat_idx].plot(np_patient_feature_chan_mean, label=f"mean", linewidth=2)
            axes[feat_idx].set_ylabel(feature_label)
            axes[feat_idx].legend()
            axes[feat_idx].set_ylim(*feature_bounds)
        plt.tight_layout()
        plt.savefig(os.path.join(path_output, f"all_data_features_rescaled_{patient_label}.png"))
        plt.close()

    # perform thresholding on mean features
    bounds_amplitude = (np.NINF, 2.3)
    bounds_zcrossings = (np.NINF, np.PINF)
    bounds_linelength = (np.NINF, 1.1)
    patients_features_thresholded = [
        [
            np.bitwise_or(
                    np_patient_feature < feature_bounds[0],
                    np_patient_feature > feature_bounds[1])
            for np_patient_feature, feature_bounds
            in zip(patient_features, (bounds_amplitude, bounds_zcrossings, bounds_linelength))
        ]
        for patient_features
        in patients_features_chan_mean]

    # visualize thresholded features on full data
    for patient_label, patient_features in zip(("a", "b"), patients_features_thresholded):
        fig, axes = plt.subplots(4, 1, sharex="all", sharey="none")
        for feat_idx, np_patient_feature, feature_label in zip(
                range(n_feats_per_channel),
                patient_features,
                ("amplitude", "zerocrossings", "linelength")):
            axes[feat_idx].plot(np_patient_feature)
            axes[feat_idx].set_ylabel(feature_label)
            axes[feat_idx].set_ylim(-0.1, 1.1)
        axes[n_feats_per_channel].plot(np.any(patient_features, axis=0))
        axes[n_feats_per_channel].set_ylabel("any")
        axes[n_feats_per_channel].set_ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(path_output, f"all_data_features_thresholded_{patient_label}.png"))
        plt.close()

    # slice data for training ridge regression
    features_train, features_test = patients_features_chan_mean
    features_train = np.array(features_train).T
    features_test = np.array(features_test).T
    targets_train = np.any(patients_features_thresholded[0], axis=0).astype(int) * 2 - 1

    # train and test ridge regression
    ridge_alpha = 0.1
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(features_train, targets_train)
    targets_train_pred = ridge.predict(features_train)
    targets_test_pred = ridge.predict(features_test)

    # visualize ridge regression results on full data
    for label, np_patient, pred in (("a", np_patient_a, targets_train_pred), ("b", np_patient_b, targets_test_pred)):
        fig, axes = plt.subplots(n_channels+1, 1, sharex="all", sharey="none")
        for chan_idx in range(n_channels):
            axes[chan_idx].plot(np_patient[chan_idx], linewidth=0.5)
        axes[n_channels].plot(np.arange(0, pred.shape[0]) * window_step_samples, pred)
        axes[n_channels].plot(np.arange(0, pred.shape[0]) * window_step_samples, (pred > 0) * 2 - 1)
        axes[n_channels].set_ylim(-1.2, 1.2)
        plt.tight_layout()
        plt.savefig(os.path.join(path_output, f"ridge_predictions_{label}.png"))
        plt.close()

    # design and apply 48db/oct digital filter (24db/oct applied once in both directions)
    # then calculate smoothed power
    order = 4
    bands_cutoff_Hz = 100
    smoothing_width_seconds = 20 / 1000
    smoothing_width_samples = int(smoothing_width_seconds * sample_rate_original)
    np_smoothing_kernel = np.ones(smoothing_width_samples) / smoothing_width_samples
    sos_highpass = sig.butter(N=order, Wn=bands_cutoff_Hz, btype="highpass", output="sos", fs=sample_rate_original)
    patients_channels_highpower = []
    for np_patient in patients:
        patient_channels_highpower = []
        for np_patient_channel in np_patient:
            np_patient_channel_highpassed = sig.sosfiltfilt(sos=sos_highpass, x=np_patient_channel)
            np_patient_channel_power = np.square(np_patient_channel_highpassed)
            np_patient_channel_smoothed = np.convolve(np_patient_channel_power, np_smoothing_kernel, mode="same")
            patient_channels_highpower.append(np_patient_channel_smoothed)
        patients_channels_highpower.append(patient_channels_highpower)

    # visualize high frequency power
    for label, np_patient_channels_highpower in zip(("a", "b"), patients_channels_highpower):
        fig, axes = plt.subplots(n_channels, 1, sharex="all", sharey="all")
        for chan_idx in range(n_channels):
            axes[chan_idx].plot(np_patient_channels_highpower[chan_idx])
        plt.tight_layout()
        plt.savefig(os.path.join(path_output, f"highpower_{label}.png"))
        plt.close()

    # compute a feature for high frequency power using same rolling window as other features
    for patient_idx, patient_channels_highpower in enumerate(patients_channels_highpower):
        np_patient_channels_highpower = np.array(patient_channels_highpower)
        np_patient_channels_highpower_rolling = np.lib.stride_tricks.sliding_window_view(
                np_patient_channels_highpower, window_width_samples, axis=1)[:, ::window_step_samples]
        np_patient_channels_highpower_max = np.max(np_patient_channels_highpower_rolling, axis=2)
        np_patient_channels_highpower_max /= np.mean(
            np_patient_channels_highpower_max[:, :baseline_duration_features], axis=1, keepdims=True)
        np_patient_chan_mean_highpower_max = np.mean(np_patient_channels_highpower_max, axis=0)
        patients_features_chan_mean[patient_idx].append(np_patient_chan_mean_highpower_max)

    # visualize rescaled features on full data
    for patient_label, patient_features_chan_mean in zip(("a", "b"), patients_features_chan_mean):
        fig, axes = plt.subplots(len(patient_features_chan_mean), 1, sharex="all", sharey="none")
        for feat_idx, np_patient_feature_chan_mean, feature_label, feature_bounds in zip(
                range(len(patient_features_chan_mean)),
                patient_features_chan_mean,
                ("amplitude", "zerocrossings", "linelength", "highpower"),
                ((0, 5), (0, 5), (0, 5), (0, 5))):
            axes[feat_idx].plot(np_patient_feature_chan_mean)
            axes[feat_idx].set_ylabel(feature_label)
            axes[feat_idx].set_ylim(*feature_bounds)
        plt.tight_layout()
        plt.savefig(os.path.join(path_output, f"all_data_features_rescaled_inc_highpower_{patient_label}.png"))
        plt.close()

    # slice data for training ridge regression
    features_train, features_test = patients_features_chan_mean
    features_train = np.array(features_train).T
    features_test = np.array(features_test).T

    # train and test ridge regression
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(features_train, targets_train)
    targets_train_pred = ridge.predict(features_train)
    targets_test_pred = ridge.predict(features_test)

    # visualize ridge regression results on full data
    for label, np_patient, pred in (("a", np_patient_a, targets_train_pred), ("b", np_patient_b, targets_test_pred)):
        fig, axes = plt.subplots(n_channels + 1, 1, sharex="all", sharey="none")
        for chan_idx in range(n_channels):
            axes[chan_idx].plot(np_patient[chan_idx], linewidth=0.5)
        axes[n_channels].plot(np.arange(0, pred.shape[0]) * window_step_samples, pred)
        axes[n_channels].plot(np.arange(0, pred.shape[0]) * window_step_samples, (pred > 0) * 2 - 1)
        axes[n_channels].set_ylim(-1.2, 1.2)
        plt.tight_layout()
        plt.savefig(os.path.join(path_output, f"ridge_predictions_inc_highpower_{label}.png"))
        plt.close()


if __name__ == '__main__':
    _path_output = r"D:\Documents\Academics\BME517\bme_lab_10_11_report"
    _path_patient_a = r"D:\Documents\Academics\BME517\bme_lab_10_11\data\Patient_A.mat"
    _path_patient_b = r"D:\Documents\Academics\BME517\bme_lab_10_11\data\Patient_B.mat"
    main(_path_output, _path_patient_a, _path_patient_b)

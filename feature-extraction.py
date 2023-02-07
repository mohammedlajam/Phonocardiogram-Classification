"""
Author: Mohammed Lajam

Phase 3: Feature Extraction:
- In this phase, different features are extracted from the preprocessed signals.
- These features are extracted from:
    - Time-Domain
    - Frequency Domain
    - Time-Frequency Representation

Objective:
The objective of this phase is to extract all the possible features from the preprocessed
signals, so that it can be used as data in the Machine Learning Phase (Phase 4).

Input:
- The input is the preprocessed signals, which are saved in 'data/denoised_signals' directory. and
it is being accessed using the function 'access_signals'

Output:
1. Numeric Features of all the used methods in a DataFrame and are being saved in csv format in
the directory 'data/extracted_features/csv'
2. Images of all the Time-Frequency Domain methods and csv file containing signal_ids and their
classes.

Note:
- All the functions and variables, which are used in this file, are imported from helpers.py
file from the feature_extraction package in the same repository.
"""


# Importing libraries:
import pandas as pd
from glob import glob
import os

import constants as c
from feature_extraction.helpers import *

import warnings
warnings.filterwarnings('ignore')


def check_create_dir():
    """Function to check if the img and csv directories exist in the data directory and
    create them if they are not existed"""
    # Check if data directory exists:
    if os.path.exists(f'{c.REPO_PATH}{c.FEATURE_EXTRACTION_PATH}'):
        pass
    else:
        os.mkdir(f'{c.REPO_PATH}{c.FEATURE_EXTRACTION_PATH}')

    # Check if csv directory exists inside data directory:
    if os.path.exists(f'{c.REPO_PATH}{c.FEATURE_EXTRACTION_PATH}/csv'):
        pass
    else:
        os.mkdir(f'{c.REPO_PATH}{c.FEATURE_EXTRACTION_PATH}/csv')

    # Check if images directory exists inside data directory:
    if os.path.exists(f'{c.REPO_PATH}{c.FEATURE_EXTRACTION_PATH}/images'):
        pass
    else:
        os.mkdir(f'{c.REPO_PATH}{c.FEATURE_EXTRACTION_PATH}/images')

    # Check if directories per image type exist inside data directory:
    image_types = ['spectrogram', 'mel_spectrogram', 'mfccs', 'scalogram']
    for image_type in image_types:
        if os.path.exists(f'{c.REPO_PATH}{c.FEATURE_EXTRACTION_PATH}/images/{image_type}'):
            pass
        else:
            os.mkdir(f'{c.REPO_PATH}{c.FEATURE_EXTRACTION_PATH}/images/{image_type}')

    # Check if directories per MFCCs type exist inside data directory:
    mfcc_types = ['mfcc', 'delta_1', 'delta_2']
    for mfcc_type in mfcc_types:
        if os.path.exists(f'{c.REPO_PATH}{c.FEATURE_EXTRACTION_PATH}/images/mfccs/{mfcc_type}'):
            pass
        else:
            os.mkdir(f'{c.REPO_PATH}{c.FEATURE_EXTRACTION_PATH}/images/mfccs/{mfcc_type}')


def access_signals(denoise_method):
    """Function to access the latest version of the preprocessed signals.
    the denoise_method is either 'emd', 'wavelet_transform', 'digital_filters', 'emd_wavelet',
    'emd_dfilters' or 'emd_wl_dfilters'
    """
    # Searching and reading the latest csv version based on with 'denoising method':
    list_of_versions = glob(f'{c.REPO_PATH}{c.SIG_PRE_PATH}/{denoise_method}/*.csv')
    latest_version = max(list_of_versions, key=os.path.getctime)
    signals = pd.read_csv(latest_version)

    # generating signal_ids:
    signal_ids = []
    for number in range(len(signals)):
        signal_id = f'signal_{number}'
        signal_ids.append(signal_id)
    signals['signal_id'] = signal_ids

    # Separate the signals from signal_id and class:
    signal_references = signals.loc[:, ['signal_id', 'class']]
    signals = signals.drop(['signal_id', 'class'], axis='columns')
    return signals, signal_references


def concatenate_dataframes(*features):
    """Function to concatenate all DataFrames"""
    all_features = [features]
    for feature in all_features:
        dataframes = pd.concat(feature, axis=1, join='inner')
    return dataframes


def save_features(dataframe, csv_version):
    """Function to save DataFrame into csv in the 'extracted_feature' directory"""
    try:
        os.path.exists(f'{c.REPO_PATH}{c.FEATURE_EXTRACTION_PATH}/csv')
    except Exception as e:
        print(e)
    else:
        while os.path.isfile(f'{c.REPO_PATH}{c.FEATURE_EXTRACTION_PATH}/csv/extracted_features_v{csv_version}.csv'):
            csv_version += 1
            continue
        else:
            dataframe.to_csv(f'{c.REPO_PATH}{c.FEATURE_EXTRACTION_PATH}/csv/extracted_features_v{csv_version}.csv',
                             index=False)
    return None


class NumericFeatureExtraction:
    """A class to extract all the features for all preprocessed signals."""
    def __init__(self, signals, sr, frame_size, hop_size, split_frequency, n_mfcc):
        self.signals = signals
        self.sr = sr
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.split_frequency = split_frequency
        self.n_mfcc = n_mfcc

    # Functions for extracting Time-Domain Features for all Signals:
    def extract_descriptive_statistics(self):
        """Extract the descriptive Statistics of every signal and return a DataFrame."""
        ds_max, ds_min, ds_mean, ds_median, ds_std = [], [], [], [], []
        for signal in range(len(self.signals)):
            maximum, minimum, mean, median, std = descriptive_statistics(signal=self.signals.iloc[signal, :])
            ds_max.append(maximum)
            ds_min.append(minimum)
            ds_mean.append(mean)
            ds_median.append(median)
            ds_std.append(std)

        # Create a Date Frame for the Descriptive Statistics of all signals:
        ds_dataframe = pd.DataFrame({'ds_max': ds_max,
                                     'ds_min': ds_min,
                                     'ds_mean': ds_mean,
                                     'ds_median': ds_median,
                                     'ds_std': ds_std})
        return ds_dataframe

    def extract_energy_power(self):
        """Extract the energy and total power of eaach signal and return a DataFrame."""
        energies, powers = [], []
        for signal in range(len(self.signals)):
            energy, power = energy_power(signal=self.signals.iloc[signal, :])
            energies.append(energy)
            powers.append(power)

        # Create a Date Frame for energy and power for all signals:
        ep_dataframe = pd.DataFrame({'energy': energies,
                                     'power': powers})
        return ep_dataframe

    def extract_amplitude_envelope(self):
        """Extract Amplitude Envelope of each frame for all signals and return the Descriptive Statistics in
        a DataFrame"""
        ae_max, ae_min, ae_mean, ae_median, ae_std = [], [], [], [], []
        for signal in range(len(self.signals)):
            maximum, minimum, mean, median, std = amplitude_envelope(signal=self.signals.iloc[signal, :],
                                                                     frame_size=self.frame_size,
                                                                     hop_size=self.hop_size,
                                                                     plot=False,
                                                                     des_stats=True)
            ae_max.append(maximum)
            ae_min.append(minimum)
            ae_mean.append(mean)
            ae_median.append(median)
            ae_std.append(std)

        # Create a Date Frame for the Descriptive Statistics of the Amplitude Envelopes for all signals:
        ae_dataframe = pd.DataFrame({'ae_max': ae_max,
                                     'ae_min': ae_min,
                                     'ae_mean': ae_mean,
                                     'ae_median': ae_median,
                                     'ae_std': ae_std})

        return ae_dataframe

    def extract_root_mean_square(self):
        """Extract Root Mean Square Energy of each frame for all signals and return the Descriptive Statistics in
        a DataFrame"""
        rm_max, rm_min, rm_mean, rm_median, rm_std = [], [], [], [], []
        for signal in range(len(self.signals)):
            maximum, minimum, mean, median, std = root_mean_square(signal=self.signals.iloc[signal, :],
                                                                   frame_size=self.frame_size,
                                                                   hop_size=self.hop_size,
                                                                   plot=False,
                                                                   des_stats=True)
            rm_max.append(maximum)
            rm_min.append(minimum)
            rm_mean.append(mean)
            rm_median.append(median)
            rm_std.append(std)

        # Create a Date Frame for the Descriptive Statistics of the Root Mean Square Energies for all signals:
        rm_dataframe = pd.DataFrame({'rm_max': rm_max,
                                     'rm_min': rm_min,
                                     'rm_mean': rm_mean,
                                     'rm_median': rm_median,
                                     'rm_std': rm_std})
        return rm_dataframe

    def extract_zero_crossing_rate(self):
        """Extract Zero-Crossing Rate of each frame for all signals and extract Zero-Crossing Rate of the complete
        Signal Return a DataFrame."""
        zcr, zcr_max, zcr_min, zcr_mean, zcr_median, zcr_std = [], [], [], [], [], []
        # Zero-Crossing Rate per Frame:
        for signal in range(len(self.signals)):
            maximum, minimum, mean, median, std = zero_crossing_rate(signal=self.signals.iloc[signal, :],
                                                                     frames=True,
                                                                     frame_size=self.frame_size,
                                                                     hop_size=self.hop_size,
                                                                     plot=False,
                                                                     des_stats=True)
            zcr_max.append(maximum)
            zcr_min.append(minimum)
            zcr_mean.append(mean)
            zcr_median.append(median)
            zcr_std.append(std)

        # Zero-Crossing Rate for complete signal:
        for signal in range(len(self.signals)):
            signal_zcr = zero_crossing_rate(signal=self.signals.iloc[signal, :],
                                            frames=False,
                                            frame_size=self.frame_size,
                                            hop_size=self.hop_size,
                                            plot=False,
                                            des_stats=False)
            zcr.append(signal_zcr)

        # Create a Date Frame for the Descriptive Statistics of the Zero-Crossing Ratio for all signals:
        zcr_dataframe = pd.DataFrame({'zcr': zcr,
                                      'zcr_max': zcr_max,
                                      'zcr_min': zcr_min,
                                      'zcr_mean': zcr_mean,
                                      'zcr_median': zcr_median,
                                      'zcr_std': zcr_std})

        return zcr_dataframe

    # Functions for extracting Frequency-Domain Features for all Signals:
    def extract_peak_frequency(self):
        """Extract Peak Amplitude and Peak Frequency for all signals and return a DataFrame"""
        peak_amplitude, peak_freq = [], []
        for signal in range(len(self.signals)):
            pa, pf = peak_frequency(signal=self.signals.iloc[signal, :],
                                    sr=self.sr,
                                    plot=False)
            peak_amplitude.append(pa)
            peak_freq.append(pf)

        # Create a Date Frame for Peak Amplitudes and Peak Frequencies for all signals:
        pa_pf_dataframe = pd.DataFrame({'peak_amplitude': peak_amplitude,
                                        'peak_frequency': peak_freq})

        return pa_pf_dataframe

    def extract_band_energy_ratio(self):
        """Extract Band Energy Ratio of each frame for all signals and return the Descriptive Statistics in
        a DataFrame"""
        ber_max, ber_min, ber_mean, ber_median, ber_std = [], [], [], [], []
        for signal in range(len(self.signals)):
            maximum, minimum, mean, median, std = band_energy_ratio(signal=self.signals.iloc[signal, :],
                                                                    sr=self.sr,
                                                                    frame_size=self.frame_size,
                                                                    hop_size=self.hop_size,
                                                                    split_frequency=self.split_frequency,
                                                                    plot=False,
                                                                    des_stats=True)
            ber_max.append(maximum)
            ber_min.append(minimum)
            ber_mean.append(mean)
            ber_median.append(median)
            ber_std.append(std)

        # Create a Date Frame for the Descriptive Statistics of Band Energy Ratio for all signals:
        ber_dataframe = pd.DataFrame({'ber_max': ber_max,
                                      'ber_min': ber_min,
                                      'ber_mean': ber_mean,
                                      'ber_median': ber_median,
                                      'ber_std': ber_std})

        return ber_dataframe

    def extract_spectral_centroid(self):
        """Extract Spectral Centroid of each frame for all signals and return the Descriptive Statistics in
        a DataFrame"""
        sc_max, sc_min, sc_mean, sc_median, sc_std = [], [], [], [], []
        for signal in range(len(self.signals)):
            maximum, minimum, mean, median, std = spectral_centroid(signal=self.signals.iloc[signal, :],
                                                                    sr=self.sr,
                                                                    frame_size=self.frame_size,
                                                                    hop_size=self.hop_size,
                                                                    plot=False,
                                                                    des_stats=True)
            sc_max.append(maximum)
            sc_min.append(minimum)
            sc_mean.append(mean)
            sc_median.append(median)
            sc_std.append(std)

        # Create a Date Frame for the Descriptive Statistics of Spectral Centroid for all signals:
        sc_dataframe = pd.DataFrame({'sc_max': sc_max,
                                     'sc_min': sc_min,
                                     'sc_mean': sc_mean,
                                     'sc_median': sc_median,
                                     'sc_std': sc_std})

        return sc_dataframe

    def extract_spectral_bandwidth(self):
        """Extract Spectral Centroid of each frame for all signals and return the Descriptive Statistics in
        a DataFrame"""
        sb_max, sb_min, sb_mean, sb_median, sb_std = [], [], [], [], []
        for signal in range(len(self.signals)):
            maximum, minimum, mean, median, std = spectral_bandwidth(signal=self.signals.iloc[signal, :],
                                                                     sr=self.sr,
                                                                     frame_size=self.frame_size,
                                                                     hop_size=self.hop_size,
                                                                     plot=False,
                                                                     des_stats=True)
            sb_max.append(maximum)
            sb_min.append(minimum)
            sb_mean.append(mean)
            sb_median.append(median)
            sb_std.append(std)

        # Create a Date Frame for the Descriptive Statistics of Spectral Bandwidth for all signals:
        sb_dataframe = pd.DataFrame({'sb_max': sb_max,
                                     'sb_min': sb_min,
                                     'sb_mean': sb_mean,
                                     'sb_median': sb_median,
                                     'sb_std': sb_std})

        return sb_dataframe

    # Functions for extracting Time-Frequency-Domain Features for all Signals:
    def extract_mfccs(self):
        """Extract mfccs, delta_1 and delta_2 for all signals and return the Descriptive Statistics in
        a DataFrame"""
        # MFCCs:
        mfcc_max, mfcc_min, mfcc_mean, mfcc_median, mfcc_std = [], [], [], [], []
        for signal in range(len(self.signals)):
            maximum, minimum, mean,\
                median, std = mel_frequency_cepstral_coefficients(signal=self.signals.iloc[signal, :],
                                                                  sr=self.sr,
                                                                  n_mfcc=self.n_mfcc,
                                                                  mfcc_type='mfccs',
                                                                  plot=False,
                                                                  save=False,
                                                                  des_stats=True)
            mfcc_max.append(maximum)
            mfcc_min.append(minimum)
            mfcc_mean.append(mean)
            mfcc_median.append(median)
            mfcc_std.append(std)

        # Delta_1:
        delta1_max, delta1_min, delta1_mean, delta1_median, delta1_std = [], [], [], [], []
        for signal in range(len(self.signals)):
            maximum, minimum, mean, \
                median, std = mel_frequency_cepstral_coefficients(signal=self.signals.iloc[signal, :],
                                                                  sr=self.sr,
                                                                  n_mfcc=self.n_mfcc,
                                                                  mfcc_type='delta_1',
                                                                  plot=False,
                                                                  save=False,
                                                                  des_stats=True)
            delta1_max.append(maximum)
            delta1_min.append(minimum)
            delta1_mean.append(mean)
            delta1_median.append(median)
            delta1_std.append(std)

        # Delta_2:
        delta2_max, delta2_min, delta2_mean, delta2_median, delta2_std = [], [], [], [], []
        for signal in range(len(self.signals)):
            maximum, minimum, mean, \
                median, std = mel_frequency_cepstral_coefficients(signal=self.signals.iloc[signal, :],
                                                                  sr=self.sr,
                                                                  n_mfcc=self.n_mfcc,
                                                                  mfcc_type='delta_2',
                                                                  plot=False,
                                                                  save=False,
                                                                  des_stats=True)
            delta2_max.append(maximum)
            delta2_min.append(minimum)
            delta2_mean.append(mean)
            delta2_median.append(median)
            delta2_std.append(std)

        # Create a Date Frame for the Descriptive Statistics of all MFCCs Coefficients for all signals:
        mfcc_dataframe = pd.DataFrame({'mfcc_max': mfcc_max,
                                       'mfcc_min': mfcc_min,
                                       'mfcc_mean': mfcc_mean,
                                       'mfcc_median': mfcc_median,
                                       'mfcc_std': mfcc_std,
                                       'delta_1_max': delta1_max,
                                       'delta_1_min': delta1_min,
                                       'delta_1_mean': delta1_mean,
                                       'delta_1_median': delta1_median,
                                       'delta_1_std': delta1_std,
                                       'delta_2_max': delta2_max,
                                       'delta_2_min': delta2_min,
                                       'delta_2_mean': delta2_mean,
                                       'delta_2_median': delta2_median,
                                       'delta_2_std': delta2_std})
        return mfcc_dataframe

    def extract_dwt_coefficients(self):
        """Extract discrete Wavelet-Transform Coefficients for all signals and return the Descriptive Statistics in
        a DataFrame"""
        ca_max, ca_min, ca_mean, ca_median, ca_std = [], [], [], [], []
        cd_max, cd_min, cd_mean, cd_median, cd_std = [], [], [], [], []
        for signal in range(len(self.signals)):
            maximum_ca, minimum_ca, mean_ca, median_ca, std_ca, \
                maximum_cd, minimum_cd, mean_cd, median_cd, std_cd = dwt_coefficients(signal=self.signals.iloc[signal, :],
                                                                                      dwt_levels=False,
                                                                                      plot=False,
                                                                                      des_stats=True)
            ca_max.append(maximum_ca)
            ca_min.append(minimum_ca)
            ca_mean.append(mean_ca)
            ca_median.append(median_ca)
            ca_std.append(std_ca)
            cd_max.append(maximum_cd)
            cd_min.append(minimum_cd)
            cd_mean.append(mean_cd)
            cd_median.append(median_cd)
            cd_std.append(std_cd)

        # Create a Date Frame for the Descriptive Statistics of Spectral Bandwidth for all signals:
        dwt_dataframe = pd.DataFrame({'ca_max': ca_max,
                                      'ca_min': ca_min,
                                      'ca_mean': ca_mean,
                                      'ca_median': ca_median,
                                      'ca_std': ca_std,
                                      'cd_max': cd_max,
                                      'cd_min': cd_min,
                                      'cd_mean': cd_mean,
                                      'cd_median': cd_median,
                                      'cd_std': cd_std})

        return dwt_dataframe


# A class to extract Time-Frequency Domain Representation:
class TimeFrequencyRepresentation:
    def __init__(self, signals, signal_references, sr, frame_size, hop_size, rep_path, feature_extraction_path,
                 csv_version):
        self.signals = signals
        self.signal_references = signal_references
        self.sr = sr
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.rep_path = rep_path
        self.feature_extraction_path = feature_extraction_path
        self.csv_version = csv_version

    def extract_spectrogram(self):
        """Function to extract Spectrogram and save it in 'feature_extracted' directory"""
        try:
            os.path.exists(f'{self.rep_path}{self.feature_extraction_path}/images/spectrogram')
        except Exception as e:
            print(e)
        else:
            for signal in range(len(self.signals)):
                spectrogram(signal=self.signals.iloc[signal],
                            sr=self.sr,
                            frame_size=self.frame_size,
                            hop_size=self.hop_size,
                            plot=False,
                            save=True,
                            img_ref=f"{self.signal_references.iloc[signal]['signal_id']}")

            # save signal_references:
            while os.path.isfile(
                    f'{self.rep_path}{self.feature_extraction_path}/images/spectrogram/signal_references_v{self.csv_version}.csv'):
                self.csv_version += 1
                continue
            else:
                self.signal_references.to_csv(f'{self.rep_path}{self.feature_extraction_path}/images/spectrogram/signal_references_v{self.csv_version}.csv',
                                              index=False)
            return None

    def extract_mel_spectrogram(self, n_mels):
        """Function to extract mel_spectrogram and save it in 'feature_extracted' directory"""
        try:
            os.path.exists(f'{self.rep_path}{self.feature_extraction_path}/images/mel_spectrogram')
        except Exception as e:
            print(e)
        else:
            for signal in range(len(self.signals)):
                mel_spectrogram(signal=self.signals.iloc[signal],
                                sr=self.sr,
                                frame_size=self.frame_size,
                                hop_size=self.hop_size,
                                n_mels=n_mels,
                                plot=False,
                                save=True,
                                img_ref=f"{self.signal_references.iloc[signal]['signal_id']}")

            # save signal_references:
            while os.path.isfile(
                    f'{self.rep_path}{self.feature_extraction_path}/images/mel_spectrogram/signal_references_v{self.csv_version}.csv'):
                self.csv_version += 1
                continue
            else:
                self.signal_references.to_csv(f'{self.rep_path}{self.feature_extraction_path}/images/mel_spectrogram/signal_references_v{self.csv_version}.csv',
                                              index=False)
            return None

    def extract_mfccs(self, mfcc_type, n_mfcc):
        """Function to extract and save mfccs based on its type as an image in 'feature_extracted' directory and
        save the signal_reference as csv in the same directory.
        mfcc_type is either "mfccs", "delta_1", "delta_2"."""
        # mfccs:
        if mfcc_type == 'mfccs':
            try:
                os.path.exists(f'{self.rep_path}{self.feature_extraction_path}/images/mfccs/mfcc')
            except Exception as e:
                print(e)
            else:
                for signal in range(len(self.signals)):
                    mel_frequency_cepstral_coefficients(signal=self.signals.iloc[signal],
                                                        sr=self.sr,
                                                        mfcc_type=mfcc_type,
                                                        n_mfcc=n_mfcc,
                                                        plot=False,
                                                        save=True,
                                                        img_ref=f"{self.signal_references.iloc[signal]['signal_id']}")

                # save signal_references:
                while os.path.isfile(f'{self.rep_path}{self.feature_extraction_path}/images/mfccs/mfcc/signal_references_v{self.csv_version}.csv'):
                    self.csv_version += 1
                    continue
                else:
                    self.signal_references.to_csv(f'{self.rep_path}{self.feature_extraction_path}/images/mfccs/mfcc/signal_references_v{self.csv_version}.csv',
                                                  index=False)
            return None

        # delta_1:
        elif mfcc_type == 'delta_1':
            try:
                os.path.exists(f'{self.rep_path}{self.feature_extraction_path}/images/mfccs/delta_1')
            except Exception as e:
                print(e)
            else:
                for signal in range(len(self.signals)):
                    mel_frequency_cepstral_coefficients(signal=self.signals.iloc[signal],
                                                        sr=self.sr,
                                                        mfcc_type=mfcc_type,
                                                        n_mfcc=n_mfcc,
                                                        plot=False,
                                                        save=True,
                                                        img_ref=f"{self.signal_references.iloc[signal]['signal_id']}")

                # Save signal_references:
                while os.path.isfile(f'{self.rep_path}{self.feature_extraction_path}/images/mfccs/delta_1/signal_references_v{self.csv_version}.csv'):
                    self.csv_version += 1
                    continue
                else:
                    self.signal_references.to_csv(f'{self.rep_path}{self.feature_extraction_path}/images/mfccs/delta_1/signal_references_v{self.csv_version}.csv',
                                                  index=False)
            return None

        # delta_2
        elif mfcc_type == 'delta_2':
            try:
                os.path.exists(f'{self.rep_path}{self.feature_extraction_path}/images/mfccs/delta_2')
            except Exception as e:
                print(e)
            else:
                for signal in range(len(self.signals)):
                    mel_frequency_cepstral_coefficients(signal=self.signals.iloc[signal],
                                                        sr=self.sr,
                                                        mfcc_type=mfcc_type,
                                                        n_mfcc=n_mfcc,
                                                        plot=False,
                                                        save=True,
                                                        img_ref=f"{self.signal_references.iloc[signal]['signal_id']}")

                # Save signal_references:
                while os.path.isfile(f'{self.rep_path}{self.feature_extraction_path}/images/mfccs/delta_2/signal_references_v{self.csv_version}.csv'):
                    self.csv_version += 1
                    continue
                else:
                    self.signal_references.to_csv(f'{self.rep_path}{self.feature_extraction_path}/images/mfccs/delta_2/signal_references_v{self.csv_version}.csv',
                                                  index=False)
            return None

    def extract_cwt_scalogram(self, num_scales, wavelet_family):
        """Function to extract cwt_scalogram and save it in 'feature_extracted' directory"""
        try:
            os.path.exists(f'{self.rep_path}{self.feature_extraction_path}/images/scalogram')
        except Exception as e:
            print(e)
        else:
            for signal in range(len(self.signals)):
                cwt_scalogram(signal=self.signals.iloc[signal],
                              num_scales=num_scales,
                              wavelet_family=wavelet_family,
                              plot=False,
                              save=True,
                              img_ref=f"{self.signal_references.iloc[signal]['signal_id']}")

            # Save signal_references:
            while os.path.isfile(
                    f'{self.rep_path}{self.feature_extraction_path}/images/scalogram/signal_references_v{self.csv_version}.csv'):
                self.csv_version += 1
                continue
            else:
                self.signal_references.to_csv(
                    f'{self.rep_path}{self.feature_extraction_path}/images/scalogram/signal_references_v{self.csv_version}.csv',
                    index=False)
        return None


if __name__ == "__main__":
    # Check directories and create if not existed:
    check_create_dir()

    # Importing Signals and References in DataFrames:
    SIGNALS, SIGNAL_REFERENCES = access_signals(denoise_method='wavelet_transform')

    # 1. Extracting Numeric Data:
    FEATURES = NumericFeatureExtraction(signals=SIGNALS,
                                        sr=c.SAMPLING_RATE,
                                        frame_size=c.FRAME_SIZE,
                                        hop_size=c.HOP_SIZE,
                                        split_frequency=c.SPLIT_FREQUENCY,
                                        n_mfcc=c.N_MFCCS)

    # Time-Domain Features:
    DESCRIPTIVE_STATISTICS = FEATURES.extract_descriptive_statistics()
    ENERGY_POWER = FEATURES.extract_energy_power()
    AMPLITUDE_ENVELOPE = FEATURES.extract_amplitude_envelope()
    ROOT_MEAN_SQUARE = FEATURES.extract_root_mean_square()
    ZERO_CROSSING_RATE = FEATURES.extract_zero_crossing_rate()

    # Fequency-Domain Features:
    PEAK_AMPLITUDE_FREQUENCY = FEATURES.extract_peak_frequency()
    BAND_ENERGY_RATIO = FEATURES.extract_band_energy_ratio()
    SPECTRAL_CENTRIOD = FEATURES.extract_spectral_centroid()
    SPECTRAL_BANDWIDTH = FEATURES.extract_spectral_bandwidth()

    # Time-Fequency-Domain Features:
    MFCC = FEATURES.extract_mfccs()
    DISCRETE_WAVELET_TRANSFORM = FEATURES.extract_dwt_coefficients()

    # Concatenating all Features in a single DataFrame:
    DATAFRAMES = concatenate_dataframes(DESCRIPTIVE_STATISTICS,
                                        ENERGY_POWER,
                                        AMPLITUDE_ENVELOPE,
                                        ROOT_MEAN_SQUARE,
                                        ZERO_CROSSING_RATE,
                                        PEAK_AMPLITUDE_FREQUENCY,
                                        BAND_ENERGY_RATIO,
                                        SPECTRAL_CENTRIOD,
                                        SPECTRAL_BANDWIDTH,
                                        MFCC,
                                        DISCRETE_WAVELET_TRANSFORM,
                                        SIGNAL_REFERENCES)

    # Saving all Numeric Data in a csv file:
    save_features(dataframe=DATAFRAMES, csv_version=1)

    # Time-Frequency Representation Features:
    IMAGES = TimeFrequencyRepresentation(signals=SIGNALS,
                                         signal_references=SIGNAL_REFERENCES,
                                         sr=c.SAMPLING_RATE,
                                         frame_size=c.FRAME_SIZE,
                                         hop_size=c.HOP_SIZE,
                                         rep_path=c.REPO_PATH,
                                         feature_extraction_path=c.FEATURE_EXTRACTION_PATH,
                                         csv_version=1)

    IMAGES.extract_spectrogram()
    IMAGES.extract_mel_spectrogram(n_mels=c.N_MELS)
    IMAGES.extract_mfccs(mfcc_type='delta_2', n_mfcc=c.N_MFCCS)
    IMAGES.extract_cwt_scalogram(num_scales=c.N_SCALES, wavelet_family='gaus1')

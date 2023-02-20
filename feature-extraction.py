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


def _check_create_dir():
    """Function to check if the img and csv directories exist in the data directory and create
    them if they are not existed"""
    # 1. SIG_PRE_PATH directory:
    if not os.path.exists(c.SIG_PRE_PATH):
        os.makedirs(c.SIG_PRE_PATH)

    signal_directories = ['original_signals', 'emd', 'wavelet_transform', 'digital_filters',
                          'emd_wavelet', 'emd_dfilters', 'emd_wl_dfilters']

    normalized_dirs = ['normalized', 'denormalized']
    for normalized_dir in normalized_dirs:
        subdirectories = [os.path.join(normalized_dir, directory) for directory in signal_directories]
        for subdirectory in subdirectories:
            if not os.path.isdir(os.path.join(c.SIG_PRE_PATH, subdirectory)):
                os.makedirs(os.path.join(c.SIG_PRE_PATH, subdirectory), exist_ok=True)

    # 2. FEATURE_EXTRACTION_PATH directory:
    # 2.1. Checking if FEATURE_EXTRACTION_PATH directory exists:
    if not os.path.exists(c.FEATURE_EXTRACTION_PATH):
        os.makedirs(c.FEATURE_EXTRACTION_PATH)

    # 2.2. Check if csv directory and its subdirectories exist:
    csv_path = os.path.join(c.FEATURE_EXTRACTION_PATH, 'csv')
    if not os.path.isdir(csv_path):
        os.makedirs(csv_path)

    csv_subdirs = [os.path.join(csv_path, normalized_dir) for normalized_dir in normalized_dirs]
    for csv_subdir in csv_subdirs:
        if not os.path.isdir(csv_subdir):
            os.makedirs(csv_subdir, exist_ok=True)

    # 2.3. Check if images directory and its subdirectories exist:
    image_path = os.path.join(c.FEATURE_EXTRACTION_PATH, 'images')
    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    image_features = ['spectrogram', 'mel_spectrogram', 'mfccs', 'scalogram']
    mfccs = ['mfcc', 'delta_1', 'delta_2']
    for image_feature in image_features:
        image_feature_path = os.path.join(image_path, image_feature)
        if not os.path.isdir(image_feature_path):
            os.makedirs(image_feature_path)
    for mfccs_type in mfccs:
        mfccs_path = os.path.join(image_path, 'mfccs', mfccs_type)
        if not os.path.isdir(mfccs_path):
            os.makedirs(mfccs_path)

    return None


def _access_signals(denoise_method, normalization: str):
    """Function to access the latest version of the preprocessed signals.
    the denoise_method is either 'emd', 'wavelet_transform', 'digital_filters', 'emd_wavelet',
    'emd_dfilters' or 'emd_wl_dfilters'. Normalization is either 'normalized' or 'denormalized'.
    """
    # Searching and reading the latest csv version based on with 'denoising method':
    if normalization not in ['normalized', 'denormalized']:
        raise ValueError("'normalization is with 'normalized' or 'denormalized'.")
    else:
        list_of_versions = glob(f'{c.SIG_PRE_PATH}/{normalization}/{denoise_method}/*.csv')
        latest_version = max(list_of_versions, key=os.path.getctime)
        audio_signals = pd.read_csv(latest_version)

    # Generating signal_ids:
    audio_signals = audio_signals.assign(signal_id=[f'signal_{number}' for number in range(len(audio_signals))])

    # Separate the signals from signal_id and class:
    signal_references = audio_signals.loc[:, ['signal_id', 'class']]
    audio_signals = audio_signals.drop(['signal_id', 'class'], axis='columns')
    return audio_signals, signal_references


def _extract_numeric_features(audio_signals):
    """Function to extract all Numeric Features from all Audio Signals. These Features are
    extracted from Time-Domain, Frequency-Domain and Time-Frequency Representation Domain.
    It returns a DataFrame including all the features."""
    ds_max, ds_min, ds_mean, ds_median, ds_std = [], [], [], [], []
    energies, powers = [], []
    ae_max, ae_min, ae_mean, ae_median, ae_std = [], [], [], [], []
    rm_max, rm_min, rm_mean, rm_median, rm_std = [], [], [], [], []
    zcr_max, zcr_min, zcr_mean, zcr_median, zcr_std = [], [], [], [], []
    zcr = []
    peak_amplitude, peak_freq = [], []
    ber_max, ber_min, ber_mean, ber_median, ber_std = [], [], [], [], []
    sc_max, sc_min, sc_mean, sc_median, sc_std = [], [], [], [], []
    sb_max, sb_min, sb_mean, sb_median, sb_std = [], [], [], [], []
    mfcc_max, mfcc_min, mfcc_mean, mfcc_median, mfcc_std = [], [], [], [], []
    delta1_max, delta1_min, delta1_mean, delta1_median, delta1_std = [], [], [], [], []
    delta2_max, delta2_min, delta2_mean, delta2_median, delta2_std = [], [], [], [], []
    ca_max, ca_min, ca_mean, ca_median, ca_std = [], [], [], [], []
    cd_max, cd_min, cd_mean, cd_median, cd_std = [], [], [], [], []

    for index in range(len(audio_signals)):
        # 1. Time Domain Features:
        signal_time_domain = TimeDomainFeatures(audio_signal=audio_signals.iloc[index, :],
                                                frame_size=c.FRAME_SIZE,
                                                hop_size=c.HOP_SIZE)
        # 1.1. Descriptive Statistics:
        maximum, minimum, mean, median, std = signal_time_domain.extract_descriptive_statistics()
        ds_max.append(maximum)
        ds_min.append(minimum)
        ds_mean.append(mean)
        ds_median.append(median)
        ds_std.append(std)

        # 1.2. Energy and Total Power:
        energy, power = signal_time_domain.extract_energy_power()
        energies.append(energy)
        powers.append(power)

        # 1.3. Amplitude Envelope:
        maximum, minimum, mean, median, std = signal_time_domain.extract_amplitude_envelope(plot=False, des_stats=True)
        ae_max.append(maximum)
        ae_min.append(minimum)
        ae_mean.append(mean)
        ae_median.append(median)
        ae_std.append(std)

        # 1.4. Root Mean Square Energy:
        maximum, minimum, mean, median, std = signal_time_domain.extract_root_mean_square(plot=False, des_stats=True)
        rm_max.append(maximum)
        rm_min.append(minimum)
        rm_mean.append(mean)
        rm_median.append(median)
        rm_std.append(std)

        # 1.5. Zero-Crossing Rate per Frame:
        maximum, minimum, mean, median, std = signal_time_domain.extract_root_mean_square(plot=False, des_stats=True)
        zcr_max.append(maximum)
        zcr_min.append(minimum)
        zcr_mean.append(mean)
        zcr_median.append(median)
        zcr_std.append(std)

        # 1.6. Zero-Crossing Rate for complete signal:
        signal_zcr = signal_time_domain.extract_zero_crossing_rate(plot=False, des_stats=False)
        zcr.append(signal_zcr)

        # 2. Frequency-Domain Features:
        signal_freq_domain = FrequencyDomainFeatures(audio_signal=audio_signals.iloc[index, :-1],
                                                     sr=c.SAMPLING_RATE,
                                                     frame_size=c.FRAME_SIZE,
                                                     hop_size=c.HOP_SIZE)
        # 2.1. Spectrum Features:
        pa, pf = signal_freq_domain.extract_spectrum_features(plot=False)
        peak_amplitude.append(pa)
        peak_freq.append(pf)

        # 2.2. Band Energy Ratio:
        maximum, minimum, mean, median, std = signal_freq_domain.extract_band_energy_ratio(
            split_frequency=c.SPLIT_FREQUENCY,
            plot=False,
            des_stats=True)
        ber_max.append(maximum)
        ber_min.append(minimum)
        ber_mean.append(mean)
        ber_median.append(median)
        ber_std.append(std)

        # 2.3. Spectral Centriod:
        maximum, minimum, mean, median, std = signal_freq_domain.extract_spectral_centroid(plot=False,
                                                                                           des_stats=True)

        sc_max.append(maximum)
        sc_min.append(minimum)
        sc_mean.append(mean)
        sc_median.append(median)
        sc_std.append(std)

        # 2.4. Spectral Bandwidth:
        maximum, minimum, mean, median, std = signal_freq_domain.extract_spectral_bandwidth(plot=False,
                                                                                            des_stats=True)
        sb_max.append(maximum)
        sb_min.append(minimum)
        sb_mean.append(mean)
        sb_median.append(median)
        sb_std.append(std)

        # 3. Time-Frequency Representation Domain:
        signal_time_freq_domain = TimeFrequencyDomainFeatures(audio_signal=audio_signals.iloc[index, :-1],
                                                              sr=c.SAMPLING_RATE,
                                                              frame_size=c.FRAME_SIZE,
                                                              hop_size=c.HOP_SIZE)
        # 3.1. MFCCs:
        # 3.1.1. mfccs_type = 'mfccs':
        maximum, minimum, mean, median, std = signal_time_freq_domain.extract_mfccs(n_mfcc=c.N_MFCCS,
                                                                                    mfcc_type='mfccs',
                                                                                    plot=False,
                                                                                    save=False,
                                                                                    des_stats=True)
        mfcc_max.append(maximum)
        mfcc_min.append(minimum)
        mfcc_mean.append(mean)
        mfcc_median.append(median)
        mfcc_std.append(std)

        # 3.1.2. mfccs_type = 'delta_1':
        maximum, minimum, mean, median, std = signal_time_freq_domain.extract_mfccs(n_mfcc=c.N_MFCCS,
                                                                                    mfcc_type='delta_1',
                                                                                    plot=False,
                                                                                    save=False,
                                                                                    des_stats=True)
        delta1_max.append(maximum)
        delta1_min.append(minimum)
        delta1_mean.append(mean)
        delta1_median.append(median)
        delta1_std.append(std)

        # 3.1.3. mfccs_type = 'delta_2':
        maximum, minimum, mean, median, std = signal_time_freq_domain.extract_mfccs(n_mfcc=c.N_MFCCS,
                                                                                    mfcc_type='delta_2',
                                                                                    plot=False,
                                                                                    save=False,
                                                                                    des_stats=True)
        delta2_max.append(maximum)
        delta2_min.append(minimum)
        delta2_mean.append(mean)
        delta2_median.append(median)
        delta2_std.append(std)

        # 3.2. Discrete Wavelet-Transform coefficients:
        maximum_ca, minimum_ca, mean_ca, \
            median_ca, std_ca, \
            maximum_cd, minimum_cd, mean_cd, median_cd, \
            std_cd = signal_time_freq_domain.extract_dwt_coefficients(dwt_levels=False,
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

    # Create a Date Frame for all extracted features:
    ds_dataframe = pd.DataFrame(
        {'ds_max': ds_max, 'ds_min': ds_min, 'ds_mean': ds_mean, 'ds_median': ds_median, 'ds_std': ds_std,
         'energy': energies, 'power': powers,
         'ae_max': ae_max, 'ae_min': ae_min, 'ae_mean': ae_mean, 'ae_median': ae_median, 'ae_std': ae_std,
         'rm_max': rm_max, 'rm_min': rm_min, 'rm_mean': rm_mean, 'rm_median': rm_median, 'rm_std': rm_std,
         'zcr': zcr, 'zcr_max': zcr_max, 'zcr_min': zcr_min, 'zcr_mean': zcr_mean, 'zcr_median': zcr_median,
         'zcr_std': zcr_std,
         'peak_amplitude': peak_amplitude, 'peak_frequency': peak_freq,
         'ber_max': ber_max, 'ber_min': ber_min, 'ber_mean': ber_mean, 'ber_median': ber_median, 'ber_std': ber_std,
         'sc_max': sc_max, 'sc_min': sc_min, 'sc_mean': sc_mean, 'sc_median': sc_median, 'sc_std': sc_std,
         'sb_max': sb_max, 'sb_min': sb_min, 'sb_mean': sb_mean, 'sb_median': sb_median, 'sb_std': sb_std,
         'mfcc_max': mfcc_max, 'mfcc_min': mfcc_min, 'mfcc_mean': mfcc_mean, 'mfcc_median': mfcc_median,
         'mfcc_std': mfcc_std,
         'delta_1_max': delta1_max, 'delta_1_min': delta1_min, 'delta_1_mean': delta1_mean,
         'delta_1_median': delta1_median, 'delta_1_std': delta1_std,
         'delta_2_max': delta2_max, 'delta_2_min': delta2_min, 'delta_2_mean': delta2_mean,
         'delta_2_median': delta2_median, 'delta_2_std': delta2_std,
         'ca_max': ca_max, 'ca_min': ca_min, 'ca_mean': ca_mean, 'ca_median': ca_median, 'ca_std': ca_std,
         'cd_max': cd_max, 'cd_min': cd_min, 'cd_mean': cd_mean, 'cd_median': cd_median, 'cd_std': cd_std})

    return ds_dataframe


def _save_features(dataframe, csv_version: int, normalization: str):
    """Function to save DataFrame into csv in the 'extracted_feature' directory."""
    if normalization not in ['normalized', 'denormalized']:
        raise ValueError("normalization must be either 'normalized' or 'denormalized'.")
    try:
        os.path.exists(f'{c.FEATURE_EXTRACTION_PATH}/csv/{normalization}')
    except Exception as e:
        print(e)
    else:
        while os.path.isfile(f'{c.FEATURE_EXTRACTION_PATH}/csv/{normalization}/extracted_features_v{csv_version}.csv'):
            csv_version += 1
            continue
        else:
            dataframe.to_csv(f'{c.FEATURE_EXTRACTION_PATH}/csv/{normalization}/extracted_features_v{csv_version}.csv',
                             index=False)
    return None


def _extract_save_images(audio_signals, references, rep_type: str):
    """Function to extract all images from all Audio Signals. These Features are extracted from
    Time-Frequency Representation Domain. It saves the images and references in
    'data/extracted_features/images'."""
    try:
        os.path.exists(f'{c.FEATURE_EXTRACTION_PATH}/images')
    except Exception as e:
        print(e)
    else:
        for index in range(len(audio_signals)):
            signal_time_freq_domain = TimeFrequencyDomainFeatures(audio_signal=audio_signals.iloc[index, :].astype(np.float32),
                                                                  sr=c.SAMPLING_RATE,
                                                                  frame_size=c.FRAME_SIZE,
                                                                  hop_size=c.HOP_SIZE)
            # 1. Spectrogram:
            if rep_type == 'spectrogram':
                signal_time_freq_domain.extract_spectrogram(save=True,
                                                            img_ref=references.loc[index, 'signal_id'])

            # 2. Mel-Spectrogram:
            elif rep_type == 'mel_spectrogram':
                signal_time_freq_domain.extract_mel_spectrogram(n_mels=c.N_MELS,
                                                                save=True,
                                                                img_ref=references.loc[index, 'signal_id'])

            # 3. MFCCS:
            # 3.1. MFCCS:
            elif rep_type == 'mfccs':
                signal_time_freq_domain.extract_mfccs(n_mfcc=c.N_MFCCS,
                                                      mfcc_type='mfccs',
                                                      save=True,
                                                      img_ref=references.loc[index, 'signal_id'])
            # 3.2. delta_1:
            elif rep_type == 'delta_1':
                signal_time_freq_domain.extract_mfccs(n_mfcc=c.N_MFCCS,
                                                      mfcc_type='delta_1',
                                                      save=True,
                                                      img_ref=references.loc[index, 'signal_id'])
            # 3.3. delta_2:
            elif rep_type == 'delta_2':
                signal_time_freq_domain.extract_mfccs(n_mfcc=c.N_MFCCS,
                                                      mfcc_type='delta_2',
                                                      save=True,
                                                      img_ref=references.loc[index, 'signal_id'])

            # 4. CWT-Scalogram:
            elif rep_type == 'scalogram':
                signal_time_freq_domain.extract_cwt_scalogram(num_scales=c.N_SCALES,
                                                              wavelet_family='shan',
                                                              save=True,
                                                              img_ref=references.loc[index, 'signal_id'])
            else:
                raise ValueError(f"Invalid representation type: {rep_type}.")

        # Saving the references:
        references.to_csv(f'{c.FEATURE_EXTRACTION_PATH}/images/{rep_type}/references.csv', index=False)
    return None


if __name__ == "__main__":
    _check_create_dir()
    # 1. Extract Numeric Features:
    # 1.1. Normalization:
    NORMALIZED_SIGNALS, REFERENCES = _access_signals('digital_filters', normalization='normalized')
    NORMALIZED_FEATURES = _extract_numeric_features(audio_signals=NORMALIZED_SIGNALS)
    NORMALIZED_FEATURES = NORMALIZED_FEATURES.join(REFERENCES)
    _save_features(dataframe=NORMALIZED_FEATURES, csv_version=1, normalization='normalized')

    # 1.2. Denormalization:
    DENORMALIZED_SIGNALS, REFERENCES = _access_signals('digital_filters', normalization='denormalized')
    DENORMALIZED_FEATURES = _extract_numeric_features(audio_signals=DENORMALIZED_SIGNALS)
    DENORMALIZED_FEATURES = DENORMALIZED_FEATURES.join(REFERENCES)
    _save_features(dataframe=DENORMALIZED_FEATURES, csv_version=1, normalization='denormalized')

    # 2. Extract Images:
    _extract_save_images(audio_signals=DENORMALIZED_SIGNALS, references=REFERENCES, rep_type='scalogram')

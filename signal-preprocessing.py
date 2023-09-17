"""
Author: Mohammed Lajam

Phase 2: Signal Preprocessing:
- In this python file, the audio data is extracted from the 'data/pcg_signals/PhysioNet_2016'
directory and being preprocessed using EMD-Digital Filter function.
- After denoising the signals, Slicing the signals takes place to a specific length, so that all
signals are equal in length and therefore the number of samples per signals are equal.

Output:
- Generating csv files for each method including all the sliced signals with references is saved
in a directory 'data/denoised_signals'

Note:
- The Dataset is downloaded from PhysioNet_2016 Challenge.
- This Repository does not contain of 'data' directory. Hence, create a directory inside the
Repository called 'data'. Inside 'data' directory, create another directory called 'pcg-signals'
and place PhysioNet_2016 data inside it.
- In 'constants.py', REPO_PATH has to be adjusted before running this file.
- All the functions and variables, which are used in this file, are imported from helpers.py
file from the signal_processing package in the same repository.
"""

# Importing libraries:
import numpy as np
import pandas as pd
from glob import glob
import os
import constants as c
from signal_preprocessing.helpers import SignalPreprocessing, extract_signal


# Function to create and check directories:
def _check_create_dir():
    """Function to check if the img and csv directories exist in the data directory and
    create them if they are not existed"""
    # Check if SIG_PRE_PATH directory exists:
    if not os.path.exists(c.SIG_PRE_PATH):
        os.mkdir(c.SIG_PRE_PATH)

    directories = ['original_signals', 'emd', 'wavelet_transform', 'digital_filters',
                   'emd_wavelet', 'emd_dfilters', 'emd_wl_dfilters']
    normalized_dirs = ['normalized', 'denormalized']
    for normalized_dir in normalized_dirs:
        subdirectories = [f'{normalized_dir}/{directory}' for directory in directories]
        for subdirectory in subdirectories:
            if not os.path.isdir(f'{c.SIG_PRE_PATH}/{subdirectory}'):
                os.makedirs(f'{c.SIG_PRE_PATH}/{subdirectory}', exist_ok=True)
    return None


# 1. Function to load all the paths of the audio files in one list:
def _create_signal_paths():
    """Function to create a list of all the Audio paths and its References."""
    # Loading all the paths of the audio files in one list:
    directories = ['a', 'b', 'c', 'd', 'e', 'f']
    audio_files = [glob(f'{c.DATASET_PATH}/training-{directory}/*.wav') for directory in directories]
    audio_files = [item for elem in audio_files for item in elem]
    audio_files = sorted(audio_files)

    # Loading the references:
    references = pd.concat([pd.read_csv(f'{c.DATASET_PATH}/training-{directory}/REFERENCE.csv', header=None) for directory in directories], ignore_index=True)
    references.columns = ['signal_id', 'class']
    return audio_files, references


# 2. Function to extract all signals:
def _extract_signals(audio_files, normalization=True):
    """Function to extract all the signals from Audio paths and joining it references.
    It returns a DataFrame"""
    audio_signals = []
    for index in range(len(audio_files)):
        audio_signal, _ = extract_signal(file_path=audio_files,
                                         audio_index=index,
                                         sr=c.SAMPLING_RATE,
                                         normalization=normalization)
        audio_signals.append(audio_signal)
    return audio_signals


def _denoise_signals(audio_signals, denoise_method: str):
    """Function to denoise all the signals and return a DataFrame. The 'audio_signals' is
    a Numpy Array. The 'denoise_method' is either emd, wavelet_transform, digital_filters,
    emd_wavelet, emd_dfilters or emd_wl_dfilters."""
    denoised_signals = []
    if denoise_method == 'emd':
        for index in range(len(audio_signals)):
            signal = SignalPreprocessing(audio_signal=audio_signals[index], sr=c.SAMPLING_RATE)
            _, denoised_signal = signal.process_emd(n_imf=c.N_IMF)
            denoised_signals.append(denoised_signal[0])

    elif denoise_method == 'wavelet_transform':
        for index in range(len(audio_signals)):
            signal = SignalPreprocessing(audio_signal=audio_signals[index], sr=c.SAMPLING_RATE)
            _, denoised_signal = signal.process_wavelet_denoising()
            denoised_signals.append(denoised_signal)

    elif denoise_method == 'digital_filters':
        for index in range(len(audio_signals)):
            signal = SignalPreprocessing(audio_signal=audio_signals[index], sr=c.SAMPLING_RATE)
            _, denoised_signal = signal.process_digital_filter(order=c.FILTER_ORDER,
                                                               low_fc=c.LOW_FC,
                                                               high_fc=c.HIGH_FC)
            denoised_signals.append(denoised_signal)

    elif denoise_method == 'emd_wavelet':
        for index in range(len(audio_signals)):
            signal = SignalPreprocessing(audio_signal=audio_signals[index], sr=c.SAMPLING_RATE)
            _, denoised_signal = signal.process_emd_wl(n_imf=c.N_IMF)
            denoised_signals.append(denoised_signal)

    elif denoise_method == 'emd_dfilters':
        for index in range(len(audio_signals)):
            signal = SignalPreprocessing(audio_signal=audio_signals[index], sr=c.SAMPLING_RATE)
            _, denoised_signal = signal.process_emd_dfilter(n_imf=c.N_IMF,
                                                            order=c.FILTER_ORDER,
                                                            low_fc=c.LOW_FC,
                                                            high_fc=c.HIGH_FC)
            denoised_signals.append(denoised_signal)

    elif denoise_method == 'emd_wl_dfilters':
        for index in range(len(audio_signals)):
            signal = SignalPreprocessing(audio_signal=audio_signals[index], sr=c.SAMPLING_RATE)
            _, denoised_signal = signal.process_emd_wl_dfilter(n_imf=c.N_IMF,
                                                               order=c.FILTER_ORDER,
                                                               low_fc=c.LOW_FC,
                                                               high_fc=c.HIGH_FC)
            denoised_signals.append(denoised_signal)
    else:
        raise ValueError(f"'{denoise_method}' is not a valid method. The 'denoised_method' is either emd, "
                         f"wavelet_transform, digital_filters, emd_wavelet, emd_dfilters or emd_wl_dfilters")

    return denoised_signals


# Function to Slice the signals:
def _slice_signals(audio_signals, references, sr, period):
    """Function to slice the signals into 5000 features each and return a DataFrame.
    Input: DataFrame."""
    audio_signals = audio_signals.drop(columns=['signal_id', 'class'])
    sliced_signals = []
    for i in range(len(audio_signals)):  # iterating over all the rows in the DataFrame
        start = 0
        end = sr * period
        for j in range(8):  # The number of slices in each row
            signal = pd.DataFrame(audio_signals.iloc[i, start:end]).T
            signal['class'] = references.loc[i, 'class']
            signal['signal_id'] = references.loc[i, 'signal_id']
            sliced_signals.append(np.array(signal))
            start += sr * period
            end += sr * period

    sliced_signals = [item for elem in sliced_signals for item in elem]

    # converting a list to DataFrame and dropping any row that contains NaN:
    sliced_signals = pd.DataFrame(sliced_signals).dropna()
    sliced_signals.reset_index(drop=True, inplace=True)
    sliced_signals = sliced_signals.rename(columns={sr*period: 'class'})
    return sliced_signals


# Function to save dataframes:
def _save_dataframe(dataframe, normalization, denoise_method, csv_version):
    """Function to save the DataFrame into the path of local machine as csv file.
    'normalization'  is either 'normalized' or 'denormalized'
    'denoise_method' is either: 'original_signals', 'emd', 'wavelet_transform',
    'digital_filters', 'emd_wavelet', 'emd_dfilters' or 'emd_wl_dfilters'."""
    directory_path = f'{c.SIG_PRE_PATH}/{normalization}/{denoise_method}'
    try:
        os.path.exists(directory_path)
    except Exception as e:
        print(e)
    else:
        while os.path.isfile(f'{directory_path}/{denoise_method}_v{csv_version}.csv'):
            csv_version += 1
        dataframe.to_csv(f'{directory_path}/{denoise_method}_v{csv_version}.csv', index=False)
    return None


if __name__ == "__main__":
    _check_create_dir()
    AUDIO_FILES, REFERENCES = _create_signal_paths()

    # 1. Normalization:
    # 1.1. Extracting the Original Signals:
    NORMALIZED_ORIGINAL_SIGNALS = _extract_signals(audio_files=AUDIO_FILES,
                                                   normalization=True)

    # 1.2. Denoising:
    NORMALIZED_DFILTERS_SIGNALS = _denoise_signals(audio_signals=NORMALIZED_ORIGINAL_SIGNALS,
                                                   denoise_method='digital_filters')

    # 1.3. Create DataFrames and References:
    NORMALIZED_ORIGINAL_SIGNALS = pd.DataFrame(NORMALIZED_ORIGINAL_SIGNALS)
    NORMALIZED_ORIGINAL_SIGNALS = NORMALIZED_ORIGINAL_SIGNALS.join(REFERENCES)

    NORMALIZED_DFILTERS_SIGNALS = pd.DataFrame(NORMALIZED_DFILTERS_SIGNALS)
    NORMALIZED_DFILTERS_SIGNALS = NORMALIZED_DFILTERS_SIGNALS.join(REFERENCES)

    # 1.4. Slicing DataFrames:
    SLICED_NORMALIZED_DFILTERS_SIGNALS = _slice_signals(audio_signals=NORMALIZED_DFILTERS_SIGNALS,
                                                        references=REFERENCES,
                                                        sr=c.SAMPLING_RATE,
                                                        period=c.PERIOD)

    # 1.5. Saving Signals:
    _save_dataframe(dataframe=NORMALIZED_ORIGINAL_SIGNALS,
                    normalization='normalized',
                    denoise_method='original_signals',
                    csv_version=1)

    _save_dataframe(dataframe=SLICED_NORMALIZED_DFILTERS_SIGNALS,
                    normalization='normalized',
                    denoise_method='digital_filters',
                    csv_version=1)

    # 2. Denormalization:
    DENORMALIZED_ORIGINAL_SIGNALS = _extract_signals(audio_files=AUDIO_FILES,
                                                     normalization=False)

    # 2.2. Denoising:
    DENORMALIZED_DFILTERS_SIGNALS = _denoise_signals(audio_signals=DENORMALIZED_ORIGINAL_SIGNALS,
                                                     denoise_method='digital_filters')

    # 2.3. Create DataFrames and References:
    DENORMALIZED_ORIGINAL_SIGNALS = pd.DataFrame(DENORMALIZED_ORIGINAL_SIGNALS)
    DENORMALIZED_ORIGINAL_SIGNALS = DENORMALIZED_ORIGINAL_SIGNALS.join(REFERENCES)

    DENORMALIZED_DFILTERS_SIGNALS = pd.DataFrame(DENORMALIZED_DFILTERS_SIGNALS)
    DENORMALIZED_DFILTERS_SIGNALS = DENORMALIZED_DFILTERS_SIGNALS.join(REFERENCES)

    # 2.4. Slicing Signals:
    SLICED_DENORMALIZED_DFILTERS_SIGNALS = _slice_signals(audio_signals=DENORMALIZED_DFILTERS_SIGNALS,
                                                          references=REFERENCES,
                                                          sr=c.SAMPLING_RATE,
                                                          period=c.PERIOD)

    # 2.5. Saving DataFrames:

    _save_dataframe(dataframe=DENORMALIZED_ORIGINAL_SIGNALS,
                    normalization='denormalized',
                    denoise_method='original_signals',
                    csv_version=1)

    _save_dataframe(dataframe=SLICED_DENORMALIZED_DFILTERS_SIGNALS,
                    normalization='denormalized',
                    denoise_method='digital_filters',
                    csv_version=1)
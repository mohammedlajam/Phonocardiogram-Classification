"""
Author: Mohammed Lajam

Phase 2: Signal Preprocessing:
- In this python file, the audio data is extracted from the 'data/pcg_signals/PhysioNet_2016'
directory and being preprocessed using EMD-Digital Filter function.
- After denoising the signals, Slicing the signals takes place to a specific length, so that all
signals are equal in length and therefore the number of samples per signals are equal.

Output:
- Generating csv files for each method including all the sliced signals with references is saved
in a folder 'denoised_signals' in 'signal_preprocessing' package.

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
import pandas as pd
from glob import glob
import constants as c
from signal_preprocessing.helpers import *


# 1. Function to load all the paths of the audio files in one list:
def create_signal_paths():
    # Loading all the paths of the audio files in one list:
    audio_files = []
    folders = ['a', 'b', 'c', 'd', 'e', 'f']
    for folder in folders:
        audio = glob(f'{c.REPO_PATH}{c.DATASET_PATH}/training-{folder}/*.wav')
        audio_files.append(audio)
    audio_files = [item for elem in audio_files for item in elem]

    # Loading the references:
    references = []
    for folder in folders:
        references_csv = pd.read_csv(f'{c.REPO_PATH}{c.DATASET_PATH}/training-{folder}/REFERENCE.csv', header=None)
        references.append(references_csv)
    references = pd.concat(references, ignore_index=True)
    references.columns = ['signal_id', 'class']
    return audio_files, references


# 2. Function to denoise all signals using one of the following methods:
# emd, wavelet_transform, digital_filters, emd_wavelet, emd_dfilters, emd_wl_dfilters:
def denoise_signals(audio_files, references, denoise_method: str):
    denoised_signals = []
    # Empirical Mode Decomposition:
    if denoise_method == 'emd':
        for audio_file in range(len(audio_files)):
            _, denoised_signal = empirical_mode_decomposition(file_path=audio_files,
                                                              audio_index=audio_file,
                                                              sr=c.SAMPLING_RATE,
                                                              plot=False)
            denoised_signals.append(denoised_signal)

    # Wavelet-Transform:
    elif denoise_method == 'wavelet_transform':
        for audio_file in range(len(audio_files)):
            _, denoised_signal = wavelet_denoising(file_path=audio_files,
                                                   audio_index=audio_file,
                                                   sr=c.SAMPLING_RATE,
                                                   plot=False)
            denoised_signals.append(denoised_signal)

    # Digital-Filters:
    elif denoise_method == 'digital_filters':
        for audio_file in range(len(audio_files)):
            _, denoised_signal = digital_filter(file_path=audio_files,
                                                audio_index=audio_file,
                                                order=c.FILTER_ORDER,
                                                low_fc=c.LOW_FC,
                                                high_fc=c.HIGH_FC,
                                                sr=c.SAMPLING_RATE,
                                                plot=False)
            denoised_signals.append(denoised_signal)

    # EMD + Wavelet-Transform:
    elif denoise_method == 'emd_wavelet':
        for audio_file in range(len(audio_files)):
            _, denoised_signal = emd_wavelet(file_path=audio_files,
                                             audio_index=audio_file,
                                             sr=c.SAMPLING_RATE,
                                             plot=False)
            denoised_signals.append(denoised_signal)

    # EMD + Digital-Filters:
    elif denoise_method == 'emd_dfilters':
        for audio_file in range(len(audio_files)):
            _, denoised_signal = emd_dfilter(file_path=audio_files,
                                             audio_index=audio_file,
                                             sr=c.SAMPLING_RATE,
                                             order=c.FILTER_ORDER,
                                             low_fc=c.LOW_FC,
                                             high_fc=c.HIGH_FC,
                                             plot=False)
            denoised_signals.append(denoised_signal)

    # EMD + Wavelet-Transform + Digital-Filters:
    elif denoise_method == 'emd_wl_dfilters':
        for audio_file in range(len(audio_files)):
            _, denoised_signal = emd_wl_dfilter(file_path=audio_files,
                                                audio_index=audio_file,
                                                sr=c.SAMPLING_RATE,
                                                order=c.FILTER_ORDER,
                                                low_fc=c.LOW_FC,
                                                high_fc=c.HIGH_FC,
                                                plot=False)
            denoised_signals.append(denoised_signal)

    # Converting 'denoised_signals' to DataFrame:
    denoised_signals = pd.DataFrame(denoised_signals)

    # Combining the 'denoised_signals' with its classes:
    denoised_signals = denoised_signals.join(references)

    # 2. Slicing the signals to 5000 samples per signal with respect to its class
    sliced_signals = slice_signals(signals=denoised_signals,
                                   period=c.PERIOD,
                                   sr=c.SAMPLING_RATE,
                                   save=True,
                                   csv_version=1,
                                   denoise_method=f'{denoise_method}')


if __name__ == "__main__":
    AUDIO_FILES, REFERENCES = create_signal_paths()
    denoise_signals(audio_files=AUDIO_FILES,
                    references=REFERENCES,
                    denoise_method='digital_filters')


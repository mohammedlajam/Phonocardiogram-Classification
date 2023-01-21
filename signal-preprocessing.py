"""
Author: Mohammed Lajam

Phase 2: Signal Preprocessing:
In this python file, the audio data is extracted from the local machine and being preprocessed
using EMD-Digital Filter function.

Phase 3: Segmentation:
In this Phase, Segmentation is used to slice the signals to a specific length, so that all
signals are equal in length and therefore the number of samples per signals are equal.

Output: A csv file including all the

Note:
- File path has to be adjusted before running the file.
- All the functions and variables, which are used in this notebook, are imported from helps.py
file from the signal_processing package in the same repository.
"""

# loading libraries:
import pandas as pd
from glob import glob
import constants as c
from signal_preprocessing.helpers import emd_dfilter, digital_filter, slice_signals

# Loading all the audio files in one list:
folders = ['a', 'b', 'c', 'd', 'e', 'f']

audio_files = []
for folder in folders:
    audio = glob(f'/Users/mohammedlajam/Documents/GitHub/Datasets/Phonocardiogram/PhysioNet_2016/training-'
                 f'{folder}/*.wav')
    audio_files.append(audio)
audio_files = [item for elem in audio_files for item in elem]

# Loading the references:
references = []
for folder in folders:
    csv_file = pd.read_csv(f'/Users/mohammedlajam/Documents/GitHub/Datasets/Phonocardiogram/PhysioNet_2016/training-'
                           f'{folder}/REFERENCE.csv', header=None)
    references.append(csv_file)
references = pd.concat(references, ignore_index=True)
references.columns = ['signal_id', 'class']

# 1. Signal Preprocessing:
# denoising the signals using Digital Filters:
FILTERED_SIGNALS = []
for audio_file in range(len(audio_files)):
    _, filtered_signal = digital_filter(file_path=audio_files,
                                        audio_index=audio_file,
                                        order=c.FILTER_ORDER,
                                        low_fc=c.LOW_FC,
                                        high_fc=c.HIGH_FC,
                                        sr=c.SAMPLING_RATE,
                                        plot=False)
    FILTERED_SIGNALS.append(filtered_signal)
FILTERED_SIGNALS = pd.DataFrame(FILTERED_SIGNALS)

# combining the FILTER_SIGNALS with its classes:
FILTERED_SIGNALS = FILTERED_SIGNALS.join(references)

# 2. Segmentation:
# slicing the signals to 5000 samples per signal with respect to its class
# the output is saved automatically in a csv file in 'filtered_signals' folder in the same directory
SLICED_SIGNALS = slice_signals(signals=FILTERED_SIGNALS,
                               period=c.PERIOD,
                               sr=c.SAMPLING_RATE,
                               save=True,
                               csv_version=1)
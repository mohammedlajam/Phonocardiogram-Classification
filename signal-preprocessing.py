"""
Author: Mohammed Lajam

Phase 2: Signal Preprocessing:
- In this python file, the audio data is extracted from the local machine and being preprocessed
using EMD-Digital Filter function.
- After denoising the signals, Slicing the signals takes place to a specific length, so that all
signals are equal in length and therefore the number of samples per signals are equal.

Output:
- A csv file including all the sliced signals with references is saved in local machine.

Note:
- In the constants.py, DATASET_PATH has to be adjusted before running this file.
- All the functions and variables, which are used in this file, are imported from helpers.py
file from the signal_processing package in the same repository.
"""

# loading libraries:
import pandas as pd
from glob import glob
import constants as c
from signal_preprocessing.helpers import emd_dfilter, digital_filter, slice_signals

# Loading all the paths of the audio files in one list:
AUDIO_FILES = []
FOLDERS = ['a', 'b', 'c', 'd', 'e', 'f']
for FOLDER in FOLDERS:
    AUDIO = glob(f'{c.DATASET_PATH}training-{FOLDER}/*.wav')
    AUDIO_FILES.append(AUDIO)
AUDIO_FILES = [item for elem in AUDIO_FILES for item in elem]

# Loading the references:
REFERENCES = []
for FOLDER in FOLDERS:
    REFERENCES_CSV = pd.read_csv(f'{c.DATASET_PATH}training-{FOLDER}/REFERENCE.csv', header=None)
    REFERENCES.append(REFERENCES_CSV)
REFERENCES = pd.concat(REFERENCES, ignore_index=True)
REFERENCES.columns = ['signal_id', 'class']

# 1. Signal Preprocessing:
# Denoising the signals using Digital Filters:
FILTERED_SIGNALS = []
for AUDIO_FILE in range(len(AUDIO_FILES)):
    _, FILTERED_SIGNAL = digital_filter(file_path=AUDIO_FILES,
                                        audio_index=AUDIO_FILE,
                                        order=c.FILTER_ORDER,
                                        low_fc=c.LOW_FC,
                                        high_fc=c.HIGH_FC,
                                        sr=c.SAMPLING_RATE,
                                        plot=False)
    FILTERED_SIGNALS.append(FILTERED_SIGNAL)
FILTERED_SIGNALS = pd.DataFrame(FILTERED_SIGNALS)

# combining the FILTER_SIGNALS with its classes:
FILTERED_SIGNALS = FILTERED_SIGNALS.join(REFERENCES)

# 2. Slicing:
# slicing the signals to 5000 samples per signal with respect to its class
# the output is saved automatically in a csv file in 'biosignal_precessing' folder in local machine
SLICED_SIGNALS = slice_signals(signals=FILTERED_SIGNALS,
                               period=c.PERIOD,
                               sr=c.SAMPLING_RATE,
                               save=True,
                               csv_version=1,
                               denoise_method='digital_filters')
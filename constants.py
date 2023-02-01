# This file contains all the variables, which is used in this repository:
# 1. Paths:
DATASET_PATH = '/Users/mohammedlajam/Documents/GitHub/Datasets/Phonocardiogram/PhysioNet_2016/'
SIG_PRE_PATH = '/Users/mohammedlajam/Documents/GitHub/Datasets/Phonocardiogram/PhysioNet_2016/biosignal_processing/signal_preprocessing'
FEATURE_EXTRACTION_PATH = '/Users/mohammedlajam/Documents/GitHub/Datasets/Phonocardiogram/PhysioNet_2016/biosignal_processing/feature_extraction'

# 2. Signal-preprocessing variables:
SAMPLING_RATE = 1000
LOW_FC = 100
HIGH_FC = 10
FILTER_ORDER = 8

# 3. Signal Segmentation variables:
PERIOD = 5

# 4. Feature Extraction variables:
FRAME_SIZE = 512
HOP_SIZE = 64
SPLIT_FREQUENCY = 256
N_MELS = 128
N_MFCCS = 13
N_SCALES = 30

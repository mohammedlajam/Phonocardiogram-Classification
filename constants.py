# This file contains all the variables, which are used in this repository:
# 1. Paths:
REPO_PATH = '/Users/mohammedlajam/Documents/GitHub/pcg-classification'
DATASET_PATH = f'{REPO_PATH}/data/pcg_signals/PhysioNet_2016'
SIG_PRE_PATH = f'{REPO_PATH}/data/denoised_signals'
FEATURE_EXTRACTION_PATH = f'{REPO_PATH}/data/extracted_features'
CROSS_VALIDATION_PATH = f'{REPO_PATH}/data/cross_validation'

# 2. Signal-preprocessing variables:
SAMPLING_RATE = 1000
N_IMF = 1
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

# 5. Data Preparation:
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_FOLDS = 6
CORR_THRESHOLD = 0.8

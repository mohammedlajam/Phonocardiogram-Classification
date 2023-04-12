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

# 6. Classification models:
# 6.1. Tabular Data:
# 6.1.1. Support Vector Machine:
SVM_KERNEL = 'rbf'
SVM_GAMMA = 1.0
SVM_C = 1.0

# 6.1.2. TabNet:
# Model Parameters for building the TabNet Model:
TB_N_D = 16  # best at 32
TB_N_A = 16  # best at 32
TB_N_STEPS = 10  # best at 10 (recommended bet. 1 to 10)
TB_GAMMA = 1
TB_N_IND = 5  # best at 5(recommended bet. 1 to 5)
TB_N_SHARED = 2  # best at 2 (recommended 2 or 4)
TB_LEARNING_RATE = 0.001
TB_MASK_TYPE = 'entmax'  # sparsemax, entmax, softmax

# Model Parameters for fitting the TabNet Model:
TB_EPOCHS = 1
TB_PATIENCE = 40
TB_BATCH_SIZE = 1024

# 6.1.3. CNN:
# Model Parameters for building the TabNet Model:
CNN_FILTER_1 = 16
CNN_FILTER_2 = 16
CNN_DENSE_1 = 500
CNN_DENSE_2 = 1000
CNN_PATIENCE = 40
CNN_OPTIMIZER = 'adam'
CNN_LOSS = 'binary_crossentropy'

# Model Parameters for fitting the TabNet Model:
CNN_EPOCHS = 1
CNN_BATCH_SIZE = 32

# 6.1.4. RNN:

# 6.1.5. C-RNN:

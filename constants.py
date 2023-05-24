# This file contains all the variables, which are used in this repository:
# 1. Paths:
REPO_PATH = '/Users/mohammedlajam/Documents/GitHub/pcg-classification'
DATASET_PATH = f'{REPO_PATH}/data/pcg_signals/PhysioNet_2016'
SIG_PRE_PATH = f'{REPO_PATH}/data/denoised_signals'
FEATURE_EXTRACTION_PATH = f'{REPO_PATH}/data/extracted_features'
PROCESSED_DATA_PATH = f'{REPO_PATH}/data/processed_data'
IMAGES_PATH = f'{FEATURE_EXTRACTION_PATH}/images'

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
# 5.1. Tabular Data:
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_FOLDS = 6
CORR_THRESHOLD = 0.8
IMG_WIDTH = 128
IMG_LENGTH = 128
IMG_CROSS_VALIDATION = True

# 6. Classification models:
# 6.1. Tabular Data:
# 6.1.1. MLP:
TB_MLP_AUTO_HP = False
# Constants for Automatic Hyper-Parameters Tuning:
TB_MLP_HP_MAX_TRIALS = 200
TB_MLP_HP_EPOCHS = 5

# Constants for Manual Hyper-Parameters Tuning:
TB_MLP_DENSE_1 = 224  # (HP-Tuning: 224) => best at 224
TB_MLP_DENSE_2 = 416  # (HP-Tuning: 416) => best at 416
TB_MLP_DENSE_1_L2 = 0.001  # (HP-Tuning: 0.001) => best at 0.001
TB_MLP_DENSE_2_L2 = 0.0  # (HP-Tuning: 0.0) => best at 0.0
TB_MLP_DROPOUT_RATE_1 = 0.1  # (HP-Tuning: 0.1) => best at 0.1
TB_MLP_DROPOUT_RATE_2 = 0.3  # (HP-Tuning: 0.3) => best at 0.3
TB_MLP_LEARNING_RATE = 0.001  # (HP-Tuning: 0.001) => best at 0.01

# Constants for Building and Fitting 1D-CNN Model:
TB_MLP_PATIENCE = 20
TB_MLP_OPTIMIZER = 'adam'
TB_MLP_LOSS = 'binary_crossentropy'
TB_MLP_EPOCHS = 300
TB_MLP_BATCH_SIZE = 32  # (HP-Tuning: 32) => best at 32

# 6.1.2. 1D-CNN:
TB_CNN_AUTO_HP = False
# Constants for Automatic Hyper-Parameters Tuning:
TB_CNN_HP_MAX_TRIALS = 200
TB_CNN_HP_EPOCHS = 5

# Constants for Manual Hyper-Parameters Tuning:
TB_CNN_FILTER_1 = 192  # (HP-Tuning: 192) => best at 192
TB_CNN_FILTER_2 = 160  # (HP-Tuning: 160) => best at 160
TB_CNN_DENSE_1 = 192  # (HP-Tuning: 192) => best at 192
TB_CNN_DENSE_2 = 256  # (HP-Tuning: 256) => best at 256
TB_CNN_FILTER_1_L2 = 0.001  # (HP-Tuning: 0) => best at 0.001
TB_CNN_FILTER_2_L2 = 0.001  # (HP-Tuning: 0) => best at 0.001
TB_CNN_DENSE_1_L2 = 0.001  # (HP-Tuning: 0) => best at 0.001
TB_CNN_DENSE_2_L2 = 0.001  # (HP-Tuning: 0) => best at 0.001
TB_CNN_DROPOUT_RATE = 0.5  # (HP-Tuning: 0) => best at 0.5
TB_CNN_LEARNING_RATE = 0.001  # (HP-Tuning: 0.001) => best at 0.001

# Constants for Building and Fitting 1D-CNN Model:
TB_CNN_PATIENCE = 20
TB_CNN_OPTIMIZER = 'adam'
TB_CNN_LOSS = 'binary_crossentropy'
TB_CNN_EPOCHS = 300  # best at 200
TB_CNN_BATCH_SIZE = 12000  # best at 10000

# 6.1.3. RNN-LSTM:
TB_LSTM_AUTO_HP = False
# Constants for Automatic Hyper-Parameters Tuning:
TB_LSTM_HP_MAX_TRIALS = 110
TB_LSTM_HP_EPOCHS = 3

# Constants for Manual Hyper-Parameters Tuning:
TB_LSTM_LSTM_1 = 256  # (HP-Tuning: 160) => best at 160
TB_LSTM_LSTM_2 = 256  # (HP-Tuning: 160) => best at 160
TB_LSTM_1_L2 = 0.01  # (HP-Tuning: 0.001) => best at 0.001
TB_LSTM_2_L2 = 0.01  # (HP-Tuning: 0.0) => best at 0.001
TB_LSTM_DROPOUT_RATE_1 = 0.5  # (HP-Tuning: 0.2) => best at 0.5
TB_LSTM_DROPOUT_RATE_2 = 0.5  # (HP-Tuning: 0.4) => best at 0.5
TB_LSTM_LEARNING_RATE = 0.001  # (HP-Tuning: 0.001) => best at 0.001

# Constants for Building and Fitting 1D-CNN Model:
TB_LSTM_PATIENCE = 40
TB_LSTM_OPTIMIZER = 'adam'
TB_LSTM_LOSS = 'binary_crossentropy'
TB_LSTM_EPOCHS = 250
TB_LSTM_BATCH_SIZE = 256  # (HP-Tuning: 256)

# 6.1.4. C-RNN:
TB_CRNN_AUTO_HP = False
# Constants for Automatic Hyper-Parameters Tuning:
TB_CRNN_HP_MAX_TRIALS = 100
TB_CRNN_HP_EPOCHS = 2

# Constants for Manual Hyper-Parameters Tuning:
TB_CRNN_FILTER_1 = 256  # (HP-Tuning: 256) => best at 256
TB_CRNN_FILTER_2 = 256  # (HP-Tuning: 256) => best at 256
TB_CRNN_LSTM_1 = 160  # (HP-Tuning: 256) => best at 160
TB_CRNN_LSTM_2 = 160  # (HP-Tuning: 160) => best at 160
TB_CRNN_FILTER_1_L2 = 0.001  # (HP-Tuning: 0.001) => best at 0.001
TB_CRNN_FILTER_2_L2 = 0.001  # (HP-Tuning: 0.0) => best at 0.001
TB_CRNN_LSTM_1_L2 = 0.001  # (HP-Tuning: 0.0) => best at 0.001
TB_CRNN_LSTM_2_L2 = 0.001  # (HP-Tuning: 0.0) => best at 0.001
TB_CRNN_DROPOUT_RATE_1 = 0.5  # (HP-Tuning: 0.3) => best at 0.5
TB_CRNN_DROPOUT_RATE_2 = 0.5  # (HP-Tuning: 0.0) => best at 0.5
TB_CRNN_LEARNING_RATE = 0.001  # (HP-Tuning: 0.001) => best at 0.001

# Constants for Building and Fitting 1D-CNN Model:
TB_CRNN_PATIENCE = 20
TB_CRNN_OPTIMIZER = 'adam'
TB_CRNN_LOSS = 'binary_crossentropy'
TB_CRNN_EPOCHS = 250
TB_CRNN_BATCH_SIZE = 1000  # (HP-Tuning: 32) => best at 1000

# 6.1.5. TabNet:
TB_TABNET_AUTO_HP = False
# Constants for Automatic Hyper-Parameters Tuning:
TB_TABNET_HP_MAX_TRIALS = 100
TB_TABNET_HP_EPOCHS = 3

# Model Hyper-Parameters for building the TabNet Model:
TB_TABNET_N_D = 32  # (HP-Tuning: 32) => best at 32
TB_TABNET_N_A = 32  # (HP-Tuning: 32) => best at 32
TB_TABNET_N_STEPS = 10  # (HP-Tuning: 4) => best at 10 (recommended bet. 1 to 10)
TB_TABNET_GAMMA = 1  # (HP-Tuning: 0.1) => best at 1
TB_TABNET_N_IND = 5  # (HP-Tuning: 1) => best at 5 (recommended bet. 1 to 5)
TB_TABNET_N_SHARED = 2  # (HP-Tuning: 2) =>best at 2 (recommended 2 or 4)
TB_TABNET_LEARNING_RATE = 0.001  # (HP-Tuning: 0.001) => best at 0.001
TB_TABNET_WEIGHT_DECAY = 0.001  # (HP-Tuning: 0.0) => best at 0.001
TB_TABNET_MASK_TYPE = 'sparsemax'  # (HP-Tuning: 'entmax') => best at 'sparsemax' (sparsemax, entmax, softmax)

# Model Hyper-Parameters for fitting the TabNet Model:
TB_TABNET_EPOCHS = 500
TB_TABNET_PATIENCE = 40
TB_TABNET_BATCH_SIZE = 1024  # best at 1024

# 6.1.6. Support Vector Machine:
TB_SVM_AUTO_HP = False
# Model Hyper-Parameters for building the TabNet Model:
TB_SVM_KERNEL = 'rbf'  # (HP-Tuning: 'rbf') => best at 'rbf'
TB_SVM_GAMMA = 5.0  # (HP-Tuning: 10) => best at 5.0
TB_SVM_C = 1.0  # (HP-Tuning: 10) => best at 1.0

# 6.2. Computer Vision (Images):
# 6.2.1. MLP:

# 6.2.2. 2D-CNN:

# 6.2.3. RNN-LSTM:

# 6.2.4. C-RNN:

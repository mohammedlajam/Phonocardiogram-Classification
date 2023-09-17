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
HIGH_FC = 0
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

# 5.2. Computer Vision
GRAY_SCALE = False
IMG_WIDTH = 128
IMG_LENGTH = 128
IMG_REP_TYPE = 'spectrogram'
IMG_CROSS_VALIDATION = True
IMG_RESAMPLING = False
IMG_NORMALIZATION = True

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
TB_MLP_LEARNING_RATE = 0.001  # (HP-Tuning: 0.001) => best at 0.001

# Constants for Building and Fitting 1D-CNN Model:
TB_MLP_PATIENCE = 20
TB_MLP_OPTIMIZER = 'adam'
TB_MLP_LOSS = 'binary_crossentropy'
TB_MLP_EPOCHS = 150
TB_MLP_BATCH_SIZE = 32  # (HP-Tuning: 32) => best at 32

# 6.1.2. 1D-CNN:
TB_CNN_AUTO_HP = False
# Constants for Automatic Hyper-Parameters Tuning:
TB_CNN_HP_MAX_TRIALS = 200
TB_CNN_HP_EPOCHS = 5

# Constants for Manual Hyper-Parameters Tuning:
TB_CNN_FILTER_1 = 192  # (HP-Tuning: 192) => best at 192
TB_CNN_FILTER_2 = 160  # (HP-Tuning: 160) => best at 160
TB_CNN_DENSE_1 = 160  # (HP-Tuning: 192) => best at 160
TB_CNN_DENSE_2 = 160  # (HP-Tuning: 256) => best at 160
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
TB_CNN_EPOCHS = 300  # best at 250
TB_CNN_BATCH_SIZE = 10000  # best at 10000

# 6.1.3. RNN-LSTM:
TB_LSTM_AUTO_HP = False
# Constants for Automatic Hyper-Parameters Tuning:
TB_LSTM_HP_MAX_TRIALS = 110
TB_LSTM_HP_EPOCHS = 3

# Constants for Manual Hyper-Parameters Tuning:
TB_LSTM_LSTM_1 = 160  # (HP-Tuning: 160) => best at 160
TB_LSTM_LSTM_2 = 160  # (HP-Tuning: 160) => best at 160
TB_LSTM_1_L2 = 0.001  # (HP-Tuning: 0.001) => best at 0.001
TB_LSTM_2_L2 = 0.001  # (HP-Tuning: 0.0) => best at 0.001
TB_LSTM_DROPOUT_RATE_1 = 0.4  # (HP-Tuning: 0.2) => best at 0.4
TB_LSTM_DROPOUT_RATE_2 = 0.4  # (HP-Tuning: 0.4) => best at 0.4
TB_LSTM_LEARNING_RATE = 0.001  # (HP-Tuning: 0.001) => best at 0.001

# Constants for Building and Fitting 1D-CNN Model:
TB_LSTM_PATIENCE = 20
TB_LSTM_OPTIMIZER = 'adam'
TB_LSTM_LOSS = 'binary_crossentropy'
TB_LSTM_EPOCHS = 300  # beat at 300
TB_LSTM_BATCH_SIZE = 256  # (HP-Tuning: 256) => best at 256

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
TB_CRNN_EPOCHS = 250  # best at 250
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
TB_TABNET_EPOCHS = 150  # best at 150
TB_TABNET_PATIENCE = 20
TB_TABNET_BATCH_SIZE = 32  # best at 32

# 6.1.6. Support Vector Machine:
TB_SVM_AUTO_HP = False
# Model Hyper-Parameters for building the TabNet Model:
TB_SVM_KERNEL = 'rbf'  # (HP-Tuning: 'rbf') => best at 'rbf'
TB_SVM_GAMMA = 5.0  # (HP-Tuning: 10) => best at 5.0
TB_SVM_C = 1.0  # (HP-Tuning: 10) => best at 1.0

# 6.2. Computer Vision (Images):
# 6.2.1. ResNet50:
# Constants for Manual Hyper-Parameters Tuning:
CV_RN50_INCLUDE_TOP = False  # Always False
CV_RN50_WEIGHTS = 'imagenet'
CV_RN50_TRAINABLE = False  # Best at False
CV_RN50_DENSE_1 = 256  # best at 256
CV_RN50_DENSE_2 = 128  # best at 128
CV_RN50_DENSE_1_L2 = 0.0  # best at 0.0
CV_RN50_DENSE_2_L2 = 0.0  # best at 0.0
CV_RN50_DROPOUT_RATE_1 = 0.0  # best at 0.0
CV_RN50_DROPOUT_RATE_2 = 0.0  # best at 0.0
CV_RN50_LEARNING_RATE = 0.001  # best at 0.001

# Constants for Building and Fitting ResNet50:
CV_RN50_PATIENCE = 4
CV_RN50_OPTIMIZER = 'adam'
CV_RN50_LOSS = 'binary_crossentropy'
CV_RN50_EPOCHS = 15
CV_RN50_BATCH_SIZE = 32  # best at 32

# 6.2.2. VGG19:
# Constants for Manual Hyper-Parameters Tuning:
CV_VGG19_INCLUDE_TOP = False  # Always False
CV_VGG19_WEIGHTS = 'imagenet'
CV_VGG19_TRAINABLE = False  # Best at False
CV_VGG19_DENSE_1 = 1024  # Best at 1024
CV_VGG19_DENSE_2 = 512  # Best at 512
CV_VGG19_DENSE_1_L2 = 0.001  # Best at 0.001
CV_VGG19_DENSE_2_L2 = 0.0  # Best at 0.0
CV_VGG19_DROPOUT_RATE_1 = 0.0  # Best at 0.0
CV_VGG19_DROPOUT_RATE_2 = 0.5  # Best at 0.5
CV_VGG19_LEARNING_RATE = 0.001  # Best at 0.001

# Constants for Building and Fitting VGG19 Model:
CV_VGG19_PATIENCE = 4
CV_VGG19_OPTIMIZER = 'adam'
CV_VGG19_LOSS = 'binary_crossentropy'
CV_VGG19_EPOCHS = 15
CV_VGG19_BATCH_SIZE = 32

# 6.2.3. InceptionV3:
# Constants for Manual Hyper-Parameters Tuning:
CV_INCEPTION_INCLUDE_TOP = False  # Always False
CV_INCEPTION_WEIGHTS = 'imagenet'
CV_INCEPTION_TRAINABLE = True  # best at True
CV_INCEPTION_DENSE_1 = 64  # Best at 64
CV_INCEPTION_DENSE_2 = 32  # Best at 32
CV_INCEPTION_DENSE_1_L2 = 0.001  # Best at 0.001
CV_INCEPTION_DENSE_2_L2 = 0.0  # Best at 0.0
CV_INCEPTION_DROPOUT_RATE_1 = 0.0  # Best at 0.0
CV_INCEPTION_DROPOUT_RATE_2 = 0.5  # Best at 0.5
CV_INCEPTION_LEARNING_RATE = 0.001  # Best at 0.001

# Constants for Building and Fitting VGG19 Model:
CV_INCEPTION_PATIENCE = 4
CV_INCEPTION_OPTIMIZER = 'adam'
CV_INCEPTION_LOSS = 'binary_crossentropy'
CV_INCEPTION_EPOCHS = 10
CV_INCEPTION_BATCH_SIZE = 32

# 6.2.4. InceptionResNetV2:
# Constants for Manual Hyper-Parameters Tuning:
CV_INCEPTIONRESNET_INCLUDE_TOP = False  # Always False
CV_INCEPTIONRESNET_WEIGHTS = 'imagenet'
CV_INCEPTIONRESNET_TRAINABLE = True  # Best at True
CV_INCEPTIONRESNET_DENSE_1 = 128  # Best at 128
CV_INCEPTIONRESNET_DENSE_2 = 64  # Best at 64
CV_INCEPTIONRESNET_DENSE_1_L2 = 0.0  # Best at 0.0
CV_INCEPTIONRESNET_DENSE_2_L2 = 0.0  # Best at 0.0
CV_INCEPTIONRESNET_DROPOUT_RATE_1 = 0.3  # Best at 0.3
CV_INCEPTIONRESNET_DROPOUT_RATE_2 = 0.5  # Best at 0.5
CV_INCEPTIONRESNET_LEARNING_RATE = 0.001  # Best at 0.001

# Constants for Building and Fitting VGG19 Model:
CV_INCEPTIONRESNET_PATIENCE = 4
CV_INCEPTIONRESNET_OPTIMIZER = 'adam'
CV_INCEPTIONRESNET_LOSS = 'binary_crossentropy'
CV_INCEPTIONRESNET_EPOCHS = 15
CV_INCEPTIONRESNET_BATCH_SIZE = 32
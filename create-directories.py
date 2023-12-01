import os
import constants as c

# Creating data directory:
DATA_DIRECTORY = f'{c.REPO_PATH}/data'

# 1. Creating main data directory
os.makedirs(DATA_DIRECTORY, exist_ok=True)

# 2. Creating the branches:
# 2.1 Creating "pcg_signals" directory and its branches:
os.makedirs(f'{DATA_DIRECTORY}/pcg_signals', exist_ok=True)

# 2.2. Creating "denoised_signal" directory and its branches:
os.makedirs(f'{DATA_DIRECTORY}/denoised_signals', exist_ok=True)
directories = ['original_signals', 'emd', 'wavelet_transform', 'digital_filters',
               'emd_wavelet', 'emd_dfilters', 'emd_wl_dfilters']
normalized_dirs = ['normalized', 'denormalized']
for normalized_dir in normalized_dirs:
    subdirectories = [f'{normalized_dir}/{directory}' for directory in directories]
    for subdirectory in subdirectories:
        if not os.path.isdir(f'{c.SIG_PRE_PATH}/{subdirectory}'):
            os.makedirs(f'{c.SIG_PRE_PATH}/{subdirectory}', exist_ok=True)

# 2.3. Creating "extracted_features" directory and its branches:
# 2.3.1. Tabular:
os.makedirs(f'{DATA_DIRECTORY}/extracted_features', exist_ok=True)
os.makedirs(f'{DATA_DIRECTORY}/extracted_features/tabular', exist_ok=True)
os.makedirs(f'{DATA_DIRECTORY}/extracted_features/tabular/normalized', exist_ok=True)
os.makedirs(f'{DATA_DIRECTORY}/extracted_features/tabular/denormalized', exist_ok=True)

# 2.3.2. Images:
os.makedirs(f'{DATA_DIRECTORY}/extracted_features/images/', exist_ok=True)
os.makedirs(f'{DATA_DIRECTORY}/extracted_features/images/spectrogram', exist_ok=True)
os.makedirs(f'{DATA_DIRECTORY}/extracted_features/images/mel_spectrogram', exist_ok=True)
os.makedirs(f'{DATA_DIRECTORY}/extracted_features/images/mfccs/', exist_ok=True)
os.makedirs(f'{DATA_DIRECTORY}/extracted_features/images/mfccs/mfcc', exist_ok=True)
os.makedirs(f'{DATA_DIRECTORY}/extracted_features/images/mfccs/delta_1', exist_ok=True)
os.makedirs(f'{DATA_DIRECTORY}/extracted_features/images/mfccs/delta_2', exist_ok=True)
os.makedirs(f'{DATA_DIRECTORY}/extracted_features/images/scalogram', exist_ok=True)

# 2.4 Creating "processed_data" directory and its branches:
# 2.4.1. Tabular:
os.makedirs(f'{DATA_DIRECTORY}/processed_data', exist_ok=True)
os.makedirs(f'{DATA_DIRECTORY}/processed_data/tabular/', exist_ok=True)
os.makedirs(f'{DATA_DIRECTORY}/processed_data/tabular/complete_features', exist_ok=True)
os.makedirs(f'{DATA_DIRECTORY}/processed_data/tabular/selected_features', exist_ok=True)

# 2.4.2. Images:
os.makedirs(f'{DATA_DIRECTORY}/processed_data/images/', exist_ok=True)
IMAGE_REPS = ['spectrogram', 'mel_spectrogram', 'scalogram']
CROSS_VALIDATION_TYPES = ['holdout_cv', 'kfold_cv']

for IMAGE_REP in IMAGE_REPS:
    for CROSS_VALIDATION_TYPE in CROSS_VALIDATION_TYPES:
        os.makedirs(f'{DATA_DIRECTORY}/processed_data/images/{IMAGE_REP}/{CROSS_VALIDATION_TYPE}', exist_ok=True)


# 2.5 Creating "models" directory and its branches:
os.makedirs(f'{DATA_DIRECTORY}/models', exist_ok=True)
MODELS_DIRECTORIES = ['tabular_svm', 'tabular_mlp', 'tabular_cnn', 'tabular_rnn', 'tabular_crnn',
                      'tabular_tabnet', 'cv_resnet50', 'cv_vgg19', 'cv_inception', 'cv_inception_resnet']
for MODELS_DIRECTORY in MODELS_DIRECTORIES:
    os.makedirs(f'{DATA_DIRECTORY}/models/{MODELS_DIRECTORY}/', exist_ok=True)

"""
Author: Mohammed Lajam

Phase 4: Data Preparation (Computer Vision):
- This Phase consists of 3 major sub-phases.
    1. Exploratory Data Analysis:
    2. Feature Engineering:

Objective:
The goal of data preprocessing is to ensure that the images is in a suitable format to be used by
machine learning phase (phase 5), and that it is free from errors, inconsistencies, and other
issues that could lead to inaccurate or biased results.

Input:
- The input is the images, which are saved in 'data/extracted_features/images' directory.

Output:
- The Output is either a Holdout-Cross Validation or Fold Cross-Validation.
- The Output is saved in a pickle files format in 'data/processed_data/images'

Note:
- EDA is not applied in this python file as it is applied in the Notebook that locates in the
data_preparation package.
- All the functions and variables, which are used in this file, are imported from helpers.py
file from the data_preparation package in the same repository.
"""

# Importing libraries:
import pickle

import constants as c
from data_preparation.helpers import *


def _load_images_labels(rep_type: str):
    """Function to load the images and its labels from 'data/extracted_features/images' and
    return a numpy array lists of images and labels."""
    try:
        os.path.exists(f'{c.FEATURE_EXTRACTION_PATH}/images/{rep_type}')
    except Exception as e:
        print(e)
    else:
        image_dir = f'{c.FEATURE_EXTRACTION_PATH}/images/{rep_type}'
        # Loading References:
        references = pd.read_csv(f'{c.FEATURE_EXTRACTION_PATH}/images/{rep_type}/references.csv')

        images = []
        labels = []
        for index, row in references.iterrows():
            img_ref = row['image_ref'] + '.png'  # Adding file extension
            img_path = os.path.join(image_dir, img_ref)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(image)
            labels.append(row['class'])

        # Convert all -1 labels to 0:
        labels = [0 if label == -1 else label for label in labels]
        return images, labels, references


def _execute_cross_validation(images, labels, references):
    """Function to execute cross validation and returns a list contains all DataFrames for
    each Fold."""
    # 1. Cross Validation (6-Folds):
    # Extracting train_indices and test_indices for each Fold:
    train_indices, test_indices = FeatureEngineeringCV.create_cross_validation(images=images,
                                                                               labels=labels,
                                                                               references=references,
                                                                               n_folds=c.N_FOLDS,
                                                                               rand_state=c.RANDOM_STATE)

    # Create DataFrames for each Fold (Train and Test) in a list:
    x_train_folds, y_train_folds, x_test_folds, y_test_folds = [], [], [], []

    # Iterate through the fold indices and create the corresponding DataFrames:
    for fold_idx in range(c.N_FOLDS):
        x_train_fold, y_train_fold = FeatureEngineeringCV.create_lists_from_indices(images=images,
                                                                                    labels=labels,
                                                                                    indices_list=train_indices,
                                                                                    index=fold_idx)

        x_train_folds.append(x_train_fold)
        y_train_folds.append(y_train_fold)

    for fold_idx in range(c.N_FOLDS):
        x_test_fold, y_test_fold = FeatureEngineeringCV.create_lists_from_indices(images=images,
                                                                                  labels=labels,
                                                                                  indices_list=test_indices,
                                                                                  index=fold_idx)
        x_test_folds.append(x_test_fold)
        y_test_folds.append(y_test_fold)

    return x_train_folds, y_train_folds, x_test_folds, y_test_folds


def _save_as_pickle_files(x_train, x_test, y_train, y_test, rep_type: str, cross_validation: bool):
    """Function to save each fold as a pickle file in 'data/processed_data/images/' directory."""
    try:
        os.path.exists(f'{c.PROCESSED_DATA_PATH}/images/{rep_type}')
    except Exception as e:
        print(e)
    else:
        if cross_validation:
            directory = f'data/processed_data/images/{rep_type}/kfold_cv'
            os.makedirs(directory, exist_ok=True)
            files = [('x_train_folds.pkl', x_train), ('x_test_folds.pkl', x_test),
                     ('y_train_folds.pkl', y_train), ('y_test_folds.pkl', y_test)]
            for filename, data in files:
                file_path = os.path.join(directory, filename)
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
        else:
            directory = f'data/processed_data/images/{rep_type}/holdout_cv'
            os.makedirs(directory, exist_ok=True)
            files = [('x_train.pkl', x_train), ('x_test.pkl', x_test), ('y_train.pkl', y_train), ('y_test.pkl', y_test)]
            for filename, data in files:
                file_path = os.path.join(directory, filename)
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
        return None


if __name__ == "__main__":
    # 1. Loading Images and its Labels:
    IMAGES, LABELS, REFERENCES = _load_images_labels(rep_type='scalogram')

    # Converting IMAGES and LABELS into Numpy arrays:
    IMAGES = np.array(IMAGES, dtype=np.float32)
    LABELS = np.array(LABELS)

    # 2. Resizing the images to 128x128:
    RESIZED_IMAGES = FeatureEngineeringCV.resize_images(images=IMAGES, new_width=c.IMG_WIDTH, new_height=c.IMG_LENGTH)

    # 3. Splitting Images to train, test sets:
    if c.IMG_CROSS_VALIDATION:
        # 3.1. K-Fold Cross-Validation (6 Folds):
        X_TRAIN_FOLDS, Y_TRAIN_FOLDS, X_TEST_FOLDS, Y_TEST_FOLDS = _execute_cross_validation(images=RESIZED_IMAGES,
                                                                                             labels=LABELS,
                                                                                             references=REFERENCES)

        # 4.1 Balancing images:
        X_TRAIN_FOLDS_RESAMPLED = []
        Y_TRAIN_FOLDS_RESAMPLED = []
        for fold_index in range(len(X_TRAIN_FOLDS)):
            X_TRAIN_RESAMPLED, Y_TRAIN_RESAMPLED = FeatureEngineeringCV.balance_images(images=X_TRAIN_FOLDS[fold_index],
                                                                                       labels=Y_TRAIN_FOLDS[fold_index],
                                                                                       rand_state=c.RANDOM_STATE)
            X_TRAIN_FOLDS_RESAMPLED.append(X_TRAIN_RESAMPLED)
            Y_TRAIN_FOLDS_RESAMPLED.append(Y_TRAIN_RESAMPLED)

        # 5.2 Normalizing images:
        X_TRAIN_FOLDS_NORMALIZED = []
        X_TEST_FOLDS_NORMALIZED = []
        for fold_index in range(len(X_TRAIN_FOLDS)):
            X_TRAIN_NORMALIZED = FeatureEngineeringCV.normalize_images(images=X_TRAIN_FOLDS_RESAMPLED[fold_index])
            X_TEST_NORMALIZED = FeatureEngineeringCV.normalize_images(images=X_TEST_FOLDS[fold_index])
            X_TRAIN_FOLDS_NORMALIZED.append(X_TRAIN_NORMALIZED)
            X_TEST_FOLDS_NORMALIZED.append(X_TEST_NORMALIZED)

        # Saving folds in form of Pickle file:
        _save_as_pickle_files(x_train=X_TRAIN_FOLDS_NORMALIZED,
                              x_test=X_TEST_FOLDS_NORMALIZED,
                              y_train=Y_TRAIN_FOLDS_RESAMPLED,
                              y_test=Y_TEST_FOLDS,
                              rep_type='scalogram',
                              cross_validation=True)

    else:
        # 3.2. Holdout Cross-Validation (One-Fold):
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = FeatureEngineeringCV.split_images(images=RESIZED_IMAGES,
                                                                             labels=LABELS,
                                                                             references=REFERENCES,
                                                                             test_size=c.TEST_SIZE,
                                                                             rand_state=c.RANDOM_STATE)

        # 4.2 Balancing images:
        X_TRAIN_RESAMPLED, Y_TRAIN_RESAMPLED = FeatureEngineeringCV.balance_images(images=X_TRAIN,
                                                                                   labels=Y_TRAIN,
                                                                                   rand_state=c.RANDOM_STATE)

        # 5.2 Normalizing images:
        X_TRAIN_NORMALIZED = FeatureEngineeringCV.normalize_images(images=X_TRAIN_RESAMPLED)
        X_TEST_NORMALIZED = FeatureEngineeringCV.normalize_images(images=X_TEST)

        # Saving images and labels in form of Pickle file:
        _save_as_pickle_files(x_train=X_TRAIN_NORMALIZED,
                              x_test=X_TEST_NORMALIZED,
                              y_train=Y_TRAIN_RESAMPLED,
                              y_test=Y_TEST,
                              rep_type='scalogram',
                              cross_validation=False)

"""
Author: Mohammed Lajam

Phase 4: Data Preparation:
- This Phase consists of 3 major sub-phases.
    1. Exploratory Data Analysis:
    2. Feature Engineering:
    3. Feature Selection:

Objective:
The goal of data preprocessing is to ensure that the data is in a suitable format to be used by
machine learning phase (phase 5), and that it is free from errors, inconsistencies, and other
issues that could lead to inaccurate or biased results.

Input:
- The input is the extracted_features, which are saved in 'data/extracted_features' directory.

Output:
- csv files for all the 6 folds of the Cross-Validation (Train and Test) after preprocessing
the data (Feature Engineering and Feature Selection) and they are saved in 'data' directory in
the same project directory.

Note:
- EDA is not applied in this python file as it is applied in the Notebook that locates in the
data_preparation package.
- All the functions and variables, which are used in this file, are imported from helpers.py
file from the data_preparation package in the same repository.
"""

# Importing libraries:
import pandas as pd
from glob import glob
import os
import pickle

import constants as c
from data_preparation.helpers import *


# Functions to be used in this python file:
def _check_create_dir():
    """Function to check if 'cross_validation' directory exists in 'data' directory and create
    it if it is not existed."""
    # 1. CROSS_VALIDATION directory:
    if not os.path.exists(c.CROSS_VALIDATION_PATH):
        os.makedirs(c.CROSS_VALIDATION_PATH)
    return None


def _load_extracted_features(normalization: str):
    """Function to access the latest version of 'extracted_features' from 'data' directory."""
    # Searching and reading the latest csv version:
    if normalization not in ['normalized', 'denormalized']:
        raise ValueError("'normalization is with 'normalized' or 'denormalized'.")
    else:
        list_of_versions = glob(f'{c.FEATURE_EXTRACTION_PATH}/csv/{normalization}/*.csv')
        latest_version = max(list_of_versions, key=os.path.getctime)
        extracted_features = pd.read_csv(latest_version)
    return extracted_features


# 1. Functions for executing Feature Engineering:
def _execute_cross_validation(dataset):
    """Function to execute cross validation and returns a list contains all DataFrames for
    each Fold."""
    # 1. Cross Validation (6-Folds):
    # extract train_indices and test_indices for each Fold:
    train_indices, test_indices = FeatureEngineering(dataset=dataset).create_cross_validation(n_folds=c.N_FOLDS,
                                                                                              rand_state=c.RANDOM_STATE)

    # Create DataFrames for each Fold (Train and Test) in a list:
    x_train_folds = []
    y_train_folds = []
    x_test_folds = []
    y_test_folds = []

    # Iterate through the fold indices and create the corresponding DataFrames:
    for fold_idx in range(c.N_FOLDS):
        x_train_fold, y_train_fold = FeatureEngineering(dataset=dataset).create_dataframe_from_indices(
            indices_list=train_indices, index=fold_idx)
        x_train_folds.append(x_train_fold)
        y_train_folds.append(y_train_fold)

    for fold_idx in range(c.N_FOLDS):
        x_test_fold, y_test_fold = FeatureEngineering(dataset=dataset).create_dataframe_from_indices(
            indices_list=test_indices, index=fold_idx)
        x_test_folds.append(x_test_fold)
        y_test_folds.append(y_test_fold)

    return x_train_folds, y_train_folds, x_test_folds, y_test_folds


def _execute_gaussian_transformation(x_train, x_test):
    """Function to execute Gaussian Transformation for all features."""
    x_train_transformed_features = []
    x_test_transformed_features = []
    # Appending the transformed features to X_TRAIN_FEATURES and X_TEST_FEATURES separately:
    # ds_max:
    x_train_ds_max_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ds_max',
                                                                                                   trans_method='cube_root')
    x_test_ds_max_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ds_max',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_ds_max_transformed)
    x_test_transformed_features.append(x_test_ds_max_transformed)

    # ds_min:
    x_train_ds_min_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ds_min',
                                                                                                   trans_method='cube_root')
    x_test_ds_min_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ds_min',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_ds_min_transformed)
    x_test_transformed_features.append(x_test_ds_min_transformed)

    # ds_mean:
    x_train_ds_mean_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ds_mean',
                                                                                                    trans_method='cube_root')
    x_test_ds_mean_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ds_mean',
                                                                                                  trans_method='cube_root')

    x_train_transformed_features.append(x_train_ds_mean_transformed)
    x_test_transformed_features.append(x_test_ds_mean_transformed)

    # ds_median:
    # Appending the transformed features to X_TRAIN_FEATURES and X_TEST_FEATURES separately:
    x_train_ds_median_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='ds_median',
        trans_method='cube_root')
    x_test_ds_median_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ds_median',
                                                                                                    trans_method='cube_root')

    x_train_transformed_features.append(x_train_ds_median_transformed)
    x_test_transformed_features.append(x_test_ds_median_transformed)

    # ds_std:
    x_train_ds_std_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ds_std',
                                                                                                   trans_method='cube_root')
    x_test_ds_std_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ds_std',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_ds_std_transformed)
    x_test_transformed_features.append(x_test_ds_std_transformed)

    # energy:
    # Appending the transformed features to X_TRAIN_FEATURES and X_TEST_FEATURES separately:
    x_train_energy_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='energy',
                                                                                                   trans_method='cube_root')
    x_test_energy_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='energy',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_energy_transformed)
    x_test_transformed_features.append(x_test_energy_transformed)

    # power:
    x_train_power_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='power',
                                                                                                  trans_method='cube_root')
    x_test_power_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='power',
                                                                                                trans_method='cube_root')

    x_train_transformed_features.append(x_train_power_transformed)
    x_test_transformed_features.append(x_test_power_transformed)

    # ae_max:
    x_train_ae_max_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ae_max',
                                                                                                   trans_method='cube_root')
    x_test_ae_max_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ae_max',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_ae_max_transformed)
    x_test_transformed_features.append(x_test_ae_max_transformed)

    # ae_min:
    x_train_ae_min_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ae_min',
                                                                                                   trans_method='cube_root')
    x_test_ae_min_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ae_min',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_ae_min_transformed)
    x_test_transformed_features.append(x_test_ae_min_transformed)

    # ae_mean:
    x_train_ae_mean_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ae_mean',
                                                                                                    trans_method='cube_root')
    x_test_ae_mean_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ae_mean',
                                                                                                  trans_method='cube_root')

    x_train_transformed_features.append(x_train_ae_mean_transformed)
    x_test_transformed_features.append(x_test_ae_mean_transformed)

    # ae_median:
    x_train_ae_median_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='ae_median',
        trans_method='cube_root')
    x_test_ae_median_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ae_median',
                                                                                                    trans_method='cube_root')

    x_train_transformed_features.append(x_train_ae_median_transformed)
    x_test_transformed_features.append(x_test_ae_median_transformed)

    # ae_std:
    x_train_ae_std_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ae_std',
                                                                                                   trans_method='cube_root')
    x_test_ae_std_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ae_std',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_ae_std_transformed)
    x_test_transformed_features.append(x_test_ae_std_transformed)

    # rm_max:
    x_train_rm_max_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='rm_max',
                                                                                                   trans_method='cube_root')
    x_test_rm_max_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='rm_max',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_rm_max_transformed)
    x_test_transformed_features.append(x_test_rm_max_transformed)

    # rm_min:
    x_train_rm_min_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='rm_min',
                                                                                                   trans_method='cube_root')
    x_test_rm_min_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='rm_min',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_rm_min_transformed)
    x_test_transformed_features.append(x_test_rm_min_transformed)

    # rm_mean:
    x_train_rm_mean_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='rm_mean',
                                                                                                    trans_method='cube_root')
    x_test_rm_mean_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='rm_mean',
                                                                                                  trans_method='cube_root')

    x_train_transformed_features.append(x_train_rm_mean_transformed)
    x_test_transformed_features.append(x_test_rm_mean_transformed)

    # rm_median:
    x_train_rm_median_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='rm_median',
        trans_method='cube_root')
    x_test_rm_median_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='rm_median',
                                                                                                    trans_method='cube_root')

    x_train_transformed_features.append(x_train_rm_median_transformed)
    x_test_transformed_features.append(x_test_rm_median_transformed)

    # rm_std:
    x_train_rm_std_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='rm_std',
                                                                                                   trans_method='cube_root')
    x_test_rm_std_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='rm_std',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_rm_std_transformed)
    x_test_transformed_features.append(x_test_rm_std_transformed)

    # zcr:
    x_train_zcr_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='zcr',
                                                                                                trans_method='exponential')
    x_test_zcr_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='zcr',
                                                                                              trans_method='exponential')

    x_train_transformed_features.append(x_train_zcr_transformed)
    x_test_transformed_features.append(x_test_zcr_transformed)

    # zcr_max:
    x_train_zcr_max_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='zcr_max',
                                                                                                    trans_method='cube_root')
    x_test_zcr_max_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='zcr_max',
                                                                                                  trans_method='cube_root')

    x_train_transformed_features.append(x_train_zcr_max_transformed)
    x_test_transformed_features.append(x_test_zcr_max_transformed)

    # zcr_min:
    x_train_zcr_min_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='zcr_min',
                                                                                                    trans_method='log')
    x_test_zcr_min_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='zcr_min',
                                                                                                  trans_method='log')

    x_train_transformed_features.append(x_train_zcr_min_transformed)
    x_test_transformed_features.append(x_test_zcr_min_transformed)

    # zcr_mean:
    x_train_zcr_mean_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='zcr_mean',
                                                                                                     trans_method='cube_root')
    x_test_zcr_mean_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='zcr_mean',
                                                                                                   trans_method='cube_root')

    x_train_transformed_features.append(x_train_zcr_mean_transformed)
    x_test_transformed_features.append(x_test_zcr_mean_transformed)

    # zcr_median:
    x_train_zcr_median_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='zcr_median',
        trans_method='cube_root')
    x_test_zcr_median_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(
        feature='zcr_median',
        trans_method='cube_root')

    x_train_transformed_features.append(x_train_zcr_median_transformed)
    x_test_transformed_features.append(x_test_zcr_median_transformed)

    # zcr_std:
    x_train_zcr_std_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='zcr_std',
                                                                                                    trans_method='cube_root')
    x_test_zcr_std_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='zcr_std',
                                                                                                  trans_method='cube_root')

    x_train_transformed_features.append(x_train_zcr_std_transformed)
    x_test_transformed_features.append(x_test_zcr_std_transformed)

    # peak_amplitude:
    x_train_peak_amp_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='peak_amplitude',
        trans_method='log')
    x_test_peak_amp_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(
        feature='peak_amplitude',
        trans_method='log')

    x_train_transformed_features.append(x_train_peak_amp_transformed)
    x_test_transformed_features.append(x_test_peak_amp_transformed)

    # peak_frequency:
    x_train_peak_freq_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='peak_frequency',
        trans_method='log')
    x_test_peak_freq_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(
        feature='peak_frequency',
        trans_method='log')

    x_train_transformed_features.append(x_train_peak_freq_transformed)
    x_test_transformed_features.append(x_test_peak_freq_transformed)

    # ber_max:
    x_train_ber_max_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ber_max',
                                                                                                    trans_method='log')
    x_test_ber_max_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ber_max',
                                                                                                  trans_method='log')

    x_train_transformed_features.append(x_train_ber_max_transformed)
    x_test_transformed_features.append(x_test_ber_max_transformed)

    # ber_min:
    x_train_ber_min_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ber_min',
                                                                                                    trans_method='log')
    x_test_ber_min_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ber_min',
                                                                                                  trans_method='log')

    x_train_transformed_features.append(x_train_ber_min_transformed)
    x_test_transformed_features.append(x_test_ber_min_transformed)

    # ber_mean:
    x_train_ber_mean_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ber_mean',
                                                                                                     trans_method='log')
    x_test_ber_mean_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ber_mean',
                                                                                                   trans_method='log')

    x_train_transformed_features.append(x_train_ber_mean_transformed)
    x_test_transformed_features.append(x_test_ber_mean_transformed)

    # ber_median:
    x_train_ber_median_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='ber_median',
        trans_method='log')
    x_test_ber_median_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(
        feature='ber_median',
        trans_method='log')

    x_train_transformed_features.append(x_train_ber_median_transformed)
    x_test_transformed_features.append(x_test_ber_median_transformed)

    # ber_std:
    x_train_ber_std_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ber_std',
                                                                                                    trans_method='log')
    x_test_ber_std_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ber_std',
                                                                                                  trans_method='log')

    x_train_transformed_features.append(x_train_ber_std_transformed)
    x_test_transformed_features.append(x_test_ber_std_transformed)

    # sc_max:
    x_train_sc_max_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='sc_max',
                                                                                                   trans_method='log')
    x_test_sc_max_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='sc_max',
                                                                                                 trans_method='log')

    x_train_transformed_features.append(x_train_sc_max_transformed)
    x_test_transformed_features.append(x_test_sc_max_transformed)

    # sc_min:
    x_train_transformed_features.append(x_train['sc_min'])
    x_test_transformed_features.append(x_test['sc_min'])

    # sc_mean:
    x_train_transformed_features.append(x_train['sc_mean'])
    x_test_transformed_features.append(x_test['sc_mean'])

    # sc_median:
    x_train_transformed_features.append(x_train['sc_median'])
    x_test_transformed_features.append(x_test['sc_median'])

    # sc_std:
    x_train_sc_std_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='sc_std',
                                                                                                   trans_method='log')
    x_test_sc_std_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='sc_std',
                                                                                                 trans_method='log')

    x_train_transformed_features.append(x_train_sc_std_transformed)
    x_test_transformed_features.append(x_test_sc_std_transformed)

    # sb_max:
    x_train_transformed_features.append(x_train['sb_max'])
    x_test_transformed_features.append(x_test['sb_max'])

    # sb_min:
    x_train_transformed_features.append(x_train['sb_min'])
    x_test_transformed_features.append(x_test['sb_min'])

    # sb_mean:
    x_train_transformed_features.append(x_train['sb_mean'])
    x_test_transformed_features.append(x_test['sb_mean'])

    # sb_median:
    x_train_transformed_features.append(x_train['sb_median'])
    x_test_transformed_features.append(x_test['sb_median'])

    # sb_std:
    x_train_sb_std_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='sb_std',
                                                                                                   trans_method='square_root')
    x_test_sb_std_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='sb_std',
                                                                                                 trans_method='square_root')

    x_train_transformed_features.append(x_train_sb_std_transformed)
    x_test_transformed_features.append(x_test_sb_std_transformed)

    # mfcc_max:
    x_train_mfcc_max_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='mfcc_max',
                                                                                                     trans_method='exponential')
    x_test_mfcc_max_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='mfcc_max',
                                                                                                   trans_method='exponential')

    x_train_transformed_features.append(x_train_mfcc_max_transformed)
    x_test_transformed_features.append(x_test_mfcc_max_transformed)

    # mfcc_min:
    x_train_transformed_features.append(x_train['mfcc_min'])
    x_test_transformed_features.append(x_test['mfcc_min'])

    # mfcc_mean:
    x_train_mfcc_mean_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='mfcc_mean',
        trans_method='square_root')
    x_test_mfcc_mean_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='mfcc_mean',
                                                                                                    trans_method='square_root')

    x_train_transformed_features.append(x_train_mfcc_mean_transformed)
    x_test_transformed_features.append(x_test_mfcc_mean_transformed)

    # mfcc_median:
    x_train_transformed_features.append(x_train['mfcc_median'])
    x_test_transformed_features.append(x_test['mfcc_median'])

    # mfcc_std:
    x_train_mfcc_std_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='mfcc_std',
                                                                                                     trans_method='square_root')
    x_test_mfcc_std_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='mfcc_std',
                                                                                                   trans_method='square_root')

    x_train_transformed_features.append(x_train_mfcc_std_transformed)
    x_test_transformed_features.append(x_test_mfcc_std_transformed)

    # delta_1_max:
    x_train_delta_1_max_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='delta_1_max',
        trans_method='square_root')
    x_test_delta_1_max_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(
        feature='delta_1_max',
        trans_method='square_root')

    x_train_transformed_features.append(x_train_delta_1_max_transformed)
    x_test_transformed_features.append(x_test_delta_1_max_transformed)

    # delta_1_min:
    x_train_delta_1_min_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='delta_1_min',
        trans_method='cube_root')
    x_test_delta_1_min_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(
        feature='delta_1_min',
        trans_method='cube_root')

    x_train_transformed_features.append(x_train_delta_1_min_transformed)
    x_test_transformed_features.append(x_test_delta_1_min_transformed)

    # delta_1_mean:
    x_train_delta_1_mean_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='delta_1_mean',
        trans_method='cube_root')
    x_test_delta_1_mean_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(
        feature='delta_1_mean',
        trans_method='cube_root')

    x_train_transformed_features.append(x_train_delta_1_mean_transformed)
    x_test_transformed_features.append(x_test_delta_1_mean_transformed)

    # delta_1_median:
    x_train_delta_1_median_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='delta_1_median',
        trans_method='cube_root')
    x_test_delta_1_median_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(
        feature='delta_1_median',
        trans_method='cube_root')

    x_train_transformed_features.append(x_train_delta_1_median_transformed)
    x_test_transformed_features.append(x_test_delta_1_median_transformed)

    # delta_1_std:
    x_train_delta_1_std_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='delta_1_std',
        trans_method='square_root')
    x_test_delta_1_std_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(
        feature='delta_1_std',
        trans_method='square_root')

    x_train_transformed_features.append(x_train_delta_1_std_transformed)
    x_test_transformed_features.append(x_test_delta_1_std_transformed)

    # delta_2_max:
    x_train_delta_2_max_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='delta_2_max',
        trans_method='exponential')
    x_test_delta_2_max_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(
        feature='delta_2_max',
        trans_method='exponential')

    x_train_transformed_features.append(x_train_delta_2_max_transformed)
    x_test_transformed_features.append(x_test_delta_2_max_transformed)

    # delta_2_min:
    x_train_transformed_features.append(x_train['delta_2_min'])
    x_test_transformed_features.append(x_test['delta_2_min'])

    # delta_2_mean:
    x_train_transformed_features.append(x_train['delta_2_mean'])
    x_test_transformed_features.append(x_test['delta_2_mean'])

    # delta_2_median:
    x_train_transformed_features.append(x_train['delta_2_median'])
    x_test_transformed_features.append(x_test['delta_2_median'])

    # delta_2_std:
    x_train_delta_2_std_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='delta_2_std',
        trans_method='exponential')
    x_test_delta_2_std_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(
        feature='delta_2_std',
        trans_method='exponential')

    x_train_transformed_features.append(x_train_delta_2_std_transformed)
    x_test_transformed_features.append(x_test_delta_2_std_transformed)

    # ca_max:
    x_train_ca_max_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ca_max',
                                                                                                   trans_method='cube_root')
    x_test_ca_max_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ca_max',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_ca_max_transformed)
    x_test_transformed_features.append(x_test_ca_max_transformed)

    # ca_min:
    x_train_ca_min_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ca_min',
                                                                                                   trans_method='cube_root')
    x_test_ca_min_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ca_min',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_ca_min_transformed)
    x_test_transformed_features.append(x_test_ca_min_transformed)

    # ca_mean:
    x_train_ca_mean_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ca_mean',
                                                                                                    trans_method='cube_root')
    x_test_ca_mean_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ca_mean',
                                                                                                  trans_method='cube_root')

    x_train_transformed_features.append(x_train_ca_mean_transformed)
    x_test_transformed_features.append(x_test_ca_mean_transformed)

    # ca_median:
    x_train_ca_median_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='ca_median',
        trans_method='cube_root')
    x_test_ca_median_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ca_median',
                                                                                                    trans_method='cube_root')

    x_train_transformed_features.append(x_train_ca_median_transformed)
    x_test_transformed_features.append(x_test_ca_median_transformed)

    # ca_std:
    x_train_ca_std_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='ca_std',
                                                                                                   trans_method='cube_root')
    x_test_ca_std_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='ca_std',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_ca_std_transformed)
    x_test_transformed_features.append(x_test_ca_std_transformed)

    # cd_max:
    x_train_cd_max_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='cd_max',
                                                                                                   trans_method='cube_root')
    x_test_cd_max_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='cd_max',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_cd_max_transformed)
    x_test_transformed_features.append(x_test_cd_max_transformed)

    # cd_min:
    x_train_cd_min_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='cd_min',
                                                                                                   trans_method='cube_root')
    x_test_cd_min_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='cd_min',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_cd_min_transformed)
    x_test_transformed_features.append(x_test_cd_min_transformed)

    # cd_mean:
    x_train_cd_mean_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='cd_mean',
                                                                                                    trans_method='cube_root')
    x_test_cd_mean_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='cd_mean',
                                                                                                  trans_method='cube_root')

    x_train_transformed_features.append(x_train_cd_mean_transformed)
    x_test_transformed_features.append(x_test_cd_mean_transformed)

    # cd_median:
    x_train_cd_median_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(
        feature='cd_median',
        trans_method='cube_root')
    x_test_cd_median_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='cd_median',
                                                                                                    trans_method='cube_root')

    x_train_transformed_features.append(x_train_cd_median_transformed)
    x_test_transformed_features.append(x_test_cd_median_transformed)

    # cd_std:
    x_train_cd_std_transformed = FeatureEngineering(dataset=x_train).apply_gaussian_transformation(feature='cd_std',
                                                                                                   trans_method='cube_root')
    x_test_cd_std_transformed = FeatureEngineering(dataset=x_test).apply_gaussian_transformation(feature='cd_std',
                                                                                                 trans_method='cube_root')

    x_train_transformed_features.append(x_train_cd_std_transformed)
    x_test_transformed_features.append(x_test_cd_std_transformed)

    # Creating a new DataFrame including all the transformed data:
    x_train_transformed = pd.concat(x_train_transformed_features, axis=1)
    x_test_transformed = pd.concat(x_test_transformed_features, axis=1)

    return x_train_transformed, x_test_transformed


# Functions for executing Feature Selection:
def _find_top_ranked_features(x_train, x_test, y_train, y_test):
    """Function to find the common top ranked features in all folds."""
    top_features_dfs = []
    for index in range(len(x_train)):
        top_features_df = FeatureSelection(x_train=x_train[index],
                                           x_test=x_test[index],
                                           y_train=y_train[index],
                                           y_test=y_test[index]).get_top_features(method='chi2', num_features=10)
        top_features_dfs.append(top_features_df)

    top_features_lists = []
    for index in range(len(top_features_dfs)):
        top_features_list = top_features_dfs[index][0].tolist()
        top_features_lists.append(top_features_list)

    common_features = list(set.intersection(*[set(top_features_lists[i]) for i in range(len(top_features_lists))]))
    return common_features


def _drop_correlated_features_all_folds(x_train_folds, x_test_folds, y_train_folds, y_test_folds, top_common_features):
    """Function to find the drop the high correlated features, which are not in top_ranked_features."""
    # Extract correlated features in each fold:
    correlated_features_folds = []
    for index in range(len(x_train_folds)):
        correlated_features = FeatureSelection(x_train=x_train_folds[index],
                                               x_test=x_test_folds[index],
                                               y_train=y_train_folds[index],
                                               y_test=y_test_folds[index]).drop_correlated_features(threshold=c.CORR_THRESHOLD,
                                                                                                    top_features=top_common_features,
                                                                                                    drop=False)
        correlated_features_folds.append(correlated_features)
    # Find the common correlated features in all folds:
    common_correlated_features_folds = list(set.intersection(*[set(correlated_features_folds[i]) for i in range(len(correlated_features_folds))]))

    # Dropping 'common_correlated_features_folds' from each Fold of 'x_train_folds' and 'x_test_folds':
    for index in range(len(x_train_folds)):
        x_train_folds[index] = x_train_folds[index].drop(common_correlated_features_folds, axis=1)
        x_test_folds[index] = x_test_folds[index].drop(common_correlated_features_folds, axis=1)
    return x_train_folds, x_test_folds, common_correlated_features_folds


def _save_as_pickle_files(x_train_folds, x_test_folds, y_train_folds, y_test_folds, feature_selection=bool):
    """Function to save each fold as a pickle file in 'data/cross_validation' directory."""
    if feature_selection:
        folds = [x_train_folds, x_test_folds, y_train_folds, y_test_folds]
        fold_names = ['x_train_folds', 'x_test_folds', 'y_train_folds', 'y_test_folds']
        for index, fold in enumerate(folds):
            with open(f'data/cross_validation/selected_features/{fold_names[index]}.pkl', 'wb') as f:
                pickle.dump(fold, f)
    else:
        folds = [x_train_folds, x_test_folds, y_train_folds, y_test_folds]
        fold_names = ['x_train_folds', 'x_test_folds', 'y_train_folds', 'y_test_folds']
        for index, fold in enumerate(folds):
            with open(f'data/cross_validation/complete_features/{fold_names[index]}.pkl', 'wb') as f:
                pickle.dump(fold, f)


if __name__ == "__main__":
    # Checking directories:
    _check_create_dir()
    # Loading dataset:
    DATASET = _load_extracted_features(normalization='denormalized')

    # 1. Feature Engineering:
    # 1.1. K-Fold Cross Validation (6 Folds):
    X_TRAIN_FOLDS, Y_TRAIN_FOLDS, X_TEST_FOLDS, Y_TEST_FOLDS = _execute_cross_validation(dataset=DATASET)

    # 1.2. Feature Scaling:
    # 1.2.1. Gaussian Transformation:
    X_TRAIN_FOLDS_TRANSFORMED = []
    X_TEST_FOLDS_TRANSFORMED = []
    for fold_index in range(len(X_TRAIN_FOLDS)):
        X_TRAIN_TRANSFORMED, X_TEST_TRANSFORMED = _execute_gaussian_transformation(x_train=X_TRAIN_FOLDS[fold_index],
                                                                                   x_test=X_TEST_FOLDS[fold_index])
        X_TRAIN_FOLDS_TRANSFORMED.append(X_TRAIN_TRANSFORMED)
        X_TEST_FOLDS_TRANSFORMED.append(X_TEST_TRANSFORMED)

    # Remove missing values from 'delta_1_max':
    for fold_index in range(len(X_TRAIN_FOLDS_TRANSFORMED)):
        FeatureEngineering(dataset=X_TRAIN_FOLDS_TRANSFORMED[fold_index]).replace_missing_values(feature='delta_1_max',
                                                                                                 replace_method='median')

    # 1.2.2. Normalization:
    X_TRAIN_FOLDS_NORMALIZED = []
    X_TEST_FOLDS_NORMALIZED = []
    for fold_index in range(len(X_TRAIN_FOLDS)):
        X_TRAIN_NORMALIZED, X_TEST_NORMALIZED = FeatureEngineering(dataset=DATASET).normalize_data(
            x_train=X_TRAIN_FOLDS_TRANSFORMED[fold_index],
            x_test=X_TRAIN_FOLDS_TRANSFORMED[fold_index])
        X_TRAIN_FOLDS_NORMALIZED.append(X_TRAIN_NORMALIZED)
        X_TEST_FOLDS_NORMALIZED.append(X_TEST_NORMALIZED)

    # 1.3. Balancing Datasets:
    X_TRAIN_FOLDS_RESAMPLED = []
    Y_TRAIN_FOLDS_RESAMPLED = []
    for fold_index in range(len(X_TRAIN_FOLDS)):
        X_TRAIN_RESAMPLED, Y_TRAIN_RESAMPLED = FeatureEngineering(dataset=DATASET).balance_dataset(
            x_train=X_TRAIN_FOLDS_NORMALIZED[fold_index],
            y_train=Y_TRAIN_FOLDS[fold_index],
            rand_state=c.RANDOM_STATE)
        X_TRAIN_FOLDS_RESAMPLED.append(X_TRAIN_RESAMPLED)
        Y_TRAIN_FOLDS_RESAMPLED.append(Y_TRAIN_RESAMPLED)

    # 2. Feature Selection:
    # 2.1. Correlation between input feature and output target:
    COMMON_FEATURES = _find_top_ranked_features(x_train=X_TRAIN_FOLDS_RESAMPLED,
                                                x_test=X_TEST_FOLDS_NORMALIZED,
                                                y_train=Y_TRAIN_FOLDS_RESAMPLED,
                                                y_test=Y_TEST_FOLDS)

    # 2.2. Correlation between input features:
    X_TRAIN_FOLDS_SELECTED, X_TEST_FOLDS_SELECTED, _ = _drop_correlated_features_all_folds(
        x_train_folds=X_TRAIN_FOLDS_RESAMPLED,
        x_test_folds=X_TEST_FOLDS_NORMALIZED,
        y_train_folds=Y_TRAIN_FOLDS_RESAMPLED,
        y_test_folds=Y_TEST_FOLDS,
        top_common_features=COMMON_FEATURES)

    # Saving all DataFrames in Local Machine:
    # All Features:
    _save_as_pickle_files(x_train_folds=X_TRAIN_FOLDS_RESAMPLED,
                          x_test_folds=X_TEST_FOLDS_NORMALIZED,
                          y_train_folds=Y_TRAIN_FOLDS_RESAMPLED,
                          y_test_folds=Y_TEST_FOLDS,
                          feature_selection=False)

    # Selected Features:
    _save_as_pickle_files(x_train_folds=X_TRAIN_FOLDS_SELECTED,
                          x_test_folds=X_TEST_FOLDS_SELECTED,
                          y_train_folds=Y_TRAIN_FOLDS_RESAMPLED,
                          y_test_folds=Y_TEST_FOLDS,
                          feature_selection=True)


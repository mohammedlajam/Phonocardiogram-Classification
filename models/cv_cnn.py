"""
Author: Mohammed Lajam

Phase 5: Classification (Computer Vision - 2D-CNN):
- In this python file, the audio data is imported from the '/data/cross_validation/selected_features/'
directory.
- After denoising the signals, Slicing the signals takes place to a specific length, so that all
signals are equal in length and therefore the number of samples per signals are equal.

Objective:
- Finding the best Hyper-Parameters by using Bayesian Optimization Technique from Keras tuners.
- Building the Model based on the best Hyper-Parameters per fold and making predictions and
evaluating the Model based on Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity and
ROC_AUC Score, etc.
- Saving all the Artifacts, Parameters and Matrices in mlflow to be used in Phase 6.

Input:
- The input data is the images and references.csv from 'data/extracted_features/images/scalogram.

Output:
- The outputs are Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity and
ROC_AUC Score saved in 'mlruns' in the same directory. Additionally, all the Artifacts, parameters
and matrices are tracked using MLFlow.

Note:
- In 'constants.py', 'CROSS_VALIDATION' path has to be adjusted before running this file.
- All the functions and variables, which are used in this file, are imported from helpers.py
file from the 'models' package in the same repository.
"""

# Loading Libraries:
import os

import constants as c


# Functions to be used in this python file:
def _check_create_dir():
    """Function to check if 'cross_validation' directory exists in 'data' directory and create
    it if it is not existed."""
    try:
        os.path.isdir(c.IMAGES_PATH)
    except FileNotFoundError:
        print("Directory does not exist!")
    except Exception as e:
        print(f"Error: {e}")
    else:
        return None


def _load_images():
    """Function to load images from loacl machine."""

    return None
"""
Author: Mohammed Lajam

This file contains a collection of helper functions that are commonly used in Classification Models
'Phase 5'. These functions provide support for various tasks such as building, predicting and
evaluating Machine and Deep Learning Models. The purpose of these helper functions is to
encapsulate repetitive and complex code into reusable and modular blocks, making it easier to
maintain and improve the overall functionality of the project.
"""

# Importing libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, \
    roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')


class ModelBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build_fit_svm(x_train, y_train, kernel: str, gamma: str, c: float, rand_state: int):
        """Function to build and fit Support Vector Machine."""
        # Building the Model:
        svm_model = svm.SVC(kernel=kernel, gamma=gamma, C=c, random_state=rand_state)
        # Fitting the Model:
        svm_model.fit(x_train, y_train)
        return svm_model

    @staticmethod
    def build_fit_tabnet(x_train, x_val, y_train, y_val, n_d: int, n_a: int, n_steps: int, gamma: float,
                         n_ind: int, n_shared: int, learning_rate: float, mask_type: str, epochs: int,
                         patience: int, batch_size: int):
        """Function to build and fit TabNet Model."""
        # Building the Model:
        tb_model = TabNetClassifier(n_d=n_d,
                                    n_a=n_a,
                                    n_steps=n_steps,
                                    gamma=gamma,
                                    n_independent=n_ind,
                                    n_shared=n_shared,
                                    optimizer_fn=torch.optim.Adam,
                                    optimizer_params=dict(lr=learning_rate, weight_decay=1e-5),
                                    scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                    scheduler_params=dict(step_size=n_steps, gamma=gamma),
                                    mask_type=mask_type)

        # Fitting the Model:
        tb_model.fit(X_train=x_train,
                     y_train=y_train,
                     eval_set=[(x_train, y_train), (x_val, y_val)],
                     eval_name=['train', 'valid'],
                     eval_metric=['accuracy'],
                     max_epochs=epochs,
                     patience=patience,
                     batch_size=batch_size,
                     drop_last=False)
        return tb_model

    @staticmethod
    def build_fit_1d_cnn(x_train, x_val, y_train, y_val, input_shape, filter_1: int, filter_2: int,
                         dense_1: int, dense_2: int, optimizer: str, loss: str, patience: int,
                         epochs: int, batch_size: int):
        """Function to build, complie and fit 1D-CNN Model. It returns Model and History."""
        # Build 1D-CNN Model:
        cnn_model = Sequential()
        cnn_model.add(Conv1D(filters=filter_1, kernel_size=3, activation='relu', input_shape=input_shape))
        cnn_model.add(MaxPooling1D(pool_size=2))
        cnn_model.add(Conv1D(filters=filter_2, kernel_size=3, activation='relu'))
        cnn_model.add(MaxPooling1D(pool_size=2))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(dense_1, activation='relu'))
        cnn_model.add(Dense(dense_2, activation='relu'))
        cnn_model.add(Dense(1, activation='sigmoid'))

        # Compile the Model:
        cnn_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1, mode='max')

        # Fitting the Model:
        history = cnn_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                                callbacks=[early_stop])
        return cnn_model, history


# 2. Class for calculating Predictions and evaluating Model:
class ModelEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def calculate_probabilities(model, x_test):
        """Function to predict the model and returns the predictions."""
        y_prob = model.predict(x_test)
        return y_prob

    @staticmethod
    def find_best_threshold(y_test, y_prob):
        # Extracting fpr, tpr and thresholds:
        _, _, thresholds = roc_curve(y_test, y_prob)
        # Calculating all the evaluation accuracy based on all the thresholds:
        accuracy_matrix = []
        for threshold in thresholds:
            y_pred = np.where(y_prob > threshold, 1, 0)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_matrix.append((threshold, accuracy))
        # Create a dataframe
        accuracy_matrix_df = pd.DataFrame(accuracy_matrix, columns=['Threshold', 'Accuracy'])
        accuracy_matrix_df.sort_values(by='Accuracy', ascending=False, inplace=True)
        best_accuracy = accuracy_matrix_df.iloc[0]['Accuracy']
        best_threshold = accuracy_matrix_df.iloc[0]['Threshold']
        return best_accuracy, best_threshold

    @staticmethod
    def evaluate_model(y_test, y_prob, threshold):
        """Function to extract the """
        # Extracting fpr, tpr, accuracy, precision
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        evaluation_metrics = []
        y_pred = np.where(y_prob > threshold, 1, 0)  # Calculating the y_pred based on best threshold
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        roc_auc = roc_auc_score(y_test, y_pred)
        auc_score = auc(fpr, tpr)
        evaluation_metrics.append((threshold, accuracy, precision, recall, f1, sensitivity, specificity, tn, fp, fn, tp,
                                   roc_auc, auc_score, fpr, tpr))

        # Creating a DataFrame for all evaluation metrics:
        evaluation_metrics_df = pd.DataFrame(evaluation_metrics,
                                             columns=['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1-score',
                                                      'Sensitivity', 'Specificity', 'TN', 'FP', 'FN', 'TP', 'ROC_AUC',
                                                      'AUC_Score', 'FPR', 'TPR'])
        return evaluation_metrics_df

    @staticmethod
    def generate_confusion_matrix(y_test, y_prob, threshold):
        y_pred = np.where(y_prob > threshold, 1, 0)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        return cm


def plot_roc_auc(fpr, tpr):
    """Function for plotting ROC_AUC Curve."""
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Charateristics (ROC) Curve')
    plt.legend()
    return plt.show()

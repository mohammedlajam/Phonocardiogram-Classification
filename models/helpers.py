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

from sklearn import svm
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras_tuner.tuners import BayesianOptimization


from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, \
    roc_curve, auc

import warnings
warnings.filterwarnings('ignore')


class HyperParametersTuner:
    def __init__(self):
        pass

    @staticmethod
    def find_best_1d_cnn_hp(x_train, x_val, y_train, y_val, input_shape, max_trials, epochs, directory):
        """Function to find the best Hyper-Parameters for 1D-CNN Model."""

        # Building the Model with Hyper-Parameters to be tuned:
        def build_1d_cnn_model(hp):
            cnn_model = Sequential()
            cnn_model.add(Conv1D(filters=hp.Int('filter_1', min_value=64, max_value=256, step=32),
                                 kernel_size=3,
                                 activation='relu',
                                 input_shape=input_shape,
                                 kernel_regularizer=l2(hp.Choice('filter_1_l2', values=[0.0, 0.01, 0.001]))))
            cnn_model.add(MaxPooling1D(pool_size=2))
            cnn_model.add(Conv1D(filters=hp.Int('filter_2', min_value=64, max_value=256, step=32),
                                 kernel_size=3,
                                 activation='relu',
                                 kernel_regularizer=l2(hp.Choice('filter_2_l2', values=[0.0, 0.01, 0.001]))))
            cnn_model.add(MaxPooling1D(pool_size=2))
            cnn_model.add(Flatten())
            cnn_model.add(Dense(hp.Int('dense_1', min_value=32, max_value=256, step=32),
                                activation='relu',
                                kernel_regularizer=l2(hp.Choice('dense_1_l2', values=[0.0, 0.01, 0.001]))))
            cnn_model.add(Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))
            cnn_model.add(Dense(hp.Int('dense_2', min_value=32, max_value=256, step=32),
                                activation='relu',
                                kernel_regularizer=l2(hp.Choice('dense_2_l2', values=[0.0, 0.01, 0.001]))))
            cnn_model.add(Dense(1, activation='sigmoid'))

            # Tune the optimizer and learning rate
            optimizer = optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[0.001, 0.0001]))
            cnn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            return cnn_model

        # Setting up the Model and Finding the best Hyper-Parameter combination based on Val_accuracy:
        tuner = BayesianOptimization(build_1d_cnn_model,
                                     objective='val_accuracy',
                                     max_trials=max_trials,
                                     directory=directory,
                                     project_name='pcg-classification')

        tuner.search(x_train, y_train,
                     epochs=epochs,
                     validation_data=(x_val, y_val))

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)
        history = best_model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val))
        val_accuracy = history.history['val_accuracy'][-1]
        return best_hps.values, val_accuracy


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
                         dense_1: int, dense_2: int, filter_1_l2: float, filter_2_l2: float, dense_1_l2: float,
                         dense_2_l2: float, dropout_rate: float, learning_rate: float, loss: str, patience: int,
                         epochs: int, batch_size: int):
        """Function to build, compile and fit 1D-CNN Model. It returns Model and History."""
        # Build 1D-CNN Model:
        cnn_model = Sequential()
        cnn_model.add(Conv1D(filters=filter_1,
                             kernel_size=3,
                             activation='relu',
                             input_shape=input_shape,
                             kernel_regularizer=l2(filter_1_l2)))
        cnn_model.add(MaxPooling1D(pool_size=2))
        cnn_model.add(Conv1D(filters=filter_2,
                             kernel_size=3,
                             activation='relu',
                             kernel_regularizer=l2(filter_2_l2)))
        cnn_model.add(MaxPooling1D(pool_size=2))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(dense_1,
                            activation='relu',
                            kernel_regularizer=l2(dense_1_l2)))
        cnn_model.add(layers.Dropout(dropout_rate))
        cnn_model.add(Dense(dense_2,
                            activation='relu',
                            kernel_regularizer=l2(dense_2_l2)))
        cnn_model.add(Dense(1, activation='sigmoid'))

        # Compile the Model:
        optimizer = optimizers.Adam(learning_rate=learning_rate)
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

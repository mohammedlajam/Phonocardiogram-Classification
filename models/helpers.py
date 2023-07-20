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
from sklearn.model_selection import GridSearchCV
from pytorch_tabnet.tab_model import TabNetClassifier

import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.layers import LSTM, Dense, Dropout, Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Flatten
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.applications import ResNet50, VGG19, InceptionV3, InceptionResNetV2
from keras_tuner.tuners import BayesianOptimization
import optuna
import torch
import torch.optim as optim

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, \
    roc_curve, auc

import warnings
warnings.filterwarnings('ignore')

# disabling ssl:
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class HyperParametersTuner:
    def __init__(self):
        pass

    # 1. Tabular Data:
    @staticmethod
    def find_best_tabular_svm_hp(x_train, y_train, rand_state: int):
        """Function to optimize Hyper-Parameters  for Support Vector Machine using GridSearchCV."""
        # Define parameter grid:
        param_grid = {
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'gamma': [0.1, 1, 10],
            'C': [0.1, 1, 10, 100]
        }

        # Building the Model:
        svm_model = svm.SVC(random_state=rand_state)

        # Performing Grid Search Cross Validation
        clf = GridSearchCV(svm_model, param_grid, n_jobs=-1)
        clf.fit(x_train, y_train)

        # Getting the best parameters and accuracy score of the best model
        best_hp = clf.best_params_
        best_score = pd.DataFrame({'Score': [clf.best_score_]})

        return best_hp, best_score

    @staticmethod
    def find_best_tabular_tabnet_hp(x_train, x_val, y_train, y_val, max_epochs: int,
                                    max_trials: int, batch_size: int):
        """Function to optimize Hyper-Parameters for TabNet Model using Optuna."""
        def objective(trial):
            # Defining Hyper-Parameters to be tuned:
            n_d = trial.suggest_categorical('n_d', [8, 16, 32])
            n_a = trial.suggest_categorical('n_a', [8, 16, 32])
            n_steps = trial.suggest_int('n_steps', 1, 10, step=1)
            gamma = trial.suggest_float('gamma', 0.1, 5.0, step=1)
            n_independent = trial.suggest_int('n_independent', 1, 5, step=1)
            n_shared = trial.suggest_categorical('n_shared', [2, 4])
            learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.0001])
            weight_decay = trial.suggest_categorical('weight_decay', [0.0, 0.1, 0.001])
            mask_type = trial.suggest_categorical('mask_type', ['sparsemax', 'entmax'])

            # Build and train the TabNet model
            tabnet_model = TabNetClassifier(n_d=n_d,
                                            n_a=n_a,
                                            n_steps=n_steps,
                                            gamma=gamma,
                                            n_independent=n_independent,
                                            n_shared=n_shared,
                                            optimizer_fn=optim.Adam,
                                            optimizer_params=dict(lr=learning_rate, weight_decay=weight_decay),
                                            scheduler_params=dict(step_size=n_steps, gamma=gamma),
                                            scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                            mask_type=mask_type)

            tabnet_model.fit(X_train=x_train,
                             y_train=y_train,
                             eval_set=[(x_train, y_train), (x_val, y_val)],
                             eval_name=['train', 'valid'],
                             eval_metric=['accuracy'],
                             max_epochs=max_epochs,
                             batch_size=batch_size,
                             drop_last=False)

            # Returning the best validation accuracy of a trial:
            return max(tabnet_model.history['valid_accuracy'])

        # Define the search space for the hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=max_trials)
        # Get the best Hyper-Parameters and the corresponding validation accuracy:
        best_hyperparams = study.best_params
        best_score = study.best_value

        return best_hyperparams, best_score

    @staticmethod
    def find_best_tabular_mlp_hp(x_train, x_val, y_train, y_val, input_shape, max_trials: int,
                                 epochs: int, directory: str):
        """Function to optimize Hyper-Parameters for MLP Model using Keras.tuner."""

        # Building the Model with Hyper-Parameters to be tuned:
        def build_tabular_mlp_model(hp):
            mlp_model = Sequential()
            mlp_model.add(Dense(hp.Int('dense_1', min_value=32, max_value=512, step=32),
                                activation='relu',
                                input_shape=input_shape,
                                kernel_regularizer=l2(hp.Choice('dense_1_l2', values=[0.0, 0.01, 0.001]))))
            mlp_model.add(Dropout(hp.Float('dropout_rate_1', min_value=0.0, max_value=0.5, step=0.1)))
            mlp_model.add(Dense(hp.Int('dense_2', min_value=32, max_value=512, step=32),
                                activation='relu',
                                kernel_regularizer=l2(hp.Choice('dense_2_l2', values=[0.0, 0.01, 0.001]))))
            mlp_model.add(Dropout(hp.Float('dropout_rate_2', min_value=0.0, max_value=0.5, step=0.1)))
            mlp_model.add(Dense(1, activation='sigmoid'))

            # Tune the optimizer and learning rate
            optimizer = optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[0.001, 0.0001]))
            mlp_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            return mlp_model

        # Setting up the Model and Finding the best Hyper-Parameter combination based on Val_accuracy:
        tuner = BayesianOptimization(build_tabular_mlp_model,
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

    @staticmethod
    def find_best_tabular_cnn_hp(x_train, x_val, y_train, y_val, input_shape, max_trials: int,
                                 epochs: int, directory: str):
        """Function to optimize Hyper-Parameters for 1D-CNN Model using Keras.tuner."""

        # Building the Model with Hyper-Parameters to be tuned:
        def build_tabular_cnn_model(hp):
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
        tuner = BayesianOptimization(build_tabular_cnn_model,
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

    @staticmethod
    def find_best_tabular_lstm_hp(x_train, x_val, y_train, y_val, input_shape, max_trials: int,
                                  epochs: int, directory: str):
        """Function to optimize Hyper-Parameters for RNN-LSTM Model using Keras.tuner."""

        # Building the Model with Hyper-Parameters to be tuned:
        def build_tabular_lstm_model(hp):
            lstm_model = Sequential()
            lstm_model.add(LSTM(units=hp.Int('lstm_1', min_value=32, max_value=256, step=32),
                                input_shape=input_shape,
                                kernel_regularizer=l2(hp.Choice('lstm_1_l2', values=[0.0, 0.01, 0.001])),
                                return_sequences=True))
            lstm_model.add(Dropout(hp.Float('dropout_rate_1', min_value=0.0, max_value=0.5, step=0.1)))
            lstm_model.add(LSTM(units=hp.Int('lstm_2', min_value=32, max_value=256, step=32),
                                kernel_regularizer=l2(hp.Choice('lstm_2_l2', values=[0.0, 0.01, 0.001]))))
            lstm_model.add(Dropout(hp.Float('dropout_rate_2', min_value=0.0, max_value=0.5, step=0.1)))
            lstm_model.add(Dense(1, activation='sigmoid'))

            # Tune the optimizer and learning rate
            optimizer = optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[0.001, 0.0001]))
            lstm_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            return lstm_model

        # Setting up the Model and Finding the best Hyper-Parameter combination based on Val_accuracy:
        tuner = BayesianOptimization(build_tabular_lstm_model,
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

    @staticmethod
    def find_best_tabular_crnn_hp(x_train, x_val, y_train, y_val, input_shape, max_trials: int,
                                  epochs: int, directory: str):
        """Function to optimize Hyper-Parameters for C-RNN Model using Keras.tuner."""

        # Building the Model with Hyper-Parameters to be tuned:
        def build_tabular_crnn_model(hp):
            crnn_model = Sequential()
            # Adding Convolutional Layers:
            crnn_model.add(Conv1D(filters=hp.Int('filter_1', min_value=64, max_value=256, step=32),
                                  kernel_size=3,
                                  activation='relu',
                                  input_shape=input_shape,
                                  kernel_regularizer=l2(hp.Choice('filter_1_l2', values=[0.0, 0.01, 0.001]))))
            crnn_model.add(MaxPooling1D(pool_size=2))
            crnn_model.add(Conv1D(filters=hp.Int('filter_2', min_value=64, max_value=256, step=32),
                                  kernel_size=3,
                                  activation='relu',
                                  kernel_regularizer=l2(hp.Choice('filter_2_l2', values=[0.0, 0.01, 0.001]))))
            crnn_model.add(MaxPooling1D(pool_size=2))

            # Adding LSTM Layers:
            crnn_model.add(LSTM(units=hp.Int('lstm_1', min_value=32, max_value=256, step=32),
                                input_shape=input_shape,
                                kernel_regularizer=l2(hp.Choice('lstm_1_l2', values=[0.0, 0.01, 0.001])),
                                return_sequences=True))
            crnn_model.add(Dropout(hp.Float('dropout_rate_1', min_value=0.0, max_value=0.5, step=0.1)))
            crnn_model.add(LSTM(units=hp.Int('lstm_2', min_value=32, max_value=256, step=32),
                                kernel_regularizer=l2(hp.Choice('lstm_2_l2', values=[0.0, 0.01, 0.001]))))
            crnn_model.add(Dropout(hp.Float('dropout_rate_2', min_value=0.0, max_value=0.5, step=0.1)))
            crnn_model.add(Flatten())
            # Adding Dense Layer as an Output:
            crnn_model.add(Dense(1, activation='sigmoid'))

            # Tune the optimizer and learning rate
            optimizer = optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[0.001, 0.0001]))
            crnn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            return crnn_model

        # Setting up the Model and Finding the best Hyper-Parameter combination based on Val_accuracy:
        tuner = BayesianOptimization(build_tabular_crnn_model,
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

    # 2. Computer Vision:
    @staticmethod
    def find_best_cv_mlp_hp(x_train, x_val, y_train, y_val, input_shape, max_trials: int,
                            epochs: int, directory: str):
        """Function to optimize Hyper-Parameters for MLP Model using Keras Tuner for image data."""

        # Reshape input data to flatten the images
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_val_flat = x_val.reshape(x_val.shape[0], -1)
        input_shape_flat = x_train_flat.shape[1:]

        # Building the Model with Hyper-Parameters to be tuned:
        def build_image_mlp_model(hp):
            mlp_model = Sequential()

            mlp_model.add(layers.Dense(hp.Int('dense_1', min_value=32, max_value=512, step=32),
                                       activation='relu',
                                       input_shape=input_shape_flat,
                                       kernel_regularizer=l2(
                                           hp.Choice('dense_1_l2', values=[0.0, 0.01, 0.001]))))
            mlp_model.add(layers.Dropout(hp.Float('dropout_rate_1', min_value=0.0, max_value=0.5, step=0.1)))

            mlp_model.add(layers.Dense(hp.Int('dense_2', min_value=32, max_value=512, step=32),
                                       activation='relu',
                                       kernel_regularizer=l2(
                                           hp.Choice('dense_2_l2', values=[0.0, 0.01, 0.001]))))
            mlp_model.add(layers.Dropout(hp.Float('dropout_rate_2', min_value=0.0, max_value=0.5, step=0.1)))

            mlp_model.add(layers.Dense(1, activation='sigmoid'))

            # Tune the optimizer and learning rate
            optimizer = optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[0.001, 0.0001]))
            mlp_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            return mlp_model

        # Setting up the Model and Finding the best Hyper-Parameter combination based on Val_accuracy:
        tuner = BayesianOptimization(build_image_mlp_model,
                                     objective='val_accuracy',
                                     max_trials=max_trials,
                                     directory=directory,
                                     project_name='image-mlp-classification')

        tuner.search(x_train_flat, y_train,
                     epochs=epochs,
                     validation_data=(x_val_flat, y_val))

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)
        history = best_model.fit(x_train_flat, y_train, epochs=epochs, validation_data=(x_val_flat, y_val))
        val_accuracy = history.history['val_accuracy'][-1]
        return best_hps.values, val_accuracy

    @staticmethod
    def find_best_cv_cnn_hp(x_train, x_val, y_train, y_val, input_shape, max_trials: int,
                            epochs: int, directory: str):
        """Function to optimize Hyper-Parameters for 2D-CNN Model using Keras.tuner."""

        # Building the Model with Hyper-Parameters to be tuned:
        def build_cv_cnn_model(hp):
            cnn_model = Sequential()
            cnn_model.add(Conv2D(filters=hp.Int('filter_1', min_value=32, max_value=128, step=32),
                                 kernel_size=3,
                                 activation='relu',
                                 input_shape=input_shape,
                                 kernel_regularizer=l2(hp.Choice('filter_1_l2', values=[0.0, 0.01, 0.001]))))
            cnn_model.add(MaxPooling2D(pool_size=2))
            cnn_model.add(Conv2D(filters=hp.Int('filter_2', min_value=32, max_value=128, step=32),
                                 kernel_size=3,
                                 activation='relu',
                                 kernel_regularizer=l2(hp.Choice('filter_2_l2', values=[0.0, 0.01, 0.001]))))
            cnn_model.add(MaxPooling2D(pool_size=2))
            cnn_model.add(Flatten())
            cnn_model.add(Dense(hp.Int('dense_1', min_value=64, max_value=512, step=64),
                                activation='relu',
                                kernel_regularizer=l2(hp.Choice('dense_1_l2', values=[0.0, 0.01, 0.001]))))
            cnn_model.add(Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))
            cnn_model.add(Dense(hp.Int('dense_2', min_value=64, max_value=512, step=64),
                                activation='relu',
                                kernel_regularizer=l2(hp.Choice('dense_2_l2', values=[0.0, 0.01, 0.001]))))
            cnn_model.add(Dense(1, activation='sigmoid'))

            # Tune the optimizer and learning rate
            optimizer = optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[0.001, 0.0001]))
            cnn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            return cnn_model

        # Setting up the Model and Finding the best Hyper-Parameter combination based on Val_accuracy:
        tuner = BayesianOptimization(build_cv_cnn_model,
                                     objective='val_accuracy',
                                     max_trials=max_trials,
                                     directory=directory,
                                     project_name='image-classification')

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

    # 1. Tabular Data:
    @staticmethod
    def build_fit_tabular_svm(x_train, y_train, kernel: str, gamma: str, c: float, rand_state: int):
        """Function to build and fit Support Vector Machine."""
        # Building the Model:
        svm_model = svm.SVC(kernel=kernel, gamma=gamma, C=c, random_state=rand_state)
        # Fitting the Model:
        svm_model.fit(x_train, y_train)
        return svm_model

    @staticmethod
    def build_fit_tabular_tabnet(x_train, x_val, y_train, y_val, n_d: int, n_a: int, n_steps: int, gamma: float,
                                 n_ind: int, n_shared: int, learning_rate: float, weight_decay: float, mask_type: str,
                                 epochs: int, patience: int, batch_size: int):
        """Function to build and fit TabNet Model."""
        # Building the Model:
        tb_model = TabNetClassifier(n_d=n_d,
                                    n_a=n_a,
                                    n_steps=n_steps,
                                    gamma=gamma,
                                    n_independent=n_ind,
                                    n_shared=n_shared,
                                    optimizer_fn=torch.optim.Adam,
                                    optimizer_params=dict(lr=learning_rate, weight_decay=weight_decay),
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
    def build_fit_tabular_mlp(x_train, x_val, y_train, y_val, input_shape, dense_1: int, dense_2: int,
                              dense_1_l2: float, dense_2_l2: float, dropout_rate_1: float, dropout_rate_2: float,
                              learning_rate: float, loss: str, patience: int, epochs: int, batch_size: int):
        """Function to build, compile and fit MLP Model. It returns Model and History."""
        # Build MLP Model:
        mlp_model = Sequential()
        mlp_model.add(Dense(dense_1,
                            input_shape=input_shape,
                            activation='relu',
                            kernel_regularizer=l2(dense_1_l2)))
        mlp_model.add(layers.Dropout(dropout_rate_1))
        mlp_model.add(Dense(dense_2,
                            activation='relu',
                            kernel_regularizer=l2(dense_2_l2)))
        mlp_model.add(layers.Dropout(dropout_rate_2))
        mlp_model.add(Dense(1, activation='sigmoid'))

        # Compile the Model:
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        mlp_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_accuracy',
                                   patience=patience,
                                   mode='max',
                                   restore_best_weights=True,
                                   verbose=True)

        # Fitting the Model:
        history = mlp_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                                callbacks=[early_stop])
        return mlp_model, history

    @staticmethod
    def build_fit_tabular_cnn(x_train, x_val, y_train, y_val, input_shape, filter_1: int, filter_2: int,
                              dense_1: int, dense_2: int, filter_1_l2: float, filter_2_l2: float,
                              dense_1_l2: float, dense_2_l2: float, dropout_rate: float, learning_rate: float,
                              loss: str, patience: int, epochs: int, batch_size: int):
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

        early_stop = EarlyStopping(monitor='val_accuracy',
                                   patience=patience,
                                   mode='max',
                                   restore_best_weights=True,
                                   verbose=True)

        # Fitting the Model:
        history = cnn_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                                callbacks=[early_stop])
        return cnn_model, history

    @staticmethod
    def build_fit_tabular_lstm(x_train, x_val, y_train, y_val, input_shape, lstm_1: int, lstm_2: int,
                               lstm_1_l2: float, lstm_2_l2: float, dropout_rate_1: float, dropout_rate_2: float,
                               learning_rate: float, loss: str, patience: int, epochs: int, batch_size: int):
        """Function to build, compile and fit LSTM Model with 2 LSTM layers. It returns Model and History."""
        # Build LSTM Model:
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=int(lstm_1),
                            input_shape=input_shape,
                            kernel_regularizer=l2(lstm_1_l2),
                            return_sequences=True))
        lstm_model.add(Dropout(dropout_rate_1))
        lstm_model.add(LSTM(units=int(lstm_2),
                            kernel_regularizer=l2(lstm_2_l2)))
        lstm_model.add(Dropout(dropout_rate_2))
        lstm_model.add(Dense(1, activation='sigmoid'))

        # Compile the Model:
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        lstm_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_accuracy',
                                   patience=patience,
                                   mode='max',
                                   restore_best_weights=True,
                                   verbose=True)

        # Fitting the Model:
        history = lstm_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                                 callbacks=[early_stop])
        return lstm_model, history

    @staticmethod
    def build_fit_tabular_crnn(x_train, x_val, y_train, y_val, input_shape, filter_1: int, filter_2: int,
                               lstm_1: int, lstm_2: int, filter_1_l2: float, filter_2_l2: float,
                               lstm_1_l2: float, lstm_2_l2: float, dropout_rate_1: float, dropout_rate_2: float,
                               learning_rate: float, loss: str, patience: int, epochs: int, batch_size: int):
        """Function to build, compile and fit C-RNN Model. It returns Model and History."""
        # Build 1D-CNN Model:
        # Adding Convolutional Layers:
        crnn_model = Sequential()
        crnn_model.add(Conv1D(filters=filter_1,
                              kernel_size=3,
                              activation='relu',
                              input_shape=input_shape,
                              kernel_regularizer=l2(filter_1_l2)))
        crnn_model.add(MaxPooling1D(pool_size=2))
        crnn_model.add(Conv1D(filters=filter_2,
                              kernel_size=3,
                              activation='relu',
                              kernel_regularizer=l2(filter_2_l2)))
        crnn_model.add(MaxPooling1D(pool_size=2))
        # Adding LSTM Layers:
        crnn_model.add(LSTM(units=int(lstm_1),
                            kernel_regularizer=l2(lstm_1_l2),
                            return_sequences=True))
        crnn_model.add(Dropout(dropout_rate_1))
        crnn_model.add(LSTM(units=int(lstm_2),
                            kernel_regularizer=l2(lstm_2_l2)))
        crnn_model.add(Dropout(dropout_rate_2))
        crnn_model.add(Flatten())
        # Adding Dense Layer as an Output:
        crnn_model.add(Dense(1, activation='sigmoid'))

        # Compile the Model:
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        crnn_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_accuracy',
                                   patience=patience,
                                   mode='max',
                                   restore_best_weights=True,
                                   verbose=True)

        # Fitting the Model:
        history = crnn_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                                 callbacks=[early_stop])
        return crnn_model, history

    # 2. Computer Vision:
    @staticmethod
    def build_fit_cv_mlp(x_train, x_val, y_train, y_val, input_shape, dense_1: int, dense_2: int,
                         dense_1_l2: float, dense_2_l2: float, dropout_rate_1: float, dropout_rate_2: float,
                         learning_rate: float, loss: str, patience: int, epochs: int, batch_size: int):
        """Function to build, compile, and fit MLP Model for image data. It returns the model and history."""

        # Reshape input data to flatten the images
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_val_flat = x_val.reshape(x_val.shape[0], -1)
        input_shape_flat = x_train_flat.shape[1:]

        # Build MLP Model:
        mlp_model = Sequential()
        mlp_model.add(layers.Dense(dense_1,
                                   input_shape=input_shape_flat,
                                   activation='relu',
                                   kernel_regularizer=l2(dense_1_l2)))
        mlp_model.add(layers.Dropout(dropout_rate_1))
        mlp_model.add(layers.Dense(dense_2,
                                   activation='relu',
                                   kernel_regularizer=l2(dense_2_l2)))
        mlp_model.add(layers.Dropout(dropout_rate_2))
        mlp_model.add(layers.Dense(1, activation='sigmoid'))

        # Compile the Model:
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        mlp_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_accuracy',
                                   patience=patience,
                                   mode='max',
                                   restore_best_weights=True,
                                   verbose=True)

        # Fitting the Model:
        history = mlp_model.fit(x_train_flat, y_train, epochs=epochs, batch_size=batch_size,
                                validation_data=(x_val_flat, y_val),
                                callbacks=[early_stop])
        return mlp_model, history

    @staticmethod
    def build_fit_cv_cnn(train_dataset, val_dataset, input_shape, filter_1: int, filter_2: int,
                         dense_1: int, dense_2: int, filter_1_l2: float, filter_2_l2: float,
                         dense_1_l2: float, dense_2_l2: float, dropout_rate: float, learning_rate: float,
                         loss: str, patience: int, epochs: int):
        """Function to build, compile and fit 2D-CNN Model. It returns Model and History."""
        # Build 2D-CNN Model:
        cnn_model = Sequential()
        cnn_model.add(Conv2D(filters=filter_1,
                             kernel_size=(3, 3),
                             activation='relu',
                             input_shape=input_shape,
                             kernel_regularizer=l2(filter_1_l2)))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
        cnn_model.add(Conv2D(filters=filter_2,
                             kernel_size=(3, 3),
                             activation='relu',
                             kernel_regularizer=l2(filter_2_l2)))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(dense_1,
                            activation='relu',
                            kernel_regularizer=l2(dense_1_l2)))
        cnn_model.add(Dropout(dropout_rate))
        cnn_model.add(Dense(dense_2,
                            activation='relu',
                            kernel_regularizer=l2(dense_2_l2)))
        cnn_model.add(Dense(1, activation='sigmoid'))

        # Compile the Model:
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        cnn_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_accuracy',
                                   patience=patience,
                                   mode='max',
                                   restore_best_weights=True,
                                   verbose=True)

        # Fitting the Model:
        history = cnn_model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[early_stop])
        return cnn_model, history


class PretrainedModel:
    def __init__(self):
        pass

    @staticmethod
    def build_fit_resnet50(train_dataset, val_dataset, input_shape, include_top: bool, resnet_weights: str,
                           trainable: bool, dense_1: int, dense_2: int, dense_1_l2: float, dense_2_l2: float,
                           dropout_rate_1: float, dropout_rate_2: float, learning_rate: float, loss: str,
                           patience: int, epochs: int):
        """Function to build, compile and fit ResNet50 Model. It returns Model and History."""
        base_model = ResNet50(include_top=include_top,
                              weights=resnet_weights,
                              input_shape=input_shape)
        base_model.trainable = trainable

        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(units=dense_1,
                        activation='relu',
                        kernel_regularizer=l2(dense_1_l2)))
        model.add(Dropout(dropout_rate_1))
        model.add(Dense(units=dense_2,
                        activation='relu',
                        kernel_regularizer=l2(dense_2_l2)))
        model.add(Dropout(dropout_rate_2))
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile the model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_accuracy',
                                       patience=patience,
                                       mode='max',
                                       restore_best_weights=True,
                                       verbose=True)

        # Train the model
        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[early_stopping])

        return model, history

    @staticmethod
    def build_fit_vgg19(train_dataset, val_dataset, input_shape, include_top: bool, vgg_weights: str,
                        trainable: bool, dense_1: int, dense_2: int, dense_1_l2: float, dense_2_l2: float,
                        dropout_rate_1: float, dropout_rate_2: float, learning_rate: float, loss: str,
                        patience: int, epochs: int):
        """Function to build, compile and fit VGG19 Model. It returns Model and History."""
        base_model = VGG19(include_top=include_top,
                           weights=vgg_weights,
                           input_shape=input_shape)
        base_model.trainable = trainable

        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(units=dense_1,
                        activation='relu',
                        kernel_regularizer=l2(dense_1_l2)))
        model.add(Dropout(dropout_rate_1))
        model.add(Dense(units=dense_2,
                        activation='relu',
                        kernel_regularizer=l2(dense_2_l2)))
        model.add(Dropout(dropout_rate_2))
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile the model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_accuracy',
                                       patience=patience,
                                       mode='max',
                                       restore_best_weights=True,
                                       verbose=True)

        # Train the model
        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[early_stopping])

        return model, history

    @staticmethod
    def build_fit_inception(train_dataset, val_dataset, input_shape, include_top: bool, inception_weights: str,
                            trainable: bool, dense_1: int, dense_2: int, dense_1_l2: float, dense_2_l2: float,
                            dropout_rate_1: float, dropout_rate_2: float, learning_rate: float, loss: str,
                            patience: int, epochs: int):
        """Function to build, compile and fit InceptionV3 Model. It returns Model and History."""
        base_model = InceptionV3(include_top=include_top,
                                 weights=inception_weights,
                                 input_shape=input_shape)
        base_model.trainable = trainable

        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(units=dense_1,
                        activation='relu',
                        kernel_regularizer=l2(dense_1_l2)))
        model.add(Dropout(dropout_rate_1))
        model.add(Dense(units=dense_2,
                        activation='relu',
                        kernel_regularizer=l2(dense_2_l2)))
        model.add(Dropout(dropout_rate_2))
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile the model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_accuracy',
                                       patience=patience,
                                       mode='max',
                                       restore_best_weights=True,
                                       verbose=True)

        # Train the model
        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[early_stopping])

        return model, history

    @staticmethod
    def build_fit_inceptionresnet(train_dataset, val_dataset, input_shape, include_top: bool,
                                  inceptionresnet_weights: str, trainable: bool, dense_1: int, dense_2: int,
                                  dense_1_l2: float, dense_2_l2: float, dropout_rate_1: float, dropout_rate_2: float,
                                  learning_rate: float, loss: str, patience: int, epochs: int):
        """Function to build, compile and fit InceptionResNetV2 Model. It returns Model and History."""
        base_model = InceptionResNetV2(include_top=include_top,
                                       weights=inceptionresnet_weights,
                                       input_shape=input_shape)
        base_model.trainable = trainable

        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(units=dense_1,
                        activation='relu',
                        kernel_regularizer=l2(dense_1_l2)))
        model.add(Dropout(dropout_rate_1))
        model.add(Dense(units=dense_2,
                        activation='relu',
                        kernel_regularizer=l2(dense_2_l2)))
        model.add(Dropout(dropout_rate_2))
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile the model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_accuracy',
                                       patience=patience,
                                       mode='max',
                                       restore_best_weights=True,
                                       verbose=True)

        # Train the model
        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[early_stopping])

        return model, history


# 2. Class for calculating Predictions and evaluating Model:
class ModelEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def calculate_probabilities(model, x_test):
        """Function to predict the model and returns the predictions."""
        # Reshape the input data to match the expected shape
        #x_test = np.reshape(x_test, (x_test.shape[0], -1))

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

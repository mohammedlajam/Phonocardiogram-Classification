"""
Author: Mohammed Lajam

Phase 5: Classification (Tabular Data - TabNet):
- In this python file, the audio data is imported from the '/data/cross_validation/selected_features/'
directory.
- After denoising the signals, Slicing the signals takes place to a specific length, so that all
signals are equal in length and therefore the number of samples per signals are equal.

Objective:
- Building, predicting, evaluating and tuning TabNet Classifier Model and save the best model
with respect to Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity and ROC_AUC Score.

Input:
- The input data is the 6 Folds of X_TRAIN, X_TEST, Y_TRAIN, Y_TEST.

Output:
- The outputs are Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity and
ROC_AUC Score saved in 'mlruns' in the same directory. Additionally, all the Artifacts, parameters
and matrices are tracked using MLFlow.

Note:
- In 'constants.py', 'CROSS_VALIDATION' path has to be adjusted before running this file.
- All the functions and variables, which are used in this file, are imported from helpers.py
file from the 'models' package in the same repository.
"""
# Importing libraries:
import os
import pickle
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

import constants as c
from models.helpers import *


# Functions to be used in this python file:
def _check_create_dir():
    """Function to check if 'cross_validation' directory exists in 'data' directory and create
    it if it is not existed."""
    try:
        os.path.isdir(c.CROSS_VALIDATION_PATH)
    except FileNotFoundError:
        print("Directory does not exist!")
    except Exception as e:
        print(f"Error: {e}")
    else:
        return None


def _load_cv_folds():
    """Function to load Cross Validation Folds from local machine."""
    try:
        with open(f'{c.CROSS_VALIDATION_PATH}/selected_features/x_train_folds.pkl', 'rb') as f:
            x_train_folds = pickle.load(f)
        with open(f'{c.CROSS_VALIDATION_PATH}/selected_features/x_test_folds.pkl', 'rb') as f:
            x_test_folds = pickle.load(f)
        with open(f'{c.CROSS_VALIDATION_PATH}/selected_features/y_train_folds.pkl', 'rb') as f:
            y_train_folds = pickle.load(f)
        with open(f'{c.CROSS_VALIDATION_PATH}/selected_features/y_test_folds.pkl', 'rb') as f:
            y_test_folds = pickle.load(f)
    except FileNotFoundError:
        print("Error: One or more files not found")
    except Exception as e:
        print(f"Error: {e}")
    else:
        return x_train_folds, x_test_folds, y_train_folds, y_test_folds


def _data_per_fold(x_train, x_test, y_train, y_test):
    """Function to prepare the data for TabNet Model."""
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    y_train = y_train.flatten()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    # Converting all -1 to 0 in Y_TRAIN and Y_TEST:
    y_train = np.where(y_train == -1, 0, y_train)
    y_test = np.where(y_test == -1, 0, y_test)

    # Splitting x_train and y_train into x_train, x_val, y_train, y_val:
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=c.TEST_SIZE,
                                                      random_state=c.RANDOM_STATE)
    return x_train, x_test, x_val, y_train, y_test, y_val


def _run_evaluate_tabnet(x_train_folds, x_test_folds, y_train_folds, y_test_folds):
    with mlflow.start_run():
        # Creating Empty Lists for all Evaluation Matrices:
        threshold_folds, accuracy_folds, precision_folds, recall_folds, f1_score_folds = [], [], [], [], []
        sensitivity_folds, specificity_folds = [], []
        tn_folds, fp_folds, fn_folds, tp_folds = [], [], [], []
        roc_auc_folds, auc_score_folds = [], []
        fpr_folds, tpr_folds = [], []
        cm_folds = []

        for fold in range(len(x_train_folds)):
            # Extracting the data per fold:
            x_train, x_test, x_val, y_train, y_test, y_val = _data_per_fold(x_train=x_train_folds[fold],
                                                                            x_test=x_test_folds[fold],
                                                                            y_train=y_train_folds[fold],
                                                                            y_test=y_test_folds[fold])

            # 1. Building and fitting TabNet Model:
            tb_model = ModelBuilder.build_fit_tabnet(x_train=x_train,
                                                     x_val=x_val,
                                                     y_train=y_train,
                                                     y_val=y_val,
                                                     n_d=c.TB_N_D,
                                                     n_a=c.TB_N_A,
                                                     n_steps=c.TB_N_STEPS,
                                                     gamma=c.TB_GAMMA,
                                                     n_ind=c.TB_N_IND,
                                                     n_shared=c.TB_N_SHARED,
                                                     learning_rate=c.TB_LEARNING_RATE,
                                                     mask_type=c.TB_MASK_TYPE,
                                                     epochs=c.TB_EPOCHS,
                                                     patience=c.TB_PATIENCE,
                                                     batch_size=c.TB_BATCH_SIZE)

            # Plotting the Accuracy each Epoch and save it in mlflow:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(tb_model.history['train_accuracy'], label=f'Fold_{fold+1} Train Accuracy')
            ax.plot(tb_model.history['valid_accuracy'], label=f'Fold_{fold+1} Validation Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy over Epochs')
            ax.legend()
            mlflow.log_figure(fig, f'Accuracy_vs_Epochs (Fold_{fold+1}).png')  # Log the plot in mlflow

            # 2. Predictions:
            # Calculating the Probabilities:
            y_prob = ModelEvaluator.calculate_probabilities(model=tb_model, x_test=x_test)

            # Finding the best threshold that provides the best accuracy:
            _, best_threshold = ModelEvaluator.find_best_threshold(y_test=y_test, y_prob=y_prob)

            # Calculating Evaluation Matrix:
            evaluation_matrix = ModelEvaluator.evaluate_model(y_test=y_test,
                                                              y_prob=y_prob,
                                                              threshold=best_threshold)

            # Generating Confusion Matrix:
            cm = ModelEvaluator.generate_confusion_matrix(y_test=y_test,
                                                          y_prob=y_prob,
                                                          threshold=best_threshold)

            # Extracting evaluations matrices based on threshold, which provides the best accuracy:
            threshold_folds.append(evaluation_matrix.iloc[0]['Threshold'])
            accuracy_folds.append(evaluation_matrix.iloc[0]['Accuracy'])
            precision_folds.append(evaluation_matrix.iloc[0]['Precision'])
            recall_folds.append(evaluation_matrix.iloc[0]['Recall'])
            f1_score_folds.append(evaluation_matrix.iloc[0]['F1-score'])
            sensitivity_folds.append(evaluation_matrix.iloc[0]['Sensitivity'])
            specificity_folds.append(evaluation_matrix.iloc[0]['Specificity'])
            tn_folds.append(evaluation_matrix.iloc[0]['TN'])
            fp_folds.append(evaluation_matrix.iloc[0]['FP'])
            fn_folds.append(evaluation_matrix.iloc[0]['FN'])
            tp_folds.append(evaluation_matrix.iloc[0]['TP'])
            roc_auc_folds.append(evaluation_matrix.iloc[0]['ROC_AUC'])
            auc_score_folds.append(evaluation_matrix.iloc[0]['AUC_Score'])
            fpr_folds.append(evaluation_matrix.iloc[0]['FPR'])
            tpr_folds.append(evaluation_matrix.iloc[0]['TPR'])
            cm_folds.append(cm)

            # 3. Calculating the Average of all Evaluation Matrices:
            mean_threshold = sum(threshold_folds) / len(threshold_folds)
            mean_accuracy = sum(accuracy_folds) / len(accuracy_folds)
            mean_precision = sum(precision_folds) / len(precision_folds)
            mean_recall = sum(recall_folds) / len(recall_folds)
            mean_f1_score = sum(f1_score_folds) / len(f1_score_folds)
            mean_sensitivity = sum(sensitivity_folds) / len(sensitivity_folds)
            mean_specificity = sum(specificity_folds) / len(specificity_folds)
            mean_tn = sum(tn_folds) / len(tn_folds)
            mean_fp = sum(fp_folds) / len(fp_folds)
            mean_fn = sum(fn_folds) / len(fn_folds)
            mean_tp = sum(tp_folds) / len(tp_folds)
            mean_roc_auc = sum(roc_auc_folds) / len(roc_auc_folds)
            mean_auc_score = sum(auc_score_folds) / len(auc_score_folds)
            cm_mean = (sum(cm_folds) / len(cm_folds)).astype(int)

            # Interpolating FPR and TPR to 500 thresholds, because they have different lengths:
            mean_thresholds = np.linspace(0, 1, 6000)

            # Interpolate the FPR and TPR values for each fold to the mean thresholds
            fpr_interpolated = []
            tpr_interpolated = []
            for fold_index in range(len(fpr_folds)):
                fpr_interpolated.append(np.interp(mean_thresholds, fpr_folds[fold_index], tpr_folds[fold_index]))
                tpr_interpolated.append(np.interp(mean_thresholds, fpr_folds[fold_index], tpr_folds[fold_index]))

            # Calculate the mean FPR and TPR across the folds
            mean_fpr = np.mean(fpr_interpolated, axis=0)
            mean_tpr = np.mean(tpr_interpolated, axis=0)

        # 4. logging all Artifacts, Parameters and Matrices into mlflow:
        # Log the Hyper-Parameters:
        mlflow.log_param('n_d', c.TB_N_D)
        mlflow.log_param('n_a', c.TB_N_A)
        mlflow.log_param('n_steps', c.TB_N_STEPS)
        mlflow.log_param('Gamma', c.TB_GAMMA)
        mlflow.log_param('n_ind', c.TB_N_IND)
        mlflow.log_param('n_shared', c.TB_N_SHARED)
        mlflow.log_param('n_steps', c.TB_N_STEPS)
        mlflow.log_param('learning_rate', c.TB_LEARNING_RATE)
        mlflow.log_param('mask_type', c.TB_MASK_TYPE)
        mlflow.log_param('epochs', c.TB_EPOCHS)
        mlflow.log_param('patience', c.TB_PATIENCE)
        mlflow.log_param('batch_size', c.TB_BATCH_SIZE)

        # Log the Matrices (Evaluation):
        mlflow.log_metric('Threshold', mean_threshold)
        mlflow.log_metric('Accuracy', mean_accuracy)
        mlflow.log_metric('Precision', mean_precision)
        mlflow.log_metric('Recall', mean_recall)
        mlflow.log_metric('F1_Score', mean_f1_score)
        mlflow.log_metric('Sensitivity', mean_sensitivity)
        mlflow.log_metric('Specificity', mean_specificity)
        mlflow.log_metric('TN', mean_tn)
        mlflow.log_metric('FP', mean_fp)
        mlflow.log_metric('FN', mean_fn)
        mlflow.log_metric('TP', mean_tp)
        mlflow.log_metric('ROC_AUC', mean_roc_auc)
        mlflow.log_metric('AUC_Score', mean_auc_score)
        mlflow.log_param('FPR', mean_fpr)  # Saving in Parameters as they are a list
        mlflow.log_param('TPR', mean_tpr)  # Saving in Parameters as they are a list

        # Logging the Average Confusion Matrix into mlflow:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_mean, annot=True, fmt='d', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")  # Log the plot in mlflow

        # Save the Model
        mlflow.sklearn.log_model(tb_model, 'model')

        # End the mlflow run:
        mlflow.end_run()
    return None


if __name__ == "__main__":
    # Checking directory:
    _check_create_dir()

    # Loading Cross Validation Folds:
    X_TRAIN_FOLDS, X_TEST_FOLDS, Y_TRAIN_FOLDS, Y_TEST_FOLDS = _load_cv_folds()
    _run_evaluate_tabnet(x_train_folds=X_TRAIN_FOLDS,
                         x_test_folds=X_TEST_FOLDS,
                         y_train_folds=Y_TRAIN_FOLDS,
                         y_test_folds=Y_TEST_FOLDS)

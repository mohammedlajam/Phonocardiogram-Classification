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
import seaborn as sns
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


def _tune_hyper_parameters(x_train_folds, x_test_folds, y_train_folds, y_test_folds):
    """Function to tune Hyper-Parameters and return a DataFrame for the best Hyper-Parameters
    per fold."""
    best_hp_folds = []
    best_validation_folds = []
    for fold in range(len(X_TRAIN_FOLDS)):
        x_train, x_test, x_val, y_train, y_test, y_val = _data_per_fold(x_train=x_train_folds[fold],
                                                                        x_test=x_test_folds[fold],
                                                                        y_train=y_train_folds[fold],
                                                                        y_test=y_test_folds[fold])

        best_hp, best_validation = HyperParametersTuner.find_best_tabular_tabnet_hp(x_train=x_train,
                                                                                    x_val=x_val,
                                                                                    y_train=y_train,
                                                                                    y_val=y_val,
                                                                                    max_trials=c.TB_TABNET_HP_MAX_TRIALS,
                                                                                    max_epochs=c.TB_TABNET_HP_EPOCHS,
                                                                                    batch_size=c.TB_TABNET_BATCH_SIZE)
        best_hp_folds.append(best_hp)
        best_validation_folds.append(best_validation)

    # Creating one DataFrame including Hyper-Parameters and Validation Scores:
    best_hp_folds = pd.DataFrame(best_hp_folds)
    best_hp_folds['val_accuracy'] = best_validation_folds
    # Arranging the rows descendingly based on the val_accuracy:
    best_hp_folds = best_hp_folds.sort_values('val_accuracy', ascending=False)
    return best_hp_folds


def _run_evaluate_tabnet_automatic_hp(x_train_folds, x_test_folds, y_train_folds, y_test_folds, best_hp_folds):
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
            tb_model = ModelBuilder.build_fit_tabular_tabnet(x_train=x_train,
                                                             x_val=x_val,
                                                             y_train=y_train,
                                                             y_val=y_val,
                                                             n_d=best_hp_folds.iloc[0]['n_d'],
                                                             n_a=best_hp_folds.iloc[0]['n_a'],
                                                             n_steps=best_hp_folds.iloc[0]['n_steps'],
                                                             gamma=best_hp_folds.iloc[0]['gamma'],
                                                             n_ind=best_hp_folds.iloc[0]['n_independent'],
                                                             n_shared=best_hp_folds.iloc[0]['n_shared'],
                                                             learning_rate=best_hp_folds.iloc[0]['learning_rate'],
                                                             weight_decay=best_hp_folds.iloc[0]['weight_decay'],
                                                             mask_type=best_hp_folds.iloc[0]['mask_type'],
                                                             epochs=c.TB_TABNET_EPOCHS,
                                                             patience=c.TB_TABNET_PATIENCE,
                                                             batch_size=c.TB_TABNET_BATCH_SIZE)

            # Save the Model
            mlflow.sklearn.log_model(tb_model, f'tabnet_model_{fold}')

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
        mean_cm = (sum(cm_folds) / len(cm_folds)).astype(int)

        # 4. Calculating the mean of all Evaluation Matrices:
        std_threshold = np.std(threshold_folds)
        std_accuracy = np.std(accuracy_folds)
        std_precision = np.std(precision_folds)
        std_recall = np.std(recall_folds)
        std_f1_score = np.std(f1_score_folds)
        std_sensitivity = np.std(sensitivity_folds)
        std_specificity = np.std(specificity_folds)
        std_tn = np.std(tn_folds)
        std_fp = np.std(fp_folds)
        std_fn = np.std(fn_folds)
        std_tp = np.std(tp_folds)
        std_roc_auc = np.std(roc_auc_folds)
        std_auc_score = np.std(auc_score_folds)

        # 5. logging all Artifacts, Parameters and Matrices into mlflow:
        # Log the Hyper-Parameters:
        mlflow.log_param('n_d', best_hp_folds.iloc[0]['n_d'])
        mlflow.log_param('n_a', best_hp_folds.iloc[0]['n_a'])
        mlflow.log_param('n_steps', best_hp_folds.iloc[0]['n_steps'])
        mlflow.log_param('gamma', best_hp_folds.iloc[0]['gamma'])
        mlflow.log_param('n_ind', best_hp_folds.iloc[0]['n_independent'])
        mlflow.log_param('n_shared', best_hp_folds.iloc[0]['n_shared'])
        mlflow.log_param('learning_rate', best_hp_folds.iloc[0]['learning_rate'])
        mlflow.log_param('weight_decay', best_hp_folds.iloc[0]['weight_decay'])
        mlflow.log_param('mask_type', best_hp_folds.iloc[0]['mask_type'])
        mlflow.log_param('epochs', c.TB_TABNET_EPOCHS)
        mlflow.log_param('patience', c.TB_TABNET_PATIENCE)
        mlflow.log_param('batch_size', c.TB_TABNET_BATCH_SIZE)

        # Log the Matrices (Evaluation):
        # Mean of Matrices:
        mlflow.log_metric('mean_threshold', mean_threshold)
        mlflow.log_metric('mean_accuracy', mean_accuracy)
        mlflow.log_metric('mean_precision', mean_precision)
        mlflow.log_metric('mean_recall', mean_recall)
        mlflow.log_metric('mean_f1_score', mean_f1_score)
        mlflow.log_metric('mean_sensitivity', mean_sensitivity)
        mlflow.log_metric('mean_specificity', mean_specificity)
        mlflow.log_metric('mean_tn', mean_tn)
        mlflow.log_metric('mean_fp', mean_fp)
        mlflow.log_metric('mean_fn', mean_fn)
        mlflow.log_metric('mean_tp', mean_tp)
        mlflow.log_metric('mean_roc_auc', mean_roc_auc)
        mlflow.log_metric('mean_auc_score', mean_auc_score)

        # Standard Deviation of Matrices:
        mlflow.log_metric('std_threshold', float(std_threshold))
        mlflow.log_metric('std_accuracy', float(std_accuracy))
        mlflow.log_metric('std_precision', float(std_precision))
        mlflow.log_metric('std_recall', float(std_recall))
        mlflow.log_metric('std_f1_score', float(std_f1_score))
        mlflow.log_metric('std_sensitivity', float(std_sensitivity))
        mlflow.log_metric('std_specificity', float(std_specificity))
        mlflow.log_metric('std_tn', float(std_tn))
        mlflow.log_metric('std_fp', float(std_fp))
        mlflow.log_metric('std_fn', float(std_fn))
        mlflow.log_metric('std_tp', float(std_tp))
        mlflow.log_metric('std_roc_auc', float(std_roc_auc))
        mlflow.log_metric('std_auc_score', float(std_auc_score))

        # Saving fpr_folds and tpr_folds as pickle file and log it into mlflow as artifact:
        with open('tabular_tabnet_fpr_folds.pkl', 'wb') as f:
            pickle.dump(fpr_folds, f)
        mlflow.log_artifact('tabular_tabnet_fpr_folds.pkl', artifact_path='fpr_tpr')
        os.remove('tabular_tabnet_fpr_folds.pkl')

        with open('tabular_tabnet_tpr_folds.pkl', 'wb') as f:
            pickle.dump(tpr_folds, f)
        mlflow.log_artifact('tabular_tabnet_tpr_folds.pkl', artifact_path='fpr_tpr')
        os.remove('tabular_tabnet_tpr_folds.pkl')

        # Logging the Average Confusion Matrix into mlflow:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(mean_cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")  # Log the plot in mlflow

        # End the mlflow run:
        mlflow.end_run()
    return None


def _run_evaluate_tabnet_manual_hp(x_train_folds, x_test_folds, y_train_folds, y_test_folds):
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
            tb_model = ModelBuilder.build_fit_tabular_tabnet(x_train=x_train,
                                                             x_val=x_val,
                                                             y_train=y_train,
                                                             y_val=y_val,
                                                             n_d=c.TB_TABNET_N_D,
                                                             n_a=c.TB_TABNET_N_A,
                                                             n_steps=c.TB_TABNET_N_STEPS,
                                                             gamma=c.TB_TABNET_GAMMA,
                                                             n_ind=c.TB_TABNET_N_IND,
                                                             n_shared=c.TB_TABNET_N_SHARED,
                                                             learning_rate=c.TB_TABNET_LEARNING_RATE,
                                                             weight_decay=c.TB_TABNET_WEIGHT_DECAY,
                                                             mask_type=c.TB_TABNET_MASK_TYPE,
                                                             epochs=c.TB_TABNET_EPOCHS,
                                                             patience=c.TB_TABNET_PATIENCE,
                                                             batch_size=c.TB_TABNET_BATCH_SIZE)

            # Save the Model
            mlflow.sklearn.log_model(tb_model, f'tabnet_model_{fold}')

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
        mean_cm = (sum(cm_folds) / len(cm_folds)).astype(int)

        # 4. Calculating the mean of all Evaluation Matrices:
        std_threshold = np.std(threshold_folds)
        std_accuracy = np.std(accuracy_folds)
        std_precision = np.std(precision_folds)
        std_recall = np.std(recall_folds)
        std_f1_score = np.std(f1_score_folds)
        std_sensitivity = np.std(sensitivity_folds)
        std_specificity = np.std(specificity_folds)
        std_tn = np.std(tn_folds)
        std_fp = np.std(fp_folds)
        std_fn = np.std(fn_folds)
        std_tp = np.std(tp_folds)
        std_roc_auc = np.std(roc_auc_folds)
        std_auc_score = np.std(auc_score_folds)

        # 5. logging all Artifacts, Parameters and Matrices into mlflow:
        # Log the Hyper-Parameters:
        mlflow.log_param('n_d', c.TB_TABNET_N_D)
        mlflow.log_param('n_a', c.TB_TABNET_N_A)
        mlflow.log_param('n_steps', c.TB_TABNET_N_STEPS)
        mlflow.log_param('gamma', c.TB_TABNET_GAMMA)
        mlflow.log_param('n_ind', c.TB_TABNET_N_IND)
        mlflow.log_param('n_shared', c.TB_TABNET_N_SHARED)
        mlflow.log_param('learning_rate', c.TB_TABNET_LEARNING_RATE)
        mlflow.log_param('weight_decay', c.TB_TABNET_WEIGHT_DECAY)
        mlflow.log_param('mask_type', c.TB_TABNET_MASK_TYPE)
        mlflow.log_param('epochs', c.TB_TABNET_EPOCHS)
        mlflow.log_param('patience', c.TB_TABNET_PATIENCE)
        mlflow.log_param('batch_size', c.TB_TABNET_BATCH_SIZE)

        # Log the Matrices (Evaluation):
        # Mean of Matrices:
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

        # Standard Deviation of Matrices:
        mlflow.log_metric('std_threshold', float(std_threshold))
        mlflow.log_metric('std_accuracy', float(std_accuracy))
        mlflow.log_metric('std_precision', float(std_precision))
        mlflow.log_metric('std_recall', float(std_recall))
        mlflow.log_metric('std_f1_score', float(std_f1_score))
        mlflow.log_metric('std_sensitivity', float(std_sensitivity))
        mlflow.log_metric('std_specificity', float(std_specificity))
        mlflow.log_metric('std_tn', float(std_tn))
        mlflow.log_metric('std_fp', float(std_fp))
        mlflow.log_metric('std_fn', float(std_fn))
        mlflow.log_metric('std_tp', float(std_tp))
        mlflow.log_metric('std_roc_auc', float(std_roc_auc))
        mlflow.log_metric('std_auc_score', float(std_auc_score))

        # Saving fpr_folds and tpr_folds as pickle file and log it into mlflow as artifact:
        with open('tabular_tabnet_fpr_folds.pkl', 'wb') as f:
            pickle.dump(fpr_folds, f)
        mlflow.log_artifact('tabular_tabnet_fpr_folds.pkl', artifact_path='fpr_tpr')
        os.remove('tabular_tabnet_fpr_folds.pkl')

        with open('tabular_tabnet_tpr_folds.pkl', 'wb') as f:
            pickle.dump(tpr_folds, f)
        mlflow.log_artifact('tabular_tabnet_tpr_folds.pkl', artifact_path='fpr_tpr')
        os.remove('tabular_tabnet_tpr_folds.pkl')

        # Logging the Average Confusion Matrix into mlflow:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(mean_cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")  # Log the plot in mlflow

        # End the mlflow run:
        mlflow.end_run()
    return None


if __name__ == "__main__":
    # Checking directory:
    _check_create_dir()

    # Loading Cross Validation Folds:
    X_TRAIN_FOLDS, X_TEST_FOLDS, Y_TRAIN_FOLDS, Y_TEST_FOLDS = _load_cv_folds()

    if c.TB_TABNET_AUTO_HP:
        # Tune the Hyper-Parameters:
        BEST_HP_FOLDS = _tune_hyper_parameters(x_train_folds=X_TRAIN_FOLDS,
                                               x_test_folds=X_TEST_FOLDS,
                                               y_train_folds=Y_TRAIN_FOLDS,
                                               y_test_folds=Y_TEST_FOLDS)

        # Building, Fitting and Evaluating TabNet Model based on best Automatic Hyper-Parameters:
        _run_evaluate_tabnet_automatic_hp(x_train_folds=X_TRAIN_FOLDS,
                                          x_test_folds=X_TEST_FOLDS,
                                          y_train_folds=Y_TRAIN_FOLDS,
                                          y_test_folds=Y_TEST_FOLDS,
                                          best_hp_folds=BEST_HP_FOLDS)

    else:
        # Building, Fitting and Evaluating TabNet Model based on best Manual Hyper-Parameters:
        _run_evaluate_tabnet_manual_hp(x_train_folds=X_TRAIN_FOLDS,
                                       x_test_folds=X_TEST_FOLDS,
                                       y_train_folds=Y_TRAIN_FOLDS,
                                       y_test_folds=Y_TEST_FOLDS)

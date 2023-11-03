"""
Author: Mohammed Lajam
/data/processed_data/tabular/selected_features
Phase 5: Classification (Tabular Data - 1D-CNN):
- In this python file, the tabular data is imported from the '/data/processed_data/tabular/selected_features/'
directory.

Objective:
- Building, predicting, evaluating and tuning Support Vector Machine Classifier Model and save the
best model with respect to Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity,
ROC_AUC Score and other evaluation matrices.

Input:
- The input data is the 6 Folds of X_TRAIN, X_TEST, Y_TRAIN, Y_TEST.

Output:
- The outputs are Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity, ROC_AUC Score
and other evaluation matrices, which are saved in 'mlruns' in the same directory. Additionally,
all the Artifacts, parameters and matrices are tracked using MLFlow.

Note:
- In 'constants.py', 'CROSS_VALIDATION' path has to be adjusted before running this file.
- All the functions and variables, which are used in this file, are imported from helpers.py
file from the 'models' package in the same repository.
"""
# Loading Libraries:
import os
import pickle
import time

import seaborn as sns

import constants as c
from models.helpers import *
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import logging


# Functions to be used in this python file:
def _check_create_dir():
    """Function to check if 'cross_validation' directory exists in 'data' directory and create
    it if it is not existed."""
    try:
        os.path.isdir(f'{c.PROCESSED_DATA_PATH}/tabular')
    except FileNotFoundError:
        print("Directory does not exist!")
    except Exception as e:
        print(f"Error: {e}")
    else:
        return None


def _load_cv_folds():
    """Function to load Cross Validation Folds from local machine."""
    try:
        with open(f'{c.PROCESSED_DATA_PATH}/tabular/selected_features/x_train_folds.pkl', 'rb') as f:
            x_train_folds = pickle.load(f)
        with open(f'{c.PROCESSED_DATA_PATH}/tabular/selected_features/x_test_folds.pkl', 'rb') as f:
            x_test_folds = pickle.load(f)
        with open(f'{c.PROCESSED_DATA_PATH}/tabular/selected_features/y_train_folds.pkl', 'rb') as f:
            y_train_folds = pickle.load(f)
        with open(f'{c.PROCESSED_DATA_PATH}/tabular/selected_features/y_test_folds.pkl', 'rb') as f:
            y_test_folds = pickle.load(f)
    except FileNotFoundError:
        print("Error: One or more files not found")
    except Exception as e:
        print(f"Error: {e}")
    else:
        return x_train_folds, x_test_folds, y_train_folds, y_test_folds


def _data_per_fold(x_train, x_test, y_train, y_test):
    """Function to prepare the data for TabNet Model."""
    # Splitting the data into train and validation sets:
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=c.TEST_SIZE,
                                                      random_state=c.RANDOM_STATE,
                                                      shuffle=True)

    # Converting all data into numpy arrays and int32:
    x_train = x_train.values.astype(np.float32)
    x_test = x_test.values.astype(np.float32)
    x_val = x_val.values.astype(np.float32)

    # Converting all -1 to 0 in Y_TRAIN and Y_TEST:
    y_train = np.where(y_train == -1, 0, y_train)
    y_test = np.where(y_test == -1, 0, y_test)
    y_val = np.where(y_val == -1, 0, y_val)
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

        # Reshaping the x_train and x_val:
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

        input_shape = x_train.shape[1:]

        best_hp, best_validation = HyperParametersTuner.find_best_tabular_cnn_hp(x_train=x_train,
                                                                                 x_val=x_val,
                                                                                 y_train=y_train,
                                                                                 y_val=y_val,
                                                                                 input_shape=input_shape,
                                                                                 max_trials=c.TB_CNN_HP_MAX_TRIALS,
                                                                                 epochs=c.TB_CNN_HP_EPOCHS,
                                                                                 directory=f'/{c.REPO_PATH}/models/hp/tabular_cnn/fold_{fold}/')
        best_hp_folds.append(best_hp)
        best_validation_folds.append(best_validation)

    # Creating one DataFrame including Hyper-Parameters and Validation Scores:
    best_hp_folds = pd.DataFrame(best_hp_folds)
    best_hp_folds['val_accuracy'] = best_validation_folds
    # Arranging the rows descendingly based on the val_accuracy:
    best_hp_folds = best_hp_folds.sort_values('val_accuracy', ascending=False)
    return best_hp_folds


def _run_evaluate_1d_cnn_automatic_hp(x_train_folds, x_test_folds, y_train_folds, y_test_folds, best_hp_folds):
    """Function to run and evaluate 1D-CNN based on Automatic adjustment of Parameters from
    best_hp_folds. It returns all the Matrices, paramters and Artifacts into mlflow."""
    with mlflow.start_run():
        # Removing the warning messages while Executing the code and keep only the Errors:
        logging.getLogger('mlflow').setLevel(logging.ERROR)

        # Creating Empty Lists for all Evaluation Matrices:
        threshold_folds, accuracy_folds, precision_folds, recall_folds, f1_score_folds = [], [], [], [], []
        sensitivity_folds, specificity_folds = [], []
        tn_folds, fp_folds, fn_folds, tp_folds = [], [], [], []
        roc_auc_folds, auc_score_folds = [], []
        fpr_folds, tpr_folds = [], []
        cm_folds = []
        test_duration_folds = []

        for fold in range(len(x_train_folds)):
            x_train, x_test, x_val, y_train, y_test, y_val = _data_per_fold(x_train=x_train_folds[fold],
                                                                            x_test=x_test_folds[fold],
                                                                            y_train=y_train_folds[fold],
                                                                            y_test=y_test_folds[fold])

            # Reshaping the x_train and x_val:
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

            input_shape = x_train.shape[1:]

            # 1. Building and fitting 1D-CNN Model:
            cnn_model, history = ModelBuilder.build_fit_tabular_cnn(x_train=x_train,
                                                                    x_val=x_val,
                                                                    y_train=y_train,
                                                                    y_val=y_val,
                                                                    input_shape=input_shape,
                                                                    filter_1=best_hp_folds.iloc[0]['filter_1'],
                                                                    filter_2=best_hp_folds.iloc[0]['filter_2'],
                                                                    dense_1=best_hp_folds.iloc[0]['dense_1'],
                                                                    dense_2=best_hp_folds.iloc[0]['dense_2'],
                                                                    filter_1_l2=best_hp_folds.iloc[0]['filter_1_l2'],
                                                                    filter_2_l2=best_hp_folds.iloc[0]['filter_2_l2'],
                                                                    dense_1_l2=best_hp_folds.iloc[0]['dense_1_l2'],
                                                                    dense_2_l2=best_hp_folds.iloc[0]['dense_2_l2'],
                                                                    dropout_rate=best_hp_folds.iloc[0]['dropout_rate'],
                                                                    learning_rate=best_hp_folds.iloc[0]['learning_rate'],
                                                                    loss=c.TB_CNN_LOSS,
                                                                    patience=c.TB_CNN_PATIENCE,
                                                                    epochs=c.TB_CNN_EPOCHS,
                                                                    batch_size=c.TB_CNN_BATCH_SIZE)

            # Saving the Model
            model_name = f"tabular_cnn_model_{fold}"
            mlflow.sklearn.log_model(cnn_model, model_name)

            # Plotting and saving Accuracy vs Epoch:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(history.history['accuracy'], label=f'Fold_{fold + 1} Train Accuracy')
            ax.plot(history.history['val_accuracy'], label=f'Fold_{fold + 1} Validation Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy over Epochs')
            ax.legend()
            mlflow.log_figure(fig, f'Accuracy_vs_Epochs (Fold_{fold + 1}).png')  # Log the plot in mlflow

            # 2. Predictions:
            # Calculating the Probabilities:
            start_time = time.time()  # Measuring the start_time of the predictions
            y_prob = ModelEvaluator.calculate_probabilities(model=cnn_model, x_test=x_test)

            # Finding the best threshold that provides the best accuracy:
            _, best_threshold = ModelEvaluator.find_best_threshold(y_test=y_test,
                                                                   y_prob=y_prob,
                                                                   evaluation_matrix='f1_score')

            # Calculating Evaluation Matrix:
            evaluation_matrix = ModelEvaluator.evaluate_model(y_test=y_test,
                                                              y_prob=y_prob,
                                                              threshold=best_threshold)

            # Generating Confusion Matrix:
            cm = ModelEvaluator.generate_confusion_matrix(y_test=y_test,
                                                          y_prob=y_prob,
                                                          threshold=best_threshold)

            end_time = time.time()  # Measuring the end_time of the predictions

            # Calculate the duration of prediction's process:
            test_duration = end_time - start_time

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
            test_duration_folds.append(test_duration)

        # 3. Calculating the mean of all Evaluation Matrices:
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
        mean_test_duration = sum(test_duration_folds) / len(test_duration_folds)

        # 4. Calculating the Standard Deviation of all Evaluation Matrices:
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
        std_test_duration = np.std(test_duration_folds)

        # 5. logging all Artifacts, Parameters and Matrices into mlflow:
        # Log the Hyper-Parameters:
        mlflow.log_param('filter_1', best_hp_folds.iloc[0]['filter_1'])
        mlflow.log_param('filter_2', best_hp_folds.iloc[0]['filter_2'])
        mlflow.log_param('dense_1', best_hp_folds.iloc[0]['dense_1'])
        mlflow.log_param('dense_2', best_hp_folds.iloc[0]['dense_2'])
        mlflow.log_param('filter_1_l2', best_hp_folds.iloc[0]['filter_1_l2'])
        mlflow.log_param('filter_2_l2', best_hp_folds.iloc[0]['filter_2_l2'])
        mlflow.log_param('dense_1_l2', best_hp_folds.iloc[0]['dense_1_l2'])
        mlflow.log_param('dense_2_l2', best_hp_folds.iloc[0]['dense_2_l2'])
        mlflow.log_param('dropout_rate', best_hp_folds.iloc[0]['dropout_rate'])
        mlflow.log_param('learning_rate', best_hp_folds.iloc[0]['learning_rate'])
        mlflow.log_param('patience', c.TB_CNN_PATIENCE)
        mlflow.log_param('epochs', c.TB_CNN_EPOCHS)
        mlflow.log_param('batch_size', c.TB_CNN_BATCH_SIZE)

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
        mlflow.log_metric('mean_test_duration', mean_test_duration)

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
        mlflow.log_metric('std_test_duration', float(std_test_duration))

        # Saving each metric of Evaluation Matrices as list:
        evaluation_matrices_folds = {"threshold_folds": threshold_folds,
                                     "accuracy_folds": accuracy_folds,
                                     "precision_folds": precision_folds,
                                     "recall_folds": recall_folds,
                                     "f1_score_folds": f1_score_folds,
                                     "sensitivity_folds": sensitivity_folds,
                                     "specificity_folds": specificity_folds,
                                     "tn_folds": tn_folds,
                                     "fp_folds": fp_folds,
                                     "fn_folds": fn_folds,
                                     "tp_folds": tp_folds,
                                     "roc_auc_folds": roc_auc_folds,
                                     "auc_score_folds": auc_score_folds,
                                     "fpr_folds": fpr_folds,
                                     "tpr_folds": tpr_folds,
                                     "cm_folds": cm_folds,
                                     "test_duration_folds": test_duration_folds}

        for key, value in evaluation_matrices_folds.items():
            with open(f'tabular_cnn_{key}.pkl', 'wb') as f:
                pickle.dump(value, f)
            mlflow.log_artifact(f'tabular_cnn_{key}.pkl', artifact_path='evaluation_matrices_folds')
            os.remove(f'tabular_cnn_{key}.pkl')

        # Saving the Model's Summary:
        artifact_path = "tabular_cnn_summary.txt"
        with open(artifact_path, "w") as f:
            cnn_model.summary(print_fn=lambda x: f.write(x + "\n"))
        mlflow.log_artifact(artifact_path, "tabular_cnn_summary.txt")
        os.remove(artifact_path)

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


def _run_evaluate_1d_cnn_manual_hp(x_train_folds, x_test_folds, y_train_folds, y_test_folds):
    """Function to run and evaluate 1D-CNN Model based on Manual adjustment of Parameters.
    It returns all the Matrices, paramters and Artifacts into mlflow."""
    with mlflow.start_run():
        # Removing the warning messages while Executing the code and keep only the Errors:
        logging.getLogger('mlflow').setLevel(logging.ERROR)

        # Creating Empty Lists for all Evaluation Matrices:
        threshold_folds, accuracy_folds, precision_folds, recall_folds, f1_score_folds = [], [], [], [], []
        sensitivity_folds, specificity_folds = [], []
        tn_folds, fp_folds, fn_folds, tp_folds = [], [], [], []
        roc_auc_folds, auc_score_folds = [], []
        fpr_folds, tpr_folds = [], []
        cm_folds = []
        test_duration_folds = []

        for fold in range(len(x_train_folds)):
            x_train, x_test, x_val, y_train, y_test, y_val = _data_per_fold(x_train=x_train_folds[fold],
                                                                            x_test=x_test_folds[fold],
                                                                            y_train=y_train_folds[fold],
                                                                            y_test=y_test_folds[fold])

            # Reshaping the x_train and x_val:
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

            input_shape = x_train.shape[1:]

            # 1. Building and fitting 1D-CNN Model:
            cnn_model, history = ModelBuilder.build_fit_tabular_cnn(x_train=x_train,
                                                                    x_val=x_val,
                                                                    y_train=y_train,
                                                                    y_val=y_val,
                                                                    input_shape=input_shape,
                                                                    filter_1=c.TB_CNN_FILTER_1,
                                                                    filter_2=c.TB_CNN_FILTER_2,
                                                                    dense_1=c.TB_CNN_DENSE_1,
                                                                    dense_2=c.TB_CNN_DENSE_2,
                                                                    filter_1_l2=c.TB_CNN_FILTER_1_L2,
                                                                    filter_2_l2=c.TB_CNN_FILTER_2_L2,
                                                                    dense_1_l2=c.TB_CNN_DENSE_1_L2,
                                                                    dense_2_l2=c.TB_CNN_DENSE_2_L2,
                                                                    dropout_rate=c.TB_CNN_DROPOUT_RATE,
                                                                    learning_rate=c.TB_CNN_LEARNING_RATE,
                                                                    loss=c.TB_CNN_LOSS,
                                                                    patience=c.TB_CNN_PATIENCE,
                                                                    epochs=c.TB_CNN_EPOCHS,
                                                                    batch_size=c.TB_CNN_BATCH_SIZE)

            # Saving the Model
            model_name = f"tabular_cnn_model_{fold}"
            mlflow.sklearn.log_model(cnn_model, model_name)
            # Saving the model in data directory:
            cnn_model.save(f'/{c.REPO_PATH}/data/models/tabular_cnn/tabular_cnn_model_{fold}/model.h5')

            # Plotting and saving Accuracy vs Epoch:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(history.history['accuracy'], label=f'Fold_{fold + 1} Train Accuracy')
            ax.plot(history.history['val_accuracy'], label=f'Fold_{fold + 1} Validation Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy over Epochs')
            ax.legend()
            mlflow.log_figure(fig, f'Accuracy_vs_Epochs (Fold_{fold + 1}).png')  # Log the plot in mlflow
            plt.savefig(f'/{c.REPO_PATH}/data/models/tabular_cnn/Accuracy_vs_Epochs_{fold + 1}')  # Save the plot

            # 2. Predictions:
            # Calculating the Probabilities:
            start_time = time.time()  # Measuring the start_time of the predictions
            y_prob = ModelEvaluator.calculate_probabilities(model=cnn_model, x_test=x_test)

            # Finding the best threshold that provides the best accuracy:
            _, best_threshold = ModelEvaluator.find_best_threshold(y_test=y_test,
                                                                   y_prob=y_prob,
                                                                   evaluation_matrix='f1_score')

            # Calculating Evaluation Matrix:
            evaluation_matrix = ModelEvaluator.evaluate_model(y_test=y_test,
                                                              y_prob=y_prob,
                                                              threshold=best_threshold)

            # Generating Confusion Matrix:
            cm = ModelEvaluator.generate_confusion_matrix(y_test=y_test,
                                                          y_prob=y_prob,
                                                          threshold=best_threshold)

            end_time = time.time()  # Measuring the end_time of the predictions

            # Calculate the duration of prediction's process:
            test_duration = end_time - start_time

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
            test_duration_folds.append(test_duration)

        # 3. Calculating the mean of all Evaluation Matrices:
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
        mean_test_duration = sum(test_duration_folds) / len(test_duration_folds)

        # 4. Calculating the Standard Deviation of all Evaluation Matrices:
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
        std_test_duration = np.std(test_duration_folds)

        # 5. logging all Artifacts, Parameters and Matrices into mlflow:
        # Log the Hyper-Parameters:
        mlflow.log_param('filter_1', c.TB_CNN_FILTER_1)
        mlflow.log_param('filter_2', c.TB_CNN_FILTER_2)
        mlflow.log_param('dense_1', c.TB_CNN_DENSE_1)
        mlflow.log_param('dense_2', c.TB_CNN_DENSE_2)
        mlflow.log_param('filter_1_l2', c.TB_CNN_FILTER_1_L2)
        mlflow.log_param('filter_2_l2', c.TB_CNN_FILTER_2_L2)
        mlflow.log_param('dense_1_l2', c.TB_CNN_DENSE_1_L2)
        mlflow.log_param('dense_2_l2', c.TB_CNN_DENSE_2_L2)
        mlflow.log_param('dropout_rate', c.TB_CNN_DROPOUT_RATE)
        mlflow.log_param('learning_rate', c.TB_CNN_LEARNING_RATE)
        mlflow.log_param('patience', c.TB_CNN_PATIENCE)
        mlflow.log_param('epochs', c.TB_CNN_EPOCHS)
        mlflow.log_param('batch_size', c.TB_CNN_BATCH_SIZE)

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
        mlflow.log_metric('mean_test_duration', mean_test_duration)

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
        mlflow.log_metric('std_test_duration', float(std_test_duration))

        # Saving each metric of Evaluation Matrices as list:
        evaluation_matrices_folds = {"threshold_folds": threshold_folds,
                                     "accuracy_folds": accuracy_folds,
                                     "precision_folds": precision_folds,
                                     "recall_folds": recall_folds,
                                     "f1_score_folds": f1_score_folds,
                                     "sensitivity_folds": sensitivity_folds,
                                     "specificity_folds": specificity_folds,
                                     "tn_folds": tn_folds,
                                     "fp_folds": fp_folds,
                                     "fn_folds": fn_folds,
                                     "tp_folds": tp_folds,
                                     "roc_auc_folds": roc_auc_folds,
                                     "auc_score_folds": auc_score_folds,
                                     "fpr_folds": fpr_folds,
                                     "tpr_folds": tpr_folds,
                                     "cm_folds": cm_folds,
                                     "test_duration_folds": test_duration_folds}

        for key, value in evaluation_matrices_folds.items():
            with open(f'tabular_cnn_{key}.pkl', 'wb') as f:
                pickle.dump(value, f)
            mlflow.log_artifact(f'tabular_cnn_{key}.pkl', artifact_path='evaluation_matrices_folds')
            os.remove(f'tabular_cnn_{key}.pkl')

        # Saving the Model's Summary:
        artifact_path = "tabular_cnn_summary.txt"
        with open(artifact_path, "w") as f:
            cnn_model.summary(print_fn=lambda x: f.write(x + "\n"))
        mlflow.log_artifact(artifact_path, "tabular_cnn_summary.txt")
        os.remove(artifact_path)

        # Logging the Average Confusion Matrix into mlflow:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(mean_cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")  # Log the plot in mlflow
        plt.savefig(f'/{c.REPO_PATH}/data/models/tabular_cnn/confusion_matrix.png')  # Save the plot

        # End the mlflow run:
        mlflow.end_run()
    return None


if __name__ == "__main__":
    # Checking directory:
    _check_create_dir()

    # Loading Cross Validation Folds:
    X_TRAIN_FOLDS, X_TEST_FOLDS, Y_TRAIN_FOLDS, Y_TEST_FOLDS = _load_cv_folds()

    if c.TB_CNN_AUTO_HP:
        # Tune the Hyper-Parameters:
        BEST_HP_FOLDS = _tune_hyper_parameters(x_train_folds=X_TRAIN_FOLDS,
                                               x_test_folds=X_TEST_FOLDS,
                                               y_train_folds=Y_TRAIN_FOLDS,
                                               y_test_folds=Y_TEST_FOLDS)

        # Building, Fitting and Evaluating 1D-CNN Model based on best Automatic Hyper-Parameters:
        _run_evaluate_1d_cnn_automatic_hp(x_train_folds=X_TRAIN_FOLDS,
                                          x_test_folds=X_TEST_FOLDS,
                                          y_train_folds=Y_TRAIN_FOLDS,
                                          y_test_folds=Y_TEST_FOLDS,
                                          best_hp_folds=BEST_HP_FOLDS)
    else:
        # Building, Fitting and Evaluating 1D-CNN Model based on best Manual Hyper-Parameters:
        _run_evaluate_1d_cnn_manual_hp(x_train_folds=X_TRAIN_FOLDS,
                                       x_test_folds=X_TEST_FOLDS,
                                       y_train_folds=Y_TRAIN_FOLDS,
                                       y_test_folds=Y_TEST_FOLDS)

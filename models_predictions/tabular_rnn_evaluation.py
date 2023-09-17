import os
import time
import seaborn as sns
import constants as c
from models_evaluation.helpers import *
import pickle
from keras.models import load_model
import mlflow.sklearn
import sys
sys.path.insert(0, '/Users/mohammedlajam/Documents/GitHub/pcg-classification')


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
    """Function to prepare the data for Support Vector Machine Model."""
    # Converting all data into numpy arrays and int32:
    x_train = x_train.values.astype(np.float32)
    x_test = x_test.values.astype(np.float32)

    # Converting all -1 to 0 in Y_TRAIN and Y_TEST:
    y_train = np.where(y_train == -1, 0, y_train)
    y_test = np.where(y_test == -1, 0, y_test)
    return x_train, x_test, y_train, y_test


def _predict_evaluate_model(x_test_folds, y_test_folds):
    with mlflow.start_run():
        # 1. Loading the models from the pickle file (6 Folds):
        lstm_models = []
        for fold in range(6):
            loaded_model = load_model(f'/{c.REPO_PATH}/data/models/tabular_rnn/tabular_lstm_model_{fold}/model.h5')
            lstm_models.append(loaded_model)

        # 2. Predictions:
        evaluation_matrices_folds = []
        cm_folds = []
        test_duration_folds = []
        for fold in range(len(lstm_models)):
            x_test, y_test = x_test_folds[fold], y_test_folds[fold]

            # Changing -1 to 0:
            y_test = np.where(y_test == -1, 0, y_test)

            # Calculating the Probabilities and best Threshold (based on F1-Score):
            start_time = time.time()  # Measuring the start_time of the predictions
            y_prob = ModelEvaluator.calculate_probabilities(model=lstm_models[fold], x_test=x_test)
            _, best_threshold = ModelEvaluator.find_best_threshold(y_test=y_test,
                                                                   y_prob=y_prob,
                                                                   evaluation_matrix='f1_score')

            evaluation_matrices = ModelEvaluator.evaluate_model(y_test=y_test,
                                                                y_prob=y_prob,
                                                                threshold=best_threshold)
            evaluation_matrices_folds.append(evaluation_matrices)

            # Generating Confusion Matrix:
            cm = ModelEvaluator.generate_confusion_matrix(y_test=y_test,
                                                          y_prob=y_prob,
                                                          threshold=best_threshold)
            cm_folds.append(cm)

            end_time = time.time()  # Measuring the end_time of the predictions
            # Calculate the duration of prediction's process:
            test_duration = end_time - start_time
            test_duration_folds.append(test_duration)

        threshold_folds, accuracy_folds, precision_folds, recall_folds, f1_score_folds = [], [], [], [], []
        sensitivity_folds, specificity_folds = [], []
        tn_folds, fp_folds, fn_folds, tp_folds = [], [], [], []
        roc_auc_folds, auc_score_folds = [], []
        fpr_folds, tpr_folds = [], []

        for index in range(len(evaluation_matrices_folds)):
            # Extracting evaluations matrices based on threshold, which provides the best accuracy:
            threshold_folds.append(evaluation_matrices_folds[index].iloc[0]['Threshold'])
            accuracy_folds.append(evaluation_matrices_folds[index].iloc[0]['Accuracy'])
            precision_folds.append(evaluation_matrices_folds[index].iloc[0]['Precision'])
            recall_folds.append(evaluation_matrices_folds[index].iloc[0]['Recall'])
            f1_score_folds.append(evaluation_matrices_folds[index].iloc[0]['F1-score'])
            sensitivity_folds.append(evaluation_matrices_folds[index].iloc[0]['Sensitivity'])
            specificity_folds.append(evaluation_matrices_folds[index].iloc[0]['Specificity'])
            tn_folds.append(evaluation_matrices_folds[index].iloc[0]['TN'])
            fp_folds.append(evaluation_matrices_folds[index].iloc[0]['FP'])
            fn_folds.append(evaluation_matrices_folds[index].iloc[0]['FN'])
            tp_folds.append(evaluation_matrices_folds[index].iloc[0]['TP'])
            roc_auc_folds.append(evaluation_matrices_folds[index].iloc[0]['ROC_AUC'])
            auc_score_folds.append(evaluation_matrices_folds[index].iloc[0]['AUC_Score'])
            fpr_folds.append(evaluation_matrices_folds[index].iloc[0]['FPR'])
            tpr_folds.append(evaluation_matrices_folds[index].iloc[0]['TPR'])

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
        mean_test_duration = sum(test_duration_folds) / len(test_duration_folds)

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
        std_test_duration = np.std(test_duration_folds)

        # 5. logging all Artifacts, Parameters and Matrices into mlflow:
        # Log the Hyper-Parameters:
        mlflow.log_param('model', 'tabular_rnn')
        mlflow.log_param('lstm_1', c.TB_LSTM_LSTM_1)
        mlflow.log_param('lstm_2', c.TB_LSTM_LSTM_2)
        mlflow.log_param('lstm_1_l2', c.TB_LSTM_1_L2)
        mlflow.log_param('lstm_2_l2', c.TB_LSTM_2_L2)
        mlflow.log_param('dropout_rate_1', c.TB_LSTM_DROPOUT_RATE_1)
        mlflow.log_param('dropout_rate_2', c.TB_LSTM_DROPOUT_RATE_2)
        mlflow.log_param('learning_rate', c.TB_LSTM_LEARNING_RATE)
        mlflow.log_param('patience', c.TB_LSTM_PATIENCE)
        mlflow.log_param('epochs', c.TB_LSTM_EPOCHS)
        mlflow.log_param('batch_size', c.TB_LSTM_BATCH_SIZE)

        # Log the Matrices (Evaluation):
        # Mean of Matrices:
        mlflow.log_metric('mean_threshold', mean_threshold)
        mlflow.log_metric('mean_accuracy', mean_accuracy)
        mlflow.log_metric('mean_precision', mean_precision)
        mlflow.log_metric('mean_recall', mean_recall)
        mlflow.log_metric('mean_f1_Score', mean_f1_score)
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
            with open(f'tabular_lstm_{key}.pkl', 'wb') as f:
                pickle.dump(value, f)
            mlflow.log_artifact(f'tabular_lstm_{key}.pkl', artifact_path='evaluation_matrices_folds')
            os.remove(f'tabular_lstm_{key}.pkl')

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
    #  Loading Datasets:
    _, X_TEST_FOLDS, _, Y_TEST_FOLDS = _load_cv_folds()

    # Loading the Models, making predictions and evaluate the Models:
    _predict_evaluate_model(x_test_folds=X_TEST_FOLDS, y_test_folds=Y_TEST_FOLDS)
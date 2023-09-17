"""
Author: Mohammed Lajam

Phase 5: Classification (Computer Vision - Pre-trained Model: "ResNet50"):
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
- The input data is the 6 Folds of X_TRAIN_FOLDS, X_TEST_FOLDS, Y_TRAIN_FOLDS, Y_TEST_FOLDS.

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
        os.path.isdir(f'{c.PROCESSED_DATA_PATH}/images')
    except FileNotFoundError:
        print("Directory does not exist!")
    except Exception as e:
        print(f"Error: {e}")
    else:
        return None


def _load_cv_folds(rep_type: str):
    """Function to load Cross Validation Folds from local machine."""
    try:
        with open(f'{c.PROCESSED_DATA_PATH}/images/{rep_type}/kfold_cv/rgb/x_train_folds.pkl', 'rb') as f:
            x_train_folds = pickle.load(f)
        with open(f'{c.PROCESSED_DATA_PATH}/images/{rep_type}/kfold_cv/rgb/x_test_folds.pkl', 'rb') as f:
            x_test_folds = pickle.load(f)
        with open(f'{c.PROCESSED_DATA_PATH}/images/{rep_type}/kfold_cv/rgb/y_train_folds.pkl', 'rb') as f:
            y_train_folds = pickle.load(f)
        with open(f'{c.PROCESSED_DATA_PATH}/images/{rep_type}/kfold_cv/rgb/y_test_folds.pkl', 'rb') as f:
            y_test_folds = pickle.load(f)
    except FileNotFoundError:
        print("Error: One or more files not found")
    except Exception as e:
        print(f"Error: {e}")
    else:
        return x_train_folds, x_test_folds, y_train_folds, y_test_folds


def _generate_val_set(x_train, x_test, y_train, y_test):
    """Function to prepare the data for ResNet50."""
    # Splitting the data into train and validation sets:
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=c.TEST_SIZE,
                                                      random_state=c.RANDOM_STATE,
                                                      shuffle=True)

    # Converting all y-sets into numpy array:
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)

    return x_train, x_test, x_val, y_train, y_test, y_val


def _run_evaluate_resnet50(x_train_folds, x_test_folds, y_train_folds, y_test_folds):
    """Function to run and evaluate ResNet50 Model based on Manual adjustment of Parameters.
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
            # Splitting x_train into x_train and x_val:
            x_train, x_test, x_val, y_train, y_test, y_val = _generate_val_set(x_train=x_train_folds[fold],
                                                                               x_test=x_test_folds[fold],
                                                                               y_train=y_train_folds[fold],
                                                                               y_test=y_test_folds[fold])

            # Convert the data to TensorFlow Dataset format:
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(c.CV_RN50_BATCH_SIZE)
            val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(c.CV_RN50_BATCH_SIZE)
            test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(c.CV_RN50_BATCH_SIZE)

            input_shape = x_train[0].shape

            # 1. Building and fitting ResNet50 Model:
            resnet50_model, history = PretrainedModel.build_fit_resnet50(train_dataset=train_dataset,
                                                                         val_dataset=val_dataset,
                                                                         input_shape=input_shape,
                                                                         include_top=c.CV_RN50_INCLUDE_TOP,
                                                                         resnet_weights=c.CV_RN50_WEIGHTS,
                                                                         trainable=c.CV_RN50_TRAINABLE,
                                                                         dense_1=c.CV_RN50_DENSE_1,
                                                                         dense_2=c.CV_RN50_DENSE_2,
                                                                         dense_1_l2=c.CV_RN50_DENSE_1_L2,
                                                                         dense_2_l2=c.CV_RN50_DENSE_2_L2,
                                                                         dropout_rate_1=c.CV_RN50_DROPOUT_RATE_1,
                                                                         dropout_rate_2=c.CV_RN50_DROPOUT_RATE_2,
                                                                         learning_rate=c.CV_RN50_LEARNING_RATE,
                                                                         loss=c.CV_RN50_LOSS,
                                                                         patience=c.CV_RN50_PATIENCE,
                                                                         epochs=c.CV_RN50_EPOCHS)

            # Saving the Model
            model_name = f"cv_resnet50_model_{fold}"
            mlflow.sklearn.log_model(resnet50_model, model_name)
            # Saving the model in data directory:
            resnet50_model.save(f'/{c.REPO_PATH}/data/models/cv_resnet50/cv_resnet_model50_{fold}/model.h5')

            # Plotting and saving Accuracy vs Epoch:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(history.history['accuracy'], label=f'Fold_{fold + 1} Train Accuracy')
            ax.plot(history.history['val_accuracy'], label=f'Fold_{fold + 1} Validation Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy over Epochs')
            ax.legend()
            mlflow.log_figure(fig, f'Accuracy_vs_Epochs (Fold_{fold + 1}).png')  # Log the plot in mlflow
            plt.savefig(f'/{c.REPO_PATH}/data/models/cv_resnet50/Accuracy_vs_Epochs_{fold + 1}')  # Save the plot

            # 2. Predictions:
            # Calculating the Probabilities:
            start_time = time.time()  # Measuring the start_time of the predictions
            y_prob = ModelEvaluator.calculate_probabilities(model=resnet50_model, x_test=test_dataset)

            # Finding the best threshold that provides the best accuracy:
            _, best_threshold = ModelEvaluator.find_best_threshold(y_test=y_test,
                                                                   y_prob=y_prob,
                                                                   evaluation_matrix='f1_score')

            # Calculating Evaluation Matrix:
            evaluation_matrix = ModelEvaluator.evaluate_model(y_test=y_test,
                                                              y_prob=y_prob,
                                                              threshold=best_threshold)

            # Generating Confusion Matrix
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
        mlflow.log_param('include_top', c.CV_RN50_INCLUDE_TOP)
        mlflow.log_param('weights', c.CV_RN50_WEIGHTS)
        mlflow.log_param('trainable', c.CV_RN50_TRAINABLE)
        mlflow.log_param('dense_1', c.CV_RN50_DENSE_1)
        mlflow.log_param('dense_2', c.CV_RN50_DENSE_2)
        mlflow.log_param('dense_1_l2', c.CV_RN50_DENSE_1_L2)
        mlflow.log_param('dense_2_l2', c.CV_RN50_DENSE_2_L2)
        mlflow.log_param('dropout_rate_1', c.CV_RN50_DROPOUT_RATE_1)
        mlflow.log_param('dropout_rate_2', c.CV_RN50_DROPOUT_RATE_2)
        mlflow.log_param('learning_rate', c.CV_RN50_LEARNING_RATE)
        mlflow.log_param('patience', c.CV_RN50_PATIENCE)
        mlflow.log_param('epochs', c.CV_RN50_EPOCHS)
        mlflow.log_param('batch_size', c.CV_RN50_BATCH_SIZE)

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
            with open(f'cv_resnet50_{key}.pkl', 'wb') as f:
                pickle.dump(value, f)
            mlflow.log_artifact(f'cv_resnet50_{key}.pkl', artifact_path='evaluation_matrices_folds')
            os.remove(f'cv_resnet50_{key}.pkl')

        # Saving the Model's Summary:
        artifact_path = "cv_resnet_summary.txt"
        with open(artifact_path, "w") as f:
            resnet50_model.summary(print_fn=lambda x: f.write(x + "\n"))
        mlflow.log_artifact(artifact_path, "cv_resnet50_summary.txt")
        os.remove(artifact_path)

        # Logging the Average Confusion Matrix into mlflow:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(mean_cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")  # Log the plot in mlflow
        plt.savefig(f'/{c.REPO_PATH}/data/models/cv_resnet50/confusion_matrix.png')  # Save the plot

        # End the mlflow run:
        mlflow.end_run()

    return None


if __name__ == "__main__":
    # Checking directory:
    _check_create_dir()

    # Loading dataset:
    X_TRAIN_FOLDS, X_TEST_FOLDS, Y_TRAIN_FOLDS, Y_TEST_FOLDS = _load_cv_folds(rep_type='spectrogram')

    # Building, Fitting and Evaluating ResNet50:
    _run_evaluate_resnet50(x_train_folds=X_TRAIN_FOLDS,
                           x_test_folds=X_TEST_FOLDS,
                           y_train_folds=Y_TRAIN_FOLDS,
                           y_test_folds=Y_TEST_FOLDS)
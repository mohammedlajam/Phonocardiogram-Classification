"""
Author: Mohammed Lajam

Phase 5: Classification (Computer Vision - Pre-trained Model: "VGG19"):
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
        with open(f'{c.PROCESSED_DATA_PATH}/images/scalogram/holdout_cv/rgb/x_train.pkl', 'rb') as f:
            x_train = pickle.load(f)
        with open(f'{c.PROCESSED_DATA_PATH}/images/scalogram/holdout_cv/rgb/x_test.pkl', 'rb') as f:
            x_test = pickle.load(f)
        with open(f'{c.PROCESSED_DATA_PATH}/images/scalogram/holdout_cv/rgb/y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        with open(f'{c.PROCESSED_DATA_PATH}/images/scalogram/holdout_cv/rgb/y_test.pkl', 'rb') as f:
            y_test = pickle.load(f)
    except FileNotFoundError:
        print("Error: One or more files not found")
    except Exception as e:
        print(f"Error: {e}")
    else:
        return x_train, x_test, y_train, y_test


def _generate_val_set(x_train, x_test, y_train, y_test):
    """Function to prepare the data for VGG19."""
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


def _run_evaluate_vgg19(x_train, x_test, y_train, y_test):
    """Function to run and evaluate VGG19 Model based on Manual adjustment of Parameters.
    It returns all the Matrices, paramters and Artifacts into mlflow."""
    with mlflow.start_run():
        # Splitting x_train into x_train and x_val:
        x_train, x_test, x_val, y_train, y_test, y_val = _generate_val_set(x_train=x_train,
                                                                           x_test=x_test,
                                                                           y_train=y_train,
                                                                           y_test=y_test)

        # Convert the data to TensorFlow Dataset format:
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(c.CV_VGG19_BATCH_SIZE)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(c.CV_VGG19_BATCH_SIZE)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(c.CV_VGG19_BATCH_SIZE)

        input_shape = x_train[0].shape

        # 1. Building and fitting MLP Model:
        vgg19_model, history = PretrainedModel.build_fit_vgg19(train_dataset=train_dataset,
                                                               val_dataset=val_dataset,
                                                               input_shape=input_shape,
                                                               include_top=c.CV_VGG19_INCLUDE_TOP,
                                                               vgg_weights=c.CV_VGG19_WEIGHTS,
                                                               trainable=c.CV_VGG19_TRAINABLE,
                                                               dense_1=c.CV_VGG19_DENSE_1,
                                                               dense_2=c.CV_VGG19_DENSE_2,
                                                               dense_1_l2=c.CV_VGG19_DENSE_1_L2,
                                                               dense_2_l2=c.CV_VGG19_DENSE_2_L2,
                                                               dropout_rate_1=c.CV_VGG19_DROPOUT_RATE_1,
                                                               dropout_rate_2=c.CV_VGG19_DROPOUT_RATE_2,
                                                               learning_rate=c.CV_VGG19_LEARNING_RATE,
                                                               loss=c.CV_VGG19_LOSS,
                                                               patience=c.CV_VGG19_PATIENCE,
                                                               epochs=c.CV_VGG19_EPOCHS)

        # Saving the Model
        model_name = f"cv_vgg19_model"
        mlflow.sklearn.log_model(vgg19_model, model_name)

        # Plotting and saving Accuracy vs Epoch:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history.history['accuracy'], label=f'Train Accuracy')
        ax.plot(history.history['val_accuracy'], label=f'Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy over Epochs')
        ax.legend()
        mlflow.log_figure(fig, f'Accuracy_vs_Epochs.png')  # Log the plot in mlflow

        # 2. Predictions:
        # Calculating the Probabilities:
        y_prob = ModelEvaluator.calculate_probabilities(model=vgg19_model, x_test=test_dataset)

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
        threshold = evaluation_matrix.iloc[0]['Threshold']
        accuracy = evaluation_matrix.iloc[0]['Accuracy']
        precision = evaluation_matrix.iloc[0]['Precision']
        recall = evaluation_matrix.iloc[0]['Recall']
        f1_score = evaluation_matrix.iloc[0]['F1-score']
        sensitivity = evaluation_matrix.iloc[0]['Sensitivity']
        specificity = evaluation_matrix.iloc[0]['Specificity']
        tn = evaluation_matrix.iloc[0]['TN']
        fp = evaluation_matrix.iloc[0]['FP']
        fn = evaluation_matrix.iloc[0]['FN']
        tp = evaluation_matrix.iloc[0]['TP']
        roc_auc = evaluation_matrix.iloc[0]['ROC_AUC']
        auc_score = evaluation_matrix.iloc[0]['AUC_Score']
        fpr = evaluation_matrix.iloc[0]['FPR']
        tpr = evaluation_matrix.iloc[0]['TPR']

        # 5. logging all Artifacts, Parameters and Matrices into mlflow:
        # Log the Hyper-Parameters:
        mlflow.log_param('include_top', c.CV_VGG19_INCLUDE_TOP)
        mlflow.log_param('weights', c.CV_VGG19_WEIGHTS)
        mlflow.log_param('trainable', c.CV_VGG19_TRAINABLE)
        mlflow.log_param('dense_1', c.CV_VGG19_DENSE_1)
        mlflow.log_param('dense_2', c.CV_VGG19_DENSE_2)
        mlflow.log_param('dense_1_l2', c.CV_VGG19_DENSE_1_L2)
        mlflow.log_param('dense_2_l2', c.CV_VGG19_DENSE_2_L2)
        mlflow.log_param('dropout_rate_1', c.CV_VGG19_DROPOUT_RATE_1)
        mlflow.log_param('dropout_rate_2', c.CV_VGG19_DROPOUT_RATE_2)
        mlflow.log_param('learning_rate', c.CV_VGG19_LEARNING_RATE)
        mlflow.log_param('patience', c.CV_VGG19_PATIENCE)
        mlflow.log_param('epochs', c.CV_VGG19_EPOCHS)
        mlflow.log_param('batch_size', c.CV_VGG19_BATCH_SIZE)

        # Log the Matrices (Evaluation):
        # Mean of Matrices:
        mlflow.log_metric('mean_threshold', threshold)
        mlflow.log_metric('mean_accuracy', accuracy)
        mlflow.log_metric('mean_precision', precision)
        mlflow.log_metric('mean_recall', recall)
        mlflow.log_metric('mean_f1_score', f1_score)
        mlflow.log_metric('mean_sensitivity', sensitivity)
        mlflow.log_metric('mean_specificity', specificity)
        mlflow.log_metric('mean_tn', tn)
        mlflow.log_metric('mean_fp', fp)
        mlflow.log_metric('mean_fn', fn)
        mlflow.log_metric('mean_tp', tp)
        mlflow.log_metric('mean_roc_auc', roc_auc)
        mlflow.log_metric('mean_auc_score', auc_score)

        # Saving fpr_folds and tpr_folds as pickle file and log it into mlflow as artifact:
        with open('cv_vgg19_fpr.pkl', 'wb') as f:
            pickle.dump(fpr, f)
        mlflow.log_artifact('cv_vgg19_fpr.pkl', artifact_path='fpr_tpr')
        os.remove('cv_vgg19_fpr.pkl')

        with open('cv_vgg19_tpr.pkl', 'wb') as f:
            pickle.dump(tpr, f)
        mlflow.log_artifact('cv_vgg19_tpr.pkl', artifact_path='fpr_tpr')
        os.remove('cv_vgg19_tpr.pkl')

        # Saving the Model's Summary:
        artifact_path = "cv_vgg19_summary.txt"
        with open(artifact_path, "w") as f:
            vgg19_model.summary(print_fn=lambda x: f.write(x + "\n"))
        mlflow.log_artifact(artifact_path, "cv_vgg19_summary.txt")
        os.remove(artifact_path)

        # Logging the Average Confusion Matrix into mlflow:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
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

    # Loading dataset:
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = _load_cv_folds()

    # Building, Fitting and Evaluating VGG19:
    _run_evaluate_vgg19(x_train=X_TRAIN,
                        x_test=X_TEST,
                        y_train=Y_TRAIN,
                        y_test=Y_TEST)
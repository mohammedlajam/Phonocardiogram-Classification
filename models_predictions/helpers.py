"""
Author: Mohammed Lajam

This file contains a collection of helper functions that are commonly used in predicting and
evaluating Machine and Deep Learning models 'Phase 6'. These functions provide support for
various tasks such as predicting and evaluating Machine and Deep Learning Models.
The purpose of these helper functions is to encapsulate repetitive and complex code into reusable
and modular blocks, making it easier to maintain and improve the overall functionality of the
project.
"""

# Importing libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, \
    roc_curve, auc

import warnings
warnings.filterwarnings('ignore')

# disabling ssl:
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


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
    def find_best_threshold(y_test, y_prob, evaluation_matrix: str):
        # Extracting fpr, tpr and thresholds:
        _, _, thresholds = roc_curve(y_test, y_prob)
        # Calculating all the evaluation accuracy based on all the thresholds:
        if evaluation_matrix == 'accuracy':
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

        elif evaluation_matrix == 'f1_score':
            f1_score_matrix = []
            for threshold in thresholds:
                y_pred = np.where(y_prob > threshold, 1, 0)
                f1 = f1_score(y_test, y_pred)
                f1_score_matrix.append((threshold, f1))
            # Create a dataframe
            f1_score_matrix_df = pd.DataFrame(f1_score_matrix, columns=['Threshold', 'Accuracy'])
            f1_score_matrix_df.sort_values(by='Accuracy', ascending=False, inplace=True)
            best_f1_score = f1_score_matrix_df.iloc[0]['Accuracy']
            best_threshold = f1_score_matrix_df.iloc[0]['Threshold']
            return best_f1_score, best_threshold

        else:
            raise ValueError(f'Invalid evaluation matrix: {evaluation_matrix}.')

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

    @staticmethod
    def plot_roc_auc(fpr, tpr):
        """Function for plotting ROC_AUC Curve."""
        plt.plot(fpr, tpr, color='red', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Charateristics (ROC) Curve')
        plt.legend()
        return plt.show()

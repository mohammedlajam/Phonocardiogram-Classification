"""
Author: Mohammed Lajam

This file contains a collection of helper functions that are commonly used in Data Preparation
'Phase 4'. These functions provide support for various tasks such as EDA, Feature Engineering and
Feature Selection. The purpose of these helper functions is to  encapsulate repetitive and complex
code into reusable and modular blocks, making it easier to maintain and improve the overall
functionality of the project.
"""
# Importing libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# EDA Libraries:
from scipy.stats import probplot
# Feature Engineering Libraries:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from imblearn.over_sampling import SMOTE
# Feature Selection Libraries:
from scipy.stats import ttest_1samp
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import pointbiserialr
from scipy.stats import pearsonr

import warnings
warnings.filterwarnings('ignore')

# Display all columns of DataFrame:
pd.pandas.set_option('display.max_columns', None)


# 1. Exploratory Data Analysis (EDA):
# 1.1. Continuous Numeric Variables:
class ExploratoryDataAnalysis:
    def __init__(self, dataset):
        self.dataset = dataset

    def extract_initial_analysis(self):
        """Function to get initial insight about the complete dataset."""
        # checking number of subjects and features:
        print(f'Number of Subjects: {self.dataset.shape[0]}')
        print(f'Number of Features: {self.dataset.shape[1]}')
        # Checking missing values:
        print(f'Number of missing values: {self.dataset.isna().sum().sum()}')
        # Checking how many classes:
        print(f'Classes: {self.dataset["class"].unique()}')
        # Checking the number of subject per class:
        print(f'Number of subjects per class:\n {self.dataset["class"].value_counts()}')

    def visualize_feature(self, feature):
        """Function to visualize Histogram, Box-Plot and Q-Q Plot in one representation."""
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        sns.histplot(self.dataset[feature], kde=True)  # Histogram
        plt.subplot(1, 3, 2)
        sns.boxplot(self.dataset[feature])  # Box-Plot
        plt.subplot(1, 3, 3)
        probplot(self.dataset[feature], plot=plt, dist='norm')  # Q-Q Plot
        plt.tight_layout()
        return plt.show()

    def count_outliers(self, feature):
        """Function to count the number of outliers and its percentage."""
        q1 = np.percentile(self.dataset[feature], 25)
        q3 = np.percentile(self.dataset[feature], 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        num_outliers = np.sum((self.dataset[feature] < lower_bound) | (self.dataset[feature] > upper_bound))

        percentage = (num_outliers / len(self.dataset[feature])) * 100
        return num_outliers, percentage

    def calculate_correlation(self, *features):
        """Function to calculate the correlation between a feature and a target."""
        correlations = pd.DataFrame(columns=['Feature', 'Correlation'])
        for feature in features:
            correlation, pval = pointbiserialr(self.dataset[feature], self.dataset['class'])
            correlations = correlations.append({'Feature': feature, 'Correlation': correlation}, ignore_index=True)
        return correlations

    def visualize_correlation(self):
        """Visualize the correlation between multiple features of the dataset as a heatmap."""
        corr = self.dataset.corr()
        sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
        return plt.show()

# 1.2. Images:


# 2. Feature Engineering:
class FeatureEngineering:
    def __init__(self, dataset):
        self.dataset = dataset

    def split_dataset(self, test_size: int, rand_state: int):
        """Function to split the dataset into Train, Test and Validation sets."""
        data = self.dataset.copy()
        features = data.drop(['signal_id', 'class'], axis=1, inplace=False)
        target = data['class']
        x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                            test_size=test_size,
                                                            random_state=rand_state,
                                                            shuffle=True)

        dfs = [x_train, y_train, x_test, y_test]
        # Loop through the list and reset the index for each DataFrame:
        for df in dfs:
            df.reset_index(drop=True, inplace=True)
        return x_train, x_test, y_train, y_test

    def create_cross_validation(self, n_folds: int, rand_state: int):
        """Function to generate the indices for train and test set of each fold of the cross
        validation. It returns two list containing the indices for train and test sets."""
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rand_state)
        train_indices = []
        test_indices = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(self.dataset, self.dataset['class'])):
            train_indices.append(train_idx)
            test_indices.append(test_idx)
        return train_indices, test_indices

    def create_dataframe_from_indices(self, indices_list, index):
        """Function to return a DataFrame for x and y based on the Indices generated from Cross Validation."""
        fold = self.dataset.loc[indices_list[index]]
        fold = fold.reset_index(drop=True)
        x = fold.drop(['signal_id', 'class'], axis=1, inplace=False)
        y = fold['class']
        return x, y

    def apply_gaussian_transformation(self, feature, trans_method: str):
        """Function to apply Gaussian Transformation. trans_method is either 'log', 'reciprocal',
        'square_root', 'cube_root', 'exponential' or 'boxcox' """
        if trans_method == 'log':
            transformed_feature = np.log(self.dataset[feature])
        elif trans_method == 'reciprocal':
            transformed_feature = 1 / self.dataset[feature]
        elif trans_method == 'square_root':
            transformed_feature = np.sqrt(self.dataset[feature])
        elif trans_method == 'cube_root':
            transformed_feature = np.cbrt(self.dataset[feature])
        elif trans_method == 'exponential':
            transformed_feature = self.dataset[feature] ** (1 / 1.2)
        elif trans_method == 'boxcox':
            # BoxCox works only with positive values:
            transformed_feature, parameter = pd.DataFrame(stats.boxcox(self.dataset[feature]))
        else:
            raise ValueError(f"'{trans_method}' is not a valid method. The 'trans_method' is either 'log', "
                             f"'reciprocal', 'square_root', 'exponential' or 'boxcox'")
        return pd.DataFrame(transformed_feature)

    def normalize_data(self, x_train, x_test):
        """Function to normalize datasets."""
        scaler = MinMaxScaler()
        x_train_normalized = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
        x_test_normalized = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
        return x_train_normalized, x_test_normalized

    def balance_dataset(self, x_train, y_train, rand_state: int):
        """Function to balance dataset using SMOTE technique."""
        smote = SMOTE(random_state=rand_state)
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
        return x_train_resampled, y_train_resampled

    def replace_missing_values(self, dataframe, feature: str, replace_method: str):
        if replace_method == 'mean':
            mean = dataframe[feature].median()
            dataframe[feature].fillna(mean, inplace=True)
        elif replace_method == 'median':
            median = dataframe[feature].median()
            dataframe[feature].fillna(median, inplace=True)
        else:
            raise ValueError(
                f"'{replace_method}' is not a valid method. The 'replace_method' is either 'mean' or 'median'.")
        return dataframe


class FeatureSelection:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def conduct_one_sample_ttest(self, *features, sample_size, alpha=0.05):
        """Function to conduct One Sample T-Test on all the features on a dataframe individually.
        If the P-Value is less or equal to 0.05, then this feature is dropped from the dataframe
        and its name is added to a list of dropped features."""
        x_train = self.x_train.copy()
        x_test = self.x_test.copy()
        dropped_features = []
        for feature in features:
            population_mean = np.mean(self.x_train[feature])
            # Creating a random Sample from the Population:
            sample = np.random.choice(self.x_train[feature], sample_size)
            # Calculating the P-Value:
            _, p_value = ttest_1samp(a=sample, popmean=population_mean)
            if p_value < alpha:
                x_train = self.x_train.drop(feature, axis=1)
                x_test = self.x_test.drop(feature, axis=1)
                dropped_features.append(feature)
        return x_train, x_test, dropped_features

    def get_top_features(self, method: str, num_features, plot=False):
        """Function to get the top ranked features (Continuous Features) to the target (Categorical). It returns
        a list of the top ranked features. method is either chi2, mutual_info, feature_importance."""
        if method == 'chi2':
            ranked_features = SelectKBest(score_func=chi2, k=len(self.x_train.columns))
            ranked_features.fit(self.x_train, self.y_train)
            features_scores = pd.DataFrame(ranked_features.scores_, columns=['Scores'])

        elif method == 'mutual_info':
            ranked_features = mutual_info_classif(self.x_train, self.y_train)
            features_scores = pd.DataFrame(ranked_features, columns=['Scores'])

        elif method == 'feature_importance':
            ranked_features = ExtraTreesClassifier()
            ranked_features.fit(self.x_train, self.y_train)
            features_scores = pd.DataFrame(ranked_features.feature_importances_, columns=['Scores'])
        else:
            raise ValueError(f"Invalid representation type: {method}. 'method' is either 'chi2',"
                             f"'mutual_info' or 'feature_importance'")

        features_columns = pd.DataFrame(self.x_train.columns)
        top_ranked_features = pd.concat([features_columns, features_scores], axis=1)
        top_ranked_features = top_ranked_features.nlargest(num_features, 'Scores')
        if plot:
            plt.figure(figsize=(30, 10))
            plt.bar(top_ranked_features.iloc[:, 0], top_ranked_features['Scores'])
            return plt.show()
        else:
            return top_ranked_features

    def drop_correlated_features(self, threshold: float, top_features: list):
        """Function to calculate the correlation between every pair of features and remove
        the correlated feature with higher skewness coefficient."""
        correlated_features = []  # Set of all the names of correlated columns
        corr_matrix = self.x_train.corr()
        skewness = self.x_train.skew()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if i == j:
                    pass
                else:
                    if abs(corr_matrix.iloc[i, j]) > threshold:
                        col_feature = corr_matrix.columns[i]  # Getting the name of column
                        row_feature = corr_matrix.iloc[j].name  # Getting the name of the row
                        # Dropping the feature from the pair, which has higher skewness
                        if abs(skewness[col_feature]) > abs(skewness[row_feature]):
                            correlated_features.append(col_feature)
                        else:
                            correlated_features.append(row_feature)
                    else:
                        pass

        correlated_features = list(set(correlated_features))
        # Remove any feature from correlated_features, which exists in the top_ranked_features:
        correlated_features = [feature for feature in correlated_features if feature not in top_features]
        x_train = self.x_train.drop(correlated_features, axis=1)
        x_test = self.x_test.drop(correlated_features, axis=1)
        return x_train, x_test, correlated_features


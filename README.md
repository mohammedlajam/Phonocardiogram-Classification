### <span style="color: hsl(30, 100%, 80%);">Project's Title:</span>

Phonocardiogram Classification techniques using Machine Learning Models: Comparative Study

### <span style="color: hsl(30, 100%, 80%);">Disclaimer:</span>
This repository is a part of a master thesis in Biomedical Engineering at Hochschule Anhalt.
- Author: Mohammed Lajam
- First supervisor: Prof. Dr. Marianne Maktabi
- Second supervisor: Prof. Dr. Stefan Twieg

### <span style="color: hsl(30, 100%, 80%);">Project's Description:</span>
A Phonocardiogram (PCG) is a graphical representation of the  sounds produced by the heart, 
particularly the heartbeats. It is created by recording the acoustic signals using a sensitive 
microphone or sensor placed on the chest. PCG signals can provide valuable information about the 
condition of the heart and are often used in medical diagnostics to detect abnormalities such as 
heart murmurs or valve disorders. These signals are an important tool in cardiology for assessing 
cardiac health. Machine and Deep Learning are important nowadays the medical sector as it 
automates the analysis of heart sounds, improving accuracy, efficiency, and aiding in the 
diagnosis and management of cardiac conditions.

### <span style="color: hsl(30, 100%, 80%);">Project Objectives:</span>
There are two major objectives:
1. Which data type, tabular or images, performs better for Phonocardiogram Classification?
2. For each Data type, which Machine or Deep Learning Model demonstrates the highest performance 
in Phonocardiogram Classification?

### <span style="color: hsl(30, 100%, 80%);">Project Methodology:</span>
There are 6 phases involved in this project in order to reach the project's objectives.
#### <font color="grey">1. Data Collection:</font>
The data used in this project is a part of [PhysioNet global competition in Cardiology (CinC)](https://physionet.org/content/challenge-2016/1.0.0/) 
in 2016. The downloaded data is in a form of Audio files (.wav).

#### <font color="grey">2. Signal Preprocessing:</font>
This phase involves two major objectives: 
1. Suppressing the noise from the signals using digital filters.
2. Slicing the signals to a specific length in order to unite the length of all the signals as well
as increasing the dataset.

#### <font color="grey">3. Feature Extraction:</font>
The major objective of this phase is to extract two different data types from the signals, which 
are tabular data and images. These features, either tabular or images, are extracted from 
time-domain, frequency-domain and time-frequency representation domain.

#### <font color="grey">4. Data Preparation:</font>
This phase involves applying different techniques to the extracted data in order to prepare them
for machine and deep learning models.
1. Exploratory Data Analysis (EDA),  which is a process of summarizing, visualizing, and 
understanding the main characteristics of a dataset.
2. Feature Engineering, which includes applying different techniques to the dataset such as
Cross Validation, Feature Scaling and Data Balancing.
3. Feature Selection, which includes techniques to select the most relevant features of a dataset.

#### <font color="grey">5. Classification:</font>
This Phase is about building and training Machine and Deep Learning models based on the train set.

__Tabular data models:__
1. Support Vector Machine (SVM)
2. Multilayer Perceptron (MLP)
3. Convolutional Neural Network (CNN)
4. Recurrent Neural Network (RNN-LSTM)
5. Convolutional Recurrent Neural Network (C-RNN)
6. TabNet

__Computer vision models:__
1. Residual Network (ResNet-50)
2. Visual Geometry Group (VGG-19)
3. Inception
4. Inception-ResNet

#### <font color="grey">6. Models' Evaluation and Comparisons:</font>
This Phase is based on making predictions and extract all the evaluation matrices and comparing 
between the models in order to conclude, which data type and model are better for PCG classification.

### <span style="color: hsl(30, 100%, 80%);">The Project's Repository:</span>
This repository includes several python files as well as packages. Python files are the files for 
executing the code and generating outputs. On the other hand, each package includes all functions
and classes used in python files.

#### <font color="grey">Python Files:</font>
1. ___Constants.py___, which includes all the constants used in this project for all the phases
2. ___signal-preprocessing.py___, which includes the code for denoising and slicing the signals
3. ___feature-extraction.py___, which includes the code for extracting both tabular data and images 
from the signals.
4. ___data-preparation-tabular.py___, which includes the code for feature engineering and feature
selection for tabular data
5. ___data-preparation-cv.py___, which includes the code for preparing the images

#### <font color="grey">Packages:</font>
1. ___signal_preprocessing___, which includes helper.py that includes all the functions and classes 
used in ___signal-preprocessing.py___. Additionally, this package two notebooks for visualizations
2. ___feature_extraction___, which includes helper.py that includes all the functions and classes 
used in ___feature-extraction.py___. Additionally, this package two notebooks for visualizations
3. ___data-preparation___, which includes helper.py that includes all the functions and classes 
used in either ___data-preparation-tabular.py___ or ___data-preparation-cv.py___. Additionally, 
this package a notebook for visualizations
4. ___models___, which includes helper.py that includes all the functions and classes used in all 
the models files. Additionally, this package includes several python file, where each python file
belong to the main code of a model.
5. ___models_predictions___, which includes helper.py that includes all the functions and classes 
used in all the models predictions files. Additionally, this package includes several python file, 
where each python file belong to the main code to make predictions.

### <span style="color: hsl(30, 100%, 80%);">Installation:</span>
1. Clone the repository in local machine
2. Install ___requirements.txt___
3. Run ___create-directories.py___
4. Download the dataset from [here](https://physionet.org/content/challenge-2016/1.0.0/) and place
the downloaded data in /data/PhysioNet/
5. Open ___constants.py___ and change ___REPO_PATH___ to the path of the repository in local 
machine
6. Run ___signal_processing.py___
7. Run ___feature_extraction.py___
8. Run ___data-preparation-tabular.py___
9. Run ___data-preparation-cv.py___
10. Open ___models___ package and run all the models files, which are the following: 
- ___tabular_svm.py___
- ___tabular_mlp.py___
- ___tabular_cnn.py___
- ___tabular_rnn.py___
- ___tabular_crnn.py___
- ___tabular_tabnet.py___
- ___cv_resnet50.py___
- ___cv_vgg19.py___
- ___cv_inception.py___
- ___cv_inception_resnet.py___

11. Open ___models_predictions___ package and run all the models predictions files, which are the following:
- ___tabular_svm_evaluation.py___
- ___tabular_mlp_evaluation.py___
- ___tabular_cnn_evaluation.py___
- ___tabular_rnn_evaluation.py___
- ___tabular_crnn_evaluation.py___
- ___tabular_tabnet_evaluation.py___
- ___cv_resnet50_evaluation.py___
- ___cv_vgg19_evaluation.py___
- ___cv_inception_evaluation.py___
- ___cv_inception_resnet_evaluation.py___

12. Open terminal and type "_mlflow ui_", which will open a web page for MLFlow including all
the runs of the models. Then download the table in .csv and place it in the ___/data/models/___ 
directory 

13. Run ___models-comparison.ipynb___ in Jupyter Notebook, which is in ___models_predictions___ package

# AFDetection
Atrial Fibrillation Detection from Electrocardiogram Signal based on Artificial Intelligence

# About
As an important biological signal in the human body, the ECG signal has non-stationary and non-linear characteristics. The analysis of ECG signals is a beneficial tool for diagnosing cardiac diseases, where atrial fibrillation is one of the most prevalent cardiac abnormalities at present. 

Initially, we used purely deep learning methods for analyzing ECG signals: A simple convolutional network (CNN) was created as a baseline. And then, we tried some relatively complex network architectures, such as ResNet and DenseNet, to make the network extract more representative features. However, the more complex the model was, the more difficult it converged during training. As a result, the final accuracy cannot be effectively improved.

This project is implemented based on the  framework for detecting the rhythm of atrial fibrillation (ENCASE), which is proposed by S. Hong et al. https://doi.org/10.22489/CinC.2017.178-245

# Structure
The project consists of the following parts
* Data Preprocessing
  * Generation of Synthetic Data
  * Generation of Train/Val Data
* Feature extraction
  * Manual feature
  * Deep feature
* Classification


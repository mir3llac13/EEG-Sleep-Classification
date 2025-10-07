# EEG Sleep Classification

A Senior Thesis Project for Automatic Sleep Stage Classification Using EEG Signals
This project leverages a 1D Convolutional Neural Network (CNN) to classify sleep stages (e.g., Wake, REM, NREM) based on EEG signals from the sleep-cassette subset of the Sleep-EDF Expanded Dataset.

**Project Overview**


**Objective**
 Automatically classify sleep stages using EEG signals to support sleep research and diagnostics.


**Dataset**
Utilizes the sleep-cassette subset of the Sleep-EDF Expanded Dataset, containing home-recorded EEG signals from healthy subjects.

**Model**
1D Convolutional Neural Network (CNN) implemented with TensorFlow/Keras.

**Tools**
Python, TensorFlow, Keras, Pandas, NumPy, MNE, Jupyter Notebook.

# Installation

To get started with this project, follow these steps:

**1.Install Required Libraries**

**pip install tensorflow numpy pandas mne**

**2.Download the Dataset**

Access the Sleep-EDF Expanded Dataset from Physionet.
Alternatively, download the dataset from Google Drive (link to be added for large files).

**3.Run the Notebook**

Open EEGCONV1D.ipynb in Jupyter Notebook or Google Colab.

# Usage

**Open the Notebook** Launch EEGCONV1D.ipynb in your preferred environment (Jupyter Notebook or Google Colab).



**Load the Dataset** Follow the instructions in the notebook to preprocess and load the EEG data.



**Train and Evaluate** Run the cells to train the 1D CNN model and evaluate its performance on sleep stage classification.

# Dataset

The project uses the Sleep-EDF Expanded Dataset (sleep-cassette subset), which includes EEG recordings from healthy subjects in a home setting. Due to the large size of the dataset, it is not included in this repository. You can obtain it from:


Physionet: Official source for the Sleep-EDF dataset.



Google Drive: (Add your Google Drive link here if uploaded).



Note: Ensure you have sufficient storage and follow the notebook instructions for data preprocessing.
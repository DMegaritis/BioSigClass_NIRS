NIRS-CNN-Classification
NIRS-CNN-Classification is a repository that provides an implementation of a convolutional neural network (CNN) designed for classifying Near-Infrared Spectroscopy (NIRS) data. This model is particularly focused on data from the vastus lateralis muscle. Developed as part of a study on muscle oxygenation and blood flow, it evaluates these parameters across different activity states (rest, unloaded exercise, exercise, and recovery) in both post-hospitalized long COVID-19 patients and healthy, age-matched individuals.

Project Overview
This project centers on classifying NIRS data related to muscle oxygenation and blood flow using a 1D CNN model. The data were collected from various muscle regions via optodes placed on the skin. The primary objective is to classify populations based on their physiological and activity states (such as rest, exercise, and recovery) by analyzing these variables.

Model Overview
Four distinct CNN models have been developed, each with unique input configurations:

Model A:

Uses the last 30 seconds of a time period.
Input data: TOI from 4 channels and one-hot encoded activity labels.
Input Shape: (151, 8)
Model B:

Uses the first 60 seconds of a time period.
Input data: TOI from 4 channels and one-hot encoded activity labels.
Input Shape: (301, 8)
Model C:

Uses the last 30 seconds of a time period without activity labels.
Input data: TOI and nTHI from 4 channels.
Input Shape: (151, 8)
Model D:

Uses the first 60 seconds of a time period without activity labels.
Input data: TOI and nTHI from 4 channels.
Input Shape: (301, 8)
Data Requirements
Input data for these models should conform to the following specifications:

Shape: (time_steps, features)
time_steps: 151 for Models A and C; 301 for Models B and D.
features: 8 features per time step, comprising:
Models A and B: TOI from 4 channels plus one-hot encoded period features (rest, warm-up, exercise, recovery).
Models C and D: TOI and nTHI from 4 channels.
Standardization: Features for Models C and D should be standardized to a mean of 0 and a standard deviation of 1.
Example Data Shapes
Model A: (151, 8)
Model B: (301, 8)
Model C: (151, 8)
Model D: (301, 8)
Installation
To use this repository:

bash
Αντιγραφή κώδικα
git clone https://github.com/DMegaritis/NIRS-CNN-Classification.git
cd NIRS-CNN-Classification
Running the Models
To run a pre-trained model, use the example script in load_model.py. Ensure your data is saved as a .npy file with the following structure:

Shape: (number_of_chunks, 151/301, 8)
151/301 refers to the number of time steps (151 for Models A and C, 301 for Models B and D).
8 refers to the number of features (consistent across all models).

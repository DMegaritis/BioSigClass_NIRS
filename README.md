# NIRS-CNN-Classification
NIRS-CNN-Classification
This repository contains the implementation of a convolutional neural network (CNN) for classifying Near-Infrared Spectroscopy (NIRS) data, particularly from measurements taken from the vastus lateralis muscle. The model was developed as part of a study evaluating muscle oxygenation and blood flow under various activity states (rest, unloaded exercise, exercise, recovery) in post hospitalised patients with long COVID-19 and healthy age matched individuals. Four distinct CNN models have been developed, each with different input configurations.

This project focuses on classifying muscle oxygenation and blood flow data from NIRS using a 1 dimensional CNN model. The NIRS data were collected from different muscle regions using optodes placed on the skin. The aim was to classify the population across different activity and physiological states (e.g., rest, exercise, recovery) based on these variables.

Models
Four different CNN models have been developed, each using different input configurations:

Model A: Uses the last 30 seconds of a time period with TOI data from four channels and one hot encoded period labels indicating different states of activity. Input shape: (151, 8)
Model B: Uses the first 60 seconds of a time period with TOI data from four channels and one hot encoded period labels indicating different states of activity. Input shape: (301, 8)
Model C: Uses the last 30 seconds of a time period, incorporating TOI and nTHI without period lables. Input shape: (151, 8)
Model D: Uses the first 60 seconds of a time period, incorporating TOI and nTHI withoout period labels. Input shape: (301, 8)

Data Requirements
The input data for these models should have the following structure:

Shape: (time_steps, features)
time_steps: The number of sequential time points. For models A and C, this is 151; for models B and D, it is 301.
Features: The number of features in each time step. For Models A and B, this is 8, which includes the Total Oxygen Index (TOI) from 4 channels and one-hot encoded period features (rest, warm-up, exercise, recovery). For Models C and D, this is again 8, which includes TOI, and nTHI from 4 channels. The features should be standardized to have a mean of 0 and a standard deviation of 1 for models C and D.
Example Data Shapes:
Model A: (151, 8)
Model B: (301, 8)
Model C: (151, 8)
Model D: (301, 8)

Installation
Clone the repository:

```
git clone https://github.com/DMegaritis/NIRS-CNN-Classification.git
cd NIRS-CNN-Classification
```

Running the Models 
To run a pre-trained model, use the example script in ```load_model.py```. Ensure your data is saved as a .npy

Shape: (number_of_chunks, 151/301, 8)

151/301 refers to the time steps (151 for Models A and C, 301 for Models B and D).

8/20 refers to the number of features (8 for all models).

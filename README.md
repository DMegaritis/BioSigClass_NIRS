# NIRS-CNN-Classification
NIRS-CNN-Classification
This repository contains the implementation of a convolutional neural network (CNN) for classifying Near-Infrared Spectroscopy (NIRS) data, particularly from measurements taken from the vastus lateralis muscle. The model was developed as part of a study evaluating muscle oxygenation and blood flow under various conditions (rest, warm-up, exercise, recovery) in post hospitalised patients with long COVID-19 and healthy age matched individuals. Four distinct CNN models have been developed, each with different input configurations.

This project focuses on classifying muscle oxygenation and blood flow data from NIRS using a 1 dimensional CNN model. The NIRS data were collected from different muscle regions using optodes placed on the skin to measure variables such as oxygenated hemoglobin (O2Hb), deoxygenated hemoglobin (HHb), total hemoglobin (nTHI), and tissue oxygenation index (TOI). The goal was to classify the population across different physiological states (e.g., rest, exercise, recovery) based on these variables.

Models
Four different CNN models have been developed, each using different input configurations:

Model A: Uses the last 30 seconds of each time period with TOI data from four channels. Input shape: (151, 8)
Model B: Uses the first 60 seconds of each time period with TOI data from four channels. Input shape: (301, 8)
Model C: Uses the last 30 seconds of each time period, incorporating all variables (O2Hb, HHb, TOI, and nTHI) from four channels. Input shape: (151, 20)
Model D: Uses the first 60 seconds of each time period, incorporating all variables (O2Hb, HHb, TOI, and nTHI) from four channels. Input shape: (301, 20)

Data Requirements
The input data for these models should have the following structure:

Shape: (time_steps, features)
time_steps: The number of sequential time points. For models A and C, this is 151; for models B and D, it is 301.
Features: The number of features in each time step. For Models A and B, this is 8, which includes the Total Oxygen Index (TOI) from 4 channels and one-hot encoded period features (rest, warm-up, exercise, recovery). For Models C and D, this is 20, which includes O2Hb, HHb, TOI, and nTHI from 4 channels, along with the one-hot encoded period features (rest, warm-up, exercise, recovery).
Example Data Shapes:
Model A: (151, 8)
Model B: (301, 8)
Model C: (151, 20)
Model D: (301, 20)

Installation
Clone the repository:

```
git clone https://github.com/DMegaritis/NIRS-CNN-Classification.git
cd NIRS-CNN-Classification
```

Running the Models
To run any of the models, you can use the following command:


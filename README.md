# BioSigClass_NIRS
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15650213.svg)](https://doi.org/10.5281/zenodo.15650213)

**BioSigClass_NIRS** is a repository that provides implementations of machine learning (K-Nearest Neighbors (KNN) with Dynamic Time Wrapping (DTW) and Canonical Interval Forests (CIF)) and deep learning models (convolutional neural networks (CNN)) for classifying Near-Infrared Spectroscopy (NIRS) data related to human tissue oxygenation. The models are particularly trained on data from the vastus lateralis muscle. Developed as part of a study on muscle oxygenation and blood flow, the models are trained on these parameters across various physical activity states (rest, unloaded exercise, exercise, and recovery) to classify between post-hospitalized long COVID-19 patients and healthy, age-matched individuals.

## Project Overview

This project centers on classifying NIRS data related to muscle oxygenation and blood flow using machine and deep learning models. The data were collected from various muscle regions via optodes placed on the skin. The primary objective is to classify populations based on their physiological responses across various physical activity states (such as resting, exercising, and recovering from exercise) by analyzing these variables.

## Feature Sets

Four distinct feature sets have been developed, each with unique input configurations:

- **Feature set A**:
  - Trained on the last 30 seconds of a time period and requires 30-second periods as input.
  - Input data: TOI from 4 channels and one-hot encoded activity labels.
  - **Input Shape**: `(151, 8)`

- **Feature set B**:  
  - Trained on the first 60 seconds of a time period and requires 60-second periods as input.
  - Input data: TOI from 4 channels and one-hot encoded activity labels.
  - **Input Shape**: `(301, 8)`

- **Feature set C**:  
  - Trained on the last 30 seconds of a time period without activity labels and requires 30-second periods as input.
  - Input data: TOI and nTHI from 4 channels.
  - **Input Shape**: `(151, 8)`

- **Feature set D**:  
  - Trained on the first 60 seconds of a time period without activity labels and requires 60-second periods as input.
  - Input data: TOI and nTHI from 4 channels.
  - **Input Shape**: `(301, 8)`

## Data Requirements

Input data for these models should conform to the following specifications:

- **Shape**: `(time_steps, features)`
  - `time_steps`: 151 for Models A and C (30 seconds); 301 for Models B and D (60 seconds).
  - `features`: 8 features per time step, comprising:
    - **Models A and B**: TOI from 4 channels plus one-hot encoded period features (rest, warm-up, exercise, recovery).
    - **Models C and D**: TOI and nTHI from 4 channels.
  - **Standardization**: Features for Models C and D should be standardized to a mean of 0 and a standard deviation of 1.

### Example Data Shapes

- **Feature set A**: `(151, 8)`
- **Feature set B**: `(301, 8)`
- **Feature set C**: `(151, 8)`
- **Feature set D**: `(301, 8)`

## Installation

To set up the environment and install the required dependencies for this repository, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/DMegaritis/BioSigClass_NIRS.git
    cd BioSigClass_NIRS
    ```

2. Install dependencies using Poetry:

    ```bash
    poetry install
    ```

This will create a virtual environment and install all necessary dependencies as specified in the `pyproject.toml` file.


## Examples

The ```examples``` folder contains:

**Sample Test Data:** ```example_data.npy``` includes example data to test and understand the model's functionality.

**Model Loading Script:** An example script ```load_model.py``` that demonstrates how to load a pre-trained model and use it with new data.


## Notes to aid re-training of the models
During training, the following hyperparameters were used for each model:

| Feature Set | CNN – Batch Size | CNN – Epochs | CIF – n_estimators | KNN – n_neighbors |
|--------------|------------------|---------------|--------------------|-------------------|
| A            | 16               | 180           | 100                | 20                |
| B            | 32               | 50            | 75                 | 20                |
| C            | 16               | 160           | 125                | 20                |
| D            | 32               | 180           | 75                 | 20         

### Initial training
### KNN
```
knn = KNN_DTW_Classifier(features, target, groups, n_splits=5, n_neighbors=15, scale=True)
knn.train()
```
### CIF
```
cif = CIF_Classifier(features, target, groups, n_splits=5, n_estimators=50)
cif.train()
```
### CNN
```python
cnn = CNN_Classifier(features=features, target=target, groups=groups, n_splits=5, epochs=50, batch_size=32, scale=True)
cnn.train()
```

## Abbreviations
TOI: Total Oxygen Index (refered to as StiO2 as well)

nTHI: normalized Total Haemoglobin Index

## Citation

If you use the pretrained models or the training procedures provided in this repository in your research, please cite:

> Dimitrios Megaritis (2025). *BioSigClass_NIRS*. Zenodo. https://doi.org/10.5281/zenodo.15650213

```bibtex
@software{megaritis_2025_biosigclassnirs,
  author       = {Dimitrios Megaritis},
  title        = {BioSigClass\_NIRS},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15650213},
  url          = {https://doi.org/10.5281/zenodo.15650213}
}


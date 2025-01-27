# BioSigClass_NIRS

**BioSigClass_NIRS** is a repository that provides implementations of convolutional neural networks (CNN) for classifying Near-Infrared Spectroscopy (NIRS) data related to human tissue oxygenation. The models are particularly focused on data from the vastus lateralis muscle. Developed as part of a study on muscle oxygenation and blood flow, the models are trained on these parameters across various activity states (rest, unloaded exercise, exercise, and recovery) to differentiate between post-hospitalized long COVID-19 patients and healthy, age-matched individuals.

## Project Overview

This project centers on classifying NIRS data related to muscle oxygenation and blood flow using a 1D CNN model. The data were collected from various muscle regions via optodes placed on the skin. The primary objective is to classify populations based on their physiological responses across various activity states (such as resting, exercising, and recovering from exercise) by analyzing these variables.

## Model Overview

Four distinct CNN models have been developed, each with unique input configurations:

- **Model A**:  
  - Uses the last 30 seconds of a time period.
  - Input data: TOI from 4 channels and one-hot encoded activity labels.
  - **Input Shape**: `(151, 8)`

- **Model B**:  
  - Uses the first 60 seconds of a time period.
  - Input data: TOI from 4 channels and one-hot encoded activity labels.
  - **Input Shape**: `(301, 8)`

- **Model C**:  
  - Uses the last 30 seconds of a time period without activity labels.
  - Input data: TOI and nTHI from 4 channels.
  - **Input Shape**: `(151, 8)`

- **Model D**:  
  - Uses the first 60 seconds of a time period without activity labels.
  - Input data: TOI and nTHI from 4 channels.
  - **Input Shape**: `(301, 8)`

## Data Requirements

Input data for these models should conform to the following specifications:

- **Shape**: `(time_steps, features)`
  - `time_steps`: 151 for Models A and C; 301 for Models B and D.
  - `features`: 8 features per time step, comprising:
    - **Models A and B**: TOI from 4 channels plus one-hot encoded period features (rest, warm-up, exercise, recovery).
    - **Models C and D**: TOI and nTHI from 4 channels.
  - **Standardization**: Features for Models C and D should be standardized to a mean of 0 and a standard deviation of 1.

### Example Data Shapes

- **Model A**: `(151, 8)`
- **Model B**: `(301, 8)`
- **Model C**: `(151, 8)`
- **Model D**: `(301, 8)`

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
During training, the following parameters were used:

| Parameter     | Value    |
|---------------|----------|
| n_splits      | 5        |
| epochs        | 50       |
| batch_size    | 32       |

### Initial training
```python
cnn_classifier = CNN_Classifier(features=features, target=target, groups=groups, n_splits=5, epochs=50, batch_size=32)
cnn_classifier.train()
```

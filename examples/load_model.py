import numpy as np
import os
from tensorflow.keras.models import load_model

# Load the model
model_path = '../pre_trained_models/model_D.h5'
model = load_model(model_path)

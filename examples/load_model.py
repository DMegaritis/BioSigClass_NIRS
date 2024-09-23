import numpy as np
import os
try:
    from tensorflow.keras.models import load_model
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")

# Load the model
model_path = '../pre_trained_models/model_D.h5'
model = load_model(model_path)

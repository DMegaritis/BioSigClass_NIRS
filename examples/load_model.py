import numpy as np
try:
    from tensorflow.keras.models import load_model
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")

# Load the model
model_path = '../pre_trained_models/model_D.h5'
model = load_model(model_path)

# Load the example data
example_data_path = '../examples/example_data.npy'
data = np.load(example_data_path)

# Make predictions
predictions = model.predict(data)
predicted_classes = (predictions >= 0.5).astype(int)
print("Predictions:", predicted_classes)

# Map the predicted class (0 or 1) to their respective labels: "Healthy" or "Covid"
predicted_labels = ["Healthy", "Covid"]
predicted_mapped = [predicted_labels[i] for i in predicted_classes.flatten()]
print(predicted_mapped)

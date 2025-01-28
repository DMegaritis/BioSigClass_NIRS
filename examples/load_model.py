import numpy as np
from sklearn.preprocessing import StandardScaler
try:
    from tensorflow.keras.models import load_model
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")

# Load the model
model_path = '../pre_trained_models/CNN/model_D.h5'
model = load_model(model_path)

# Optionally we can recompile the model (using the same loss and optimizer as during training)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the example data
example_data_path = '../examples/example_data.npy'
data = np.load(example_data_path)

# Reshaping the data to scale (example data have a shape of (5, 301, 8))
reshaped_data = data.reshape(-1, 8)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(reshaped_data)
# Reshaping back to the original 3D shape of the example data (5, 301, 8)
scaled_data = scaled_data.reshape(5, 301, 8)

# Make predictions
predictions = model.predict(scaled_data)
predicted_classes = (predictions >= 0.5).astype(int)
print("Predictions:", predicted_classes)

# Map the predicted class (0 or 1) to their respective labels: "Healthy" or "Covid"
predicted_labels = ["Healthy", "Covid"]
predicted_mapped = [predicted_labels[i] for i in predicted_classes.flatten()]
print(predicted_mapped)

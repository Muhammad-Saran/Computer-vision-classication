import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.preprocessing import image

# Load the trained model
model = keras.models.load_model("model.h5")

# Function to preprocess and predict on a new image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  # Load image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)[0][0]  # Get prediction score

    # Interpret results
    if prediction > 0.5:
        print(f"Prediction: Dog ({prediction:.4f})")
    else:
        print(f"Prediction: Cat ({1 - prediction:.4f})")

# Example usage
predict_image("path_to_your_test_image.jpg")  # Replace with the path to your test image

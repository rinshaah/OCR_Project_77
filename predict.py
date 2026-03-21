import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("ocr_model.h5")

# Load image (change this to your image name)
img = cv2.imread("four.jpg", cv2.IMREAD_GRAYSCALE)

# Resize to 28x28
img = cv2.resize(img, (28, 28))

# Normalize
img = img / 255.0

# Reshape for model
img = img.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(img)

print("Predicted digit:", np.argmax(prediction))

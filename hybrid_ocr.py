import cv2
import numpy as np
import tensorflow as tf
import pytesseract

# Load CNN model
model = tf.keras.models.load_model("ocr_model.h5")

# Load image
input_img = cv2.imread("text5.png")

#convert image to grayscale

gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

# Resize for CNN check
resized = cv2.resize(gray, (28, 28))
norm = resized / 255.0
norm = norm.reshape(1, 28, 28, 1)

# Check if image looks like a single character
# (simple heuristic: small width or height)
h, w = gray.shape

if h < 50 and w < 50:
    # Use CNN
    prediction = model.predict(norm)
    print("CNN Prediction:", np.argmax(prediction))

else:
    # Use Tesseract for text
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    text = pytesseract.image_to_string(
        thresh,
        config='--psm 6'
    )

    print("Tesseract Output:")
    print(text)

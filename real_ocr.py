import cv2
import pytesseract

# If needed, specify path (usually not required in Linux)
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Load image
img = cv2.imread("K.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding (important for OCR)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# OCR
text = pytesseract.image_to_string(thresh)

print("Detected Text:")
print(text)

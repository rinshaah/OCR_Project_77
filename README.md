# Hybrid OCR System using CNN & Tesseract

## Project Overview
Manual data entry from images is time-consuming and prone to human error. 
This project develops a robust **Optical Character Recognition (OCR)** system that leverages a hybrid approach:
1. **Custom CNN:** Optimized for high-accuracy single-character and digit recognition.
2. **Tesseract OCR:** Utilized for complex full-text extraction and document processing.

---

## Problem Statement
Manual data entry from images is time-consuming and error-prone. This project aims to develop an OCR system 
that can automatically detect and extract text from images, including both single characters and full text, 
using a hybrid approach combining CNN and Tesseract OCR.

---

## Objectives
* **Character Recognition:** Use a Convolutional Neural Network (CNN) for precise digit and letter identification.
* **Text Extraction:** Integrate Tesseract OCR for multi-line and full-sentence recognition.
* **Hybrid Logic:** Build a system for better accuracy.
* **Automation:** Process images and extract readable text automatically.

---

## Tools & Technologies
* **Language:** Python 3.12.3
* **Deep Learning:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **OCR Engine:** Tesseract OCR
* **Data Science:** NumPy
* **Environment:** Ubuntu Linux / Git

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com
cd ocr-cnn-tesseract
```


### 2. Create and Activate Virtual Environment
```bash
python3 -m venv ocr_env
source ocr_env/bin/activate
```

### 3. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR
```bash
sudo apt install tesseract-ocr
```

---

## Execution Procedure
* Place the input image in the project folder.
Run the OCR program:
```bash
python hybrid_ocr.py
```

### The System will:
* Detect whether the input is a single character or full text.
* Use CNN or Tesseract accordingly.
* Display the recognized text as output.

---

## Output Screenshots
* Input Image: (Insert image here)
* Terminal execution: (Insert image here)
* Sample Output:
```bash
Enter image path: text4.png
```
```bash
OCR_Result: HELLO WORLD
```

---

## Conclusion
This project successfully demonstrates a hybrid OCR system capable of recognizing both single 
characters and full text from images. The combination of CNN and Tesseract improves accuracy and flexibility. 
The system can be further enhanced for real-time recognition and handwritten text detection.

---

## Future Scope
* Improve accuracy using advanced deep learning models.
* Add support for handwritten text recognition.
* Develop a graphical user interface (GUI).
* Deploy as a web or mobile application.

---

## Author:
 Rinsha

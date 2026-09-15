# Computer Vision: Object Detection with OpenCV & Python

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x%20%7C%205.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Coursera Project](https://img.shields.io/badge/Coursera-Guided%20Project-0056D2?logo=coursera&logoColor=white)](Coursera%20Y9JV2HPE7VUZ.pdf)

A comprehensive hands-on computer vision project implementing real-time object detection using **Haar Feature-based Cascade Classifiers** with **OpenCV** and **Python**. This project covers static image detection (faces, eyes, vehicle license plates) and dynamic video stream processing (pedestrians, vehicles).

---

## Table of Contents

- [Overview](#overview)
- [How Haar Cascade Classifiers Work](#how-haar-cascade-classifiers-work)
- [Project Modules](#project-modules)
  - [1. Face Detection](#1-face-detection)
  - [2. Eye Detection](#2-eye-detection)
  - [3. Combined Face & Eye Detection](#3-combined-face--eye-detection)
  - [4. Pedestrian Detection in Video](#4-pedestrian-detection-in-video)
  - [5. Vehicle Detection in Video](#5-vehicle-detection-in-video)
  - [6. License Plate Detection & Privacy Blurring](#6-license-plate-detection--privacy-blurring)
- [Detection Results Gallery](#detection-results-gallery)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebooks](#running-the-notebooks)
- [Key Hyperparameters](#key-hyperparameters)
- [Certificate](#certificate)
- [License](#license)

---

## Overview

Object detection is a cornerstone discipline in Computer Vision. While modern deep learning architectures (YOLO, SSD, Faster R-CNN) dominate complex multi-class detection, classical **Haar Cascade Classifiers** (proposed by Paul Viola and Michael Jones in 2001) remain widely favored for edge computing, low-resource embedded hardware, and ultra-fast real-time applications.

This repository demonstrates how to configure, fine-tune, and execute pre-trained Haar Cascades across multiple object categories using Python and OpenCV.

---

## How Haar Cascade Classifiers Work

The Viola-Jones detection framework operates through four fundamental innovations:

1. **Haar Features**: Convolutional-like rectangular windows that capture contrasts in pixel intensity (edge features, line features, center-surround features).
2. **Integral Images**: An intermediate representation that computes the sum of pixel values in any arbitrary bounding rectangle in constant time $O(1)$.
3. **AdaBoost Training**: An adaptive boosting algorithm that selects a small number of critical features from hundreds of thousands of candidate features.
4. **Cascading Architecture**: Classifiers are grouped into stages of increasing complexity. Regions of interest that fail an early stage are immediately rejected, saving computing power for potential target regions.

---

## Project Modules

| # | Notebook | Target | Cascade Model | Media Type |
|---|---|---|---|---|
| **01** | [`01_Face_Detection.ipynb`](01_Face_Detection.ipynb) | Human Faces | `haarcascade_frontalface_default.xml` | Image |
| **02** | [`02_Eyes_Detection.ipynb`](02_Eyes_Detection.ipynb) | Human Eyes | `haarcascade_eye.xml` | Image / ROI |
| **03** | [`03_Face_and_Eyes_Detection.ipynb`](03_Face_and_Eyes_Detection.ipynb) | Faces + Eyes | `haarcascade_frontalface_default.xml` + `haarcascade_eye.xml` | Hierarchical ROI |
| **04** | [`04_Pedestrians_Detection.ipynb`](04_Pedestrians_Detection.ipynb) | Walking Pedestrians | `haarcascade_fullbody.xml` | Video Stream |
| **05** | [`05_Car_Detection.ipynb`](05_Car_Detection.ipynb) | Road Vehicles | `haarcascade_car.xml` | Video Stream |
| **06** | [`06_Car_Plates_Detection.ipynb`](06_Car_Plates_Detection.ipynb) | License Plates & Anonymization | `haarcascade_plate_number.xml` | Image + Gaussian Blur |

---

### 1. Face Detection
- **Notebook**: [`01_Face_Detection.ipynb`](01_Face_Detection.ipynb)
- **Classifier**: `haarcascade_frontalface_default.xml`
- **Workflow**:
  1. Load image via `cv2.imread()`.
  2. Convert BGR color space to Grayscale using `cv2.cvtColor()`.
  3. Detect multi-scale facial bounding boxes via `detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)`.
  4. Overlay bounding rectangles via `cv2.rectangle()`.

### 2. Eye Detection
- **Notebook**: [`02_Eyes_Detection.ipynb`](02_Eyes_Detection.ipynb)
- **Classifier**: `haarcascade_eye.xml`
- **Workflow**:
  1. Detect ocular features across the full image.
  2. Crop and isolate detected eye regions of interest (ROI) for individual inspection.

### 3. Combined Face & Eye Detection
- **Notebook**: [`03_Face_and_Eyes_Detection.ipynb`](03_Face_and_Eyes_Detection.ipynb)
- **Workflow**:
  1. Detect the outer facial bounding box first.
  2. Constrain eye detection strictly within the detected face ROI (`gray[y:y+h, x:x+w]`).
  3. Drastically cuts down false-positive eye detections outside the face region.

### 4. Pedestrian Detection in Video
- **Notebook**: [`04_Pedestrians_Detection.ipynb`](04_Pedestrians_Detection.ipynb)
- **Classifier**: `haarcascade_fullbody.xml`
- **Workflow**:
  1. Open video stream using `cv2.VideoCapture('assets/video/People_Walking.mp4')`.
  2. Loop frame-by-frame, convert to grayscale.
  3. Detect full-body silhouettes (`scaleFactor=1.2`, `minNeighbors=3`).
  4. Draw bounding boxes and render live preview with `cv2.imshow()`.

### 5. Vehicle Detection in Video
- **Notebook**: [`05_Car_Detection.ipynb`](05_Car_Detection.ipynb)
- **Classifier**: `haarcascade_car.xml`
- **Workflow**:
  1. Process traffic surveillance video (`assets/video/Vehicles.mp4`).
  2. Detect passing vehicles across varying illumination conditions.
  3. Render tracking bounding boxes on moving cars.

### 6. License Plate Detection & Privacy Blurring
- **Notebook**: [`06_Car_Plates_Detection.ipynb`](06_Car_Plates_Detection.ipynb)
- **Classifier**: `haarcascade_plate_number.xml`
- **Workflow**:
  1. Detect vehicular license plates using custom plate cascade.
  2. Extract plate coordinates $(x, y, w, h)$.
  3. Apply an intense Gaussian Blur kernel (`cv2.medianBlur()` or `cv2.GaussianBlur()`) over the plate ROI to preserve privacy while keeping the rest of the image intact.

---

## Detection Results Gallery

| Detection Task | Output Preview |
|---|---|
| **Face Detection** | ![Face Detection Result](assets/images/face_detection_result.png) |
| **Eye Detection** | ![Eye Detection Result](assets/images/eye_detection_result.png) |
| **Face & Eyes Detection** | ![Face & Eyes Detection Result](assets/images/face_and_eyes_detection_result.png) |
| **Plate Detection & Blurring** | ![Car Plate Detection Result](assets/images/car_plate_detection_result.png) |


---

## Getting Started

### Prerequisites
- Python 3.8 to 3.12+
- `pip` package manager
- Webcam (optional, for live video testing)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mohd-faizy/13P_Object-Detection-with-OpenCV-&-Python.git
   cd 13P_Object-Detection-with-OpenCV-&-Python
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate

   # On Windows:
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebooks

Launch the Jupyter Notebook interface:
```bash
jupyter notebook
```
Open any of the 6 notebooks (`01_Face_Detection.ipynb` through `06_Car_Plates_Detection.ipynb`) and select **Run All Cells**.

---

## Key Hyperparameters

When invoking `detectMultiScale()`, three key hyperparameters dictate accuracy and speed:

$$\text{detectMultiScale}(\text{image}, \text{scaleFactor}, \text{minNeighbors}, \text{minSize})$$

- **`scaleFactor`** (e.g., `1.1` to `1.3`):
  Specifies how much the image size is reduced at each image scale. Smaller values (e.g., `1.05`) increase accuracy for small objects at the cost of higher compute time.
- **`minNeighbors`** (e.g., `3` to `6`):
  Specifies how many neighbors each candidate rectangle should retain to pass the detection threshold. Higher values eliminate false positives but may miss subtle detections.
- **`minSize`** (e.g., `(30, 30)`):
  Minimum possible object size; smaller candidate regions are discarded immediately.

---

## Quick Code Example

```python
import cv2
import matplotlib.pyplot as plt

# 1. Load the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

# 2. Read and preprocess the image
img = cv2.imread('assets/images/eye_face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# 4. Draw bounding boxes
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

# 5. Display the result
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
```

---

## Certificate

This project was completed as part of the Coursera Guided Project **"Computer Vision - Object Detection with OpenCV and Python"**.  
Official credential verification: [Coursera Certificate PDF](Coursera%20Y9JV2HPE7VUZ.pdf).

---

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for complete details.


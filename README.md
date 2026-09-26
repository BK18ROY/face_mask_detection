# Face Mask Detection System

A production-style, real-time Computer Vision and Deep Learning system for human face detection and mask compliance monitoring. Built with **Python 3.10+**, **OpenCV**, and **TensorFlow / Keras**.

---

## Overview

The **Face Mask Detection System** identifies human faces from live webcam streams or pre-recorded video files in real time. Each localized face is extracted, normalized, and classified into one of two states:
1. **MASK** (Compliant)
2. **NO MASK** (Non-compliant)

The system displays dynamic bounding boxes with confidence scores, maintains a live compliance HUD counter, triggers visual and optional audio warnings upon violation detection, and captures rate-limited violation screenshots for auditing.

---

## Features

- **Real-Time Face Detection**: Powered by OpenCV Haar Cascade classifier for high-framerate face localization on CPU and GPU.
- **Custom Deep CNN Classifier**: Built from scratch using TensorFlow/Keras (no pre-trained transfer learning black boxes), engineered specifically for binary mask classification.
- **Dual Stream Support**: Seamlessly processes live webcams (`python main.py webcam`) and pre-recorded video files (`python main.py video --input <path>`).
- **Calibrated Confidence Scores**: True class-specific confidence percentages derived from the sigmoid decision boundary.
- **Real-Time Compliance HUD**: Live on-screen counters showing total detected faces, masked individuals, and unmasked individuals.
- **Visual & Audio Alert System**: Instant flashing alert banners (`WARNING: MASK REQUIRED`) and system beeps when non-compliance is detected.
- **Surveillance Snapshot Engine**: Automated timestamped screenshot capture saved to `outputs/screenshots/` with configurable cooldown to prevent storage saturation.
- **Optional Video Recording**: Write annotated video streams with bounding boxes and compliance overlays directly to `outputs/videos/`.
- **Comprehensive Evaluation Pipeline**: Confusion matrices, precision-recall metrics, F1 scores, and classification reports saved to `outputs/reports/`.
- **Training Visualization**: Automated generation of publication-quality loss and accuracy curves saved to `outputs/plots/`.
- **100% Offline & Private**: Zero cloud uploads; all inference and image processing occur strictly on the local machine.

---

## Architecture Pipeline

```text
Camera / Video Stream
        ↓
OpenCV Video Capture
        ↓
Haar Cascade Face Detection
        ↓
Face ROI Crop (x, y, w, h)
        ↓
Resize (128x128) & RGB Normalization [0, 1]
        ↓
Custom CNN Inference (Sigmoid Probability)
        ↓
Binary Decision (Threshold = 0.5)
   ├── Probability < 0.5  → [MASK] (Green Box + Score)
   └── Probability >= 0.5 → [NO MASK] (Red Box + Score)
        ↓
Compliance HUD Counters Updated
        ↓
Violation Trigger Check
   ├── If NO MASK > 0 → Display Warning Banner + Cooldown Screenshot
   └── Video Frame Annotation & Display (OpenCV GUI)
```

---

## Tech Stack

- **Language**: Python 3.10 / 3.11
- **Computer Vision**: OpenCV (`opencv-python`)
- **Deep Learning**: TensorFlow & Keras
- **Numerical Computing**: NumPy
- **Data Visualization**: Matplotlib
- **Metrics & Evaluation**: scikit-learn
- **Testing**: pytest

---

## Installation

### 1. Clone or Open the Repository

```bash
cd FACE_MASK_DETECTION
```

### 2. Create and Activate Virtual Environment

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt / PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

The CNN requires raw training images divided into `with_mask` and `without_mask` directories:

```text
data/
└── raw/
    ├── with_mask/
    │   ├── 0001.jpg
    │   └── ...
    └── without_mask/
        ├── 0001.jpg
        └── ...
```

### Where to Obtain Public Datasets

1. **Kaggle Face Mask Dataset (by Omkar Gurav)**:
   - Link: [Kaggle Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
   - Contains ~7,553 images (3,725 with mask, 3,828 without mask).
2. **Prajna Bhandary Observations Dataset**:
   - Link: [GitHub observations dataset](https://github.com/prajnasb/observations/tree/master/experiements/data)

Download the dataset images and place them into `data/raw/with_mask/` and `data/raw/without_mask/`.

### Preparing and Splitting Data

Run the preparation pipeline to validate images, remove corrupted files, and partition them into 80% train, 10% validation, and 10% test splits:

```bash
python main.py prepare-data
```

*(Optional bootstrap mode: if you do not have the raw dataset downloaded yet and want to verify the entire pipeline end-to-end, execute `python main.py prepare-data --generate-samples` to generate benchmark samples.)*

---

## Usage Guide

### 1. Model Training

Train the custom CNN model with Early Stopping, ReduceLROnPlateau, and ModelCheckpoint callbacks:

```bash
python main.py train
```

Customizable training parameters:
```bash
python main.py train --epochs 25 --batch-size 32 --lr 0.0001
```

Outputs generated:
- Best model: `models/face_mask_detector_best.keras`
- Final model: `models/face_mask_detector.keras`
- Learning curves: `outputs/plots/training_history.png`

---

### 2. Model Evaluation

Evaluate the trained model on the independent test dataset:

```bash
python main.py evaluate
```

Outputs generated:
- Confusion matrix plot: `outputs/reports/confusion_matrix.png`
- Classification report: `outputs/reports/classification_report.txt`

---

### 3. Live Webcam Detection

Launch real-time webcam surveillance:

```bash
python main.py webcam
```

Options:
```bash
# Specify camera index (default: 0)
python main.py webcam --camera-index 0

# Adjust decision threshold (default: 0.5)
python main.py webcam --threshold 0.55

# Set violation screenshot cooldown in seconds (default: 5.0)
python main.py webcam --cooldown 3.0

# Record video session to outputs/videos/
python main.py webcam --record
```

Press **Q** or **ESC** in the OpenCV window to exit.

---

### 4. Video File Processing

Process any pre-recorded video:

```bash
python main.py video --input path/to/video.mp4
```

With recording enabled:
```bash
python main.py video --input path/to/video.mp4 --record
```

---

### 5. Running Automated Tests

Run the unit tests covering model architecture, shapes, inference bounds, and image preprocessing:

```bash
pytest
```

or via CLI:

```bash
python main.py test
```

---

## Project Structure

```text
FACE_MASK_DETECTION/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Production dependencies
├── .gitignore                         # Git exclusion rules
├── LICENSE                            # MIT open-source license
├── config.py                          # Centralized configuration & hyperparams
├── main.py                            # Unified CLI application
│
├── data/
│   ├── raw/
│   │   ├── with_mask/                 # Raw images with face masks
│   │   └── without_mask/              # Raw images without face masks
│   │
│   └── processed/
│       ├── train/                     # 80% Training split (augmented)
│       │   ├── with_mask/
│       │   └── without_mask/
│       ├── validation/                # 10% Validation split
│       │   ├── with_mask/
│       │   └── without_mask/
│       └── test/                      # 10% Independent test split
│           ├── with_mask/
│           └── without_mask/
│
├── models/
│   ├── .gitkeep
│   ├── face_mask_detector_best.keras  # Best checkpoint (val_loss)
│   └── face_mask_detector.keras       # Final trained model
│
├── outputs/
│   ├── screenshots/                   # Violation capture snapshots
│   ├── plots/                         # Accuracy & loss graphs
│   ├── reports/                       # Confusion matrix & metrics text
│   └── videos/                        # Recorded video files
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # Preprocessing, normalization & tf.data
│   │   └── prepare_dataset.py         # Image validation, cleaning & split
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn_model.py               # Custom deep CNN architecture
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py                   # Model training and callbacks
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py                # Test set evaluation & report generation
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── webcam.py                  # Live video inference & HUD overlay
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                  # Standardized logging setup
│       └── visualization.py           # Drawing routines & matplotlib plotting
│
├── tests/
│   ├── __init__.py
│   ├── test_model.py                  # CNN model architecture & shape tests
│   └── test_preprocessing.py          # Image resize & normalization tests
│
└── assets/
    ├── README.md                      # Asset usage guidelines
    └── haarcascade_frontalface_default.xml # Haar Cascade face detector
```

---

## Model Architecture Details

The custom deep Convolutional Neural Network is built specifically for binary mask classification without transfer learning overhead:

```text
Layer (type)                     Output Shape          Param #
=================================================================
face_input (InputLayer)          (None, 128, 128, 3)   0
conv1_1 (Conv2D - 32, 3x3)       (None, 128, 128, 32)  896
bn1 (BatchNormalization)         (None, 128, 128, 32)  128
relu1_1 (Activation)             (None, 128, 128, 32)  0
conv1_2 (Conv2D - 32, 3x3)       (None, 128, 128, 32)  9,248
relu1_2 (Activation)             (None, 128, 128, 32)  0
pool1 (MaxPooling2D 2x2)         (None, 64, 64, 32)    0
-----------------------------------------------------------------
conv2_1 (Conv2D - 64, 3x3)       (None, 64, 64, 64)    18,496
bn2 (BatchNormalization)         (None, 64, 64, 64)    256
relu2_1 (Activation)             (None, 64, 64, 64)    0
conv2_2 (Conv2D - 64, 3x3)       (None, 64, 64, 64)    36,928
relu2_2 (Activation)             (None, 64, 64, 64)    0
pool2 (MaxPooling2D 2x2)         (None, 32, 32, 64)    0
-----------------------------------------------------------------
conv3_1 (Conv2D - 128, 3x3)      (None, 32, 32, 128)   73,856
bn3 (BatchNormalization)         (None, 32, 32, 128)   512
relu3_1 (Activation)             (None, 32, 32, 128)   0
conv3_2 (Conv2D - 128, 3x3)      (None, 32, 32, 128)   147,584
relu3_2 (Activation)             (None, 32, 32, 128)   0
pool3 (MaxPooling2D 2x2)         (None, 16, 16, 128)   0
-----------------------------------------------------------------
conv4 (Conv2D - 256, 3x3)        (None, 16, 16, 256)   295,168
bn4 (BatchNormalization)         (None, 16, 16, 256)   1,024
relu4 (Activation)               (None, 16, 16, 256)   0
global_avg_pool (GlobalAvgPool)  (None, 256)           0
-----------------------------------------------------------------
dense1 (Dense - 128)             (None, 128)           32,896
head_bn (BatchNormalization)     (None, 128)           512
dropout (Dropout - 0.5)          (None, 128)           0
mask_output (Dense - 1, Sigmoid) (None, 1)             129
=================================================================
Total params: 616,833 (2.35 MB)
Trainable params: 615,617 (2.35 MB)
Non-trainable params: 1,216 (4.75 KB)
```

- **Sigmoid Output**: Represents the estimated posterior probability $P(\text{NO\_MASK} \mid x)$.
- **Confidence Computation**:
  $$\text{Confidence} = \begin{cases} P(\text{NO\_MASK} \mid x) & \text{if } P \ge 0.5 \text{ (NO MASK)} \\ 1.0 - P(\text{NO\_MASK} \mid x) & \text{if } P < 0.5 \text{ (MASK)} \end{cases}$$

---

## Experimental Results

> [!NOTE]
> Metrics below reflect model evaluation following training execution on the prepared dataset.

- **Accuracy** : *Generated after training (`python main.py evaluate`)*
- **Precision**: *Generated after evaluation*
- **Recall**   : *Generated after evaluation*
- **F1 Score** : *Generated after evaluation*

Detailed metrics will be written to `outputs/reports/classification_report.txt` and `outputs/reports/confusion_matrix.png`.

---

## Security & Privacy

1. **Local Execution**: All frame captures and predictions run strictly in local memory. Video data is never transmitted to external APIs or remote servers.
2. **Controlled Storage**: Screenshots are stored strictly on the local filesystem (`outputs/screenshots/`) only when non-compliance is triggered, with a strict cooldown timer to prevent accidental disk saturation.

---

## Limitations

1. **Haar Cascade Sensitivity**: The Haar Cascade detector is optimized for frontal faces. Profile faces, extreme head angles, or severe tilts may escape detection.
2. **Lighting Conditions**: Severe underexposure or harsh backlighting can degrade face localization.
3. **Partial Facial Occlusion**: Wearing heavy sunglasses or scarfs that obscure the eyes might affect bounding box extraction.
4. **Demonstration Scope**: This software is designed for educational and compliance monitoring assistance and is not certified as a safety-critical life-support system.

---

## License

This project is licensed under the MIT License - see the [LICENSE](file:///C:/Users/PC/Desktop/FACE_MASK_DETECTION/LICENSE) file for details.

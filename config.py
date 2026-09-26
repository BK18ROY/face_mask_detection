"""Configuration module for Face Mask Detection System.

Centralizes all paths, hyperparameters, model constants, and runtime settings.
Uses pathlib for platform-independent path handling.
"""

from pathlib import Path
import cv2

# Project root directory (absolute path to project root)
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_WITH_MASK_DIR = RAW_DATA_DIR / "with_mask"
RAW_WITHOUT_MASK_DIR = RAW_DATA_DIR / "without_mask"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_DIR = PROCESSED_DATA_DIR / "train"
VAL_DATA_DIR = PROCESSED_DATA_DIR / "validation"
TEST_DATA_DIR = PROCESSED_DATA_DIR / "test"

# Model directory and artifact paths
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "face_mask_detector.keras"
BEST_MODEL_PATH = MODEL_DIR / "face_mask_detector_best.keras"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
VIDEOS_DIR = OUTPUT_DIR / "videos"

# Training plots and evaluation report paths
TRAINING_PLOT_PATH = PLOTS_DIR / "training_history.png"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

# Assets directory
ASSETS_DIR = PROJECT_ROOT / "assets"

# Image and Preprocessing Hyperparameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
CHANNELS = 3

# Data splitting ratio
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
RANDOM_SEED = 42

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

# Inference and Detection settings
# Label definition: 0 = WITH_MASK, 1 = WITHOUT_MASK
LABEL_WITH_MASK = 0
LABEL_WITHOUT_MASK = 1

LABEL_NAMES = {
    LABEL_WITH_MASK: "MASK",
    LABEL_WITHOUT_MASK: "NO MASK",
}

# BGR Colors for OpenCV visualization
COLOR_WITH_MASK = (0, 255, 0)      # Green in BGR
COLOR_WITHOUT_MASK = (0, 0, 255)   # Red in BGR

# Sigmoid probability threshold for class WITHOUT_MASK
# If prob >= MASK_THRESHOLD, classified as NO MASK (1), else MASK (0)
MASK_THRESHOLD = 0.5

# Surveillance & Alert settings
SCREENSHOT_COOLDOWN = 5.0  # Minimum seconds between violation screenshots
ENABLE_AUDIO_ALERT = True  # Optional sound alert if OS supports it
WEBCAM_INDEX = 0           # Default camera device index

# Haar Cascade Face Detector Path
# Default to OpenCV's built-in cascade file
DEFAULT_HAAR_FILENAME = "haarcascade_frontalface_default.xml"
OPENCV_HAAR_PATH = Path(cv2.data.haarcascades) / DEFAULT_HAAR_FILENAME
LOCAL_HAAR_PATH = ASSETS_DIR / DEFAULT_HAAR_FILENAME


def get_haar_cascade_path() -> Path:
    """Return an available Haar cascade XML path.

    Checks OpenCV's built-in cascade location first, then local assets directory.

    Returns:
        Path: Path to haarcascade_frontalface_default.xml

    Raises:
        FileNotFoundError: If no Haar cascade XML file is found.
    """
    if OPENCV_HAAR_PATH.is_file():
        return OPENCV_HAAR_PATH
    if LOCAL_HAAR_PATH.is_file():
        return LOCAL_HAAR_PATH
    raise FileNotFoundError(
        f"Haar cascade XML file '{DEFAULT_HAAR_FILENAME}' could not be located in:\n"
        f"  1. OpenCV data path: {OPENCV_HAAR_PATH}\n"
        f"  2. Local assets path: {LOCAL_HAAR_PATH}\n"
        "Please install opencv-python or place the XML file in the assets/ directory."
    )


def ensure_directories() -> None:
    """Create all required directory structures if they do not exist."""
    directories = [
        RAW_WITH_MASK_DIR,
        RAW_WITHOUT_MASK_DIR,
        TRAIN_DATA_DIR / "with_mask",
        TRAIN_DATA_DIR / "without_mask",
        VAL_DATA_DIR / "with_mask",
        VAL_DATA_DIR / "without_mask",
        TEST_DATA_DIR / "with_mask",
        TEST_DATA_DIR / "without_mask",
        MODEL_DIR,
        SCREENSHOTS_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
        VIDEOS_DIR,
        ASSETS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

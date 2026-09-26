# Project Assets and Cascades

This directory holds auxiliary assets, model definition files, Haar cascades, and sample test media.

---

## 1. Haar Cascade Classifier

- **File**: `haarcascade_frontalface_default.xml`
- **Purpose**: Pre-trained OpenCV Haar Cascade model used for real-time frontal face detection and bounding box localization.
- **Source**: Official OpenCV repository ([opencv/opencv data/haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)).
- **Resolution Strategy**: The system first attempts to load the cascade from the installed `opencv-python` package data path. If unavailable in the system site-packages, it automatically falls back to this local file.

---

## 2. Sample Inputs

To test video-based inference with pre-recorded files:
1. Place standard video files (`.mp4`, `.avi`, `.mov`) in this folder or in `data/`.
2. Example execution:
   ```bash
   python main.py video --input assets/sample_input.mp4
   ```

---

## 3. Sample Outputs

The system saves surveillance events and artifacts into the `outputs/` directory:
- **Screenshots**: Stored in `outputs/screenshots/no_mask_YYYY-MM-DD_HH-MM-SS.jpg` whenever an unmasked face is detected (throttled by cooldown).
- **Recorded Video**: Stored in `outputs/videos/recorded_YYYY-MM-DD_HH-MM-SS.avi` when running with `--record`.
- **Learning Plots**: Stored in `outputs/plots/training_history.png`.
- **Evaluation Reports**: Stored in `outputs/reports/classification_report.txt` and `outputs/reports/confusion_matrix.png`.

"""Real-time Face Mask Detection inference on Webcam or Video Stream.

Performs face detection via Haar Cascade, crops and normalizes detected faces,
runs CNN prediction, overlays bounding boxes with confidence scores, maintains
real-time compliance statistics, generates visual warnings, and captures
violation screenshots with a cooldown timer.
"""

from datetime import datetime
from pathlib import Path
import time
from typing import Optional, Union
import cv2
import numpy as np

import config
from src.data.dataset import load_and_preprocess_image
from src.models.cnn_model import load_trained_model, predict_mask
from src.utils.logger import get_logger
from src.utils.visualization import (
    draw_compliance_banner,
    draw_detection_overlay,
    draw_statistics,
)

logger = get_logger(__name__)


def play_optional_alert() -> None:
    """Trigger an audio beep alert on supported platforms without blocking or crashing."""
    try:
        import winsound  # Available on Windows
        winsound.Beep(1200, 150)
    except Exception:
        # Cross-platform fallback: terminal bell
        print("\a", end="", flush=True)


def get_face_detector() -> cv2.CascadeClassifier:
    """Load and validate the OpenCV Haar Cascade face detector.

    Returns:
        cv2.CascadeClassifier: Initialized face detector.

    Raises:
        FileNotFoundError: If Haar cascade XML cannot be loaded.
    """
    cascade_path = config.get_haar_cascade_path()
    logger.info("Loading Haar Cascade face detector from: %s", cascade_path)
    detector = cv2.CascadeClassifier(str(cascade_path))

    if detector.empty():
        raise FileNotFoundError(
            f"Failed to initialize Haar Cascade from: {cascade_path}\n"
            "Please verify OpenCV installation or download 'haarcascade_frontalface_default.xml' "
            "into the assets/ directory."
        )
    return detector


def run_inference(
    source: Union[int, str, Path] = config.WEBCAM_INDEX,
    model_path: Optional[Path] = None,
    threshold: float = config.MASK_THRESHOLD,
    cooldown_seconds: float = config.SCREENSHOT_COOLDOWN,
    record: bool = False,
    output_video_path: Optional[Path] = None,
    headless: bool = False,
) -> None:
    """Run live inference loop on webcam or video file.

    Args:
        source: Camera device index (int) or path to video file (str or Path).
        model_path: Optional path to trained .keras model.
        threshold: Classification probability threshold for NO_MASK.
        cooldown_seconds: Minimum cooldown time between violation screenshots.
        record: Whether to record annotated frames to a video file.
        output_video_path: Optional destination video file path.
        headless: If True, do not call cv2.imshow (useful for automated testing/servers).

    Raises:
        FileNotFoundError: If video file or model file is missing.
        RuntimeError: If camera stream cannot be initialized.
    """
    logger.info("=== Initializing Face Mask Detection Inference ===")
    config.ensure_directories()

    # Determine model file to load (prefer best model, fallback to standard)
    if model_path is None:
        if config.BEST_MODEL_PATH.exists():
            target_model_path = config.BEST_MODEL_PATH
        elif config.MODEL_PATH.exists():
            target_model_path = config.MODEL_PATH
        else:
            raise FileNotFoundError(
                f"No trained model found at {config.BEST_MODEL_PATH} or {config.MODEL_PATH}.\n"
                "Please train the model first by executing: python main.py train"
            )
    else:
        target_model_path = Path(model_path)
        if not target_model_path.exists():
            raise FileNotFoundError(f"Specified model path does not exist: {target_model_path}")

    # Load model and face detector ONCE
    model = load_trained_model(target_model_path)
    detector = get_face_detector()

    # Open video capture device or file
    is_webcam = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
    if is_webcam:
        cam_index = int(source)
        logger.info("Opening webcam at index %d...", cam_index)
        cap = cv2.VideoCapture(cam_index)
    else:
        video_file = Path(source)
        if not video_file.exists():
            raise FileNotFoundError(
                f"Video input file not found: {video_file}\n"
                "Please verify the provided file path."
            )
        logger.info("Opening video source: %s", video_file)
        cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        source_desc = f"webcam device (index {source})" if is_webcam else f"video file '{source}'"
        raise RuntimeError(
            f"Failed to open {source_desc}.\n"
            "Troubleshooting tips:\n"
            "  - For webcam: Check camera permissions, ensure no other application is using it, "
            "or test index 1 (python main.py webcam --camera-index 1).\n"
            "  - For video: Verify the file codec and path."
        )

    logger.info("Video stream opened successfully.")

    # Configure video recorder if enabled
    writer = None
    if record:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        if output_video_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_video_path = config.VIDEOS_DIR / f"recorded_{timestamp}.avi"

        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
        logger.info("Recording enabled. Saving output to: %s", output_video_path)

    last_screenshot_time = 0.0
    frame_count = 0

    window_name = "Face Mask Detection System (Press Q to Exit)"
    logger.info("Starting detection loop. Press 'q' or 'ESC' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if not is_webcam:
                    logger.info("Reached end of video stream.")
                else:
                    logger.warning("Failed to grab frame from webcam stream.")
                break

            frame_count += 1

            # Convert frame to grayscale for Haar face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect frontal faces
            detected_faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(36, 36),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            total_faces = len(detected_faces)
            masked_count = 0
            no_mask_count = 0

            # Process each detected face ROI
            for (x, y, w, h) in detected_faces:
                # Add minor padding to ROI to capture the full chin and nose area
                pad_x = int(0.05 * w)
                pad_y = int(0.05 * h)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_y)

                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                    continue

                try:
                    preprocessed_face = load_and_preprocess_image(face_roi, target_size=config.IMG_SIZE)
                    label, confidence, is_mask = predict_mask(model, preprocessed_face, threshold=threshold)

                    if is_mask:
                        masked_count += 1
                    else:
                        no_mask_count += 1

                    draw_detection_overlay(frame, (x, y, w, h), label, confidence, is_mask)
                except Exception as err:
                    logger.debug("Error processing face ROI: %s", err)
                    continue

            # Overlay compliance dashboard HUD
            draw_statistics(frame, total_faces, masked_count, no_mask_count)

            # Warning banner and screenshot capture on violation
            if no_mask_count > 0:
                draw_compliance_banner(frame, "WARNING: MASK REQUIRED")

                current_time = time.time()
                if current_time - last_screenshot_time >= cooldown_seconds:
                    last_screenshot_time = current_time
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    screenshot_file = config.SCREENSHOTS_DIR / f"no_mask_{timestamp}.jpg"
                    cv2.imwrite(str(screenshot_file), frame)
                    logger.warning("Violation detected! Screenshot captured: %s", screenshot_file)

                    if config.ENABLE_AUDIO_ALERT:
                        play_optional_alert()

            # Record frame if enabled
            if writer is not None:
                writer.write(frame)

            # Display window unless running in headless mode
            if not headless:
                try:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                        logger.info("Exit key pressed by user. Terminating inference.")
                        break
                except cv2.error as e:
                    logger.warning("GUI display not available (%s). Switching to headless mode.", e)
                    headless = True

    finally:
        # Clean up all resources
        cap.release()
        logger.info("Video capture device released.")

        if writer is not None:
            writer.release()
            logger.info("Video recording finalized.")

        if not headless:
            cv2.destroyAllWindows()
            logger.info("OpenCV display windows closed.")

        logger.info("Inference session completed. Total frames processed: %d", frame_count)

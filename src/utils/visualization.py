"""Visualization utilities for Face Mask Detection System.

Provides reusable routines for:
- Drawing bounding boxes, labels, and confidence overlays on OpenCV frames.
- Drawing compliance statistics overlays (Faces, Masked, No Mask).
- Drawing prominent warning alerts when violations are detected.
- Generating training history accuracy/loss plots.
- Generating confusion matrix visualization plots.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

import config


def draw_detection_overlay(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    is_mask: bool,
) -> np.ndarray:
    """Draw a bounding box with class label and confidence score on a frame.

    Args:
        frame: OpenCV image frame (BGR).
        bbox: Bounding box as (x, y, w, h).
        label: Prediction label (e.g. 'MASK' or 'NO MASK').
        confidence: Prediction confidence score between 0.0 and 1.0.
        is_mask: Boolean indicating if face is wearing a mask.

    Returns:
        np.ndarray: Modified frame with drawn bounding box and label.
    """
    x, y, w, h = bbox
    color = config.COLOR_WITH_MASK if is_mask else config.COLOR_WITHOUT_MASK
    thickness = 2

    # Draw face bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    # Format text: e.g. "MASK 98.45%" or "NO MASK 91.20%"
    display_text = f"{label} {confidence * 100:.1f}%"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    text_thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, text_thickness)
    label_y = max(y - 10, text_h + 10)

    # Label background rectangle for high contrast readability
    cv2.rectangle(
        frame,
        (x, label_y - text_h - 4),
        (x + text_w + 8, label_y + baseline),
        color,
        cv2.FILLED,
    )

    # Text in white (or black for bright backgrounds)
    text_color = (255, 255, 255)
    cv2.putText(
        frame,
        display_text,
        (x + 4, label_y - 2),
        font,
        font_scale,
        text_color,
        text_thickness,
        lineType=cv2.LINE_AA,
    )

    return frame


def draw_statistics(
    frame: np.ndarray,
    total_faces: int,
    masked_count: int,
    no_mask_count: int,
) -> np.ndarray:
    """Draw a semi-transparent HUD statistics panel showing face counts.

    Args:
        frame: OpenCV image frame (BGR).
        total_faces: Total number of faces detected in the current frame.
        masked_count: Number of masked faces.
        no_mask_count: Number of unmasked faces.

    Returns:
        np.ndarray: Modified frame with HUD statistics overlay.
    """
    h, w, _ = frame.shape
    panel_x1, panel_y1 = 10, 10
    panel_x2, panel_y2 = 230, 115

    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (20, 20, 20),
        cv2.FILLED,
    )
    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (80, 80, 80),
        1,
    )

    alpha = 0.70
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thickness = 1

    # Text lines
    stats = [
        (f"Faces: {total_faces}", (255, 255, 255)),
        (f"Masked: {masked_count}", (0, 255, 0)),
        (f"No Mask: {no_mask_count}", (0, 0, 255) if no_mask_count > 0 else (200, 200, 200)),
        ("Press Q to quit", (180, 180, 180)),
    ]

    start_y = panel_y1 + 22
    for text, text_color in stats:
        cv2.putText(
            frame,
            text,
            (panel_x1 + 12, start_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )
        start_y += 22

    return frame


def draw_compliance_banner(frame: np.ndarray, warning_text: str = "WARNING: MASK REQUIRED") -> np.ndarray:
    """Draw a bold flashing warning banner at the top/bottom when violations occur.

    Args:
        frame: OpenCV image frame (BGR).
        warning_text: Text message to display.

    Returns:
        np.ndarray: Frame with warning banner rendered.
    """
    h, w, _ = frame.shape
    banner_height = 42

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 200), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.75
    thickness = 2
    (tw, th), _ = cv2.getTextSize(warning_text, font, font_scale, thickness)
    tx = max(10, (w - tw) // 2)
    ty = (banner_height + th) // 2

    cv2.putText(
        frame,
        warning_text,
        (tx, ty),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return frame


def plot_training_history(history: Dict[str, List[float]], output_path: Path) -> None:
    """Plot training and validation accuracy and loss curves and save to disk.

    Args:
        history: Dictionary containing 'accuracy', 'val_accuracy', 'loss', 'val_loss'.
        output_path: Path where PNG plot will be saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs_range = range(1, len(history.get("accuracy", [])) + 1)

    plt.figure(figsize=(12, 5))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    if "accuracy" in history:
        plt.plot(epochs_range, history["accuracy"], label="Training Accuracy", color="#1f77b4", linewidth=2)
    if "val_accuracy" in history:
        plt.plot(epochs_range, history["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", linewidth=2, linestyle="--")
    plt.title("Model Accuracy over Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)

    # Plot Loss
    plt.subplot(1, 2, 2)
    if "loss" in history:
        plt.plot(epochs_range, history["loss"], label="Training Loss", color="#d62728", linewidth=2)
    if "val_loss" in history:
        plt.plot(epochs_range, history["val_loss"], label="Validation Loss", color="#2ca02c", linewidth=2, linestyle="--")
    plt.title("Model Loss over Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Crossentropy Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Plot confusion matrix with counts and percentages and save to disk.

    Args:
        cm: 2x2 confusion matrix numpy array.
        class_names: List of class names (e.g. ['MASK', 'NO MASK']).
        output_path: Path where PNG plot will be saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13,
                fontweight="bold",
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

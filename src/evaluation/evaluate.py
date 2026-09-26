"""Evaluation module for Face Mask Detection CNN.

Evaluates the trained model against the independent test dataset, computing
classification accuracy, precision, recall, F1-score, confusion matrix,
and a comprehensive classification report.
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config
from src.data.dataset import create_dataset_from_directory
from src.models.cnn_model import load_trained_model
from src.utils.logger import get_logger
from src.utils.visualization import plot_confusion_matrix

logger = get_logger(__name__)


def run_evaluation(
    model_path: Path = config.BEST_MODEL_PATH,
    test_dir: Path = config.TEST_DATA_DIR,
    threshold: float = config.MASK_THRESHOLD,
    report_save_path: Path = config.CLASSIFICATION_REPORT_PATH,
    cm_save_path: Path = config.CONFUSION_MATRIX_PATH,
) -> Dict[str, float]:
    """Evaluate trained model on the test dataset.

    Args:
        model_path: Path to trained .keras model (defaults to best checkpoint).
        test_dir: Path to test dataset split directory.
        threshold: Decision boundary probability threshold for NO_MASK class.
        report_save_path: Path to save classification report text file.
        cm_save_path: Path to save confusion matrix visualization PNG.

    Returns:
        Dict[str, float]: Computed evaluation metrics.

    Raises:
        FileNotFoundError: If model file or test directory does not exist.
    """
    logger.info("=== Starting Model Evaluation ===")

    # Fall back to standard model path if best model checkpoint is missing
    target_model_path = model_path
    if not target_model_path.exists():
        if config.MODEL_PATH.exists():
            logger.info("Best model checkpoint not found at %s. Falling back to %s", model_path, config.MODEL_PATH)
            target_model_path = config.MODEL_PATH
        else:
            raise FileNotFoundError(
                f"No trained model found at {model_path} or {config.MODEL_PATH}.\n"
                "Please train the model first by executing: python main.py train"
            )

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test dataset directory not found at {test_dir}.\n"
            "Please run: python main.py prepare-data"
        )

    # Load model
    model = load_trained_model(target_model_path)

    # Load test dataset without shuffling
    test_ds = create_dataset_from_directory(
        test_dir,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        augment=False,
    )

    y_true_list = []
    y_pred_probs_list = []

    logger.info("Running inference across test set batches...")
    for images, labels in test_ds:
        probs = model(images, training=False).numpy()
        y_pred_probs_list.extend(probs.flatten())
        y_true_list.extend(labels.numpy().flatten())

    y_true = np.array(y_true_list, dtype=int)
    y_pred_probs = np.array(y_pred_probs_list, dtype=float)
    y_pred = (y_pred_probs >= threshold).astype(int)

    if len(y_true) == 0:
        raise ValueError("Test dataset contains 0 samples. Cannot evaluate model.")

    # Calculate metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))

    class_names = [config.LABEL_NAMES[0], config.LABEL_NAMES[1]]  # ["MASK", "NO MASK"]
    report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Print results to terminal in the requested format
    terminal_output = (
        "\n"
        "=============================\n"
        "      MODEL EVALUATION       \n"
        "=============================\n"
        f"Accuracy : {accuracy * 100:.2f}%\n"
        f"Precision: {precision * 100:.2f}%\n"
        f"Recall   : {recall * 100:.2f}%\n"
        f"F1 Score : {f1 * 100:.2f}%\n"
        "=============================\n\n"
        "Detailed Classification Report:\n"
        f"{report_str}\n"
    )
    print(terminal_output)

    # Save classification report to disk
    report_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write(terminal_output)
    logger.info("Classification report saved to %s", report_save_path)

    # Save confusion matrix plot to disk
    plot_confusion_matrix(cm, class_names, cm_save_path)
    logger.info("Confusion matrix plot saved to %s", cm_save_path)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

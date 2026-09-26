"""Custom Convolutional Neural Network (CNN) architecture for Face Mask Detection.

Defines a multi-block deep CNN with Batch Normalization, ReLU activations,
Dropout regularization, and a Sigmoid classification head for binary prediction.
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_cnn_model(
    input_shape: Tuple[int, int, int] = (config.IMG_WIDTH, config.IMG_HEIGHT, config.CHANNELS),
    learning_rate: float = config.LEARNING_RATE,
) -> keras.Model:
    """Build and compile a custom multi-block CNN for face mask detection.

    Architecture:
      Input (128x128x3)
      Block 1: Conv2D(32) -> BatchNormalization -> ReLU -> Conv2D(32) -> ReLU -> MaxPooling2D
      Block 2: Conv2D(64) -> BatchNormalization -> ReLU -> Conv2D(64) -> ReLU -> MaxPooling2D
      Block 3: Conv2D(128) -> BatchNormalization -> ReLU -> Conv2D(128) -> ReLU -> MaxPooling2D
      Block 4: Conv2D(256) -> BatchNormalization -> ReLU -> GlobalAveragePooling2D
      Head: Dense(128, ReLU) -> BatchNormalization -> Dropout(0.5) -> Dense(1, Sigmoid)

    The output sigmoid neuron outputs the probability of class 1 (WITHOUT_MASK).

    Args:
        input_shape: Shape tuple of input images (H, W, C).
        learning_rate: Optimizer learning rate.

    Returns:
        keras.Model: Compiled TensorFlow/Keras model.
    """
    inputs = keras.Input(shape=input_shape, name="face_input")

    # Convolutional Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Convolutional Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Convolutional Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Convolutional Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Classification Dense Head
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="mask_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="custom_face_mask_cnn")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    logger.info("Custom Face Mask CNN successfully built and compiled.")
    return model


def save_model(model: keras.Model, filepath: Path) -> None:
    """Save model weights and architecture to disk in native .keras format.

    Args:
        model: Trained Keras model instance.
        filepath: Destination path (e.g. models/face_mask_detector.keras).
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(filepath))
    logger.info("Model saved to %s", filepath)


def load_trained_model(filepath: Path) -> keras.Model:
    """Load a saved Keras model from disk.

    Args:
        filepath: Path to the .keras model file.

    Returns:
        keras.Model: Loaded Keras model.

    Raises:
        FileNotFoundError: If the model file is not found.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Trained model not found at {filepath}.\n"
            "Please train the model first by executing: python main.py train"
        )

    logger.info("Loading model from %s", filepath)
    model = keras.models.load_model(str(filepath))
    logger.info("Model loaded successfully.")
    return model


def predict_mask(
    model: keras.Model,
    preprocessed_face: np.ndarray,
    threshold: float = config.MASK_THRESHOLD,
) -> Tuple[str, float, bool]:
    """Perform single-face inference to classify MASK vs NO MASK.

    Calculates the true confidence corresponding to the predicted class.
    Sigmoid output:
      - 0.0 -> WITH_MASK
      - 1.0 -> WITHOUT_MASK

    If raw probability >= threshold:
      Predicted: NO MASK
      Confidence: raw probability
      is_mask: False
    Else:
      Predicted: MASK
      Confidence: 1.0 - raw probability
      is_mask: True

    Args:
        model: Compiled or loaded Keras model.
        preprocessed_face: Normalized face image of shape (128, 128, 3) or (1, 128, 128, 3).
        threshold: Decision boundary probability threshold (default 0.5).

    Returns:
        Tuple[str, float, bool]: (label_string, confidence_score, is_mask_bool)
    """
    # Ensure 4D batch dimension: (1, 128, 128, 3)
    if len(preprocessed_face.shape) == 3:
        input_tensor = np.expand_dims(preprocessed_face, axis=0)
    elif len(preprocessed_face.shape) == 4:
        input_tensor = preprocessed_face
    else:
        raise ValueError(f"Invalid input shape for face tensor: {preprocessed_face.shape}")

    # Run inference without verbose progress bars
    raw_pred = model(input_tensor, training=False).numpy()
    no_mask_prob = float(raw_pred[0][0])

    if no_mask_prob >= threshold:
        label = "NO MASK"
        confidence = no_mask_prob
        is_mask = False
    else:
        label = "MASK"
        confidence = 1.0 - no_mask_prob
        is_mask = True

    return label, confidence, is_mask

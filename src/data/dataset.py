"""Dataset loading, image preprocessing, and augmentation pipelines.

Provides functions to preprocess individual face crops for real-time inference,
as well as tf.data.Dataset loaders with data augmentation for CNN training.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_augmentation_layer() -> keras.Sequential:
    """Build a Keras data augmentation pipeline with realistic transformations.

    Includes horizontal flipping, subtle rotation, slight zoom, and minor translation.

    Returns:
        keras.Sequential: Sequential layer containing augmentation steps.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=config.RANDOM_SEED),
            layers.RandomRotation(0.08, seed=config.RANDOM_SEED),
            layers.RandomZoom(0.08, seed=config.RANDOM_SEED),
            layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05,
                seed=config.RANDOM_SEED,
            ),
        ],
        name="data_augmentation",
    )


def load_and_preprocess_image(
    image_input: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = config.IMG_SIZE,
) -> np.ndarray:
    """Load, convert to RGB, resize, and normalize an image to [0, 1].

    Accepts either a file path (str or Path) or an in-memory BGR numpy array
    (such as a face ROI cropped from an OpenCV video frame).

    Args:
        image_input: Path to image file, or OpenCV BGR image array.
        target_size: Target (width, height) tuple, defaults to (128, 128).

    Returns:
        np.ndarray: Preprocessed float32 image array with shape (H, W, 3) in [0.0, 1.0].

    Raises:
        ValueError: If image cannot be read or array is invalid.
    """
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")
        # Read with OpenCV
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise ValueError(f"Failed to read image at {img_path}. File may be corrupted.")
    elif isinstance(image_input, np.ndarray):
        bgr = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    if bgr.size == 0 or len(bgr.shape) < 2:
        raise ValueError("Provided image array is empty or has invalid dimensions.")

    # Convert BGR (OpenCV default) or Grayscale to RGB
    if len(bgr.shape) == 2:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    elif bgr.shape[2] == 4:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
    elif bgr.shape[2] == 3:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unexpected number of channels: {bgr.shape[2]}")

    # Resize to target dimensions (width, height)
    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)

    # Normalize pixel values to [0.0, 1.0]
    normalized = resized.astype(np.float32) / 255.0

    return normalized


def create_dataset_from_directory(
    directory_path: Path,
    batch_size: int = config.BATCH_SIZE,
    target_size: Tuple[int, int] = config.IMG_SIZE,
    shuffle: bool = True,
    augment: bool = False,
    seed: int = config.RANDOM_SEED,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from a directory structured with class subfolders.

    Directory must contain 'with_mask' and 'without_mask' subdirectories.
    Labels are mapped deterministically:
        - 'with_mask': 0
        - 'without_mask': 1

    Args:
        directory_path: Path to dataset split folder.
        batch_size: Mini-batch size.
        target_size: (width, height) tuple for images.
        shuffle: Whether to shuffle data.
        augment: Whether to apply training data augmentation.
        seed: Random seed for shuffling.

    Returns:
        tf.data.Dataset: Configured TensorFlow dataset.

    Raises:
        FileNotFoundError: If directory does not exist or has missing subdirectories.
    """
    if not directory_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {directory_path}")

    class_names = ["with_mask", "without_mask"]
    for cname in class_names:
        sub = directory_path / cname
        if not sub.exists():
            raise FileNotFoundError(
                f"Missing class folder '{cname}' in {directory_path}. "
                "Ensure data preparation has been run."
            )

    # Height, width expected by keras image_dataset_from_directory
    image_shape = (target_size[1], target_size[0])

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=str(directory_path),
        labels="inferred",
        label_mode="binary",
        class_names=class_names,  # index 0: with_mask, index 1: without_mask
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_shape,
        shuffle=shuffle,
        seed=seed,
    )

    # Normalize images from [0, 255] to [0.0, 1.0]
    rescale_layer = layers.Rescaling(1.0 / 255.0)
    dataset = dataset.map(lambda x, y: (rescale_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Apply data augmentation if requested (training split only)
    if augment:
        augmentation_layer = build_augmentation_layer()
        dataset = dataset.map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Optimize pipeline with prefetching
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_dataset_generators(
    train_dir: Path = config.TRAIN_DATA_DIR,
    val_dir: Path = config.VAL_DATA_DIR,
    test_dir: Optional[Path] = config.TEST_DATA_DIR,
    batch_size: int = config.BATCH_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset]]:
    """Convenience helper to retrieve train, validation, and optional test datasets.

    Args:
        train_dir: Path to training data.
        val_dir: Path to validation data.
        test_dir: Path to testing data.
        batch_size: Batch size.

    Returns:
        Tuple: (train_ds, val_ds, test_ds)
    """
    logger.info("Loading training dataset from %s", train_dir)
    train_ds = create_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
    )

    logger.info("Loading validation dataset from %s", val_dir)
    val_ds = create_dataset_from_directory(
        val_dir,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )

    test_ds = None
    if test_dir and test_dir.exists():
        logger.info("Loading test dataset from %s", test_dir)
        test_ds = create_dataset_from_directory(
            test_dir,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
        )

    return train_ds, val_ds, test_ds

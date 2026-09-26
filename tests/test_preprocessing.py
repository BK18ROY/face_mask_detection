"""Unit tests for dataset preprocessing and image pipeline."""

from pathlib import Path
import numpy as np
import pytest

import config
from src.data.dataset import load_and_preprocess_image


def test_preprocessing_from_bgr_array():
    """Test preprocessing a synthetic BGR image array."""
    # Arbitrary input dimensions (e.g. 250 x 300 x 3)
    raw_bgr = np.random.randint(0, 256, size=(250, 300, 3), dtype=np.uint8)

    processed = load_and_preprocess_image(raw_bgr, target_size=(128, 128))

    # Assert shape is (128, 128, 3)
    assert processed.shape == (128, 128, 3)
    # Assert normalized to [0.0, 1.0]
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_preprocessing_grayscale_array():
    """Test preprocessing a 2D grayscale array converts to 3-channel RGB."""
    raw_gray = np.random.randint(0, 256, size=(200, 200), dtype=np.uint8)

    processed = load_and_preprocess_image(raw_gray, target_size=(128, 128))

    assert processed.shape == (128, 128, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_preprocessing_custom_target_size():
    """Test resizing to custom dimensions."""
    raw_bgr = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    custom_size = (64, 64)
    processed = load_and_preprocess_image(raw_bgr, target_size=custom_size)

    assert processed.shape == (64, 64, 3)


def test_preprocessing_empty_array_raises_error():
    """Test that passing an empty array raises ValueError."""
    empty_array = np.array([])
    with pytest.raises(ValueError):
        load_and_preprocess_image(empty_array)


def test_preprocessing_nonexistent_file_raises_error():
    """Test that passing a non-existent file path raises FileNotFoundError."""
    missing_file = Path("non_existent_image_path_12345.jpg")
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_image(missing_file)


def test_batch_dimension_expansion():
    """Test that preprocessed image can be expanded to 4D batch tensor."""
    raw_bgr = np.random.randint(0, 256, size=(150, 150, 3), dtype=np.uint8)
    processed = load_and_preprocess_image(raw_bgr, target_size=(128, 128))

    batch = np.expand_dims(processed, axis=0)
    assert batch.shape == (1, 128, 128, 3)

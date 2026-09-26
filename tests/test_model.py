"""Unit tests for Face Mask CNN model architecture and inference."""

import numpy as np
import pytest
import tensorflow as tf

import config
from src.models.cnn_model import build_cnn_model, predict_mask


@pytest.fixture(scope="module")
def initialized_model():
    """Fixture providing a compiled CNN model instance."""
    return build_cnn_model(input_shape=(128, 128, 3), learning_rate=1e-4)


def test_model_creation(initialized_model):
    """Test that the CNN model can be initialized without error."""
    assert initialized_model is not None
    assert isinstance(initialized_model, tf.keras.Model)
    assert initialized_model.name == "custom_face_mask_cnn"


def test_model_output_shape(initialized_model):
    """Test that the CNN model produces expected output tensor shape (batch, 1)."""
    assert initialized_model.input_shape == (None, 128, 128, 3)
    assert initialized_model.output_shape == (None, 1)


def test_model_inference_batch(initialized_model):
    """Test that the model performs forward pass on a batch of synthetic images."""
    batch_size = 4
    dummy_input = np.random.uniform(0.0, 1.0, size=(batch_size, 128, 128, 3)).astype(np.float32)

    predictions = initialized_model(dummy_input, training=False).numpy()
    assert predictions.shape == (batch_size, 1)
    # Binary sigmoid outputs must be strictly in [0.0, 1.0]
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)


def test_predict_mask_confidence_masked(initialized_model):
    """Test predict_mask confidence and label calculation when prob < threshold."""
    dummy_face = np.random.uniform(0.0, 1.0, size=(128, 128, 3)).astype(np.float32)

    # Mock model call returning 0.1 (low prob of NO_MASK -> MASK)
    class MockModelMask:
        def __call__(self, tensor, training=False):
            return tf.constant([[0.1]], dtype=tf.float32)

    label, confidence, is_mask = predict_mask(MockModelMask(), dummy_face, threshold=0.5)

    assert label == "MASK"
    assert is_mask is True
    assert pytest.approx(confidence, 0.01) == 0.90  # 1.0 - 0.1 = 0.9


def test_predict_mask_confidence_unmasked(initialized_model):
    """Test predict_mask confidence and label calculation when prob >= threshold."""
    dummy_face = np.random.uniform(0.0, 1.0, size=(128, 128, 3)).astype(np.float32)

    # Mock model call returning 0.85 (high prob of NO_MASK -> NO MASK)
    class MockModelNoMask:
        def __call__(self, tensor, training=False):
            return tf.constant([[0.85]], dtype=tf.float32)

    label, confidence, is_mask = predict_mask(MockModelNoMask(), dummy_face, threshold=0.5)

    assert label == "NO MASK"
    assert is_mask is False
    assert pytest.approx(confidence, 0.01) == 0.85

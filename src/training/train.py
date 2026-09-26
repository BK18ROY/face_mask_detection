"""Training pipeline for Face Mask Detection CNN.

Handles dataset loading, model compilation, training with callbacks
(ModelCheckpoint, EarlyStopping, ReduceLROnPlateau), model serialization,
and learning curve visualization.
"""

from pathlib import Path
from typing import Optional
import tensorflow as tf
from tensorflow import keras

import config
from src.data.dataset import create_dataset_from_directory
from src.data.prepare_dataset import prepare_and_split_dataset
from src.models.cnn_model import build_cnn_model, save_model
from src.utils.logger import get_logger
from src.utils.visualization import plot_training_history

logger = get_logger(__name__)


def check_and_prepare_data_if_needed() -> None:
    """Check if processed train and validation sets exist, else run data preparation."""
    train_wm = config.TRAIN_DATA_DIR / "with_mask"
    train_wom = config.TRAIN_DATA_DIR / "without_mask"
    val_wm = config.VAL_DATA_DIR / "with_mask"
    val_wom = config.VAL_DATA_DIR / "without_mask"

    has_data = (
        train_wm.exists()
        and any(train_wm.iterdir())
        and train_wom.exists()
        and any(train_wom.iterdir())
        and val_wm.exists()
        and any(val_wm.iterdir())
        and val_wom.exists()
        and any(val_wom.iterdir())
    )

    if not has_data:
        logger.info("Processed dataset not detected. Attempting to prepare dataset from raw data...")
        prepare_and_split_dataset(allow_synthetic=False)


def run_training(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    train_dir: Path = config.TRAIN_DATA_DIR,
    val_dir: Path = config.VAL_DATA_DIR,
    model_save_path: Path = config.MODEL_PATH,
    best_model_save_path: Path = config.BEST_MODEL_PATH,
    plot_save_path: Path = config.TRAINING_PLOT_PATH,
) -> keras.callbacks.History:
    """Execute the end-to-end model training workflow.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Initial learning rate for Adam optimizer.
        train_dir: Path to processed train folder.
        val_dir: Path to processed validation folder.
        model_save_path: Destination path for final model.
        best_model_save_path: Destination path for best model checkpoint.
        plot_save_path: Destination path for training curve plot.

    Returns:
        keras.callbacks.History: Training history object.
    """
    logger.info("=== Starting Face Mask Detection Model Training ===")
    config.ensure_directories()

    # Verify or prepare data
    check_and_prepare_data_if_needed()

    # Load datasets
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

    # Build CNN architecture
    model = build_cnn_model(learning_rate=learning_rate)
    model.summary(print_fn=lambda x: logger.info("%s", x))

    # Configure training callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_save_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    logger.info("Initiating model training for up to %d epochs...", epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )

    # Save final model
    save_model(model, model_save_path)
    logger.info("Final model saved to %s", model_save_path)

    # Ensure best model is also present if early stopping restored best weights
    if not best_model_save_path.exists():
        save_model(model, best_model_save_path)
        logger.info("Best model saved to %s", best_model_save_path)

    # Plot and save learning curves
    logger.info("Plotting training history curves...")
    plot_training_history(history.history, plot_save_path)
    logger.info("Training history plot saved to %s", plot_save_path)

    logger.info("=== Model Training Completed Successfully ===")
    return history

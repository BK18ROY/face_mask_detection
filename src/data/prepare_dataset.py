"""Dataset validation, cleaning, and train/val/test partitioning.

Validates raw images, filters corrupted files, ensures valid dimensions/color spaces,
and splits data into stratified train/val/test sets reproducibly.
"""

from pathlib import Path
import random
import shutil
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np

import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_image(image_path: Path) -> bool:
    """Validate whether an image file is readable and uncorrupted.

    Args:
        image_path: Path to the image file.

    Returns:
        bool: True if image is valid, False otherwise.
    """
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False

    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify file integrity

        # Reopen to verify loading pixel data and dimensions
        with Image.open(image_path) as img:
            img.load()
            w, h = img.size
            if w < 10 or h < 10:
                logger.warning("Image too small (%dx%d): %s", w, h, image_path.name)
                return False
        return True
    except Exception as err:
        logger.warning("Corrupted or unreadable image skipped: %s (Reason: %s)", image_path.name, err)
        return False


def collect_valid_images(directory: Path) -> List[Path]:
    """Scan a directory and return a list of all valid image paths.

    Args:
        directory: Directory containing raw image files.

    Returns:
        List[Path]: List of validated image paths.
    """
    if not directory.exists() or not directory.is_dir():
        logger.warning("Directory does not exist: %s", directory)
        return []

    valid_images: List[Path] = []
    for item in sorted(directory.iterdir()):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            if validate_image(item):
                valid_images.append(item)

    return valid_images


def split_image_list(
    images: List[Path],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split a list of images into train, validation, and test subsets reproducibly.

    Args:
        images: List of image paths.
        train_ratio: Proportion of training data (e.g. 0.80).
        val_ratio: Proportion of validation data (e.g. 0.10).
        seed: Random seed for deterministic shuffling.

    Returns:
        Tuple[List[Path], List[Path], List[Path]]: (train, validation, test) lists.
    """
    shuffled = list(images)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train: n_train + n_val]
    test_files = shuffled[n_train + n_val:]

    return train_files, val_files, test_files


def copy_files_to_split(files: List[Path], destination_dir: Path) -> int:
    """Copy a list of files to the target split directory.

    Args:
        files: List of file paths to copy.
        destination_dir: Target directory path.

    Returns:
        int: Number of files copied.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for file_path in files:
        target_path = destination_dir / file_path.name
        # If duplicate name from another folder, prepend unique index
        if target_path.exists():
            target_path = destination_dir / f"{copied}_{file_path.name}"
        shutil.copy2(file_path, target_path)
        copied += 1
    return copied


def create_synthetic_sample_data(num_samples_per_class: int = 40) -> None:
    """Generate synthetic sample face images with and without masks for testing.

    Used when no raw dataset is present to verify full pipeline execution.

    Args:
        num_samples_per_class: Number of synthetic images per category.
    """
    import cv2

    logger.info("Generating %d synthetic benchmark samples per category...", num_samples_per_class)
    config.RAW_WITH_MASK_DIR.mkdir(parents=True, exist_ok=True)
    config.RAW_WITHOUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.RANDOM_SEED)

    for i in range(num_samples_per_class):
        # Create base face-like oval background
        img = np.full((128, 128, 3), (210, 220, 230), dtype=np.uint8)

        # Draw face oval
        face_color = (int(rng.integers(140, 190)), int(rng.integers(170, 215)), int(rng.integers(210, 245)))
        cv2.ellipse(img, (64, 64), (42, 54), 0, 0, 360, face_color, -1)

        # Eyes
        cv2.circle(img, (48, 52), 5, (50, 40, 30), -1)
        cv2.circle(img, (80, 52), 5, (50, 40, 30), -1)

        # Mouth / Mask
        # Without mask: draw lips/mouth
        no_mask_img = img.copy()
        cv2.ellipse(no_mask_img, (64, 88), (14, 6), 0, 0, 360, (60, 60, 180), -1)
        no_mask_file = config.RAW_WITHOUT_MASK_DIR / f"sample_nomask_{i:03d}.jpg"
        cv2.imwrite(str(no_mask_file), no_mask_img)

        # With mask: draw surgical / cloth mask covering nose and mouth
        mask_img = img.copy()
        mask_color = (int(rng.integers(200, 255)), int(rng.integers(180, 240)), int(rng.integers(70, 120)))
        mask_pts = np.array([[34, 68], [94, 68], [86, 106], [42, 106]], np.int32)
        cv2.fillPoly(mask_img, [mask_pts], mask_color)
        cv2.polylines(mask_img, [mask_pts], isClosed=True, color=(100, 100, 100), thickness=1)
        mask_file = config.RAW_WITH_MASK_DIR / f"sample_mask_{i:03d}.jpg"
        cv2.imwrite(str(mask_file), mask_img)

    logger.info("Synthetic benchmark dataset successfully generated in %s", config.RAW_DATA_DIR)


def prepare_and_split_dataset(
    train_ratio: float = config.TRAIN_SPLIT,
    val_ratio: float = config.VAL_SPLIT,
    seed: int = config.RANDOM_SEED,
    allow_synthetic: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Clean raw images and split into train, validation, and test directories.

    Args:
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.
        seed: Random seed for reproducibility.
        allow_synthetic: Generate synthetic samples if raw folders are empty.

    Returns:
        Dict: Statistics breakdown per split and category.

    Raises:
        FileNotFoundError: If raw dataset is missing and allow_synthetic is False.
    """
    config.ensure_directories()

    with_mask_files = collect_valid_images(config.RAW_WITH_MASK_DIR)
    without_mask_files = collect_valid_images(config.RAW_WITHOUT_MASK_DIR)

    if not with_mask_files or not without_mask_files:
        if allow_synthetic:
            logger.info("Raw dataset empty. Generating synthetic benchmark samples...")
            create_synthetic_sample_data(num_samples_per_class=50)
            with_mask_files = collect_valid_images(config.RAW_WITH_MASK_DIR)
            without_mask_files = collect_valid_images(config.RAW_WITHOUT_MASK_DIR)
        else:
            raise FileNotFoundError(
                f"\nDataset not found or contains no valid images!\n"
                f"Checked:\n"
                f"  - With Mask:    {config.RAW_WITH_MASK_DIR} (found {len(with_mask_files)} valid)\n"
                f"  - Without Mask: {config.RAW_WITHOUT_MASK_DIR} (found {len(without_mask_files)} valid)\n\n"
                f"How to acquire the dataset:\n"
                f"  1. Download the 'Face Mask Dataset' from Kaggle or GitHub:\n"
                f"     https://www.kaggle.com/datasets/omkargurav/face-mask-dataset\n"
                f"     https://github.com/prajnasb/observations/tree/master/experiements/data\n"
                f"  2. Place images into:\n"
                f"       {config.RAW_WITH_MASK_DIR} (images wearing masks)\n"
                f"       {config.RAW_WITHOUT_MASK_DIR} (images without masks)\n"
                f"  3. Alternatively, re-run with: python main.py prepare-data --generate-samples\n"
            )

    logger.info("Raw Dataset Statistics:")
    logger.info("  With Mask:    %d images", len(with_mask_files))
    logger.info("  Without Mask: %d images", len(without_mask_files))
    logger.info("  Total:        %d images", len(with_mask_files) + len(without_mask_files))

    # Split each category reproducibly
    wm_train, wm_val, wm_test = split_image_list(with_mask_files, train_ratio, val_ratio, seed)
    wom_train, wom_val, wom_test = split_image_list(without_mask_files, train_ratio, val_ratio, seed)

    # Clean existing processed directories before writing fresh split
    splits = [
        ("train", config.TRAIN_DATA_DIR, wm_train, wom_train),
        ("validation", config.VAL_DATA_DIR, wm_val, wom_val),
        ("test", config.TEST_DATA_DIR, wm_test, wom_test),
    ]

    stats: Dict[str, Dict[str, int]] = {}

    for split_name, split_dir, wm_list, wom_list in splits:
        wm_dest = split_dir / "with_mask"
        wom_dest = split_dir / "without_mask"

        # Clear existing
        if wm_dest.exists():
            shutil.rmtree(wm_dest)
        if wom_dest.exists():
            shutil.rmtree(wom_dest)

        copied_wm = copy_files_to_split(wm_list, wm_dest)
        copied_wom = copy_files_to_split(wom_list, wom_dest)

        stats[split_name] = {
            "with_mask": copied_wm,
            "without_mask": copied_wom,
            "total": copied_wm + copied_wom,
        }

        logger.info(
            "Split '%s': %d with_mask, %d without_mask (Total: %d)",
            split_name,
            copied_wm,
            copied_wom,
            copied_wm + copied_wom,
        )

    return stats

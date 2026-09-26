"""Command Line Interface (CLI) for Face Mask Detection System.

Provides high-level commands for:
  - prepare-data : Validate, clean, and split raw dataset
  - train        : Train the custom CNN model with early stopping & checkpointing
  - evaluate     : Evaluate model performance on the test dataset
  - webcam       : Run live webcam face mask detection
  - video        : Run face mask detection on a pre-recorded video file
  - test         : Execute test suite using pytest
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional

import config
from src.utils.logger import get_logger

logger = get_logger("face_mask_cli")


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the CLI argument parser with subcommands.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="face-mask-detection",
        description="Production Face Mask Detection System powered by OpenCV & TensorFlow CNN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Optional top-level source flag for direct invocation: python main.py --source webcam
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Inference source: 'webcam' or path to video file (e.g. video.mp4).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record annotated inference frames to outputs/videos/.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. prepare-data
    prep_parser = subparsers.add_parser(
        "prepare-data",
        help="Validate raw images and partition into train/val/test splits.",
    )
    prep_parser.add_argument(
        "--generate-samples",
        "--allow-synthetic",
        dest="generate_samples",
        action="store_true",
        help="Generate synthetic benchmark sample images if raw folders are empty.",
    )
    prep_parser.add_argument(
        "--train-split",
        type=float,
        default=config.TRAIN_SPLIT,
        help=f"Fraction of data for training (default: {config.TRAIN_SPLIT}).",
    )
    prep_parser.add_argument(
        "--val-split",
        type=float,
        default=config.VAL_SPLIT,
        help=f"Fraction of data for validation (default: {config.VAL_SPLIT}).",
    )
    prep_parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {config.RANDOM_SEED}).",
    )

    # 2. train
    train_parser = subparsers.add_parser(
        "train",
        help="Train the custom CNN face mask classifier.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help=f"Number of training epochs (default: {config.EPOCHS}).",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Mini-batch size (default: {config.BATCH_SIZE}).",
    )
    train_parser.add_argument(
        "--learning-rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=config.LEARNING_RATE,
        help=f"Adam optimizer learning rate (default: {config.LEARNING_RATE}).",
    )

    # 3. evaluate
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate trained model on the test dataset and generate reports.",
    )
    eval_parser.add_argument(
        "--model",
        type=Path,
        default=config.BEST_MODEL_PATH,
        help=f"Path to model file (default: {config.BEST_MODEL_PATH}).",
    )
    eval_parser.add_argument(
        "--threshold",
        type=float,
        default=config.MASK_THRESHOLD,
        help=f"Decision boundary probability threshold (default: {config.MASK_THRESHOLD}).",
    )

    # 4. webcam
    webcam_parser = subparsers.add_parser(
        "webcam",
        help="Run real-time face mask detection on live webcam feed.",
    )
    webcam_parser.add_argument(
        "--camera-index",
        "--index",
        dest="camera_index",
        type=int,
        default=config.WEBCAM_INDEX,
        help=f"Camera device index (default: {config.WEBCAM_INDEX}).",
    )
    webcam_parser.add_argument(
        "--record",
        action="store_true",
        help="Record annotated video to outputs/videos/.",
    )
    webcam_parser.add_argument(
        "--threshold",
        type=float,
        default=config.MASK_THRESHOLD,
        help=f"Classification threshold (default: {config.MASK_THRESHOLD}).",
    )
    webcam_parser.add_argument(
        "--cooldown",
        type=float,
        default=config.SCREENSHOT_COOLDOWN,
        help=f"Violation screenshot cooldown in seconds (default: {config.SCREENSHOT_COOLDOWN}).",
    )
    webcam_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI window (useful for headless servers / CI).",
    )

    # 5. video
    video_parser = subparsers.add_parser(
        "video",
        help="Run face mask detection on a local video file.",
    )
    video_parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to input video file (e.g. data/sample.mp4).",
    )
    video_parser.add_argument(
        "--record",
        action="store_true",
        help="Record annotated video output to outputs/videos/.",
    )
    video_parser.add_argument(
        "--threshold",
        type=float,
        default=config.MASK_THRESHOLD,
        help=f"Classification threshold (default: {config.MASK_THRESHOLD}).",
    )
    video_parser.add_argument(
        "--cooldown",
        type=float,
        default=config.SCREENSHOT_COOLDOWN,
        help=f"Violation screenshot cooldown in seconds (default: {config.SCREENSHOT_COOLDOWN}).",
    )
    video_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI window (useful for headless servers / CI).",
    )

    # 6. test
    subparsers.add_parser(
        "test",
        help="Run automated test suite (pytest).",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint.

    Args:
        argv: Command-line arguments list (defaults to sys.argv[1:]).

    Returns:
        int: Exit status code (0 for success, non-zero for error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Ensure required directory tree exists
    config.ensure_directories()

    try:
        # Handle top-level --source flag if provided
        if args.source is not None:
            from src.inference.webcam import run_inference
            source_target = (
                config.WEBCAM_INDEX
                if args.source.lower() == "webcam"
                else Path(args.source)
            )
            run_inference(
                source=source_target,
                record=args.record,
            )
            return 0

        # Handle subcommand routing
        if args.command == "prepare-data":
            from src.data.prepare_dataset import prepare_and_split_dataset
            prepare_and_split_dataset(
                train_ratio=args.train_split,
                val_ratio=args.val_split,
                seed=args.seed,
                allow_synthetic=args.generate_samples,
            )
            return 0

        if args.command == "train":
            from src.training.train import run_training
            run_training(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
            return 0

        if args.command == "evaluate":
            from src.evaluation.evaluate import run_evaluation
            run_evaluation(
                model_path=args.model,
                threshold=args.threshold,
            )
            return 0

        if args.command == "webcam":
            from src.inference.webcam import run_inference
            run_inference(
                source=args.camera_index,
                record=args.record,
                threshold=args.threshold,
                cooldown_seconds=args.cooldown,
                headless=args.headless,
            )
            return 0

        if args.command == "video":
            from src.inference.webcam import run_inference
            run_inference(
                source=args.input,
                record=args.record,
                threshold=args.threshold,
                cooldown_seconds=args.cooldown,
                headless=args.headless,
            )
            return 0

        if args.command == "test":
            import pytest
            logger.info("Running pytest test suite...")
            exit_code = pytest.main(["-v", "tests"])
            return int(exit_code)

        # No command given: print help
        parser.print_help()
        return 0

    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Exiting cleanly.")
        return 0
    except Exception as err:
        logger.error("Execution error: %s", err, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

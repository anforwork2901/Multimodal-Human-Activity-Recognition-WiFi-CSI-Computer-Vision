#!/usr/bin/env python3
"""
CLI Training Entry Point for Multimodal Activity Recognition.

Usage examples:
    python scripts/train.py --model Fusion
    python scripts/train.py --model all --epochs 30 --batch-size 16
"""

import argparse
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import load_config
from src.use_cases.train_pipeline import TrainPipelineUseCase


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Multimodal Activity Recognition Models (CV + WiFi CSI)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Fusion",
        choices=["Vision-Only", "CSI-Only", "Fusion", "all"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Overrides from CLI if specified
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr

    models_to_train = (
        ["Vision-Only", "CSI-Only", "Fusion"]
        if args.model == "all"
        else [args.model]
    )

    pipeline = TrainPipelineUseCase(config)
    pipeline.execute(models_to_train=models_to_train)


if __name__ == "__main__":
    main()

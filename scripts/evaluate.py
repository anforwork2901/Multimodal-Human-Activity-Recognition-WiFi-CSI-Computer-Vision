#!/usr/bin/env python3
"""
CLI Evaluation Entry Point for Multimodal Activity Recognition.

Usage:
    python scripts/evaluate.py --model Fusion --checkpoint Output/models/Fusion_model.pth
"""

import argparse
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import load_config
from src.use_cases.evaluate_pipeline import EvaluatePipelineUseCase


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Multimodal Activity Recognition Checkpoint."
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
        choices=["Vision-Only", "CSI-Only", "Fusion"],
        help="Model architecture name.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to saved .pth model checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    eval_pipeline = EvaluatePipelineUseCase(config)
    eval_pipeline.execute(model_name=args.model, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()

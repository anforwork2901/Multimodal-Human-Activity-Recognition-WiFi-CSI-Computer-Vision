"""Application Use Case: Standalone Model Evaluation & Checkpoint Testing."""

import os
from typing import Optional
import torch

from src.config.settings import AppConfig
from src.data.datamodule import ActivityDataModule
from src.domain.entities import EvaluationResult
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualizer import plot_confusion_matrix
from src.models.fusion_net import HybridFPNFusionNet
from src.models.vision_only import VisionOnlyCNN
from src.models.wifi_only import WiFiOnlyNN


class EvaluatePipelineUseCase:
    """Evaluates a saved model checkpoint against the test dataset."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.device = config.get_device()
        self.data_module = ActivityDataModule(config)
        self.evaluator = ModelEvaluator(self.device)

    def execute(self, model_name: str, checkpoint_path: str) -> EvaluationResult:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        num_classes = self.config.data.num_classes
        csi_length = self.config.data.csi_length

        if model_name == "Vision-Only":
            model = VisionOnlyCNN(num_classes=num_classes)
        elif model_name == "CSI-Only":
            model = WiFiOnlyNN(csi_length=csi_length, num_classes=num_classes)
        elif model_name == "Fusion":
            model = HybridFPNFusionNet(csi_length=csi_length, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)

        test_loader = self.data_module.get_test_dataloader()
        eval_res = self.evaluator.evaluate(model, model_name, test_loader)

        plot_confusion_matrix(eval_res, save_dir=self.config.paths.plots_save_path)
        return eval_res

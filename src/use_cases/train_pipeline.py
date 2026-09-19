"""Application Use Case: Comprehensive Multimodal Training & Benchmarking Pipeline."""

import json
import os
from typing import Dict, List, Optional, Tuple
import torch

from src.config.settings import AppConfig
from src.data.datamodule import ActivityDataModule
from src.domain.entities import EvaluationResult, TrainingHistory
from src.domain.interfaces import IActivityModel
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualizer import (
    plot_comparative_results,
    plot_confusion_matrix,
    plot_training_history,
)
from src.models.fusion_net import HybridFPNFusionNet
from src.models.vision_only import VisionOnlyCNN
from src.models.wifi_only import WiFiOnlyNN
from src.training.trainer import ActivityTrainer


class TrainPipelineUseCase:
    """
    Coordinates data preparation, model instantiation, training loops,
    evaluation benchmarks, and visualization reporting.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.config.paths.ensure_directories()
        self.device = config.get_device()

        print(f"🔧 Initializing Training Pipeline on device: {self.device}")
        self.data_module = ActivityDataModule(config)
        self.trainer = ActivityTrainer(config)
        self.evaluator = ModelEvaluator(self.device)

    def _build_model(self, model_name: str) -> IActivityModel:
        num_classes = self.config.data.num_classes
        csi_length = self.config.data.csi_length

        if model_name == "Vision-Only":
            return VisionOnlyCNN(num_classes=num_classes)
        elif model_name == "CSI-Only":
            return WiFiOnlyNN(csi_length=csi_length, num_classes=num_classes)
        elif model_name == "Fusion":
            return HybridFPNFusionNet(csi_length=csi_length, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model architecture: '{model_name}'")

    def execute(
        self, models_to_train: Optional[List[str]] = None
    ) -> Tuple[Dict[str, TrainingHistory], List[EvaluationResult]]:
        target_models = models_to_train or self.config.training.models_to_train

        train_loader = self.data_module.get_train_dataloader()
        val_loader = self.data_module.get_val_dataloader()
        test_loader = self.data_module.get_test_dataloader()

        print("\n📊 DATASET STATISTICS:")
        print(f"   Train samples: {len(self.data_module.train_dataset)}")
        print(f"   Val samples:   {len(self.data_module.val_dataset)}")
        print(f"   Test samples:  {len(self.data_module.test_dataset)}")
        print(f"   Batch size:    {self.config.data.batch_size}")

        if len(self.data_module.train_dataset) == 0:
            raise FileNotFoundError(
                f"No training samples found in '{self.data_module.resolved_path}'. "
                "Please place the dataset in 'Dataset/Data_acti_CSI_CV/Activity_CV_CSI_Dataset' "
                "or 'data_activity/' as described in README.md."
            )

        histories: Dict[str, TrainingHistory] = {}
        eval_results: List[EvaluationResult] = []

        for model_name in target_models:
            print(f"\n{'=' * 75}")
            print(f"🎯 STARTING PIPELINE STAGE: {model_name}")
            print(f"{'=' * 75}")

            model = self._build_model(model_name)
            trained_model, history = self.trainer.train(
                model=model,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
            )
            histories[model_name] = history

            # Plot training curves
            plot_training_history(
                history=history,
                model_name=model_name,
                save_dir=self.config.paths.plots_save_path,
            )

            # Evaluate on unseen test dataset
            eval_res = self.evaluator.evaluate(
                model=trained_model,
                model_name=model_name,
                test_loader=test_loader,
            )
            eval_results.append(eval_res)

            # Plot confusion matrix
            plot_confusion_matrix(
                result=eval_res, save_dir=self.config.paths.plots_save_path
            )

            # Save detailed metrics JSON
            json_path = os.path.join(
                self.config.paths.results_save_path, f"{model_name}_results.json"
            )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_name": eval_res.model_name,
                        "accuracy": eval_res.accuracy,
                        "loss": eval_res.loss,
                        "classification_report": eval_res.classification_report,
                    },
                    f,
                    indent=2,
                )
            print(f"📄 Detailed results saved to: {json_path}")

        # Comparative summary across models
        if len(eval_results) > 1:
            plot_comparative_results(
                results=eval_results, save_dir=self.config.paths.plots_save_path
            )

        print("\n" + "=" * 75)
        print("🏆 MULTIMODAL BENCHMARK RESULTS:")
        print("=" * 75)
        for res in eval_results:
            print(f"  {res.model_name:15s} | Accuracy: {res.accuracy:6.2f}% | Loss: {res.loss:.4f}")

        best = max(eval_results, key=lambda x: x.accuracy)
        print(f"\n🌟 TOP PERFORMING MODEL: {best.model_name} ({best.accuracy:.2f}%)")
        return histories, eval_results

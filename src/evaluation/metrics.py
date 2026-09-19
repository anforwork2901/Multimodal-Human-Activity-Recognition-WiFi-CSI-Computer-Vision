"""Evaluation metrics computation implementing IEvaluator."""

from typing import List
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.domain.constants import CLASS_NAMES
from src.domain.entities import EvaluationResult
from src.domain.interfaces import IActivityModel, IEvaluator


class ModelEvaluator(IEvaluator):
    """Computes comprehensive evaluation metrics on test datasets."""

    def __init__(self, device: torch.device):
        self.device = device

    def evaluate(
        self, model: IActivityModel, model_name: str, test_loader: DataLoader
    ) -> EvaluationResult:
        print(f"\n--- Evaluating {model_name} on Test Set ---")
        model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions: List[int] = []
        all_targets: List[int] = []
        all_probabilities: List[List[float]] = []
        test_batches = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                try:
                    images = batch["images"].to(self.device)
                    csi = batch["csi"].to(self.device)
                    labels = batch["class_ids"]

                    if isinstance(labels, list):
                        labels = torch.tensor(labels)
                    labels = labels.to(self.device).long()

                    outputs, _ = model(images=images, csi=csi)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()

                    probs = F.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    test_total += labels.size(0)
                    test_correct += (preds == labels).sum().item()
                    test_batches += 1

                    all_predictions.extend(preds.cpu().numpy().tolist())
                    all_targets.extend(labels.cpu().numpy().tolist())
                    all_probabilities.extend(probs.cpu().numpy().tolist())
                except Exception:
                    continue

        if test_batches == 0 or test_total == 0:
            return EvaluationResult(
                model_name=model_name,
                accuracy=0.0,
                loss=float("inf"),
            )

        test_acc = 100.0 * test_correct / test_total
        avg_loss = test_loss / test_batches

        report = {}
        if len(all_targets) > 0 and len(all_predictions) > 0:
            try:
                report = classification_report(
                    all_targets,
                    all_predictions,
                    target_names=CLASS_NAMES,
                    output_dict=True,
                )
            except Exception:
                pass

        print(f"📊 Test Accuracy: {test_acc:.2f}% | Test Loss: {avg_loss:.4f}")

        return EvaluationResult(
            model_name=model_name,
            accuracy=test_acc,
            loss=avg_loss,
            predictions=all_predictions,
            targets=all_targets,
            probabilities=all_probabilities,
            classification_report=report,
        )

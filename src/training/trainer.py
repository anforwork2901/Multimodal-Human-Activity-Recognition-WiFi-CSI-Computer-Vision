"""Model training engine implementing ITrainer."""

import os
from typing import Optional, Tuple
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config.settings import AppConfig
from src.domain.entities import TrainingHistory
from src.domain.interfaces import IActivityModel, ITrainer
from src.losses.triplet_loss import TripletLoss, create_triplets
from src.training.callbacks import EarlyStopping, ModelCheckpoint


class ActivityTrainer(ITrainer):
    """
    Standard Model Trainer for Vision-Only, CSI-Only, and Multimodal Fusion models.
    Supports gradient clipping, LR scheduling, Triplet Loss for GaitFi metric learning,
    and automatic best checkpoint restoration.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.device = config.get_device()

    def _compute_loss(
        self,
        model: IActivityModel,
        batch: dict,
        model_name: str,
        criterion: nn.CrossEntropyLoss,
        triplet_criterion: Optional[TripletLoss] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = batch["images"].to(self.device)
        csi = batch["csi"].to(self.device)
        labels = batch["class_ids"]

        if isinstance(labels, list):
            labels = torch.tensor(labels)
        labels = labels.to(self.device).long()

        # Forward pass
        outputs, features = model(images=images, csi=csi)
        classification_loss = criterion(outputs, labels)
        total_loss = classification_loss

        # Optional Triplet Metric Loss for Fusion
        if (
            model_name == "Fusion"
            and triplet_criterion is not None
            and features is not None
        ):
            try:
                anchors, positives, negatives = create_triplets(features, labels)
                if anchors is not None:
                    t_loss = triplet_criterion(anchors, positives, negatives)
                    total_loss = (
                        classification_loss
                        + self.config.model.triplet_loss_weight * t_loss
                    )
            except Exception:
                pass

        return total_loss, outputs

    def train(
        self,
        model: IActivityModel,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Tuple[IActivityModel, TrainingHistory]:
        print(f"\n{'=' * 70}")
        print(f"🚀 TRAINING {model_name.upper()} ON {self.device}")
        print(f"{'=' * 70}")

        model_nn = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model_nn.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.config.training.scheduler_factor,
            patience=self.config.training.scheduler_patience,
        )

        triplet_criterion = TripletLoss(margin=self.config.model.triplet_margin).to(
            self.device
        )
        early_stopping = EarlyStopping(
            patience=self.config.training.patience, mode="max"
        )

        save_path = os.path.join(
            self.config.paths.model_save_path, f"{model_name}_model.pth"
        )
        checkpoint = ModelCheckpoint(filepath=save_path)

        history = TrainingHistory()
        best_val_acc = 0.0

        for epoch in range(self.config.training.epochs):
            lr_current = optimizer.param_groups[0]["lr"]
            history.learning_rates.append(lr_current)
            print(f"\nEpoch {epoch + 1}/{self.config.training.epochs} | LR: {lr_current:.6f}")

            # 1. Training Phase
            model_nn.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_batches = 0

            with tqdm(train_loader, desc=f"Train {model_name}") as pbar:
                for batch in train_loader:
                    try:
                        optimizer.zero_grad()
                        loss, outputs = self._compute_loss(
                            model, batch, model_name, criterion, triplet_criterion
                        )

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model_nn.parameters(),
                            max_norm=self.config.training.gradient_clip_max_norm,
                        )
                        optimizer.step()

                        train_loss += loss.item()
                        _, preds = torch.max(outputs, 1)

                        labels = batch["class_ids"]
                        if isinstance(labels, list):
                            labels = torch.tensor(labels)
                        labels = labels.to(self.device)

                        train_total += labels.size(0)
                        train_correct += (preds == labels).sum().item()
                        train_batches += 1

                        cur_acc = (
                            100.0 * train_correct / train_total if train_total > 0 else 0.0
                        )
                        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{cur_acc:.2f}%"})
                    except Exception as e:
                        continue

            if train_batches == 0:
                print("❌ No valid training batches processed.")
                continue

            epoch_train_loss = train_loss / train_batches
            epoch_train_acc = 100.0 * train_correct / train_total

            # 2. Validation Phase
            model_nn.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    try:
                        loss, outputs = self._compute_loss(
                            model, batch, model_name, criterion, triplet_criterion
                        )
                        val_loss += loss.item()
                        _, preds = torch.max(outputs, 1)

                        labels = batch["class_ids"]
                        if isinstance(labels, list):
                            labels = torch.tensor(labels)
                        labels = labels.to(self.device)

                        val_total += labels.size(0)
                        val_correct += (preds == labels).sum().item()
                        val_batches += 1
                    except Exception:
                        continue

            if val_batches == 0:
                print("❌ No valid validation batches processed.")
                continue

            epoch_val_loss = val_loss / val_batches
            epoch_val_acc = 100.0 * val_correct / val_total

            history.train_loss.append(epoch_train_loss)
            history.train_acc.append(epoch_train_acc)
            history.val_loss.append(epoch_val_loss)
            history.val_acc.append(epoch_val_acc)

            print(
                f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%"
            )
            print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}%")

            scheduler.step(epoch_val_acc)

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                checkpoint.save_best(model_nn)
                print(f"🌟 Best validation accuracy: {best_val_acc:.2f}% (Model saved)")

            if not early_stopping.step(epoch_val_acc):
                print(
                    f"⏳ Early stopping patience: {early_stopping.counter}/{self.config.training.patience}"
                )

            if early_stopping.early_stop:
                print(f"\n⏹️ Early stopping triggered at epoch {epoch + 1}.")
                break

        # Restore best checkpoint
        checkpoint.restore_best(model_nn)
        print(f"\n✨ Training finished! Restored best model (Val Acc: {best_val_acc:.2f}%).")
        return model, history

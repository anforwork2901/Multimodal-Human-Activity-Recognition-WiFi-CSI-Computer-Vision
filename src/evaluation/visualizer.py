"""Visualization utilities for training progress, confusion matrices, and model benchmarks."""

import os
from typing import List
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from src.domain.constants import CLASS_NAMES
from src.domain.entities import EvaluationResult, TrainingHistory


def plot_training_history(
    history: TrainingHistory, model_name: str, save_dir: str = "Output/plots"
) -> None:
    """Plot Loss, Accuracy, and Learning Rate curves for a model."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{model_name} Training History", fontsize=16, fontweight="bold")

    epochs = range(1, len(history.train_loss) + 1)

    # 1. Loss
    axes[0, 0].plot(epochs, history.train_loss, "b-", label="Train Loss", linewidth=2)
    axes[0, 0].plot(epochs, history.val_loss, "r-", label="Val Loss", linewidth=2)
    axes[0, 0].set_title("Model Loss", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Accuracy
    axes[0, 1].plot(epochs, history.train_acc, "b-", label="Train Acc", linewidth=2)
    axes[0, 1].plot(epochs, history.val_acc, "r-", label="Val Acc", linewidth=2)
    axes[0, 1].set_title("Model Accuracy (%)", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Learning Rate
    if history.learning_rates:
        axes[1, 0].plot(
            range(1, len(history.learning_rates) + 1),
            history.learning_rates,
            "g-",
            linewidth=2,
        )
        axes[1, 0].set_title("Learning Rate Schedule", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("LR")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Summary Text
    best_val = max(history.val_acc) if history.val_acc else 0.0
    final_train = history.train_acc[-1] if history.train_acc else 0.0
    final_val = history.val_acc[-1] if history.val_acc else 0.0

    axes[1, 1].text(0.1, 0.7, f"Best Val Accuracy: {best_val:.2f}%", fontsize=14, fontweight="bold")
    axes[1, 1].text(0.1, 0.5, f"Final Train Accuracy: {final_train:.2f}%", fontsize=12)
    axes[1, 1].text(0.1, 0.3, f"Final Val Accuracy: {final_val:.2f}%", fontsize=12)
    axes[1, 1].text(0.1, 0.1, f"Total Epochs: {len(history.train_loss)}", fontsize=12)
    axes[1, 1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_training_history.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Training curve saved to: {save_path}")


def plot_confusion_matrix(
    result: EvaluationResult, save_dir: str = "Output/plots"
) -> None:
    """Plot Confusion Matrix heatmap."""
    os.makedirs(save_dir, exist_ok=True)
    if not result.targets or not result.predictions:
        return

    cm = confusion_matrix(result.targets, result.predictions)

    plt.figure(figsize=(10, 8))
    if HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
        )
    else:
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.colorbar()
        tick_marks = range(len(CLASS_NAMES))
        plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
        plt.yticks(tick_marks, CLASS_NAMES)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
    plt.title(
        f"{result.model_name} - Confusion Matrix (Acc: {result.accuracy:.2f}%)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{result.model_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📈 Confusion matrix saved to: {save_path}")


def plot_comparative_results(
    results: List[EvaluationResult], save_dir: str = "Output/plots"
) -> None:
    """Plot comparative benchmark bar charts across all tested models."""
    os.makedirs(save_dir, exist_ok=True)
    if not results:
        return

    model_names = [r.model_name for r in results]
    accuracies = [r.accuracy for r in results]
    losses = [r.loss for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Multimodal Activity Recognition Benchmark", fontsize=16, fontweight="bold")

    colors = ["#4A90E2", "#E94E77", "#50E3C2"][: len(model_names)]

    # Accuracy comparison
    bars1 = axes[0].bar(model_names, accuracies, color=colors)
    axes[0].set_title("Test Accuracy Comparison (%)", fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 105)
    for bar, acc in zip(bars1, accuracies):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{acc:.2f}%",
            ha="center",
            fontweight="bold",
        )

    # Loss comparison
    bars2 = axes[1].bar(model_names, losses, color=colors)
    axes[1].set_title("Test Cross-Entropy Loss Comparison", fontweight="bold")
    axes[1].set_ylabel("Loss")
    for bar, loss in zip(bars2, losses):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{loss:.4f}",
            ha="center",
            fontweight="bold",
        )

    plt.tight_layout()
    save_path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🏆 Comparative benchmark plot saved to: {save_path}")

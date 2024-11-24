from typing import Union, Dict, Any, Optional

import matplotlib
import torch
import torch.nn as nn
import wandb
from torch import Tensor
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

import matplotlib.pyplot as plt
import seaborn as sns

# Set Matplotlib backend to Agg (non-GUI backend)
matplotlib.use("Agg")


class BinaryClassificationModel(LightningModule):
    def __init__(
            self,
            classifier: nn.Module,
            learning_rate: float = 1e-3,
            pos_weight: float = 1.5,
            monitor_metric: str = "val/loss",
            **kwargs
    ):
        super().__init__()

        # Hyperparameters
        self.save_hyperparameters(ignore=['classifier'])
        self.monitor_metric = monitor_metric
        self.learning_rate = learning_rate

        # Single classifier model for binary classification
        self.classifier = classifier

        # Binary Cross Entropy Loss
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        # Metrics
        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")
        self.confusion_matrix = ConfusionMatrix(task="binary")

        # Storage for predictions and targets during test
        self.test_preds = []
        self.test_targets = []

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self.monitor_metric}

    def training_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any]]:
        x, y = batch
        y = y.float().view(-1, 1)  # Ensure y is reshaped to (batch_size, 1) for BCE Loss
        preds = self(x)
        loss = self.loss_fn(preds, y)

        # Calculate metrics
        acc = self.accuracy(preds, y.int())
        precision = self.precision(preds, y.int())
        recall = self.recall(preds, y.int())
        f1 = self.f1(preds, y.int())

        # Log metrics
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        self.log("train/precision", precision, prog_bar=True)
        self.log("train/recall", recall, prog_bar=True)
        self.log("train/f1", f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any]]:
        x, y = batch
        y = y.float().view(-1, 1)
        preds = self(x)
        loss = self.loss_fn(preds, y)

        # Calculate metrics
        acc = self.accuracy(preds, y.int())
        precision = self.precision(preds, y.int())
        recall = self.recall(preds, y.int())
        f1 = self.f1(preds, y.int())

        # Log metrics
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/precision", precision, prog_bar=True)
        self.log("val/recall", recall, prog_bar=True)
        self.log("val/f1", f1, prog_bar=True)

        return {"preds": preds, "targets": y}

    def test_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any]]:
        x, y = batch
        y = y.float().view(-1, 1)
        preds = self(x)
        loss = self.loss_fn(preds, y)

        # Store predictions and targets for confusion matrix
        self.test_preds.append(preds)
        self.test_targets.append(y.int())

        # Calculate metrics
        acc = self.accuracy(preds, y.int())
        precision = self.precision(preds, y.int())
        recall = self.recall(preds, y.int())
        f1 = self.f1(preds, y.int())

        # Log metrics
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)
        self.log("test/precision", precision, prog_bar=True)
        self.log("test/recall", recall, prog_bar=True)
        self.log("test/f1", f1, prog_bar=True)

        return {"test_loss": loss, "test_acc": acc, "test_precision": precision, "test_recall": recall, "test_f1": f1}

    def on_test_end(self):
        """Logs the confusion matrix at the end of testing."""
        # Concatenate all predictions and targets across batches
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)

        # Generate confusion matrix
        confusion_matrix = self.confusion_matrix(preds, targets).cpu().numpy()

        # Turn off interactive mode
        plt.ioff()

        # Plot the confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", square=True, cbar=False)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")

        # Log the plot as an image to WandB
        self.logger.experiment.log({"test/confusion_matrix": [wandb.Image(plt)]})

        # Show / Close the image after run
        # plt.show()
        plt.close()

        # Clear stored predictions and targets
        self.test_preds.clear()
        self.test_targets.clear()

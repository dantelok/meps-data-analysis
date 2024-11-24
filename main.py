import argparse
import time
from multiprocessing import freeze_support

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.datamodule import CSVDataModule
from src.models.classifier import Classifier
from src.models.classification_model import BinaryClassificationModel

# Define the Sweep Configuration for Bayesian Optimization
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/recall", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-2},
        "pos_weight": {"distribution": "uniform", "min": 3.0, "max": 5.0},
        "batch_size": {"values": [16, 32, 64]},
        "dropout_rate": {"distribution": "uniform", "min": 0.1, "max": 0.5},
        "hidden_dims": {
            "values": [
                [128, 256, 512, 256, 128, 64],
                [64, 128, 256, 128, 64],
                [128, 256, 256, 128]
            ]
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Binary Classification Training with PyTorch Lightning")

    # Flag of WandB sweep
    parser.add_argument("--sweep", type=bool, default=False, help="Run the WandB sweep for hyperparameter optimization")

    # Trainer arguments
    parser.add_argument("--trainer_max_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--trainer_gpus", type=int, default=1, help="Number of GPUs to use")

    # Data module arguments
    parser.add_argument("--datamodule_batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--datamodule_apply_smote", type=bool, default=False, help="Apply SMOTE")

    # Model arguments
    parser.add_argument("--model_learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model_pos_weight", type=float, default=4.0, help="Positive class weight")
    parser.add_argument("--model_dropout_rate", type=float, default=0.3, help="Dropout rate")

    return parser.parse_args()


def train(args):
    # Initialize WandbLogger
    wandb_logger = WandbLogger(project="meps-data-analysis", name=f"run_{int(time.time())}")

    # Set up the Data Module
    data_module = CSVDataModule(
        data_path='data/100-highBPDiagnosed-clean-data.csv',
        target_column='highBPDiagnosed',
        train_val_test_split=(0.7, 0.15, 0.15),
        batch_size=args.datamodule_batch_size,
        num_workers=4,
        pin_memory=True,
        apply_smote=args.datamodule_apply_smote,
    )

    # Define the Classifier
    input_dim = 115
    hidden_dims = [128, 256, 512, 256, 128, 64]
    classifier = Classifier(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=args.model_dropout_rate)

    # Initialize the BinaryClassificationModel with the Complex Classifier
    model = BinaryClassificationModel(
        classifier=classifier,
        learning_rate=args.model_learning_rate,
        pos_weight=args.model_pos_weight,
        monitor_metric="val/recall",
    )

    # Set up Callbacks (Model Checkpointing and Early Stopping)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="max",
        save_top_k=1,
        filename="best_model-{epoch:02d}-{val/loss:.2f}"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/loss",
        mode="max",
        patience=5
    )

    # Initialize Trainer with updated arguments
    trainer = pl.Trainer(
        max_epochs=args.trainer_max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="gpu" if (torch.cuda.is_available() or torch.mps.is_available()) and args.trainer_gpus > 0 else "cpu",
        devices=1,  # Use 1 GPU or 1 CPU
        log_every_n_steps=10
    )

    # Train the Model
    trainer.fit(model, datamodule=data_module)

    # Test the Model
    trainer.test(datamodule=data_module, ckpt_path="best")


if __name__ == '__main__':
    # freeze_support()

    # Parse command-line arguments
    args = parse_args()

    # Check if we should run the WandB sweep
    if args.sweep:
        # Run Bayesian Optimization Sweep if sweep is True
        num_search = 20
        sweep_id = wandb.sweep(sweep_config, project="meps-data-analysis")
        wandb.agent(sweep_id, function=lambda: train(args), count=num_search)
    else:
        # Run training with user-defined hyperparameters
        train(args)

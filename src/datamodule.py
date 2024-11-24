from typing import Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


class CustomCSVDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y


class CSVDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        target_column: str,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        apply_smote: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.target_column = target_column
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.apply_smote = apply_smote

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Load and preprocess data from CSV."""
        # Read the CSV
        df = pd.read_csv(self.data_path)

        # Separate features and target
        X = df.drop(columns=['id', self.target_column]).values
        y = df[self.target_column].values

        # Normalize features using StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split into train, val, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.train_val_test_split[1] + self.train_val_test_split[2], stratify=y
        )
        val_size = self.train_val_test_split[1] / (self.train_val_test_split[1] + self.train_val_test_split[2])
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, stratify=y_temp)

        # Apply SMOTE to the training data if enabled
        if self.apply_smote:
            smote = SMOTE()
            X_train, y_train = smote.fit_resample(X_train, y_train)

        # Convert to PyTorch Datasets
        self.data_train = CustomCSVDataset(X_train, y_train)
        self.data_val = CustomCSVDataset(X_val, y_val)
        self.data_test = CustomCSVDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=False,
        )
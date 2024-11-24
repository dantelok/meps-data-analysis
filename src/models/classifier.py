import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout_rate: float = 0.3):
        super(Classifier, self).__init__()

        # Define a list to hold layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())

        # Hidden layers with dropout and batch normalization
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())  # Sigmoid activation for binary classification

        # Wrap all layers in a Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

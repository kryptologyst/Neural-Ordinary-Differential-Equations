"""Baseline models for comparison with Neural ODEs."""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LinearBaseline(nn.Module):
    """Linear regression baseline."""
    
    def __init__(self, input_dim: int = 1, output_dim: int = 1):
        """Initialize linear baseline.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(x)


class PolynomialBaseline(nn.Module):
    """Polynomial regression baseline."""
    
    def __init__(self, input_dim: int = 1, degree: int = 3, output_dim: int = 1):
        """Initialize polynomial baseline.
        
        Args:
            input_dim: Input dimension
            degree: Polynomial degree
            output_dim: Output dimension
        """
        super().__init__()
        
        # Create polynomial features
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        
        # Linear layer for final output
        n_features = self.poly_features.n_output_features_
        self.linear = nn.Linear(n_features, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Convert to numpy for sklearn preprocessing
        x_np = x.detach().cpu().numpy()
        
        # Apply polynomial features
        x_poly = self.poly_features.fit_transform(x_np)
        
        # Convert back to tensor
        x_poly = torch.tensor(x_poly, dtype=x.float32, device=x.device)
        
        return self.linear(x_poly)


class MLPBaseline(nn.Module):
    """Multi-layer perceptron baseline."""
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: list = [64, 32],
        output_dim: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        """Initialize MLP baseline.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions
            output_dim: Output dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        # Define activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class RNNBaseline(nn.Module):
    """RNN baseline for time series modeling."""
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        rnn_type: str = "LSTM",
        dropout: float = 0.1,
    ):
        """Initialize RNN baseline.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of RNN layers
            output_dim: Output dimension
            rnn_type: Type of RNN ("LSTM", "GRU", "RNN")
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Choose RNN type
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # RNN forward pass
        rnn_out, _ = self.rnn(x)
        
        # Apply output layer
        output = self.output_layer(rnn_out)
        
        return output


class SklearnBaseline:
    """Wrapper for sklearn baselines."""
    
    def __init__(self, model_type: str = "linear", **kwargs):
        """Initialize sklearn baseline.
        
        Args:
            model_type: Type of sklearn model
            **kwargs: Additional arguments for the model
        """
        self.model_type = model_type
        
        if model_type == "linear":
            self.model = LinearRegression(**kwargs)
        elif model_type == "ridge":
            self.model = Ridge(**kwargs)
        elif model_type == "polynomial":
            degree = kwargs.pop("degree", 3)
            self.model = Pipeline([
                ("poly", PolynomialFeatures(degree=degree)),
                ("linear", LinearRegression())
            ])
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnBaseline":
        """Fit the model."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        return self.model.score(X, y)


class EnsembleBaseline(nn.Module):
    """Ensemble of baseline models."""
    
    def __init__(
        self,
        models: list,
        weights: Optional[list] = None,
    ):
        """Initialize ensemble baseline.
        
        Args:
            models: List of baseline models
            weights: Optional weights for ensemble members
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = sum(w * pred for w, pred in zip(self.weights, predictions))
        
        return weighted_pred

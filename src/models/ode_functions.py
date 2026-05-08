"""Neural ODE function definitions."""

from typing import Optional

import torch
import torch.nn as nn


class BasicODEFunc(nn.Module):
    """Basic ODE function for Neural ODEs.
    
    Implements a simple neural network that defines the derivative dy/dt = f(t, y).
    """
    
    def __init__(
        self,
        hidden_dim: int = 50,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """Initialize ODE function.
        
        Args:
            hidden_dim: Hidden dimension size
            activation: Activation function ("relu", "tanh", "sigmoid")
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Define activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Neural network layers
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of ODE function.
        
        Args:
            t: Time tensor
            y: State tensor
            
        Returns:
            Derivative dy/dt
        """
        # Concatenate time and state for input
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        
        # Use only the state as input (time-independent ODE)
        return self.net(y)


class TimeDependentODEFunc(nn.Module):
    """Time-dependent ODE function.
    
    Implements an ODE function that explicitly depends on time: dy/dt = f(t, y).
    """
    
    def __init__(
        self,
        hidden_dim: int = 50,
        activation: str = "tanh",
        dropout: float = 0.0,
    ):
        """Initialize time-dependent ODE function.
        
        Args:
            hidden_dim: Hidden dimension size
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Define activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Neural network that takes both time and state as input
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 2 inputs: time and state
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of time-dependent ODE function.
        
        Args:
            t: Time tensor
            y: State tensor
            
        Returns:
            Derivative dy/dt
        """
        # Ensure proper dimensions
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        
        # Concatenate time and state
        inputs = torch.cat([t, y], dim=-1)
        return self.net(inputs)


class HamiltonianODEFunc(nn.Module):
    """Hamiltonian ODE function for energy-conserving systems.
    
    Implements a Hamiltonian neural network that preserves energy.
    """
    
    def __init__(
        self,
        hidden_dim: int = 50,
        activation: str = "tanh",
    ):
        """Initialize Hamiltonian ODE function.
        
        Args:
            hidden_dim: Hidden dimension size
            activation: Activation function
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Define activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Hamiltonian network (scalar function)
        self.hamiltonian_net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # position and momentum
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Forward pass of Hamiltonian ODE function.
        
        Args:
            t: Time tensor
            state: State tensor [position, momentum]
            
        Returns:
            Derivative of state
        """
        # Ensure proper dimensions
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(-1)
        
        # Compute Hamiltonian
        hamiltonian = self.hamiltonian_net(state)
        
        # Compute gradients for Hamiltonian equations
        # dq/dt = dH/dp, dp/dt = -dH/dq
        grad_h = torch.autograd.grad(
            hamiltonian.sum(),
            state,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Hamiltonian equations
        dq_dt = grad_h[:, 1:2]  # dH/dp
        dp_dt = -grad_h[:, 0:1]  # -dH/dq
        
        return torch.cat([dq_dt, dp_dt], dim=-1)


class AugmentedODEFunc(nn.Module):
    """Augmented Neural ODE function.
    
    Implements an augmented ODE function that includes additional dimensions
    to improve expressiveness and training stability.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 50,
        augmented_dim: int = 0,
        activation: str = "tanh",
        dropout: float = 0.0,
    ):
        """Initialize augmented ODE function.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension size
            augmented_dim: Additional augmented dimensions
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.augmented_dim = augmented_dim
        self.total_dim = input_dim + augmented_dim
        
        # Define activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Neural network
        self.net = nn.Sequential(
            nn.Linear(self.total_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.total_dim),
        )
    
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of augmented ODE function.
        
        Args:
            t: Time tensor
            y: State tensor (may include augmented dimensions)
            
        Returns:
            Derivative dy/dt
        """
        # Ensure proper dimensions
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        
        return self.net(y)

"""Tests for Neural ODE models."""

import pytest
import torch
import torch.nn as nn

from src.models import (
    BasicODEFunc,
    NeuralODE,
    AugmentedNeuralODE,
    LinearBaseline,
    MLPBaseline,
)


class TestBasicODEFunc:
    """Test BasicODEFunc class."""
    
    def test_initialization(self):
        """Test ODE function initialization."""
        ode_func = BasicODEFunc(hidden_dim=32, activation="relu")
        assert isinstance(ode_func, nn.Module)
        assert ode_func.hidden_dim == 32
    
    def test_forward_pass(self):
        """Test forward pass."""
        ode_func = BasicODEFunc(hidden_dim=32)
        
        t = torch.tensor(0.0)
        y = torch.tensor([[1.0]])
        
        output = ode_func(t, y)
        assert output.shape == (1, 1)
        assert isinstance(output, torch.Tensor)
    
    def test_different_activations(self):
        """Test different activation functions."""
        activations = ["relu", "tanh", "sigmoid"]
        
        for activation in activations:
            ode_func = BasicODEFunc(hidden_dim=16, activation=activation)
            
            t = torch.tensor(0.0)
            y = torch.tensor([[1.0]])
            
            output = ode_func(t, y)
            assert output.shape == (1, 1)


class TestNeuralODE:
    """Test NeuralODE class."""
    
    def test_initialization(self):
        """Test Neural ODE initialization."""
        ode_func = BasicODEFunc(hidden_dim=32)
        neural_ode = NeuralODE(ode_func=ode_func)
        
        assert isinstance(neural_ode, nn.Module)
        assert neural_ode.ode_func == ode_func
    
    def test_forward_pass(self):
        """Test forward pass."""
        ode_func = BasicODEFunc(hidden_dim=32)
        neural_ode = NeuralODE(ode_func=ode_func)
        
        x0 = torch.tensor([[0.0]])
        t = torch.linspace(0, 1, 10)
        
        output = neural_ode(x0, t)
        assert output.shape == (10, 1, 1)  # (time_steps, batch_size, state_dim)
    
    def test_different_solvers(self):
        """Test different ODE solvers."""
        ode_func = BasicODEFunc(hidden_dim=16)
        
        solvers = ["dopri5", "rk4", "euler"]
        
        for method in solvers:
            neural_ode = NeuralODE(ode_func=ode_func, method=method)
            
            x0 = torch.tensor([[0.0]])
            t = torch.linspace(0, 1, 5)
            
            output = neural_ode(x0, t)
            assert output.shape == (5, 1, 1)


class TestAugmentedNeuralODE:
    """Test AugmentedNeuralODE class."""
    
    def test_initialization(self):
        """Test augmented Neural ODE initialization."""
        ode_func = BasicODEFunc(hidden_dim=32)
        aug_neural_ode = AugmentedNeuralODE(
            ode_func=ode_func,
            input_dim=1,
            augmented_dim=2
        )
        
        assert isinstance(aug_neural_ode, nn.Module)
        assert aug_neural_ode.input_dim == 1
        assert aug_neural_ode.augmented_dim == 2
    
    def test_forward_pass(self):
        """Test forward pass with augmentation."""
        ode_func = BasicODEFunc(hidden_dim=32)
        aug_neural_ode = AugmentedNeuralODE(
            ode_func=ode_func,
            input_dim=1,
            augmented_dim=1
        )
        
        x0 = torch.tensor([[0.0]])
        t = torch.linspace(0, 1, 5)
        
        output = aug_neural_ode(x0, t)
        assert output.shape == (5, 1, 1)  # Only original dimensions returned


class TestBaselineModels:
    """Test baseline models."""
    
    def test_linear_baseline(self):
        """Test linear baseline."""
        model = LinearBaseline(input_dim=1, output_dim=1)
        
        x = torch.tensor([[1.0]])
        output = model(x)
        
        assert output.shape == (1, 1)
        assert isinstance(output, torch.Tensor)
    
    def test_mlp_baseline(self):
        """Test MLP baseline."""
        model = MLPBaseline(
            input_dim=1,
            hidden_dims=[32, 16],
            output_dim=1
        )
        
        x = torch.tensor([[1.0]])
        output = model(x)
        
        assert output.shape == (1, 1)
        assert isinstance(output, torch.Tensor)
    
    def test_mlp_different_activations(self):
        """Test MLP with different activations."""
        activations = ["relu", "tanh", "sigmoid"]
        
        for activation in activations:
            model = MLPBaseline(
                input_dim=1,
                hidden_dims=[16],
                output_dim=1,
                activation=activation
            )
            
            x = torch.tensor([[1.0]])
            output = model(x)
            
            assert output.shape == (1, 1)


class TestModelIntegration:
    """Test model integration and training."""
    
    def test_training_step(self):
        """Test a single training step."""
        # Create model
        ode_func = BasicODEFunc(hidden_dim=16)
        model = NeuralODE(ode_func=ode_func)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        x0 = torch.tensor([[0.0]])
        t = torch.linspace(0, 1, 10)
        target = torch.sin(t).unsqueeze(-1)
        
        output = model(x0, t)
        loss = criterion(output.squeeze(), target)
        
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0
        assert output.shape == (10, 1, 1)
    
    def test_model_parameters(self):
        """Test model parameter counting."""
        ode_func = BasicODEFunc(hidden_dim=32)
        model = NeuralODE(ode_func=ode_func)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable


if __name__ == "__main__":
    pytest.main([__file__])

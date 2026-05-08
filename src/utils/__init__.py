"""Utility functions for Neural ODEs project."""

import logging
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", "mps")
        
    Returns:
        PyTorch device object
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA GPU")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple Silicon GPU (MPS)")
        else:
            device = "cpu"
            print("Using CPU")
    
    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        filepath: Path to save checkpoint
        metadata: Additional metadata to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
) -> Tuple[int, float, Dict[str, Any]]:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model to load state into
        optimizer: Optimizer to load state into
        filepath: Path to checkpoint file
        
    Returns:
        Tuple of (epoch, loss, metadata)
    """
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"], checkpoint["metadata"]


def validate_config(config: DictConfig) -> None:
    """Validate configuration parameters.
    
    Args:
        config: Configuration object
        
    Raises:
        ValueError: If configuration is invalid
    """
    if config.seed < 0:
        raise ValueError("Seed must be non-negative")
    
    if config.data.train_ratio + config.data.val_ratio + config.data.test_ratio != 1.0:
        raise ValueError("Data split ratios must sum to 1.0")
    
    if config.training.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    
    if config.training.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")


def print_safety_disclaimer() -> None:
    """Print safety and ethics disclaimer."""
    print("\n" + "="*80)
    print("SAFETY AND ETHICS DISCLAIMER")
    print("="*80)
    print("This Neural ODE implementation is for RESEARCH and EDUCATIONAL purposes only.")
    print("It is NOT intended for production use or real-world decision making.")
    print("")
    print("Key limitations:")
    print("- Models may not generalize to unseen data distributions")
    print("- Continuous-time modeling assumptions may not hold in practice")
    print("- No guarantees about numerical stability or convergence")
    print("- Results should be interpreted with appropriate uncertainty")
    print("")
    print("Please use responsibly and consider ethical implications of your research.")
    print("="*80 + "\n")

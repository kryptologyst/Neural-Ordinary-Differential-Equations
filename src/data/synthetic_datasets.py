"""Synthetic datasets for Neural ODEs."""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig


class SyntheticSineDataset(Dataset):
    """Synthetic dataset for continuous-time function approximation.
    
    Generates synthetic data for testing Neural ODEs on function approximation tasks.
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        time_range: Tuple[float, float] = (0.0, 25.0),
        noise_level: float = 0.1,
        function_type: str = "sine",
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        """Initialize synthetic dataset.
        
        Args:
            n_samples: Total number of samples to generate
            time_range: Time range (start, end) for the function
            noise_level: Standard deviation of Gaussian noise
            function_type: Type of function ("sine", "cosine", "polynomial", "exponential")
            split: Data split ("train", "val", "test")
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.time_range = time_range
        self.noise_level = noise_level
        self.function_type = function_type
        self.split = split
        self.seed = seed
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Generate data
        self.times, self.values = self._generate_data()
        
        # Split data
        self._split_data(train_ratio, val_ratio, test_ratio)
    
    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic time series data.
        
        Returns:
            Tuple of (times, values) arrays
        """
        times = np.linspace(self.time_range[0], self.time_range[1], self.n_samples)
        
        if self.function_type == "sine":
            values = np.sin(times)
        elif self.function_type == "cosine":
            values = np.cos(times)
        elif self.function_type == "polynomial":
            # Cubic polynomial
            values = 0.1 * times**3 - 0.5 * times**2 + 0.3 * times
        elif self.function_type == "exponential":
            values = np.exp(-0.1 * times) * np.sin(times)
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")
        
        # Add noise
        noise = np.random.normal(0, self.noise_level, size=values.shape)
        values = values + noise
        
        return times, values
    
    def _split_data(self, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
        """Split data into train/val/test sets.
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
        """
        n_train = int(self.n_samples * train_ratio)
        n_val = int(self.n_samples * val_ratio)
        
        if self.split == "train":
            self.times = self.times[:n_train]
            self.values = self.values[:n_train]
        elif self.split == "val":
            self.times = self.times[n_train:n_train + n_val]
            self.values = self.values[n_train:n_train + n_val]
        elif self.split == "test":
            self.times = self.times[n_train + n_val:]
            self.values = self.values[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.times)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data point.
        
        Args:
            idx: Index of the data point
            
        Returns:
            Dictionary containing time and value tensors
        """
        return {
            "time": torch.tensor(self.times[idx], dtype=torch.float32),
            "value": torch.tensor(self.values[idx], dtype=torch.float32),
        }


class ContinuousTimeDataset(Dataset):
    """Dataset for continuous-time modeling with irregular sampling.
    
    Extends the basic synthetic dataset to support irregular time sampling
    and multiple trajectories.
    """
    
    def __init__(
        self,
        n_trajectories: int = 100,
        time_range: Tuple[float, float] = (0.0, 10.0),
        n_points_per_traj: int = 50,
        noise_level: float = 0.05,
        irregular_sampling: bool = True,
        split: str = "train",
        seed: int = 42,
    ):
        """Initialize continuous-time dataset.
        
        Args:
            n_trajectories: Number of independent trajectories
            time_range: Time range for each trajectory
            n_points_per_traj: Number of points per trajectory
            noise_level: Noise level for observations
            irregular_sampling: Whether to use irregular time sampling
            split: Data split
            seed: Random seed
        """
        self.n_trajectories = n_trajectories
        self.time_range = time_range
        self.n_points_per_traj = n_points_per_traj
        self.noise_level = noise_level
        self.irregular_sampling = irregular_sampling
        self.split = split
        self.seed = seed
        
        np.random.seed(seed)
        self.trajectories = self._generate_trajectories()
    
    def _generate_trajectories(self) -> List[Dict[str, np.ndarray]]:
        """Generate multiple trajectories.
        
        Returns:
            List of trajectory dictionaries
        """
        trajectories = []
        
        for i in range(self.n_trajectories):
            if self.irregular_sampling:
                # Irregular time sampling
                times = np.sort(np.random.uniform(
                    self.time_range[0], 
                    self.time_range[1], 
                    self.n_points_per_traj
                ))
            else:
                # Regular time sampling
                times = np.linspace(
                    self.time_range[0], 
                    self.time_range[1], 
                    self.n_points_per_traj
                )
            
            # Generate trajectory using a simple ODE: dy/dt = -0.5 * y + sin(t)
            # This has analytical solution: y(t) = C * exp(-0.5*t) + particular solution
            y0 = np.random.uniform(-2, 2)  # Random initial condition
            
            # Numerical integration (Euler method for simplicity)
            values = np.zeros_like(times)
            values[0] = y0
            
            for j in range(1, len(times)):
                dt = times[j] - times[j-1]
                values[j] = values[j-1] + dt * (-0.5 * values[j-1] + np.sin(times[j-1]))
            
            # Add noise
            noise = np.random.normal(0, self.noise_level, size=values.shape)
            values = values + noise
            
            trajectories.append({
                "times": times,
                "values": values,
                "trajectory_id": i,
            })
        
        return trajectories
    
    def __len__(self) -> int:
        """Return number of trajectories."""
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a trajectory.
        
        Args:
            idx: Trajectory index
            
        Returns:
            Dictionary containing trajectory data
        """
        traj = self.trajectories[idx]
        return {
            "times": torch.tensor(traj["times"], dtype=torch.float32),
            "values": torch.tensor(traj["values"], dtype=torch.float32),
            "trajectory_id": torch.tensor(traj["trajectory_id"], dtype=torch.long),
        }


def create_data_loaders(
    config: DictConfig,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train/val/test splits.
    
    Args:
        config: Configuration object
        device: PyTorch device
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = config.data._target_(
        split="train",
        seed=config.seed,
        **{k: v for k, v in config.data.items() if k not in ["_target_", "batch_size", "num_workers", "pin_memory"]}
    )
    
    val_dataset = config.data._target_(
        split="val",
        seed=config.seed + 1,  # Different seed for validation
        **{k: v for k, v in config.data.items() if k not in ["_target_", "batch_size", "num_workers", "pin_memory"]}
    )
    
    test_dataset = config.data._target_(
        split="test",
        seed=config.seed + 2,  # Different seed for test
        **{k: v for k, v in config.data.items() if k not in ["_target_", "batch_size", "num_workers", "pin_memory"]}
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and device.type == "cuda",
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and device.type == "cuda",
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and device.type == "cuda",
    )
    
    return train_loader, val_loader, test_loader

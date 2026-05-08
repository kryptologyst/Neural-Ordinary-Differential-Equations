"""Tests for data loading and preprocessing."""

import pytest
import torch
import numpy as np

from src.data import SyntheticSineDataset, ContinuousTimeDataset, create_data_loaders


class TestSyntheticSineDataset:
    """Test SyntheticSineDataset class."""
    
    def test_initialization(self):
        """Test dataset initialization."""
        dataset = SyntheticSineDataset(
            n_samples=100,
            time_range=(0, 10),
            noise_level=0.1,
            function_type="sine",
            split="train"
        )
        
        assert len(dataset) == 100
        assert dataset.function_type == "sine"
        assert dataset.noise_level == 0.1
    
    def test_data_generation(self):
        """Test data generation."""
        dataset = SyntheticSineDataset(
            n_samples=50,
            time_range=(0, 5),
            noise_level=0.05,
            function_type="sine",
            split="train"
        )
        
        # Check data shape
        sample = dataset[0]
        assert "time" in sample
        assert "value" in sample
        assert isinstance(sample["time"], torch.Tensor)
        assert isinstance(sample["value"], torch.Tensor)
    
    def test_different_function_types(self):
        """Test different function types."""
        function_types = ["sine", "cosine", "polynomial", "exponential"]
        
        for func_type in function_types:
            dataset = SyntheticSineDataset(
                n_samples=20,
                function_type=func_type,
                split="train"
            )
            
            assert len(dataset) == 20
            assert dataset.function_type == func_type
    
    def test_data_splits(self):
        """Test data splitting."""
        splits = ["train", "val", "test"]
        
        for split in splits:
            dataset = SyntheticSineDataset(
                n_samples=100,
                split=split,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2
            )
            
            assert len(dataset) > 0
            assert dataset.split == split
    
    def test_reproducibility(self):
        """Test data reproducibility with same seed."""
        dataset1 = SyntheticSineDataset(n_samples=50, seed=42, split="train")
        dataset2 = SyntheticSineDataset(n_samples=50, seed=42, split="train")
        
        # Check that datasets are identical
        for i in range(len(dataset1)):
            sample1 = dataset1[i]
            sample2 = dataset2[i]
            
            assert torch.allclose(sample1["time"], sample2["time"])
            assert torch.allclose(sample1["value"], sample2["value"])


class TestContinuousTimeDataset:
    """Test ContinuousTimeDataset class."""
    
    def test_initialization(self):
        """Test continuous time dataset initialization."""
        dataset = ContinuousTimeDataset(
            n_trajectories=10,
            time_range=(0, 5),
            n_points_per_traj=20,
            irregular_sampling=True,
            split="train"
        )
        
        assert len(dataset) == 10
        assert dataset.n_trajectories == 10
    
    def test_trajectory_generation(self):
        """Test trajectory generation."""
        dataset = ContinuousTimeDataset(
            n_trajectories=5,
            time_range=(0, 3),
            n_points_per_traj=15,
            split="train"
        )
        
        # Check trajectory structure
        trajectory = dataset[0]
        assert "times" in trajectory
        assert "values" in trajectory
        assert "trajectory_id" in trajectory
        
        assert len(trajectory["times"]) == 15
        assert len(trajectory["values"]) == 15
    
    def test_irregular_sampling(self):
        """Test irregular vs regular sampling."""
        # Regular sampling
        dataset_regular = ContinuousTimeDataset(
            n_trajectories=3,
            irregular_sampling=False,
            split="train"
        )
        
        # Irregular sampling
        dataset_irregular = ContinuousTimeDataset(
            n_trajectories=3,
            irregular_sampling=True,
            split="train"
        )
        
        # Check that irregular sampling produces different time points
        traj_regular = dataset_regular[0]
        traj_irregular = dataset_irregular[0]
        
        times_regular = traj_regular["times"].numpy()
        times_irregular = traj_irregular["times"].numpy()
        
        # Regular sampling should have evenly spaced times
        time_diffs_regular = np.diff(times_regular)
        assert np.allclose(time_diffs_regular, time_diffs_regular[0])
        
        # Irregular sampling should have varying time differences
        time_diffs_irregular = np.diff(times_irregular)
        assert not np.allclose(time_diffs_irregular, time_diffs_irregular[0])


class TestDataLoaders:
    """Test data loader creation."""
    
    def test_create_data_loaders(self):
        """Test data loader creation."""
        # Mock config
        class MockConfig:
            data = type('obj', (object,), {
                '_target_': SyntheticSineDataset,
                'n_samples': 100,
                'time_range': (0, 10),
                'noise_level': 0.1,
                'function_type': 'sine',
                'batch_size': 16,
                'num_workers': 0,
                'pin_memory': False,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
            })()
            seed = 42
        
        config = MockConfig()
        device = torch.device("cpu")
        
        train_loader, val_loader, test_loader = create_data_loaders(config, device)
        
        # Check that loaders are created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Check batch structure
        batch = next(iter(train_loader))
        assert "time" in batch
        assert "value" in batch
        assert batch["time"].shape[0] <= 16  # batch_size
        assert batch["value"].shape[0] <= 16  # batch_size


class TestDataConsistency:
    """Test data consistency and edge cases."""
    
    def test_empty_dataset(self):
        """Test edge case with very small dataset."""
        dataset = SyntheticSineDataset(
            n_samples=1,
            split="train",
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0
        )
        
        assert len(dataset) == 1
        sample = dataset[0]
        assert "time" in sample
        assert "value" in sample
    
    def test_zero_noise(self):
        """Test dataset with zero noise."""
        dataset = SyntheticSineDataset(
            n_samples=20,
            noise_level=0.0,
            function_type="sine",
            split="train"
        )
        
        # Check that values follow sine function exactly
        times = [dataset[i]["time"].item() for i in range(len(dataset))]
        values = [dataset[i]["value"].item() for i in range(len(dataset))]
        
        # Sort by time
        sorted_data = sorted(zip(times, values))
        times_sorted, values_sorted = zip(*sorted_data)
        
        # Check that values are close to sine function
        expected_values = np.sin(times_sorted)
        np.testing.assert_allclose(values_sorted, expected_values, atol=1e-6)
    
    def test_extreme_time_ranges(self):
        """Test extreme time ranges."""
        # Very small range
        dataset_small = SyntheticSineDataset(
            n_samples=10,
            time_range=(0, 0.1),
            split="train"
        )
        
        # Very large range
        dataset_large = SyntheticSineDataset(
            n_samples=10,
            time_range=(0, 1000),
            split="train"
        )
        
        assert len(dataset_small) == 10
        assert len(dataset_large) == 10
        
        # Check time ranges
        times_small = [dataset_small[i]["time"].item() for i in range(len(dataset_small))]
        times_large = [dataset_large[i]["time"].item() for i in range(len(dataset_large))]
        
        assert max(times_small) <= 0.1
        assert max(times_large) <= 1000


if __name__ == "__main__":
    pytest.main([__file__])

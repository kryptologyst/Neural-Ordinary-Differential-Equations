#!/usr/bin/env python3
"""Main training script for Neural ODEs project."""

import argparse
import logging
import os
from typing import Dict, Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.data import create_data_loaders
from src.eval import Evaluator
from src.models import (
    BasicODEFunc,
    NeuralODE,
    LinearBaseline,
    MLPBaseline,
    RNNBaseline,
)
from src.train import Trainer
from src.utils import (
    get_device,
    print_safety_disclaimer,
    set_seed,
    setup_logging,
    validate_config,
)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function.
    
    Args:
        config: Hydra configuration object
    """
    # Print safety disclaimer
    if config.safety_disclaimer:
        print_safety_disclaimer()
    
    # Setup logging
    logger = setup_logging(config.log_level)
    logger.info("Starting Neural ODEs training...")
    
    # Set random seed
    set_seed(config.seed)
    logger.info(f"Random seed set to {config.seed}")
    
    # Get device
    device = get_device(config.device)
    
    # Validate configuration
    validate_config(config)
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.output_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)
    logger.info(f"Configuration saved to {config_path}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config, device)
    
    # Create models
    logger.info("Creating models...")
    models = create_models(config, device)
    
    # Train and evaluate models
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            logger=logger,
        )
        
        # Train model
        history = trainer.train()
        
        # Create evaluator
        evaluator = Evaluator(
            model=model,
            test_loader=test_loader,
            config=config,
            device=device,
            logger=logger,
        )
        
        # Evaluate model
        eval_results = evaluator.evaluate()
        
        # Store results
        results[model_name] = {
            'model': model,
            'history': history,
            'eval_results': eval_results,
        }
        
        logger.info(f"Completed training and evaluation of {model_name}")
    
    # Create leaderboard
    logger.info("Creating leaderboard...")
    create_leaderboard(results, config, logger)
    
    logger.info("Training completed successfully!")


def create_models(config: DictConfig, device: torch.device) -> Dict[str, torch.nn.Module]:
    """Create models for training.
    
    Args:
        config: Configuration object
        device: PyTorch device
        
    Returns:
        Dictionary mapping model names to models
    """
    models = {}
    
    # Neural ODE
    ode_func = BasicODEFunc(
        hidden_dim=config.model.ode_func.hidden_dim,
        activation=config.model.ode_func.activation,
        dropout=config.model.ode_func.dropout,
    )
    
    neural_ode = NeuralODE(
        ode_func=ode_func,
        rtol=config.model.rtol,
        atol=config.model.atol,
        method=config.model.method,
        adjoint=config.model.adjoint,
    )
    
    models['neural_ode'] = neural_ode
    
    # Baseline models
    models['linear'] = LinearBaseline(input_dim=1, output_dim=1)
    models['mlp'] = MLPBaseline(
        input_dim=1,
        hidden_dims=[64, 32],
        output_dim=1,
        dropout=0.1,
    )
    models['rnn'] = RNNBaseline(
        input_dim=1,
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
        rnn_type="LSTM",
    )
    
    return models


def create_leaderboard(
    results: Dict[str, Any],
    config: DictConfig,
    logger: logging.Logger,
) -> None:
    """Create and save leaderboard.
    
    Args:
        results: Dictionary of model results
        config: Configuration object
        logger: Logger instance
    """
    # Extract metrics
    leaderboard_data = []
    
    for model_name, result in results.items():
        metrics = result['eval_results']['metrics']
        
        leaderboard_data.append({
            'Model': model_name,
            'MSE': metrics.get('mse', float('inf')),
            'MAE': metrics.get('mae', float('inf')),
            'RMSE': metrics.get('rmse', float('inf')),
            'R²': metrics.get('r2_score', float('-inf')),
            'MAPE': metrics.get('mape', float('inf')),
        })
    
    # Sort by MSE (lower is better)
    leaderboard_data.sort(key=lambda x: x['MSE'])
    
    # Print leaderboard
    print("\n" + "="*80)
    print("MODEL PERFORMANCE LEADERBOARD")
    print("="*80)
    print(f"{'Rank':<4} {'Model':<20} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'MAPE':<10}")
    print("-"*80)
    
    for rank, data in enumerate(leaderboard_data, 1):
        print(
            f"{rank:<4} {data['Model']:<20} "
            f"{data['MSE']:<10.6f} {data['MAE']:<10.6f} "
            f"{data['RMSE']:<10.6f} {data['R²']:<10.6f} {data['MAPE']:<10.6f}"
        )
    
    print("="*80)
    
    # Save leaderboard
    leaderboard_path = os.path.join(config.output_dir, 'leaderboard.txt')
    with open(leaderboard_path, 'w') as f:
        f.write("MODEL PERFORMANCE LEADERBOARD\n")
        f.write("="*80 + "\n")
        f.write(f"{'Rank':<4} {'Model':<20} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'MAPE':<10}\n")
        f.write("-"*80 + "\n")
        
        for rank, data in enumerate(leaderboard_data, 1):
            f.write(
                f"{rank:<4} {data['Model']:<20} "
                f"{data['MSE']:<10.6f} {data['MAE']:<10.6f} "
                f"{data['RMSE']:<10.6f} {data['R²']:<10.6f} {data['MAPE']:<10.6f}\n"
            )
        
        f.write("="*80 + "\n")
    
    logger.info(f"Leaderboard saved to {leaderboard_path}")


if __name__ == "__main__":
    main()

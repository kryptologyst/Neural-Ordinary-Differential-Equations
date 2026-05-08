#!/usr/bin/env python3
"""Script to run comprehensive evaluation of all models."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from data import create_data_loaders
from eval import Evaluator
from models import (
    BasicODEFunc,
    NeuralODE,
    AugmentedNeuralODE,
    LinearBaseline,
    MLPBaseline,
    RNNBaseline,
)
from train import Trainer
from utils import get_device, set_seed, setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Run comprehensive evaluation."""
    # Setup
    logger = setup_logging("INFO")
    set_seed(config.seed)
    device = get_device(config.device)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config, device)
    
    # Define models to evaluate
    models_to_evaluate = {
        "Linear": LinearBaseline(input_dim=1, output_dim=1),
        "MLP": MLPBaseline(input_dim=1, hidden_dims=[64, 32], output_dim=1),
        "RNN": RNNBaseline(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1),
        "Neural ODE": NeuralODE(
            ode_func=BasicODEFunc(hidden_dim=50, activation="tanh"),
            rtol=1e-3, atol=1e-4, method="dopri5"
        ),
        "Augmented Neural ODE": AugmentedNeuralODE(
            ode_func=BasicODEFunc(hidden_dim=50, activation="tanh"),
            input_dim=1, augmented_dim=1,
            rtol=1e-3, atol=1e-4, method="dopri5"
        ),
    }
    
    # Train and evaluate each model
    results = {}
    
    for model_name, model in models_to_evaluate.items():
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
        
        # Train
        history = trainer.train()
        
        # Evaluate
        evaluator = Evaluator(
            model=model,
            test_loader=test_loader,
            config=config,
            device=device,
            logger=logger,
        )
        
        eval_results = evaluator.evaluate()
        
        results[model_name] = {
            "history": history,
            "eval_results": eval_results,
        }
    
    # Create comprehensive report
    create_report(results, config, logger)


def create_report(results: dict, config: DictConfig, logger: logging.Logger) -> None:
    """Create comprehensive evaluation report."""
    logger.info("Creating evaluation report...")
    
    # Extract metrics
    metrics_data = []
    for model_name, result in results.items():
        metrics = result["eval_results"]["metrics"]
        metrics_data.append({
            "Model": model_name,
            "MSE": metrics.get("mse", float("inf")),
            "MAE": metrics.get("mae", float("inf")),
            "RMSE": metrics.get("rmse", float("inf")),
            "R²": metrics.get("r2_score", float("-inf")),
            "MAPE": metrics.get("mape", float("inf")),
        })
    
    # Sort by MSE
    metrics_data.sort(key=lambda x: x["MSE"])
    
    # Print leaderboard
    print("\n" + "="*100)
    print("COMPREHENSIVE MODEL EVALUATION LEADERBOARD")
    print("="*100)
    print(f"{'Rank':<4} {'Model':<25} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R²':<12} {'MAPE':<12}")
    print("-"*100)
    
    for rank, data in enumerate(metrics_data, 1):
        print(
            f"{rank:<4} {data['Model']:<25} "
            f"{data['MSE']:<12.6f} {data['MAE']:<12.6f} "
            f"{data['RMSE']:<12.6f} {data['R²']:<12.6f} {data['MAPE']:<12.6f}"
        )
    
    print("="*100)
    
    # Save detailed report
    report_path = os.path.join(config.output_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
        f.write("="*100 + "\n\n")
        
        f.write("LEADERBOARD\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Rank':<4} {'Model':<25} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R²':<12} {'MAPE':<12}\n")
        f.write("-"*100 + "\n")
        
        for rank, data in enumerate(metrics_data, 1):
            f.write(
                f"{rank:<4} {data['Model']:<25} "
                f"{data['MSE']:<12.6f} {data['MAE']:<12.6f} "
                f"{data['RMSE']:<12.6f} {data['R²']:<12.6f} {data['MAPE']:<12.6f}\n"
            )
        
        f.write("="*100 + "\n\n")
        
        # Detailed results for each model
        f.write("DETAILED RESULTS\n")
        f.write("-"*50 + "\n")
        
        for model_name, result in results.items():
            f.write(f"\n{model_name}:\n")
            f.write("-" * (len(model_name) + 1) + "\n")
            
            metrics = result["eval_results"]["metrics"]
            for metric_name, metric_value in metrics.items():
                f.write(f"  {metric_name}: {metric_value:.6f}\n")
            
            # Training history summary
            history = result["history"]
            final_train_loss = history["train_loss"][-1]
            final_val_loss = history["val_loss"][-1]
            f.write(f"  Final Train Loss: {final_train_loss:.6f}\n")
            f.write(f"  Final Val Loss: {final_val_loss:.6f}\n")
    
    logger.info(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    main()

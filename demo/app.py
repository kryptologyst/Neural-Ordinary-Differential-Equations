"""Streamlit demo application for Neural ODEs."""

import os
import tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchdiffeq import odeint

from src.models import BasicODEFunc, NeuralODE, LinearBaseline, MLPBaseline
from src.data import SyntheticSineDataset


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Neural ODEs Demo",
        page_icon="🧠",
        layout="wide",
    )
    
    # Title and description
    st.title("🧠 Neural Ordinary Differential Equations Demo")
    st.markdown("""
    This demo showcases Neural ODEs for continuous-time modeling and function approximation.
    
    **⚠️ Safety Notice**: This is a research/educational demo. Not for production use.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Dataset parameters
    st.sidebar.subheader("Dataset Parameters")
    function_type = st.sidebar.selectbox(
        "Function Type",
        ["sine", "cosine", "polynomial", "exponential"],
        index=0
    )
    n_samples = st.sidebar.slider("Number of Samples", 100, 2000, 1000)
    time_range = st.sidebar.slider("Time Range", 1.0, 50.0, 25.0)
    noise_level = st.sidebar.slider("Noise Level", 0.01, 0.5, 0.1)
    
    # Model parameters
    st.sidebar.subheader("Neural ODE Parameters")
    hidden_dim = st.sidebar.slider("Hidden Dimension", 10, 200, 50)
    activation = st.sidebar.selectbox("Activation", ["relu", "tanh", "sigmoid"], index=1)
    dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.0)
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    epochs = st.sidebar.slider("Epochs", 50, 1000, 500)
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
    batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
    
    # ODE solver parameters
    st.sidebar.subheader("ODE Solver Parameters")
    rtol = st.sidebar.slider("Relative Tolerance", 1e-5, 1e-1, 1e-3)
    atol = st.sidebar.slider("Absolute Tolerance", 1e-6, 1e-2, 1e-4)
    method = st.sidebar.selectbox("Solver Method", ["dopri5", "rk4", "euler"], index=0)
    
    # Generate data
    if st.sidebar.button("Generate New Data"):
        st.session_state.data_generated = True
    
    if not hasattr(st.session_state, 'data_generated'):
        st.session_state.data_generated = True
    
    if st.session_state.data_generated:
        # Generate synthetic data
        dataset = SyntheticSineDataset(
            n_samples=n_samples,
            time_range=(0.0, time_range),
            noise_level=noise_level,
            function_type=function_type,
            split="train",
            seed=42
        )
        
        # Convert to tensors
        times = torch.linspace(0.0, time_range, n_samples)
        values = torch.tensor([dataset[i]['value'] for i in range(len(dataset))])
        
        st.session_state.times = times
        st.session_state.values = values
        st.session_state.data_generated = False
    
    # Display data
    st.subheader("Generated Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Statistics:**")
        st.write(f"- Number of samples: {len(st.session_state.values)}")
        st.write(f"- Time range: [0, {time_range:.1f}]")
        st.write(f"- Value range: [{st.session_state.values.min():.3f}, {st.session_state.values.max():.3f}]")
        st.write(f"- Noise level: {noise_level:.3f}")
    
    with col2:
        # Plot data
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(st.session_state.times.numpy(), st.session_state.values.numpy(), 'b-', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'Generated {function_type} Function with Noise')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Train models
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            # Create models
            models = create_models(hidden_dim, activation, dropout, rtol, atol, method)
            
            # Train models
            results = train_models(models, st.session_state.times, st.session_state.values, 
                                epochs, learning_rate, batch_size)
            
            st.session_state.results = results
        
        st.success("Training completed!")
    
    # Display results
    if hasattr(st.session_state, 'results'):
        st.subheader("Model Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Metrics", "Training History", "Model Comparison"])
        
        with tab1:
            plot_predictions(st.session_state.results, st.session_state.times, st.session_state.values)
        
        with tab2:
            display_metrics(st.session_state.results)
        
        with tab3:
            plot_training_history(st.session_state.results)
        
        with tab4:
            plot_model_comparison(st.session_state.results, st.session_state.times, st.session_state.values)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Author**: [kryptologyst](https://github.com/kryptologyst) | 
    **GitHub**: https://github.com/kryptologyst
    
    This demo is for research and educational purposes only. Not for production use.
    """)


def create_models(
    hidden_dim: int,
    activation: str,
    dropout: float,
    rtol: float,
    atol: float,
    method: str,
) -> Dict[str, nn.Module]:
    """Create models for training.
    
    Args:
        hidden_dim: Hidden dimension for Neural ODE
        activation: Activation function
        dropout: Dropout probability
        rtol: Relative tolerance for ODE solver
        atol: Absolute tolerance for ODE solver
        method: ODE solver method
        
    Returns:
        Dictionary of models
    """
    models = {}
    
    # Neural ODE
    ode_func = BasicODEFunc(
        hidden_dim=hidden_dim,
        activation=activation,
        dropout=dropout,
    )
    
    neural_ode = NeuralODE(
        ode_func=ode_func,
        rtol=rtol,
        atol=atol,
        method=method,
        adjoint=True,
    )
    
    models['Neural ODE'] = neural_ode
    
    # Baseline models
    models['Linear'] = LinearBaseline(input_dim=1, output_dim=1)
    models['MLP'] = MLPBaseline(
        input_dim=1,
        hidden_dims=[hidden_dim, hidden_dim//2],
        output_dim=1,
        dropout=dropout,
    )
    
    return models


def train_models(
    models: Dict[str, nn.Module],
    times: torch.Tensor,
    values: torch.Tensor,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> Dict[str, Dict]:
    """Train models and return results.
    
    Args:
        models: Dictionary of models
        times: Time points
        values: Target values
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        
    Returns:
        Dictionary of training results
    """
    results = {}
    
    for model_name, model in models.items():
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training history
        train_losses = []
        
        # Train model
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            if model_name == 'Neural ODE':
                # Neural ODE forward pass
                x0 = values[0:1].unsqueeze(0)  # Initial condition
                predictions = model(x0, times)
                loss = criterion(predictions.squeeze(), values)
            else:
                # Regular neural network
                predictions = model(times.unsqueeze(-1))
                loss = criterion(predictions.squeeze(), values)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            if model_name == 'Neural ODE':
                x0 = values[0:1].unsqueeze(0)
                predictions = model(x0, times)
            else:
                predictions = model(times.unsqueeze(-1))
            
            # Compute metrics
            mse = criterion(predictions.squeeze(), values).item()
            mae = torch.mean(torch.abs(predictions.squeeze() - values)).item()
            rmse = torch.sqrt(torch.mean((predictions.squeeze() - values) ** 2)).item()
            
            # R² score
            ss_res = torch.sum((values - predictions.squeeze()) ** 2)
            ss_tot = torch.sum((values - torch.mean(values)) ** 2)
            r2 = 1 - ss_res / ss_tot
            
            results[model_name] = {
                'model': model,
                'predictions': predictions.squeeze(),
                'train_losses': train_losses,
                'metrics': {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R²': r2.item(),
                }
            }
    
    return results


def plot_predictions(results: Dict[str, Dict], times: torch.Tensor, values: torch.Tensor):
    """Plot model predictions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot ground truth
    ax.plot(times.numpy(), values.numpy(), 'b-', label='Ground Truth', alpha=0.8, linewidth=2)
    
    # Plot predictions for each model
    colors = ['red', 'green', 'orange', 'purple']
    for idx, (model_name, result) in enumerate(results.items()):
        predictions = result['predictions'].detach().numpy()
        ax.plot(times.numpy(), predictions, '--', 
                color=colors[idx % len(colors)], 
                label=f'{model_name}', alpha=0.8, linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Model Predictions vs Ground Truth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)


def display_metrics(results: Dict[str, Dict]):
    """Display model metrics."""
    # Create metrics table
    metrics_data = []
    for model_name, result in results.items():
        metrics = result['metrics']
        metrics_data.append({
            'Model': model_name,
            'MSE': f"{metrics['MSE']:.6f}",
            'MAE': f"{metrics['MAE']:.6f}",
            'RMSE': f"{metrics['RMSE']:.6f}",
            'R²': f"{metrics['R²']:.6f}",
        })
    
    # Sort by MSE
    metrics_data.sort(key=lambda x: float(x['MSE']))
    
    # Display table
    st.table(metrics_data)
    
    # Best model
    best_model = metrics_data[0]['Model']
    st.success(f"🏆 Best performing model: **{best_model}**")


def plot_training_history(results: Dict[str, Dict]):
    """Plot training history."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red', 'green', 'orange', 'purple']
    for idx, (model_name, result) in enumerate(results.items()):
        train_losses = result['train_losses']
        ax.plot(train_losses, color=colors[idx % len(colors)], 
                label=f'{model_name}', alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    st.pyplot(fig)


def plot_model_comparison(results: Dict[str, Dict], times: torch.Tensor, values: torch.Tensor):
    """Plot model comparison."""
    n_models = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot predictions for each model
    for idx, (model_name, result) in enumerate(results.items()):
        if idx < 4:  # Limit to 4 models
            row, col = idx // 2, idx % 2
            
            predictions = result['predictions'].detach().numpy()
            
            axes[row, col].plot(times.numpy(), values.numpy(), 'b-', 
                               label='Ground Truth', alpha=0.7)
            axes[row, col].plot(times.numpy(), predictions, 'r--', 
                               label='Predictions', alpha=0.7)
            axes[row, col].set_xlabel('Time')
            axes[row, col].set_ylabel('Value')
            axes[row, col].set_title(f'{model_name}')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, 4):
        row, col = idx // 2, idx % 2
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()

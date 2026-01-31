"""
Training pipeline with callbacks, early stopping, and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import callbacks
from typing import Dict, Optional, List, Tuple
import os
from datetime import datetime

import config
from model import create_model
TRAINING_CONFIG = config.TRAINING_CONFIG


def create_callbacks(
    config: dict = None,
    model_checkpoint_path: Optional[str] = None,
    tensorboard_log_dir: Optional[str] = None
) -> List[callbacks.Callback]:
    """
    Create training callbacks.
    
    Args:
        config: Training configuration dict (uses TRAINING_CONFIG if None)
        model_checkpoint_path: Path to save best model
        tensorboard_log_dir: Directory for TensorBoard logs
        
    Returns:
        List of Keras callbacks
    """
    config = config or TRAINING_CONFIG
    
    callback_list = []
    
    # Early stopping
    early_stopping = callbacks.EarlyStopping(
        monitor=config.get('early_stopping_monitor', 'val_loss'),
        patience=config.get('early_stopping_patience', 15),
        restore_best_weights=True,
        verbose=1
    )
    callback_list.append(early_stopping)
    
    # Reduce learning rate on plateau
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.get('reduce_lr_factor', 0.5),
        patience=config.get('reduce_lr_patience', 5),
        min_lr=config.get('reduce_lr_min_lr', 1e-7),
        verbose=1
    )
    callback_list.append(reduce_lr)
    
    # Model checkpoint (save best model)
    if model_checkpoint_path:
        os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=model_checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callback_list.append(model_checkpoint)
    
    # TensorBoard
    if tensorboard_log_dir:
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        tensorboard = callbacks.TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=1,
            write_graph=True
        )
        callback_list.append(tensorboard)
    
    return callback_list


def calculate_class_weights(y_train: np.ndarray, method: str = 'balanced') -> Optional[Dict[int, float]]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y_train: Training labels
        method: Method to calculate weights ('balanced', 'custom', or None)
        
    Returns:
        Dictionary mapping class to weight, or None
    """
    if method is None or method == 'none':
        return None
    
    if method == 'balanced':
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.array([0, 1])
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        return {cls: weight for cls, weight in zip(classes, weights)}
    
    elif method == 'custom':
        # Custom logic - you can modify this
        class_0_count = np.sum(y_train == 0)
        class_1_count = np.sum(y_train == 1)
        total = len(y_train)
        
        return {
            0: total / (2 * class_0_count) if class_0_count > 0 else 1.0,
            1: total / (2 * class_1_count) if class_1_count > 0 else 1.0
        }
    
    return None


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model: Optional[keras.Model] = None,
    config: dict = None,
    model_checkpoint_path: Optional[str] = None,
    tensorboard_log_dir: Optional[str] = None,
    verbose: int = None
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Train the neural network model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model: Pre-created model (creates new if None)
        config: Training configuration dict
        model_checkpoint_path: Path to save best model
        tensorboard_log_dir: Directory for TensorBoard logs
        verbose: Verbosity level (overrides config)
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    config = config or TRAINING_CONFIG
    
    # Create model if not provided
    if model is None:
        model = create_model(input_dim=X_train.shape[1])
    
    # Calculate class weights
    class_weight = calculate_class_weights(
        y_train,
        method=config.get('class_weight_method', 'balanced')
    )
    
    if class_weight:
        print(f"Using class weights: {class_weight}")
    
    # Create callbacks
    callback_list = create_callbacks(
        config=config,
        model_checkpoint_path=model_checkpoint_path,
        tensorboard_log_dir=tensorboard_log_dir
    )
    
    # Training parameters
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 32)
    verbose_level = verbose if verbose is not None else config.get('verbose', 1)
    
    # Train the model
    print(f"\nStarting training for {epochs} epochs with batch size {batch_size}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callback_list,
        verbose=verbose_level
    )
    
    print("\nTraining completed!")
    
    return model, history


def plot_training_history(
    history: keras.callbacks.History,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot training history (loss and metrics).
    
    Args:
        history: Keras training history object
        save_path: Path to save the plot (displays if None)
        figsize: Figure size
    """
    history_dict = history.history
    
    # Determine which metrics are available
    metrics = [key for key in history_dict.keys() if not key.startswith('val_')]
    
    # Create subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot training metric
        ax.plot(history_dict[metric], label=f'Training {metric}', linewidth=2)
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history_dict:
            ax.plot(history_dict[val_metric], label=f'Validation {metric}', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def get_training_summary(history: keras.callbacks.History) -> Dict:
    """
    Get summary statistics from training history.
    
    Args:
        history: Keras training history object
        
    Returns:
        Dictionary with training summary
    """
    history_dict = history.history
    
    summary = {
        'total_epochs': len(history_dict['loss']),
        'final_train_loss': float(history_dict['loss'][-1]),
        'final_val_loss': float(history_dict['val_loss'][-1]),
        'best_val_loss': float(min(history_dict['val_loss'])),
        'best_val_loss_epoch': int(np.argmin(history_dict['val_loss']) + 1),
    }
    
    # Add other metrics if available
    for metric in ['accuracy', 'precision', 'recall', 'auc']:
        if metric in history_dict:
            summary[f'final_train_{metric}'] = float(history_dict[metric][-1])
        if f'val_{metric}' in history_dict:
            summary[f'final_val_{metric}'] = float(history_dict[f'val_{metric}'][-1])
            summary[f'best_val_{metric}'] = float(max(history_dict[f'val_{metric}']))
            summary[f'best_val_{metric}_epoch'] = int(np.argmax(history_dict[f'val_{metric}']) + 1)
    
    return summary


def print_training_summary(history: keras.callbacks.History):
    """
    Print training summary.
    
    Args:
        history: Keras training history object
    """
    summary = get_training_summary(history)
    
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total epochs: {summary['total_epochs']}")
    print(f"Best validation loss: {summary['best_val_loss']:.4f} (epoch {summary['best_val_loss_epoch']})")
    print()
    
    print("Final Metrics:")
    print(f"  Train Loss: {summary['final_train_loss']:.4f}")
    print(f"  Val Loss:   {summary['final_val_loss']:.4f}")
    
    if 'final_train_accuracy' in summary:
        print(f"  Train Acc:  {summary['final_train_accuracy']:.4f}")
        print(f"  Val Acc:    {summary['final_val_accuracy']:.4f}")
    
    if 'final_train_precision' in summary:
        print(f"  Train Prec: {summary['final_train_precision']:.4f}")
        print(f"  Val Prec:   {summary['final_val_precision']:.4f}")
    
    if 'final_train_recall' in summary:
        print(f"  Train Rec:  {summary['final_train_recall']:.4f}")
        print(f"  Val Rec:    {summary['final_val_recall']:.4f}")
    
    if 'final_train_auc' in summary:
        print(f"  Train AUC:  {summary['final_train_auc']:.4f}")
        print(f"  Val AUC:    {summary['final_val_auc']:.4f}")
    
    print("=" * 60)


def create_training_run_name() -> str:
    """
    Create a unique name for the training run.
    
    Returns:
        Training run name with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}"

"""
Neural network model architecture using TensorFlow/Keras
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from typing import Optional, List, Dict
import numpy as np

import config
MODEL_CONFIG = config.MODEL_CONFIG


def create_model(
    input_dim: int,
    config: dict = None
) -> keras.Model:
    """
    Create a binary classification neural network model.
    
    Args:
        input_dim: Number of input features
        config: Model configuration dict (uses MODEL_CONFIG if None)
        
    Returns:
        Compiled Keras model
        
    Example:
        >>> model = create_model(input_dim=50)
        >>> model.summary()
    """
    config = config or MODEL_CONFIG
    
    # Extract configuration
    layer_sizes = config.get('layers', [128, 64, 32])
    dropout_rates = config.get('dropout_rates', [0.3, 0.2, 0.2])
    activation = config.get('activation', 'relu')
    output_activation = config.get('output_activation', 'sigmoid')
    learning_rate = config.get('learning_rate', 0.001)
    optimizer_name = config.get('optimizer', 'adam')
    
    # Ensure dropout_rates has same length as layer_sizes
    if len(dropout_rates) < len(layer_sizes):
        dropout_rates = dropout_rates + [dropout_rates[-1]] * (len(layer_sizes) - len(dropout_rates))
    
    # Build model
    model = models.Sequential(name='stock_prediction_model')
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,), name='input'))
    
    # Hidden layers with dropout
    for i, (units, dropout) in enumerate(zip(layer_sizes, dropout_rates)):
        model.add(layers.Dense(
            units,
            activation=activation,
            name=f'hidden_{i+1}'
        ))
        if dropout > 0:
            model.add(layers.Dropout(dropout, name=f'dropout_{i+1}'))
    
    # Output layer (binary classification)
    model.add(layers.Dense(1, activation=output_activation, name='output'))
    
    # Create optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def create_custom_model(
    input_dim: int,
    architecture: List[Dict],
    learning_rate: float = 0.001,
    optimizer_name: str = 'adam'
) -> keras.Model:
    """
    Create a custom model with specified architecture.
    
    Args:
        input_dim: Number of input features
        architecture: List of layer specifications, each dict with 'type', 'units', etc.
        learning_rate: Learning rate for optimizer
        optimizer_name: Name of optimizer ('adam', 'sgd', 'rmsprop')
        
    Returns:
        Compiled Keras model
        
    Example:
        >>> architecture = [
        ...     {'type': 'dense', 'units': 256, 'activation': 'relu'},
        ...     {'type': 'dropout', 'rate': 0.4},
        ...     {'type': 'dense', 'units': 128, 'activation': 'relu'},
        ...     {'type': 'dropout', 'rate': 0.3},
        ... ]
        >>> model = create_custom_model(50, architecture)
    """
    model = models.Sequential(name='custom_model')
    model.add(layers.Input(shape=(input_dim,)))
    
    for i, layer_spec in enumerate(architecture):
        layer_type = layer_spec.get('type', 'dense').lower()
        
        if layer_type == 'dense':
            model.add(layers.Dense(
                units=layer_spec.get('units', 64),
                activation=layer_spec.get('activation', 'relu'),
                name=f'dense_{i}'
            ))
        elif layer_type == 'dropout':
            model.add(layers.Dropout(
                rate=layer_spec.get('rate', 0.2),
                name=f'dropout_{i}'
            ))
        elif layer_type == 'batchnorm':
            model.add(layers.BatchNormalization(name=f'batchnorm_{i}'))
    
    # Add output layer
    model.add(layers.Dense(1, activation='sigmoid', name='output'))
    
    # Create optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    # Compile
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def get_model_summary_string(model: keras.Model) -> str:
    """
    Get model summary as a string.
    
    Args:
        model: Keras model
        
    Returns:
        String containing model summary
    """
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return '\n'.join(string_list)


def count_parameters(model: keras.Model) -> Dict[str, int]:
    """
    Count trainable and non-trainable parameters.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with parameter counts
    """
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    return {
        'trainable': int(trainable_count),
        'non_trainable': int(non_trainable_count),
        'total': int(trainable_count + non_trainable_count)
    }


def predict_proba(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Get prediction probabilities.
    
    Args:
        model: Trained Keras model
        X: Input features
        
    Returns:
        Array of probabilities for positive class
    """
    return model.predict(X, verbose=0).flatten()


def predict_class(
    model: keras.Model,
    X: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Get binary class predictions.
    
    Args:
        model: Trained Keras model
        X: Input features
        threshold: Decision threshold (default: 0.5)
        
    Returns:
        Array of binary predictions (0 or 1)
    """
    probabilities = predict_proba(model, X)
    return (probabilities >= threshold).astype(int)


def evaluate_model(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    verbose: int = 0
) -> Dict[str, float]:
    """
    Evaluate model on given data.
    
    Args:
        model: Trained Keras model
        X: Input features
        y: True labels
        batch_size: Batch size for evaluation
        verbose: Verbosity level
        
    Returns:
        Dictionary with evaluation metrics
    """
    results = model.evaluate(X, y, batch_size=batch_size, verbose=verbose, return_dict=True)
    return results


def print_model_info(model: keras.Model):
    """
    Print comprehensive model information.
    
    Args:
        model: Keras model
    """
    print("=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    model.summary()
    print()
    
    params = count_parameters(model)
    print("=" * 60)
    print("PARAMETER COUNT")
    print("=" * 60)
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    print(f"Total parameters: {params['total']:,}")
    print("=" * 60)

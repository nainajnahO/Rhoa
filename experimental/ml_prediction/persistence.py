"""
Model persistence: save and load trained models and preprocessing objects
"""

import os
import json
import pickle
from typing import Dict, Tuple, Optional, List
from tensorflow import keras
import warnings

import config
PERSISTENCE_CONFIG = config.PERSISTENCE_CONFIG


def save_model(
    model: keras.Model,
    model_path: str,
    save_format: str = None
):
    """
    Save trained Keras model.
    
    Args:
        model: Trained Keras model
        model_path: Path to save model
        save_format: Save format ('h5' or 'tf'), uses config if None
    """
    save_format = save_format or PERSISTENCE_CONFIG.get('model_save_format', 'h5')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if save_format == 'h5':
        if not model_path.endswith('.h5'):
            model_path = model_path + '.h5'
        model.save(model_path)
    elif save_format == 'tf':
        # SavedModel format
        model.save(model_path, save_format='tf')
    else:
        raise ValueError(f"Unknown save_format: {save_format}. Use 'h5' or 'tf'")
    
    print(f"Model saved to: {model_path}")


def load_model(model_path: str) -> keras.Model:
    """
    Load trained Keras model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    
    return model


def save_scaler(scaler, scaler_path: str):
    """
    Save fitted scaler using pickle.
    
    Args:
        scaler: Fitted scaler object (StandardScaler or MinMaxScaler)
        scaler_path: Path to save scaler
    """
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Scaler saved to: {scaler_path}")


def load_scaler(scaler_path: str):
    """
    Load fitted scaler.
    
    Args:
        scaler_path: Path to saved scaler
        
    Returns:
        Loaded scaler object
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Scaler loaded from: {scaler_path}")
    
    return scaler


def save_config(config: dict, config_path: str):
    """
    Save configuration as JSON.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config saved to: {config_path}")


def load_config(config_path: str) -> dict:
    """
    Load configuration from JSON.
    
    Args:
        config_path: Path to saved config
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Config loaded from: {config_path}")
    
    return config


def save_feature_columns(feature_columns: List[str], path: str):
    """
    Save list of feature column names.
    
    Args:
        feature_columns: List of feature names
        path: Path to save feature columns
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump({'feature_columns': feature_columns}, f, indent=2)
    
    print(f"Feature columns saved to: {path}")


def load_feature_columns(path: str) -> List[str]:
    """
    Load list of feature column names.
    
    Args:
        path: Path to saved feature columns
        
    Returns:
        List of feature names
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature columns not found at: {path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    feature_columns = data.get('feature_columns', [])
    print(f"Feature columns loaded from: {path} ({len(feature_columns)} features)")
    
    return feature_columns


def save_pipeline(
    model: keras.Model,
    scaler,
    feature_columns: List[str],
    base_path: str,
    feature_config: dict = None,
    model_config: dict = None,
    save_format: str = None
):
    """
    Save complete ML pipeline (model, scaler, configs).
    
    Args:
        model: Trained Keras model
        scaler: Fitted scaler
        feature_columns: List of feature column names
        base_path: Base directory path to save pipeline
        feature_config: Feature engineering configuration
        model_config: Model architecture configuration
        save_format: Model save format
    """
    os.makedirs(base_path, exist_ok=True)
    
    # Save model
    model_path = os.path.join(base_path, 'model.h5')
    save_model(model, model_path, save_format=save_format)
    
    # Save scaler
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    save_scaler(scaler, scaler_path)
    
    # Save feature columns
    features_path = os.path.join(base_path, 'feature_columns.json')
    save_feature_columns(feature_columns, features_path)
    
    # Save configs if provided
    if feature_config:
        feature_config_path = os.path.join(base_path, 'feature_config.json')
        save_config(feature_config, feature_config_path)
    
    if model_config:
        model_config_path = os.path.join(base_path, 'model_config.json')
        save_config(model_config, model_config_path)
    
    # Save metadata
    metadata = {
        'num_features': len(feature_columns),
        'save_format': save_format or PERSISTENCE_CONFIG.get('model_save_format', 'h5'),
        'scaler_type': type(scaler).__name__,
    }
    metadata_path = os.path.join(base_path, 'metadata.json')
    save_config(metadata, metadata_path)
    
    print(f"\nComplete pipeline saved to: {base_path}")


def load_pipeline(base_path: str) -> Dict:
    """
    Load complete ML pipeline.
    
    Args:
        base_path: Base directory path where pipeline was saved
        
    Returns:
        Dictionary containing:
            - 'model': Loaded Keras model
            - 'scaler': Loaded scaler
            - 'feature_columns': List of feature names
            - 'feature_config': Feature configuration (if available)
            - 'model_config': Model configuration (if available)
            - 'metadata': Pipeline metadata
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Pipeline not found at: {base_path}")
    
    pipeline = {}
    
    # Load model
    model_path = os.path.join(base_path, 'model.h5')
    if not os.path.exists(model_path):
        # Try TF SavedModel format
        model_path = base_path
    pipeline['model'] = load_model(model_path)
    
    # Load scaler
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    pipeline['scaler'] = load_scaler(scaler_path)
    
    # Load feature columns
    features_path = os.path.join(base_path, 'feature_columns.json')
    pipeline['feature_columns'] = load_feature_columns(features_path)
    
    # Load configs if available
    feature_config_path = os.path.join(base_path, 'feature_config.json')
    if os.path.exists(feature_config_path):
        pipeline['feature_config'] = load_config(feature_config_path)
    else:
        pipeline['feature_config'] = None
    
    model_config_path = os.path.join(base_path, 'model_config.json')
    if os.path.exists(model_config_path):
        pipeline['model_config'] = load_config(model_config_path)
    else:
        pipeline['model_config'] = None
    
    # Load metadata
    metadata_path = os.path.join(base_path, 'metadata.json')
    if os.path.exists(metadata_path):
        pipeline['metadata'] = load_config(metadata_path)
    else:
        pipeline['metadata'] = None
    
    print(f"\nComplete pipeline loaded from: {base_path}")
    
    return pipeline


def predict_with_pipeline(
    pipeline: Dict,
    X_new,
    threshold: float = 0.5
) -> Tuple:
    """
    Make predictions using loaded pipeline.
    
    Args:
        pipeline: Dictionary from load_pipeline
        X_new: New data (DataFrame or array) with same features as training
        threshold: Classification threshold
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    import pandas as pd
    import numpy as np
    
    model = pipeline['model']
    scaler = pipeline['scaler']
    feature_columns = pipeline['feature_columns']
    
    # Handle DataFrame input
    if isinstance(X_new, pd.DataFrame):
        # Ensure we have all required features
        missing_features = set(feature_columns) - set(X_new.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        X_new = X_new[feature_columns].values
    
    # Scale features
    X_scaled = scaler.transform(X_new)
    
    # Predict
    probabilities = model.predict(X_scaled, verbose=0).flatten()
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions, probabilities


def save_training_history(history: keras.callbacks.History, path: str):
    """
    Save training history to JSON.
    
    Args:
        history: Keras training history object
        path: Path to save history
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert history to serializable format
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]
    
    with open(path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved to: {path}")


def load_training_history(path: str) -> dict:
    """
    Load training history from JSON.
    
    Args:
        path: Path to saved history
        
    Returns:
        Dictionary with training history
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training history not found at: {path}")
    
    with open(path, 'r') as f:
        history_dict = json.load(f)
    
    print(f"Training history loaded from: {path}")
    
    return history_dict


def list_saved_models(base_dir: str = None) -> List[str]:
    """
    List all saved models in the base directory.
    
    Args:
        base_dir: Base directory to search (uses config if None)
        
    Returns:
        List of model directory paths
    """
    base_dir = base_dir or PERSISTENCE_CONFIG.get('model_dir', 'experimental/ml_prediction/saved_models')
    
    if not os.path.exists(base_dir):
        print(f"No saved models found in: {base_dir}")
        return []
    
    models = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains model files
            has_model = any(
                os.path.exists(os.path.join(item_path, f))
                for f in ['model.h5', 'scaler.pkl', 'feature_columns.json']
            )
            if has_model:
                models.append(item_path)
    
    return models

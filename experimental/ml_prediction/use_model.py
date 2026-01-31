"""
Simple script to load and use the high-precision model (100% precision at threshold 0.67)
"""
import os
import sys

# Add parent directory to path for rhoa imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from persistence import load_pipeline
from feature_engineering import create_features
from target_generation import generate_targets

# Path to the high-precision model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_models/high_precision_model')


def load_model(model_path=MODEL_PATH):
    """
    Load the high-precision model.
    
    Returns:
        Dictionary containing model, scaler, feature_columns, and metadata
    """
    return load_pipeline(model_path)


def predict(model_pipeline, df, threshold=0.67):
    """
    Make predictions on new stock data.
    
    Args:
        model_pipeline: Dictionary from load_model()
        df: DataFrame with OHLCV data (Open, High, Low, Close, Volume)
        threshold: Classification threshold (default 0.67 for 100% precision)
        
    Returns:
        Tuple of (predictions, probabilities)
        - predictions: Binary array (0 or 1)
        - probabilities: Probability scores (0 to 1)
    """
    # Generate features
    df_features = create_features(df)
    
    # Get feature columns
    feature_columns = model_pipeline['feature_columns']
    
    # Select features
    X = df_features[feature_columns].values
    
    # Scale features
    X_scaled = model_pipeline['scaler'].transform(X)
    
    # Make predictions
    probabilities = model_pipeline['model'].predict(X_scaled, verbose=0).flatten()
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions, probabilities


def evaluate(model_pipeline, df, threshold_percent=5.0, horizon_days=42, pred_threshold=0.67):
    """
    Evaluate model on data with known outcomes.
    
    Args:
        model_pipeline: Dictionary from load_model()
        df: DataFrame with OHLCV data
        threshold_percent: Minimum return percentage to consider as buy signal (default 5.0%)
        horizon_days: Days forward to measure return (default 42)
        pred_threshold: Prediction threshold (default 0.67)
        
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import confusion_matrix, precision_score, recall_score
    
    # Generate features and target
    df_features = create_features(df)
    df_with_target = generate_targets(df_features, 
                                      threshold_percent=threshold_percent,
                                      horizon_days=horizon_days)
    
    # Remove NaN rows
    df_clean = df_with_target.dropna()
    
    # Get features and target
    feature_columns = model_pipeline['feature_columns']
    X = df_clean[feature_columns].values
    y_true = df_clean['target'].values
    
    # Make predictions
    predictions, probabilities = predict(model_pipeline, df_clean, threshold=pred_threshold)
    
    # Calculate metrics
    cm = confusion_matrix(y_true, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    
    return {
        'confusion_matrix': cm,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'precision': precision,
        'recall': recall,
        'total_signals': tp + fp,
        'threshold': pred_threshold
    }


def main():
    """
    Example usage of the high-precision model.
    """
    print("="*80)
    print("HIGH-PRECISION STOCK PREDICTION MODEL")
    print("="*80)
    
    # Load model
    print("\n### Loading Model ###")
    model_pipeline = load_model()
    print(f"✓ Model loaded: {len(model_pipeline['feature_columns'])} features")
    
    # Load example data
    print("\n### Loading Example Data ###")
    data_path = os.path.join(os.path.dirname(__file__), '../../tests/data.csv')
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} rows")
    
    # Evaluate on example data
    print("\n### Evaluating Model ###")
    results = evaluate(model_pipeline, df, pred_threshold=0.67)
    
    print(f"\nResults at threshold {results['threshold']}:")
    print(f"  Precision:  {results['precision']:.1%}")
    print(f"  Recall:     {results['recall']:.1%}")
    print(f"  Signals:    {results['total_signals']}")
    print(f"  Correct:    {results['true_positives']}")
    print(f"  Wrong:      {results['false_positives']}")
    
    print("\nConfusion Matrix:")
    print(f"  TN: {results['true_negatives']:3d}  |  FP: {results['false_positives']:3d}")
    print(f"  FN: {results['false_negatives']:3d}  |  TP: {results['true_positives']:3d}")
    
    # Try different thresholds
    print("\n### Testing Different Thresholds ###")
    for threshold in [0.5, 0.6, 0.67]:
        results = evaluate(model_pipeline, df, pred_threshold=threshold)
        print(f"Threshold {threshold:.2f}: Precision {results['precision']:5.1%}, "
              f"Recall {results['recall']:5.1%}, Signals {results['total_signals']:2d} "
              f"({results['true_positives']} correct)")
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()

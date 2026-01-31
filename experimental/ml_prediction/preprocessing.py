"""
Data preprocessing, normalization, and train/validation/test splitting
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

import config
PREPROCESSING_CONFIG = config.PREPROCESSING_CONFIG
SPLIT_CONFIG = config.SPLIT_CONFIG


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'drop',
    min_valid_rows: int = None
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: DataFrame with features
        method: Method to handle NaN ('drop', 'forward_fill', 'interpolate')
        min_valid_rows: Minimum number of valid rows required
        
    Returns:
        DataFrame with missing values handled
        
    Raises:
        ValueError: If insufficient valid rows after handling NaN
    """
    config = PREPROCESSING_CONFIG
    method = method or config.get('handle_nan', 'drop')
    min_valid_rows = min_valid_rows or config.get('min_valid_rows', 250)
    
    df = df.copy()
    initial_rows = len(df)
    
    if method == 'drop':
        df = df.dropna()
    elif method == 'forward_fill':
        df = df.fillna(method='ffill')
        df = df.dropna()  # Drop any remaining NaN at the beginning
    elif method == 'interpolate':
        df = df.interpolate(method='linear')
        df = df.dropna()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'drop', 'forward_fill', or 'interpolate'")
    
    remaining_rows = len(df)
    
    if remaining_rows < min_valid_rows:
        raise ValueError(
            f"Insufficient valid rows after handling NaN: {remaining_rows} < {min_valid_rows}. "
            f"Lost {initial_rows - remaining_rows} rows."
        )
    
    if remaining_rows < initial_rows:
        warnings.warn(
            f"Dropped {initial_rows - remaining_rows} rows with NaN values. "
            f"Remaining: {remaining_rows} rows.",
            UserWarning
        )
    
    return df


def create_scaler(scaler_type: str = 'standard'):
    """
    Create a scaler object for feature normalization.
    
    Args:
        scaler_type: Type of scaler ('standard' or 'minmax')
        
    Returns:
        Scaler object (StandardScaler or MinMaxScaler)
    """
    if scaler_type == 'standard':
        return StandardScaler()
    elif scaler_type == 'minmax':
        return MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}. Use 'standard' or 'minmax'")


def scale_features(
    X: pd.DataFrame,
    scaler=None,
    fit: bool = True
) -> Tuple[pd.DataFrame, object]:
    """
    Scale features using the provided or created scaler.
    
    Args:
        X: DataFrame with features to scale
        scaler: Pre-fitted scaler (if None, creates new one)
        fit: Whether to fit the scaler (True for training, False for test)
        
    Returns:
        Tuple of (scaled_features_df, fitted_scaler)
    """
    if scaler is None:
        scaler = create_scaler(PREPROCESSING_CONFIG.get('scaler_type', 'standard'))
    
    if fit:
        scaled_values = scaler.fit_transform(X)
    else:
        scaled_values = scaler.transform(X)
    
    # Create DataFrame with same index and columns
    X_scaled = pd.DataFrame(
        scaled_values,
        index=X.index,
        columns=X.columns
    )
    
    return X_scaled, scaler


def time_series_split(
    df: pd.DataFrame,
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets respecting time order.
    
    Args:
        df: DataFrame to split
        train_ratio: Proportion for training (default from SPLIT_CONFIG)
        val_ratio: Proportion for validation (default from SPLIT_CONFIG)
        test_ratio: Proportion for test (default from SPLIT_CONFIG)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
        
    Example:
        >>> df = pd.DataFrame({'feature': range(1000), 'target': range(1000)})
        >>> train, val, test = time_series_split(df, 0.7, 0.15, 0.15)
        >>> # train: 0-699, val: 700-849, test: 850-999
    """
    config = SPLIT_CONFIG
    train_ratio = train_ratio or config.get('train_ratio', 0.70)
    val_ratio = val_ratio or config.get('val_ratio', 0.15)
    test_ratio = test_ratio or config.get('test_ratio', 0.15)
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df


def prepare_ml_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = 'target',
    do_scaling: bool = True,
    scaler=None
) -> Tuple[pd.DataFrame, pd.Series, Optional[object]]:
    """
    Prepare features and target for ML model.
    
    Args:
        df: DataFrame with features and target
        feature_columns: List of feature column names
        target_column: Name of target column
        do_scaling: Whether to scale features
        scaler: Pre-fitted scaler (None to create new one)
        
    Returns:
        Tuple of (X_scaled, y, scaler) where:
            - X_scaled: Scaled feature DataFrame
            - y: Target Series
            - scaler: Fitted scaler object (or None if do_scaling=False)
    """
    # Extract features and target
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # Scale features if requested
    if do_scaling:
        X_scaled, scaler = scale_features(X, scaler=scaler, fit=(scaler is None))
        return X_scaled, y, scaler
    else:
        return X, y, None


def preprocess_pipeline(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = 'target',
    handle_nan_method: str = None,
    scaler_type: str = None,
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None
) -> dict:
    """
    Complete preprocessing pipeline: handle NaN, split, and scale.
    
    Args:
        df: DataFrame with features and target
        feature_columns: List of feature column names
        target_column: Name of target column
        handle_nan_method: Method to handle missing values
        scaler_type: Type of scaler to use
        train_ratio: Training set proportion
        val_ratio: Validation set proportion
        test_ratio: Test set proportion
        
    Returns:
        Dictionary containing:
            - 'X_train', 'y_train': Training features and labels
            - 'X_val', 'y_val': Validation features and labels
            - 'X_test', 'y_test': Test features and labels
            - 'scaler': Fitted scaler object
            - 'feature_columns': List of feature names
            - 'stats': Dictionary with split statistics
    """
    # Handle missing values
    df_clean = handle_missing_values(df, method=handle_nan_method)
    
    # Split into train/val/test
    train_df, val_df, test_df = time_series_split(
        df_clean,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # Prepare training data (fit scaler)
    X_train, y_train, scaler = prepare_ml_data(
        train_df,
        feature_columns=feature_columns,
        target_column=target_column,
        do_scaling=True,
        scaler=None
    )
    
    # Prepare validation data (use fitted scaler)
    X_val, y_val, _ = prepare_ml_data(
        val_df,
        feature_columns=feature_columns,
        target_column=target_column,
        do_scaling=True,
        scaler=scaler
    )
    
    # Prepare test data (use fitted scaler)
    X_test, y_test, _ = prepare_ml_data(
        test_df,
        feature_columns=feature_columns,
        target_column=target_column,
        do_scaling=True,
        scaler=scaler
    )
    
    # Compute statistics
    stats = {
        'total_samples': len(df_clean),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'num_features': len(feature_columns),
        'train_class_0': int((y_train == 0).sum()),
        'train_class_1': int((y_train == 1).sum()),
        'val_class_0': int((y_val == 0).sum()),
        'val_class_1': int((y_val == 1).sum()),
        'test_class_0': int((y_test == 0).sum()),
        'test_class_1': int((y_test == 1).sum()),
    }
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'stats': stats
    }


def get_class_weights_dict(y: pd.Series) -> dict:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Target labels (0 or 1)
        
    Returns:
        Dictionary mapping class labels to weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.array([0, 1])
    weights = compute_class_weight('balanced', classes=classes, y=y.values)
    
    return {cls: weight for cls, weight in zip(classes, weights)}


def print_preprocessing_summary(stats: dict):
    """
    Print a summary of preprocessing results.
    
    Args:
        stats: Statistics dictionary from preprocess_pipeline
    """
    print("=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Number of features: {stats['num_features']}")
    print()
    print(f"Train set: {stats['train_samples']} samples")
    print(f"  - Class 0: {stats['train_class_0']} ({stats['train_class_0']/stats['train_samples']*100:.1f}%)")
    print(f"  - Class 1: {stats['train_class_1']} ({stats['train_class_1']/stats['train_samples']*100:.1f}%)")
    print()
    print(f"Validation set: {stats['val_samples']} samples")
    print(f"  - Class 0: {stats['val_class_0']} ({stats['val_class_0']/stats['val_samples']*100:.1f}%)")
    print(f"  - Class 1: {stats['val_class_1']} ({stats['val_class_1']/stats['val_samples']*100:.1f}%)")
    print()
    print(f"Test set: {stats['test_samples']} samples")
    print(f"  - Class 0: {stats['test_class_0']} ({stats['test_class_0']/stats['test_samples']*100:.1f}%)")
    print(f"  - Class 1: {stats['test_class_1']} ({stats['test_class_1']/stats['test_samples']*100:.1f}%)")
    print("=" * 60)

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


def time_series_split_multi_stock(
    df: pd.DataFrame,
    stock_col: str = 'stock_id',
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None,
    holdout_stocks: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split multi-stock data into train/validation/test sets respecting time order within each stock.
    
    This function ensures:
    1. Each stock is split in chronological order (no data leakage within stock)
    2. Optional: Some stocks can be held out entirely for testing (cross-stock generalization)
    3. Class distribution is preserved across splits
    
    Args:
        df: DataFrame with multi-stock data (must have stock_col column)
        stock_col: Name of the column containing stock identifiers
        train_ratio: Proportion for training (default from SPLIT_CONFIG)
        val_ratio: Proportion for validation (default from SPLIT_CONFIG)
        test_ratio: Proportion for test (default from SPLIT_CONFIG)
        holdout_stocks: Optional list of stock IDs to reserve entirely for test set
        
    Returns:
        Tuple of (train_df, val_df, test_df)
        
    Example:
        >>> # Each stock split 70/15/15
        >>> train, val, test = time_series_split_multi_stock(df, 'stock_id')
        
        >>> # Hold out specific stocks for testing
        >>> train, val, test = time_series_split_multi_stock(
        ...     df, 'stock_id', holdout_stocks=['AAPL', 'MSFT']
        ... )
    """
    config = SPLIT_CONFIG
    train_ratio = train_ratio or config.get('train_ratio', 0.70)
    val_ratio = val_ratio or config.get('val_ratio', 0.15)
    test_ratio = test_ratio or config.get('test_ratio', 0.15)
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Check if stock_col exists
    if stock_col not in df.columns:
        raise ValueError(f"Column '{stock_col}' not found in DataFrame")
    
    # Get unique stocks
    all_stocks = df[stock_col].unique()
    
    # Handle holdout stocks
    if holdout_stocks:
        holdout_set = set(holdout_stocks)
        training_stocks = [s for s in all_stocks if s not in holdout_set]
        
        # Validate holdout stocks exist
        missing_stocks = holdout_set - set(all_stocks)
        if missing_stocks:
            raise ValueError(f"Holdout stocks not found in data: {missing_stocks}")
        
        print(f"Using {len(training_stocks)} stocks for train/val, {len(holdout_stocks)} stocks held out for test")
        
        # Split training stocks normally
        train_dfs = []
        val_dfs = []
        
        for stock in training_stocks:
            stock_df = df[df[stock_col] == stock].copy()
            n = len(stock_df)
            train_end = int(n * train_ratio / (train_ratio + val_ratio))
            
            train_dfs.append(stock_df.iloc[:train_end])
            val_dfs.append(stock_df.iloc[train_end:])
        
        # All holdout stocks go to test
        test_dfs = [df[df[stock_col] == stock].copy() for stock in holdout_stocks]
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
    else:
        # Split each stock according to ratios
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for stock in all_stocks:
            stock_df = df[df[stock_col] == stock].copy()
            n = len(stock_df)
            
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_dfs.append(stock_df.iloc[:train_end])
            val_dfs.append(stock_df.iloc[train_end:val_end])
            test_dfs.append(stock_df.iloc[val_end:])
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Print split summary
    print(f"\nMulti-stock split summary:")
    print(f"  Train: {len(train_df)} samples from {train_df[stock_col].nunique()} stocks")
    print(f"  Val:   {len(val_df)} samples from {val_df[stock_col].nunique()} stocks")
    print(f"  Test:  {len(test_df)} samples from {test_df[stock_col].nunique()} stocks")
    
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
    test_ratio: float = None,
    stock_col: Optional[str] = None,
    holdout_stocks: Optional[List[str]] = None
) -> dict:
    """
    Complete preprocessing pipeline: handle NaN, split, and scale.
    
    Supports both single-stock and multi-stock data. If stock_col is provided,
    uses multi-stock splitting strategy.
    
    Args:
        df: DataFrame with features and target
        feature_columns: List of feature column names
        target_column: Name of target column
        handle_nan_method: Method to handle missing values
        scaler_type: Type of scaler to use
        train_ratio: Training set proportion
        val_ratio: Validation set proportion
        test_ratio: Test set proportion
        stock_col: Optional stock identifier column (for multi-stock data)
        holdout_stocks: Optional list of stocks to reserve entirely for testing
        
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
    
    # Split into train/val/test (multi-stock or single-stock)
    if stock_col and stock_col in df_clean.columns:
        train_df, val_df, test_df = time_series_split_multi_stock(
            df_clean,
            stock_col=stock_col,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            holdout_stocks=holdout_stocks
        )
    else:
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
    
    # Add stock information if available
    if stock_col and stock_col in df_clean.columns:
        stats['num_stocks'] = df_clean[stock_col].nunique()
        stats['train_stocks'] = train_df[stock_col].nunique()
        stats['val_stocks'] = val_df[stock_col].nunique()
        stats['test_stocks'] = test_df[stock_col].nunique()
    
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

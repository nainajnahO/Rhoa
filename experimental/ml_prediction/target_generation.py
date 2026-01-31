"""
Target generation for stock return prediction
Generates binary classification labels based on forward returns
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

import config
TARGET_HORIZON_DAYS = config.TARGET_HORIZON_DAYS
RETURN_THRESHOLD_PERCENT = config.RETURN_THRESHOLD_PERCENT


def calculate_forward_returns(
    prices: pd.Series,
    horizon_days: int = TARGET_HORIZON_DAYS
) -> pd.Series:
    """
    Calculate forward returns over a specified horizon.
    
    Args:
        prices: Series of stock prices (typically Close prices)
        horizon_days: Number of days to look forward (default: 42 for ~2 months)
        
    Returns:
        Series containing percentage returns (last N rows will be NaN)
        
    Example:
        >>> prices = pd.Series([100, 102, 105, 110, 115])
        >>> forward_returns = calculate_forward_returns(prices, horizon_days=2)
        >>> # forward_returns[0] = (105 - 100) / 100 * 100 = 5.0%
    """
    # Shift prices backward to get future prices
    future_prices = prices.shift(-horizon_days)
    
    # Calculate percentage return
    returns = ((future_prices - prices) / prices) * 100
    
    return returns


def create_binary_labels(
    returns: pd.Series,
    threshold_percent: float = RETURN_THRESHOLD_PERCENT
) -> pd.Series:
    """
    Create binary classification labels based on return threshold.
    
    Args:
        returns: Series of percentage returns
        threshold_percent: Threshold for positive class (default: 5.0%)
        
    Returns:
        Series with binary labels:
            0: Return < threshold (don't buy)
            1: Return >= threshold (buy signal)
            
    Example:
        >>> returns = pd.Series([2.0, 7.5, -3.0, 5.0, 10.0])
        >>> labels = create_binary_labels(returns, threshold_percent=5.0)
        >>> # labels = [0, 1, 0, 1, 1]
    """
    labels = (returns >= threshold_percent).astype(int)
    return labels


def generate_targets(
    df: pd.DataFrame,
    price_column: str = 'Close',
    horizon_days: int = TARGET_HORIZON_DAYS,
    threshold_percent: float = RETURN_THRESHOLD_PERCENT,
    add_returns: bool = True
) -> pd.DataFrame:
    """
    Generate target labels and optionally add return values to DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        price_column: Name of the price column to use (default: 'Close')
        horizon_days: Forward-looking horizon in days (default: 42)
        threshold_percent: Return threshold for classification (default: 5.0%)
        add_returns: Whether to add the raw return values to the DataFrame
        
    Returns:
        DataFrame with added columns:
            - 'forward_return': Percentage returns (if add_returns=True)
            - 'target': Binary classification labels (0 or 1)
            
    Example:
        >>> df = pd.DataFrame({'Close': [100, 102, 105, 108, 110]})
        >>> df_with_targets = generate_targets(df)
        >>> # df_with_targets will have 'forward_return' and 'target' columns
    """
    df = df.copy()
    
    # Calculate forward returns
    forward_returns = calculate_forward_returns(
        df[price_column],
        horizon_days=horizon_days
    )
    
    # Create binary labels
    labels = create_binary_labels(
        forward_returns,
        threshold_percent=threshold_percent
    )
    
    # Add to dataframe
    if add_returns:
        df['forward_return'] = forward_returns
    df['target'] = labels
    
    return df


def get_class_distribution(labels: pd.Series) -> dict:
    """
    Get the distribution of classes in the target labels.
    
    Args:
        labels: Series of binary labels (0 or 1)
        
    Returns:
        Dictionary with class counts and percentages
        
    Example:
        >>> labels = pd.Series([0, 0, 1, 0, 1, 1, 1])
        >>> dist = get_class_distribution(labels)
        >>> # {'class_0': 3, 'class_1': 4, 'class_0_pct': 42.86, 'class_1_pct': 57.14}
    """
    valid_labels = labels.dropna()
    
    class_counts = valid_labels.value_counts().sort_index()
    total = len(valid_labels)
    
    distribution = {}
    for cls in [0, 1]:
        count = class_counts.get(cls, 0)
        distribution[f'class_{cls}'] = int(count)
        distribution[f'class_{cls}_pct'] = (count / total * 100) if total > 0 else 0
    
    distribution['total'] = int(total)
    distribution['imbalance_ratio'] = (
        distribution['class_1'] / distribution['class_0']
        if distribution['class_0'] > 0 else float('inf')
    )
    
    return distribution


def calculate_class_weights(labels: pd.Series) -> dict:
    """
    Calculate balanced class weights for training.
    
    Args:
        labels: Series of binary labels (0 or 1)
        
    Returns:
        Dictionary mapping class labels to weights
        
    Example:
        >>> labels = pd.Series([0, 0, 0, 1, 1])  # 60% class 0, 40% class 1
        >>> weights = calculate_class_weights(labels)
        >>> # weights = {0: 0.833, 1: 1.25} - upweight minority class
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    valid_labels = labels.dropna()
    
    classes = np.array([0, 1])
    weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=valid_labels.values
    )
    
    return {cls: weight for cls, weight in zip(classes, weights)}


def split_with_targets(
    df: pd.DataFrame,
    target_col: str = 'target'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into rows with valid targets and rows without.
    
    Args:
        df: DataFrame with target column
        target_col: Name of the target column
        
    Returns:
        Tuple of (valid_df, invalid_df) where:
            - valid_df: Rows with non-NaN targets (for training/testing)
            - invalid_df: Rows with NaN targets (most recent data)
    """
    valid_mask = df[target_col].notna()
    
    valid_df = df[valid_mask].copy()
    invalid_df = df[~valid_mask].copy()
    
    return valid_df, invalid_df

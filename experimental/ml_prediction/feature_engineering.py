"""
Feature engineering using technical indicators from rhoa.indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
import os

# Add rhoa to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import rhoa

import config
FEATURE_CONFIG = config.FEATURE_CONFIG


def add_price_features(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Add basic price-based features (returns, price changes).
    
    Args:
        df: DataFrame with OHLCV data
        config: Feature configuration dict (uses FEATURE_CONFIG if None)
        
    Returns:
        DataFrame with added price features
    """
    config = config or FEATURE_CONFIG
    df = df.copy()
    
    # Price returns over different periods
    for period in config.get('return_periods', [1, 5, 10, 20]):
        df[f'return_{period}d'] = df['Close'].pct_change(period) * 100
    
    # Price relative to Open (intraday change)
    df['intraday_change'] = ((df['Close'] - df['Open']) / df['Open']) * 100
    
    # High-Low range (daily volatility proxy)
    df['high_low_range'] = ((df['High'] - df['Low']) / df['Close']) * 100
    
    # Close position in daily range
    df['close_position'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low']))
    df['close_position'] = df['close_position'].fillna(0.5)  # Fill when High == Low
    
    return df


def add_volume_features(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Add volume-based features if volume data is available.
    
    Args:
        df: DataFrame with OHLCV data
        config: Feature configuration dict
        
    Returns:
        DataFrame with added volume features
    """
    df = df.copy()
    
    if 'Volume' not in df.columns:
        return df
    
    # Volume change
    df['volume_change'] = df['Volume'].pct_change() * 100
    
    # Volume moving averages
    df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
    
    # Price-Volume relationship
    df['price_volume'] = df['return_1d'] * np.sign(df['volume_change'])
    
    return df


def add_moving_averages(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Add Simple and Exponential Moving Averages.
    
    Args:
        df: DataFrame with price data
        config: Feature configuration dict (uses FEATURE_CONFIG if None)
        
    Returns:
        DataFrame with added moving average features
    """
    config = config or FEATURE_CONFIG
    df = df.copy()
    
    # Simple Moving Averages
    for window in config.get('sma_windows', [5, 10, 20, 50, 200]):
        df[f'sma_{window}'] = df['Close'].indicators.sma(window_size=window)
        # Distance from SMA (percentage)
        df[f'close_to_sma_{window}'] = ((df['Close'] - df[f'sma_{window}']) / df[f'sma_{window}']) * 100
    
    # Exponential Weighted Moving Averages
    for window in config.get('ewma_windows', [5, 10, 20, 50]):
        df[f'ewma_{window}'] = df['Close'].indicators.ewma(window_size=window)
        df[f'close_to_ewma_{window}'] = ((df['Close'] - df[f'ewma_{window}']) / df[f'ewma_{window}']) * 100
    
    # Moving average crossovers
    if 5 in config.get('sma_windows', []) and 20 in config.get('sma_windows', []):
        df['sma_5_20_cross'] = df['sma_5'] - df['sma_20']
    
    if 10 in config.get('sma_windows', []) and 50 in config.get('sma_windows', []):
        df['sma_10_50_cross'] = df['sma_10'] - df['sma_50']
    
    return df


def add_momentum_indicators(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Add momentum-based technical indicators.
    
    Args:
        df: DataFrame with OHLCV data
        config: Feature configuration dict (uses FEATURE_CONFIG if None)
        
    Returns:
        DataFrame with added momentum indicators
    """
    config = config or FEATURE_CONFIG
    df = df.copy()
    
    # RSI
    rsi_window = config.get('rsi_window', 14)
    df['rsi'] = df['Close'].indicators.rsi(window_size=rsi_window)
    
    # CCI
    cci_window = config.get('cci_window', 20)
    df['cci'] = df['Close'].indicators.cci(df['High'], df['Low'], window_size=cci_window)
    
    # Stochastic Oscillator
    k_window = config.get('stochastic_k_window', 14)
    d_window = config.get('stochastic_d_window', 3)
    stoch = df['Close'].indicators.stochastic(df['High'], df['Low'], k_window=k_window, d_window=d_window)
    df['stoch_k'] = stoch['%K']
    df['stoch_d'] = stoch['%D']
    df['stoch_k_d_diff'] = df['stoch_k'] - df['stoch_d']
    
    # Williams %R
    wr_window = config.get('williams_r_window', 14)
    df['williams_r'] = df['Close'].indicators.williams_r(df['High'], df['Low'], window_size=wr_window)
    
    return df


def add_volatility_indicators(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Add volatility-based technical indicators.
    
    Args:
        df: DataFrame with OHLCV data
        config: Feature configuration dict (uses FEATURE_CONFIG if None)
        
    Returns:
        DataFrame with added volatility indicators
    """
    config = config or FEATURE_CONFIG
    df = df.copy()
    
    # Bollinger Bands
    bb_window = config.get('bollinger_window', 20)
    bb_std = config.get('bollinger_std', 2.0)
    bb = df['Close'].indicators.bollinger_bands(window_size=bb_window, num_std=bb_std)
    df['bb_upper'] = bb['upper_band']
    df['bb_middle'] = bb['middle_band']
    df['bb_lower'] = bb['lower_band']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR (Average True Range)
    atr_window = config.get('atr_window', 14)
    df['atr'] = df['Close'].indicators.atr(df['High'], df['Low'], window_size=atr_window)
    df['atr_percent'] = (df['atr'] / df['Close']) * 100
    
    # Exponentially Weighted Moving Standard Deviation
    for window in config.get('ewmstd_windows', [10, 20]):
        df[f'ewmstd_{window}'] = df['Close'].indicators.ewmstd(window_size=window)
    
    # Exponentially Weighted Moving Variance
    for window in config.get('ewmv_windows', [10, 20]):
        df[f'ewmv_{window}'] = df['Close'].indicators.ewmv(window_size=window)
    
    return df


def add_trend_indicators(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Add trend-based technical indicators.
    
    Args:
        df: DataFrame with OHLCV data
        config: Feature configuration dict (uses FEATURE_CONFIG if None)
        
    Returns:
        DataFrame with added trend indicators
    """
    config = config or FEATURE_CONFIG
    df = df.copy()
    
    # MACD
    macd_short = config.get('macd_short', 12)
    macd_long = config.get('macd_long', 26)
    macd_signal = config.get('macd_signal', 9)
    macd = df['Close'].indicators.macd(
        short_window=macd_short,
        long_window=macd_long,
        signal_window=macd_signal
    )
    df['macd'] = macd['macd']
    df['macd_signal'] = macd['signal']
    df['macd_histogram'] = macd['histogram']
    
    # ADX (Average Directional Index)
    adx_window = config.get('adx_window', 14)
    adx = df['Close'].indicators.adx(df['High'], df['Low'], window_size=adx_window)
    df['adx'] = adx['ADX']
    df['adx_plus_di'] = adx['+DI']
    df['adx_minus_di'] = adx['-DI']
    df['adx_di_diff'] = df['adx_plus_di'] - df['adx_minus_di']
    
    # Parabolic SAR
    psar_af_start = config.get('psar_af_start', 0.02)
    psar_af_increment = config.get('psar_af_increment', 0.02)
    psar_af_maximum = config.get('psar_af_maximum', 0.2)
    df['psar'] = df['Close'].indicators.parabolic_sar(
        df['High'],
        df['Low'],
        af_start=psar_af_start,
        af_increment=psar_af_increment,
        af_maximum=psar_af_maximum
    )
    df['close_to_psar'] = ((df['Close'] - df['psar']) / df['psar']) * 100
    
    return df


def add_lagged_features(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Add lagged features (previous values).
    
    Args:
        df: DataFrame with features
        config: Feature configuration dict (uses FEATURE_CONFIG if None)
        
    Returns:
        DataFrame with added lagged features
    """
    config = config or FEATURE_CONFIG
    df = df.copy()
    
    lag_periods = config.get('lag_periods', [1, 2, 3, 5])
    
    # Lagged returns
    for lag in lag_periods:
        df[f'return_1d_lag_{lag}'] = df['return_1d'].shift(lag)
    
    # Lagged RSI
    for lag in lag_periods[:2]:  # Only 1 and 2 day lags for RSI
        df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
    
    return df


def create_features(
    df: pd.DataFrame,
    config: dict = None,
    include_volume: bool = True
) -> pd.DataFrame:
    """
    Create all features using technical indicators.
    
    Args:
        df: DataFrame with OHLCV data (must have: Open, High, Low, Close, optionally Volume)
        config: Feature configuration dict (uses FEATURE_CONFIG if None)
        include_volume: Whether to include volume features (if Volume column exists)
        
    Returns:
        DataFrame with all engineered features
        
    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv('stock_data.csv')
        >>> df_with_features = create_features(df)
    """
    config = config or FEATURE_CONFIG
    df = df.copy()
    
    # Add all feature groups
    df = add_price_features(df, config)
    df = add_moving_averages(df, config)
    df = add_momentum_indicators(df, config)
    df = add_volatility_indicators(df, config)
    df = add_trend_indicators(df, config)
    
    if include_volume and 'Volume' in df.columns:
        df = add_volume_features(df, config)
    
    # Add lagged features last (after all other features are computed)
    df = add_lagged_features(df, config)
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature column names (excluding OHLCV and target columns).
    
    Args:
        df: DataFrame with features
        
    Returns:
        List of feature column names
    """
    # Columns to exclude (base data and targets)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 
                    'target', 'forward_return', 'Adj Close']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    return feature_cols


def get_feature_importance_names(config: dict = None) -> Dict[str, List[str]]:
    """
    Get feature names grouped by category for analysis.
    
    Args:
        config: Feature configuration dict
        
    Returns:
        Dictionary mapping feature categories to feature name patterns
    """
    config = config or FEATURE_CONFIG
    
    categories = {
        'price_features': ['return_', 'intraday_change', 'high_low_range', 'close_position'],
        'moving_averages': ['sma_', 'ewma_', 'close_to_sma_', 'close_to_ewma_', '_cross'],
        'momentum': ['rsi', 'cci', 'stoch_', 'williams_r'],
        'volatility': ['bb_', 'atr', 'ewmstd_', 'ewmv_'],
        'trend': ['macd', 'adx', 'psar', 'close_to_psar'],
        'volume': ['volume_'],
        'lagged': ['_lag_'],
    }
    
    return categories

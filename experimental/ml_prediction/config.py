"""
Configuration parameters for ML stock prediction model
"""

# Target generation parameters
TARGET_HORIZON_DAYS = 42  # Approximately 2 months of trading days
RETURN_THRESHOLD_PERCENT = 5.0  # 5% return threshold for binary classification

# Feature engineering parameters
FEATURE_CONFIG = {
    # Moving averages
    'sma_windows': [5, 10, 20, 50, 200],
    'ewma_windows': [5, 10, 20, 50],
    
    # Momentum indicators
    'rsi_window': 14,
    'cci_window': 20,
    'stochastic_k_window': 14,
    'stochastic_d_window': 3,
    'williams_r_window': 14,
    
    # Volatility indicators
    'bollinger_window': 20,
    'bollinger_std': 2.0,
    'atr_window': 14,
    'ewmstd_windows': [10, 20],
    'ewmv_windows': [10, 20],
    
    # Trend indicators
    'macd_short': 12,
    'macd_long': 26,
    'macd_signal': 9,
    'adx_window': 14,
    
    # Parabolic SAR
    'psar_af_start': 0.02,
    'psar_af_increment': 0.02,
    'psar_af_maximum': 0.2,
    
    # Additional features
    'return_periods': [1, 5, 10, 20],  # Calculate returns over these periods
    'lag_periods': [1, 2, 3, 5],  # Lagged features
}

# Data preprocessing parameters
PREPROCESSING_CONFIG = {
    'scaler_type': 'standard',  # 'standard' or 'minmax'
    'handle_nan': 'drop',  # 'drop', 'forward_fill', or 'interpolate'
    'min_valid_rows': 250,  # Minimum rows after dropping NaN
}

# Train/validation/test split
SPLIT_CONFIG = {
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
}

# Model architecture parameters
MODEL_CONFIG = {
    'layers': [128, 64, 32],  # Hidden layer sizes
    'dropout_rates': [0.3, 0.2, 0.2],  # Dropout after each hidden layer
    'activation': 'relu',
    'output_activation': 'sigmoid',  # Binary classification
    'learning_rate': 0.001,
    'optimizer': 'adam',
}

# Training parameters
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.0,  # We use explicit validation set
    'early_stopping_patience': 15,
    'early_stopping_monitor': 'val_loss',
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    'reduce_lr_min_lr': 1e-7,
    'class_weight_method': None,
    'verbose': 1,
}

# Walk-forward validation parameters
BACKTEST_CONFIG = {
    'initial_train_size': 500,  # Initial training window size (days)
    'test_size': 50,  # Test window size (days)
    'step_size': 25,  # Step size for rolling window (days)
    'retrain_frequency': 2,  # Retrain every N steps
    'min_train_size': 300,  # Minimum training size
}

# Model persistence
PERSISTENCE_CONFIG = {
    'model_save_format': 'h5',  # 'h5' or 'tf' (SavedModel)
    'model_dir': 'experimental/ml_prediction/saved_models',
    'scaler_filename': 'scaler.pkl',
    'config_filename': 'feature_config.json',
}

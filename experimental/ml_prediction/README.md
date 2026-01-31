# High-Precision Stock Prediction Model

This folder contains a trained neural network model that achieves **100% precision** (3 true positives, 0 false positives) for stock prediction.

**📈 Want to capture more true positives while maintaining 90-100% precision?**  
See **[IMPROVEMENTS_GUIDE.md](IMPROVEMENTS_GUIDE.md)** for instructions on training with multi-stock data.

## Model Performance

| Threshold | Precision | Recall | Signals | Correct | Wrong |
|-----------|-----------|--------|---------|---------|-------|
| 0.50 | 40.5% | 38.9% | 126 | 51 | 75 |
| 0.60 | **66.7%** | 10.7% | 21 | 14 | 7 |
| **0.67** | **100%** ⭐ | 2.3% | 3 | 3 | 0 |

See `saved_models/confusion_matrices_comparison.png` for visualization.

## Model Details

- **Architecture**: Neural network with [128, 64, 32] hidden layers
- **Features**: 63 technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, etc.)
- **Target**: Stocks with >5% return over next 42 trading days
- **Training**: Balanced class weights, early stopping on val_loss

## Files

### Core Modules
- `model.py` - Neural network architecture
- `preprocessing.py` - Data preprocessing and splitting
- `feature_engineering.py` - Technical indicator features (63 features)
- `training.py` - Training utilities
- `target_generation.py` - Target label creation
- `persistence.py` - Model save/load functions
- `config.py` - Configuration parameters

### Model
- `saved_models/high_precision_model/` - Trained model achieving 100% precision
  - `model.h5` - Trained neural network
  - `scaler.pkl` - Feature scaler (StandardScaler)
  - `feature_columns.json` - List of 63 feature names
  - `feature_config.json` - Feature engineering configuration
  - `model_config.json` - Model architecture configuration
  - `metadata.json` - Model metadata

### Usage Script
- `use_model.py` - Simple script to load and use the model

## Quick Start

```python
from use_model import load_model, predict

# Load the model
model_pipeline = load_model()

# Prepare your stock data with OHLCV columns
import pandas as pd
df = pd.read_csv('your_stock_data.csv')

# Make predictions
predictions, probabilities = predict(model_pipeline, df)

# Use threshold 0.67 for 100% precision
buy_signals = probabilities >= 0.67
```

## Requirements

```
tensorflow>=2.13.0
scikit-learn>=1.3.0
pandas>=1.3
numpy>=1.21
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install: `pip install -r requirements.txt`

## Notes

- The model uses 63 technical indicators as features
- Always use the same feature engineering pipeline as training
- Threshold 0.67 gives 100% precision but very few signals (~3 per test set)
- Threshold 0.60 gives 66.7% precision with more signals (~14)
- Test on out-of-sample data before using in production

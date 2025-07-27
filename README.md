# Noor

Noor is a Python package providing pandas DataFrame extension accessors for technical analysis.

## Installation

```bash
pip install noor
```

## Usage

```python
import pandas as pd
import noor  # Registers the `.indicators` accessor on Series

# Example price data
df = pd.DataFrame({
    "high":  [10, 11, 12, 13, 12, 11, 12],
    "low":   [8, 9, 10, 11, 10, 9, 10],
    "close": [9, 10, 11, 12, 11, 10, 11]
})

# Apply technical indicators
sma = df["close"].indicators.sma(window_size=3)
rsi = df["close"].indicators.rsi(window_size=3)
macd_df = df["close"].indicators.macd()
bb_df = df["close"].indicators.bollinger_bands(window_size=3)
atr = df["close"].indicators.atr(high=df["high"], low=df["low"])
cci = df["close"].indicators.cci(high=df["high"], low=df["low"])
```

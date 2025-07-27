# Noor

Noor is a Python package providing pandas DataFrame extension accessors for technical analysis.

## Installation

```bash
pip install Noor
```

## Usage

```python
import pandas as pd
import Noor

df = pd.DataFrame(...)
df_clean = df.preprocess.drop_na()
sma = df.indicators.sma(window=20)
```

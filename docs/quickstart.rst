Quick Start
===========

This guide will help you get started with Rhoa in just a few minutes.

Overview
--------

Rhoa provides pandas DataFrame extension accessors for:

- **Technical indicators** via ``.indicators`` accessor on pandas Series
- **ML target generation** via the ``generate_target_combinations()`` function
- **Visualization** via ``.plots`` accessor on pandas DataFrame

Installation
------------

If you haven't installed Rhoa yet:

.. code-block:: bash

   pip install rhoa

Your First Indicators
---------------------

Let's calculate some basic technical indicators on stock price data.

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import rhoa

   # Load your price data (assuming you have a CSV with Close prices)
   df = pd.read_csv('stock_prices.csv')

   # Calculate Simple Moving Average
   df['SMA_20'] = df['Close'].indicators.sma(window_size=20)

   # Calculate RSI (Relative Strength Index)
   df['RSI_14'] = df['Close'].indicators.rsi(window_size=14)

   # Calculate Exponential Moving Average
   df['EMA_12'] = df['Close'].indicators.ewma(span=12)

   print(df[['Close', 'SMA_20', 'RSI_14', 'EMA_12']].tail())

Working with OHLC Data
~~~~~~~~~~~~~~~~~~~~~~

For indicators that require High, Low, and Close prices:

.. code-block:: python

   # Calculate Average True Range
   df['ATR_14'] = df['Close'].indicators.atr(
       high=df['High'],
       low=df['Low'],
       window_size=14
   )

   # Calculate Stochastic Oscillator
   stoch = df['Close'].indicators.stochastic(
       high=df['High'],
       low=df['Low'],
       k_window=14,
       d_window=3
   )
   df['Stoch_K'] = stoch['%K']
   df['Stoch_D'] = stoch['%D']

Multiple Indicators at Once
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate multiple indicators
   close = df['Close']
   high = df['High']
   low = df['Low']

   # Build comprehensive technical analysis DataFrame
   df['SMA_50'] = close.indicators.sma(50)
   df['SMA_200'] = close.indicators.sma(200)
   df['RSI'] = close.indicators.rsi(14)
   df['ATR'] = close.indicators.atr(high, low, 14)

   macd = close.indicators.macd()
   df['MACD'] = macd['macd']
   df['MACD_Signal'] = macd['signal']
   df['MACD_Hist'] = macd['histogram']

   bb = close.indicators.bollinger_bands(window_size=20)
   df['BB_Upper'] = bb['upper_band']
   df['BB_Middle'] = bb['middle_band']
   df['BB_Lower'] = bb['lower_band']

Generating ML Targets
---------------------

Rhoa can automatically generate optimized binary targets for machine learning.

Auto Mode (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

Let Rhoa find the optimal lookback period and threshold:

.. code-block:: python

   from rhoa.targets import generate_target_combinations

   # Generate targets with Pareto optimization
   targets, metadata = generate_target_combinations(
       df,
       mode='auto',
       target_class_balance=0.5  # Aim for 50% positive class
   )

   # Inspect what was found optimal for each target
   for method, params in metadata.items():
       print(f"{method}: period={params['period']}, "
             f"threshold={params['threshold']}%, "
             f"instances={params['instances']}")

This generates 8 different target types in the returned DataFrame:

- Target_1: Close[N]/Close[0]
- Target_2: Close[N]/High[0]
- Target_3: High[N]/Close[0]
- Target_4: High[N]/High[0]
- Target_5: MaxClose/Close[0]
- Target_6: MaxClose/High[0]
- Target_7: MaxHigh/Close[0]
- Target_8: MaxHigh/High[0]

Manual Mode
~~~~~~~~~~~

Specify a fixed lookback period and let Rhoa optimize thresholds:

.. code-block:: python

   # Generate targets with fixed 5-day lookback
   targets, metadata = generate_target_combinations(
       df,
       mode='manual',
       lookback_periods=5
   )

   # All targets use 5-day period with optimized thresholds
   print(targets.head())

Building an ML Pipeline
-----------------------

Here's a complete example combining features and targets:

.. code-block:: python

   from rhoa.targets import generate_target_combinations
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import classification_report

   # Step 1: Load data
   df = pd.read_csv('stock_prices.csv')

   # Step 2: Generate technical indicator features
   close = df['Close']
   high = df['High']
   low = df['Low']

   df['SMA_20'] = close.indicators.sma(20)
   df['SMA_50'] = close.indicators.sma(50)
   df['RSI_14'] = close.indicators.rsi(14)
   df['ATR_14'] = close.indicators.atr(high, low, 14)
   df['Returns'] = close.pct_change()

   # Step 3: Generate optimized targets
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       target_class_balance=0.3  # 30% positive instances
   )

   # Step 4: Prepare ML dataset
   # Combine features and targets
   ml_df = pd.concat([df, targets], axis=1).dropna()

   # Select features and target
   feature_cols = ['SMA_20', 'SMA_50', 'RSI_14', 'ATR_14', 'Returns']
   X = ml_df[feature_cols]
   y = ml_df['Target_7']  # Using MaxHigh/Close[0]

   # Step 5: Train-test split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # Step 6: Train model
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)

   # Step 7: Evaluate
   y_pred = model.predict(X_test)
   print(classification_report(y_test, y_pred))
   print(f"Accuracy: {model.score(X_test, y_test):.2%}")

Visualizing Predictions
-----------------------

Visualize your model's predictions on a price chart:

.. code-block:: python

   # After training your model
   predictions = model.predict(X_test)

   # Create visualization with confusion matrix
   fig = df.loc[X_test.index].plots.signal(
       y_pred=predictions,
       y_true=y_test,
       date_col='Date',
       price_col='Close',
       title='Stock Predictions'
   )

   # Save to file
   fig.savefig('predictions.png', dpi=300, bbox_inches='tight')

The visualization shows:

- **Stock price line** over time
- **Light green dots**: True opportunities (y_true=1)
- **Bright green dots**: Model predictions (y_pred=1)
- **Red X markers**: False positives
- **Orange circles**: False negatives (missed opportunities)
- **Confusion matrix**: Performance summary

Common Patterns
---------------

Find Overbought/Oversold Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate RSI
   rsi = df['Close'].indicators.rsi(14)

   # Find overbought (RSI > 70)
   overbought = df[rsi > 70]
   print(f"Overbought periods: {len(overbought)}")

   # Find oversold (RSI < 30)
   oversold = df[rsi < 30]
   print(f"Oversold periods: {len(oversold)}")

Detect Moving Average Crossovers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate moving averages
   sma_50 = df['Close'].indicators.sma(50)
   sma_200 = df['Close'].indicators.sma(200)

   # Detect golden cross (50 crosses above 200)
   golden_cross = (sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1))

   # Detect death cross (50 crosses below 200)
   death_cross = (sma_50 < sma_200) & (sma_50.shift(1) >= sma_200.shift(1))

   print(f"Golden crosses: {golden_cross.sum()}")
   print(f"Death crosses: {death_cross.sum()}")

Identify MACD Signals
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate MACD
   macd_data = df['Close'].indicators.macd()
   macd = macd_data['macd']
   signal = macd_data['signal']

   # Bullish crossover
   bullish = (macd > signal) & (macd.shift(1) <= signal.shift(1))

   # Bearish crossover
   bearish = (macd < signal) & (macd.shift(1) >= signal.shift(1))

   print(f"Bullish signals: {bullish.sum()}")
   print(f"Bearish signals: {bearish.sum()}")

Check Bollinger Band Breakouts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate Bollinger Bands
   bb = df['Close'].indicators.bollinger_bands(window_size=20, num_std=2.0)

   # Price touching upper band (potential reversal)
   upper_touch = df['Close'] >= bb['upper_band']

   # Price touching lower band (potential bounce)
   lower_touch = df['Close'] <= bb['lower_band']

   print(f"Upper band touches: {upper_touch.sum()}")
   print(f"Lower band touches: {lower_touch.sum()}")

Next Steps
----------

Now that you understand the basics, explore:

- :doc:`user_guide/index` - In-depth conceptual guides
- :doc:`examples/index` - More practical examples
- :doc:`api/index` - Complete API reference
- :doc:`faq` - Common questions and solutions

Tips for Success
----------------

1. **Always import rhoa** before using accessors
2. **Handle NaN values** - Indicators create NaN for initial periods
3. **Choose appropriate window sizes** - Longer = smoother but more lag
4. **Test multiple targets** - Different targets work for different strategies
5. **Validate on out-of-sample data** - Avoid overfitting
6. **Document your metadata** - Save target generation parameters for reproducibility

Need Help?
----------

- Check the :doc:`faq` for common issues
- Read the :doc:`user_guide/index` for detailed explanations
- Browse :doc:`examples/index` for more use cases
- Open an issue on `GitHub <https://github.com/nainajnahO/Rhoa/issues>`_

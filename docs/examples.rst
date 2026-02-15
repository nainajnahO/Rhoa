Examples
========

Basic Usage
-----------

Simple Moving Average
~~~~~~~~~~~~~~~~~~~~~

Calculate a 20-period Simple Moving Average:

.. code-block:: python

   import pandas as pd
   import rhoa

   # Load your price data
   df = pd.read_csv('your_data.csv')

   # Calculate 20-period SMA
   sma_20 = df.rhoa.indicators.sma(window_size=20)
   print(sma_20.head())

RSI (Relative Strength Index)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate the RSI indicator:

.. code-block:: python

   # Calculate 14-period RSI
   rsi = df.rhoa.indicators.rsi(window_size=14)
   
   # Find overbought conditions (RSI > 70)
   overbought = rsi > 70
   print(f"Overbought periods: {overbought.sum()}")

MACD Indicator
~~~~~~~~~~~~~~

Calculate MACD with custom parameters:

.. code-block:: python

   # Calculate MACD with default parameters (12, 26, 9)
   macd_data = df.rhoa.indicators.macd()
   
   # Access individual components
   macd_line = macd_data['macd']
   signal_line = macd_data['signal']
   histogram = macd_data['histogram']
   
   # Find bullish crossovers (MACD crosses above signal)
   bullish_crossover = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))

Bollinger Bands
~~~~~~~~~~~~~~~

Calculate Bollinger Bands:

.. code-block:: python

   # Calculate Bollinger Bands (20-period, 2 standard deviations)
   bb = df.rhoa.indicators.bollinger_bands(window_size=20, num_std=2.0)

   # Check for price touching upper band
   touching_upper = df['close'] >= bb['upper_band']

   # Check for price touching lower band
   touching_lower = df['close'] <= bb['lower_band']

Multiple Indicators
~~~~~~~~~~~~~~~~~~~

Combine multiple indicators for analysis:

.. code-block:: python

   # Calculate multiple indicators
   sma_50 = df.rhoa.indicators.sma(window_size=50)
   rsi_14 = df.rhoa.indicators.rsi(window_size=14)
   bb = df.rhoa.indicators.bollinger_bands()

   # Create a comprehensive analysis DataFrame
   analysis = pd.DataFrame({
       'price': df['close'],
       'sma_50': sma_50,
       'rsi': rsi_14,
       'bb_upper': bb['upper_band'],
       'bb_middle': bb['middle_band'],
       'bb_lower': bb['lower_band']
   })
   
   print(analysis.head())

Advanced Usage with High/Low/Close Data
---------------------------------------

For indicators that require OHLC data:

.. code-block:: python

   # Assuming you have OHLC data

   # Average True Range
   atr = df.rhoa.indicators.atr(window_size=14)

   # Stochastic Oscillator
   stoch = df.rhoa.indicators.stochastic(k_window=14, d_window=3)
   k_percent = stoch['%K']
   d_percent = stoch['%D']

   # ADX (Average Directional Index)
   adx_data = df.rhoa.indicators.adx(window_size=14)
   adx = adx_data['ADX']
   plus_di = adx_data['+DI']
   minus_di = adx_data['-DI']

Target Generation for Machine Learning
---------------------------------------

Generate optimized binary targets for ML models:

Auto Mode (Pareto Optimization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automatically finds optimal period and threshold:

.. code-block:: python

   from rhoa.targets import generate_target_combinations

   # Load OHLC data
   df = pd.read_csv('prices.csv')

   # Generate targets with Pareto optimization
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       target_class_balance=0.5  # 50% positive instances
   )

   # Check optimal parameters found
   print(meta['method_7'])  # MaxHigh/Close[0]
   # {'period': 6, 'threshold': 4.0, 'instances': 249, 'pct_of_max': 8.9}

   # Use in ML pipeline
   print(targets.head())
   #    Target_1  Target_2  ...  Target_8
   # 0     False     False  ...     False
   # 1      True     False  ...      True

Manual Mode (Elbow Method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fixed period with elbow-optimized thresholds:

.. code-block:: python

   # Generate targets with fixed 5-day lookback
   targets, meta = generate_target_combinations(
       df,
       mode='manual',
       lookback_periods=5
   )

   # All methods use period=5, with elbow-detected thresholds
   print(meta['method_1'])
   # {'period': 5, 'threshold': 6.0, 'instances': 22, 'pct_of_max': 1.4}

ML Pipeline Integration
~~~~~~~~~~~~~~~~~~~~~~~

Complete example with features and model training:

.. code-block:: python

   from rhoa.targets import generate_target_combinations
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split

   # Generate features
   df['SMA_20'] = df['Close'].rolling(20).mean()
   df['Returns'] = df['Close'].pct_change()

   # Generate optimized targets
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       target_class_balance=0.3  # 30% positive instances
   )

   # Combine features and targets
   ml_df = pd.concat([df, targets], axis=1).dropna()

   # Train model on Method 7 (MaxHigh/Close[0])
   X = ml_df[['SMA_20', 'Returns']]
   y = ml_df['Target_7']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   model = RandomForestClassifier().fit(X_train, y_train)

   print(f"Accuracy: {model.score(X_test, y_test):.2%}")
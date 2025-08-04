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
   prices = df['close']
   
   # Calculate 20-period SMA
   sma_20 = prices.indicators.sma(window_size=20)
   print(sma_20.head())

RSI (Relative Strength Index)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate the RSI indicator:

.. code-block:: python

   # Calculate 14-period RSI
   rsi = prices.indicators.rsi(window_size=14)
   
   # Find overbought conditions (RSI > 70)
   overbought = rsi > 70
   print(f"Overbought periods: {overbought.sum()}")

MACD Indicator
~~~~~~~~~~~~~~

Calculate MACD with custom parameters:

.. code-block:: python

   # Calculate MACD with default parameters (12, 26, 9)
   macd_data = prices.indicators.macd()
   
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
   bb = prices.indicators.bollinger_bands(window_size=20, num_std=2.0)
   
   # Check for price touching upper band
   touching_upper = prices >= bb['upper_band']
   
   # Check for price touching lower band  
   touching_lower = prices <= bb['lower_band']

Multiple Indicators
~~~~~~~~~~~~~~~~~~~

Combine multiple indicators for analysis:

.. code-block:: python

   # Calculate multiple indicators
   sma_50 = prices.indicators.sma(window_size=50)
   rsi_14 = prices.indicators.rsi(window_size=14)
   bb = prices.indicators.bollinger_bands()
   
   # Create a comprehensive analysis DataFrame
   analysis = pd.DataFrame({
       'price': prices,
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
   high = df['high']
   low = df['low'] 
   close = df['close']
   
   # Average True Range
   atr = close.indicators.atr(high, low, window_size=14)
   
   # Stochastic Oscillator
   stoch = close.indicators.stochastic(high, low, k_window=14, d_window=3)
   k_percent = stoch['%K']
   d_percent = stoch['%D']
   
   # ADX (Average Directional Index)
   adx_data = close.indicators.adx(high, low, window_size=14)
   adx = adx_data['ADX']
   plus_di = adx_data['+DI']
   minus_di = adx_data['-DI']
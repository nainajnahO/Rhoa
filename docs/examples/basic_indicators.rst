Basic Indicators
================

This page covers basic technical indicators that work with a single price series.

.. _basic-sma:

Simple Moving Average (SMA)
----------------------------

The Simple Moving Average is the foundation of technical analysis.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import rhoa

   # Load your price data
   df = pd.read_csv('prices.csv')
   prices = df['Close']

   # Calculate 20-period SMA
   sma_20 = prices.rhoa.indicators.sma(window_size=20)
   print(sma_20.head())

Multiple SMAs
~~~~~~~~~~~~~

Calculate multiple moving averages to identify trends:

.. code-block:: python

   # Calculate short, medium, and long-term SMAs
   df['SMA_20'] = df['Close'].rhoa.indicators.sma(window_size=20)
   df['SMA_50'] = df['Close'].rhoa.indicators.sma(window_size=50)
   df['SMA_200'] = df['Close'].rhoa.indicators.sma(window_size=200)

   # Identify trend: price above all SMAs = strong uptrend
   strong_uptrend = (
       (df['Close'] > df['SMA_20']) &
       (df['SMA_20'] > df['SMA_50']) &
       (df['SMA_50'] > df['SMA_200'])
   )
   print(f"Strong uptrend periods: {strong_uptrend.sum()}")

Moving Average Crossovers
~~~~~~~~~~~~~~~~~~~~~~~~~~

Detect when faster MA crosses slower MA:

.. code-block:: python

   sma_50 = df['Close'].rhoa.indicators.sma(50)
   sma_200 = df['Close'].rhoa.indicators.sma(200)

   # Golden Cross: 50 crosses above 200 (bullish)
   golden_cross = (sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1))

   # Death Cross: 50 crosses below 200 (bearish)
   death_cross = (sma_50 < sma_200) & (sma_50.shift(1) >= sma_200.shift(1))

   # Find dates of crossovers
   golden_dates = df.loc[golden_cross, 'Date']
   death_dates = df.loc[death_cross, 'Date']

   print(f"Golden crosses: {len(golden_dates)}")
   print(f"Death crosses: {len(death_dates)}")

.. _basic-ema:

Exponential Moving Average (EWMA)
----------------------------------

EWMA gives more weight to recent prices, making it more responsive than SMA.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   # Calculate 12-period EWMA
   ema_12 = df['Close'].rhoa.indicators.ewma(span=12)

   # Calculate 26-period EWMA
   ema_26 = df['Close'].rhoa.indicators.ewma(span=26)

   df['EMA_12'] = ema_12
   df['EMA_26'] = ema_26

EMA Crossover Strategy
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ema_12 = df['Close'].rhoa.indicators.ewma(span=12)
   ema_26 = df['Close'].rhoa.indicators.ewma(span=26)

   # Buy signal: fast EMA crosses above slow EMA
   buy_signal = (ema_12 > ema_26) & (ema_12.shift(1) <= ema_26.shift(1))

   # Sell signal: fast EMA crosses below slow EMA
   sell_signal = (ema_12 < ema_26) & (ema_12.shift(1) >= ema_26.shift(1))

   df['Signal'] = 0
   df.loc[buy_signal, 'Signal'] = 1
   df.loc[sell_signal, 'Signal'] = -1

.. _basic-rsi:

Relative Strength Index (RSI)
------------------------------

RSI measures momentum and identifies overbought/oversold conditions.

Basic RSI Calculation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate 14-period RSI
   rsi = df['Close'].rhoa.indicators.rsi(window_size=14)
   df['RSI_14'] = rsi

   print(f"RSI range: {rsi.min():.2f} to {rsi.max():.2f}")

Find Overbought/Oversold
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   rsi = df['Close'].rhoa.indicators.rsi(window_size=14)

   # Traditional thresholds
   overbought = rsi > 70
   oversold = rsi < 30

   # Find specific periods
   overbought_periods = df[overbought]
   oversold_periods = df[oversold]

   print(f"Overbought periods: {len(overbought_periods)}")
   print(f"Oversold periods: {len(oversold_periods)}")

RSI Divergence Detection
~~~~~~~~~~~~~~~~~~~~~~~~~

Detect when price and RSI diverge (advanced pattern):

.. code-block:: python

   # Calculate RSI
   rsi = df['Close'].rhoa.indicators.rsi(14)

   # Find local peaks in price and RSI
   price_peaks = df['Close'].rolling(5).max() == df['Close']
   rsi_peaks = rsi.rolling(5).max() == rsi

   # Bearish divergence: price makes higher high, RSI makes lower high
   # (Simplified version - production code needs more sophisticated logic)
   df['Price_Peak'] = price_peaks
   df['RSI_Peak'] = rsi_peaks

Moving Variance and Standard Deviation
---------------------------------------

Measure volatility using exponentially weighted statistics.

Volatility Calculation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate exponentially weighted moving standard deviation
   vol = df['Close'].rhoa.indicators.ewmstd(span=20)
   df['Volatility'] = vol

   # High volatility periods
   high_vol = vol > vol.quantile(0.75)
   print(f"High volatility periods: {high_vol.sum()}")

Relative Volatility
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate returns
   returns = df['Close'].pct_change()

   # Calculate rolling volatility
   vol_20 = df['Close'].rhoa.indicators.ewmstd(span=20)

   # Normalize by average volatility
   avg_vol = vol_20.mean()
   relative_vol = vol_20 / avg_vol

   # Find regime changes
   low_vol_regime = relative_vol < 0.8
   high_vol_regime = relative_vol > 1.2

Williams %R
-----------

Williams %R is a momentum indicator similar to RSI.

Basic Calculation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate Williams %R
   williams_r = df['Close'].rhoa.indicators.williams_r(
       high=df['High'],
       low=df['Low'],
       window_size=14
   )
   df['Williams_R'] = williams_r

   # Overbought: above -20
   # Oversold: below -80
   overbought = williams_r > -20
   oversold = williams_r < -80

Combining Multiple Indicators
------------------------------

Use multiple indicators together for confirmation.

Multi-Indicator Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate multiple indicators
   prices = df['Close']

   df['SMA_50'] = prices.rhoa.indicators.sma(50)
   df['RSI'] = prices.rhoa.indicators.rsi(14)
   df['EMA_12'] = prices.rhoa.indicators.ewma(span=12)
   df['EMA_26'] = prices.rhoa.indicators.ewma(span=26)

   # Bullish conditions (all must be true)
   bullish = (
       (df['Close'] > df['SMA_50']) &  # Price above MA
       (df['RSI'] < 70) &                # Not overbought
       (df['EMA_12'] > df['EMA_26'])    # Short EMA above long
   )

   # Bearish conditions
   bearish = (
       (df['Close'] < df['SMA_50']) &   # Price below MA
       (df['RSI'] > 30) &                # Not oversold
       (df['EMA_12'] < df['EMA_26'])    # Short EMA below long
   )

   print(f"Bullish periods: {bullish.sum()}")
   print(f"Bearish periods: {bearish.sum()}")

Indicator Dashboard
~~~~~~~~~~~~~~~~~~~

Create a comprehensive view of multiple indicators:

.. code-block:: python

   import pandas as pd
   import rhoa

   def create_indicator_dashboard(df):
       """Create dashboard with multiple indicators."""
       close = df['Close']

       # Trend indicators
       df['SMA_20'] = close.rhoa.indicators.sma(20)
       df['SMA_50'] = close.rhoa.indicators.sma(50)
       df['EMA_12'] = close.rhoa.indicators.ewma(span=12)

       # Momentum indicators
       df['RSI_14'] = close.rhoa.indicators.rsi(14)

       # Volatility
       df['Volatility'] = close.rhoa.indicators.ewmstd(span=20)

       # Summary metrics
       trend = "Bullish" if close.iloc[-1] > df['SMA_50'].iloc[-1] else "Bearish"
       rsi_status = "Overbought" if df['RSI_14'].iloc[-1] > 70 else \
                    "Oversold" if df['RSI_14'].iloc[-1] < 30 else "Neutral"

       print(f"Trend: {trend}")
       print(f"RSI Status: {rsi_status}")
       print(f"Current RSI: {df['RSI_14'].iloc[-1]:.2f}")

       return df

   # Use the dashboard
   df = pd.read_csv('prices.csv')
   df = create_indicator_dashboard(df)

Next Steps
----------

Continue to :doc:`advanced_indicators` to learn about more sophisticated indicators
like MACD, Bollinger Bands, and ADX.

Advanced Indicators
===================

This page covers advanced technical indicators that require OHLC (Open, High, Low, Close) data.

.. _advanced-macd:

MACD (Moving Average Convergence Divergence)
---------------------------------------------

MACD is a trend-following momentum indicator showing the relationship between two moving averages.

Basic MACD
~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import rhoa

   df = pd.read_csv('prices.csv')

   # Calculate MACD with default parameters (12, 26, 9)
   macd_data = df['Close'].rhoa.indicators.macd()

   # Extract components
   df['MACD'] = macd_data['macd']
   df['MACD_Signal'] = macd_data['signal']
   df['MACD_Histogram'] = macd_data['histogram']

   print(df[['Close', 'MACD', 'MACD_Signal', 'MACD_Histogram']].tail())

Custom Parameters
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use custom parameters: fast=8, slow=21, signal=5
   macd_data = df['Close'].rhoa.indicators.macd(
       fast_period=8,
       slow_period=21,
       signal_period=5
   )

MACD Crossover Signals
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   macd_data = df['Close'].rhoa.indicators.macd()
   macd = macd_data['macd']
   signal = macd_data['signal']

   # Bullish crossover: MACD crosses above signal line
   bullish_cross = (macd > signal) & (macd.shift(1) <= signal.shift(1))

   # Bearish crossover: MACD crosses below signal line
   bearish_cross = (macd < signal) & (macd.shift(1) >= signal.shift(1))

   # Zero line crossovers
   macd_above_zero = (macd > 0) & (macd.shift(1) <= 0)
   macd_below_zero = (macd < 0) & (macd.shift(1) >= 0)

   print(f"Bullish crossovers: {bullish_cross.sum()}")
   print(f"Bearish crossovers: {bearish_cross.sum()}")
   print(f"MACD crossed above zero: {macd_above_zero.sum()}")
   print(f"MACD crossed below zero: {macd_below_zero.sum()}")

MACD Histogram Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   macd_data = df['Close'].rhoa.indicators.macd()
   histogram = macd_data['histogram']

   # Histogram turning positive (momentum shift)
   momentum_shift_bullish = (histogram > 0) & (histogram.shift(1) <= 0)

   # Histogram divergence (simplified)
   # When histogram makes higher lows while price makes lower lows
   df['MACD_Hist'] = histogram

.. _advanced-bollinger:

Bollinger Bands
---------------

Bollinger Bands measure volatility and identify potential overbought/oversold conditions.

Basic Bollinger Bands
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate Bollinger Bands (20-period, 2 std devs)
   bb = df['Close'].rhoa.indicators.bollinger_bands(window_size=20, num_std=2.0)

   df['BB_Upper'] = bb['upper_band']
   df['BB_Middle'] = bb['middle_band']
   df['BB_Lower'] = bb['lower_band']

   print(df[['Close', 'BB_Upper', 'BB_Middle', 'BB_Lower']].tail())

Bollinger Band Squeeze
~~~~~~~~~~~~~~~~~~~~~~~

Identify low volatility periods that often precede large moves:

.. code-block:: python

   bb = df['Close'].rhoa.indicators.bollinger_bands(window_size=20, num_std=2.0)

   # Calculate band width
   band_width = (bb['upper_band'] - bb['lower_band']) / bb['middle_band']

   # Squeeze: band width in lowest 20% historically
   squeeze_threshold = band_width.quantile(0.20)
   squeeze = band_width < squeeze_threshold

   print(f"Squeeze periods: {squeeze.sum()}")
   print(f"Average band width: {band_width.mean():.4f}")

Bollinger Band Breakouts
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   bb = df['Close'].rhoa.indicators.bollinger_bands(window_size=20, num_std=2.0)

   # Price touching or exceeding upper band
   upper_touch = df['Close'] >= bb['upper_band']

   # Price touching or exceeding lower band
   lower_touch = df['Close'] <= bb['lower_band']

   # Strong breakout: close outside bands
   upper_breakout = df['Close'] > bb['upper_band']
   lower_breakout = df['Close'] < bb['lower_band']

   print(f"Upper band touches: {upper_touch.sum()}")
   print(f"Lower band touches: {lower_touch.sum()}")

%B Indicator
~~~~~~~~~~~~

Calculate where price is relative to the bands:

.. code-block:: python

   bb = df['Close'].rhoa.indicators.bollinger_bands(window_size=20, num_std=2.0)

   # %B = (Close - Lower Band) / (Upper Band - Lower Band)
   # %B > 1: above upper band
   # %B < 0: below lower band
   # %B = 0.5: at middle band

   percent_b = (df['Close'] - bb['lower_band']) / (bb['upper_band'] - bb['lower_band'])
   df['Percent_B'] = percent_b

   # Identify extremes
   extremely_overbought = percent_b > 1.0
   extremely_oversold = percent_b < 0.0

ADX (Average Directional Index)
--------------------------------

ADX measures trend strength without indicating direction.

Basic ADX
~~~~~~~~~

.. code-block:: python

   # Calculate ADX with 14-period window
   adx_data = df['Close'].rhoa.indicators.adx(
       high=df['High'],
       low=df['Low'],
       window_size=14
   )

   df['ADX'] = adx_data['ADX']
   df['+DI'] = adx_data['+DI']
   df['-DI'] = adx_data['-DI']

   print(df[['Close', 'ADX', '+DI', '-DI']].tail())

Interpret ADX Values
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   adx_data = df['Close'].rhoa.indicators.adx(df['High'], df['Low'], 14)
   adx = adx_data['ADX']

   # ADX interpretation
   # < 20: Weak trend or ranging market
   # 20-25: Emerging trend
   # 25-50: Strong trend
   # > 50: Very strong trend

   weak_trend = adx < 20
   strong_trend = adx > 25
   very_strong = adx > 50

   print(f"Weak trend periods: {weak_trend.sum()}")
   print(f"Strong trend periods: {strong_trend.sum()}")
   print(f"Very strong trend periods: {very_strong.sum()}")

Directional Movement Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   adx_data = df['Close'].rhoa.indicators.adx(df['High'], df['Low'], 14)

   adx = adx_data['ADX']
   plus_di = adx_data['+DI']
   minus_di = adx_data['-DI']

   # Strong uptrend: +DI > -DI and ADX > 25
   strong_uptrend = (plus_di > minus_di) & (adx > 25)

   # Strong downtrend: -DI > +DI and ADX > 25
   strong_downtrend = (minus_di > plus_di) & (adx > 25)

   # Ranging market: ADX < 20
   ranging = adx < 20

   print(f"Strong uptrend periods: {strong_uptrend.sum()}")
   print(f"Strong downtrend periods: {strong_downtrend.sum()}")
   print(f"Ranging periods: {ranging.sum()}")

Average True Range (ATR)
------------------------

ATR measures market volatility.

Basic ATR
~~~~~~~~~

.. code-block:: python

   # Calculate 14-period ATR
   atr = df['Close'].rhoa.indicators.atr(
       high=df['High'],
       low=df['Low'],
       window_size=14
   )
   df['ATR_14'] = atr

   print(f"Average ATR: {atr.mean():.2f}")
   print(f"Current ATR: {atr.iloc[-1]:.2f}")

Volatility Regimes
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   atr = df['Close'].rhoa.indicators.atr(df['High'], df['Low'], 14)

   # Define volatility regimes using percentiles
   low_vol = atr < atr.quantile(0.33)
   normal_vol = (atr >= atr.quantile(0.33)) & (atr < atr.quantile(0.67))
   high_vol = atr >= atr.quantile(0.67)

   print(f"Low volatility: {low_vol.sum()} days")
   print(f"Normal volatility: {normal_vol.sum()} days")
   print(f"High volatility: {high_vol.sum()} days")

ATR-Based Position Sizing
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   atr = df['Close'].rhoa.indicators.atr(df['High'], df['Low'], 14)

   # Risk per trade: 2% of account
   account_size = 100000
   risk_per_trade = account_size * 0.02  # $2000

   # Position size based on ATR stop loss
   # Stop loss = 2x ATR
   stop_loss_atr = 2 * atr
   position_size = risk_per_trade / stop_loss_atr

   df['Position_Size'] = position_size
   print(df[['Close', 'ATR_14', 'Position_Size']].tail())

Stochastic Oscillator
---------------------

The Stochastic Oscillator compares closing price to the price range over time.

Basic Stochastic
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate Stochastic with 14-period %K and 3-period %D
   stoch = df['Close'].rhoa.indicators.stochastic(
       high=df['High'],
       low=df['Low'],
       k_window=14,
       d_window=3
   )

   df['Stoch_K'] = stoch['%K']
   df['Stoch_D'] = stoch['%D']

   print(df[['Close', 'Stoch_K', 'Stoch_D']].tail())

Stochastic Signals
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   stoch = df['Close'].rhoa.indicators.stochastic(df['High'], df['Low'], 14, 3)
   k = stoch['%K']
   d = stoch['%D']

   # Overbought/Oversold levels
   overbought = k > 80
   oversold = k < 20

   # Crossover signals
   bullish_cross = (k > d) & (k.shift(1) <= d.shift(1))
   bearish_cross = (k < d) & (k.shift(1) >= d.shift(1))

   # Buy signal: bullish cross in oversold region
   buy_signal = bullish_cross & (k < 20)

   # Sell signal: bearish cross in overbought region
   sell_signal = bearish_cross & (k > 80)

   print(f"Buy signals: {buy_signal.sum()}")
   print(f"Sell signals: {sell_signal.sum()}")

CCI (Commodity Channel Index)
------------------------------

CCI identifies cyclical trends and potential reversals.

Basic CCI
~~~~~~~~~

.. code-block:: python

   # Calculate 20-period CCI
   cci = df['Close'].rhoa.indicators.cci(
       high=df['High'],
       low=df['Low'],
       window_size=20
   )
   df['CCI_20'] = cci

   print(f"CCI range: {cci.min():.2f} to {cci.max():.2f}")

CCI Trading Signals
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   cci = df['Close'].rhoa.indicators.cci(df['High'], df['Low'], 20)

   # Traditional levels
   # > +100: Overbought
   # < -100: Oversold

   overbought = cci > 100
   oversold = cci < -100

   # Extreme levels
   extremely_overbought = cci > 200
   extremely_oversold = cci < -200

   # Zero line crossovers
   bullish_cross = (cci > 0) & (cci.shift(1) <= 0)
   bearish_cross = (cci < 0) & (cci.shift(1) >= 0)

   print(f"Overbought periods: {overbought.sum()}")
   print(f"Oversold periods: {oversold.sum()}")

Parabolic SAR
-------------

Parabolic SAR identifies potential reversal points.

Basic SAR
~~~~~~~~~

.. code-block:: python

   # Calculate Parabolic SAR
   sar = df['Close'].rhoa.indicators.parabolic_sar(
       high=df['High'],
       low=df['Low']
   )
   df['SAR'] = sar

   # SAR below price = uptrend
   # SAR above price = downtrend
   uptrend = df['Close'] > sar
   downtrend = df['Close'] < sar

   print(f"Uptrend periods: {uptrend.sum()}")
   print(f"Downtrend periods: {downtrend.sum()}")

SAR Reversals
~~~~~~~~~~~~~

.. code-block:: python

   sar = df['Close'].rhoa.indicators.parabolic_sar(df['High'], df['Low'])

   # Detect trend reversals
   uptrend = df['Close'] > sar
   downtrend = df['Close'] < sar

   # Bullish reversal: downtrend to uptrend
   bullish_reversal = uptrend & ~uptrend.shift(1)

   # Bearish reversal: uptrend to downtrend
   bearish_reversal = downtrend & ~downtrend.shift(1)

   print(f"Bullish reversals: {bullish_reversal.sum()}")
   print(f"Bearish reversals: {bearish_reversal.sum()}")

Multi-Indicator Strategies
---------------------------

Combine multiple advanced indicators for robust signals.

Trend + Momentum Confirmation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate indicators
   adx_data = df['Close'].rhoa.indicators.adx(df['High'], df['Low'], 14)
   macd_data = df['Close'].rhoa.indicators.macd()
   bb = df['Close'].rhoa.indicators.bollinger_bands(20, 2.0)

   # Strong trend confirmation
   strong_trend = adx_data['ADX'] > 25
   bullish_momentum = macd_data['macd'] > macd_data['signal']
   price_above_ma = df['Close'] > bb['middle_band']

   # All conditions must be true
   confirmed_uptrend = strong_trend & bullish_momentum & price_above_ma

   print(f"Confirmed uptrend periods: {confirmed_uptrend.sum()}")

Volatility Breakout System
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Combine Bollinger Bands + ATR + ADX
   bb = df['Close'].rhoa.indicators.bollinger_bands(20, 2.0)
   atr = df['Close'].rhoa.indicators.atr(df['High'], df['Low'], 14)
   adx_data = df['Close'].rhoa.indicators.adx(df['High'], df['Low'], 14)

   # Squeeze: low volatility
   band_width = (bb['upper_band'] - bb['lower_band']) / bb['middle_band']
   squeeze = band_width < band_width.quantile(0.20)

   # Breakout conditions
   price_breakout = df['Close'] > bb['upper_band']
   increasing_vol = atr > atr.shift(5)
   trend_forming = adx_data['ADX'] > 20

   # Breakout signal
   breakout_signal = squeeze.shift(1) & price_breakout & increasing_vol & trend_forming

   print(f"Breakout signals: {breakout_signal.sum()}")

Complete Technical Analysis Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def technical_dashboard(df):
       """Create comprehensive technical analysis dashboard."""
       close = df['Close']
       high = df['High']
       low = df['Low']

       # Trend indicators
       macd = close.rhoa.indicators.macd()
       adx = close.rhoa.indicators.adx(high, low, 14)

       # Momentum
       rsi = close.rhoa.indicators.rsi(14)
       stoch = close.rhoa.indicators.stochastic(high, low, 14, 3)

       # Volatility
       bb = close.rhoa.indicators.bollinger_bands(20, 2.0)
       atr = close.rhoa.indicators.atr(high, low, 14)

       # Summary
       print("=" * 50)
       print("TECHNICAL ANALYSIS DASHBOARD")
       print("=" * 50)
       print(f"\nTrend:")
       print(f"  ADX: {adx['ADX'].iloc[-1]:.2f} "
             f"({'Strong' if adx['ADX'].iloc[-1] > 25 else 'Weak'} trend)")
       print(f"  MACD: {macd['macd'].iloc[-1]:.2f}")

       print(f"\nMomentum:")
       print(f"  RSI: {rsi.iloc[-1]:.2f}")
       print(f"  Stochastic %K: {stoch['%K'].iloc[-1]:.2f}")

       print(f"\nVolatility:")
       print(f"  ATR: {atr.iloc[-1]:.2f}")
       print(f"  BB Width: {(bb['upper_band'].iloc[-1] - bb['lower_band'].iloc[-1]):.2f}")

       return df

   # Use it
   df = pd.read_csv('prices.csv')
   df = technical_dashboard(df)

Next Steps
----------

Continue to :doc:`target_generation` to learn how to create optimized targets for
machine learning models.

Indicators Guide
================

This comprehensive guide covers all 13 technical indicators available in Rhoa, including when to use each, parameter tuning, mathematical formulas, and best practices.

Overview
--------

Rhoa provides 13 carefully selected technical indicators accessible through the ``.indicators`` accessor on both pandas DataFrame and Series objects. These indicators cover the four main categories:

- **Trend Indicators**: SMA, EWMA, ADX, Parabolic SAR
- **Momentum Indicators**: RSI, MACD, Stochastic, Williams %R
- **Volatility Indicators**: ATR, EWMV, EWMSTD, Bollinger Bands
- **Oscillators**: RSI, CCI, Stochastic, Williams %R

All indicators are implemented with:

- Industry-standard default parameters
- Full parameter customization
- Comprehensive documentation
- Type hints for IDE support
- Proper NaN handling

Accessing Indicators
--------------------

Rhoa offers two ways to access indicators:

**DataFrame accessor** (recommended for OHLC indicators) — auto-detects Close, High, and Low columns:

.. code-block:: python

   import pandas as pd
   import rhoa

   df = pd.read_csv('prices.csv')

   # OHLC indicators — no need to pass columns manually
   atr = df.rhoa.indicators.atr(window_size=14)
   stoch = df.rhoa.indicators.stochastic(k_window=14)
   adx_data = df.rhoa.indicators.adx(window_size=14)

   # Single-series indicators default to the Close column
   sma = df.rhoa.indicators.sma(window_size=20)
   rsi = df.rhoa.indicators.rsi(window_size=14)

   # Override auto-detection with explicit params
   atr_custom = df.rhoa.indicators.atr(close=df['Adj Close'], window_size=14)

**Series accessor** — call on a single column and pass others explicitly:

.. code-block:: python

   # Series-level: you choose which column to call on
   sma = df['Close'].rhoa.indicators.sma(window_size=20)

   # OHLC indicators require passing high/low explicitly
   atr = df['Close'].rhoa.indicators.atr(df['High'], df['Low'], window_size=14)

Column auto-detection is case-insensitive. For the close column, ``Adj Close`` is also checked as a fallback.

Quick Reference Table
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 20 45

   * - Indicator
     - Category
     - Default Period
     - Primary Use Case
   * - SMA
     - Trend
     - 20
     - Identify trend direction, support/resistance
   * - EWMA
     - Trend
     - 20
     - Faster trend following than SMA
   * - EWMV
     - Volatility
     - 20
     - Measure variance with recent emphasis
   * - EWMSTD
     - Volatility
     - 20
     - Measure volatility for risk management
   * - RSI
     - Momentum/Oscillator
     - 14
     - Overbought/oversold conditions
   * - MACD
     - Momentum
     - 12/26/9
     - Trend changes and momentum shifts
   * - Bollinger Bands
     - Volatility
     - 20, 2σ
     - Volatility bands, mean reversion
   * - ATR
     - Volatility
     - 14
     - Position sizing, stop-loss placement
   * - CCI
     - Oscillator
     - 20
     - Cyclical turning points
   * - Stochastic
     - Momentum/Oscillator
     - 14, 3
     - Overbought/oversold, momentum
   * - Williams %R
     - Momentum/Oscillator
     - 14
     - Overbought/oversold conditions
   * - ADX
     - Trend
     - 14
     - Trend strength measurement
   * - Parabolic SAR
     - Trend
     - 0.02/0.2
     - Trailing stops, trend reversals

Trend Indicators
----------------

Simple Moving Average (SMA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Smooth price data to identify the underlying trend direction.

**Mathematical Formula**:

.. math::

   SMA_t = \frac{1}{n} \sum_{i=0}^{n-1} Price_{t-i}

Where:
- :math:`n` = window size
- :math:`Price_t` = price at time t

**Parameters**:

- ``window_size`` (int, default=20): Number of periods to average
- ``min_periods`` (int, optional): Minimum observations required
- ``center`` (bool, default=False): Center the window

**Code Example**:

.. code-block:: python

   import pandas as pd
   import rhoa

   # Load price data
   df = pd.read_csv('prices.csv')

   # Basic SMA
   sma_20 = df.rhoa.indicators.sma(window_size=20)

   # Multiple SMAs for crossover strategy
   sma_50 = df.rhoa.indicators.sma(window_size=50)
   sma_200 = df.rhoa.indicators.sma(window_size=200)

   # Golden cross: SMA 50 crosses above SMA 200
   golden_cross = (sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1))

**When to Use**:

- Identifying long-term trend direction
- Support and resistance levels
- Crossover strategies (fast SMA crosses slow SMA)
- Filtering out market noise

**Best Practices**:

- Use 20-period for short-term, 50 for medium-term, 200 for long-term
- Combine multiple SMAs for confirmation
- SMA lags price, not suitable for fast trading
- Works best in trending markets, less effective in sideways markets

**Interpretation**:

- Price above SMA: Uptrend
- Price below SMA: Downtrend
- SMA slope: Trend strength (steep = strong)
- SMA acting as support (uptrend) or resistance (downtrend)

Exponential Weighted Moving Average (EWMA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Moving average that gives more weight to recent prices, responding faster to changes than SMA.

**Mathematical Formula**:

.. math::

   EWMA_t = \alpha \cdot Price_t + (1 - \alpha) \cdot EWMA_{t-1}

Where:
- :math:`\alpha = \frac{2}{span + 1}` (smoothing factor)
- :math:`span` = window_size

**Parameters**:

- ``window_size`` (int, default=20): Span of the exponential window
- ``adjust`` (bool, default=False): Normalize weights by sum
- ``min_periods`` (int, optional): Minimum observations required

**Code Example**:

.. code-block:: python

   # Basic EWMA
   ewma_20 = df.rhoa.indicators.ewma(window_size=20)

   # Multiple EMAs for trend confirmation
   ema_12 = df.rhoa.indicators.ewma(window_size=12)
   ema_26 = df.rhoa.indicators.ewma(window_size=26)

   # EMA crossover strategy
   bullish = (ema_12 > ema_26) & (ema_12.shift(1) <= ema_26.shift(1))

**When to Use**:

- When you need faster response to price changes than SMA
- Short-term trading strategies
- As a component of MACD
- Dynamic support/resistance levels

**Best Practices**:

- Use shorter periods (12, 26) for active trading
- EWMA responds faster but is more sensitive to noise
- Combine with other indicators for confirmation
- More suitable than SMA for volatile markets

**Interpretation**:

- Same as SMA but more responsive
- Crossovers happen earlier than SMA
- Better for catching trend changes
- Can generate more false signals in choppy markets

Average Directional Index (ADX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Measure the strength of a trend, regardless of direction.

**Mathematical Formula**:

.. math::

   +DM = High_t - High_{t-1} \text{ (if positive and greater than -DM)}

   -DM = Low_{t-1} - Low_t \text{ (if positive and greater than +DM)}

   +DI = 100 \times \frac{EMA(+DM)}{ATR}

   -DI = 100 \times \frac{EMA(-DM)}{ATR}

   DX = 100 \times \frac{|+DI - -DI|}{+DI + -DI}

   ADX = EMA(DX)

**Parameters**:

- ``high`` (Series): High prices
- ``low`` (Series): Low prices
- ``window_size`` (int, default=14): Smoothing period

**Code Example**:

.. code-block:: python

   # Calculate ADX
   adx_data = df.rhoa.indicators.adx(window_size=14)

   adx = adx_data['ADX']
   plus_di = adx_data['+DI']
   minus_di = adx_data['-DI']

   # Strong trend identification
   strong_trend = adx > 25

   # Trend direction
   uptrend = (plus_di > minus_di) & strong_trend
   downtrend = (minus_di > plus_di) & strong_trend

**When to Use**:

- Determining if a market is trending or ranging
- Filtering trades (only trade in strong trends)
- Confirming breakouts
- Avoiding choppy, directionless markets

**Best Practices**:

- ADX > 25: Strong trend (good for trend-following strategies)
- ADX < 20: Weak trend or ranging (good for mean reversion)
- Use +DI/-DI crossovers for direction, ADX for strength
- Rising ADX confirms trend strengthening
- Falling ADX indicates trend weakening

**Interpretation**:

- 0-20: Weak or absent trend
- 20-25: Emerging trend
- 25-50: Strong trend
- 50-75: Very strong trend
- 75-100: Extremely strong trend (rare)

Parabolic SAR
~~~~~~~~~~~~~

**Purpose**: Identify potential reversal points and provide trailing stop levels.

**Parameters**:

- ``high`` (Series): High prices
- ``low`` (Series): Low prices
- ``af_start`` (float, default=0.02): Initial acceleration factor
- ``af_increment`` (float, default=0.02): AF increase per new extreme
- ``af_maximum`` (float, default=0.2): Maximum AF value

**Code Example**:

.. code-block:: python

   # Calculate Parabolic SAR
   sar = df.rhoa.indicators.parabolic_sar(
       af_start=0.02,
       af_increment=0.02,
       af_maximum=0.2
   )

   # Trend identification
   uptrend = df['Close'] > sar
   downtrend = df['Close'] < sar

   # Detect trend reversals
   reversal_up = (df['Close'] > sar) & (df['Close'].shift(1) <= sar.shift(1))
   reversal_down = (df['Close'] < sar) & (df['Close'].shift(1) >= sar.shift(1))

   # Use as trailing stop
   stop_loss = sar

**When to Use**:

- Identifying trend direction
- Setting trailing stop-loss levels
- Spotting potential reversal points
- Trend-following strategies

**Best Practices**:

- SAR works best in trending markets
- Generates many false signals in sideways markets
- Combine with ADX (trade SAR only when ADX > 25)
- Use conservative AF values for long-term trends
- Increase AF for more responsive signals (but more whipsaws)

**Interpretation**:

- Dots below price: Uptrend (long position)
- Dots above price: Downtrend (short position)
- SAR flip: Potential trend reversal
- Use SAR dots as stop-loss levels

Momentum Indicators
-------------------

Relative Strength Index (RSI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Measure momentum and identify overbought/oversold conditions.

**Mathematical Formula**:

.. math::

   RS = \frac{EMA(Gains)}{EMA(Losses)}

   RSI = 100 - \frac{100}{1 + RS}

**Parameters**:

- ``window_size`` (int, default=14): Period for calculation
- ``edge_case_value`` (float, default=100.0): Value when no losses occur

**Code Example**:

.. code-block:: python

   # Basic RSI
   rsi = df.rhoa.indicators.rsi(window_size=14)

   # Identify overbought/oversold
   overbought = rsi > 70
   oversold = rsi < 30

   # RSI divergence (advanced)
   # Price makes new high but RSI doesn't
   price_high = df['Close'] == df['Close'].rolling(20).max()
   rsi_lower = rsi < rsi.shift(20)
   bearish_divergence = price_high & rsi_lower

**When to Use**:

- Identifying overbought/oversold conditions
- Spotting momentum divergences
- Confirming trend strength
- Mean reversion strategies

**Best Practices**:

- 70/30 levels for general use
- 80/20 levels for strong trends
- 60/40 levels for ranging markets
- Wait for RSI to exit extreme zones before trading
- Combine with price action for confirmation

**Interpretation**:

- 0-30: Oversold (potential buy signal)
- 30-70: Neutral zone
- 70-100: Overbought (potential sell signal)
- Above 50: Bullish momentum
- Below 50: Bearish momentum

MACD (Moving Average Convergence Divergence)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Identify trend changes and momentum shifts through moving average relationships.

**Mathematical Formula**:

.. math::

   MACD = EMA_{12} - EMA_{26}

   Signal = EMA_9(MACD)

   Histogram = MACD - Signal

**Parameters**:

- ``short_window`` (int, default=12): Fast EMA period
- ``long_window`` (int, default=26): Slow EMA period
- ``signal_window`` (int, default=9): Signal line EMA period

**Code Example**:

.. code-block:: python

   # Calculate MACD
   macd_data = df.rhoa.indicators.macd(
       short_window=12,
       long_window=26,
       signal_window=9
   )

   macd_line = macd_data['macd']
   signal_line = macd_data['signal']
   histogram = macd_data['histogram']

   # Bullish crossover
   bullish = (macd_line > signal_line) & \
             (macd_line.shift(1) <= signal_line.shift(1))

   # Bearish crossover
   bearish = (macd_line < signal_line) & \
             (macd_line.shift(1) >= signal_line.shift(1))

   # Histogram growing = strengthening trend
   momentum_increasing = histogram > histogram.shift(1)

**When to Use**:

- Identifying trend reversals
- Confirming trend direction
- Measuring momentum strength
- Finding divergences

**Best Practices**:

- Most reliable in trending markets
- Use histogram for early momentum signals
- Look for MACD-price divergences
- Combine with other indicators for confirmation
- Adjust periods for different timeframes

**Interpretation**:

- MACD > Signal: Bullish momentum
- MACD < Signal: Bearish momentum
- MACD crosses above Signal: Buy signal
- MACD crosses below Signal: Sell signal
- Histogram expanding: Momentum strengthening
- Histogram contracting: Momentum weakening

Stochastic Oscillator
~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Compare closing price to price range over time to identify momentum.

**Mathematical Formula**:

.. math::

   \%K = 100 \times \frac{Close - Lowest\_Low}{Highest\_High - Lowest\_Low}

   \%D = SMA_3(\%K)

**Parameters**:

- ``high`` (Series): High prices
- ``low`` (Series): Low prices
- ``k_window`` (int, default=14): %K calculation period
- ``d_window`` (int, default=3): %D smoothing period

**Code Example**:

.. code-block:: python

   # Calculate Stochastic
   stoch = df.rhoa.indicators.stochastic(
       k_window=14,
       d_window=3
   )

   k_percent = stoch['%K']
   d_percent = stoch['%D']

   # Overbought/oversold
   overbought = k_percent > 80
   oversold = k_percent < 20

   # Crossover signals
   bullish = (k_percent > d_percent) & \
             (k_percent.shift(1) <= d_percent.shift(1)) & \
             (k_percent < 50)

**When to Use**:

- Identifying overbought/oversold conditions
- Confirming trend reversals
- Finding divergences
- Range-bound markets

**Best Practices**:

- 80/20 levels for general use
- Look for %K crossing %D in extreme zones
- More reliable in ranging markets
- Can give premature signals in strong trends
- Use with trend confirmation

**Interpretation**:

- 0-20: Oversold zone
- 80-100: Overbought zone
- %K above %D: Bullish
- %K below %D: Bearish
- Crossovers in extreme zones: Strong signals

Williams %R
~~~~~~~~~~~

**Purpose**: Momentum indicator measuring overbought/oversold levels (inverted Stochastic).

**Mathematical Formula**:

.. math::

   Williams\%R = -100 \times \frac{Highest\_High - Close}{Highest\_High - Lowest\_Low}

**Parameters**:

- ``high`` (Series): High prices
- ``low`` (Series): Low prices
- ``window_size`` (int, default=14): Lookback period

**Code Example**:

.. code-block:: python

   # Calculate Williams %R
   wr = df.rhoa.indicators.williams_r(window_size=14)

   # Identify conditions
   overbought = wr > -20
   oversold = wr < -80

   # Exit signals
   exit_overbought = (wr < -20) & (wr.shift(1) >= -20)
   exit_oversold = (wr > -80) & (wr.shift(1) <= -80)

**When to Use**:

- Quick overbought/oversold identification
- Short-term trading signals
- Confirming price momentum
- Similar use cases to Stochastic

**Best Practices**:

- -20/-80 are traditional levels
- Look for exits from extreme zones
- Works well for short-term reversals
- Combine with price action
- Can stay extreme during strong trends

**Interpretation**:

- 0 to -20: Overbought
- -20 to -80: Normal range
- -80 to -100: Oversold
- Rising from oversold: Bullish
- Falling from overbought: Bearish

Volatility Indicators
---------------------

Exponentially Weighted Moving Variance (EWMV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Measure variance with more weight on recent observations.

**Parameters**:

- ``window_size`` (int, default=20): Span for weighting
- ``adjust`` (bool, default=True): Normalize weights
- ``min_periods`` (int, optional): Minimum observations

**Code Example**:

.. code-block:: python

   # Calculate EWMV
   ewmv = df.rhoa.indicators.ewmv(window_size=20)

   # Identify high volatility periods
   mean_variance = ewmv.rolling(50).mean()
   high_volatility = ewmv > mean_variance * 1.5

   # Volatility regime changes
   volatility_increasing = ewmv > ewmv.shift(5)

**When to Use**:

- Risk measurement
- Volatility regime detection
- Position sizing
- Option pricing adjustments

**Best Practices**:

- Use relative comparisons (current vs. historical)
- Pair with price action
- Higher variance = higher risk
- Consider different timeframes

Exponentially Weighted Moving Standard Deviation (EWMSTD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Measure volatility with exponential weighting (square root of EWMV).

**Parameters**:

- ``window_size`` (int, default=20): Span for weighting
- ``adjust`` (bool, default=True): Normalize weights
- ``min_periods`` (int, optional): Minimum observations

**Code Example**:

.. code-block:: python

   # Calculate EWMSTD
   ewmstd = df.rhoa.indicators.ewmstd(window_size=20)

   # Position sizing based on volatility
   base_position = 10000
   target_volatility = 0.02
   position_size = (base_position * target_volatility) / ewmstd

   # Stop-loss based on volatility
   stop_distance = 2 * ewmstd

**When to Use**:

- Position sizing
- Stop-loss calculation
- Risk management
- Volatility-adjusted strategies

**Best Practices**:

- Use for ATR-style position sizing
- Scale positions inversely with volatility
- Combine with price levels for stops
- Monitor for volatility spikes

Bollinger Bands
~~~~~~~~~~~~~~~

**Purpose**: Volatility bands showing standard deviation channels around moving average.

**Mathematical Formula**:

.. math::

   Middle = SMA(Close, n)

   Upper = Middle + (k \times \sigma)

   Lower = Middle - (k \times \sigma)

Where:
- :math:`n` = window_size
- :math:`k` = num_std
- :math:`\sigma` = standard deviation

**Parameters**:

- ``window_size`` (int, default=20): SMA and std period
- ``num_std`` (float, default=2.0): Number of standard deviations
- ``min_periods`` (int, optional): Minimum observations
- ``center`` (bool, default=False): Center the window

**Code Example**:

.. code-block:: python

   # Calculate Bollinger Bands
   bb = df.rhoa.indicators.bollinger_bands(
       window_size=20,
       num_std=2.0
   )

   upper = bb['upper_band']
   middle = bb['middle_band']
   lower = bb['lower_band']

   # Band squeeze (low volatility)
   band_width = upper - lower
   squeeze = band_width < band_width.rolling(50).quantile(0.25)

   # Mean reversion signals
   oversold = df['Close'] <= lower
   overbought = df['Close'] >= upper

   # Trend following: price riding upper band
   strong_uptrend = (df['Close'] > middle) & (df['Close'].shift(1) > middle.shift(1))

**When to Use**:

- Mean reversion strategies
- Volatility analysis
- Breakout identification
- Overbought/oversold conditions

**Best Practices**:

- 20-period, 2 std is standard
- Narrow bands = low volatility (squeeze)
- Wide bands = high volatility (expansion)
- Price at bands is not necessarily reversal signal
- Look for band squeeze followed by expansion
- Combine with other indicators

**Interpretation**:

- Price touching upper band: Strong upward momentum
- Price touching lower band: Strong downward momentum
- Band squeeze: Consolidation, potential breakout ahead
- Band expansion: High volatility, strong trending
- Walking the bands: Very strong trend

Average True Range (ATR)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Measure market volatility using true range.

**Mathematical Formula**:

.. math::

   True\ Range = max[(High - Low), |High - Close_{prev}|, |Low - Close_{prev}|]

   ATR = SMA(True\ Range, n)

**Parameters**:

- ``high`` (Series): High prices
- ``low`` (Series): Low prices
- ``window_size`` (int, default=14): Averaging period

**Code Example**:

.. code-block:: python

   # Calculate ATR
   atr = df.rhoa.indicators.atr(window_size=14)

   # Position sizing
   risk_per_trade = 1000  # $1000 risk per trade
   atr_multiple = 2  # Stop at 2x ATR
   position_size = risk_per_trade / (atr_multiple * atr)

   # Dynamic stop-loss
   stop_loss_long = df['Close'] - (2 * atr)
   stop_loss_short = df['Close'] + (2 * atr)

   # Profit targets
   profit_target_long = df['Close'] + (3 * atr)

**When to Use**:

- Position sizing
- Stop-loss placement
- Profit target setting
- Volatility filtering

**Best Practices**:

- Use 1.5-3x ATR for stops
- Use 2-5x ATR for targets
- Scale position size inversely with ATR
- Higher ATR = wider stops needed
- Compare current ATR to historical average

**Interpretation**:

- High ATR: High volatility, wider stops needed
- Low ATR: Low volatility, tighter stops possible
- Rising ATR: Volatility increasing
- Falling ATR: Volatility decreasing

Commodity Channel Index (CCI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Identify cyclical trends and overbought/oversold conditions.

**Mathematical Formula**:

.. math::

   Typical\ Price = \frac{High + Low + Close}{3}

   CCI = \frac{Typical\ Price - SMA(Typical\ Price)}{0.015 \times Mean\ Deviation}

**Parameters**:

- ``high`` (Series): High prices
- ``low`` (Series): Low prices
- ``window_size`` (int, default=20): Calculation period

**Code Example**:

.. code-block:: python

   # Calculate CCI
   cci = df.rhoa.indicators.cci(window_size=20)

   # Traditional signals
   overbought = cci > 100
   oversold = cci < -100

   # Extreme signals
   very_overbought = cci > 200
   very_oversold = cci < -200

   # Zero-line crosses
   bullish = (cci > 0) & (cci.shift(1) <= 0)
   bearish = (cci < 0) & (cci.shift(1) >= 0)

**When to Use**:

- Identifying cyclical turning points
- Overbought/oversold detection
- Trend strength confirmation
- Divergence analysis

**Best Practices**:

- +100/-100 are traditional levels
- Can exceed 100 in strong trends
- Look for exits from extreme zones
- Use with price action confirmation
- Works well in cyclical markets

**Interpretation**:

- CCI > +100: Overbought, strong uptrend
- CCI < -100: Oversold, strong downtrend
- CCI near 0: Neutral
- Rising CCI: Increasing momentum
- Falling CCI: Decreasing momentum

Combining Indicators
--------------------

Multi-Indicator Strategy Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import rhoa

   # Load data
   df = pd.read_csv('stock_data.csv')

   # Calculate multiple indicators
   sma_50 = df.rhoa.indicators.sma(50)
   sma_200 = df.rhoa.indicators.sma(200)
   rsi = df.rhoa.indicators.rsi(14)
   macd_data = df.rhoa.indicators.macd()
   adx_data = df.rhoa.indicators.adx(window_size=14)
   atr = df.rhoa.indicators.atr(window_size=14)

   # Define trading conditions
   # 1. Trend: Price above 50 SMA, 50 SMA above 200 SMA
   uptrend = (df['Close'] > sma_50) & (sma_50 > sma_200)

   # 2. Momentum: MACD bullish, RSI not overbought
   momentum_good = (macd_data['macd'] > macd_data['signal']) & \
                   (rsi < 70) & (rsi > 30)

   # 3. Trend strength: ADX confirms strong trend
   strong_trend = adx_data['ADX'] > 25

   # 4. Combined entry signal
   buy_signal = uptrend & momentum_good & strong_trend

   # 5. Position sizing and risk management
   risk_amount = 1000  # Risk $1000 per trade
   stop_distance = 2 * atr
   position_size = risk_amount / stop_distance

   # 6. Exit signals
   stop_loss = df['Close'] - stop_distance
   profit_target = df['Close'] + (3 * stop_distance)  # 3:1 reward:risk

Best Practices Summary
----------------------

General Guidelines
~~~~~~~~~~~~~~~~~~

1. **Never Use a Single Indicator**
   - Always confirm with multiple indicators
   - Combine trend, momentum, and volatility
   - Use indicators from different categories

2. **Understand Market Context**
   - Trending vs. ranging markets
   - High vs. low volatility environments
   - Different indicators work in different conditions

3. **Adjust Parameters**
   - Shorter periods for active trading
   - Longer periods for position trading
   - Test different parameters on your data

4. **Handle NaN Values Properly**
   - All indicators create NaN for initial periods
   - Use ``dropna()`` or ``fillna()`` appropriately
   - Consider ``min_periods`` parameter

5. **Backtest Everything**
   - Never trade on indicators alone
   - Backtest your indicator combinations
   - Use proper walk-forward validation

Common Mistakes to Avoid
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Indicator Overload**
   - Using too many correlated indicators
   - All momentum indicators will agree
   - 3-5 indicators maximum

2. **Ignoring Market Regime**
   - RSI works great in ranges, poor in trends
   - MACD works great in trends, poor in ranges
   - Match indicator to market condition

3. **Not Considering Timeframes**
   - 14-period RSI on 1-minute chart ≠ daily chart
   - Adjust parameters for your timeframe
   - Use multiple timeframe analysis

4. **Forgetting Transaction Costs**
   - More signals = more commissions
   - Slippage on every trade
   - Factor costs into backtest

5. **Curve Fitting**
   - Over-optimizing parameters
   - Use standard parameters first
   - Test on out-of-sample data

Performance Tips
----------------

For Large Datasets
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Pre-calculate and store
   df['SMA_50'] = df.rhoa.indicators.sma(50)
   df['RSI'] = df.rhoa.indicators.rsi(14)

   # Rather than recalculating each time
   # sma = df.rhoa.indicators.sma(50)  # Recalculates

Using with NumPy
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Indicators return pandas Series
   rsi = df.rhoa.indicators.rsi(14)

   # Convert to numpy for faster operations
   rsi_array = rsi.values

   # Or use pandas optimized operations
   above_70 = (rsi > 70).sum()  # Fast pandas operation

Further Reading
---------------

For more information:

- :doc:`/examples/basic_indicators` - Hands-on examples
- :doc:`/examples/advanced_indicators` - Advanced techniques
- :doc:`targets_guide` - Using indicators for ML
- :doc:`visualization_guide` - Plotting indicators
- :doc:`/api/indicators` - Complete API reference

Common Questions
----------------

**Q: Which indicator is best?**
   No single "best" indicator exists. Use combinations based on market conditions and strategy.

**Q: Why do my indicators give different values than TradingView?**
   Small differences may occur due to calculation methods, especially for exponential averages. Rhoa uses industry-standard formulas.

**Q: Can I use indicators on intraday data?**
   Yes, indicators work on any timeframe. Adjust parameters accordingly (shorter periods for intraday).

**Q: How do I choose parameter values?**
   Start with defaults, then optimize on training data. Validate on separate test data. Avoid overfitting.

**Q: Should I normalize indicator values?**
   Some indicators (RSI, Stochastic) are already normalized (0-100). Others benefit from normalization for ML.

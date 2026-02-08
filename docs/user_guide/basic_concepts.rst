Basic Concepts
==============

This guide introduces fundamental concepts you need to understand before using Rhoa effectively.

Pandas DataFrame Extension
---------------------------

Rhoa extends pandas using the **accessor API**, which allows adding custom methods to pandas objects.

How Accessors Work
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import rhoa  # This registers the accessors

   # Now Series objects have .indicators accessor
   prices = pd.Series([100, 102, 105, 103, 107])
   sma = prices.indicators.sma(window_size=3)

   # DataFrame objects have .plots accessor
   df.plots.signal(y_pred=predictions, y_true=targets)

**Key Points:**
- You must import rhoa to register the accessors
- Accessors feel like native pandas methods
- They return standard pandas objects (Series, DataFrame)
- Perfect for method chaining

Technical Indicators
--------------------

Technical indicators are mathematical calculations based on historical price, volume, or open interest.

Categories of Indicators
~~~~~~~~~~~~~~~~~~~~~~~~~

**Trend Indicators**
   Show the direction of price movement (up, down, sideways).

   Examples: SMA, EMA, ADX, Parabolic SAR

**Momentum Indicators**
   Measure the speed and strength of price movements.

   Examples: RSI, MACD, Stochastic, Williams %R

**Volatility Indicators**
   Measure the rate of price changes (high/low volatility).

   Examples: ATR, Bollinger Bands, Standard Deviation

**Oscillators**
   Bounded indicators that fluctuate between fixed levels.

   Examples: RSI (0-100), Stochastic (0-100), CCI

Indicator Properties
~~~~~~~~~~~~~~~~~~~~

**Window Size (Period)**
   Most indicators use a rolling window. Larger windows = smoother but more lag.

   .. code-block:: python

      sma_20 = prices.indicators.sma(window_size=20)  # Slower, smoother
      sma_5 = prices.indicators.sma(window_size=5)    # Faster, noisier

**NaN Values**
   Indicators create NaN for initial periods where insufficient data exists.

   .. code-block:: python

      sma_10 = prices.indicators.sma(window_size=10)
      # First 9 values will be NaN

**Lagging vs. Leading**
   - **Lagging**: Based on past prices (most indicators)
   - **Leading**: Attempts to predict future moves (rare, often unreliable)

Time Series Considerations
---------------------------

Financial time series have unique properties that affect analysis.

Stationarity
~~~~~~~~~~~~

**Definition**: A stationary series has constant mean, variance, and autocorrelation over time.

**Why It Matters**: Most ML models assume stationarity. Raw prices are non-stationary.

**Solutions**:
- Use returns instead of prices
- Use indicators (already somewhat stationary)
- Apply differencing or detrending

.. code-block:: python

   # Non-stationary (price levels)
   prices = df['Close']

   # More stationary (returns)
   returns = df['Close'].pct_change()

   # Stationary indicator
   rsi = df['Close'].indicators.rsi(14)

Autocorrelation
~~~~~~~~~~~~~~~

Financial data is often autocorrelated (today's price depends on yesterday's).

**Implications**:
- Can't use random train/test splits
- Must use time-based splits
- Cross-validation requires special handling (TimeSeriesSplit)

**Correct Split**:

.. code-block:: python

   # Time-based split (correct)
   split_date = '2024-01-01'
   train = df[df['Date'] < split_date]
   test = df[df['Date'] >= split_date]

**Wrong Split**:

.. code-block:: python

   # Random split (WRONG for time series!)
   from sklearn.model_selection import train_test_split
   train, test = train_test_split(df)  # Don't do this!

Look-Ahead Bias
~~~~~~~~~~~~~~~

**Definition**: Using future information in training that wouldn't be available in production.

**Common Mistakes**:
1. Normalizing before splitting (uses future statistics)
2. Generating targets on full dataset then splitting
3. Using future price data in features

**Avoiding Bias**:

.. code-block:: python

   # WRONG: Normalize then split
   df_norm = (df - df.mean()) / df.std()
   train, test = split(df_norm)

   # CORRECT: Split then normalize
   train, test = split(df)
   train_norm = (train - train.mean()) / train.std()
   test_norm = (test - train.mean()) / train.std()  # Use TRAIN statistics!

Machine Learning for Trading
-----------------------------

Applying ML to trading requires special considerations.

Binary Classification
~~~~~~~~~~~~~~~~~~~~~~

Most trading strategies can be framed as binary classification:
- **Class 1**: Buy signal (price will increase enough to profit)
- **Class 0**: No signal (price won't increase enough, or will decrease)

**Key Decisions**:
- What threshold defines "enough to profit"?
- What time horizon to consider?
- How to handle transaction costs?

Rhoa's target generation addresses these questions systematically.

Class Imbalance
~~~~~~~~~~~~~~~

Real trading data often has severe class imbalance:
- True opportunities may be rare (5-20% of days)
- Too many positives = frequent trading, high costs
- Too few positives = model rarely trades

**Solutions**:
- Adjust target thresholds (Rhoa's auto mode)
- Use appropriate metrics (precision, recall, F1, not just accuracy)
- Consider cost-sensitive learning

.. code-block:: python

   # Control class balance
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       target_class_balance=0.3  # 30% positive instances
   )

Evaluation Metrics
~~~~~~~~~~~~~~~~~~

For trading strategies, different metrics matter:

**Precision (How many signals are correct?)**
   Critical for avoiding false trades and transaction costs.

**Recall (How many opportunities are caught?)**
   Important for not missing profitable trades.

**F1 Score (Harmonic mean of precision and recall)**
   Balances both concerns.

**Sharpe Ratio (Risk-adjusted returns)**
   The ultimate metric for trading strategies.

.. code-block:: python

   from sklearn.metrics import precision_score, recall_score, f1_score

   precision = precision_score(y_test, y_pred)  # Want this high!
   recall = recall_score(y_test, y_pred)        # And this!
   f1 = f1_score(y_test, y_pred)                # Compromise

Data Requirements
-----------------

Quality Over Quantity
~~~~~~~~~~~~~~~~~~~~~

**Minimum Data**:
- At least 200-500 data points for basic indicators
- 1000+ data points for reliable ML training
- More data for longer-period indicators

**Data Quality**:
- No missing values (or properly handled)
- Correct OHLC relationships (High ≥ Close ≥ Low)
- Adjusted for splits and dividends
- Consistent time intervals

Required Columns
~~~~~~~~~~~~~~~~

**For Basic Indicators**:
- Close price (minimum requirement)

**For Advanced Indicators**:
- Open, High, Low, Close (OHLC)
- Volume (optional but recommended)

**For ML**:
- Date/Timestamp column
- OHLC data
- Sufficient history

.. code-block:: python

   # Minimal DataFrame structure
   df = pd.DataFrame({
       'Date': pd.date_range('2023-01-01', periods=100),
       'Open': [100, 102, ...],
       'High': [105, 106, ...],
       'Low': [98, 100, ...],
       'Close': [103, 104, ...],
       'Volume': [1000000, 1200000, ...]
   })

Common Pitfalls
---------------

**Overfitting**
   - Using too many features
   - Training on too little data
   - Not using proper cross-validation
   - Optimizing on test set

**Data Leakage**
   - Using future information
   - Improper normalization
   - Including target in features

**Unrealistic Assumptions**
   - Ignoring transaction costs
   - Assuming perfect execution
   - Not accounting for slippage
   - Ignoring liquidity constraints

**Poor Risk Management**
   - No stop losses
   - Over-leveraging
   - No position sizing
   - Correlated positions

Next Steps
----------

Now that you understand the basics, dive into specific topics:

- :doc:`indicators_guide` - Learn about each indicator
- :doc:`targets_guide` - Master target generation
- :doc:`visualization_guide` - Evaluate your models
- :doc:`/examples/index` - See practical examples

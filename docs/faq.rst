Frequently Asked Questions
==========================

Common questions and answers about installing, using, and troubleshooting Rhoa.

Installation & Setup
--------------------

How do I install Rhoa?
~~~~~~~~~~~~~~~~~~~~~~

The simplest method is using pip:

.. code-block:: bash

   pip install rhoa

For all optional features:

.. code-block:: bash

   pip install rhoa[all]

See the :doc:`installation` guide for detailed instructions.

Why can't I import rhoa?
~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: ``ModuleNotFoundError: No module named 'rhoa'``

**Solutions**:

1. **Check installation**:

   .. code-block:: bash

      pip list | grep rhoa

   If not listed, install it:

   .. code-block:: bash

      pip install rhoa

2. **Check Python environment**:

   .. code-block:: bash

      # Make sure pip and python are from same environment
      which python
      which pip

3. **Verify in correct environment**:

   If using virtual environment:

   .. code-block:: bash

      source venv/bin/activate  # Activate first
      pip install rhoa

4. **Try upgrading pip**:

   .. code-block:: bash

      python -m pip install --upgrade pip
      pip install rhoa

Why doesn't the .indicators accessor work?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: ``AttributeError: 'Series' object has no attribute 'indicators'``

**Solution**: You must import rhoa to register the accessor:

.. code-block:: python

   import rhoa  # This line is required!
   import pandas as pd

   prices = pd.Series([100, 102, 104])
   sma = prices.rhoa.indicators.sma(20)  # Now works

**Why**: Rhoa uses pandas' accessor API, which requires importing to register.

What Python version do I need?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Requirement**: Python 3.9 or higher

**Check your version**:

.. code-block:: bash

   python --version

**If too old**:

.. code-block:: bash

   # Install newer Python
   # On Ubuntu/Debian:
   sudo apt-get install python3.9

   # On macOS with Homebrew:
   brew install python@3.9

   # On Windows: Download from python.org

What are the required dependencies?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Core requirements**:

- pandas >= 1.3
- numpy >= 1.21

**Optional for target generation**:

- kneed (for elbow method)
- paretoset (for Pareto optimization)

**Optional for visualization**:

- matplotlib
- seaborn
- scikit-learn (for confusion matrix)

**Check versions**:

.. code-block:: bash

   pip show pandas numpy matplotlib

Using Indicators
----------------

Why do my indicators return NaN values?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**This is normal behavior**. Indicators need a minimum number of observations to calculate.

**Example**:

.. code-block:: python

   prices = pd.Series([100, 102, 104, 106, 108])
   sma_20 = prices.rhoa.indicators.sma(window_size=20)
   # All values will be NaN because we only have 5 data points

**Solution**: Ensure you have enough data:

.. code-block:: python

   # For 20-period indicator, need at least 20 data points
   prices = pd.Series(range(100))  # 100 data points
   sma_20 = prices.rhoa.indicators.sma(window_size=20)
   # First 19 will be NaN, then valid values start

**Handling NaN**:

.. code-block:: python

   # Drop NaN values
   df_clean = df.dropna()

   # Or forward fill
   df['SMA_20'] = df['SMA_20'].fillna(method='ffill')

How do I choose indicator parameters?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**General guidelines**:

**For Window Size**:

- **Short-term trading**: 5-20 periods
- **Medium-term trading**: 20-50 periods
- **Long-term trading**: 50-200 periods

**Start with defaults**:

.. code-block:: python

   # These are industry standards
   rsi = prices.rhoa.indicators.rsi(window_size=14)        # Standard
   macd = prices.rhoa.indicators.macd(12, 26, 9)           # Standard
   bb = prices.rhoa.indicators.bollinger_bands(20, 2.0)    # Standard

**Then optimize if needed**:

.. code-block:: python

   # Test different values
   for window in [10, 14, 20]:
       rsi = prices.rhoa.indicators.rsi(window_size=window)
       # Evaluate performance
       score = evaluate_strategy(rsi)
       print(f"Window {window}: {score}")

**Avoid over-optimization**: Don't tune parameters too much on training data (overfitting).

Why do my indicators give different values than TradingView?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Small differences can occur due to:

1. **Calculation method**: Rhoa uses exponential weighting (EWM) for smoothing, while some platforms use different methods.

2. **Data alignment**: Check that dates and prices match exactly.

3. **Rounding**: Different precision in calculations.

**Verify with reference**:

.. code-block:: python

   # Check MACD calculation step by step
   prices = pd.Series([100, 102, 104, ...])

   ema_12 = prices.ewm(span=12, adjust=False).mean()
   ema_26 = prices.ewm(span=26, adjust=False).mean()
   macd_line = ema_12 - ema_26
   signal_line = macd_line.ewm(span=9, adjust=False).mean()

   # Compare with your reference source

**Note**: Rhoa uses industry-standard formulas. Small differences are normal and rarely significant for trading decisions.

Can I use indicators with intraday data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Yes!** Indicators work with any timeframe.

.. code-block:: python

   # Load 5-minute bars
   df_5min = pd.read_csv('intraday_5min.csv')

   # Calculate indicators (adjust window sizes for timeframe)
   df_5min['SMA_20'] = df_5min['Close'].rhoa.indicators.sma(20)  # 100 minutes
   df_5min['RSI'] = df_5min['Close'].rhoa.indicators.rsi(14)     # 70 minutes

**Parameter adjustment**:

- Daily data: 14-period RSI = 14 days
- 5-minute data: 14-period RSI = 70 minutes
- For similar effect, use more periods: 84 periods = ~7 hours

How do I combine multiple indicators?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate multiple indicators
   close = df['Close']
   high = df['High']
   low = df['Low']

   # Trend
   sma_50 = close.rhoa.indicators.sma(50)
   sma_200 = close.rhoa.indicators.sma(200)

   # Momentum
   rsi = close.rhoa.indicators.rsi(14)
   macd_data = close.rhoa.indicators.macd()

   # Volatility
   atr = close.rhoa.indicators.atr(high, low, 14)

   # Store in DataFrame
   df['SMA_50'] = sma_50
   df['SMA_200'] = sma_200
   df['RSI'] = rsi
   df['MACD'] = macd_data['macd']
   df['ATR'] = atr

   # Combine conditions
   uptrend = close > sma_50
   strong_trend = sma_50 > sma_200
   not_overbought = rsi < 70
   bullish_macd = macd_data['macd'] > macd_data['signal']

   buy_signal = uptrend & strong_trend & not_overbought & bullish_macd

See :doc:`user_guide/indicators_guide` for more details.

Target Generation
-----------------

What's the difference between auto and manual mode?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Auto Mode**:

- Searches across both period AND threshold
- Uses Pareto optimization
- Finds parameters to match target class balance
- Best for: Initial exploration, production systems

.. code-block:: python

   targets, meta = generate_target_combinations(
       df, mode='auto', target_class_balance=0.5
   )
   # Finds: period=6, threshold=4.2%

**Manual Mode**:

- Fixed period (you choose)
- Uses elbow method to find threshold
- Best for: Specific timeframes, hypothesis testing

.. code-block:: python

   targets, meta = generate_target_combinations(
       df, mode='manual', lookback_periods=5
   )
   # Uses: period=5, threshold=6.1% (elbow-detected)

**Recommendation**: Start with auto mode, then use manual mode to explore specific periods.

See :doc:`user_guide/targets_guide` for detailed comparison.

What is target_class_balance?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Definition**: The percentage of positive instances you want in your target.

.. code-block:: python

   # 50% positive (balanced)
   targets, meta = generate_target_combinations(
       df, mode='auto', target_class_balance=0.5
   )
   # Result: 512 positive, 512 negative

   # 30% positive (conservative)
   targets, meta = generate_target_combinations(
       df, mode='auto', target_class_balance=0.3
   )
   # Result: 307 positive, 717 negative

**Tradeoffs**:

- **Higher balance (0.7)**: More signals, more training data, but lower quality
- **Lower balance (0.3)**: Fewer signals, higher quality, but less training data
- **Balanced (0.5)**: Good starting point

**Consider**:

- Transaction costs (lower balance = fewer trades)
- Training data needs (higher balance = more examples)
- Risk tolerance (lower balance = more selective)

Which target method should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are 8 target methods. Here's a quick guide:

**For Training (want more data)**:

- **Method 7** (MaxHigh/Close): Most generous, maximum profit potential
- **Method 5** (MaxClose/Close): Slightly more conservative

.. code-block:: python

   # Training: Use generous target
   y_train = targets_train['Target_7']

**For Validation (want realism)**:

- **Method 1** (Close/Close): Most conservative, actual entry/exit
- **Method 3** (High/Close): Allows for intraday profit

.. code-block:: python

   # Validation: Use conservative target
   y_val = targets_val['Target_1']

**Why the difference?**:

- Train on generous targets → model learns from more examples
- Validate on conservative targets → realistic performance estimate

**Compare all methods**:

.. code-block:: python

   # See which method works best for your strategy
   results = {}
   for i in range(1, 9):
       y = targets[f'Target_{i}']
       model.fit(X_train, y)
       results[f'Method_{i}'] = model.score(X_test, y_test)

   print(results)

Why do my targets have NaN values?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**This is expected**. The last N rows will be NaN because we can't know the future.

.. code-block:: python

   targets, meta = generate_target_combinations(df, mode='auto')

   # If period=5, last 5 rows will be NaN
   print(targets.tail(10))
   #         Target_1  Target_2  ...
   # 995        True     False  ...
   # 996       False     False  ...
   # 997         NaN       NaN  ...  ← Last 5 rows
   # 998         NaN       NaN  ...
   # 999         NaN       NaN  ...

**Solution**: Drop NaN before training:

.. code-block:: python

   # Combine features and targets
   ml_data = pd.concat([features, targets], axis=1)

   # Drop NaN
   clean_data = ml_data.dropna()

   # Now split
   X = clean_data[feature_cols]
   y = clean_data['Target_1']

How do I apply the same parameters to test data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Important**: Generate targets on training data only, then apply parameters to test data.

.. code-block:: python

   # 1. Split FIRST
   train_df, test_df = train_test_split_by_date(df)

   # 2. Generate targets on training data
   targets_train, meta = generate_target_combinations(
       train_df, mode='auto'
   )

   # 3. Extract parameters for Method 7
   period = meta['method_7']['period']           # e.g., 6
   threshold = meta['method_7']['threshold']     # e.g., 4.2

   # 4. Apply same parameters to test data
   future_max_high = test_df['High'].shift(-period).rolling(
       window=period, min_periods=1
   ).max().shift(period)

   test_target = (future_max_high / test_df['Close'] - 1 >= threshold / 100)

**Why?** This avoids look-ahead bias - your model can't "see" the test data during target optimization.

See :doc:`user_guide/targets_guide` for complete workflow.

Visualization
-------------

How do I visualize my predictions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``.rhoa.plots.signal()`` method:

.. code-block:: python

   import rhoa

   # Get predictions
   y_pred = model.predict(X_test)
   y_true = test_df['Target']

   # Visualize
   fig = test_df.rhoa.plots.signal(
       y_pred=y_pred,
       y_true=y_true,
       date_col='Date',
       price_col='Close'
   )

This creates:

- Confusion matrix with precision/recall
- Price chart with signals overlaid
- False positive/negative identification

See :doc:`user_guide/visualization_guide` for details.

What do the colors mean in the visualization?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**On the price chart**:

- **Blue line**: Stock price
- **Light green background dots**: All true opportunities (if y_true provided)
- **Bright green dots**: Model's buy signals
- **Red X markers**: False positives (wrong predictions)
- **Orange circles**: False negatives (missed opportunities)

**In the confusion matrix**:

- Darker blue = more instances in that cell
- Numbers show count and percentage

How do I save the visualization?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``save_path`` parameter:

.. code-block:: python

   fig = df.rhoa.plots.signal(
       y_pred=predictions,
       y_true=targets,
       save_path='results/my_model.png',
       dpi=300  # High quality
   )

**Don't show, just save**:

.. code-block:: python

   fig = df.rhoa.plots.signal(
       y_pred=predictions,
       y_true=targets,
       save_path='results/my_model.png',
       show=False  # Don't display
   )

What if I don't have ground truth labels?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just omit ``y_true``:

.. code-block:: python

   # For future predictions (no ground truth yet)
   fig = df.rhoa.plots.signal(
       y_pred=predictions,
       # No y_true parameter
       date_col='Date',
       price_col='Close'
   )

This shows:

- Price chart with predicted signals
- No confusion matrix (can't evaluate without truth)

Machine Learning
----------------

Can I use scikit-learn with Rhoa?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Yes!** Rhoa is designed to work seamlessly with scikit-learn:

.. code-block:: python

   import pandas as pd
   import rhoa
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split

   # Create features using Rhoa
   df['SMA_20'] = df['Close'].rhoa.indicators.sma(20)
   df['RSI'] = df['Close'].rhoa.indicators.rsi(14)

   # Create targets using Rhoa
   from rhoa.targets import generate_target_combinations
   targets, meta = generate_target_combinations(df, mode='auto')

   # Use with scikit-learn
   X = df[['SMA_20', 'RSI']].dropna()
   y = targets['Target_7'].dropna()

   model = RandomForestClassifier()
   model.fit(X, y)

Why should I use time-based splits?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Financial data is autocorrelated**: Today's price depends on yesterday's price.

**Random splits leak information**:

.. code-block:: python

   # WRONG - random split
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y)
   # Training data contains future information!

**Time-based splits are correct**:

.. code-block:: python

   # CORRECT - time-based split
   split_idx = int(len(df) * 0.8)
   train = df[:split_idx]
   test = df[split_idx:]

   # Or use a specific date
   train = df[df['Date'] < '2024-01-01']
   test = df[df['Date'] >= '2024-01-01']

**Why it matters**:

- Random splits make your model look better than it is
- Time-based splits reflect real trading (you can't trade the past)
- Random splits = look-ahead bias = overfitting

See :doc:`user_guide/basic_concepts` for more on time series considerations.

How do I avoid overfitting?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Common causes of overfitting**:

1. **Too many features**
2. **Too little data**
3. **Model too complex**
4. **Optimizing on test set**

**Solutions**:

.. code-block:: python

   # 1. Use feature selection
   from sklearn.feature_selection import SelectKBest, f_classif

   selector = SelectKBest(f_classif, k=10)
   X_selected = selector.fit_transform(X_train, y_train)

   # 2. Use regularization
   from sklearn.linear_model import LogisticRegression

   model = LogisticRegression(C=0.1)  # Stronger regularization

   # 3. Use cross-validation (time-series aware)
   from sklearn.model_selection import TimeSeriesSplit

   tscv = TimeSeriesSplit(n_splits=5)
   for train_idx, val_idx in tscv.split(X):
       X_train_cv, X_val_cv = X[train_idx], X[val_idx]
       # Train and validate

   # 4. Simplify model
   from sklearn.ensemble import RandomForestClassifier

   # Instead of
   model = RandomForestClassifier(n_estimators=500, max_depth=20)

   # Use
   model = RandomForestClassifier(n_estimators=100, max_depth=5)

   # 5. Get more data
   # Collect more historical data if possible

How do I handle class imbalance?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rhoa's target generation handles this automatically:

.. code-block:: python

   # Control class balance
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       target_class_balance=0.5  # 50% positive
   )

**Additional techniques**:

.. code-block:: python

   # 1. Adjust class weights
   from sklearn.ensemble import RandomForestClassifier

   model = RandomForestClassifier(class_weight='balanced')
   model.fit(X, y)

   # 2. Use different evaluation metrics
   from sklearn.metrics import f1_score, precision_score, recall_score

   # Don't use accuracy for imbalanced data
   # Use precision, recall, or F1 instead

   # 3. Use SMOTE (synthetic oversampling)
   from imblearn.over_sampling import SMOTE

   smote = SMOTE()
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

   # 4. Adjust prediction threshold
   proba = model.predict_proba(X_test)[:, 1]
   y_pred = (proba > 0.7).astype(int)  # Higher threshold = fewer predictions

Performance & Optimization
--------------------------

Is Rhoa slow with large datasets?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rhoa uses pandas and numpy, which are optimized for large datasets. However:

**For indicators**:

.. code-block:: python

   # Indicators are fast - they use vectorized pandas operations
   %timeit prices.rhoa.indicators.sma(20)
   # ~1ms for 100k rows

   # Pre-calculate and store if using repeatedly
   df['SMA_20'] = df['Close'].rhoa.indicators.sma(20)  # Calculate once
   # Rather than
   # sma = df['Close'].rhoa.indicators.sma(20)  # Every time

**For target generation**:

.. code-block:: python

   # Auto mode searches parameter space - can be slow
   # Reduce search space for faster results
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       period_step=2,  # Check every 2 periods instead of 1
       step=2          # Check every 2% threshold instead of 1%
   )
   # 4x faster with minimal accuracy loss

**Cache results**:

.. code-block:: python

   import joblib

   # Save targets
   joblib.dump((targets, meta), 'targets_cache.pkl')

   # Load later
   targets, meta = joblib.load('targets_cache.pkl')

How do I optimize indicator calculations?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Batch calculation**:

.. code-block:: python

   # Calculate all indicators at once
   df['SMA_20'] = df['Close'].rhoa.indicators.sma(20)
   df['SMA_50'] = df['Close'].rhoa.indicators.sma(50)
   df['RSI'] = df['Close'].rhoa.indicators.rsi(14)

   # Store in DataFrame
   df.to_pickle('data_with_indicators.pkl')

   # Load when needed
   df = pd.read_pickle('data_with_indicators.pkl')

**Use appropriate window sizes**:

.. code-block:: python

   # Smaller windows = faster
   sma_5 = prices.rhoa.indicators.sma(5)    # Fast

   # Larger windows = slower
   sma_200 = prices.rhoa.indicators.sma(200)  # Slower

**Avoid recalculation**:

.. code-block:: python

   # SLOW - recalculates every time
   for i in range(100):
       sma = df['Close'].rhoa.indicators.sma(20)
       # Use sma

   # FAST - calculate once
   sma = df['Close'].rhoa.indicators.sma(20)
   for i in range(100):
       # Use sma

Data Issues
-----------

How do I handle missing data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Check for missing values**:

.. code-block:: python

   print(df.isnull().sum())
   # Close    5
   # High     3
   # Low      2

**Solutions**:

.. code-block:: python

   # 1. Drop rows with missing values
   df_clean = df.dropna()

   # 2. Forward fill (use previous value)
   df['Close'] = df['Close'].fillna(method='ffill')

   # 3. Backward fill
   df['Close'] = df['Close'].fillna(method='bfill')

   # 4. Interpolate
   df['Close'] = df['Close'].interpolate()

**For financial data**, forward fill is usually appropriate (use last known price).

**Avoid**:

.. code-block:: python

   # DON'T use mean/median for time series
   df['Close'].fillna(df['Close'].mean())  # WRONG!
   # This creates unrealistic prices

My OHLC data has inconsistencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Check OHLC relationships**:

.. code-block:: python

   # Verify data quality
   assert (df['High'] >= df['Close']).all(), "High should be >= Close"
   assert (df['Close'] >= df['Low']).all(), "Close should be >= Low"
   assert (df['High'] >= df['Low']).all(), "High should be >= Low"
   assert (df['High'] >= df['Open']).all(), "High should be >= Open"
   assert (df['Open'] >= df['Low']).all(), "Open should be >= Low"

**Fix inconsistencies**:

.. code-block:: python

   # Fix: Ensure High is maximum
   df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)

   # Fix: Ensure Low is minimum
   df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)

**Common causes**:

- Data provider errors
- Stock splits not adjusted
- Currency conversion issues
- Bad data entry

How do I handle stock splits?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use adjusted prices**:

.. code-block:: python

   # Most data providers offer adjusted prices
   # Yahoo Finance: Download with 'Adj Close'
   # These already account for splits and dividends

   import yfinance as yf
   df = yf.download('AAPL', start='2020-01-01')
   # Automatically adjusted for splits

**Manual adjustment**:

.. code-block:: python

   # If you have split information
   split_date = '2024-06-10'
   split_ratio = 4  # 4:1 split

   # Adjust prices before split date
   mask = df['Date'] < split_date
   df.loc[mask, ['Open', 'High', 'Low', 'Close']] /= split_ratio
   df.loc[mask, 'Volume'] *= split_ratio

Error Messages
--------------

"Index out of bounds" error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Likely cause**: Not enough data for indicator window.

.. code-block:: python

   # Error
   prices = pd.Series([100, 102])  # Only 2 values
   sma_20 = prices.rhoa.indicators.sma(20)  # Need 20 values!

**Solution**: Ensure sufficient data.

.. code-block:: python

   # Check data length
   print(f"Data points: {len(prices)}")
   print(f"Window size: 20")

   if len(prices) < 20:
       print("Not enough data!")

"Unable to parse string" in date column
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Date column not in correct format.

.. code-block:: python

   # Error
   df['Date'] = '2024-01-01'  # String, not datetime

**Solution**: Convert to datetime.

.. code-block:: python

   df['Date'] = pd.to_datetime(df['Date'])

   # Specify format if needed
   df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

   # For US format
   df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

"KeyError: 'Close'" or missing column error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Column name mismatch.

.. code-block:: python

   # Your data has 'close' (lowercase)
   print(df.columns)
   # ['Date', 'open', 'high', 'low', 'close']

**Solution**: Rename or specify column name.

.. code-block:: python

   # Option 1: Rename columns
   df.columns = ['Date', 'Open', 'High', 'Low', 'Close']

   # Option 2: Use lowercase
   sma = df['close'].rhoa.indicators.sma(20)

   # Option 3: Specify in function call
   targets, meta = generate_target_combinations(
       df,
       close_col='close',  # Specify your column name
       high_col='high'
   )

Getting Help
------------

Where can I find more examples?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`examples/index` - Comprehensive examples
- :doc:`user_guide/index` - Conceptual guides
- :doc:`api/index` - API reference
- `GitHub repository <https://github.com/nainajnahO/Rhoa>`_ - Source code and examples

How do I report a bug?
~~~~~~~~~~~~~~~~~~~~~~~

1. **Check if it's a known issue**: Search `GitHub Issues <https://github.com/nainajnahO/Rhoa/issues>`_

2. **Create a minimal reproducible example**:

   .. code-block:: python

      import pandas as pd
      import rhoa

      # Minimal code that reproduces the bug
      df = pd.DataFrame({
          'Close': [100, 102, 104, 106, 108]
      })
      result = df['Close'].rhoa.indicators.sma(20)  # Bug occurs here
      print(result)

3. **Include**:

   - Your Rhoa version: ``rhoa.__version__``
   - Your Python version: ``python --version``
   - Operating system
   - Full error traceback

4. **Submit**: Create an issue on GitHub with this information

What features are planned?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Upcoming features** (see roadmap):

- More indicators (volume-based, pattern recognition)
- Strategy backtesting framework
- Preprocessing utilities
- Data connectors (more exchanges/sources)
- Performance optimization

**Request features**: Open a GitHub issue with the "enhancement" label.

Additional Resources
--------------------

**Documentation**:

- :doc:`quickstart` - Get started quickly
- :doc:`user_guide/basic_concepts` - Fundamental concepts
- :doc:`user_guide/indicators_guide` - Indicator details
- :doc:`user_guide/targets_guide` - Target generation guide
- :doc:`user_guide/visualization_guide` - Visualization guide

**Support**:

- `GitHub Issues <https://github.com/nainajnahO/Rhoa/issues>`_ - Report bugs and request features
- `Examples <https://github.com/nainajnahO/Rhoa/tree/main/examples>`_ - More code samples

**Related Libraries**:

- `pandas <https://pandas.pydata.org/>`_ - Data manipulation
- `scikit-learn <https://scikit-learn.org/>`_ - Machine learning
- `matplotlib <https://matplotlib.org/>`_ - Visualization
- `yfinance <https://github.com/ranaroussi/yfinance>`_ - Market data

Still Have Questions?
---------------------

If your question isn't answered here:

1. **Search the documentation**: Use the search bar at the top
2. **Check GitHub Issues**: See if it's been reported or discussed
3. **Open a GitHub Issue**: Report bugs or ask questions

**When reporting issues**:

- Describe what you're trying to do
- Show your code (minimal example)
- Include any error messages
- Mention Rhoa version and Python version

We're here to help you succeed with Rhoa!

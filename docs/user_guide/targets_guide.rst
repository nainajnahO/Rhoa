Targets Guide
=============

A comprehensive guide to Rhoa's advanced target generation system for machine learning. Learn how to create optimized binary classification targets for trading strategies.

Overview
--------

Rhoa's ``generate_target_combinations`` function creates 8 different binary classification targets, each representing different ways to define a "profitable trade." The system automatically finds optimal parameters using either:

1. **Auto Mode**: Pareto optimization to find optimal period AND threshold
2. **Manual Mode**: Elbow method to find optimal threshold for fixed period

This guide explains both modes in depth, the mathematics behind them, and best practices for production use.

Why Target Generation Matters
------------------------------

The Problem
~~~~~~~~~~~

When building ML models for trading, you need to define what constitutes a "buy signal." This requires answering:

1. **How far ahead should the price move?** (lookback period)
2. **How much should it move?** (threshold percentage)
3. **Which price metric to use?** (close-to-close, high-to-close, etc.)

Poor choices lead to:

- **Too many signals**: High transaction costs, low precision
- **Too few signals**: Insufficient training data
- **Unrealistic targets**: Model learns patterns that don't translate to profit

The Solution
~~~~~~~~~~~~

Rhoa's target generation:

- Tests all 8 common target definitions
- Automatically finds optimal parameters
- Balances class distribution
- Provides detailed metadata
- Ensures reproducibility

Quick Start
-----------

Auto Mode (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rhoa.targets import generate_target_combinations
   import pandas as pd

   # Load your OHLC data
   df = pd.read_csv('stock_data.csv')

   # Generate optimized targets
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       target_class_balance=0.5  # 50% positive instances
   )

   # Check what was found
   print(f"Method 7 uses period {meta['method_7']['period']}, "
         f"threshold {meta['method_7']['threshold']}%")
   # Method 7 uses period 6, threshold 4.0%

   # Use in ML pipeline
   print(targets.head())
   #    Target_1  Target_2  Target_3  ...  Target_8
   # 0     False     False     False  ...     False
   # 1      True     False      True  ...      True

Manual Mode
~~~~~~~~~~~

.. code-block:: python

   # Fixed 5-day lookback, optimize thresholds
   targets, meta = generate_target_combinations(
       df,
       mode='manual',
       lookback_periods=5
   )

   # All methods use period=5
   print(meta['method_1'])
   # {'period': 5, 'threshold': 6.0, 'instances': 22, 'pct_of_max': 1.4}

The 8 Target Methods
--------------------

Each method defines "success" differently:

.. list-table::
   :header-rows: 1
   :widths: 10 35 25 30

   * - Method
     - Definition
     - Formula
     - Use Case
   * - 1
     - Close[N] / Close[0]
     - Future close vs. current close
     - Conservative, actual exit
   * - 2
     - Close[N] / High[0]
     - Future close vs. current high
     - Buy at top of range
   * - 3
     - High[N] / Close[0]
     - Future high vs. current close
     - Intraday profit potential
   * - 4
     - High[N] / High[0]
     - Future high vs. current high
     - Very conservative
   * - 5
     - MaxClose / Close[0]
     - Best close in period vs. current
     - Best exit timing
   * - 6
     - MaxClose / High[0]
     - Best close vs. current high
     - Optimal buy at top
   * - 7
     - MaxHigh / Close[0]
     - Best high in period vs. current
     - Maximum profit potential
   * - 8
     - MaxHigh / High[0]
     - Best high vs. current high
     - Ultra-conservative max

Method Details
~~~~~~~~~~~~~~

**Method 1: Close[N] / Close[0] - 1 >= threshold**

Most conservative. Represents buying at close today, selling at close N days later.

**Method 7: MaxHigh / Close[0] - 1 >= threshold**

Most generous. Represents the maximum profit achievable in the next N days, assuming perfect intraday timing.

**Recommendation**: Start with Method 7 for training data abundance, then validate with Method 1 for conservative estimates.

Auto Mode Deep Dive
-------------------

How Pareto Optimization Works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Auto mode searches across both period (time) and threshold (percentage) dimensions to find optimal combinations.

**Objective**: Find parameters that:

1. Maximize threshold (higher quality signals)
2. Minimize period (faster trades)
3. Achieve target class balance (e.g., 50% positive instances)

**Mathematical Formulation**:

For each method, we solve:

.. math::

   \text{maximize} \quad & threshold \\
   \text{minimize} \quad & period \\
   \text{minimize} \quad & |instances - target\_instances| \\
   \text{subject to} \quad & period \in [min\_period, max\_period] \\
                           & threshold \in [min\_pct, max\_pct]

This is a multi-objective optimization problem solved using Pareto frontier analysis.

The Algorithm
~~~~~~~~~~~~~

**Step 1: Find Maximum Instances**

For each method and period, calculate maximum possible instances (threshold=0%):

.. code-block:: python

   for period in range(1, 21):
       future_close = df['Close'].shift(-period)
       max_instances = (future_close / df['Close'] - 1 >= 0).sum()

**Step 2: Calculate Target Instances**

.. code-block:: python

   target_instances = max_instances * target_class_balance
   # e.g., max_instances=500, balance=0.5 → target=250

**Step 3: Search Parameter Space**

Test all combinations of period and threshold:

.. code-block:: python

   results = []
   for period in range(1, 21):
       for threshold_pct in range(0, 100):
           instances = count_instances(period, threshold_pct)
           deviation = abs(instances - target_instances)
           results.append({
               'period': period,
               'threshold': threshold_pct,
               'instances': instances,
               'deviation': deviation
           })

**Step 4: Pareto Optimization**

Find Pareto-optimal solutions (non-dominated solutions):

.. code-block:: python

   from paretoset import paretoset

   # Pareto optimization: max threshold, min period, min deviation
   data = results[['threshold', 'period', 'deviation']].values
   mask = paretoset(data, sense=["max", "min", "min"])
   pareto_solutions = results[mask]

**Step 5: Select Best Solution**

From Pareto frontier, choose solution closest to target balance:

.. code-block:: python

   pareto_solutions['distance'] = abs(
       pareto_solutions['instances'] / max_instances - target_class_balance
   )
   best = pareto_solutions.loc[pareto_solutions['distance'].idxmin()]

Pareto Frontier Example
~~~~~~~~~~~~~~~~~~~~~~~

Imagine these solutions for Method 7:

.. code-block:: text

   Period  Threshold  Instances  Deviation
   ------  ---------  ---------  ---------
      3       8.0        180        70      ← Pareto optimal
      5       6.0        210        40      ← Pareto optimal
      6       4.0        249         1      ← BEST (closest to 250)
     10       3.0        255         5      ← Pareto optimal
     15       2.0        240        10

Solutions at period 6, threshold 4.0% is selected because:

- It's on the Pareto frontier (not dominated)
- Instances (249) closest to target (250)
- Good trade-off: moderate period, reasonable threshold

Parameters Explained
~~~~~~~~~~~~~~~~~~~~

**target_class_balance** (float, default=0.5)
   Target percentage of positive instances.

   - 0.3 (30%): Conservative, higher quality signals
   - 0.5 (50%): Balanced, plenty of training data
   - 0.7 (70%): Aggressive, many signals

   **Example**:

   .. code-block:: python

      # Conservative: only 30% positive
      targets, meta = generate_target_combinations(
          df, mode='auto', target_class_balance=0.3
      )

**min_period / max_period** (int, defaults: 1/20)
   Period range to search.

   - Shorter periods (1-5): Day trading
   - Medium periods (5-15): Swing trading
   - Longer periods (15-30): Position trading

   **Example**:

   .. code-block:: python

      # Search only swing trading range
      targets, meta = generate_target_combinations(
          df,
          mode='auto',
          min_period=5,
          max_period=15
      )

**min_pct / max_pct / step** (int, defaults: 0/100/1)
   Threshold range and granularity.

   **Example**:

   .. code-block:: python

      # Search 2-10% in 0.5% increments
      targets, meta = generate_target_combinations(
          df,
          mode='auto',
          min_pct=2,
          max_pct=10,
          step=0.5  # Finer granularity
      )

Manual Mode Deep Dive
---------------------

How the Elbow Method Works
~~~~~~~~~~~~~~~~~~~~~~~~~~

Manual mode fixes the lookback period and finds the optimal threshold using the "elbow method."

**Concept**: As threshold increases, instances decrease. The "elbow" is where diminishing returns begin - the optimal balance between quality and quantity.

**Mathematical Basis**:

Plot instances vs. threshold:

.. code-block:: text

   Threshold (%)    Instances
   -------------    ---------
         0             500     ← Max instances
         1             450
         2             380
         3             300
         4             240     ← Elbow point
         5             220
         6             205
        ...            ...
        20               5     ← Very few

The Elbow Algorithm
~~~~~~~~~~~~~~~~~~~

**Step 1: Calculate Instances Across Thresholds**

.. code-block:: python

   from kneed import KneeLocator
   import numpy as np

   thresholds = np.arange(0, 100, 1)
   instances = []

   for threshold in thresholds:
       count = (future_price / current_price - 1 >= threshold/100).sum()
       instances.append(count)

**Step 2: Find Knee Point**

.. code-block:: python

   kn = KneeLocator(
       thresholds,
       instances,
       curve='convex',      # Curve shape
       direction='decreasing'  # Instances decrease as threshold increases
   )

   optimal_threshold = kn.elbow

**Step 3: Generate Targets**

Use the detected elbow threshold:

.. code-block:: python

   target = (future_price / current_price - 1 >= optimal_threshold / 100)

Visual Example
~~~~~~~~~~~~~~

.. code-block:: text

         Instances
          |
      500 |*
          |
      400 | *
          |
      300 |  *
          |   ╲
      200 |    *  ← Elbow at threshold ≈ 4%
          |     ╲
      100 |      ╲___
          |          ╲____
        0 |_______________╲____
          0   2   4   6   8   10  Threshold (%)

The elbow at 4% represents the optimal threshold where:
- Still have substantial instances (200)
- Threshold is meaningful (4% return)
- Diminishing returns begin beyond this point

Parameters Explained
~~~~~~~~~~~~~~~~~~~~

**lookback_periods** (int, default=5)
   Fixed number of periods to look forward.

   **Example**:

   .. code-block:: python

      # 10-day lookback
      targets, meta = generate_target_combinations(
          df,
          mode='manual',
          lookback_periods=10
      )

      # All methods use period=10
      for i in range(1, 9):
          assert meta[f'method_{i}']['period'] == 10

Metadata Structure
------------------

Understanding the Output
~~~~~~~~~~~~~~~~~~~~~~~~

The metadata dictionary contains rich information:

.. code-block:: python

   targets, meta = generate_target_combinations(df, mode='auto')

   # Metadata structure
   meta = {
       'mode': 'auto',  # or 'manual'
       'method_1': {
           'period': 5,          # Lookback period
           'threshold': 3.5,     # Threshold percentage
           'instances': 247,     # Number of positive instances
           'pct_of_max': 45.2    # Percentage of maximum possible instances
       },
       # ... method_2 through method_8 ...
   }

**Field Meanings**:

- ``period``: How many days/periods to look forward
- ``threshold``: Minimum return % required for positive label
- ``instances``: How many data points are positive
- ``pct_of_max``: What % of theoretical maximum this represents

Using Metadata
~~~~~~~~~~~~~~

**Compare Methods**:

.. code-block:: python

   import pandas as pd

   # Create comparison DataFrame
   comparison = pd.DataFrame([
       meta[f'method_{i}'] for i in range(1, 9)
   ])
   comparison.index = [f'Method_{i}' for i in range(1, 9)]

   print(comparison)
   #           period  threshold  instances  pct_of_max
   # Method_1       5        3.5        247        45.2
   # Method_2       4        5.0        198        38.1
   # ...

**Save for Reproducibility**:

.. code-block:: python

   import json

   # Save metadata
   with open('target_metadata.json', 'w') as f:
       json.dump(meta, f, indent=2)

   # Load later
   with open('target_metadata.json', 'r') as f:
       loaded_meta = json.load(f)

**Apply to New Data**:

.. code-block:: python

   # Apply same parameters to test set
   def apply_target_params(df, method_meta):
       period = method_meta['period']
       threshold = method_meta['threshold'] / 100

       future_close = df['Close'].shift(-period)
       return (future_close / df['Close'] - 1 >= threshold)

   # Apply Method 7 parameters to test data
   test_target = apply_target_params(test_df, meta['method_7'])

Choosing Between Modes
-----------------------

Use Auto Mode When:
~~~~~~~~~~~~~~~~~~~

- You want optimal parameters for your specific data
- Class balance is critical (e.g., balanced dataset for training)
- You're exploring different timeframes
- You need reproducible, data-driven decisions
- You have sufficient data (500+ rows)

**Example Use Cases**:

- Initial model development
- Production systems with regular retraining
- Research and strategy development

Use Manual Mode When:
~~~~~~~~~~~~~~~~~~~~~~

- You have a specific trading timeframe in mind
- You want to compare performance across fixed horizons
- You're validating a hypothesis
- You have domain knowledge about appropriate periods
- You want simpler, more interpretable parameters

**Example Use Cases**:

- Backtesting specific strategies (e.g., "5-day swing trades")
- Regulatory or operational constraints on holding periods
- Comparing different stocks on equal footing

Comparison Example
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Auto mode: Let optimizer decide everything
   auto_targets, auto_meta = generate_target_combinations(
       df, mode='auto', target_class_balance=0.5
   )
   # Result: period=7, threshold=3.8%, instances=512 (50.1% of max)

   # Manual mode: You control the period
   manual_targets, manual_meta = generate_target_combinations(
       df, mode='manual', lookback_periods=7
   )
   # Result: period=7, threshold=5.2%, instances=384 (37.5% of max)

   # Auto mode found lower threshold to hit target balance

Complete Workflow Example
--------------------------

End-to-End ML Pipeline
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import numpy as np
   from rhoa.targets import generate_target_combinations
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import classification_report
   import json

   # 1. Load and prepare data
   df = pd.read_csv('stock_data.csv')
   df['Date'] = pd.to_datetime(df['Date'])
   df = df.sort_values('Date').reset_index(drop=True)

   # 2. Time-based split (IMPORTANT: split before target generation)
   split_idx = int(len(df) * 0.8)
   train_df = df[:split_idx].copy()
   test_df = df[split_idx:].copy()

   # 3. Generate targets on TRAINING data only
   targets_train, meta = generate_target_combinations(
       train_df,
       mode='auto',
       target_class_balance=0.4  # 40% positive
   )

   print(f"Using Method 7: {meta['method_7']}")
   # {'period': 6, 'threshold': 4.2, 'instances': 201, 'pct_of_max': 39.8}

   # 4. Save metadata for reproducibility
   with open('target_config.json', 'w') as f:
       json.dump(meta, f, indent=2)

   # 5. Create features
   train_df['SMA_20'] = train_df['Close'].rolling(20).mean()
   train_df['SMA_50'] = train_df['Close'].rolling(50).mean()
   train_df['Returns'] = train_df['Close'].pct_change()
   train_df['Volatility'] = train_df['Returns'].rolling(20).std()

   # 6. Combine features and target
   train_df['Target'] = targets_train['Target_7']
   train_clean = train_df.dropna()

   # 7. Train model
   feature_cols = ['SMA_20', 'SMA_50', 'Returns', 'Volatility']
   X_train = train_clean[feature_cols]
   y_train = train_clean['Target']

   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)

   # 8. Apply SAME parameters to test data
   test_period = meta['method_7']['period']
   test_threshold = meta['method_7']['threshold'] / 100

   future_close = test_df['Close'].shift(-test_period)
   future_high = test_df['High'].shift(-test_period)
   future_max_high = test_df['High'].shift(-test_period).rolling(
       window=test_period, min_periods=1
   ).max().shift(test_period)

   test_df['Target'] = (future_max_high / test_df['Close'] - 1 >= test_threshold)

   # 9. Create test features
   test_df['SMA_20'] = test_df['Close'].rolling(20).mean()
   test_df['SMA_50'] = test_df['Close'].rolling(50).mean()
   test_df['Returns'] = test_df['Close'].pct_change()
   test_df['Volatility'] = test_df['Returns'].rolling(20).std()

   test_clean = test_df.dropna()

   # 10. Evaluate
   X_test = test_clean[feature_cols]
   y_test = test_clean['Target']
   y_pred = model.predict(X_test)

   print(classification_report(y_test, y_pred))
   #               precision    recall  f1-score
   #          0       0.88      0.91      0.90
   #          1       0.76      0.69      0.72

   # 11. Visualize results
   test_clean.plots.signal(
       y_pred=y_pred,
       y_true=y_test,
       date_col='Date',
       price_col='Close',
       threshold=meta['method_7']['threshold'],
       title=f"Method 7 Predictions (Period={test_period}, Threshold={test_threshold*100:.1f}%)"
   )

Best Practices
--------------

Data Preparation
~~~~~~~~~~~~~~~~

**Always Split Before Target Generation**:

.. code-block:: python

   # CORRECT
   train, test = split(df)
   targets_train, meta = generate_target_combinations(train)
   # Apply meta parameters to test

   # WRONG - Look-ahead bias!
   targets, meta = generate_target_combinations(df)
   train, test = split(df)

**Handle NaN Values**:

.. code-block:: python

   # Targets create NaN for last N rows (future unknown)
   print(targets.tail())
   # Last 'period' rows will be NaN

   # Always drop NaN before training
   combined = pd.concat([features, targets], axis=1)
   clean_data = combined.dropna()

**Ensure Data Quality**:

.. code-block:: python

   # Check for missing values
   assert df[['Open', 'High', 'Low', 'Close']].isnull().sum().sum() == 0

   # Check OHLC relationships
   assert (df['High'] >= df['Close']).all()
   assert (df['Close'] >= df['Low']).all()
   assert (df['High'] >= df['Low']).all()

Parameter Selection
~~~~~~~~~~~~~~~~~~~

**Start Conservative**:

.. code-block:: python

   # Start with lower balance for higher quality
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       target_class_balance=0.3  # Only best 30%
   )

**Consider Your Trading Style**:

.. code-block:: python

   # Day trading: short periods
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       min_period=1,
       max_period=5
   )

   # Swing trading: medium periods
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       min_period=5,
       max_period=15
   )

   # Position trading: long periods
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       min_period=15,
       max_period=30
   )

**Match Threshold to Costs**:

.. code-block:: python

   # If trading costs are 0.5% round-trip
   # Minimum profitable threshold should be > 0.5%
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       min_pct=1,  # Start at 1% minimum
       max_pct=20
   )

Method Selection
~~~~~~~~~~~~~~~~

**For Training (generous)**:

.. code-block:: python

   # Use Method 7 or 8 for more positive examples
   y_train = targets_train['Target_7']  # MaxHigh/Close
   # More data for model to learn from

**For Validation (conservative)**:

.. code-block:: python

   # Use Method 1 for realistic validation
   y_val = targets_val['Target_1']  # Close/Close
   # More realistic profit expectations

**Compare Multiple Methods**:

.. code-block:: python

   # Train on each method, compare results
   results = {}
   for i in range(1, 9):
       X_train, y_train = features, targets[f'Target_{i}']
       model.fit(X_train, y_train)
       results[f'Method_{i}'] = evaluate(model, X_test, y_test)

   # Choose method with best out-of-sample performance

Reproducibility
~~~~~~~~~~~~~~~

**Always Save Metadata**:

.. code-block:: python

   # Save with timestamp
   import datetime

   meta_with_timestamp = {
       'generated_at': datetime.datetime.now().isoformat(),
       'data_shape': df.shape,
       'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
       **meta
   }

   with open('target_metadata.json', 'w') as f:
       json.dump(meta_with_timestamp, f, indent=2)

**Version Your Data and Targets**:

.. code-block:: python

   # Save targets alongside data
   df_with_targets = pd.concat([df, targets], axis=1)
   df_with_targets.to_csv(f'data_with_targets_{datetime.date.today()}.csv')

Common Pitfalls
---------------

Pitfall 1: Look-Ahead Bias
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Generating targets on full dataset before splitting.

.. code-block:: python

   # WRONG
   targets, meta = generate_target_combinations(df)  # Uses all data
   train, test = split(df)  # Then split

   # Why wrong: Optimization saw test data

**Solution**: Generate on train only.

.. code-block:: python

   # CORRECT
   train, test = split(df)
   targets_train, meta = generate_target_combinations(train)
   # Apply meta params to test

Pitfall 2: Ignoring Class Imbalance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Using default 50% balance with limited data.

.. code-block:: python

   # With 100 samples and 50% balance
   targets, meta = generate_target_combinations(
       small_df,  # Only 100 rows
       target_class_balance=0.5
   )
   # Result: Only 50 positive examples (too few!)

**Solution**: Adjust balance or get more data.

.. code-block:: python

   # Better
   targets, meta = generate_target_combinations(
       small_df,
       target_class_balance=0.7  # 70 positive examples
   )

Pitfall 3: Unrealistic Thresholds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Thresholds below transaction costs.

.. code-block:: python

   # Transaction cost = 0.5% round-trip
   # But threshold = 0.3%
   # Every "profitable" trade actually loses money!

**Solution**: Set minimum threshold above costs.

.. code-block:: python

   targets, meta = generate_target_combinations(
       df,
       min_pct=1,  # 1% minimum (above 0.5% costs)
   )

Pitfall 4: Over-Optimizing on Single Stock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Perfect parameters for AAPL may not work for others.

**Solution**: Test on multiple assets.

.. code-block:: python

   # Find parameters on basket of stocks
   all_targets = []
   for ticker in ['AAPL', 'MSFT', 'GOOGL']:
       df = load_data(ticker)
       targets, meta = generate_target_combinations(df)
       all_targets.append(targets)

   # Use common parameters across stocks

Advanced Topics
---------------

Multi-Target Ensemble
~~~~~~~~~~~~~~~~~~~~~

Train separate models on different target methods and ensemble:

.. code-block:: python

   # Train models on different targets
   models = {}
   for i in [1, 5, 7]:  # Conservative, medium, aggressive
       X, y = features, targets[f'Target_{i}']
       model = RandomForestClassifier()
       model.fit(X, y)
       models[i] = model

   # Ensemble: require 2 out of 3 to agree
   pred_1 = models[1].predict_proba(X_test)[:, 1]
   pred_5 = models[5].predict_proba(X_test)[:, 1]
   pred_7 = models[7].predict_proba(X_test)[:, 1]

   ensemble_pred = ((pred_1 > 0.5) + (pred_5 > 0.5) + (pred_7 > 0.5)) >= 2

Custom Target Generation
~~~~~~~~~~~~~~~~~~~~~~~~

Create your own target based on Rhoa's patterns:

.. code-block:: python

   def custom_target(df, period, threshold, cost_pct=0.5):
       """Target that accounts for transaction costs."""
       future_high = df['High'].shift(-period)
       future_low = df['Low'].shift(-period)

       # Best case profit
       profit = (future_high / df['Close'] - 1) - (cost_pct / 100)

       # Worst case loss (stop loss at 2%)
       loss = (future_low / df['Close'] - 1) - (cost_pct / 100)

       # Positive if profit > threshold AND loss > -2%
       return (profit >= threshold / 100) & (loss >= -0.02)

   # Use custom target
   target = custom_target(df, period=5, threshold=3.0)

Performance Tips
----------------

For Large Datasets
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Reduce search space
   targets, meta = generate_target_combinations(
       large_df,
       mode='auto',
       period_step=2,  # Check every 2 periods instead of 1
       step=2          # Check every 2% threshold instead of 1%
   )
   # Runs 4x faster with minimal accuracy loss

For Production
~~~~~~~~~~~~~~

.. code-block:: python

   # Cache targets if data doesn't change
   import joblib

   cache_file = 'targets_cache.pkl'

   if os.path.exists(cache_file):
       targets, meta = joblib.load(cache_file)
   else:
       targets, meta = generate_target_combinations(df)
       joblib.dump((targets, meta), cache_file)

Further Reading
---------------

- :doc:`indicators_guide` - Using indicators as features
- :doc:`visualization_guide` - Evaluating target quality
- :doc:`/examples/target_generation` - Hands-on examples
- :doc:`/examples/complete_pipeline` - Full ML pipeline
- :doc:`/api/targets` - API reference

Summary
-------

Key takeaways:

1. **Auto mode** for optimal parameters, **manual mode** for fixed periods
2. Always split data **before** generating targets
3. Save metadata for reproducibility
4. Choose target method based on use case (Method 7 for training, Method 1 for validation)
5. Account for transaction costs in threshold selection
6. Test parameters across multiple assets
7. Validate on true out-of-sample data

The target generation system is designed to remove guesswork and provide data-driven, reproducible trading signal definitions for machine learning.

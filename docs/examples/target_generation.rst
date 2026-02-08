Target Generation
=================

Learn how to generate optimized binary classification targets for machine learning models.

Overview
--------

Rhoa's target generation creates 8 different binary target types based on future price movements:

1. **Target_1**: Close[N]/Close[0] - Future close vs current close
2. **Target_2**: Close[N]/High[0] - Future close vs current high
3. **Target_3**: High[N]/Close[0] - Future high vs current close
4. **Target_4**: High[N]/High[0] - Future high vs current high
5. **Target_5**: MaxClose/Close[0] - Max future close vs current close
6. **Target_6**: MaxClose/High[0] - Max future close vs current high
7. **Target_7**: MaxHigh/Close[0] - Max future high vs current close
8. **Target_8**: MaxHigh/High[0] - Max future high vs current high

Two modes are available:
- **Auto mode**: Pareto optimization finds optimal lookback period AND threshold
- **Manual mode**: Fixed lookback period with elbow method for thresholds

.. _target-auto:

Auto Mode (Pareto Optimization)
--------------------------------

Auto mode automatically finds the best combination of lookback period and threshold.

Basic Auto Mode
~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from rhoa.targets import generate_target_combinations

   # Load OHLC data
   df = pd.read_csv('prices.csv')

   # Generate targets with auto mode
   targets, metadata = generate_target_combinations(
       df,
       mode='auto',
       target_class_balance=0.5  # Aim for 50% positive instances
   )

   # Inspect the targets
   print(targets.head())
   print(f"\nShape: {targets.shape}")
   print(f"\nColumns: {targets.columns.tolist()}")

Understanding Metadata
~~~~~~~~~~~~~~~~~~~~~~

The metadata dictionary contains parameters for each target:

.. code-block:: python

   # Check what parameters were found optimal
   for method, params in metadata.items():
       print(f"{method}:")
       print(f"  Period: {params['period']} days")
       print(f"  Threshold: {params['threshold']}%")
       print(f"  Positive instances: {params['instances']}")
       print(f"  % of maximum: {params['pct_of_max']:.1f}%")
       print()

Example output:

.. code-block:: text

   method_1:
     Period: 5 days
     Threshold: 3.5%
     Positive instances: 142
     % of maximum: 12.3%

   method_7:
     Period: 6 days
     Threshold: 4.0%
     Positive instances: 249
     % of maximum: 21.5%

Custom Class Balance
~~~~~~~~~~~~~~~~~~~~

Adjust the target class balance based on your needs:

.. code-block:: python

   # Conservative: 30% positive instances (higher precision)
   targets_conservative, meta = generate_target_combinations(
       df,
       mode='auto',
       target_class_balance=0.3
   )

   # Aggressive: 70% positive instances (higher recall)
   targets_aggressive, meta = generate_target_combinations(
       df,
       mode='auto',
       target_class_balance=0.7
   )

   # Balanced: 50% positive instances
   targets_balanced, meta = generate_target_combinations(
       df,
       mode='auto',
       target_class_balance=0.5
   )

Custom Search Ranges
~~~~~~~~~~~~~~~~~~~~

Control the optimization search space:

.. code-block:: python

   # Search only short-term periods (1-5 days)
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       period_range=(1, 5),
       threshold_range=(1.0, 5.0),
       target_class_balance=0.5
   )

   # Search longer-term periods (10-30 days)
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       period_range=(10, 30),
       threshold_range=(5.0, 20.0),
       target_class_balance=0.4
   )

.. _target-manual:

Manual Mode (Elbow Method)
---------------------------

Manual mode uses a fixed lookback period and finds optimal thresholds using the elbow method.

Basic Manual Mode
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate targets with fixed 5-day lookback
   targets, metadata = generate_target_combinations(
       df,
       mode='manual',
       lookback_periods=5
   )

   # Check the detected thresholds
   for method, params in metadata.items():
       print(f"{method}: threshold={params['threshold']}%, "
             f"instances={params['instances']}")

Different Lookback Periods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Short-term: 3 days
   targets_short, meta_short = generate_target_combinations(
       df,
       mode='manual',
       lookback_periods=3
   )

   # Medium-term: 10 days
   targets_medium, meta_medium = generate_target_combinations(
       df,
       mode='manual',
       lookback_periods=10
   )

   # Long-term: 20 days
   targets_long, meta_long = generate_target_combinations(
       df,
       mode='manual',
       lookback_periods=20
   )

   # Compare signal counts
   print(f"Short-term (3d): {targets_short['Target_7'].sum()} signals")
   print(f"Medium-term (10d): {targets_medium['Target_7'].sum()} signals")
   print(f"Long-term (20d): {targets_long['Target_7'].sum()} signals")

Choosing the Right Target
--------------------------

Different targets serve different trading strategies.

Target Characteristics
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   targets, meta = generate_target_combinations(df, mode='auto')

   # Analyze each target
   for i in range(1, 9):
       target_col = f'Target_{i}'
       positive_pct = targets[target_col].mean() * 100

       print(f"{target_col}:")
       print(f"  Method: {meta[f'method_{i}']}")
       print(f"  Positive: {positive_pct:.1f}%")
       print(f"  Negative: {100-positive_pct:.1f}%")
       print()

Compare Target Types
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Target 1: Conservative (Close[N]/Close[0])
   # Requires price to be higher at specific future date

   # Target 7: Moderate (MaxHigh/Close[0])
   # Requires max high within period to exceed threshold
   # More achievable than Target 1

   # Target 8: Aggressive (MaxHigh/High[0])
   # Most stringent - requires exceeding current high

   # Count positives for each
   for i in [1, 7, 8]:
       count = targets[f'Target_{i}'].sum()
       pct = targets[f'Target_{i}'].mean() * 100
       print(f"Target {i}: {count} positives ({pct:.1f}%)")

Recommended Targets by Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   """
   Swing Trading (hold 3-10 days):
     - Target_7 (MaxHigh/Close[0])
     - Captures upward swings
     - Moderate signal frequency

   Day Trading (hold intraday):
     - Target_1 (Close[N]/Close[0])
     - Quick entries and exits
     - Higher frequency

   Position Trading (hold weeks/months):
     - Target_5 (MaxClose/Close[0])
     - Sustained movements
     - Lower frequency, higher conviction

   Mean Reversion:
     - Target_1 with lower thresholds (1-3%)
     - Quick profit taking

   Momentum/Breakout:
     - Target_7/8 with higher thresholds (>5%)
     - Captures strong moves
   """

Validating Targets
------------------

Always validate target quality before using in production.

Class Balance Check
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   targets, meta = generate_target_combinations(df, mode='auto')

   # Check class balance for all targets
   print("Class Distribution:")
   print("-" * 40)
   for col in targets.columns:
       pos = targets[col].sum()
       neg = len(targets) - pos
       ratio = pos / len(targets) * 100
       print(f"{col}: {pos} positive ({ratio:.1f}%), {neg} negative")

Temporal Validation
~~~~~~~~~~~~~~~~~~~

Check if targets are distributed across time:

.. code-block:: python

   import matplotlib.pyplot as plt

   targets, meta = generate_target_combinations(df, mode='auto')

   # Check Target_7 distribution over time
   # Assuming df has Date column
   df['Target_7'] = targets['Target_7']

   # Group by month
   df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')
   monthly = df.groupby('Month')['Target_7'].mean()

   print("Monthly positive rate:")
   print(monthly)

   # Should not have months with 0% or 100% positive rate

Forward-Looking Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Never validate on data used to generate targets:

.. code-block:: python

   # WRONG: Using same data for generation and validation
   targets, meta = generate_target_combinations(df, mode='auto')
   X_train, X_test, y_train, y_test = train_test_split(X, targets['Target_7'])
   # This will overfit!

   # CORRECT: Time-based split
   split_date = '2024-01-01'

   # Generate targets on training data only
   train_df = df[df['Date'] < split_date]
   targets_train, meta = generate_target_combinations(train_df, mode='auto')

   # Apply same parameters to test data
   test_df = df[df['Date'] >= split_date]
   # Use meta['method_7']['period'] and meta['method_7']['threshold']
   # to generate test targets with same parameters

Practical Examples
------------------

Example 1: Multi-Timeframe Targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate targets for different timeframes
   df = pd.read_csv('prices.csv')

   # Short-term (3 days)
   targets_3d, meta_3d = generate_target_combinations(
       df, mode='manual', lookback_periods=3
   )
   df['Target_3d'] = targets_3d['Target_7']

   # Medium-term (10 days)
   targets_10d, meta_10d = generate_target_combinations(
       df, mode='manual', lookback_periods=10
   )
   df['Target_10d'] = targets_10d['Target_7']

   # Use both targets for hierarchical prediction
   # First predict 3d, then 10d

Example 2: Target Ensemble
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use multiple targets for ensemble approach
   targets, meta = generate_target_combinations(df, mode='auto')

   # Create ensemble target: positive if ANY of multiple targets true
   ensemble = (
       targets['Target_5'] |
       targets['Target_6'] |
       targets['Target_7']
   ).astype(int)

   print(f"Ensemble positive rate: {ensemble.mean():.1%}")

Example 3: Conservative Target
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create very conservative target for high-precision trading
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       target_class_balance=0.1,  # Only 10% positive
       threshold_range=(10.0, 30.0)  # High thresholds
   )

   # This should give fewer but higher-quality signals
   print(f"Positive signals: {targets['Target_7'].sum()}")
   print(f"Average threshold: {meta['method_7']['threshold']}%")

Saving and Loading Metadata
----------------------------

Always save target generation parameters for reproducibility.

Save Metadata
~~~~~~~~~~~~~

.. code-block:: python

   import json

   targets, meta = generate_target_combinations(df, mode='auto')

   # Save metadata
   with open('target_metadata.json', 'w') as f:
       json.dump(meta, f, indent=2)

   # Save targets
   targets.to_csv('targets.csv', index=False)

Load and Apply
~~~~~~~~~~~~~~

.. code-block:: python

   import json

   # Load metadata
   with open('target_metadata.json', 'r') as f:
       meta = json.load(f)

   # Apply to new data using same parameters
   # (You'll need to implement the logic using meta parameters)
   period = meta['method_7']['period']
   threshold = meta['method_7']['threshold']

   print(f"Using period={period}, threshold={threshold}%")

Best Practices
--------------

1. **Always validate targets** on out-of-sample data
2. **Save metadata** for reproducibility
3. **Check class balance** - extreme imbalance causes issues
4. **Use appropriate timeframes** - match your trading style
5. **Start with auto mode** - it finds good defaults
6. **Test multiple targets** - different targets for different strategies
7. **Consider transaction costs** - adjust thresholds accordingly
8. **Avoid data leakage** - never peek at future data during training

Next Steps
----------

Continue to :doc:`complete_pipeline` to see how to build end-to-end ML pipelines
using generated targets.

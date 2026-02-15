Visualization Guide
===================

Learn how to create professional visualizations for evaluating machine learning predictions using Rhoa's plots accessor.

Overview
--------

Rhoa provides a powerful visualization system through the ``.plots`` accessor on pandas DataFrames. Currently focused on the ``signal()`` method, which creates comprehensive visualizations showing:

- Stock price charts with predicted buy signals
- Confusion matrices with detailed metrics
- False positive and false negative identification
- Professional styling suitable for presentations and reports

Quick Start
-----------

Basic Signal Plot
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import rhoa

   # Load your data with predictions
   df = pd.read_csv('stock_data.csv')
   df['Date'] = pd.to_datetime(df['Date'])

   # Assume you have predictions and ground truth
   predictions = model.predict(X_test)
   ground_truth = y_test

   # Create visualization
   fig = df.rhoa.plots.signal(
       y_pred=predictions,
       y_true=ground_truth,
       date_col='Date',
       price_col='Close'
   )

This creates a two-panel visualization:

- **Top panel**: Confusion matrix with precision/recall metrics
- **Bottom panel**: Price chart with signals overlaid

The signal() Method
-------------------

Complete API
~~~~~~~~~~~~

.. code-block:: python

   df.rhoa.plots.signal(
       y_pred,                    # Required: predictions
       y_true=None,              # Optional: ground truth
       date_col='Date',          # Date column name
       price_col='Close',        # Price column to plot
       threshold=None,           # Prediction threshold used
       title=None,               # Custom title
       figsize=(18, 10),         # Figure size
       cmap='Blues',             # Confusion matrix colormap
       save_path=None,           # Path to save figure
       dpi=300,                  # Resolution for saved figure
       show=True                 # Whether to display
   )

Parameters Explained
~~~~~~~~~~~~~~~~~~~~

**y_pred** (required)
   Binary predictions array (0 or 1) from your model.

   .. code-block:: python

      # From scikit-learn model
      y_pred = model.predict(X_test)

      # From probability predictions
      y_pred_proba = model.predict_proba(X_test)[:, 1]
      y_pred = (y_pred_proba > 0.6).astype(int)  # Custom threshold

**y_true** (optional)
   Ground truth labels. When provided:

   - Adds confusion matrix panel
   - Shows precision and recall metrics
   - Highlights false positives (red X)
   - Highlights false negatives (orange circles)

   .. code-block:: python

      # With ground truth (full visualization)
      fig = df.rhoa.plots.signal(y_pred=predictions, y_true=targets)

      # Without ground truth (predictions only)
      fig = df.rhoa.plots.signal(y_pred=predictions)

**date_col** (default='Date')
   Name of the date/timestamp column.

   .. code-block:: python

      # If your date column is named differently
      fig = df.rhoa.plots.signal(
          y_pred=predictions,
          y_true=targets,
          date_col='Timestamp'  # Custom column name
      )

**price_col** (default='Close')
   Which price to plot (usually 'Close', but can be 'Open', 'High', 'Low').

   .. code-block:: python

      # Plot with High prices instead of Close
      fig = df.rhoa.plots.signal(
          y_pred=predictions,
          y_true=targets,
          price_col='High'
      )

**threshold** (optional)
   The prediction threshold that was used. Displayed in title for reference.

   .. code-block:: python

      threshold = 0.67
      y_pred = (model.predict_proba(X)[:, 1] > threshold).astype(int)

      fig = df.rhoa.plots.signal(
          y_pred=y_pred,
          y_true=y_true,
          threshold=threshold  # Shows "Threshold: 0.67" in title
      )

**title** (optional)
   Custom title for the plot.

   .. code-block:: python

      fig = df.rhoa.plots.signal(
          y_pred=predictions,
          y_true=targets,
          title='AAPL Random Forest Model'
      )

**figsize** (default=(18, 10))
   Figure size as (width, height) in inches.

   .. code-block:: python

      # Larger figure for presentations
      fig = df.rhoa.plots.signal(
          y_pred=predictions,
          y_true=targets,
          figsize=(24, 14)
      )

      # Smaller figure for reports
      fig = df.rhoa.plots.signal(
          y_pred=predictions,
          y_true=targets,
          figsize=(12, 8)
      )

**cmap** (default='Blues')
   Colormap for confusion matrix. Options: 'Blues', 'Greens', 'Reds', 'Purples', etc.

   .. code-block:: python

      # Green theme for positive emphasis
      fig = df.rhoa.plots.signal(
          y_pred=predictions,
          y_true=targets,
          cmap='Greens'
      )

**save_path** (optional)
   Path to save the figure. If None, figure is not saved.

   .. code-block:: python

      fig = df.rhoa.plots.signal(
          y_pred=predictions,
          y_true=targets,
          save_path='results/aapl_predictions.png'
      )

**dpi** (default=300)
   Resolution for saved figure (dots per inch). 300 is publication quality.

   .. code-block:: python

      # High resolution for printing
      fig = df.rhoa.plots.signal(
          y_pred=predictions,
          y_true=targets,
          save_path='report.png',
          dpi=600  # Very high quality
      )

**show** (default=True)
   Whether to display the plot. Set to False when saving only.

   .. code-block:: python

      # Save without displaying
      fig = df.rhoa.plots.signal(
          y_pred=predictions,
          y_true=targets,
          save_path='output.png',
          show=False
      )

Understanding the Visualization
--------------------------------

Confusion Matrix Panel
~~~~~~~~~~~~~~~~~~~~~~

The confusion matrix shows how well your model performed:

.. code-block:: text

   ┌─────────────────────────────────────┐
   │        Confusion Matrix             │
   │  Threshold: 0.67 | Precision: 85.3% │
   │                                      │
   │              Predicted               │
   │           No Buy(0)  Buy(1)          │
   │  True  ┌─────────────────────┐      │
   │  No(0) │  TN: 420  │ FP: 23  │      │
   │        │  (95%)    │  (5%)   │      │
   │        ├───────────┼─────────┤      │
   │  Buy(1)│  FN: 18   │ TP: 105 │      │
   │        │  (15%)    │ (85%)   │      │
   │        └───────────┴─────────┘      │
   │                                      │
   │  Metrics Summary:                    │
   │  TP: 105  FP: 23                    │
   │  TN: 420  FN: 18                    │
   │  Total Signals: 128                  │
   │  Correct: 105 (85.3%)               │
   └─────────────────────────────────────┘

**Reading the Matrix**:

- **True Negatives (TN)**: Correctly predicted no-buy (420 cases, 95%)
- **False Positives (FP)**: Predicted buy but shouldn't have (23 cases, 5%)
- **False Negatives (FN)**: Missed opportunities (18 cases, 15%)
- **True Positives (TP)**: Correctly predicted buy signals (105 cases, 85%)

**Key Metrics**:

- **Precision = TP / (TP + FP) = 105 / 128 = 85.3%**

  *"Of all buy signals, 85.3% were correct"*

  High precision means fewer false alarms, lower transaction costs.

- **Recall = TP / (TP + FN) = 105 / 123 = 85.4%**

  *"Of all true opportunities, we caught 85.4%"*

  High recall means fewer missed opportunities, more profit potential.

Price Chart Panel
~~~~~~~~~~~~~~~~~

The price chart shows predictions overlaid on price:

.. code-block:: text

   ┌────────────────────────────────────────┐
   │     Close Price with Buy Signals       │
   │                                        │
   │  $150 ─────────────────────────────   │
   │                   ●                    │
   │       ○                 ○              │ ○ = True opportunities (light green)
   │  $140 ────●────────────────────────   │ ● = Model predictions (bright green)
   │         ✗                              │ ✗ = False positives (red)
   │  $130 ─────────────────────────────   │ ◯ = Missed opportunities (orange)
   │       ○                                │
   │  $120 ─────────────────────────────   │
   │                                        │
   │    Jan    Feb    Mar    Apr    May    │
   └────────────────────────────────────────┘

**Visual Elements**:

1. **Blue Line**: Stock price over time
2. **Light Green Background Dots**: All true buy opportunities (when y_true provided)
3. **Bright Green Dots**: Model's buy signal predictions
4. **Red X Markers**: False positives (predicted buy, but wasn't a real opportunity)
5. **Orange Circles**: False negatives (missed opportunities)

**What to Look For**:

- **Clustered green dots**: Model finding real patterns
- **Red X markers**: Where model made mistakes (investigate these)
- **Orange circles**: Opportunities the model missed (could improve recall)
- **Position on price chart**: Are signals coming at good entry points?

Interpreting Results
--------------------

Good Model Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~

**High Precision (> 70%)**

.. code-block:: python

   # Precision: 85.3%
   # Of 128 signals, 105 were correct

- Few false positives (red X marks)
- Most green dots align with light green background
- Signals are reliable
- Lower transaction costs

**High Recall (> 60%)**

.. code-block:: python

   # Recall: 85.4%
   # Caught 105 of 123 opportunities

- Few orange circles (missed opportunities)
- Capturing most profitable trades
- Higher profit potential

**Signals at Good Entry Points**

Look at the price chart:

- Are signals coming near local lows? (Good!)
- Are signals coming near local highs? (Bad - buying at top)
- Do signals cluster before price increases? (Good!)

Poor Model Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Low Precision (< 50%)**

.. code-block:: python

   # Precision: 45.2%
   # Of 200 signals, only 90 correct

- Many red X marks (false positives)
- Model is too aggressive
- High transaction costs will eat profits
- **Solution**: Increase prediction threshold or retrain

**Low Recall (< 40%)**

.. code-block:: python

   # Recall: 38.5%
   # Only caught 70 of 182 opportunities

- Many orange circles (missed opportunities)
- Model is too conservative
- Missing too many profitable trades
- **Solution**: Decrease prediction threshold or add features

**Random-Looking Signals**

If signals appear random on the price chart:

- No clear pattern relative to price movements
- Equal distribution of correct/incorrect
- Model hasn't learned meaningful patterns
- **Solution**: Feature engineering, more data, or different algorithm

Example Interpretations
-----------------------

Scenario 1: High Precision, Low Recall
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Confusion Matrix:
   Precision: 92.3% | Recall: 42.1%

   TP: 48   FP: 4
   TN: 480  FN: 66

**Interpretation**:

- Model is very conservative
- When it predicts buy, it's usually right (92.3%)
- But it misses many opportunities (66 missed)
- Only trading 52 times instead of possible 114

**Appropriate For**:

- High transaction costs
- Risk-averse strategies
- Need high win rate

**How to Improve**:

.. code-block:: python

   # Lower prediction threshold
   y_pred_conservative = (y_pred_proba > 0.8).astype(int)  # Current
   y_pred_balanced = (y_pred_proba > 0.6).astype(int)      # Better balance

Scenario 2: Low Precision, High Recall
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Confusion Matrix:
   Precision: 58.7% | Recall: 88.2%

   TP: 105  FP: 74
   TN: 360  FN: 14

**Interpretation**:

- Model is very aggressive
- Catches almost all opportunities (88.2%)
- But generates many false signals (74)
- Trading 179 times (many unnecessary)

**Appropriate For**:

- Low transaction costs
- Market-making strategies
- Exploration phase

**How to Improve**:

.. code-block:: python

   # Raise prediction threshold
   y_pred_aggressive = (y_pred_proba > 0.4).astype(int)  # Current
   y_pred_balanced = (y_pred_proba > 0.6).astype(int)    # Better balance

Scenario 3: Balanced Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Confusion Matrix:
   Precision: 76.5% | Recall: 72.3%

   TP: 89   FP: 27
   TN: 420  FN: 34

**Interpretation**:

- Good balance between precision and recall
- 116 total signals, 89 correct (76.5%)
- Caught 89 of 123 opportunities (72.3%)
- F1 score ≈ 74.3% (harmonic mean)

**This Is Ideal**: Good balance suitable for most trading strategies.

Scenario 4: Poor Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Confusion Matrix:
   Precision: 52.1% | Recall: 49.5%

   TP: 45   FP: 41
   TN: 402  FN: 46

**Interpretation**:

- Barely better than random (50%)
- As many false as true positives
- Missing half the opportunities
- Model hasn't learned useful patterns

**Actions**:

1. Check for data leakage
2. Improve feature engineering
3. Try different algorithms
4. Get more/better quality data

Practical Examples
------------------

Complete Workflow
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import numpy as np
   import rhoa
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split

   # 1. Load data
   df = pd.read_csv('AAPL.csv')
   df['Date'] = pd.to_datetime(df['Date'])

   # 2. Create features
   df['SMA_20'] = df['Close'].rolling(20).mean()
   df['SMA_50'] = df['Close'].rolling(50).mean()
   df['RSI'] = df.rhoa.indicators.rsi(14)
   df['Returns'] = df['Close'].pct_change()

   # 3. Create target
   from rhoa.targets import generate_target_combinations
   targets, meta = generate_target_combinations(df, mode='auto')
   df['Target'] = targets['Target_7']

   # 4. Prepare data
   df_clean = df.dropna()
   features = ['SMA_20', 'SMA_50', 'RSI', 'Returns']
   X = df_clean[features]
   y = df_clean['Target']

   # 5. Split
   split_idx = int(len(X) * 0.8)
   X_train, X_test = X[:split_idx], X[split_idx:]
   y_train, y_test = y[:split_idx], y[split_idx:]
   df_test = df_clean[split_idx:]

   # 6. Train
   model = RandomForestClassifier(n_estimators=100)
   model.fit(X_train, y_train)

   # 7. Predict
   y_pred = model.predict(X_test)

   # 8. Visualize
   fig = df_test.rhoa.plots.signal(
       y_pred=y_pred,
       y_true=y_test,
       date_col='Date',
       price_col='Close',
       title='AAPL Random Forest Predictions',
       save_path='aapl_results.png'
   )

Comparing Models
~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
   from sklearn.linear_model import LogisticRegression

   # Train multiple models
   models = {
       'Random Forest': RandomForestClassifier(n_estimators=100),
       'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
       'Logistic Regression': LogisticRegression()
   }

   # Visualize each
   for name, model in models.items():
       model.fit(X_train, y_train)
       y_pred = model.predict(X_test)

       fig = df_test.rhoa.plots.signal(
           y_pred=y_pred,
           y_true=y_test,
           title=f'{name} - AAPL Predictions',
           save_path=f'results/{name.lower().replace(" ", "_")}.png',
           show=False  # Don't show, just save
       )

   # Now compare the saved images side by side

Threshold Optimization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get probabilities
   y_pred_proba = model.predict_proba(X_test)[:, 1]

   # Try different thresholds
   thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]

   for threshold in thresholds:
       y_pred = (y_pred_proba > threshold).astype(int)

       fig = df_test.rhoa.plots.signal(
           y_pred=y_pred,
           y_true=y_test,
           threshold=threshold,
           title=f'AAPL - Threshold {threshold}',
           save_path=f'threshold_analysis/threshold_{threshold}.png',
           show=False
       )

   # Review images to find optimal threshold

Predictions Only (No Ground Truth)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # When you don't have ground truth (production/future predictions)
   future_data = pd.read_csv('new_data.csv')
   future_data['Date'] = pd.to_datetime(future_data['Date'])

   # Create same features
   future_data['SMA_20'] = future_data['Close'].rolling(20).mean()
   future_data['SMA_50'] = future_data['Close'].rolling(50).mean()
   future_data['RSI'] = future_data['Close'].rhoa.indicators.rsi(14)
   future_data['Returns'] = future_data['Close'].pct_change()

   future_clean = future_data.dropna()
   X_future = future_clean[features]

   # Predict
   y_pred_future = model.predict(X_future)

   # Visualize (no ground truth, no confusion matrix)
   fig = future_clean.rhoa.plots.signal(
       y_pred=y_pred_future,
       # No y_true parameter
       date_col='Date',
       price_col='Close',
       title='AAPL Future Predictions',
       save_path='future_signals.png'
   )

Customization Options
---------------------

Color Schemes
~~~~~~~~~~~~~

.. code-block:: python

   # Professional blue theme (default)
   fig = df.rhoa.plots.signal(y_pred=pred, y_true=true, cmap='Blues')

   # Success/green theme
   fig = df.rhoa.plots.signal(y_pred=pred, y_true=true, cmap='Greens')

   # Warning/red theme
   fig = df.rhoa.plots.signal(y_pred=pred, y_true=true, cmap='Reds')

   # Purple theme
   fig = df.rhoa.plots.signal(y_pred=pred, y_true=true, cmap='Purples')

Figure Sizes for Different Uses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For presentations (large, high DPI)
   fig = df.rhoa.plots.signal(
       y_pred=pred, y_true=true,
       figsize=(24, 14),
       dpi=150,
       save_path='presentation.png'
   )

   # For papers/publications (standard, very high DPI)
   fig = df.rhoa.plots.signal(
       y_pred=pred, y_true=true,
       figsize=(12, 8),
       dpi=600,
       save_path='publication.png'
   )

   # For reports (medium)
   fig = df.rhoa.plots.signal(
       y_pred=pred, y_true=true,
       figsize=(15, 9),
       dpi=300,
       save_path='report.png'
   )

   # For web/dashboard (smaller, lower DPI)
   fig = df.rhoa.plots.signal(
       y_pred=pred, y_true=true,
       figsize=(10, 6),
       dpi=100,
       save_path='dashboard.png'
   )

Further Customization
~~~~~~~~~~~~~~~~~~~~~

The method returns a matplotlib Figure object, allowing further customization:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Get figure
   fig = df.rhoa.plots.signal(y_pred=pred, y_true=true, show=False)

   # Access axes
   axes = fig.get_axes()
   confusion_ax = axes[0]  # First panel
   price_ax = axes[1]      # Second panel

   # Customize
   price_ax.set_ylabel('Price (USD)', fontsize=14)
   price_ax.grid(True, linestyle='--', alpha=0.5)

   # Add annotations
   price_ax.annotate(
       'Important Event',
       xy=('2024-01-15', 150),
       xytext=('2024-02-01', 160),
       arrowprops=dict(arrowstyle='->', color='red')
   )

   # Save customized version
   plt.savefig('customized.png', dpi=300, bbox_inches='tight')
   plt.show()

Best Practices
--------------

Do's
~~~~

1. **Always Validate on Out-of-Sample Data**

   .. code-block:: python

      # Use time-based split
      split_idx = int(len(df) * 0.8)
      train, test = df[:split_idx], df[split_idx:]

2. **Save Visualizations for Documentation**

   .. code-block:: python

      fig = df.rhoa.plots.signal(
          y_pred=pred, y_true=true,
          save_path=f'results/{model_name}_{datetime.date.today()}.png'
      )

3. **Include Threshold in Filename When Comparing**

   .. code-block:: python

      for t in [0.5, 0.6, 0.7]:
          y_pred = (proba > t).astype(int)
          df.rhoa.plots.signal(
              y_pred=y_pred, y_true=y_true,
              save_path=f'threshold_{t:.1f}.png',
              show=False
          )

4. **Use Descriptive Titles**

   .. code-block:: python

      title = f'{ticker} - {model_type} - {date_range} - Threshold {threshold}'
      fig = df.rhoa.plots.signal(y_pred=pred, y_true=true, title=title)

Don'ts
~~~~~~

1. **Don't Visualize Training Data Performance**

   .. code-block:: python

      # WRONG - visualizing training data
      model.fit(X_train, y_train)
      y_train_pred = model.predict(X_train)
      df_train.rhoa.plots.signal(y_pred=y_train_pred, y_true=y_train)
      # This will look artificially good!

      # CORRECT - visualize test data
      y_test_pred = model.predict(X_test)
      df_test.rhoa.plots.signal(y_pred=y_test_pred, y_true=y_test)

2. **Don't Compare Models on Different Date Ranges**

   .. code-block:: python

      # WRONG
      model1_pred = model1.predict(df['2024-01':'2024-06'])
      model2_pred = model2.predict(df['2024-03':'2024-08'])

      # CORRECT - same test period
      test_period = df['2024-06':'2024-12']
      model1_pred = model1.predict(test_period)
      model2_pred = model2.predict(test_period)

3. **Don't Ignore the Price Chart**

   Don't just look at precision/recall numbers. Always check:
   - Are signals at reasonable entry points?
   - Do false positives have a pattern?
   - Are missed opportunities (FN) avoidable?

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue**: "Length of y_pred doesn't match DataFrame length"

.. code-block:: python

   # Problem: NaN values were dropped
   y_pred = model.predict(X_test)  # Length: 100
   df_test                         # Length: 120 (includes NaN rows)

   # Solution: Drop NaN before splitting
   df_clean = df.dropna()
   # Then split and use df_clean for plotting

**Issue**: Confusion matrix shows all zeros

.. code-block:: python

   # Problem: y_true or y_pred are all same class
   print(y_pred.sum())  # Should not be 0 or len(y_pred)

   # Solution: Check model is actually predicting both classes
   print(np.unique(y_pred, return_counts=True))

**Issue**: Figure doesn't display

.. code-block:: python

   # If using Jupyter
   %matplotlib inline

   # If using script
   import matplotlib.pyplot as plt
   fig = df.rhoa.plots.signal(y_pred=pred, y_true=true)
   plt.show()  # Explicitly show

**Issue**: Date axis is crowded/unreadable

.. code-block:: python

   # Solution: Rotate labels automatically applied
   # But you can customize further
   fig = df.rhoa.plots.signal(y_pred=pred, y_true=true, show=False)
   ax = fig.get_axes()[1]  # Price chart
   ax.tick_params(axis='x', rotation=45, labelsize=10)
   plt.tight_layout()
   plt.show()

Performance Tips
----------------

For Large Datasets
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Subsample for visualization if very large
   if len(df_test) > 1000:
       sample_idx = np.linspace(0, len(df_test)-1, 1000, dtype=int)
       df_sample = df_test.iloc[sample_idx]
       y_pred_sample = y_pred[sample_idx]
       y_true_sample = y_true[sample_idx]

       fig = df_sample.rhoa.plots.signal(
           y_pred=y_pred_sample,
           y_true=y_true_sample
       )

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process multiple stocks
   tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

   for ticker in tickers:
       df = load_data(ticker)
       # ... train model, get predictions ...

       fig = df_test.rhoa.plots.signal(
           y_pred=y_pred,
           y_true=y_true,
           title=f'{ticker} Predictions',
           save_path=f'results/{ticker}.png',
           show=False  # Don't show, just save
       )
       plt.close(fig)  # Free memory

Summary
-------

The ``plots.signal()`` method provides:

- **Comprehensive visualization** of model performance
- **Confusion matrix** with precision/recall metrics
- **Price chart** with predictions overlaid
- **False positive/negative identification**
- **Professional styling** for reports and presentations
- **Flexible customization** options

Key points:

1. Use ``y_true`` for full evaluation with confusion matrix
2. Omit ``y_true`` for future predictions visualization
3. Check both metrics AND price chart patterns
4. Save visualizations for documentation
5. Compare multiple models/thresholds visually
6. Customize for your specific presentation needs

Further Reading
---------------

- :doc:`targets_guide` - Understanding what to predict
- :doc:`indicators_guide` - Features for prediction
- :doc:`/examples/complete_pipeline` - End-to-end example
- :doc:`/api/plots` - Complete API reference

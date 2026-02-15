User Guide
==========

Welcome to the Rhoa User Guide. This section provides in-depth conceptual documentation
to help you understand and effectively use Rhoa's features.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   basic_concepts
   indicators_guide
   targets_guide
   visualization_guide

Overview
--------

Rhoa is designed to streamline technical analysis and machine learning workflows for
financial time series data. It extends pandas with specialized accessors that integrate
seamlessly with your existing data science pipelines.

Core Philosophy
---------------

**Pandas-Native Integration**
   Rhoa extends pandas rather than replacing it. Use familiar pandas syntax with
   powerful financial analysis capabilities.

**Sensible Defaults**
   Common technical indicators work out of the box with industry-standard parameters.
   Advanced users can customize every aspect.

**ML-First Design**
   The target generation system is built specifically for machine learning,
   with automatic optimization and proper validation handling.

**Production-Ready**
   Type hints, comprehensive tests, and clear error messages make Rhoa suitable
   for production environments.

Who This Guide Is For
----------------------

**Beginners**
   If you're new to technical analysis or machine learning, start with
   :doc:`basic_concepts` to understand core concepts.

**Data Scientists**
   Experienced with ML but new to finance? Jump to :doc:`targets_guide` to learn
   how Rhoa's target generation works.

**Traders**
   Familiar with technical indicators? See :doc:`indicators_guide` for
   implementation details and best practices.

**Everyone**
   The :doc:`visualization_guide` shows how to create publication-quality charts
   and evaluate model performance.

Getting Started
---------------

If you haven't installed Rhoa yet, see the :doc:`/installation` guide.

For a quick introduction, check out the :doc:`/quickstart` tutorial.

For hands-on examples, browse the :doc:`/examples/index` section.

Module Organization
-------------------

Rhoa is organized into focused modules:

**indicators**
   Technical analysis indicators accessible via ``.indicators`` accessor on pandas Series.
   Includes trend, momentum, volatility, and oscillator indicators.

**targets**
   ML target generation with automatic optimization. Creates binary classification
   targets optimized for your data and trading strategy.

**plots**
   Visualization tools for model evaluation. Plot predictions, confusion matrices,
   and performance analysis.

**data**
   Data utilities for importing from various sources (currently Google Sheets,
   with more coming).

**strategy** (planned)
   Backtesting and strategy evaluation framework.

**preprocess** (planned)
   Data cleaning, normalization, and feature engineering utilities.

Best Practices
--------------

**Always Import Rhoa**
   The accessors only become available after importing:

   .. code-block:: python

      import rhoa  # Required to register accessors

**Handle NaN Values**
   Technical indicators create NaN for initial periods. Always handle these:

   .. code-block:: python

      df = df.dropna()  # Or use appropriate forward/backward fill

**Use Time-Based Splits**
   For financial data, always use time-based train/test splits, never random splits:

   .. code-block:: python

      # CORRECT
      split_idx = int(len(df) * 0.8)
      train, test = df[:split_idx], df[split_idx:]

      # WRONG
      train, test = train_test_split(df)  # Don't do this!

**Save Metadata**
   When generating targets, always save the metadata for reproducibility:

   .. code-block:: python

      targets, meta = generate_target_combinations(df, mode='auto')

      import json
      with open('target_meta.json', 'w') as f:
          json.dump(meta, f)

**Validate Out-of-Sample**
   Never validate on data used for target generation or training:

   .. code-block:: python

      # Generate targets on training data only
      train_targets, meta = generate_target_combinations(train_df, mode='auto')

      # Apply same parameters to test data
      # (using meta parameters)

Common Patterns
---------------

**Multi-Indicator Analysis**

   .. code-block:: python

      # Calculate multiple indicators
      close = df['Close']
      df['SMA_50'] = close.rhoa.indicators.sma(50)
      df['RSI'] = close.rhoa.indicators.rsi(14)
      macd = close.rhoa.indicators.macd()
      df['MACD'] = macd['macd']

**Feature Engineering**

   .. code-block:: python

      # Create features for ML
      df['SMA_20'] = df['Close'].rhoa.indicators.sma(20)
      df['Returns'] = df['Close'].pct_change()
      df['Volatility'] = df['Close'].rhoa.indicators.ewmstd(span=20)

**Target Generation**

   .. code-block:: python

      from rhoa.targets import generate_target_combinations

      targets, meta = generate_target_combinations(
          df, mode='auto', target_class_balance=0.4
      )

**Model Evaluation**

   .. code-block:: python

      # Visualize predictions
      fig = df.rhoa.plots.signal(
          y_pred=predictions,
          y_true=ground_truth,
          date_col='Date',
          price_col='Close'
      )

Next Steps
----------

Continue to the individual guides:

- :doc:`basic_concepts` - Fundamental concepts and terminology
- :doc:`indicators_guide` - Deep dive into technical indicators
- :doc:`targets_guide` - Understanding target generation
- :doc:`visualization_guide` - Creating effective visualizations

Need Help?
----------

- Check the :doc:`/faq` for common questions
- See :doc:`/examples/index` for practical code examples
- Visit the `GitHub repository <https://github.com/nainajnahO/Rhoa>`_ to report issues

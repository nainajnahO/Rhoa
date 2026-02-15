Rhoa Documentation
==================

**Rhoa** is a Python package providing pandas DataFrame extension accessors for technical analysis and machine learning in financial markets.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/basic_concepts
   user_guide/indicators_guide
   user_guide/targets_guide
   user_guide/visualization_guide

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index
   examples/basic_indicators
   examples/advanced_indicators
   examples/target_generation
   examples/complete_pipeline

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/rhoa
   api/indicators
   api/targets
   api/data
   api/preprocess

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   faq
   changelog
   GitHub Repository <https://github.com/nainajnahO/Rhoa>

Overview
--------

Rhoa streamlines technical analysis and machine learning workflows by extending pandas with specialized accessors for:

- **📊 Technical Indicators** - 13 professional-grade indicators
- **🤖 ML Target Generation** - Optimized binary targets with auto/manual modes
- **📈 Visualization** - Model evaluation with confusion matrices and price charts
- **🔗 Pandas Integration** - Seamless integration with existing workflows

Quick Example
-------------

.. code-block:: python

   import pandas as pd
   import rhoa

   # Load your price data
   df = pd.read_csv('stock_prices.csv')

   # Calculate technical indicators
   df['SMA_20'] = df['Close'].rhoa.indicators.sma(window_size=20)
   df['RSI_14'] = df['Close'].rhoa.indicators.rsi(window_size=14)

   # Generate ML targets
   from rhoa.targets import generate_target_combinations
   targets, meta = generate_target_combinations(df, mode='auto')

   # Visualize predictions
   fig = df.rhoa.plots.signal(y_pred=predictions, y_true=targets['Target_7'])

Key Features
------------

**Pandas-Native Interface**
   All functionality accessible via pandas accessors. No need to learn a new API.

**Intelligent Target Generation**
   Automatic optimization finds the best lookback period and threshold for your data.

**Production-Ready**
   Type hints, comprehensive tests, and proper error handling throughout.

**Well-Documented**
   Every function has detailed docstrings with examples and best practices.

Installation
------------

Install via pip:

.. code-block:: bash

   pip install rhoa

For detailed installation instructions, see the :doc:`installation` guide.

Getting Help
------------

- **New Users**: Start with the :doc:`quickstart` tutorial
- **Examples**: Browse :doc:`examples/index` for practical code
- **Concepts**: Read the :doc:`user_guide/index` for in-depth explanations
- **API Details**: Check the :doc:`api/index` for complete reference
- **Questions**: See the :doc:`faq` for common issues
- **Issues**: Report bugs on `GitHub <https://github.com/nainajnahO/Rhoa/issues>`_

Project Information
-------------------

- **Version**: 0.1.7
- **License**: GNU General Public License v3.0
- **Author**: nainajnahO
- **Python**: 3.9+
- **Status**: Pre-Alpha

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

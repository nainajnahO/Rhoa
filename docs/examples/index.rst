Examples
========

This section provides practical examples demonstrating how to use Rhoa for various technical analysis and machine learning tasks.

.. toctree::
   :maxdepth: 2
   :caption: Example Categories

   basic_indicators
   advanced_indicators
   target_generation
   complete_pipeline

Overview
--------

The examples are organized by complexity and use case:

**Basic Indicators**
   Simple examples using individual technical indicators like SMA, RSI, and moving averages.
   Perfect for beginners.

**Advanced Indicators**
   Examples using complex indicators that require OHLC data: MACD, Bollinger Bands,
   ADX, Stochastic Oscillator, and more.

**Target Generation**
   Learn how to generate optimized binary targets for machine learning models using
   both auto and manual modes.

**Complete Pipeline**
   End-to-end examples showing how to build complete ML pipelines from data loading
   to model training and evaluation.

Quick Reference
---------------

Common Tasks
~~~~~~~~~~~~

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Task
     - Example Link
   * - Calculate Simple Moving Average
     - :ref:`basic-sma`
   * - Find Overbought Conditions (RSI)
     - :ref:`basic-rsi`
   * - Detect MACD Crossovers
     - :ref:`advanced-macd`
   * - Calculate Bollinger Bands
     - :ref:`advanced-bollinger`
   * - Generate ML Targets (Auto Mode)
     - :ref:`target-auto`
   * - Generate ML Targets (Manual Mode)
     - :ref:`target-manual`
   * - Build Complete ML Pipeline
     - :ref:`pipeline-ml`

Example Data
------------

Most examples assume you have OHLC (Open, High, Low, Close) price data in a CSV file:

.. code-block:: python

   import pandas as pd
   import rhoa

   # Load your data
   df = pd.read_csv('your_prices.csv')

   # Expected columns: Date, Open, High, Low, Close, Volume
   # Date should be parseable as datetime

Sample Data Format
~~~~~~~~~~~~~~~~~~

Your CSV file should look like this:

.. code-block:: text

   Date,Open,High,Low,Close,Volume
   2024-01-01,100.0,105.0,99.0,103.0,1000000
   2024-01-02,103.0,107.0,102.0,106.0,1200000
   2024-01-03,106.0,108.0,104.0,105.0,950000
   ...

Getting Test Data
~~~~~~~~~~~~~~~~~

For testing, you can use the sample data included in the tests directory:

.. code-block:: python

   import pandas as pd

   # Using Rhoa's test data
   df = pd.read_csv('https://raw.githubusercontent.com/nainajnahO/Rhoa/main/tests/data.csv')

Or download financial data using yfinance:

.. code-block:: python

   import yfinance as yf

   # Download stock data
   ticker = yf.Ticker("AAPL")
   df = ticker.history(period="1y")
   df = df.reset_index()

Next Steps
----------

Start with :doc:`basic_indicators` if you're new to Rhoa, then progress to more
advanced examples as you become comfortable with the basics.

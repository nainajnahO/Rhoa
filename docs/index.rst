.. raw:: html

   <div class="logo-container">
      <img src="_static/logo.png" alt="Rhoa Logo" class="main-logo" />
   </div>

Rhoa Documentation
==================

Rhoa is a Python package providing pandas DataFrame extension accessors for technical analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index
   examples

Installation
------------

.. code-block:: bash

   pip install rhoa

Quick Start
-----------

.. code-block:: python

   import pandas as pd
   import rhoa

   # Load your price data
   df = pd.read_csv('your_data.csv')
   
   # Calculate Simple Moving Average
   sma = df['close'].indicators.sma(window_size=20)
   
   # Calculate RSI
   rsi = df['close'].indicators.rsi(window_size=14)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
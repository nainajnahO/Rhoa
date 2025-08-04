rhoa.indicators module
======================

.. automodule:: rhoa.indicators
   :members:
   :undoc-members:
   :show-inheritance:

The indicators module provides technical analysis indicators through pandas Series accessor.

indicators Class
----------------

.. autoclass:: rhoa.indicators.indicators
   :members:
   :undoc-members:
   :show-inheritance:

Function Reference
------------------

Moving Averages
~~~~~~~~~~~~~~~

.. autofunction:: rhoa.indicators.indicators.sma
.. autofunction:: rhoa.indicators.indicators.ewma  
.. autofunction:: rhoa.indicators.indicators.ewmv
.. autofunction:: rhoa.indicators.indicators.ewmstd

Momentum Oscillators
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rhoa.indicators.indicators.rsi
.. autofunction:: rhoa.indicators.indicators.stochastic
.. autofunction:: rhoa.indicators.indicators.williams_r
.. autofunction:: rhoa.indicators.indicators.cci

Trend Indicators
~~~~~~~~~~~~~~~~

.. autofunction:: rhoa.indicators.indicators.macd
.. autofunction:: rhoa.indicators.indicators.adx
.. autofunction:: rhoa.indicators.indicators.parabolic_sar

Volatility Indicators
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rhoa.indicators.indicators.bollinger_bands
.. autofunction:: rhoa.indicators.indicators.atr
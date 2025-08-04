rhoa.indicators module
======================

.. automodule:: rhoa.indicators
   :members:
   :undoc-members:
   :show-inheritance:

The indicators module provides technical analysis indicators through pandas Series accessor.

IndicatorsAccessor Class
------------------------

.. autoclass:: rhoa.indicators.IndicatorsAccessor
   :members:
   :undoc-members:
   :show-inheritance:

Function Reference
------------------

Moving Averages
~~~~~~~~~~~~~~~

.. autofunction:: rhoa.indicators.IndicatorsAccessor.sma
.. autofunction:: rhoa.indicators.IndicatorsAccessor.ewma  
.. autofunction:: rhoa.indicators.IndicatorsAccessor.ewmv
.. autofunction:: rhoa.indicators.IndicatorsAccessor.ewmstd

Momentum Oscillators
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rhoa.indicators.IndicatorsAccessor.rsi
.. autofunction:: rhoa.indicators.IndicatorsAccessor.stochastic
.. autofunction:: rhoa.indicators.IndicatorsAccessor.williams_r
.. autofunction:: rhoa.indicators.IndicatorsAccessor.cci

Trend Indicators
~~~~~~~~~~~~~~~~

.. autofunction:: rhoa.indicators.IndicatorsAccessor.macd
.. autofunction:: rhoa.indicators.IndicatorsAccessor.adx
.. autofunction:: rhoa.indicators.IndicatorsAccessor.parabolic_sar

Volatility Indicators
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rhoa.indicators.IndicatorsAccessor.bollinger_bands
.. autofunction:: rhoa.indicators.IndicatorsAccessor.atr
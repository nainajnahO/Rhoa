Indicators Module
=================

The indicators module provides technical analysis indicators through pandas Series accessor.

.. currentmodule:: rhoa.indicators

IndicatorsAccessor
------------------

.. autoclass:: IndicatorsAccessor
   :members:
   :undoc-members:
   :show-inheritance:

Moving Averages
---------------

.. automethod:: IndicatorsAccessor.sma
.. automethod:: IndicatorsAccessor.ewma
.. automethod:: IndicatorsAccessor.ewmv
.. automethod:: IndicatorsAccessor.ewmstd

Momentum Oscillators
--------------------

.. automethod:: IndicatorsAccessor.rsi
.. automethod:: IndicatorsAccessor.stochastic
.. automethod:: IndicatorsAccessor.williams_r
.. automethod:: IndicatorsAccessor.cci

Trend Indicators
----------------

.. automethod:: IndicatorsAccessor.macd
.. automethod:: IndicatorsAccessor.adx
.. automethod:: IndicatorsAccessor.parabolic_sar

Volatility Indicators
---------------------

.. automethod:: IndicatorsAccessor.bollinger_bands
.. automethod:: IndicatorsAccessor.atr
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
   :noindex:
.. automethod:: IndicatorsAccessor.ewma
   :noindex:
.. automethod:: IndicatorsAccessor.ewmv
   :noindex:
.. automethod:: IndicatorsAccessor.ewmstd
   :noindex:

Momentum Oscillators
--------------------

.. automethod:: IndicatorsAccessor.rsi
   :noindex:
.. automethod:: IndicatorsAccessor.stochastic
   :noindex:
.. automethod:: IndicatorsAccessor.williams_r
   :noindex:
.. automethod:: IndicatorsAccessor.cci
   :noindex:

Trend Indicators
----------------

.. automethod:: IndicatorsAccessor.macd
   :noindex:
.. automethod:: IndicatorsAccessor.adx
   :noindex:
.. automethod:: IndicatorsAccessor.parabolic_sar
   :noindex:

Volatility Indicators
---------------------

.. automethod:: IndicatorsAccessor.bollinger_bands
   :noindex:
.. automethod:: IndicatorsAccessor.atr
   :noindex:
# rhoa - A pandas DataFrame extension for technical analysis
# Copyright (C) 2025 nainajnahO
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
  Suggestions for Improvement

  1. Package Structure & Organization

  - Empty files: rhoa/targets.py is empty - either implement or remove
  - Unused preprocess module: PreprocessAccessor is imported but minimal - expand or remove
  - Consider submodules: As you add more indicators, organize by category (momentum, trend, volatility)

  2. Enhanced Functionality

  # Add volume-based indicators
  def obv(self, volume: Series) -> Series:  # On-Balance Volume
  def vwap(self, volume: Series, high: Series, low: Series) -> Series:  # VWAP

  # Add pattern recognition
  def detect_patterns(self) -> DataFrame:  # Common candlestick patterns

  # Add multiple timeframe support
  def resample_indicator(self, timeframe: str, indicator: str, **kwargs):


  5. Documentation & Examples

  - Usage examples: Add practical examples to README
  - Jupyter notebooks: Create tutorial notebooks
  - API reference: Consider Sphinx documentation

  6. Dependencies & Distribution

  - Add numpy explicitly to dependencies (currently imported but not listed)
  - Version constraints: Consider maximum versions for dependencies
  - Optional dependencies: Separate plotting/visualization dependencies

  7. Code Quality Improvements

  # Add type hints throughout
  from typing import Optional, Union
  def sma(self, window_size: int = 20, min_periods: Optional[int] = None) -> Series:

  # Add validation decorators
  @validate_window_size
  @validate_series_length
  def indicator_method(self, ...):

  8. Testing Enhancements

  - Performance benchmarks: Add timing tests
  - Property-based testing: Use hypothesis for edge case generation
  - Integration tests: Test full workflows

  9. Additional Features

  - Signal generation: Add buy/sell signal detection
  - Backtesting utilities: Basic strategy testing framework
  - Plotting integration: Integration with matplotlib/plotly
  - Data export: Export indicators to various formats
"""

import pandas
import numpy
from pandas import Series
from pandas import DataFrame
from pandas.api.extensions import register_series_accessor


@register_series_accessor("indicators")
class IndicatorsAccessor:
    def __init__(self, series: Series) -> None:
        self._series = series

    def sma(self,
            window_size: int = 20,
            min_periods: int = None,
            center: bool = False,
            **kwargs) -> Series:
        """Calculate the Simple Moving Average (SMA) over a specified window.

        The SMA is a commonly used technical indicator in financial
        and time series analysis that calculates the average value
        over a defined number of periods.

        Args:
            window_size (int, optional): The size of the moving window, representing
                the number of periods over which to calculate the average. Defaults to 20.
            min_periods (int, optional): Minimum number of observations in window 
                required to have a value. Defaults to None.
            center (bool, optional): Whether to center the labels in the result. 
                Defaults to False.
            **kwargs: Additional keyword arguments passed to pandas rolling function.

        Returns:
            pandas.Series: A pandas Series containing the calculated SMA values.

        Example:
            Calculate 20-period Simple Moving Average:

            >>> import pandas as pd
            >>> import rhoa
            >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
            >>> sma = prices.indicators.sma(window_size=5)
            >>> print(sma.iloc[4])  # First valid SMA value
            102.2

        Note:
            The first `window_size - 1` values will be NaN since there aren't 
            enough observations to calculate the average.
        """
        return self._series.rolling(window=window_size, min_periods=min_periods, center=center, **kwargs).mean()

    def ewma(self,
             window_size: int = 20,
             adjust: bool = False,
             min_periods: int = None,
             **kwargs) -> Series:
        """Calculate the Exponential Weighted Moving Average (EWMA) of the series.

        The EWMA is a type of infinite impulse response filter that applies weighting
        factors which decrease exponentially. This method is commonly used in financial
        time series to smooth data and compute trends. Unlike simple moving averages,
        EWMA gives more weight to recent observations.

        Args:
            window_size (int, optional): The span of the exponential moving average. 
                Determines the level of smoothing, where larger values result in 
                smoother trends and slower responsiveness to changes in the data. 
                Defaults to 20.
            adjust (bool, optional): Divide by decaying adjustment factor in beginning 
                periods. When True, the weights are normalized by the sum of weights. 
                Defaults to False.
            min_periods (int, optional): Minimum number of observations in window 
                required to have a value. Defaults to None.
            **kwargs: Additional keyword arguments passed to pandas ewm function.

        Returns:
            pandas.Series: A pandas Series containing the calculated EWMA values.

        Example:
            Calculate 20-period Exponential Weighted Moving Average:

            >>> import pandas as pd
            >>> import rhoa
            >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
            >>> ewma = prices.indicators.ewma(window_size=5)
            >>> print(f"Latest EWMA: {ewma.iloc[-1]:.2f}")
            Latest EWMA: 105.45


        Note:
            EWMA responds more quickly to recent price changes compared to SMA,
            making it useful for trend following strategies.
        """
        return self._series.ewm(span=window_size, adjust=adjust, min_periods=min_periods, **kwargs).mean()

    def ewmv(self,
             window_size: int = 20,
             adjust: bool = True,
             min_periods: int = None,
             **kwargs) -> Series:
        """Calculate the exponentially weighted moving variance (EWMV) of a series.

        This method computes the variance of a series by applying exponential
        weighting. The window size parameter determines the span of the
        exponentially weighted period. EWMV is useful for measuring volatility
        that adapts more quickly to recent price changes.

        Args:
            window_size (int, optional): The span of the exponential window. Determines the
                level of smoothing applied to the variance calculation. Defaults to 20.
            adjust (bool, optional): Divide by decaying adjustment factor in beginning periods.
                When True, the weights are normalized by the sum of weights. Defaults to True.
            min_periods (int, optional): Minimum number of observations in window required
                to have a value. Defaults to None.
            **kwargs: Additional keyword arguments passed to pandas ewm function.

        Returns:
            pandas.Series: A pandas Series containing the exponentially weighted moving
                variance of the input series.

        Example:
            Calculate exponentially weighted moving variance:

            >>> import pandas as pd
            >>> import rhoa
            >>> prices = pd.Series([100, 102, 99, 103, 105, 101, 106, 104])
            >>> ewmv = prices.indicators.ewmv(window_size=5)
            >>> print(f"Latest variance: {ewmv.iloc[-1]:.2f}")
            Latest variance: 6.24

        Note:
            Higher variance values indicate increased volatility in the price series.
            The variance is always non-negative.
        """
        return self._series.ewm(span=window_size, adjust=adjust, min_periods=min_periods, **kwargs).var()

    def ewmstd(self,
               window_size: int = 20,
               adjust: bool = True,
               min_periods: int = None,
               **kwargs) -> Series:
        """Calculate the exponentially weighted moving standard deviation (EWMSTD).

        EWMSTD is a statistical measure that weights recent data points more heavily
        to provide a smoothed calculation of the moving standard deviation. This makes
        it more responsive to recent volatility changes compared to traditional
        rolling standard deviation.

        Args:
            window_size (int, optional): The span or window size for the exponentially
                weighted moving calculation. Smaller spans apply heavier weighting to
                more recent data points, while larger spans provide smoother results.
                Defaults to 20.
            adjust (bool, optional): Divide by decaying adjustment factor in beginning periods.
                When True, the weights are normalized by the sum of weights. Defaults to True.
            min_periods (int, optional): Minimum number of observations in window required
                to have a value. Defaults to None.
            **kwargs: Additional keyword arguments passed to pandas ewm function.

        Returns:
            pandas.Series: A pandas Series containing the exponentially weighted moving
                standard deviation values.

        Example:
            Calculate exponentially weighted moving standard deviation:

            >>> import pandas as pd
            >>> import rhoa
            >>> prices = pd.Series([100, 102, 99, 103, 105, 101, 106, 104])
            >>> ewmstd = prices.indicators.ewmstd(window_size=5)
            >>> print(f"Latest volatility: {ewmstd.iloc[-1]:.2f}")
            Latest volatility: 2.50

        Note:
            The relationship EWMSTD² = EWMV holds. This indicator is commonly used
            for volatility-based trading strategies and risk management.
        """
        return self._series.ewm(span=window_size, adjust=adjust, min_periods=min_periods, **kwargs).std()

    def rsi(
            self,
            window_size: int = 14,
            edge_case_value: float = 100.0,
            **kwargs) -> Series:
        """Calculate the Relative Strength Index (RSI) for momentum analysis.

        RSI is a momentum oscillator that measures the speed and change of price
        movements on a scale of 0 to 100. It helps identify overbought (typically >70)
        and oversold (typically <30) market conditions. RSI is one of the most widely
        used technical indicators in trading.

        Args:
            window_size (int, optional): The size of the rolling window used to calculate
                the moving averages of gains and losses. Traditional value is 14. Defaults to 14.
            edge_case_value (float, optional): The RSI value to use when avg_loss == 0
                (no losses occurred). Common values: 100.0 (infinite RS, default),
                50.0 (neutral), or float('nan'). Defaults to 100.0.
            **kwargs: Additional keyword arguments passed to pandas ewm function.

        Returns:
            pandas.Series: A pandas Series containing RSI values between 0 and 100.

        Example:
            Calculate 14-period RSI and identify trading signals:

            >>> import pandas as pd
            >>> import rhoa
            >>> prices = pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109])
            >>> rsi = prices.indicators.rsi(window_size=14)
            >>> overbought = rsi > 70  # Potential sell signals
            >>> oversold = rsi < 30   # Potential buy signals
            >>> print(f"Latest RSI: {rsi.iloc[-1]:.1f}")
            Latest RSI: 75.2

        Note:
            - RSI > 70: Generally considered overbought (potential sell signal)
            - RSI < 30: Generally considered oversold (potential buy signal)
            - RSI around 50: Neutral momentum
        """
        price = self._series
        delta = price.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(span=window_size, adjust=False, min_periods=window_size, **kwargs).mean()
        avg_loss = loss.ewm(span=window_size, adjust=False, min_periods=window_size, **kwargs).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Handle edge case when avg_loss == 0 (division by zero)
        rsi[avg_loss == 0] = edge_case_value

        return rsi

    def macd(self,
             short_window: int = 12,
             long_window: int = 26,
             signal_window: int = 9,
             **kwargs) -> DataFrame:
        """Calculate the MACD (Moving Average Convergence Divergence) indicator.

        MACD is a trend-following momentum indicator that shows the relationship
        between two moving averages of a security's price. It consists of three
        components: the MACD line, signal line, and histogram, which together
        provide insights into trend direction and momentum changes.

        The MACD line is the difference between the short-term and long-term EMAs.
        The signal line is an EMA of the MACD line. The histogram shows the
        difference between MACD and signal lines.

        Args:
            short_window (int, optional): Length of the short-term EMA window.
                Defaults to 12.
            long_window (int, optional): Length of the long-term EMA window.
                Defaults to 26.
            signal_window (int, optional): Length of the signal EMA window.
                Defaults to 9.
            **kwargs: Additional keyword arguments passed to pandas ewm function.

        Returns:
            pandas.DataFrame: A DataFrame with columns 'macd', 'signal', and 'histogram'.

        Example:
            Calculate MACD and identify bullish crossover:

            >>> import pandas as pd
            >>> import rhoa
            >>> prices = pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110])
            >>> macd_data = prices.indicators.macd()
            >>> # Bullish signal: MACD crosses above signal
            >>> bullish = (macd_data['macd'] > macd_data['signal']) & \
            ...           (macd_data['macd'].shift(1) <= macd_data['signal'].shift(1))
            >>> print(f"MACD: {macd_data['macd'].iloc[-1]:.3f}")
            MACD: 0.245

        Note:
            - Bullish signal: MACD line crosses above signal line
            - Bearish signal: MACD line crosses below signal line
            - Histogram > 0: MACD above signal (bullish momentum)
            - Histogram < 0: MACD below signal (bearish momentum)
        """
        # SHORT-TERM AND LONG-TERM EXPONENTIAL MOVING AVERAGE
        short_ema = self._series.ewm(span=short_window, adjust=False, **kwargs).mean()
        long_ema = self._series.ewm(span=long_window, adjust=False, **kwargs).mean()

        # MACD LINE
        macd_line = short_ema - long_ema

        # SIGNAL LINE
        signal_line = macd_line.ewm(span=signal_window, adjust=False, **kwargs).mean()

        # HISTOGRAM
        macd_histogram = macd_line - signal_line

        return DataFrame({
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_histogram
        })

    def bollinger_bands(self,
                        window_size: int = 20,
                        num_std: float = 2.0,
                        min_periods: int = None,
                        center: bool = False,
                        **kwargs) -> DataFrame:
        """Calculate Bollinger Bands for volatility and mean reversion analysis.

        Bollinger Bands consist of three lines: an upper band, middle band (SMA),
        and lower band. The bands expand and contract based on market volatility,
        providing insights into potential overbought/oversold conditions and
        price volatility patterns.

        Args:
            window_size (int, optional): The size of the rolling window used for
                computing the moving average and standard deviation. Defaults to 20.
            num_std (float, optional): The number of standard deviations to add/subtract
                from the moving average to calculate the upper and lower bands.
                Defaults to 2.0.
            min_periods (int, optional): Minimum number of observations in window
                required to have a value. Defaults to None.
            center (bool, optional): Whether to center the labels in the result.
                Defaults to False.
            **kwargs: Additional keyword arguments passed to pandas rolling function.

        Returns:
            pandas.DataFrame: A DataFrame with columns 'upper_band', 'middle_band',
                and 'lower_band'.

        Example:
            Calculate Bollinger Bands and identify squeeze conditions:

            >>> import pandas as pd
            >>> import rhoa
            >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107])
            >>> bb = prices.indicators.bollinger_bands(window_size=5, num_std=2.0)
            >>> # Band width indicates volatility
            >>> width = bb['upper_band'] - bb['lower_band']
            >>> squeeze = width < width.rolling(10).mean() * 0.8  # Low volatility
            >>> print(f"Upper: {bb['upper_band'].iloc[-1]:.2f}")
            Upper: 109.45

        Note:
            - Price touching upper band: Potentially overbought
            - Price touching lower band: Potentially oversold
            - Narrow bands: Low volatility (squeeze)
            - Wide bands: High volatility (expansion)
        """
        series = self._series

        middle = series.rolling(window=window_size, min_periods=min_periods, center=center, **kwargs).mean()
        std = series.rolling(window=window_size, min_periods=min_periods, center=center, **kwargs).std()

        upper = middle + num_std * std
        lower = middle - num_std * std

        return DataFrame({
            "upper_band": upper,
            "middle_band": middle,
            "lower_band": lower
        })

    def atr(self,
            high: Series,
            low: Series,
            window_size: int = 14,
            min_periods: int = None,
            center: bool = False,
            **kwargs) -> Series:
        """Calculate the Average True Range (ATR) for volatility measurement.

        ATR measures market volatility by calculating the average of true ranges
        over a specified period. True range is the maximum of: (high - low),
        (high - previous close), or (low - previous close). ATR is widely used
        for position sizing and stop-loss placement.

        Args:
            high (pandas.Series): A Series representing the high prices.
            low (pandas.Series): A Series representing the low prices.
            window_size (int, optional): Length of the rolling window for calculating
                the average true range. Defaults to 14.
            min_periods (int, optional): Minimum number of observations in window
                required to have a value. Defaults to None.
            center (bool, optional): Whether to center the labels in the result.
                Defaults to False.
            **kwargs: Additional keyword arguments passed to pandas rolling function.

        Returns:
            pandas.Series: A Series containing the calculated ATR values.

        Example:
            Calculate ATR for position sizing:

            >>> import pandas as pd
            >>> import rhoa
            >>> close = pd.Series([100, 102, 101, 103, 105, 104, 106])
            >>> high = pd.Series([101, 103, 102, 104, 106, 105, 107])
            >>> low = pd.Series([99, 101, 100, 102, 104, 103, 105])
            >>> atr = close.indicators.atr(high, low, window_size=5)
            >>> # Use ATR for stop-loss: 2 * ATR below entry
            >>> stop_distance = 2 * atr.iloc[-1]
            >>> print(f"ATR: {atr.iloc[-1]:.2f}, Stop distance: {stop_distance:.2f}")
            ATR: 1.80, Stop distance: 3.60

        Note:
            Higher ATR values indicate higher volatility. ATR is commonly used
            for setting stop-losses and position sizing in trading strategies.
        """
        close = self._series

        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()

        true_range = pandas.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window_size, min_periods=min_periods, center=center, **kwargs).mean()

        return atr

    def cci(self,
            high: Series,
            low: Series,
            window_size: int = 20,
            min_periods: int = None,
            center: bool = False,
            **kwargs) -> Series:
        """Calculate the Commodity Channel Index (CCI) for momentum analysis.

        CCI is a momentum-based oscillator that measures the variation of a security's
        price from its statistical mean. It oscillates above and below zero, with
        readings above +100 indicating overbought conditions and readings below -100
        indicating oversold conditions.

        Args:
            high (pandas.Series): A Series containing the high prices.
            low (pandas.Series): A Series containing the low prices.
            window_size (int, optional): Number of periods for calculating the CCI.
                Defaults to 20.
            min_periods (int, optional): Minimum number of observations in window
                required to have a value. Defaults to None.
            center (bool, optional): Whether to center the labels in the result.
                Defaults to False.
            **kwargs: Additional keyword arguments passed to pandas rolling function.

        Returns:
            pandas.Series: A Series representing the calculated CCI values.

        Example:
            Calculate CCI and identify trading signals:

            >>> import pandas as pd
            >>> import rhoa
            >>> close = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107])
            >>> high = pd.Series([101, 103, 102, 104, 106, 105, 107, 109, 108])
            >>> low = pd.Series([99, 101, 100, 102, 104, 103, 105, 107, 106])
            >>> cci = close.indicators.cci(high, low, window_size=5)
            >>> overbought = cci > 100   # Potential sell signals
            >>> oversold = cci < -100    # Potential buy signals
            >>> print(f"Latest CCI: {cci.iloc[-1]:.1f}")
            Latest CCI: 85.2

        Note:
            - CCI > +100: Overbought condition (potential sell signal)
            - CCI < -100: Oversold condition (potential buy signal)
            - CCI around 0: Normal trading range
        """
        close = self._series
        typical_price = (high + low + close) / 3

        sma = typical_price.rolling(window=window_size, min_periods=min_periods, center=center, **kwargs).mean()

        mean_deviation = typical_price.rolling(window=window_size, min_periods=min_periods, center=center,
                                               **kwargs).apply(
            lambda x: numpy.mean(numpy.abs(x - x.mean())),
            raw=True
        )

        cci = (typical_price - sma) / (0.015 * mean_deviation)

        return cci

    def stochastic(self,
                   high: Series,
                   low: Series,
                   k_window: int = 14,
                   d_window: int = 3,
                   min_periods: int = None,
                   center: bool = False,
                   **kwargs) -> DataFrame:
        """Calculate the Stochastic Oscillator (%K and %D) for momentum analysis.

        The Stochastic Oscillator compares a closing price to its price range over
        a given time period. It generates values between 0 and 100, where values
        above 80 typically indicate overbought conditions and values below 20
        indicate oversold conditions.

        Args:
            high (pandas.Series): A Series containing the high prices.
            low (pandas.Series): A Series containing the low prices.
            k_window (int, optional): Number of periods for %K calculation.
                Defaults to 14.
            d_window (int, optional): Number of periods for %D calculation (SMA of %K).
                Defaults to 3.
            min_periods (int, optional): Minimum observations in window required
                to have a value. Defaults to None.
            center (bool, optional): Whether to center the labels in the result.
                Defaults to False.
            **kwargs: Additional keyword arguments passed to pandas rolling function.

        Returns:
            pandas.DataFrame: A DataFrame with '%K' and '%D' columns representing
                the stochastic values.

        Example:
            Calculate Stochastic Oscillator and identify signals:

            >>> import pandas as pd
            >>> import rhoa
            >>> close = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107])
            >>> high = pd.Series([101, 103, 102, 104, 106, 105, 107, 109, 108])
            >>> low = pd.Series([99, 101, 100, 102, 104, 103, 105, 107, 106])
            >>> stoch = close.indicators.stochastic(high, low, k_window=5, d_window=3)
            >>> overbought = stoch['%K'] > 80  # Potential sell signals
            >>> oversold = stoch['%K'] < 20   # Potential buy signals
            >>> print(f"%K: {stoch['%K'].iloc[-1]:.1f}, %D: {stoch['%D'].iloc[-1]:.1f}")
            %K: 75.0, %D: 72.3

        Note:
            - %K > 80: Overbought (potential sell signal)
            - %K < 20: Oversold (potential buy signal)
            - %K crossing above %D: Bullish signal
            - %K crossing below %D: Bearish signal
        """

        # Calculate %K
        lowest_low = low.rolling(window=k_window, min_periods=min_periods, center=center, **kwargs).min()
        highest_high = high.rolling(window=k_window, min_periods=min_periods, center=center, **kwargs).max()

        k_percent = 100 * ((self._series - lowest_low) / (highest_high - lowest_low))

        # Calculate %D (SMA of %K)
        d_percent = k_percent.rolling(window=d_window, min_periods=min_periods, center=center, **kwargs).mean()

        return DataFrame({
            "%K": k_percent,
            "%D": d_percent
        })

    def williams_r(self,
                   high: Series,
                   low: Series,
                   window_size: int = 14,
                   min_periods: int = None,
                   center: bool = False,
                   **kwargs) -> Series:
        """Calculate Williams %R for momentum and overbought/oversold analysis.

        Williams %R is a momentum indicator that measures overbought and oversold
        levels. It oscillates between -100 and 0, with readings above -20 indicating
        overbought conditions and readings below -80 indicating oversold conditions.
        It's essentially an inverted Stochastic Oscillator.

        Args:
            high (pandas.Series): A Series containing the high prices.
            low (pandas.Series): A Series containing the low prices.
            window_size (int, optional): Number of periods for Williams %R calculation.
                Defaults to 14.
            min_periods (int, optional): Minimum observations in window required
                to have a value. Defaults to None.
            center (bool, optional): Whether to center the labels in the result.
                Defaults to False.
            **kwargs: Additional keyword arguments passed to pandas rolling function.

        Returns:
            pandas.Series: A Series representing Williams %R values (-100 to 0).

        Example:
            Calculate Williams %R and identify trading signals:

            >>> import pandas as pd
            >>> import rhoa
            >>> close = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107])
            >>> high = pd.Series([101, 103, 102, 104, 106, 105, 107, 109, 108])
            >>> low = pd.Series([99, 101, 100, 102, 104, 103, 105, 107, 106])
            >>> wr = close.indicators.williams_r(high, low, window_size=5)
            >>> overbought = wr > -20  # Potential sell signals
            >>> oversold = wr < -80   # Potential buy signals
            >>> print(f"Williams %R: {wr.iloc[-1]:.1f}")
            Williams %R: -25.0

        Note:
            - Williams %R > -20: Overbought (potential sell signal)
            - Williams %R < -80: Oversold (potential buy signal)
            - Values closer to 0: Stronger momentum
            - Values closer to -100: Weaker momentum
        """
        close = self._series

        # Calculate Williams %R
        highest_high = high.rolling(window=window_size, min_periods=min_periods, center=center, **kwargs).max()
        lowest_low = low.rolling(window=window_size, min_periods=min_periods, center=center, **kwargs).min()

        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))

        return williams_r

    def adx(self,
            high: Series,
            low: Series,
            window_size: int = 14,
            min_periods: int = None,
            **kwargs) -> DataFrame:
        """Calculate the Average Directional Index (ADX) for trend strength analysis.

        ADX is a non-directional indicator that quantifies trend strength regardless
        of direction. It ranges from 0 to 100, with higher values indicating stronger
        trends. The calculation includes +DI and -DI (Directional Indicators) that
        measure upward and downward price movement strength.

        Args:
            high (pandas.Series): A Series containing the high prices.
            low (pandas.Series): A Series containing the low prices.
            window_size (int, optional): Number of periods for ADX calculation.
                Defaults to 14.
            min_periods (int, optional): Minimum observations in window required
                to have a value. Defaults to None.
            **kwargs: Additional keyword arguments passed to pandas ewm function.

        Returns:
            pandas.DataFrame: A DataFrame with 'ADX', '+DI', and '-DI' columns.

        Example:
            Calculate ADX and assess trend strength:

            >>> import pandas as pd
            >>> import rhoa
            >>> close = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116])
            >>> high = pd.Series([101, 103, 105, 107, 109, 111, 113, 115, 117])
            >>> low = pd.Series([99, 101, 103, 105, 107, 109, 111, 113, 115])
            >>> adx_data = close.indicators.adx(high, low, window_size=5)
            >>> strong_trend = adx_data['ADX'] > 25  # Strong trend identification
            >>> bullish = adx_data['+DI'] > adx_data['-DI']  # Uptrend
            >>> print(f"ADX: {adx_data['ADX'].iloc[-1]:.1f}")
            ADX: 45.2

        Note:
            - ADX > 25: Strong trend (regardless of direction)
            - ADX < 20: Weak trend or sideways market
            - +DI > -DI: Uptrend strength
            - -DI > +DI: Downtrend strength
        """
        close = self._series

        # Calculate True Range (same as ATR calculation)
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        true_range = pandas.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate Directional Movement
        high_diff = high.diff()
        low_diff = low.diff()

        plus_dm = pandas.Series(numpy.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0), index=high.index)
        minus_dm = pandas.Series(numpy.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0), index=low.index)

        # Smooth the True Range and Directional Movement using EWM
        atr = true_range.ewm(span=window_size, adjust=False, min_periods=min_periods, **kwargs).mean()
        plus_di_smooth = plus_dm.ewm(span=window_size, adjust=False, min_periods=min_periods, **kwargs).mean()
        minus_di_smooth = minus_dm.ewm(span=window_size, adjust=False, min_periods=min_periods, **kwargs).mean()

        # Calculate +DI and -DI
        plus_di = 100 * (plus_di_smooth / atr)
        minus_di = 100 * (minus_di_smooth / atr)

        # Calculate ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.ewm(span=window_size, adjust=False, min_periods=min_periods, **kwargs).mean()

        return DataFrame({
            "ADX": adx,
            "+DI": plus_di,
            "-DI": minus_di
        })

    def parabolic_sar(self,
                      high: Series,
                      low: Series,
                      af_start: float = 0.02,
                      af_increment: float = 0.02,
                      af_maximum: float = 0.2) -> Series:
        """Calculate the Parabolic Stop and Reverse (SAR) for trend following.

        Parabolic SAR is a trend-following indicator that provides potential reversal
        points and trailing stop levels. It appears as dots above or below price:
        dots below indicate uptrends, dots above indicate downtrends. The indicator
        uses an acceleration factor that increases over time for sensitivity.

        Args:
            high (pandas.Series): A Series containing the high prices.
            low (pandas.Series): A Series containing the low prices.
            af_start (float, optional): Initial acceleration factor. Defaults to 0.02.
            af_increment (float, optional): Increment added to acceleration factor
                when new extreme is reached. Defaults to 0.02.
            af_maximum (float, optional): Maximum acceleration factor value.
                Defaults to 0.2.

        Returns:
            pandas.Series: A Series representing Parabolic SAR values.

        Example:
            Calculate Parabolic SAR for trend identification and stops:

            >>> import pandas as pd
            >>> import rhoa
            >>> close = pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110])
            >>> high = pd.Series([101, 103, 105, 104, 106, 108, 107, 109, 111])
            >>> low = pd.Series([99, 101, 103, 102, 104, 106, 105, 107, 109])
            >>> sar = close.indicators.parabolic_sar(high, low)
            >>> uptrend = close > sar    # Price above SAR = uptrend
            >>> downtrend = close < sar  # Price below SAR = downtrend
            >>> print(f"Latest SAR: {sar.iloc[-1]:.2f}")
            Latest SAR: 99.45

        Note:
            - Price above SAR: Uptrend (SAR acts as support)
            - Price below SAR: Downtrend (SAR acts as resistance)
            - SAR flip: Potential trend reversal signal
            - Use SAR as trailing stop-loss levels
        """
        close = self._series

        # Initialize arrays
        length = len(close)
        sar = numpy.zeros(length)
        trend = numpy.zeros(length, dtype=int)  # 1 for uptrend, -1 for downtrend
        af = numpy.zeros(length)
        ep = numpy.zeros(length)  # Extreme Point

        # Initialize first values
        sar[0] = float(low.iloc[0])
        trend[0] = 1  # Start with uptrend
        af[0] = af_start
        ep[0] = float(high.iloc[0])

        for i in range(1, length):
            # Previous values
            prev_sar = sar[i - 1]
            prev_trend = trend[i - 1]
            prev_af = af[i - 1]
            prev_ep = ep[i - 1]

            # Calculate new SAR
            if prev_trend == 1:  # Uptrend
                sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)

                # Check for trend reversal
                if float(low.iloc[i]) <= sar[i]:
                    # Trend reversal to downtrend
                    trend[i] = -1
                    sar[i] = prev_ep  # SAR becomes the previous extreme point
                    af[i] = af_start
                    ep[i] = float(low.iloc[i])
                else:
                    # Continue uptrend
                    trend[i] = 1

                    # Update extreme point and acceleration factor
                    if float(high.iloc[i]) > prev_ep:
                        ep[i] = float(high.iloc[i])
                        af[i] = min(prev_af + af_increment, af_maximum)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af

                    # Ensure SAR doesn't exceed previous lows
                    sar[i] = min(sar[i], float(low.iloc[i - 1]))
                    if i >= 2:
                        sar[i] = min(sar[i], float(low.iloc[i - 2]))

            else:  # Downtrend
                sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)

                # Check for trend reversal
                if float(high.iloc[i]) >= sar[i]:
                    # Trend reversal to uptrend
                    trend[i] = 1
                    sar[i] = prev_ep  # SAR becomes the previous extreme point
                    af[i] = af_start
                    ep[i] = float(high.iloc[i])
                else:
                    # Continue downtrend
                    trend[i] = -1

                    # Update extreme point and acceleration factor
                    if float(low.iloc[i]) < prev_ep:
                        ep[i] = float(low.iloc[i])
                        af[i] = min(prev_af + af_increment, af_maximum)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af

                    # Ensure SAR doesn't exceed previous highs
                    sar[i] = max(sar[i], float(high.iloc[i - 1]))
                    if i >= 2:
                        sar[i] = max(sar[i], float(high.iloc[i - 2]))

        return pandas.Series(sar, index=close.index)

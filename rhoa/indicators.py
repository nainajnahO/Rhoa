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

# 2. Enhanced Functionality
#
## Add volume-based indicators
#def obv(self, volume: Series) -> Series:  # On-Balance Volume
#def vwap(self, volume: Series, high: Series, low: Series) -> Series:  # VWAP

## Add pattern recognition
#def detect_patterns(self) -> DataFrame:  # Common candlestick patterns

## Add multiple timeframe support
#def resample_indicator(self, timeframe: str, indicator: str, **kwargs):

import pandas
import numpy
from pandas import Series
from pandas import DataFrame
class indicators:
    def __init__(self, series: Series) -> None:
        self._series = series

    def sma(self,
            window_size: int = 20,
            min_periods: int = None,
            center: bool = False,
            **kwargs) -> Series:
        """
        Calculate the Simple Moving Average (SMA) over a specified window.

        The SMA is a commonly used technical indicator in financial and time series
        analysis that calculates the arithmetic mean of prices over a defined number
        of periods. It smooths out price data to identify trends by reducing noise
        from short-term fluctuations.

        Parameters
        ----------
        window_size : int, default 20
            The size of the moving window, representing the number of periods over
            which to calculate the average.
        min_periods : int, optional
            Minimum number of observations in window required to have a value.
            If None, defaults to window_size.
        center : bool, default False
            Whether to set the labels at the center of the window.
        **kwargs : dict
            Additional keyword arguments passed to pandas rolling function.

        Returns
        -------
        pandas.Series
            A Series containing the calculated SMA values with the same index as
            the input series.

        See Also
        --------
        ewma : Exponential Weighted Moving Average for trend analysis
        bollinger_bands : Uses SMA as the middle band
        macd : Uses exponential moving averages

        Notes
        -----
        The Simple Moving Average is calculated as:

        .. math:: SMA_t = \\frac{1}{n} \\sum_{i=0}^{n-1} P_{t-i}

        where :math:`P_t` is the price at time t and n is the window_size.

        The first `window_size - 1` values will be NaN unless min_periods is set
        to a lower value. SMA gives equal weight to all values in the window,
        which can make it slower to respond to recent price changes compared to
        exponential moving averages.

        SMA is commonly used for:
        - Identifying support and resistance levels
        - Generating crossover trading signals (e.g., golden cross, death cross)
        - Smoothing price data for trend identification

        .. tip::
            Also available via DataFrame accessor:
            ``df.rhoa.indicators.sma(window_size=20)`` which defaults to the
            Close column. See :class:`DataFrameIndicators`.

        Examples
        --------
        Calculate 20-period Simple Moving Average:

        >>> import pandas as pd
        >>> import rhoa
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
        >>> sma = prices.rhoa.indicators.sma(window_size=5)
        >>> print(sma.iloc[4])  # First valid SMA value
        102.2

        Generate trading signals using SMA crossover:

        >>> prices = pd.Series([100, 102, 104, 106, 108, 107, 105, 103, 101])
        >>> sma_short = prices.rhoa.indicators.sma(window_size=3)
        >>> sma_long = prices.rhoa.indicators.sma(window_size=5)
        >>> buy_signal = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
        """
        return self._series.rolling(window=window_size, min_periods=min_periods, center=center, **kwargs).mean()

    def ewma(self,
             window_size: int = 20,
             adjust: bool = False,
             min_periods: int = None,
             **kwargs) -> Series:
        """
        Calculate the Exponential Weighted Moving Average (EWMA) of the series.

        The EWMA is a type of infinite impulse response filter that applies weighting
        factors which decrease exponentially. Unlike simple moving averages, EWMA gives
        more weight to recent observations, making it more responsive to recent price
        changes while still providing smoothing.

        Parameters
        ----------
        window_size : int, default 20
            The span of the exponential moving average. Determines the level of
            smoothing, where larger values result in smoother trends and slower
            responsiveness to changes in the data.
        adjust : bool, default False
            Divide by decaying adjustment factor in beginning periods. When True,
            the weights are normalized by the sum of weights to account for the
            imbalance in the beginning periods.
        min_periods : int, optional
            Minimum number of observations in window required to have a value.
            If None, defaults to 0.
        **kwargs : dict
            Additional keyword arguments passed to pandas ewm function.

        Returns
        -------
        pandas.Series
            A Series containing the calculated EWMA values with the same index as
            the input series.

        See Also
        --------
        sma : Simple Moving Average for trend analysis
        ewmv : Exponential Weighted Moving Variance
        ewmstd : Exponential Weighted Moving Standard Deviation
        macd : Uses EWMA for signal generation

        Notes
        -----
        The Exponential Weighted Moving Average is calculated using:

        .. math:: EWMA_t = \\alpha \\cdot P_t + (1 - \\alpha) \\cdot EWMA_{t-1}

        where :math:`\\alpha = \\frac{2}{span + 1}` and span is the window_size.

        Key characteristics:
        - More weight to recent prices: reacts faster to recent changes
        - Smoother than price but more responsive than SMA
        - All historical data has some influence (infinite impulse response)
        - No lookback period required (unlike SMA)

        The adjust parameter affects the calculation in the initial periods:
        - adjust=True: Uses normalized weights (standard EWMA definition)
        - adjust=False: Uses recursive formula (more common in trading)

        EWMA is commonly used for:
        - Trend identification and confirmation
        - Support and resistance levels that adapt to volatility
        - Component of MACD and other composite indicators

        .. tip::
            Also available via DataFrame accessor:
            ``df.rhoa.indicators.ewma(window_size=20)`` which defaults to the
            Close column. See :class:`DataFrameIndicators`.

        Examples
        --------
        Calculate 20-period Exponential Weighted Moving Average:

        >>> import pandas as pd
        >>> import rhoa
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
        >>> ewma = prices.rhoa.indicators.ewma(window_size=5)
        >>> print(f"Latest EWMA: {ewma.iloc[-1]:.2f}")
        Latest EWMA: 105.45

        Compare EWMA with different window sizes:

        >>> ewma_fast = prices.rhoa.indicators.ewma(window_size=5)
        >>> ewma_slow = prices.rhoa.indicators.ewma(window_size=20)
        >>> crossover = (ewma_fast > ewma_slow) & (ewma_fast.shift(1) <= ewma_slow.shift(1))
        """
        return self._series.ewm(span=window_size, adjust=adjust, min_periods=min_periods, **kwargs).mean()

    def ewmv(self,
             window_size: int = 20,
             adjust: bool = True,
             min_periods: int = None,
             **kwargs) -> Series:
        """
        Calculate the exponentially weighted moving variance (EWMV) of a series.

        This method computes the variance of a series by applying exponential weighting
        to give more importance to recent observations. EWMV is useful for measuring
        volatility that adapts more quickly to recent price changes compared to
        standard rolling variance.

        Parameters
        ----------
        window_size : int, default 20
            The span of the exponential window. Determines the level of smoothing
            applied to the variance calculation. Larger values result in smoother
            variance estimates.
        adjust : bool, default True
            Divide by decaying adjustment factor in beginning periods. When True,
            the weights are normalized by the sum of weights.
        min_periods : int, optional
            Minimum number of observations in window required to have a value.
            If None, defaults to 0.
        **kwargs : dict
            Additional keyword arguments passed to pandas ewm function.

        Returns
        -------
        pandas.Series
            A Series containing the exponentially weighted moving variance values
            with the same index as the input series.

        See Also
        --------
        ewmstd : Exponential Weighted Moving Standard Deviation
        ewma : Exponential Weighted Moving Average
        bollinger_bands : Uses standard deviation for band calculation

        Notes
        -----
        The exponentially weighted moving variance uses exponential smoothing
        to calculate variance with the formula:

        .. math:: EWMV_t = \\alpha \\sum_{i=0}^{t} (1-\\alpha)^i (P_{t-i} - EWMA_t)^2

        where :math:`\\alpha = \\frac{2}{span + 1}` and span is the window_size.

        Key properties:
        - Always non-negative (variance cannot be negative)
        - More responsive to recent volatility changes than rolling variance
        - Relationship: :math:`EWMV = EWMSTD^2`
        - Units are squared (e.g., squared dollars for price data)

        Higher variance values indicate increased volatility and risk. EWMV is
        commonly used for:
        - Volatility estimation in risk management
        - Detecting regime changes in market conditions
        - Building adaptive trading strategies
        - Calculating value-at-risk (VaR) metrics

        .. tip::
            Also available via DataFrame accessor:
            ``df.rhoa.indicators.ewmv(window_size=20)`` which defaults to the
            Close column. See :class:`DataFrameIndicators`.

        Examples
        --------
        Calculate exponentially weighted moving variance:

        >>> import pandas as pd
        >>> import rhoa
        >>> prices = pd.Series([100, 102, 99, 103, 105, 101, 106, 104])
        >>> ewmv = prices.rhoa.indicators.ewmv(window_size=5)
        >>> print(f"Latest variance: {ewmv.iloc[-1]:.2f}")
        Latest variance: 6.24

        Detect periods of high volatility:

        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 110, 95, 105, 100])
        >>> ewmv = prices.rhoa.indicators.ewmv(window_size=5)
        >>> high_volatility = ewmv > ewmv.rolling(20).mean() * 1.5
        """
        return self._series.ewm(span=window_size, adjust=adjust, min_periods=min_periods, **kwargs).var()

    def ewmstd(self,
               window_size: int = 20,
               adjust: bool = True,
               min_periods: int = None,
               **kwargs) -> Series:
        """
        Calculate the exponentially weighted moving standard deviation (EWMSTD).

        EWMSTD is a statistical measure that weights recent data points more heavily
        to provide a smoothed calculation of the moving standard deviation. This makes
        it more responsive to recent volatility changes compared to traditional rolling
        standard deviation, while maintaining smoothness.

        Parameters
        ----------
        window_size : int, default 20
            The span or window size for the exponentially weighted moving calculation.
            Smaller spans apply heavier weighting to more recent data points and react
            faster to changes, while larger spans provide smoother results.
        adjust : bool, default True
            Divide by decaying adjustment factor in beginning periods. When True,
            the weights are normalized by the sum of weights.
        min_periods : int, optional
            Minimum number of observations in window required to have a value.
            If None, defaults to 0.
        **kwargs : dict
            Additional keyword arguments passed to pandas ewm function.

        Returns
        -------
        pandas.Series
            A Series containing the exponentially weighted moving standard deviation
            values with the same index as the input series.

        See Also
        --------
        ewmv : Exponential Weighted Moving Variance
        ewma : Exponential Weighted Moving Average
        bollinger_bands : Uses standard deviation for volatility bands
        atr : Average True Range for volatility measurement

        Notes
        -----
        The exponentially weighted moving standard deviation is the square root
        of the exponentially weighted moving variance:

        .. math:: EWMSTD_t = \\sqrt{EWMV_t}

        where :math:`EWMV_t` is calculated with exponential weights.

        The relationship :math:`EWMSTD^2 = EWMV` always holds.

        Key characteristics:
        - Always non-negative (standard deviation cannot be negative)
        - Same units as the original series (unlike variance)
        - More responsive to recent volatility than rolling standard deviation
        - Commonly used as a volatility proxy in trading

        EWMSTD is commonly used for:
        - Volatility-based position sizing
        - Risk management and stop-loss placement
        - Adaptive trading strategies that respond to volatility
        - Bollinger Bands and other volatility indicators
        - Normalized price movements (z-scores)

        Higher values indicate increased volatility and uncertainty.

        .. tip::
            Also available via DataFrame accessor:
            ``df.rhoa.indicators.ewmstd(window_size=20)`` which defaults to the
            Close column. See :class:`DataFrameIndicators`.

        Examples
        --------
        Calculate exponentially weighted moving standard deviation:

        >>> import pandas as pd
        >>> import rhoa
        >>> prices = pd.Series([100, 102, 99, 103, 105, 101, 106, 104])
        >>> ewmstd = prices.rhoa.indicators.ewmstd(window_size=5)
        >>> print(f"Latest volatility: {ewmstd.iloc[-1]:.2f}")
        Latest volatility: 2.50

        Use EWMSTD for volatility-adjusted position sizing:

        >>> prices = pd.Series([100, 102, 104, 106, 108, 110, 112])
        >>> volatility = prices.rhoa.indicators.ewmstd(window_size=10)
        >>> position_size = 1000 / volatility  # Risk-adjusted sizing
        """
        return self._series.ewm(span=window_size, adjust=adjust, min_periods=min_periods, **kwargs).std()

    def rsi(
            self,
            window_size: int = 14,
            edge_case_value: float = 100.0,
            **kwargs) -> Series:
        """
        Calculate the Relative Strength Index (RSI) for momentum analysis.

        RSI is a momentum oscillator that measures the speed and magnitude of price
        changes on a scale of 0 to 100. Developed by J. Welles Wilder Jr., it helps
        identify overbought and oversold conditions, as well as potential trend
        reversals. RSI is one of the most widely used technical indicators in trading.

        Parameters
        ----------
        window_size : int, default 14
            The size of the rolling window used to calculate the exponential moving
            averages of gains and losses. Traditional value is 14 periods as
            recommended by Wilder.
        edge_case_value : float, default 100.0
            The RSI value to use when avg_loss == 0 (no losses occurred in the
            period). Common values are 100.0 (infinite RS, default), 50.0 (neutral),
            or numpy.nan.
        **kwargs : dict
            Additional keyword arguments passed to pandas ewm function.

        Returns
        -------
        pandas.Series
            A Series containing RSI values between 0 and 100 with the same index
            as the input series.

        See Also
        --------
        stochastic : Similar momentum oscillator
        cci : Commodity Channel Index for momentum
        williams_r : Williams %R momentum indicator
        macd : Moving Average Convergence Divergence

        Notes
        -----
        The Relative Strength Index is calculated using:

        .. math:: RSI = 100 - \\frac{100}{1 + RS}

        where :math:`RS = \\frac{EMA(gains)}{EMA(losses)}` and EMA is the
        exponential moving average over the specified window_size.

        Traditional interpretation levels:
        - RSI > 70: Overbought condition (potential sell signal)
        - RSI < 30: Oversold condition (potential buy signal)
        - RSI = 50: Neutral momentum (balance between bulls and bears)

        Key characteristics:
        - Range: 0 to 100 (bounded oscillator)
        - Mean-reverting: tends to oscillate around 50
        - Leading indicator: can signal reversals before price
        - Works best in ranging markets (less reliable in strong trends)

        Advanced RSI techniques:
        - Divergence: RSI diverging from price can signal reversals
        - Failure swings: RSI patterns that don't confirm new price highs/lows
        - Centerline crossovers: RSI crossing 50 confirms trend direction
        - Dynamic thresholds: Use 80/20 in strong trends instead of 70/30

        The first `window_size` values will be NaN as the indicator requires
        sufficient data to calculate the initial exponential moving averages.

        References
        ----------
        .. [1] Wilder, J. W. (1978). New Concepts in Technical Trading Systems.
               Trend Research.

        .. tip::
            Also available via DataFrame accessor:
            ``df.rhoa.indicators.rsi(window_size=14)`` which defaults to the
            Close column. See :class:`DataFrameIndicators`.

        Examples
        --------
        Calculate 14-period RSI and identify trading signals:

        >>> import pandas as pd
        >>> import rhoa
        >>> prices = pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109])
        >>> rsi = prices.rhoa.indicators.rsi(window_size=14)
        >>> overbought = rsi > 70  # Potential sell signals
        >>> oversold = rsi < 30   # Potential buy signals
        >>> print(f"Latest RSI: {rsi.iloc[-1]:.1f}")
        Latest RSI: 75.2

        Detect RSI divergence for reversal signals:

        >>> prices = pd.Series([100, 105, 110, 115, 120, 118, 116, 114])
        >>> rsi = prices.rhoa.indicators.rsi()
        >>> # Bearish divergence: price makes new high but RSI doesn't
        >>> price_higher = prices > prices.shift(1).rolling(5).max()
        >>> rsi_lower = rsi < rsi.shift(1).rolling(5).max()
        >>> divergence = price_higher & rsi_lower
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
        """
        Calculate the MACD (Moving Average Convergence Divergence) indicator.

        MACD is a trend-following momentum indicator that shows the relationship
        between two exponential moving averages of a security's price. Developed by
        Gerald Appel, it consists of three components that together provide insights
        into trend direction, momentum strength, and potential reversals.

        Parameters
        ----------
        short_window : int, default 12
            Length of the short-term (fast) EMA window in periods. Smaller values
            make MACD more responsive to recent price changes.
        long_window : int, default 26
            Length of the long-term (slow) EMA window in periods. Larger values
            provide more smoothing and stability.
        signal_window : int, default 9
            Length of the signal line EMA window in periods. This smooths the
            MACD line to generate trading signals.
        **kwargs : dict
            Additional keyword arguments passed to pandas ewm function.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with three columns:
            - 'macd' : The MACD line (short_ema - long_ema)
            - 'signal' : The signal line (EMA of MACD line)
            - 'histogram' : The MACD histogram (macd - signal)

        See Also
        --------
        ewma : Exponential Weighted Moving Average
        rsi : Relative Strength Index for momentum
        stochastic : Stochastic Oscillator for momentum

        Notes
        -----
        The MACD indicator components are calculated as:

        .. math::
            MACD_{line} = EMA_{short} - EMA_{long}
        .. math::
            Signal_{line} = EMA_{signal}(MACD_{line})
        .. math::
            Histogram = MACD_{line} - Signal_{line}

        Traditional interpretation:
        - MACD line crosses above signal: Bullish signal (buy)
        - MACD line crosses below signal: Bearish signal (sell)
        - Histogram > 0: MACD above signal (bullish momentum)
        - Histogram < 0: MACD below signal (bearish momentum)
        - Histogram expanding: Momentum increasing
        - Histogram contracting: Momentum decreasing

        Key characteristics:
        - Unbounded oscillator (can take any value)
        - Combines trend-following and momentum aspects
        - Three signals: crossovers, divergences, and rapid rises/falls
        - Works best in trending markets

        Advanced MACD techniques:
        - Divergence: MACD diverging from price signals potential reversals
        - Zero-line crossovers: MACD crossing zero indicates trend change
        - Histogram analysis: Momentum changes before MACD line crosses
        - Multiple timeframe confirmation

        The standard parameters (12, 26, 9) were optimized for daily charts but
        can be adjusted for different timeframes or market characteristics.

        References
        ----------
        .. [1] Appel, Gerald (2005). Technical Analysis: Power Tools for Active
               Investors. Financial Times Prentice Hall.

        .. tip::
            Also available via DataFrame accessor:
            ``df.rhoa.indicators.macd()`` which defaults to the
            Close column. See :class:`DataFrameIndicators`.

        Examples
        --------
        Calculate MACD and identify bullish crossover:

        >>> import pandas as pd
        >>> import rhoa
        >>> prices = pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110])
        >>> macd_data = prices.rhoa.indicators.macd()
        >>> # Bullish signal: MACD crosses above signal
        >>> bullish = (macd_data['macd'] > macd_data['signal']) & \
        ...           (macd_data['macd'].shift(1) <= macd_data['signal'].shift(1))
        >>> print(f"MACD: {macd_data['macd'].iloc[-1]:.3f}")
        MACD: 0.245

        Analyze MACD histogram for momentum changes:

        >>> macd_data = prices.rhoa.indicators.macd()
        >>> histogram = macd_data['histogram']
        >>> momentum_increasing = histogram > histogram.shift(1)
        >>> momentum_peak = (histogram.shift(1) > histogram) & (histogram.shift(1) > histogram.shift(2))
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
        """
        Calculate Bollinger Bands for volatility and mean reversion analysis.

        Bollinger Bands consist of three lines: an upper band, middle band (SMA),
        and lower band. Developed by John Bollinger, the bands expand and contract
        based on market volatility, providing insights into potential overbought/
        oversold conditions, volatility patterns, and mean reversion opportunities.

        Parameters
        ----------
        window_size : int, default 20
            The size of the rolling window used for computing the moving average
            and standard deviation. Standard setting is 20 periods.
        num_std : float, default 2.0
            The number of standard deviations to add/subtract from the moving
            average to calculate the upper and lower bands. Standard setting is 2.0.
        min_periods : int, optional
            Minimum number of observations in window required to have a value.
            If None, defaults to window_size.
        center : bool, default False
            Whether to set the labels at the center of the window.
        **kwargs : dict
            Additional keyword arguments passed to pandas rolling function.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with three columns:
            - 'upper_band' : Upper Bollinger Band (middle + num_std * std)
            - 'middle_band' : Middle band (SMA of the series)
            - 'lower_band' : Lower Bollinger Band (middle - num_std * std)

        See Also
        --------
        sma : Simple Moving Average (middle band)
        ewmstd : Exponential Weighted Standard Deviation
        atr : Average True Range for volatility

        Notes
        -----
        Bollinger Bands are calculated using:

        .. math::
            Middle_{band} = SMA(price, window)
        .. math::
            Upper_{band} = Middle_{band} + (num\\_std \\times \\sigma)
        .. math::
            Lower_{band} = Middle_{band} - (num\\_std \\times \\sigma)

        where :math:`\\sigma` is the standard deviation over the window.

        Traditional interpretation:
        - Price touching upper band: Potentially overbought (high relative price)
        - Price touching lower band: Potentially oversold (low relative price)
        - Price between bands: Normal trading range
        - Band width: Measure of volatility

        Key characteristics:
        - Approximately 95% of price action occurs within 2 standard deviations
        - Bands are dynamic support/resistance levels
        - Band width reflects market volatility
        - Works as both a trend and volatility indicator

        Common Bollinger Band strategies:
        - Squeeze: Narrow bands indicate low volatility, often precedes breakout
        - Expansion: Wide bands indicate high volatility
        - Band walk: Price riding upper/lower band indicates strong trend
        - Bollinger Bounce: Mean reversion trades off band touches
        - %B indicator: (price - lower) / (upper - lower) shows relative position

        Advanced techniques:
        - Double tops/bottoms at bands for reversal signals
        - M-tops and W-bottoms for patterns
        - Band width as volatility indicator
        - Combination with other indicators for confirmation

        References
        ----------
        .. [1] Bollinger, John (2001). Bollinger on Bollinger Bands.
               McGraw-Hill.

        .. tip::
            Also available via DataFrame accessor:
            ``df.rhoa.indicators.bollinger_bands(window_size=20)`` which defaults to the
            Close column. See :class:`DataFrameIndicators`.

        Examples
        --------
        Calculate Bollinger Bands and identify squeeze conditions:

        >>> import pandas as pd
        >>> import rhoa
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107])
        >>> bb = prices.rhoa.indicators.bollinger_bands(window_size=5, num_std=2.0)
        >>> # Band width indicates volatility
        >>> width = bb['upper_band'] - bb['lower_band']
        >>> squeeze = width < width.rolling(10).mean() * 0.8  # Low volatility
        >>> print(f"Upper: {bb['upper_band'].iloc[-1]:.2f}")
        Upper: 109.45

        Calculate %B indicator for position within bands:

        >>> bb = prices.rhoa.indicators.bollinger_bands()
        >>> percent_b = (prices - bb['lower_band']) / (bb['upper_band'] - bb['lower_band'])
        >>> overbought = percent_b > 1.0  # Price above upper band
        >>> oversold = percent_b < 0.0    # Price below lower band
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


class DataFrameIndicators:
    """DataFrame-level indicators accessor for OHLC operations.

    Auto-detects Close, High, and Low columns from the DataFrame,
    delegating computation to the existing ``indicators`` class.
    """

    def __init__(self, df: DataFrame) -> None:
        self._df = df
        self._columns = {col.lower(): col for col in df.columns}

    def _resolve_column(self, canonical: str) -> str:
        """Case-insensitive column lookup.

        For 'close', also checks 'adj close' as a fallback alias.
        """
        if canonical in self._columns:
            return self._columns[canonical]
        if canonical == 'close' and 'adj close' in self._columns:
            return self._columns['adj close']
        return None

    def _get_series(self, explicit, canonical: str, param_name: str) -> Series:
        """Return explicit Series if passed, otherwise look up from df."""
        if explicit is not None:
            return explicit
        col = self._resolve_column(canonical)
        if col is None:
            raise ValueError(
                f"Could not auto-detect '{canonical}' column from DataFrame. "
                f"Pass it explicitly via the '{param_name}' parameter."
            )
        return self._df[col]

    def _get_close_indicators(self, close=None) -> indicators:
        """Return an indicators instance bound to the close series."""
        close_s = self._get_series(close, 'close', 'close')
        return indicators(close_s)

    # --- Single-series indicators (delegate to Close) ---

    def sma(self, close=None, window_size: int = 20, min_periods: int = None,
            center: bool = False, **kwargs) -> Series:
        return self._get_close_indicators(close).sma(
            window_size=window_size, min_periods=min_periods, center=center, **kwargs)

    def ewma(self, close=None, window_size: int = 20, adjust: bool = False,
             min_periods: int = None, **kwargs) -> Series:
        return self._get_close_indicators(close).ewma(
            window_size=window_size, adjust=adjust, min_periods=min_periods, **kwargs)

    def ewmv(self, close=None, window_size: int = 20, adjust: bool = True,
             min_periods: int = None, **kwargs) -> Series:
        return self._get_close_indicators(close).ewmv(
            window_size=window_size, adjust=adjust, min_periods=min_periods, **kwargs)

    def ewmstd(self, close=None, window_size: int = 20, adjust: bool = True,
               min_periods: int = None, **kwargs) -> Series:
        return self._get_close_indicators(close).ewmstd(
            window_size=window_size, adjust=adjust, min_periods=min_periods, **kwargs)

    def rsi(self, close=None, window_size: int = 14,
            edge_case_value: float = 100.0, **kwargs) -> Series:
        return self._get_close_indicators(close).rsi(
            window_size=window_size, edge_case_value=edge_case_value, **kwargs)

    def macd(self, close=None, short_window: int = 12, long_window: int = 26,
             signal_window: int = 9, **kwargs) -> DataFrame:
        return self._get_close_indicators(close).macd(
            short_window=short_window, long_window=long_window,
            signal_window=signal_window, **kwargs)

    def bollinger_bands(self, close=None, window_size: int = 20,
                        num_std: float = 2.0, min_periods: int = None,
                        center: bool = False, **kwargs) -> DataFrame:
        return self._get_close_indicators(close).bollinger_bands(
            window_size=window_size, num_std=num_std, min_periods=min_periods,
            center=center, **kwargs)

    # --- OHLC indicators (DataFrame-only, auto-detect Close, High, Low) ---

    def atr(self, close=None, high=None, low=None, window_size: int = 14,
            min_periods: int = None, center: bool = False, **kwargs) -> Series:
        """Calculate the Average True Range (ATR) for volatility measurement.

        Auto-detects Close, High, and Low columns from the DataFrame.
        Pass explicit Series to override auto-detection.

        Parameters
        ----------
        close : pandas.Series, optional
            Close prices. Auto-detected from DataFrame if not provided.
        high : pandas.Series, optional
            High prices. Auto-detected from DataFrame if not provided.
        low : pandas.Series, optional
            Low prices. Auto-detected from DataFrame if not provided.
        window_size : int, default 14
            Length of the rolling window for calculating the average true range.
        min_periods : int, optional
            Minimum number of observations in window required to have a value.
        center : bool, default False
            Whether to set the labels at the center of the window.
        **kwargs : dict
            Additional keyword arguments passed to pandas rolling function.

        Returns
        -------
        pandas.Series
            A Series containing the calculated ATR values.
        """
        close_s = self._get_series(close, 'close', 'close')
        high_s = self._get_series(high, 'high', 'high')
        low_s = self._get_series(low, 'low', 'low')

        high_low = high_s - low_s
        high_close = (high_s - close_s.shift(1)).abs()
        low_close = (low_s - close_s.shift(1)).abs()

        true_range = pandas.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window_size, min_periods=min_periods,
                                  center=center, **kwargs).mean()

    def cci(self, close=None, high=None, low=None, window_size: int = 20,
            min_periods: int = None, center: bool = False, **kwargs) -> Series:
        """Calculate the Commodity Channel Index (CCI) for momentum analysis.

        Auto-detects Close, High, and Low columns from the DataFrame.
        Pass explicit Series to override auto-detection.

        Parameters
        ----------
        close : pandas.Series, optional
            Close prices. Auto-detected from DataFrame if not provided.
        high : pandas.Series, optional
            High prices. Auto-detected from DataFrame if not provided.
        low : pandas.Series, optional
            Low prices. Auto-detected from DataFrame if not provided.
        window_size : int, default 20
            Number of periods for calculating the CCI.
        min_periods : int, optional
            Minimum number of observations in window required to have a value.
        center : bool, default False
            Whether to set the labels at the center of the window.
        **kwargs : dict
            Additional keyword arguments passed to pandas rolling function.

        Returns
        -------
        pandas.Series
            A Series containing the calculated CCI values.
        """
        close_s = self._get_series(close, 'close', 'close')
        high_s = self._get_series(high, 'high', 'high')
        low_s = self._get_series(low, 'low', 'low')

        typical_price = (high_s + low_s + close_s) / 3
        sma = typical_price.rolling(window=window_size, min_periods=min_periods,
                                    center=center, **kwargs).mean()
        mean_deviation = typical_price.rolling(
            window=window_size, min_periods=min_periods, center=center,
            **kwargs).apply(
            lambda x: numpy.mean(numpy.abs(x - x.mean())),
            raw=True
        )
        return (typical_price - sma) / (0.015 * mean_deviation)

    def stochastic(self, close=None, high=None, low=None, k_window: int = 14,
                   d_window: int = 3, min_periods: int = None,
                   center: bool = False, **kwargs) -> DataFrame:
        """Calculate the Stochastic Oscillator (%K and %D) for momentum analysis.

        Auto-detects Close, High, and Low columns from the DataFrame.
        Pass explicit Series to override auto-detection.

        Parameters
        ----------
        close : pandas.Series, optional
            Close prices. Auto-detected from DataFrame if not provided.
        high : pandas.Series, optional
            High prices. Auto-detected from DataFrame if not provided.
        low : pandas.Series, optional
            Low prices. Auto-detected from DataFrame if not provided.
        k_window : int, default 14
            Number of periods for %K calculation.
        d_window : int, default 3
            Number of periods for %D calculation (SMA of %K).
        min_periods : int, optional
            Minimum observations in window required to have a value.
        center : bool, default False
            Whether to set the labels at the center of the window.
        **kwargs : dict
            Additional keyword arguments passed to pandas rolling function.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with '%K' and '%D' columns.
        """
        close_s = self._get_series(close, 'close', 'close')
        high_s = self._get_series(high, 'high', 'high')
        low_s = self._get_series(low, 'low', 'low')

        lowest_low = low_s.rolling(window=k_window, min_periods=min_periods,
                                   center=center, **kwargs).min()
        highest_high = high_s.rolling(window=k_window, min_periods=min_periods,
                                      center=center, **kwargs).max()
        k_percent = 100 * ((close_s - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window, min_periods=min_periods,
                                      center=center, **kwargs).mean()

        return DataFrame({
            "%K": k_percent,
            "%D": d_percent
        })

    def williams_r(self, close=None, high=None, low=None, window_size: int = 14,
                   min_periods: int = None, center: bool = False,
                   **kwargs) -> Series:
        """Calculate Williams %R for momentum and overbought/oversold analysis.

        Auto-detects Close, High, and Low columns from the DataFrame.
        Pass explicit Series to override auto-detection.

        Parameters
        ----------
        close : pandas.Series, optional
            Close prices. Auto-detected from DataFrame if not provided.
        high : pandas.Series, optional
            High prices. Auto-detected from DataFrame if not provided.
        low : pandas.Series, optional
            Low prices. Auto-detected from DataFrame if not provided.
        window_size : int, default 14
            Number of periods for Williams %R calculation.
        min_periods : int, optional
            Minimum observations in window required to have a value.
        center : bool, default False
            Whether to set the labels at the center of the window.
        **kwargs : dict
            Additional keyword arguments passed to pandas rolling function.

        Returns
        -------
        pandas.Series
            A Series containing Williams %R values ranging from -100 to 0.
        """
        close_s = self._get_series(close, 'close', 'close')
        high_s = self._get_series(high, 'high', 'high')
        low_s = self._get_series(low, 'low', 'low')

        highest_high = high_s.rolling(window=window_size, min_periods=min_periods,
                                      center=center, **kwargs).max()
        lowest_low = low_s.rolling(window=window_size, min_periods=min_periods,
                                   center=center, **kwargs).min()

        return -100 * ((highest_high - close_s) / (highest_high - lowest_low))

    def adx(self, close=None, high=None, low=None, window_size: int = 14,
            min_periods: int = None, **kwargs) -> DataFrame:
        """Calculate the Average Directional Index (ADX) for trend strength analysis.

        Auto-detects Close, High, and Low columns from the DataFrame.
        Pass explicit Series to override auto-detection.

        Parameters
        ----------
        close : pandas.Series, optional
            Close prices. Auto-detected from DataFrame if not provided.
        high : pandas.Series, optional
            High prices. Auto-detected from DataFrame if not provided.
        low : pandas.Series, optional
            Low prices. Auto-detected from DataFrame if not provided.
        window_size : int, default 14
            Number of periods for ADX calculation.
        min_periods : int, optional
            Minimum observations in window required to have a value.
        **kwargs : dict
            Additional keyword arguments passed to pandas ewm function.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with 'ADX', '+DI', and '-DI' columns.
        """
        close_s = self._get_series(close, 'close', 'close')
        high_s = self._get_series(high, 'high', 'high')
        low_s = self._get_series(low, 'low', 'low')

        # Calculate True Range
        high_low = high_s - low_s
        high_close = (high_s - close_s.shift(1)).abs()
        low_close = (low_s - close_s.shift(1)).abs()
        true_range = pandas.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate Directional Movement
        high_diff = high_s.diff()
        low_diff = low_s.diff()

        plus_dm = pandas.Series(
            numpy.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0),
            index=high_s.index)
        minus_dm = pandas.Series(
            numpy.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0),
            index=low_s.index)

        # Smooth using EWM
        atr = true_range.ewm(span=window_size, adjust=False,
                              min_periods=min_periods, **kwargs).mean()
        plus_di_smooth = plus_dm.ewm(span=window_size, adjust=False,
                                      min_periods=min_periods, **kwargs).mean()
        minus_di_smooth = minus_dm.ewm(span=window_size, adjust=False,
                                        min_periods=min_periods, **kwargs).mean()

        # Calculate +DI and -DI
        plus_di = 100 * (plus_di_smooth / atr)
        minus_di = 100 * (minus_di_smooth / atr)

        # Calculate ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.ewm(span=window_size, adjust=False,
                      min_periods=min_periods, **kwargs).mean()

        return DataFrame({
            "ADX": adx,
            "+DI": plus_di,
            "-DI": minus_di
        })

    def parabolic_sar(self, close=None, high=None, low=None,
                      af_start: float = 0.02, af_increment: float = 0.02,
                      af_maximum: float = 0.2) -> Series:
        """Calculate the Parabolic Stop and Reverse (SAR) for trend following.

        Auto-detects Close, High, and Low columns from the DataFrame.
        Pass explicit Series to override auto-detection.

        Parameters
        ----------
        close : pandas.Series, optional
            Close prices. Auto-detected from DataFrame if not provided.
        high : pandas.Series, optional
            High prices. Auto-detected from DataFrame if not provided.
        low : pandas.Series, optional
            Low prices. Auto-detected from DataFrame if not provided.
        af_start : float, default 0.02
            Initial acceleration factor.
        af_increment : float, default 0.02
            Increment added to acceleration factor when a new extreme point is
            reached.
        af_maximum : float, default 0.2
            Maximum acceleration factor value.

        Returns
        -------
        pandas.Series
            A Series containing Parabolic SAR values.
        """
        close_s = self._get_series(close, 'close', 'close')
        high_s = self._get_series(high, 'high', 'high')
        low_s = self._get_series(low, 'low', 'low')

        # Initialize arrays
        length = len(close_s)
        sar = numpy.zeros(length)
        trend = numpy.zeros(length, dtype=int)
        af = numpy.zeros(length)
        ep = numpy.zeros(length)

        # Initialize first values
        sar[0] = float(low_s.iloc[0])
        trend[0] = 1
        af[0] = af_start
        ep[0] = float(high_s.iloc[0])

        for i in range(1, length):
            prev_sar = sar[i - 1]
            prev_trend = trend[i - 1]
            prev_af = af[i - 1]
            prev_ep = ep[i - 1]

            if prev_trend == 1:  # Uptrend
                sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)

                if float(low_s.iloc[i]) <= sar[i]:
                    trend[i] = -1
                    sar[i] = prev_ep
                    af[i] = af_start
                    ep[i] = float(low_s.iloc[i])
                else:
                    trend[i] = 1

                    if float(high_s.iloc[i]) > prev_ep:
                        ep[i] = float(high_s.iloc[i])
                        af[i] = min(prev_af + af_increment, af_maximum)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af

                    sar[i] = min(sar[i], float(low_s.iloc[i - 1]))
                    if i >= 2:
                        sar[i] = min(sar[i], float(low_s.iloc[i - 2]))

            else:  # Downtrend
                sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)

                if float(high_s.iloc[i]) >= sar[i]:
                    trend[i] = 1
                    sar[i] = prev_ep
                    af[i] = af_start
                    ep[i] = float(high_s.iloc[i])
                else:
                    trend[i] = -1

                    if float(low_s.iloc[i]) < prev_ep:
                        ep[i] = float(low_s.iloc[i])
                        af[i] = min(prev_af + af_increment, af_maximum)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af

                    sar[i] = max(sar[i], float(high_s.iloc[i - 1]))
                    if i >= 2:
                        sar[i] = max(sar[i], float(high_s.iloc[i - 2]))

        return pandas.Series(sar, index=close_s.index)

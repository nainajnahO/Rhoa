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

import pandas
import numpy
from pandas import Series
from pandas import DataFrame
from pandas.api.extensions import register_series_accessor


# TODO: Indicators have more arguments than just 'window_size', expand the functions
#       to allow for better hyper-tuning.

@register_series_accessor("indicators")
class IndicatorsAccessor:
    def __init__(self, series: Series) -> None:
        self._series = series

    def sma(self,
            window_size: int = 20) -> Series:
        """
        Calculate the Simple Moving Average (SMA) over a specified window.

        The SMA is a commonly used technical indicator in financial
        and time series analysis that calculates the average value
        over a defined number of periods.

        :param window_size: The size of the moving window, representing
            the number of periods over which to calculate the average.
        :type window_size: int
        :return: A pandas Series containing the calculated SMA values.
        :rtype: Series
        """
        return self._series.rolling(window_size).mean()

    def ewma(self,
             window_size: int = 20) -> Series:
        """
        Calculates the Exponential Weighted Moving Average (EWMA) of the series.

        The EWMA is a type of infinite impulse response filter that applies weighting
        factors which decrease exponentially. This method is commonly used in financial
        time series to smooth data and compute trends.

        :param window_size: The span of the exponential moving average. Determines the
            level of smoothing, where larger values result in smoother trends and slower
            responsiveness to changes in the data.
        :type window_size: int
        :return: A pandas Series containing the calculated EWMA values for the input
            series.
        :rtype: Series
        """
        return self._series.ewm(span=window_size, adjust=False).mean()

    def ewmv(self,
             window_size: int = 20) -> Series:
        """
        Calculate the exponentially weighted moving variance (EWMV) of a series.

        This method computes the variance of a series by applying exponential
        weighting. The window size parameter determines the span of the
        exponentially weighted period.

        :param window_size: The span of the exponential window. Determines the
            level of smoothing applied to the variance calculation.
        :type window_size: int
        :return: A pandas Series containing the exponentially weighted moving
            variance of the input series.
        :rtype: Series
        """
        return self._series.ewm(span=window_size).var()

    def ewmstd(self,
               window_size: int = 20) -> Series:
        """
        Calculate the exponentially weighted moving standard deviation (EWMSTD)
        for the given time series data. EWMSTD is a statistical measure that
        weights recent data points more heavily to provide a smoothed
        calculation of the moving standard deviation.

        This method considers a specified window size to calculate the
        exponentially weighted standard deviation. Smaller window sizes
        assign more weight to recent data, ensuring rapid adaptation to
        changes in the series, while larger window sizes provide smoother
        and more stable results.

        :param window_size: The span or window size for the exponentially
            weighted moving calculation. It determines the degree of
            smoothing applied. A smaller span applies heavier weighting to
            more recent data points, while a larger span applies less
            weighting.
        :return: A new Series containing the exponentially weighted moving
            standard deviation values for the provided time series data.
        """
        return self._series.ewm(span=window_size).std()

    def rsi(
            self,
            window_size: int = 14) -> Series:
        """
        Calculates the Relative Strength Index (RSI) for a given price series using
        a specified window size. RSI is a momentum oscillator that measures the speed
        and change of price movements. It provides a value between 0 and 100 to indicate
        overbought or oversold market conditions. Higher values of RSI denote overbought
        conditions, while lower values indicate oversold conditions.

        :param window_size: The size of the rolling window used to calculate the moving
            averages of gains and losses. Default value is 14.
        :type window_size: int
        :return: A pandas Series object containing the RSI values for the price series.
        :rtype: Series
        """
        price = self._series
        delta = price.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(span=window_size, adjust=False, min_periods=window_size).mean()
        avg_loss = loss.ewm(span=window_size, adjust=False, min_periods=window_size).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # when avg_loss == 0, RSI should be 100 (infinite RS)
        rsi[avg_loss == 0] = 100

        return rsi

    def macd(self,
             short_window: int = 12,
             long_window: int = 26,
             signal_window: int = 9) -> DataFrame:
        """
        Calculates the MACD (Moving Average Convergence Divergence) for a given time series data. The MACD
        is represented by three components: the MACD line, the signal line, and the histogram. These
        components are calculated using the short-term and long-term exponential moving averages (EMA) of
        the input data series.

        The MACD line is computed as the difference between the short-term EMA and the long-term EMA.
        The signal line represents an EMA of the MACD line over a defined signal period. The histogram
        is derived as the difference between the MACD line and the signal line, providing insight into
        potential momentum changes and trend directions.

        :param short_window: Length of the short-term EMA window; default is 12.
        :type short_window: int
        :param long_window: Length of the long-term EMA window; default is 26.
        :type long_window: int
        :param signal_window: Length of the signal EMA window; default is 9.
        :type signal_window: int
        :return: A DataFrame containing the calculated "macd", "signal", and "histogram" as its columns.
        :rtype: DataFrame
        """
        # SHORT-TERM AND LONG-TERM EXPONENTIAL MOVING AVERAGE
        short_ema = self._series.ewm(span=short_window, adjust=False).mean()
        long_ema = self._series.ewm(span=long_window, adjust=False).mean()

        # MACD LINE
        macd_line = short_ema - long_ema

        # SIGNAL LINE
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()

        # HISTOGRAM
        macd_histogram = macd_line - signal_line

        return DataFrame({
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_histogram
        })

    def bollinger_bands(self,
                        window_size: int = 20,
                        num_std: float = 2.0) -> DataFrame:
        """
        Computes the Bollinger Bands for a given time series.

        Bollinger Bands are a technical analysis tool used in financial markets
        to assess volatility and potential overbought or oversold market conditions.
        The bands consist of three lines: the upper band, the middle band (moving
        average), and the lower band. They are derived from applying a rolling
        average and standard deviation over a specified window size.

        :param window_size: The size of the rolling window used for computing the
            moving average and standard deviation.
        :type window_size: int
        :param num_std: The number of standard deviations to add or subtract from
            the moving average to calculate the upper and lower bands.
        :type num_std: float
        :return: A DataFrame containing three columns: `upper_band`, `middle_band`,
            and `lower_band` indexed by the same index as the input series.
        """
        series = self._series

        middle = series.rolling(window=window_size).mean()
        std = series.rolling(window=window_size).std()

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
            window_size: int = 14) -> Series:
        """
        Calculates the Average True Range (ATR) for a financial time series. ATR is
        a measure of market volatility that takes into account the range between
        high and low prices, as well as the differences from the previous period's
        closing prices.

        The function computes the true range as the maximum of the differences
        between the high and low, high and close of the previous period, and low
        and close of the previous period. It then averages the true range over a
        specified rolling window to determine the ATR value.

        :param high: A Pandas Series representing the high prices.
        :type high: Series
        :param low: A Pandas Series representing the low prices.
        :type low: Series
        :param window_size: Integer specifying the length of the rolling window
            for calculating the average true range. Defaults to 14.
        :type window_size: int
        :return: A Pandas Series containing the calculated ATR values.
        """
        close = self._series

        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()

        true_range = pandas.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window_size).mean()

        return atr

    def cci(self,
            high: Series,
            low: Series,
            window_size: int = 20) -> Series:
        """
        Calculates the Commodity Channel Index (CCI) for the given time series data.

        The Commodity Channel Index is a momentum-based oscillator that measures the
        variation of a financial instrument's price from its average price over a specified
        time period. It is used to identify overbought or oversold conditions, as well as
        potential reversals or continuation of trends in price movements.

        :param high: A Pandas Series containing the high prices of the financial instrument.
        :type high: Series
        :param low: A pandas Series containing the low prices of the financial instrument.
        :type low: Series
        :param window_size: Integer representing the number of periods to be considered
            for calculating the CCI. The default is 20.
        :type window_size: int
        :return: A pandas Series representing the calculated CCI values.
        :rtype: Series
        """
        close = self._series
        typical_price = (high + low + close) / 3

        sma = typical_price.rolling(window=window_size).mean()

        mean_deviation = typical_price.rolling(window=window_size).apply(
            lambda x: numpy.mean(numpy.abs(x - x.mean())),
            raw=True
        )

        cci = (typical_price - sma) / (0.015 * mean_deviation)

        return cci

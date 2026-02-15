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

from pandas.api.extensions import register_dataframe_accessor, register_series_accessor

from .plots import PlotsAccessor
from .preprocess import PreprocessAccessor
from .strategy import StrategyAccessor
from .indicators import indicators, DataFrameIndicators


@register_dataframe_accessor("rhoa")
class RhoaDataFrameAccessor:
    def __init__(self, pandas_obj):
        self._df = pandas_obj

    @property
    def plots(self) -> PlotsAccessor:
        return PlotsAccessor(self._df)

    @property
    def preprocess(self) -> PreprocessAccessor:
        return PreprocessAccessor(self._df)

    @property
    def strategy(self) -> StrategyAccessor:
        return StrategyAccessor(self._df)

    @property
    def indicators(self) -> DataFrameIndicators:
        return DataFrameIndicators(self._df)


@register_series_accessor("rhoa")
class RhoaSeriesAccessor:
    def __init__(self, series):
        self._series = series

    @property
    def indicators(self) -> indicators:
        return indicators(self._series)

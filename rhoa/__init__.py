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

__version__ = "0.2.0"

from typing import TYPE_CHECKING

import pandas as pd

from .preprocess import PreprocessAccessor
from .indicators import indicators
from .strategy import StrategyAccessor
from .plots import PlotsAccessor
from . import accessor
from . import data
from . import targets

if TYPE_CHECKING:
    from .accessor import RhoaDataFrameAccessor, RhoaSeriesAccessor
    from typing import overload, Any

    class Series(pd.Series):  # type: ignore[no-redef]
        @property
        def rhoa(self) -> RhoaSeriesAccessor: ...

    class DataFrame(pd.DataFrame):  # type: ignore[no-redef]
        @property
        def rhoa(self) -> RhoaDataFrameAccessor: ...
        @overload
        def __getitem__(self, key: str) -> Series: ...
        @overload
        def __getitem__(self, key: list[str]) -> DataFrame: ...
        def __getitem__(self, key): ...

    def read_csv(*args: Any, **kwargs: Any) -> DataFrame: ...
    def read_excel(*args: Any, **kwargs: Any) -> DataFrame: ...
else:
    DataFrame = pd.DataFrame
    Series = pd.Series
    read_csv = pd.read_csv
    read_excel = pd.read_excel

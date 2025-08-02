# Rhoa - A pandas DataFrame extension for technical analysis
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

import pandas as pd
import unittest
from rhoa.indicators import TechnicalSeriesAccessor


class TestTechnicalSeriesAccessor(unittest.TestCase):

    def setUp(self):
        self.sample_data = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        self.sample_data.indicators = TechnicalSeriesAccessor(self.sample_data)

    def test_sma_with_custom_window(self):
        result = self.sample_data.indicators.sma(window=3)
        expected = pd.Series([None, None, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
        expected.index = self.sample_data.index
        pd.testing.assert_series_equal(result, expected, check_exact=False, check_dtype=False)

    def test_sma_with_window_larger_than_data(self):
        result = self.sample_data.indicators.sma(window=20)
        expected = pd.Series([float('nan')] * len(self.sample_data), dtype='float64')
        expected.index = self.sample_data.index
        pd.testing.assert_series_equal(result, expected, check_exact=True)


if __name__ == '__main__':
    unittest.main()
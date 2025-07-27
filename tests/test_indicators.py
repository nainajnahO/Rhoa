import pandas as pd
import unittest
from noor.indicators import TechnicalSeriesAccessor


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
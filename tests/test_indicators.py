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
import rhoa.indicators
import os


class BaseIndicatorTest(unittest.TestCase):
    """Base test class with shared data loading functionality"""
    
    @classmethod
    def setUpClass(cls):
        # Load real market data from CSV once for all tests
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data.csv')
        cls.df = pd.read_csv(csv_path)
        cls.df['Date'] = pd.to_datetime(cls.df['Date'])
        cls.df.set_index('Date', inplace=True)
        
        # Use Close prices for testing - take a subset for faster tests
        cls.price_data = cls.df['Close'].iloc[:100]
        
        # Additional data series for multi-input indicators
        cls.high_data = cls.df['High'].iloc[:100]
        cls.low_data = cls.df['Low'].iloc[:100]


class TestSMAIndicator(BaseIndicatorTest):
    """Test cases for Simple Moving Average indicator"""
    
    def test_sma_basic_calculation(self):
        result = self.price_data.indicators.sma(window_size=5)
        
        # Test that result is a pandas Series
        self.assertIsInstance(result, pd.Series)
        
        # Test that first 4 values are NaN (not enough data for window)
        self.assertTrue(result.iloc[:4].isna().all())
        
        # Test that we have valid values after the window period
        self.assertFalse(result.iloc[4:].isna().all())
        
        # Test manual calculation for 5th value (index 4)
        expected_5th = self.price_data.iloc[:5].mean()
        self.assertAlmostEqual(result.iloc[4], expected_5th, places=5)
    
    def test_sma_different_window_sizes(self):
        for window in [3, 10, 20]:
            result = self.price_data.indicators.sma(window_size=window)
            
            # Test that we get NaN for initial values
            self.assertTrue(result.iloc[:window-1].isna().all())
            
            # Test that we get valid values after window period
            if self.price_data.shape[0] >= window:
                self.assertFalse(result.iloc[window-1:].isna().all())
    
    def test_sma_window_larger_than_data(self):
        result = self.price_data.indicators.sma(window_size=200)
        # Should return all NaN when window is larger than data
        self.assertTrue(result.isna().all())


class TestRSIIndicator(BaseIndicatorTest):
    """Test cases for Relative Strength Index indicator"""
    
    def test_rsi_basic_calculation(self):
        result = self.price_data.indicators.rsi(window_size=14)
        
        # Test that result is a pandas Series
        self.assertIsInstance(result, pd.Series)
        
        # Test that RSI values are between 0 and 100
        valid_values = result.dropna()
        self.assertTrue((valid_values >= 0).all())
        self.assertTrue((valid_values <= 100).all())
        
        # Test that we have some valid values
        self.assertTrue(len(valid_values) > 0)
    
    def test_rsi_different_window_sizes(self):
        for window in [7, 14, 21]:
            result = self.price_data.indicators.rsi(window_size=window)
            
            # Test value range
            valid_values = result.dropna()
            if len(valid_values) > 0:
                self.assertTrue((valid_values >= 0).all())
                self.assertTrue((valid_values <= 100).all())
    
    def test_rsi_constant_prices_default(self):
        # Test RSI with constant prices using default edge case value (100)
        constant_prices = pd.Series([100] * 50)
        result = constant_prices.indicators.rsi(window_size=14)
        
        # RSI should be 100 for constant prices (no losses, infinite RS)
        valid_values = result.dropna()
        if len(valid_values) > 0:
            # All values should be 100 (no price movement means no losses)
            self.assertTrue(all(abs(val - 100) < 0.01 for val in valid_values))
    
    def test_rsi_constant_prices_neutral(self):
        # Test RSI with constant prices using neutral edge case value (50)
        constant_prices = pd.Series([100] * 50)
        result = constant_prices.indicators.rsi(window_size=14, edge_case_value=50.0)
        
        # RSI should be 50 for constant prices when using neutral setting
        valid_values = result.dropna()
        if len(valid_values) > 0:
            # All values should be 50 (neutral interpretation)
            self.assertTrue(all(abs(val - 50) < 0.01 for val in valid_values))
    
    def test_rsi_constant_prices_nan(self):
        # Test RSI with constant prices using NaN edge case value
        constant_prices = pd.Series([100] * 50)
        result = constant_prices.indicators.rsi(window_size=14, edge_case_value=float('nan'))
        
        # RSI should be NaN for constant prices when using NaN setting
        valid_values = result.dropna()
        # Should have no valid values (all NaN for the edge case)
        self.assertEqual(len(valid_values), 0)


class TestBollingerBandsIndicator(BaseIndicatorTest):
    """Test cases for Bollinger Bands indicator"""
    
    def test_bollinger_bands_basic_calculation(self):
        result = self.price_data.indicators.bollinger_bands(window_size=20, num_std=2.0)
        
        # Test that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Test that it has the correct columns
        expected_columns = ['upper_band', 'middle_band', 'lower_band']
        self.assertEqual(list(result.columns), expected_columns)
        
        # Test that upper > middle > lower (where all are valid)
        valid_rows = result.dropna()
        if len(valid_rows) > 0:
            self.assertTrue(all(valid_rows['upper_band'] >= valid_rows['middle_band']))
            self.assertTrue(all(valid_rows['middle_band'] >= valid_rows['lower_band']))
    
    def test_bollinger_bands_different_std(self):
        for std_dev in [1.0, 2.0, 2.5]:
            result = self.price_data.indicators.bollinger_bands(window_size=20, num_std=std_dev)
            
            valid_rows = result.dropna()
            if len(valid_rows) > 0:
                # Test band relationships
                upper_ge_middle = valid_rows['upper_band'] >= valid_rows['middle_band']
                middle_ge_lower = valid_rows['middle_band'] >= valid_rows['lower_band']
                self.assertTrue(upper_ge_middle.all())
                self.assertTrue(middle_ge_lower.all())
                
                # Test that middle band is SMA
                expected_middle = self.price_data.rolling(window=20).mean()
                pd.testing.assert_series_equal(
                    result['middle_band'], 
                    expected_middle, 
                    check_names=False
                )
    
    def test_bollinger_bands_width_scaling(self):
        result_1std = self.price_data.indicators.bollinger_bands(window_size=20, num_std=1.0)
        result_2std = self.price_data.indicators.bollinger_bands(window_size=20, num_std=2.0)
        
        # 2std bands should be wider than 1std bands
        valid_rows_1 = result_1std.dropna()
        valid_rows_2 = result_2std.dropna()
        
        if len(valid_rows_1) > 0 and len(valid_rows_2) > 0:
            # Compare same time periods
            common_index = valid_rows_1.index.intersection(valid_rows_2.index)
            if len(common_index) > 0:
                width_1std = (valid_rows_1.loc[common_index, 'upper_band'] - 
                              valid_rows_1.loc[common_index, 'lower_band'])
                width_2std = (valid_rows_2.loc[common_index, 'upper_band'] - 
                              valid_rows_2.loc[common_index, 'lower_band'])
                
                self.assertTrue(all(width_2std > width_1std))


class TestMACDIndicator(BaseIndicatorTest):
    """Test cases for MACD indicator"""
    
    def test_macd_basic_calculation(self):
        result = self.price_data.indicators.macd(short_window=12, long_window=26, signal_window=9)
        
        # Test that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Test that it has the correct columns
        expected_columns = ['macd', 'signal', 'histogram']
        self.assertEqual(list(result.columns), expected_columns)
        
        # Test that histogram = macd - signal (where all are valid)
        valid_rows = result.dropna()
        if len(valid_rows) > 0:
            calculated_histogram = valid_rows['macd'] - valid_rows['signal']
            pd.testing.assert_series_equal(
                calculated_histogram, 
                valid_rows['histogram'], 
                check_names=False,
                atol=1e-10
            )
    
    def test_macd_component_relationships(self):
        result = self.price_data.indicators.macd(short_window=12, long_window=26, signal_window=9)
        
        # Test that MACD line is difference of EMAs
        short_ema = self.price_data.ewm(span=12, adjust=False).mean()
        long_ema = self.price_data.ewm(span=26, adjust=False).mean()
        expected_macd = short_ema - long_ema
        
        pd.testing.assert_series_equal(
            result['macd'], 
            expected_macd, 
            check_names=False,
            atol=1e-10
        )
    
    def test_macd_different_parameters(self):
        # Test with different parameter combinations
        param_sets = [
            (5, 10, 3),
            (12, 26, 9),
            (8, 21, 5)
        ]
        
        for short, long, signal in param_sets:
            result = self.price_data.indicators.macd(
                short_window=short, 
                long_window=long, 
                signal_window=signal
            )
            
            # Basic structure tests
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(list(result.columns), ['macd', 'signal', 'histogram'])
            
            # Histogram calculation test
            valid_rows = result.dropna()
            if len(valid_rows) > 0:
                calculated_histogram = valid_rows['macd'] - valid_rows['signal']
                pd.testing.assert_series_equal(
                    calculated_histogram, 
                    valid_rows['histogram'], 
                    check_names=False,
                    atol=1e-10
                )


if __name__ == '__main__':
    unittest.main()
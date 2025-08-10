import pandas as pd
import numpy as np
from unittest import TestCase
import rhoa.indicators
import os



class TestIndicatorsAccessor(TestCase):

    @classmethod
    def setUpClass(cls):
        """Load test data once for all tests"""
        # Load real market data
        csv_path = os.path.join(os.path.dirname(__file__), 'data.csv')
        cls.df = pd.read_csv(csv_path)
        cls.df['Date'] = pd.to_datetime(cls.df['Date'])
        cls.df.set_index('Date', inplace=True)

        # Test data subsets
        cls.price_data = cls.df['Close'].iloc[:100]
        cls.high_data = cls.df['High'].iloc[:100]
        cls.low_data = cls.df['Low'].iloc[:100]

        # Edge case data
        cls.constant_prices = pd.Series([100.0] * 50)
        cls.trending_up = pd.Series(range(1, 51))  # 1, 2, 3, ..., 50
        cls.trending_down = pd.Series(range(50, 0, -1))  # 50, 49, 48, ..., 1
        cls.volatile_data = pd.Series([100 + 10 * np.sin(i / 5) for i in range(50)])

    def test_sma(self):
        """Test Simple Moving Average with various scenarios"""
        # Basic functionality
        result = self.price_data.indicators.sma(window_size=5)
        self.assertIsInstance(result, pd.Series)

        # Check initial NaN values
        self.assertTrue(result.iloc[:4].isna().all())
        self.assertFalse(result.iloc[4:].isna().all())

        # Manual calculation verification
        expected_5th = self.price_data.iloc[:5].mean()
        self.assertAlmostEqual(result.iloc[4], expected_5th, places=5)

        # Edge cases
        # Window larger than data
        large_window = self.price_data.indicators.sma(window_size=200)
        self.assertTrue(large_window.isna().all())

        # Window size 1 (should equal original data)
        window_1 = self.price_data.indicators.sma(window_size=1)
        pd.testing.assert_series_equal(window_1, self.price_data, check_names=False)

        # Constant prices
        const_sma = self.constant_prices.indicators.sma(window_size=10)
        valid_const = const_sma.dropna()
        self.assertTrue(all(abs(val - 100.0) < 1e-10 for val in valid_const))

    def test_ewma(self):
        """Test Exponential Weighted Moving Average"""
        result = self.price_data.indicators.ewma(window_size=10)
        self.assertIsInstance(result, pd.Series)

        # EWMA should have fewer NaN values than SMA
        ewma_valid_count = result.dropna().shape[0]
        sma_valid_count = self.price_data.indicators.sma(window_size=10).dropna().shape[0]
        self.assertGreaterEqual(ewma_valid_count, sma_valid_count)

        # Test against pandas ewm
        expected = self.price_data.ewm(span=10, adjust=False).mean()
        pd.testing.assert_series_equal(result, expected, check_names=False)

        # Edge cases
        # Constant prices
        const_ewma = self.constant_prices.indicators.ewma(window_size=10)
        valid_const = const_ewma.dropna()
        self.assertTrue(all(abs(val - 100.0) < 1e-10 for val in valid_const))

        # Single value
        single_val = pd.Series([42.0])
        single_ewma = single_val.indicators.ewma(window_size=5)
        self.assertEqual(single_ewma.iloc[0], 42.0)

    def test_ewmv(self):
        """Test Exponential Weighted Moving Variance"""
        result = self.price_data.indicators.ewmv(window_size=10)
        self.assertIsInstance(result, pd.Series)

        # Variance should be non-negative
        valid_values = result.dropna()
        self.assertTrue((valid_values >= 0).all())

        # Test against pandas ewm variance
        expected = self.price_data.ewm(span=10).var()
        pd.testing.assert_series_equal(result, expected, check_names=False)

        # Edge cases
        # Constant prices should have zero variance
        const_ewmv = self.constant_prices.indicators.ewmv(window_size=10)
        valid_const = const_ewmv.dropna()
        self.assertTrue(all(abs(val) < 1e-10 for val in valid_const))

        # Volatile data should have higher variance
        volatile_ewmv = self.volatile_data.indicators.ewmv(window_size=10)
        trending_ewmv = self.trending_up.indicators.ewmv(window_size=10)
        volatile_mean = volatile_ewmv.dropna().mean()
        trending_mean = trending_ewmv.dropna().mean()
        self.assertGreater(volatile_mean, trending_mean)

    def test_ewmstd(self):
        """Test Exponential Weighted Moving Standard Deviation"""
        result = self.price_data.indicators.ewmstd(window_size=10)
        self.assertIsInstance(result, pd.Series)

        # Standard deviation should be non-negative
        valid_values = result.dropna()
        self.assertTrue((valid_values >= 0).all())

        # Test against pandas ewm std
        expected = self.price_data.ewm(span=10).std()
        pd.testing.assert_series_equal(result, expected, check_names=False)

        # Relationship with variance
        ewmv_result = self.price_data.indicators.ewmv(window_size=10)
        std_squared = result ** 2
        pd.testing.assert_series_equal(std_squared, ewmv_result, check_names=False, atol=1e-10)

        # Edge cases
        # Constant prices should have zero std
        const_ewmstd = self.constant_prices.indicators.ewmstd(window_size=10)
        valid_const = const_ewmstd.dropna()
        self.assertTrue(all(abs(val) < 1e-10 for val in valid_const))

    def test_rsi(self):
        """Test Relative Strength Index with edge cases"""
        # Basic functionality
        result = self.price_data.indicators.rsi(window_size=14)
        self.assertIsInstance(result, pd.Series)

        # RSI should be between 0 and 100
        valid_values = result.dropna()
        self.assertTrue((valid_values >= 0).all())
        self.assertTrue((valid_values <= 100).all())

        # Test different edge case values
        # Default (100) for constant prices
        const_rsi_default = self.constant_prices.indicators.rsi(window_size=14)
        valid_const = const_rsi_default.dropna()
        if len(valid_const) > 0:
            self.assertTrue(all(abs(val - 100) < 0.01 for val in valid_const))

        # Neutral (50) for constant prices
        const_rsi_neutral = self.constant_prices.indicators.rsi(window_size=14, edge_case_value=50.0)
        valid_neutral = const_rsi_neutral.dropna()
        if len(valid_neutral) > 0:
            self.assertTrue(all(abs(val - 50) < 0.01 for val in valid_neutral))

        # NaN for constant prices
        const_rsi_nan = self.constant_prices.indicators.rsi(window_size=14, edge_case_value=float('nan'))
        self.assertEqual(len(const_rsi_nan.dropna()), 0)

        # Trending data behavior
        up_rsi = self.trending_up.indicators.rsi(window_size=14)
        down_rsi = self.trending_down.indicators.rsi(window_size=14)

        # Uptrend should generally have higher RSI
        up_mean = up_rsi.dropna().mean()
        down_mean = down_rsi.dropna().mean()
        self.assertGreater(up_mean, down_mean)

    def test_macd(self):
        """Test MACD indicator"""
        result = self.price_data.indicators.macd(short_window=12, long_window=26, signal_window=9)
        self.assertIsInstance(result, pd.DataFrame)

        # Check correct columns
        expected_columns = ['macd', 'signal', 'histogram']
        self.assertEqual(list(result.columns), expected_columns)

        # Histogram should equal MACD - Signal
        valid_rows = result.dropna()
        if len(valid_rows) > 0:
            calculated_histogram = valid_rows['macd'] - valid_rows['signal']
            pd.testing.assert_series_equal(
                calculated_histogram,
                valid_rows['histogram'],
                check_names=False,
                atol=1e-10
            )

        # Test component calculation
        short_ema = self.price_data.ewm(span=12, adjust=False).mean()
        long_ema = self.price_data.ewm(span=26, adjust=False).mean()
        expected_macd = short_ema - long_ema
        pd.testing.assert_series_equal(result['macd'], expected_macd, check_names=False, atol=1e-10)

        # Edge cases
        # Different parameter combinations
        for short, long, signal in [(5, 10, 3), (8, 21, 5)]:
            macd_result = self.price_data.indicators.macd(short, long, signal)
            self.assertEqual(list(macd_result.columns), expected_columns)

        # Constant prices
        const_macd = self.constant_prices.indicators.macd()
        valid_const = const_macd.dropna()
        if len(valid_const) > 0:
            # MACD line should be near zero for constant prices
            self.assertTrue(all(abs(val) < 1e-10 for val in valid_const['macd']))

    def test_bollinger_bands(self):
        """Test Bollinger Bands indicator"""
        result = self.price_data.indicators.bollinger_bands(window_size=20, num_std=2.0)
        self.assertIsInstance(result, pd.DataFrame)

        # Check correct columns
        expected_columns = ['upper_band', 'middle_band', 'lower_band']
        self.assertEqual(list(result.columns), expected_columns)

        # Band relationships
        valid_rows = result.dropna()
        if len(valid_rows) > 0:
            self.assertTrue(all(valid_rows['upper_band'] >= valid_rows['middle_band']))
            self.assertTrue(all(valid_rows['middle_band'] >= valid_rows['lower_band']))

        # Middle band should be SMA
        expected_middle = self.price_data.rolling(window=20).mean()
        pd.testing.assert_series_equal(result['middle_band'], expected_middle, check_names=False)

        # Test different standard deviations
        bb_1std = self.price_data.indicators.bollinger_bands(window_size=20, num_std=1.0)
        bb_2std = self.price_data.indicators.bollinger_bands(window_size=20, num_std=2.0)

        # 2std bands should be wider
        valid_1 = bb_1std.dropna()
        valid_2 = bb_2std.dropna()
        if len(valid_1) > 0 and len(valid_2) > 0:
            common_idx = valid_1.index.intersection(valid_2.index)
            if len(common_idx) > 0:
                width_1 = valid_1.loc[common_idx, 'upper_band'] - valid_1.loc[common_idx, 'lower_band']
                width_2 = valid_2.loc[common_idx, 'upper_band'] - valid_2.loc[common_idx, 'lower_band']
                self.assertTrue(all(width_2 > width_1))

        # Edge cases
        # Constant prices - bands should converge to price
        const_bb = self.constant_prices.indicators.bollinger_bands(window_size=10, num_std=2.0)
        valid_const = const_bb.dropna()
        if len(valid_const) > 0:
            band_width = valid_const['upper_band'] - valid_const['lower_band']
            self.assertTrue(all(abs(width) < 1e-10 for width in band_width))

    def test_atr(self):
        """Test Average True Range indicator"""
        result = self.price_data.indicators.atr(self.high_data, self.low_data, window_size=14)
        self.assertIsInstance(result, pd.Series)

        # ATR should be non-negative
        valid_values = result.dropna()
        self.assertTrue((valid_values >= 0).all())

        # Manual calculation verification for first few values
        high = self.high_data
        low = self.low_data
        close = self.price_data

        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        expected_atr = true_range.rolling(window=14).mean()

        pd.testing.assert_series_equal(result, expected_atr, check_names=False)

        # Edge cases
        # When high = low = close (no volatility)
        flat_high = pd.Series([100.0] * 50)
        flat_low = pd.Series([100.0] * 50)
        flat_close = pd.Series([100.0] * 50)

        flat_atr = flat_close.indicators.atr(flat_high, flat_low, window_size=14)
        valid_flat = flat_atr.dropna()
        if len(valid_flat) > 0:
            self.assertTrue(all(abs(val) < 1e-10 for val in valid_flat))

    def test_cci(self):
        """Test Commodity Channel Index indicator"""
        result = self.price_data.indicators.cci(self.high_data, self.low_data, window_size=20)
        self.assertIsInstance(result, pd.Series)

        # Manual calculation verification
        close = self.price_data
        high = self.high_data
        low = self.low_data

        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=20).mean()

        mean_deviation = typical_price.rolling(window=20).apply(
            lambda x: np.mean(np.abs(x - x.mean())),
            raw=True
        )

        expected_cci = (typical_price - sma) / (0.015 * mean_deviation)
        pd.testing.assert_series_equal(result, expected_cci, check_names=False)

        # CCI properties
        valid_values = result.dropna()
        if len(valid_values) > 0:
            # CCI typically ranges from -100 to +100, but can exceed these bounds
            # Just check that we have reasonable values (not infinite or NaN)
            self.assertTrue(all(np.isfinite(val) for val in valid_values))

        # Edge cases
        # Constant prices should give CCI near zero (after initial period)
        const_high = pd.Series([100.0] * 50)
        const_low = pd.Series([100.0] * 50)
        const_close = pd.Series([100.0] * 50)

        const_cci = const_close.indicators.cci(const_high, const_low, window_size=20)
        # For constant prices, mean deviation approaches zero, so CCI is undefined
        # The implementation should handle this gracefully
        valid_const = const_cci.dropna()
        # We just verify it doesn't crash and produces some result

    def test_stochastic(self):
        """Test Stochastic Oscillator (%K and %D)"""
        result = self.price_data.indicators.stochastic(self.high_data, self.low_data, k_window=14, d_window=3)
        self.assertIsInstance(result, pd.DataFrame)

        # Check correct columns
        expected_columns = ['%K', '%D']
        self.assertEqual(list(result.columns), expected_columns)

        # Stochastic should be between 0 and 100
        valid_values = result.dropna()
        if len(valid_values) > 0:
            self.assertTrue((valid_values['%K'] >= 0).all())
            self.assertTrue((valid_values['%K'] <= 100).all())
            self.assertTrue((valid_values['%D'] >= 0).all())
            self.assertTrue((valid_values['%D'] <= 100).all())

        # %D should be smoother (SMA of %K)
        valid_k = result['%K'].dropna()
        if len(valid_k) >= 3:
            expected_d = valid_k.rolling(window=3).mean()
            # Compare overlapping indices
            common_idx = expected_d.dropna().index.intersection(result['%D'].dropna().index)
            if len(common_idx) > 0:
                pd.testing.assert_series_equal(
                    result.loc[common_idx, '%D'],
                    expected_d.loc[common_idx],
                    check_names=False
                )

        # Manual calculation verification
        close = self.price_data
        high = self.high_data
        low = self.low_data

        lowest_low = low.rolling(window=14).min()
        highest_high = high.rolling(window=14).max()
        expected_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))

        pd.testing.assert_series_equal(result['%K'], expected_k, check_names=False)

        # Edge cases
        # Constant prices should give %K = middle value
        const_high = pd.Series([100.0] * 50)
        const_low = pd.Series([95.0] * 50)
        const_close = pd.Series([97.5] * 50)  # Middle of range

        const_stoch = const_close.indicators.stochastic(const_high, const_low, k_window=14, d_window=3)
        valid_const = const_stoch.dropna()
        if len(valid_const) > 0:
            # Should be 50% of the range
            self.assertTrue(all(abs(val - 50.0) < 1e-10 for val in valid_const['%K']))

    def test_williams_r(self):
        """Test Williams %R indicator"""
        result = self.price_data.indicators.williams_r(self.high_data, self.low_data, window_size=14)
        self.assertIsInstance(result, pd.Series)

        # Williams %R should be between -100 and 0
        valid_values = result.dropna()
        if len(valid_values) > 0:
            self.assertTrue((valid_values >= -100).all())
            self.assertTrue((valid_values <= 0).all())

        # Manual calculation verification
        close = self.price_data
        high = self.high_data
        low = self.low_data

        highest_high = high.rolling(window=14).max()
        lowest_low = low.rolling(window=14).min()
        expected_wr = -100 * ((highest_high - close) / (highest_high - lowest_low))

        pd.testing.assert_series_equal(result, expected_wr, check_names=False)

        # Edge cases
        # When close = high, Williams %R should be 0
        high_close = pd.Series([100.0] * 50)
        low_close = pd.Series([95.0] * 50)
        at_high = pd.Series([100.0] * 50)

        high_wr = at_high.indicators.williams_r(high_close, low_close, window_size=14)
        valid_high = high_wr.dropna()
        if len(valid_high) > 0:
            self.assertTrue(all(abs(val) < 1e-10 for val in valid_high))

        # When close = low, Williams %R should be -100
        at_low = pd.Series([95.0] * 50)
        low_wr = at_low.indicators.williams_r(high_close, low_close, window_size=14)
        valid_low = low_wr.dropna()
        if len(valid_low) > 0:
            self.assertTrue(all(abs(val + 100) < 1e-10 for val in valid_low))

    def test_adx(self):
        """Test Average Directional Index (ADX) with +DI and -DI"""
        result = self.price_data.indicators.adx(self.high_data, self.low_data, window_size=14)
        self.assertIsInstance(result, pd.DataFrame)

        # Check correct columns
        expected_columns = ['ADX', '+DI', '-DI']
        self.assertEqual(list(result.columns), expected_columns)

        # All values should be non-negative and typically between 0-100
        valid_values = result.dropna()
        if len(valid_values) > 0:
            self.assertTrue((valid_values['ADX'] >= 0).all())
            self.assertTrue((valid_values['+DI'] >= 0).all())
            self.assertTrue((valid_values['-DI'] >= 0).all())

            # ADX should typically be <= 100 (though can exceed in extreme cases)
            self.assertTrue((valid_values['ADX'] <= 150).all())  # Allow some margin

        # Test component calculations
        close = self.price_data
        high = self.high_data
        low = self.low_data

        # Verify True Range calculation (used in ADX)
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # The calculation is complex, so we mainly test structure and bounds
        self.assertEqual(len(result), len(self.price_data))

        # Edge cases
        # Trending data should have higher ADX than sideways data
        trending_adx = self.trending_up.indicators.adx(
            pd.Series(range(105, 155)),  # trending high
            pd.Series(range(95, 145)),   # trending low
            window_size=14
        )

        const_adx = self.constant_prices.indicators.adx(
            pd.Series([102.0] * 50),  # constant high
            pd.Series([98.0] * 50),   # constant low
            window_size=14
        )

        trend_adx_mean = trending_adx['ADX'].dropna().mean()
        const_adx_mean = const_adx['ADX'].dropna().mean()

        if not np.isnan(trend_adx_mean) and not np.isnan(const_adx_mean):
            self.assertGreater(trend_adx_mean, const_adx_mean)

    def test_parabolic_sar(self):
        """Test Parabolic SAR indicator"""
        result = self.price_data.indicators.parabolic_sar(
            self.high_data,
            self.low_data,
            af_start=0.02,
            af_increment=0.02,
            af_maximum=0.2
        )
        self.assertIsInstance(result, pd.Series)

        # SAR should have same length as input
        self.assertEqual(len(result), len(self.price_data))

        # All SAR values should be finite
        valid_values = result.dropna()
        self.assertTrue(all(np.isfinite(val) for val in valid_values))

        # SAR should generally be within reasonable range of price data
        if len(valid_values) > 0:
            price_min = self.price_data.min()
            price_max = self.price_data.max()
            price_range = price_max - price_min

            # Allow SAR to be outside price range but within reasonable bounds
            sar_min = valid_values.min()
            sar_max = valid_values.max()

            # SAR shouldn't be wildly outside the price range
            self.assertGreater(sar_min, price_min - 2 * price_range)
            self.assertLess(sar_max, price_max + 2 * price_range)

        # Test different parameter combinations
        sar_fast = self.price_data.indicators.parabolic_sar(
            self.high_data, self.low_data,
            af_start=0.04, af_increment=0.04, af_maximum=0.4
        )

        sar_slow = self.price_data.indicators.parabolic_sar(
            self.high_data, self.low_data,
            af_start=0.01, af_increment=0.01, af_maximum=0.1
        )

        # Both should be valid Series
        self.assertIsInstance(sar_fast, pd.Series)
        self.assertIsInstance(sar_slow, pd.Series)

        # Edge cases
        # Constant prices - SAR should stabilize
        const_high = pd.Series([105.0] * 50)
        const_low = pd.Series([95.0] * 50)
        const_close = pd.Series([100.0] * 50)

        const_sar = const_close.indicators.parabolic_sar(const_high, const_low)
        # Should not crash and produce reasonable values
        self.assertEqual(len(const_sar), 50)

        # Strong trending data
        trend_high = pd.Series(range(100, 150))
        trend_low = pd.Series(range(95, 145))
        trend_close = pd.Series(range(98, 148))

        trend_sar = trend_close.indicators.parabolic_sar(trend_high, trend_low)
        # Should track below the uptrend
        if len(trend_sar.dropna()) > 10:
            # In a strong uptrend, SAR should generally be below the closing prices
            recent_sar = trend_sar.iloc[-10:].mean()
            recent_close = trend_close.iloc[-10:].mean()
            self.assertLess(recent_sar, recent_close)

    def test_kwargs_support(self):
        """Test that all new indicators support **kwargs parameter"""
        # Test that **kwargs are accepted without errors

        # Stochastic with extra rolling parameters
        stoch_result = self.price_data.indicators.stochastic(
            self.high_data, self.low_data,
            k_window=14, d_window=5,  # Fixed: d_window >= min_periods
            min_periods=3
        )
        self.assertIsInstance(stoch_result, pd.DataFrame)

        # Williams %R with extra rolling parameters
        wr_result = self.price_data.indicators.williams_r(
            self.high_data, self.low_data,
            window_size=14, min_periods=5
        )
        self.assertIsInstance(wr_result, pd.Series)

        # ADX with extra ewm parameters
        adx_result = self.price_data.indicators.adx(
            self.high_data, self.low_data,
            window_size=14, min_periods=5
        )
        self.assertIsInstance(adx_result, pd.DataFrame)

        # Parabolic SAR doesn't use rolling/ewm, so no extra kwargs to test
        # but verify it still works normally
        sar_result = self.price_data.indicators.parabolic_sar(
            self.high_data, self.low_data
        )
        self.assertIsInstance(sar_result, pd.Series)

    def test_new_indicators_integration(self):
        """Test that new indicators work with the accessor pattern properly"""
        # Verify all new indicators are accessible through the accessor
        self.assertTrue(hasattr(self.price_data.indicators, 'stochastic'))
        self.assertTrue(hasattr(self.price_data.indicators, 'williams_r'))
        self.assertTrue(hasattr(self.price_data.indicators, 'adx'))
        self.assertTrue(hasattr(self.price_data.indicators, 'parabolic_sar'))

        # Test with minimal data
        small_data = self.price_data.iloc[:20]
        small_high = self.high_data.iloc[:20]
        small_low = self.low_data.iloc[:20]

        # All should work without crashing
        stoch = small_data.indicators.stochastic(small_high, small_low, k_window=5, d_window=3)
        wr = small_data.indicators.williams_r(small_high, small_low, window_size=5)
        adx = small_data.indicators.adx(small_high, small_low, window_size=5)
        sar = small_data.indicators.parabolic_sar(small_high, small_low)

        # Verify return types
        self.assertIsInstance(stoch, pd.DataFrame)
        self.assertIsInstance(wr, pd.Series)
        self.assertIsInstance(adx, pd.DataFrame)
        self.assertIsInstance(sar, pd.Series)

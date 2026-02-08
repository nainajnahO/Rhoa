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

import pytest
import pandas as pd
import numpy as np
from rhoa.targets import generate_target_combinations


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
    high_prices = close_prices + np.random.rand(100) * 3

    return pd.DataFrame({
        'Date': dates,
        'Close': close_prices,
        'High': high_prices
    })


class TestGenerateTargetCombinations:
    """Tests for generate_target_combinations function."""

    def test_auto_mode_basic(self, sample_df):
        """Test auto mode with default parameters."""
        targets, meta = generate_target_combinations(
            sample_df,
            mode='auto',
            target_class_balance=0.5,
            max_period=10
        )

        # Check output structure
        assert targets.shape[1] == 8, "Should have 8 target columns"
        assert len(targets) == len(sample_df), "Should have same length as input"
        assert meta['mode'] == 'auto', "Mode should be 'auto'"

        # Check metadata structure
        for i in range(1, 9):
            method_key = f'method_{i}'
            assert method_key in meta, f"Missing {method_key} in metadata"
            assert 'period' in meta[method_key]
            assert 'threshold' in meta[method_key]
            assert 'instances' in meta[method_key]
            assert 'pct_of_max' in meta[method_key]

    def test_manual_mode_basic(self, sample_df):
        """Test manual mode with default parameters."""
        targets, meta = generate_target_combinations(
            sample_df,
            mode='manual',
            lookback_periods=5
        )

        # Check output structure
        assert targets.shape[1] == 8, "Should have 8 target columns"
        assert len(targets) == len(sample_df), "Should have same length as input"
        assert meta['mode'] == 'manual', "Mode should be 'manual'"

        # Check that all methods use the same period
        for i in range(1, 9):
            method_key = f'method_{i}'
            assert meta[method_key]['period'] == 5, "All methods should use period=5"

    def test_auto_mode_different_class_balances(self, sample_df):
        """Test auto mode with different target class balances."""
        for balance in [0.1, 0.5, 0.8]:
            targets, meta = generate_target_combinations(
                sample_df,
                mode='auto',
                target_class_balance=balance,
                max_period=10,
                step=5
            )

            assert targets.shape[1] == 8
            assert meta['mode'] == 'auto'

    def test_manual_mode_different_periods(self, sample_df):
        """Test manual mode with different lookback periods."""
        for period in [3, 7, 14]:
            targets, meta = generate_target_combinations(
                sample_df,
                mode='manual',
                lookback_periods=period
            )

            # All methods should use the specified period
            for i in range(1, 9):
                assert meta[f'method_{i}']['period'] == period

    def test_target_column_names(self, sample_df):
        """Test that target columns have correct names."""
        targets, _ = generate_target_combinations(sample_df, mode='manual')

        expected_columns = [f'Target_{i}' for i in range(1, 9)]
        assert list(targets.columns) == expected_columns

    def test_target_boolean_dtype(self, sample_df):
        """Test that target columns are boolean."""
        targets, _ = generate_target_combinations(sample_df, mode='manual')

        for col in targets.columns:
            assert targets[col].dtype == bool, f"{col} should be boolean"

    def test_invalid_mode_raises_error(self, sample_df):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be 'auto' or 'manual'"):
            generate_target_combinations(sample_df, mode='invalid')

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="DataFrame is empty"):
            generate_target_combinations(empty_df)

    def test_missing_close_column_raises_error(self, sample_df):
        """Test that missing Close column raises ValueError."""
        df_no_close = sample_df.drop(columns=['Close'])

        with pytest.raises(ValueError, match="Column 'Close' not found"):
            generate_target_combinations(df_no_close)

    def test_missing_high_column_raises_error(self, sample_df):
        """Test that missing High column raises ValueError."""
        df_no_high = sample_df.drop(columns=['High'])

        with pytest.raises(ValueError, match="Column 'High' not found"):
            generate_target_combinations(df_no_high)

    def test_custom_column_names(self, sample_df):
        """Test with custom column names."""
        df = sample_df.rename(columns={'Close': 'close_price', 'High': 'high_price'})

        targets, meta = generate_target_combinations(
            df,
            mode='manual',
            lookback_periods=5,
            close_col='close_price',
            high_col='high_price'
        )

        assert targets.shape[1] == 8
        assert meta['mode'] == 'manual'

    def test_metadata_numeric_types(self, sample_df):
        """Test that metadata contains correct numeric types."""
        targets, meta = generate_target_combinations(sample_df, mode='auto', max_period=5)

        for i in range(1, 9):
            method = meta[f'method_{i}']
            assert isinstance(method['period'], int)
            assert isinstance(method['threshold'], float)
            assert isinstance(method['instances'], int)
            assert isinstance(method['pct_of_max'], float)

    def test_threshold_range(self, sample_df):
        """Test that thresholds are within specified range."""
        targets, meta = generate_target_combinations(
            sample_df,
            mode='manual',
            lookback_periods=5,
            min_pct=0,
            max_pct=50
        )

        for i in range(1, 9):
            threshold = meta[f'method_{i}']['threshold']
            assert 0 <= threshold < 50, f"Threshold {threshold} out of range"

    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({
            'Close': [100.0],
            'High': [105.0]
        })

        targets, meta = generate_target_combinations(df, mode='manual', lookback_periods=1)

        assert targets.shape == (1, 8)

    def test_constant_prices(self):
        """Test with constant prices (no variation)."""
        df = pd.DataFrame({
            'Close': [100.0] * 50,
            'High': [100.0] * 50
        })

        targets, meta = generate_target_combinations(df, mode='manual', lookback_periods=5)

        # With constant prices and threshold > 0, all targets should be False
        for col in targets.columns:
            if meta[f"method_{col.split('_')[1]}"]['threshold'] > 0:
                assert targets[col].sum() == 0, f"{col} should have no True values"

    def test_reproducibility_auto_mode(self, sample_df):
        """Test that auto mode produces reproducible results."""
        targets1, meta1 = generate_target_combinations(
            sample_df,
            mode='auto',
            target_class_balance=0.5,
            max_period=10
        )

        targets2, meta2 = generate_target_combinations(
            sample_df,
            mode='auto',
            target_class_balance=0.5,
            max_period=10
        )

        # Results should be identical
        pd.testing.assert_frame_equal(targets1, targets2)
        assert meta1 == meta2

    def test_reproducibility_manual_mode(self, sample_df):
        """Test that manual mode produces reproducible results."""
        targets1, meta1 = generate_target_combinations(
            sample_df,
            mode='manual',
            lookback_periods=5
        )

        targets2, meta2 = generate_target_combinations(
            sample_df,
            mode='manual',
            lookback_periods=5
        )

        # Results should be identical
        pd.testing.assert_frame_equal(targets1, targets2)
        assert meta1 == meta2

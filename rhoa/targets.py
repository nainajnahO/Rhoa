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
Target Generation for Financial Time Series Analysis
=====================================================

This module provides tools for generating optimized binary classification targets
based on future price movements in financial time series data. It supports two
optimization strategies: Pareto multi-objective optimization and elbow method
threshold selection.

Core Functionality
------------------
- Generate 8 target combinations using different entry/exit price definitions
- Automatic parameter optimization for lookback periods and thresholds
- Support for both end-of-period and maximum-during-period gain calculations
- Flexible class balance targeting or elbow-based threshold selection

Target Methods
--------------
The module generates eight distinct target definitions, each representing a
different combination of entry price (Close[t] or High[t]) and exit price
(end-of-period or maximum-during-period):

1. Close[t+N] / Close[t] - Conservative entry, end-point exit
2. Close[t+N] / High[t] - Aggressive entry, end-point exit
3. High[t+N] / Close[t] - Conservative entry, high exit
4. High[t+N] / High[t] - Aggressive entry, high exit
5. max(Close[t+1:t+N]) / Close[t] - Conservative entry, optimal close exit
6. max(Close[t+1:t+N]) / High[t] - Aggressive entry, optimal close exit
7. max(High[t+1:t+N]) / Close[t] - Conservative entry, optimal high exit
8. max(High[t+1:t+N]) / High[t] - Aggressive entry, optimal high exit

Optimization Modes
------------------
Auto Mode (Pareto Optimization)
    Searches the space of (period, threshold) combinations to find Pareto-optimal
    solutions that balance three objectives:
    - Maximize threshold (higher precision requirements)
    - Minimize period (shorter holding time)
    - Minimize deviation from target class balance

Manual Mode (Elbow Method)
    Uses a fixed lookback period and finds the optimal threshold using the
    elbow/knee point detection on the curve of instance counts vs. thresholds.

Mathematical Background
-----------------------
Pareto Optimization
    A solution is Pareto-optimal if no other solution exists that improves at
    least one objective without worsening any other. For point A to dominate B:
    - A must be better than B in at least one objective
    - A must be no worse than B in all other objectives

Elbow Method
    Identifies the point of maximum curvature on a convex decreasing curve by
    finding the point with maximum perpendicular distance from the line
    connecting curve endpoints.

Dependencies
------------
- pandas : DataFrame operations and time series handling
- numpy : Numerical computations
- kneed : Elbow/knee point detection (KneeLocator)
- paretoset : Pareto frontier computation

Notes
-----
**Look-ahead Bias Warning:**
    These targets use future price information and should only be used for
    target creation in supervised learning. Never use future data for feature
    engineering or training, only for defining what the model should predict.

**NaN Handling:**
    The last N rows (where N is the lookback period) will contain NaN values
    due to insufficient future data. These should be excluded from analysis.

**Class Imbalance:**
    Very low target_class_balance (< 0.1) may result in insufficient positive
    examples. Very high values (> 0.9) may result in overly easy targets with
    poor discrimination.

Examples
--------
Basic usage with auto mode:

>>> import pandas as pd
>>> import numpy as np
>>> from rhoa.targets import generate_target_combinations
>>>
>>> # Load your OHLC data
>>> df = pd.read_csv('prices.csv', index_col='Date', parse_dates=True)
>>>
>>> # Generate targets with 50% class balance
>>> targets, metadata = generate_target_combinations(
...     df, mode='auto', target_class_balance=0.5
... )
>>>
>>> print(f"Generated {len(targets.columns)} targets")
>>> print(f"Target_1 has {targets['Target_1'].sum()} positive instances")

Manual mode with fixed period:

>>> targets, metadata = generate_target_combinations(
...     df, mode='manual', lookback_periods=10
... )
>>> print(f"Method 1 threshold: {metadata['method_1']['threshold']}%")

See Also
--------
generate_target_combinations : Main function for target generation.

References
----------
.. [1] Pareto, V. (1906). "Manuale di economia politica"
.. [2] Satopää, V., et al. (2011). "Finding a 'Kneedle' in a Haystack:
       Detecting Knee Points in System Behavior"
.. [3] Deb, K., et al. (2002). "A fast and elitist multiobjective genetic
       algorithm: NSGA-II"
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from kneed import KneeLocator
from paretoset import paretoset


def generate_target_combinations(
    df: pd.DataFrame,
    mode: str = 'auto',
    # Manual mode parameters:
    lookback_periods: int = 5,
    # Auto mode parameters:
    target_class_balance: float = 0.5,
    min_period: int = 1,
    max_period: int = 20,
    period_step: int = 1,
    # Common parameters:
    min_pct: int = 0,
    max_pct: int = 100,
    step: int = 1,
    close_col: str = "Close",
    high_col: str = "High"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate eight target combinations with optimized thresholds and lookback periods.

    This function creates binary classification targets based on future price movements,
    supporting two optimization modes: automatic Pareto-based optimization and manual
    elbow-based threshold selection. Each target represents a different way of measuring
    future price gains relative to current entry prices.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing OHLC (Open, High, Low, Close) price data.
        Must have at least the columns specified by `close_col` and `high_col`.
        Index should be a time series (e.g., DatetimeIndex).
    mode : {'auto', 'manual'}, default='auto'
        Optimization mode to use:

        - 'auto' : Uses Pareto optimization to find optimal lookback period and
          threshold that balance multiple objectives (maximize threshold, minimize
          period, achieve target class balance).
        - 'manual' : Uses fixed lookback period with elbow method to find optimal
          threshold based on the curve of instance counts vs. thresholds.
    lookback_periods : int, default=5
        Number of periods to look forward for future price calculations.
        Only used when `mode='manual'`.
        Must be >= 1 and < len(df).
    target_class_balance : float, default=0.5
        Target proportion of positive class instances (range: 0.0 to 1.0).
        Only used when `mode='auto'`.
        For example, 0.5 means aim for 50% positive instances, 0.3 means 30%.
    min_period : int, default=1
        Minimum lookback period to consider in optimization search space.
        Only used when `mode='auto'`.
        Must be >= 1.
    max_period : int, default=20
        Maximum lookback period to consider in optimization search space.
        Only used when `mode='auto'`.
        Must be > min_period and < len(df).
    period_step : int, default=1
        Increment step for lookback period search.
        Only used when `mode='auto'`.
        Must be >= 1.
    min_pct : int, default=0
        Minimum threshold percentage to consider (e.g., 0 for 0%).
        Must be >= 0 and < max_pct.
    max_pct : int, default=100
        Maximum threshold percentage to consider (e.g., 100 for 100%).
        Must be > min_pct and <= 100.
    step : int, default=1
        Increment step for threshold search in percentage points.
        Must be >= 1.
    close_col : str, default='Close'
        Name of the close price column in the DataFrame.
    high_col : str, default='High'
        Name of the high price column in the DataFrame.

    Returns
    -------
    targets_df : pd.DataFrame
        DataFrame with same index as input `df`, containing 8 boolean columns:

        - Target_1 : (Close[t+N] / Close[t]) - 1 >= threshold
        - Target_2 : (Close[t+N] / High[t]) - 1 >= threshold
        - Target_3 : (High[t+N] / Close[t]) - 1 >= threshold
        - Target_4 : (High[t+N] / High[t]) - 1 >= threshold
        - Target_5 : (max(Close[t+1:t+N]) / Close[t]) - 1 >= threshold
        - Target_6 : (max(Close[t+1:t+N]) / High[t]) - 1 >= threshold
        - Target_7 : (max(High[t+1:t+N]) / Close[t]) - 1 >= threshold
        - Target_8 : (max(High[t+1:t+N]) / High[t]) - 1 >= threshold

        Where N is the optimized lookback period, and threshold is the optimized
        percentage gain threshold.

    metadata : dict
        Dictionary containing optimization results and configuration with keys:

        - 'mode' : str
            The mode used ('auto' or 'manual').
        - 'method_1' through 'method_8' : dict
            Each method dictionary contains:

            - 'period' : int
                Optimal lookback period in number of time steps.
            - 'threshold' : float
                Optimal threshold as percentage (e.g., 5.0 for 5%).
            - 'instances' : int
                Number of positive instances at the optimal parameters.
            - 'pct_of_max' : float
                Percentage of maximum possible instances (at 0% threshold).

    Raises
    ------
    ValueError
        If the DataFrame is empty.
    ValueError
        If `close_col` or `high_col` not found in DataFrame columns.
    ValueError
        If `mode` is not 'auto' or 'manual'.

    See Also
    --------
    _find_optimal_params_pareto : Pareto optimization for auto mode.
    _find_optimal_params_elbow : Elbow method for manual mode.
    _generate_targets : Generates target columns from optimal parameters.

    Notes
    -----
    **Target Interpretation:**

    - Targets 1-4 measure end-of-period gains (single point in time at t+N).
    - Targets 5-8 measure maximum gains during the period (any time in [t+1, t+N]).
    - Using High[t] as denominator (Targets 2, 4, 6, 8) is more conservative than
      Close[t], as it requires overcoming intraday peaks.
    - Maximum-based targets (5-8) capture exit opportunities that might occur before
      the end of the lookback period.

    **Pareto Optimization (Auto Mode):**

    The Pareto optimization finds solutions that are not dominated by any other
    solution in the objective space. A solution A dominates solution B if:

    - A is better than B in at least one objective
    - A is no worse than B in all other objectives

    For this problem, we optimize three objectives:

    1. Maximize threshold (prefer higher gain requirements)
    2. Minimize period (prefer shorter holding periods)
    3. Minimize deviation from target class balance

    From the Pareto-optimal set, we select the solution closest to the target
    class balance.

    **Elbow Method (Manual Mode):**

    The elbow method finds the "knee" or "elbow" point on the curve of instance
    counts vs. thresholds. This point represents the optimal trade-off where:

    - Increasing threshold further causes steep drops in instances (high cost)
    - Decreasing threshold provides diminishing returns in instances

    Mathematically, the elbow is found by maximizing the distance from the curve
    to the line connecting the endpoints.

    **Common Pitfalls:**

    - Insufficient data: Ensure df has enough rows for the lookback period.
    - Look-ahead bias: Do not use future data for training; only for target creation.
    - Class imbalance: Very low or very high target_class_balance may yield poor results.
    - NaN handling: Last N rows will have NaN targets due to insufficient future data.

    Examples
    --------
    **Example 1: Auto mode with default parameters**

    >>> import pandas as pd
    >>> import numpy as np
    >>> from rhoa.targets import generate_target_combinations
    >>>
    >>> # Create sample OHLC data
    >>> np.random.seed(42)
    >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
    >>> df = pd.DataFrame({
    ...     'Close': 100 + np.cumsum(np.random.randn(100)),
    ...     'High': 100 + np.cumsum(np.random.randn(100)) + 1
    ... }, index=dates)
    >>>
    >>> # Generate targets with auto mode
    >>> targets, meta = generate_target_combinations(df, mode='auto')
    >>>
    >>> # Check results
    >>> print(f"Mode: {meta['mode']}")
    Mode: auto
    >>> print(f"Target_7 period: {meta['method_7']['period']}")
    Target_7 period: 6
    >>> print(f"Target_7 threshold: {meta['method_7']['threshold']}%")
    Target_7 threshold: 4.0%
    >>> print(f"Positive instances: {targets['Target_7'].sum()}")
    Positive instances: 249

    **Example 2: Manual mode with custom lookback period**

    >>> # Generate targets with manual mode
    >>> targets, meta = generate_target_combinations(
    ...     df,
    ...     mode='manual',
    ...     lookback_periods=10,
    ...     min_pct=0,
    ...     max_pct=20,
    ...     step=1
    ... )
    >>>
    >>> # Check results for method 1
    >>> method_1 = meta['method_1']
    >>> print(f"Period: {method_1['period']}, Threshold: {method_1['threshold']}%")
    Period: 10, Threshold: 6.0%
    >>> print(f"Instances: {method_1['instances']} ({method_1['pct_of_max']:.1f}% of max)")
    Instances: 22 (1.4% of max)

    **Example 3: Target specific class balance**

    >>> # Aim for 30% positive instances
    >>> targets, meta = generate_target_combinations(
    ...     df,
    ...     mode='auto',
    ...     target_class_balance=0.3,
    ...     min_period=1,
    ...     max_period=15
    ... )
    >>>
    >>> # Verify class balance for each target
    >>> for i in range(1, 9):
    ...     positive_pct = targets[f'Target_{i}'].sum() / len(targets) * 100
    ...     print(f"Target_{i}: {positive_pct:.1f}% positive")
    Target_1: 29.5% positive
    Target_2: 30.2% positive
    ...

    **Example 4: Custom column names**

    >>> # DataFrame with different column names
    >>> df_custom = df.rename(columns={'Close': 'close_price', 'High': 'high_price'})
    >>> targets, meta = generate_target_combinations(
    ...     df_custom,
    ...     mode='auto',
    ...     close_col='close_price',
    ...     high_col='high_price'
    ... )
    >>> print(targets.columns.tolist())
    ['Target_1', 'Target_2', 'Target_3', 'Target_4', 'Target_5', 'Target_6', 'Target_7', 'Target_8']

    References
    ----------
    .. [1] Pareto, V. (1906). "Manuale di economia politica"
    .. [2] Satopää, V., et al. (2011). "Finding a 'Kneedle' in a Haystack:
           Detecting Knee Points in System Behavior"
    """
    # 1. Validate inputs
    _validate_inputs(df, mode, close_col, high_col)

    if mode == 'auto':
        # 2a. Auto mode: Pareto optimization
        optimal_params = _find_optimal_params_pareto(
            df, target_class_balance, min_period, max_period,
            period_step, min_pct, max_pct, step, close_col, high_col
        )
    else:  # mode == 'manual'
        # 2b. Manual mode: Fixed period, find elbow
        optimal_params = _find_optimal_params_elbow(
            df, lookback_periods, min_pct, max_pct, step, close_col, high_col
        )

    # 3. Generate 8 target columns using optimal params
    targets_df = _generate_targets(df, optimal_params, close_col, high_col)

    # 4. Create metadata dict
    metadata = {
        'mode': mode,
        **optimal_params  # Contains method_1 through method_8 dicts
    }

    return targets_df, metadata


def _validate_inputs(df: pd.DataFrame, mode: str, close_col: str, high_col: str) -> None:
    """
    Validate input parameters for target generation.

    Ensures the DataFrame is not empty, contains required columns, and the mode
    parameter is valid before proceeding with target generation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate.
    mode : str
        Optimization mode to validate. Must be 'auto' or 'manual'.
    close_col : str
        Name of the close price column to check for existence.
    high_col : str
        Name of the high price column to check for existence.

    Raises
    ------
    ValueError
        If DataFrame is empty.
    ValueError
        If `close_col` not found in DataFrame columns.
    ValueError
        If `high_col` not found in DataFrame columns.
    ValueError
        If `mode` is not 'auto' or 'manual'.

    Notes
    -----
    This is a helper function that performs early validation to provide clear
    error messages before expensive computation begins.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'Close': [100, 101], 'High': [102, 103]})
    >>> _validate_inputs(df, 'auto', 'Close', 'High')  # No error
    >>> _validate_inputs(df, 'invalid', 'Close', 'High')  # Raises ValueError
    Traceback (most recent call last):
        ...
    ValueError: mode must be 'auto' or 'manual', got 'invalid'
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    if close_col not in df.columns:
        raise ValueError(f"Column '{close_col}' not found in DataFrame")

    if high_col not in df.columns:
        raise ValueError(f"Column '{high_col}' not found in DataFrame")

    if mode not in ['auto', 'manual']:
        raise ValueError(f"mode must be 'auto' or 'manual', got '{mode}'")


def _calculate_future_values(
    df: pd.DataFrame,
    period: int,
    close_col: str,
    high_col: str
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate future price values and maximum values over the lookback period.

    This function computes four forward-looking price series that are used to
    create the eight target definitions. It handles both point-in-time future
    values (at period N) and maximum values over the entire period (1 to N).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing price data.
    period : int
        Number of periods to look forward. Must be >= 1.
    close_col : str
        Name of the close price column.
    high_col : str
        Name of the high price column.

    Returns
    -------
    future_close : pd.Series
        Close price at time t+period. Shape matches input df.
        Last `period` values will be NaN due to insufficient future data.
    future_high : pd.Series
        High price at time t+period. Shape matches input df.
        Last `period` values will be NaN.
    future_max_close : pd.Series
        Maximum close price over the window [t+1, t+period].
        Last `period` values will be NaN.
    future_max_high : pd.Series
        Maximum high price over the window [t+1, t+period].
        Last `period` values will be NaN.

    Notes
    -----
    **Shifting Logic:**

    - `shift(-period)` moves values backward in time, making future values
      available at current time index.
    - For max values, we shift forward, calculate rolling max, then shift back
      to align the window correctly with [t+1, t+period].

    **Rolling Window:**

    The rolling window calculation uses `min_periods=1` to handle edge cases
    at the start of the series, but in practice, the initial values are not
    used due to the shifting operations.

    **Memory Efficiency:**

    The function creates four new Series objects but does not copy the entire
    DataFrame, making it efficient for large datasets.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'Close': [100, 102, 101, 105, 103, 107],
    ...     'High': [101, 103, 102, 106, 104, 108]
    ... })
    >>> future_close, future_high, future_max_close, future_max_high = \\
    ...     _calculate_future_values(df, period=2, close_col='Close', high_col='High')
    >>>
    >>> # future_close is Close shifted back by 2
    >>> print(future_close.values)
    [101. 105. 103. 107.  nan  nan]
    >>>
    >>> # future_max_close is max(Close) over next 2 periods
    >>> print(future_max_close.values)
    [102. 105. 105. 107.  nan  nan]

    See Also
    --------
    pd.Series.shift : Shift index by desired number of periods.
    pd.Series.rolling : Provide rolling window calculations.
    """
    future_close = df[close_col].shift(-period)
    future_high = df[high_col].shift(-period)

    # For max values: shift forward, calculate rolling max, then shift back
    future_max_close = df[close_col].shift(-period).rolling(
        window=period, min_periods=1
    ).max().shift(period)

    future_max_high = df[high_col].shift(-period).rolling(
        window=period, min_periods=1
    ).max().shift(period)

    return future_close, future_high, future_max_close, future_max_high


def _find_optimal_params_pareto(
    df: pd.DataFrame,
    target_class_balance: float,
    min_period: int,
    max_period: int,
    period_step: int,
    min_pct: int,
    max_pct: int,
    step: int,
    close_col: str,
    high_col: str
) -> Dict:
    """
    Find optimal parameters using Pareto multi-objective optimization.

    This function searches the parameter space of (period, threshold) combinations
    to find Pareto-optimal solutions that balance three competing objectives:
    maximize threshold, minimize period, and minimize deviation from target class
    balance. From the Pareto-optimal set, it selects the solution closest to the
    desired class balance for each of the 8 target methods.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing OHLC price data.
    target_class_balance : float
        Target proportion of positive class instances (0.0 to 1.0).
        For example, 0.5 means aim for 50% positive instances.
    min_period : int
        Minimum lookback period to consider. Must be >= 1.
    max_period : int
        Maximum lookback period to consider. Must be > min_period.
    period_step : int
        Increment for period search. Must be >= 1.
    min_pct : int
        Minimum threshold percentage (e.g., 0 for 0%).
    max_pct : int
        Maximum threshold percentage (e.g., 100 for 100%).
    step : int
        Increment for threshold search in percentage points.
    close_col : str
        Name of the close price column.
    high_col : str
        Name of the high price column.

    Returns
    -------
    optimal_params : dict
        Dictionary with keys 'method_1' through 'method_8', where each value
        is a dictionary containing:

        - 'period' : int
            Optimal lookback period.
        - 'threshold' : float
            Optimal threshold as percentage (e.g., 5.0 for 5%).
        - 'instances' : int
            Number of positive instances at optimal point.
        - 'pct_of_max' : float
            Percentage of maximum possible instances (at 0% threshold).

    Notes
    -----
    **Pareto Optimization:**

    A solution (period, threshold) is Pareto-optimal if no other solution exists
    that is better in at least one objective and no worse in all others.

    Mathematically, solution A dominates solution B if:

    .. math::

        \\forall i: f_i(A) \\geq f_i(B) \\text{ and } \\exists j: f_j(A) > f_j(B)

    where :math:`f_i` are the objective functions (with appropriate sense).

    **Objectives:**

    1. **Maximize threshold**: Higher thresholds mean more stringent requirements,
       leading to higher precision predictions.

       .. math::

           \\text{maximize } \\theta

    2. **Minimize period**: Shorter holding periods reduce risk and capital
       allocation time.

       .. math::

           \\text{minimize } N

    3. **Minimize deviation from target balance**: Stay close to the desired
       class distribution.

       .. math::

           \\text{minimize } |\\text{instances}(\\theta, N) - \\text{target}|

    **Two-Pass Algorithm:**

    1. **First pass**: Calculate maximum instances at 0% threshold for each period
       to establish the upper bound and compute target instance counts.

    2. **Second pass**: Evaluate all (period, threshold) combinations, storing
       results for each method along with deviation from target.

    3. **Selection**: Apply Pareto dominance to find optimal set, then select
       solution closest to target class balance.

    **Computational Complexity:**

    Time complexity: O(P × T × M) where:

    - P = number of periods = (max_period - min_period) / period_step
    - T = number of thresholds = (max_pct - min_pct) / step
    - M = number of methods = 8

    Space complexity: O(P × T × M) to store all candidate solutions.

    **Common Pitfalls:**

    - If target_class_balance is too low (e.g., < 0.01), the search may not
      find feasible solutions within the threshold range.
    - Large search spaces (many periods and thresholds) increase computation time.
    - Insufficient data relative to max_period can lead to poor statistics.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     'Close': 100 + np.cumsum(np.random.randn(100)),
    ...     'High': 102 + np.cumsum(np.random.randn(100))
    ... })
    >>> optimal = _find_optimal_params_pareto(
    ...     df, target_class_balance=0.5, min_period=1, max_period=10,
    ...     period_step=1, min_pct=0, max_pct=20, step=1,
    ...     close_col='Close', high_col='High'
    ... )
    >>> print(optimal['method_1'])
    {'period': 5, 'threshold': 7.0, 'instances': 48, 'pct_of_max': 49.8}

    See Also
    --------
    paretoset : Library for computing Pareto-optimal sets.
    _find_optimal_params_elbow : Alternative elbow method for manual mode.

    References
    ----------
    .. [1] Pareto, V. (1906). "Manuale di economia politica"
    .. [2] Deb, K., et al. (2002). "A fast and elitist multiobjective genetic
           algorithm: NSGA-II"
    """
    # First pass: find maximum instances for each method at 0% threshold
    max_instances = [0] * 8

    for period in range(min_period, max_period + 1, period_step):
        future_close, future_high, future_max_close, future_max_high = \
            _calculate_future_values(df, period, close_col, high_col)

        # Calculate max instances for each method at threshold=0
        method_instances = [
            (future_close / df[close_col] - 1 >= 0).sum(),
            (future_close / df[high_col] - 1 >= 0).sum(),
            (future_high / df[close_col] - 1 >= 0).sum(),
            (future_high / df[high_col] - 1 >= 0).sum(),
            (future_max_close / df[close_col] - 1 >= 0).sum(),
            (future_max_close / df[high_col] - 1 >= 0).sum(),
            (future_max_high / df[close_col] - 1 >= 0).sum(),
            (future_max_high / df[high_col] - 1 >= 0).sum()
        ]

        for i in range(8):
            max_instances[i] = max(max_instances[i], method_instances[i])

    # Calculate target instances for each method
    target_instances = [int(mi * target_class_balance) for mi in max_instances]

    # Second pass: collect all combinations for each method
    results = {f'method_{i+1}': [] for i in range(8)}

    for period in range(min_period, max_period + 1, period_step):
        future_close, future_high, future_max_close, future_max_high = \
            _calculate_future_values(df, period, close_col, high_col)

        for threshold_pct in range(min_pct, max_pct, step):
            threshold = threshold_pct / 100

            # Calculate instances for all 8 methods
            instances = [
                (future_close / df[close_col] - 1 >= threshold).sum(),
                (future_close / df[high_col] - 1 >= threshold).sum(),
                (future_high / df[close_col] - 1 >= threshold).sum(),
                (future_high / df[high_col] - 1 >= threshold).sum(),
                (future_max_close / df[close_col] - 1 >= threshold).sum(),
                (future_max_close / df[high_col] - 1 >= threshold).sum(),
                (future_max_high / df[close_col] - 1 >= threshold).sum(),
                (future_max_high / df[high_col] - 1 >= threshold).sum()
            ]

            # Store results for each method
            for i in range(8):
                deviation = abs(instances[i] - target_instances[i])
                pct_of_max = (instances[i] / max_instances[i] * 100) if max_instances[i] > 0 else 0

                results[f'method_{i+1}'].append({
                    'period': period,
                    'threshold': threshold_pct,
                    'instances': instances[i],
                    'pct_of_max': pct_of_max,
                    'deviation': deviation
                })

    # Find Pareto-optimal solution for each method
    optimal_params = {}

    for method_idx in range(8):
        method_key = f'method_{method_idx + 1}'
        method_results = pd.DataFrame(results[method_key])

        # Pareto optimization: maximize threshold, minimize period, minimize deviation
        data = method_results[['threshold', 'period', 'deviation']].values
        mask = paretoset(data, sense=["max", "min", "min"])
        pareto_df = method_results[mask].copy()

        # Select solution closest to target_class_balance
        pareto_df['abs_dev_from_target'] = (
            pareto_df['pct_of_max'] / 100 - target_class_balance
        ).abs()

        best_solution = pareto_df.loc[pareto_df['abs_dev_from_target'].idxmin()]

        optimal_params[method_key] = {
            'period': int(best_solution['period']),
            'threshold': float(best_solution['threshold']),
            'instances': int(best_solution['instances']),
            'pct_of_max': float(best_solution['pct_of_max'])
        }

    return optimal_params


def _find_optimal_params_elbow(
    df: pd.DataFrame,
    lookback_periods: int,
    min_pct: int,
    max_pct: int,
    step: int,
    close_col: str,
    high_col: str
) -> Dict:
    """
    Find optimal thresholds using the elbow method with a fixed lookback period.

    The elbow method identifies the "knee" or "elbow" point on the curve of
    instance counts versus threshold percentages. This point represents an optimal
    trade-off where increasing the threshold further results in disproportionately
    fewer positive instances, while decreasing it provides diminishing returns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing OHLC price data.
    lookback_periods : int
        Fixed number of periods to look forward. Must be >= 1 and < len(df).
    min_pct : int
        Minimum threshold percentage (e.g., 0 for 0%).
    max_pct : int
        Maximum threshold percentage (e.g., 100 for 100%).
    step : int
        Increment for threshold search in percentage points. Must be >= 1.
    close_col : str
        Name of the close price column.
    high_col : str
        Name of the high price column.

    Returns
    -------
    optimal_params : dict
        Dictionary with keys 'method_1' through 'method_8', where each value
        is a dictionary containing:

        - 'period' : int
            Lookback period (same value `lookback_periods` for all methods).
        - 'threshold' : float
            Elbow threshold as percentage (e.g., 6.0 for 6%).
        - 'instances' : int
            Number of positive instances at the elbow point.
        - 'pct_of_max' : float
            Percentage of maximum possible instances (at 0% threshold).

    Notes
    -----
    **Elbow Method:**

    The elbow method finds the point of maximum curvature on a curve. For a
    decreasing convex curve (instances vs. threshold), the elbow represents
    the threshold where:

    - Below the elbow: Small threshold increases cause gradual instance decreases
    - Above the elbow: Small threshold increases cause steep instance decreases

    **Mathematical Formulation:**

    Given points on curve :math:`(x_i, y_i)` for thresholds :math:`x_i` and
    instance counts :math:`y_i`, the elbow maximizes the perpendicular distance
    from the curve to the line connecting the endpoints.

    For a line from :math:`(x_0, y_0)` to :math:`(x_n, y_n)`, the distance from
    point :math:`(x_i, y_i)` is:

    .. math::

        d_i = \\frac{|(y_n - y_0)x_i - (x_n - x_0)y_i + x_n y_0 - y_n x_0|}
              {\\sqrt{(y_n - y_0)^2 + (x_n - x_0)^2}}

    The elbow is at :math:`\\arg\\max_i d_i`.

    **KneeLocator Implementation:**

    This function uses the `kneed` library's `KneeLocator` class with:

    - `curve='convex'`: The instance count curve is convex (curves downward)
    - `direction='decreasing'`: Instance counts decrease as threshold increases

    **Fallback Behavior:**

    If no elbow is detected (e.g., monotonic curve with no clear knee), the
    function defaults to `min_pct` to ensure a valid result.

    **Advantages:**

    - Simple and interpretable method
    - No need to specify target class balance
    - Automatic threshold selection based on curve shape

    **Limitations:**

    - Fixed lookback period (no period optimization)
    - May not find elbow if curve is too smooth or too noisy
    - Assumes convex decreasing relationship

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     'Close': 100 + np.cumsum(np.random.randn(100)),
    ...     'High': 102 + np.cumsum(np.random.randn(100))
    ... })
    >>> optimal = _find_optimal_params_elbow(
    ...     df, lookback_periods=5, min_pct=0, max_pct=50, step=1,
    ...     close_col='Close', high_col='High'
    ... )
    >>> print(optimal['method_1'])
    {'period': 5, 'threshold': 6.0, 'instances': 22, 'pct_of_max': 22.4}
    >>>
    >>> # All methods use the same period
    >>> periods = [optimal[f'method_{i}']['period'] for i in range(1, 9)]
    >>> print(all(p == 5 for p in periods))
    True

    See Also
    --------
    KneeLocator : Knee/elbow detection algorithm from kneed library.
    _find_optimal_params_pareto : Pareto optimization for auto mode.

    References
    ----------
    .. [1] Satopää, V., et al. (2011). "Finding a 'Kneedle' in a Haystack:
           Detecting Knee Points in System Behavior"
    .. [2] Zhao, Q., et al. (2008). "Knee Point Detection in BIC for Detecting
           the Number of Clusters"
    """
    # Calculate future values for fixed period
    future_close, future_high, future_max_close, future_max_high = \
        _calculate_future_values(df, lookback_periods, close_col, high_col)

    # Calculate instance counts across threshold range for all 8 methods
    x = np.array(range(min_pct, max_pct, step))

    methods_data = [
        [(future_close / df[close_col] - 1 >= i / 100).sum() for i in range(min_pct, max_pct, step)],
        [(future_close / df[high_col] - 1 >= i / 100).sum() for i in range(min_pct, max_pct, step)],
        [(future_high / df[close_col] - 1 >= i / 100).sum() for i in range(min_pct, max_pct, step)],
        [(future_high / df[high_col] - 1 >= i / 100).sum() for i in range(min_pct, max_pct, step)],
        [(future_max_close / df[close_col] - 1 >= i / 100).sum() for i in range(min_pct, max_pct, step)],
        [(future_max_close / df[high_col] - 1 >= i / 100).sum() for i in range(min_pct, max_pct, step)],
        [(future_max_high / df[close_col] - 1 >= i / 100).sum() for i in range(min_pct, max_pct, step)],
        [(future_max_high / df[high_col] - 1 >= i / 100).sum() for i in range(min_pct, max_pct, step)]
    ]

    # Find elbow points for all methods
    optimal_params = {}

    for method_idx, pct_lst in enumerate(methods_data):
        method_key = f'method_{method_idx + 1}'

        # Find elbow using KneeLocator
        kn = KneeLocator(x, pct_lst, curve='convex', direction='decreasing')
        elbow_threshold = kn.elbow if kn.elbow is not None else min_pct

        # Get instances at elbow
        elbow_idx = int((elbow_threshold - min_pct) / step)
        instances = pct_lst[elbow_idx] if elbow_idx < len(pct_lst) else 0

        # Calculate max instances (at threshold=0)
        max_inst = pct_lst[0] if len(pct_lst) > 0 else 1
        pct_of_max = (instances / max_inst * 100) if max_inst > 0 else 0

        optimal_params[method_key] = {
            'period': lookback_periods,
            'threshold': float(elbow_threshold),
            'instances': int(instances),
            'pct_of_max': float(pct_of_max)
        }

    return optimal_params


def _generate_targets(
    df: pd.DataFrame,
    optimal_params: Dict,
    close_col: str,
    high_col: str
) -> pd.DataFrame:
    """
    Generate eight binary target columns using optimized parameters.

    Creates boolean target columns based on whether future price gains exceed
    specified thresholds. Each target uses its own optimized period and threshold
    determined by either Pareto optimization or the elbow method.

    Parameters
    ----------
    df : pd.DataFrame
        Original input DataFrame containing price data.
    optimal_params : dict
        Dictionary with keys 'method_1' through 'method_8', where each value
        contains 'period' and 'threshold' parameters.
    close_col : str
        Name of the close price column.
    high_col : str
        Name of the high price column.

    Returns
    -------
    targets_df : pd.DataFrame
        DataFrame with same index as input `df`, containing 8 boolean columns
        named 'Target_1' through 'Target_8'. Each column has True where the
        gain exceeds the threshold, False otherwise, and NaN for rows without
        sufficient future data.

    Notes
    -----
    **Target Formulas:**

    For time t, lookback period N, threshold :math:`\\theta`:

    - **Target_1**: :math:`\\frac{\\text{Close}[t+N]}{\\text{Close}[t]} - 1 \\geq \\theta`
    - **Target_2**: :math:`\\frac{\\text{Close}[t+N]}{\\text{High}[t]} - 1 \\geq \\theta`
    - **Target_3**: :math:`\\frac{\\text{High}[t+N]}{\\text{Close}[t]} - 1 \\geq \\theta`
    - **Target_4**: :math:`\\frac{\\text{High}[t+N]}{\\text{High}[t]} - 1 \\geq \\theta`
    - **Target_5**: :math:`\\frac{\\max_{i=1}^{N}(\\text{Close}[t+i])}{\\text{Close}[t]} - 1 \\geq \\theta`
    - **Target_6**: :math:`\\frac{\\max_{i=1}^{N}(\\text{Close}[t+i])}{\\text{High}[t]} - 1 \\geq \\theta`
    - **Target_7**: :math:`\\frac{\\max_{i=1}^{N}(\\text{High}[t+i])}{\\text{Close}[t]} - 1 \\geq \\theta`
    - **Target_8**: :math:`\\frac{\\max_{i=1}^{N}(\\text{High}[t+i])}{\\text{High}[t]} - 1 \\geq \\theta`

    **Entry Price Interpretation:**

    - Close[t]: Assume entry at closing price of day t
    - High[t]: Assume entry at intraday high of day t (more conservative)

    **Exit Price Interpretation:**

    - Point targets (1-4): Exit at specific time t+N
    - Maximum targets (5-8): Exit at any optimal time in [t+1, t+N]

    **NaN Handling:**

    The last N rows of each target will contain NaN values because there is
    insufficient future data to calculate the gains. These rows should be
    excluded from training and evaluation.

    **Memory Efficiency:**

    Each target calculation recomputes future values for its specific period,
    which allows different periods per target but requires multiple passes
    over the data. For large datasets, consider caching intermediate results.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'Close': [100, 102, 101, 105, 103, 107, 106, 110],
    ...     'High': [101, 103, 102, 106, 104, 108, 107, 111]
    ... })
    >>> optimal_params = {
    ...     'method_1': {'period': 2, 'threshold': 5.0},
    ...     'method_2': {'period': 2, 'threshold': 5.0},
    ...     'method_3': {'period': 2, 'threshold': 5.0},
    ...     'method_4': {'period': 2, 'threshold': 5.0},
    ...     'method_5': {'period': 3, 'threshold': 7.0},
    ...     'method_6': {'period': 3, 'threshold': 7.0},
    ...     'method_7': {'period': 3, 'threshold': 7.0},
    ...     'method_8': {'period': 3, 'threshold': 7.0}
    ... }
    >>> targets = _generate_targets(df, optimal_params, 'Close', 'High')
    >>> print(targets.columns.tolist())
    ['Target_1', 'Target_2', 'Target_3', 'Target_4', 'Target_5', 'Target_6', 'Target_7', 'Target_8']
    >>> print(targets['Target_1'].dtype)
    bool
    >>>
    >>> # Check target calculation for first row
    >>> # Close[2]/Close[0] = 101/100 = 1.01, gain = 1%
    >>> # 1% < 5% threshold, so False
    >>> print(targets['Target_1'].iloc[0])
    False

    See Also
    --------
    _calculate_future_values : Computes future price series.
    """
    targets = {}

    for method_idx in range(8):
        method_key = f'method_{method_idx + 1}'
        params = optimal_params[method_key]
        period = params['period']
        threshold = params['threshold'] / 100  # Convert percentage to decimal

        # Calculate future values for this method's period
        future_close, future_high, future_max_close, future_max_high = \
            _calculate_future_values(df, period, close_col, high_col)

        # Generate target based on method
        if method_idx == 0:  # Method 1: Close[N]/Close[0]
            targets[f'Target_{method_idx + 1}'] = (future_close / df[close_col] - 1 >= threshold)
        elif method_idx == 1:  # Method 2: Close[N]/High[0]
            targets[f'Target_{method_idx + 1}'] = (future_close / df[high_col] - 1 >= threshold)
        elif method_idx == 2:  # Method 3: High[N]/Close[0]
            targets[f'Target_{method_idx + 1}'] = (future_high / df[close_col] - 1 >= threshold)
        elif method_idx == 3:  # Method 4: High[N]/High[0]
            targets[f'Target_{method_idx + 1}'] = (future_high / df[high_col] - 1 >= threshold)
        elif method_idx == 4:  # Method 5: MaxClose/Close[0]
            targets[f'Target_{method_idx + 1}'] = (future_max_close / df[close_col] - 1 >= threshold)
        elif method_idx == 5:  # Method 6: MaxClose/High[0]
            targets[f'Target_{method_idx + 1}'] = (future_max_close / df[high_col] - 1 >= threshold)
        elif method_idx == 6:  # Method 7: MaxHigh/Close[0]
            targets[f'Target_{method_idx + 1}'] = (future_max_high / df[close_col] - 1 >= threshold)
        elif method_idx == 7:  # Method 8: MaxHigh/High[0]
            targets[f'Target_{method_idx + 1}'] = (future_max_high / df[high_col] - 1 >= threshold)

    return pd.DataFrame(targets, index=df.index)

"""
Visualization module for stock prediction analysis.

This module provides a pandas DataFrame accessor for creating publication-quality
visualizations of stock prediction models. It includes methods for plotting buy/sell
signals, confusion matrices, and performance metrics overlaid on price charts.

The visualizations are designed to help interpret model predictions, identify
false positives/negatives, and understand prediction quality at a glance.

Examples
--------
Basic usage with the DataFrame accessor:

>>> import pandas as pd
>>> import rhoa
>>> df = rhoa.read_csv('stock_data.csv')
>>> df['Date'] = pd.to_datetime(df['Date'])
>>> # Visualize predictions
>>> fig = df.rhoa.plots.signal(y_pred=predictions, y_true=targets)

Notes
-----
All plotting methods return matplotlib Figure objects that can be further
customized using standard matplotlib/seaborn APIs.

See Also
--------
rhoa.targets : Module for generating target labels for stock prediction

"""
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, Tuple


class PlotsAccessor:
    """
    Pandas DataFrame accessor for stock prediction visualization.

    This accessor provides methods for creating professional visualizations of
    stock prediction model results. It is automatically available on any pandas
    DataFrame through the `.plots` attribute after importing rhoa.

    The accessor specializes in visualizing:
    - Buy/sell signals overlaid on price charts
    - Confusion matrices for classification performance
    - False positive and false negative analysis
    - Model performance metrics (precision, recall)

    Attributes
    ----------
    _df : pandas.DataFrame
        The DataFrame instance this accessor is attached to. Should contain
        stock price data with at minimum a date column and price column.

    Examples
    --------
    Access the plotting methods through any DataFrame:

    >>> import pandas as pd
    >>> import rhoa
    >>> df = rhoa.read_csv('stock_data.csv')
    >>> # The .plots accessor is now available
    >>> fig = df.rhoa.plots.signal(y_pred=predictions)

    Notes
    -----
    This accessor is registered automatically when rhoa is imported. The
    DataFrame must contain the columns specified in each method's parameters
    (typically 'Date' and 'Close' at minimum).

    See Also
    --------
    rhoa.targets : Generate target labels for predictions
    pandas.api.extensions.register_dataframe_accessor : pandas accessor registration

    """

    def __init__(self, pandas_obj):
        """
        Initialize the plots accessor.

        Parameters
        ----------
        pandas_obj : pandas.DataFrame
            The DataFrame instance to attach the accessor to.

        """
        self._df = pandas_obj

    def signal(
            self,
            y_pred: Union[np.ndarray, pd.Series],
            y_true: Optional[Union[np.ndarray, pd.Series]] = None,
            date_col: str = 'Date',
            price_col: str = 'Close',
            threshold: Optional[float] = None,
            title: Optional[str] = None,
            figsize: Tuple[int, int] = (18, 10),
            cmap: str = 'Blues',
            save_path: Optional[str] = None,
            dpi: int = 300,
            show: bool = True,
            **kwargs
    ) -> plt.Figure:
        """
        Visualize stock predictions with buy signals overlaid on price chart.

        Creates a publication-quality visualization showing model predictions on a
        stock price chart. The visualization includes the price time series with
        predicted buy signals marked as bright green dots. When ground truth labels
        are provided, the plot includes a confusion matrix panel and marks false
        positives (red X) and false negatives (orange circles) on the chart.

        This method is designed for evaluating machine learning models for stock
        trading by providing immediate visual feedback on prediction quality,
        timing of signals, and model errors in temporal context.

        Parameters
        ----------
        y_pred : array-like of shape (n_samples,)
            Binary predictions array (0 or 1) indicating predicted buy signals.
            Must have the same length as the DataFrame. Typically output from
            a classification model like RandomForest, XGBoost, or neural network.
            Values should be 0 (no buy signal) or 1 (buy signal).
        y_true : array-like of shape (n_samples,), optional
            Ground truth binary labels (0 or 1) for validation. When provided,
            the visualization adds:
            - A confusion matrix panel showing TP, FP, TN, FN
            - Light green background dots for true buy opportunities
            - Red X markers for false positive predictions
            - Orange circles for missed opportunities (false negatives)
            Default is None (predictions only, no validation).
        date_col : str, default='Date'
            Name of the date/datetime column in the DataFrame. This column is
            used for the x-axis of the price chart. The column should be of
            datetime type or convertible to datetime.
        price_col : str, default='Close'
            Name of the price column to plot on the y-axis. Typically 'Close'
            for closing prices, but can be 'Open', 'High', 'Low', or any other
            price column. The signals will be plotted at this price level.
        threshold : float, optional
            The prediction probability threshold that was used to generate y_pred.
            This is displayed in the plot title for reference. For example, if
            you used `y_pred = (y_pred_proba >= 0.67).astype(int)`, set
            threshold=0.67. Default is None (threshold not shown).
        title : str, optional
            Custom title prefix for the plot. If None, generates an automatic
            title based on the price_col. The title will be expanded with
            " - Confusion Matrix" or " - Price with Buy Signals" depending on
            the panel. Default is None.
        figsize : tuple of int, default=(18, 10)
            Figure size as (width, height) in inches. Large default size ensures
            readability with multiple panels and detailed annotations. Adjust for
            different display sizes or publication requirements.
        cmap : str, default='Blues'
            Matplotlib colormap name for the confusion matrix heatmap. Options:
            - 'Blues': Professional blue gradient (recommended)
            - 'Greens': Green gradient for emphasizing positive signals
            - 'Reds': Red gradient (use cautiously)
            - Any valid matplotlib colormap name
        save_path : str, optional
            File path to save the figure. Supports common formats: .png, .jpg,
            .pdf, .svg. If None, the figure is not saved to disk. Directory
            must exist. Default is None.
        dpi : int, default=300
            Resolution (dots per inch) for saved figures. 300 dpi is suitable
            for publications. Use 150 for presentations, 600+ for print.
            Only applies when save_path is provided.
        show : bool, default=True
            Whether to display the plot using plt.show(). Set to False if you
            want to further customize the figure or prevent display in non-
            interactive environments.
        **kwargs : dict, optional
            Additional keyword arguments passed to matplotlib for advanced
            customization. Currently not used but reserved for future extensions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure object containing one or two subplots depending
            on whether y_true is provided. Can be further customized using
            standard matplotlib methods or saved to different formats.

        Raises
        ------
        ValueError
            If the specified date_col is not found in the DataFrame.
        ValueError
            If the specified price_col is not found in the DataFrame.
        ValueError
            If the length of y_pred does not match the DataFrame length.
        ValueError
            If y_true is provided but its length does not match the DataFrame.

        See Also
        --------
        rhoa.targets.future_return : Generate target labels based on future returns
        rhoa.targets.drawdown : Generate target labels based on drawdown thresholds
        matplotlib.pyplot.savefig : Save the figure to file
        sklearn.metrics.confusion_matrix : Compute confusion matrix

        Notes
        -----
        **Visualization Components:**

        When y_true is None (predictions only):
        - Single panel showing price chart with predicted buy signals (bright green)
        - No confusion matrix or error markers

        When y_true is provided (validation mode):
        - Top panel: Confusion matrix with counts, percentages, and metrics
        - Bottom panel: Price chart with all signal types:
          * Light green background: True buy opportunities (ground truth)
          * Bright green dots: Model predicted buy signals
          * Red X markers: False positives (wrong predictions)
          * Orange circles: False negatives (missed opportunities)

        **Interpreting the Visualization:**

        - **Dense bright green clusters**: Model is actively predicting buy signals
        - **Green dots on light green background**: True positives (correct predictions)
        - **Red X markers**: False alarms - model predicted buy but shouldn't have
        - **Orange circles**: Missed trades - model failed to predict real opportunities
        - **Gaps in signals**: Periods where model predicts no buy opportunities

        **Best Practices:**

        1. Always provide y_true during model development for full diagnostics
        2. Look for temporal patterns in false positives (e.g., during volatility)
        3. Check if false negatives occur at specific price levels or market conditions
        4. Compare precision/recall from confusion matrix with your trading strategy
        5. Use threshold parameter to document the decision boundary
        6. Save high-quality versions (dpi=300+) for documentation

        **Performance Metrics:**

        The confusion matrix panel displays:
        - Precision: Of all buy signals, what percentage were correct?
        - Recall: Of all true opportunities, what percentage did we catch?
        - Counts: Absolute numbers of TP, FP, TN, FN

        Examples
        --------
        Visualize predictions with full validation metrics:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import rhoa
        >>>
        >>> # Load stock data
        >>> df = pd.read_csv('AAPL_stock_data.csv')
        >>> df['Date'] = pd.to_datetime(df['Date'])
        >>>
        >>> # Generate targets using rhoa (7% return threshold)
        >>> targets = df.targets.future_return(
        ...     threshold=0.07,
        ...     holding_period=10,
        ...     return_type='pct'
        ... )
        >>>
        >>> # Assume you have model predictions
        >>> predictions = model.predict(features)
        >>>
        >>> # Create comprehensive visualization
        >>> fig = df.rhoa.plots.signal(
        ...     y_pred=predictions,
        ...     y_true=targets,
        ...     threshold=0.67,
        ...     title='AAPL Random Forest Model',
        ...     save_path='outputs/aapl_predictions.png',
        ...     dpi=300
        ... )

        Visualize predictions only (no ground truth available):

        >>> # When you don't have labels (e.g., predicting future)
        >>> fig = df.rhoa.plots.signal(
        ...     y_pred=predictions,
        ...     date_col='Date',
        ...     price_col='Close',
        ...     title='AAPL Future Predictions',
        ...     cmap='Greens'
        ... )

        Customize for different price columns:

        >>> # Use opening prices instead of closing
        >>> fig = df.rhoa.plots.signal(
        ...     y_pred=predictions,
        ...     y_true=targets,
        ...     price_col='Open',
        ...     title='Entry Signals (Open Prices)'
        ... )

        Save without displaying (batch processing):

        >>> # Useful for generating reports for multiple stocks
        >>> for ticker in ['AAPL', 'GOOGL', 'MSFT']:
        ...     df = load_stock_data(ticker)
        ...     predictions = model.predict(df)
        ...     fig = df.rhoa.plots.signal(
        ...         y_pred=predictions,
        ...         save_path=f'reports/{ticker}_signals.png',
        ...         show=False  # Don't display, just save
        ...     )
        ...     plt.close(fig)  # Free memory

        Compare different thresholds visually:

        >>> # Generate predictions at different thresholds
        >>> proba = model.predict_proba(features)[:, 1]
        >>>
        >>> for thresh in [0.5, 0.67, 0.8]:
        ...     preds = (proba >= thresh).astype(int)
        ...     fig = df.rhoa.plots.signal(
        ...         y_pred=preds,
        ...         y_true=targets,
        ...         threshold=thresh,
        ...         title=f'Model Threshold {thresh}',
        ...         save_path=f'outputs/threshold_{thresh}.png',
        ...         show=False
        ...     )

        """
        # Validate inputs
        self._validate_signal_inputs(y_pred, y_true, date_col, price_col)

        # Convert predictions to numpy arrays
        y_pred = np.asarray(y_pred)
        if y_true is not None:
            y_true = np.asarray(y_true)

        # Determine layout: 2 panels if y_true provided, 1 panel otherwise
        has_ground_truth = y_true is not None

        # Create figure
        fig = plt.figure(figsize=figsize)

        if has_ground_truth:
            # Two-panel layout: confusion matrix + price chart
            ax_confusion = plt.subplot(2, 1, 1)
            ax_price = plt.subplot(2, 1, 2)

            # Plot confusion matrix
            self._plot_confusion_matrix(
                ax_confusion, y_true, y_pred, threshold, cmap, title
            )
        else:
            # Single panel: price chart only
            ax_price = plt.subplot(1, 1, 1)

        # Plot price chart with signals
        self._plot_price_signals(
            ax_price, y_pred, y_true, date_col, price_col,
            threshold, title, has_ground_truth
        )

        # Apply tight layout
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

        # Show if requested
        if show:
            plt.show()

        return fig

    def _validate_signal_inputs(
            self,
            y_pred: Union[np.ndarray, pd.Series],
            y_true: Optional[Union[np.ndarray, pd.Series]],
            date_col: str,
            price_col: str
    ) -> None:
        """
        Validate inputs for the signal plotting method.

        Performs comprehensive validation of all input parameters to ensure they
        meet the requirements for creating the signal visualization. Checks for
        column existence, correct array lengths, and data integrity.

        Parameters
        ----------
        y_pred : array-like
            Binary predictions array to validate. Must have same length as DataFrame.
        y_true : array-like or None
            Ground truth labels to validate if provided. Must have same length as
            DataFrame when not None.
        date_col : str
            Name of the date column to check for existence in DataFrame.
        price_col : str
            Name of the price column to check for existence in DataFrame.

        Returns
        -------
        None
            This method returns nothing on success.

        Raises
        ------
        ValueError
            If date_col is not found in the DataFrame columns.
        ValueError
            If price_col is not found in the DataFrame columns.
        ValueError
            If the length of y_pred does not match the DataFrame length.
        ValueError
            If y_true is provided and its length does not match the DataFrame length.

        Notes
        -----
        This is an internal validation method called automatically by the signal()
        method before creating visualizations. It ensures data consistency and
        provides clear error messages for common input mistakes.

        Examples
        --------
        This method is called internally:

        >>> # These validations happen automatically in signal()
        >>> fig = df.rhoa.plots.signal(y_pred=predictions)
        >>> # Raises ValueError if predictions length doesn't match df

        """
        # Check required columns exist
        if date_col not in self._df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        if price_col not in self._df.columns:
            raise ValueError(f"Price column '{price_col}' not found in DataFrame")

        # Check prediction length matches DataFrame
        if len(y_pred) != len(self._df):
            raise ValueError(
                f"Length of y_pred ({len(y_pred)}) must match DataFrame length ({len(self._df)})"
            )

        # Check ground truth length if provided
        if y_true is not None and len(y_true) != len(self._df):
            raise ValueError(
                f"Length of y_true ({len(y_true)}) must match DataFrame length ({len(self._df)})"
            )

    def _plot_confusion_matrix(
            self,
            ax: plt.Axes,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            threshold: Optional[float],
            cmap: str,
            title: Optional[str]
    ) -> None:
        """
        Plot confusion matrix with custom annotations and metrics.

        Creates a detailed confusion matrix heatmap with counts, percentages,
        and classification metrics (precision and recall). The matrix uses
        custom annotations to show both absolute counts and row-normalized
        percentages for each cell. A metrics summary box is added to the right
        side of the matrix.

        This is an internal method called by signal() when ground truth labels
        are provided.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to plot the confusion matrix on. Should be from a
            subplot in the main figure.
        y_true : numpy.ndarray of shape (n_samples,)
            Ground truth binary labels (0 or 1). These form the rows of the
            confusion matrix.
        y_pred : numpy.ndarray of shape (n_samples,)
            Predicted binary labels (0 or 1). These form the columns of the
            confusion matrix.
        threshold : float or None
            The prediction threshold used to generate y_pred. If provided,
            displayed in the title. If None, threshold is not shown.
        cmap : str
            Matplotlib colormap name for the heatmap background. Common options
            are 'Blues', 'Greens', 'Reds'. The text color automatically adjusts
            for readability (white on dark cells, black on light cells).
        title : str or None
            Custom title prefix. If provided, the confusion matrix title becomes
            "{title} - Confusion Matrix". If None, just "Confusion Matrix" is used.

        Returns
        -------
        None
            This method modifies the provided axes object in-place and returns nothing.

        Notes
        -----
        **Matrix Layout:**

        The confusion matrix follows sklearn's convention:
        - Rows represent true labels (actual class)
        - Columns represent predicted labels (model output)
        - Cell [0,0]: True Negatives (TN) - correctly predicted no-buy
        - Cell [0,1]: False Positives (FP) - incorrectly predicted buy
        - Cell [1,0]: False Negatives (FN) - missed buy opportunities
        - Cell [1,1]: True Positives (TP) - correctly predicted buy

        **Annotations:**

        Each cell displays:
        - Top number: Absolute count of samples
        - Bottom number: Row-normalized percentage (e.g., "85.0%")

        **Metrics Displayed:**

        - Precision = TP / (TP + FP): Of all buy signals, how many were correct?
        - Recall = TP / (TP + FN): Of all true opportunities, how many were caught?
        - Total Signals: TP + FP (number of buy predictions)

        **Metrics Box:**

        A summary box on the right shows:
        - TP, FP, TN, FN counts
        - Total signals generated
        - Correct signals (precision percentage)
        - Wrong signals (false positive count)

        **Color Coding:**

        Text color automatically adapts based on cell background:
        - White text on dark cells (>50% normalized value)
        - Black text on light cells (<=50% normalized value)

        See Also
        --------
        sklearn.metrics.confusion_matrix : Compute confusion matrix
        sklearn.metrics.precision_score : Compute precision
        sklearn.metrics.recall_score : Compute recall
        seaborn.heatmap : Create annotated heatmaps

        Examples
        --------
        This method is called internally by signal():

        >>> # When y_true is provided, signal() creates confusion matrix
        >>> fig = df.rhoa.plots.signal(y_pred=predictions, y_true=targets)
        >>> # The confusion matrix is automatically plotted in the top panel

        """
        from sklearn.metrics import confusion_matrix as compute_cm

        # Compute confusion matrix
        cm = compute_cm(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Normalize for percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot heatmap
        sns.heatmap(cm, annot=False, fmt='d', cmap=cmap, ax=ax, cbar=False)

        # Add custom annotations with counts and percentages
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                percentage = cm_norm[i, j]
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                ax.text(
                    j + 0.5, i + 0.5,
                    f'{count}\n({percentage:.1%})',
                    ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color=color
                )

        # Calculate metrics
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

        # Set labels and title
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')

        if title:
            title_text = f'{title} - Confusion Matrix'
        else:
            title_text = 'Confusion Matrix'

        if threshold is not None:
            title_text += f'\nThreshold: {threshold:.2f} | Precision: {precision:.1f}% | Recall: {recall:.1f}%'
        else:
            title_text += f'\nPrecision: {precision:.1f}% | Recall: {recall:.1f}%'

        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticklabels(['No Buy (0)', 'Buy (1)'], fontsize=11)
        ax.set_yticklabels(['No Buy (0)', 'Buy (1)'], fontsize=11, rotation=0)

        # Add metrics text box
        metrics_text = f"""
True Positives (TP):  {tp:>4d}
False Positives (FP): {fp:>4d}
True Negatives (TN):  {tn:>4d}
False Negatives (FN): {fn:>4d}

Total Signals: {tp + fp}
Correct: {tp} ({precision:.1f}%)
Wrong: {fp}
"""
        ax.text(
            1.15, 0.5, metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace'
        )

    def _plot_price_signals(
            self,
            ax: plt.Axes,
            y_pred: np.ndarray,
            y_true: Optional[np.ndarray],
            date_col: str,
            price_col: str,
            threshold: Optional[float],
            title: Optional[str],
            has_ground_truth: bool
    ) -> None:
        """
        Plot price chart with overlaid buy signals and error markers.

        Creates a time series plot of stock prices with buy signal predictions
        overlaid as colored markers. When ground truth is provided, also displays
        true opportunities, false positives, and false negatives using a distinct
        color-coding system.

        This is the main visualization component that shows temporal patterns in
        predictions and helps identify when and where the model makes errors.

        This is an internal method called by signal() for the price chart panel.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to plot the price chart on. Should be from a subplot
            in the main figure (either the only panel or the bottom panel).
        y_pred : numpy.ndarray of shape (n_samples,)
            Binary predictions array (0 or 1) indicating predicted buy signals.
            Values of 1 are plotted as bright green markers on the chart.
        y_true : numpy.ndarray of shape (n_samples,) or None
            Ground truth binary labels (0 or 1). When provided, enables error
            visualization with false positive and false negative markers. When
            None, only predictions are shown.
        date_col : str
            Name of the date column in the DataFrame to use for the x-axis.
            Must be a valid column name in self._df.
        price_col : str
            Name of the price column in the DataFrame to use for the y-axis.
            Typically 'Close', but can be any price column. Must be a valid
            column name in self._df.
        threshold : float or None
            The prediction threshold used. If provided, appended to the plot
            title as "(Threshold X.XX)". If None, threshold is not shown.
        title : str or None
            Custom title prefix. If provided, becomes "{title} - Price with Buy
            Signals". If None, defaults to "{price_col} Price with Buy Signals".
        has_ground_truth : bool
            Flag indicating whether ground truth labels are provided. Used to
            determine whether to plot error markers. Should match whether y_true
            is not None.

        Returns
        -------
        None
            This method modifies the provided axes object in-place and returns nothing.

        Notes
        -----
        **Marker Color Scheme:**

        The visualization uses a carefully designed color scheme to make different
        signal types immediately distinguishable:

        - **Blue line**: Stock price time series (primary data)
          * Color: #2E86AB (professional blue)
          * Linewidth: 2 (prominent but not overwhelming)
          * Z-order: 1 (bottom layer)

        - **Light green background dots**: True buy opportunities (when y_true provided)
          * Color: 'lightgreen'
          * Size: 50 points
          * Alpha: 0.4 (semi-transparent to not obscure other markers)
          * Z-order: 2 (second layer)
          * Shows where the target labels indicate a buy opportunity

        - **Bright green dots**: Model predicted buy signals
          * Color: 'lime' with 'darkgreen' edge
          * Size: 120 points (largest for visibility)
          * Edge width: 2 (prominent border)
          * Z-order: 3 (third layer, on top of true opportunities)
          * These are the main predictions from your model

        - **Red X markers**: False positives (when y_true provided)
          * Color: 'red'
          * Marker: 'x'
          * Size: 300 points (very large for warning visibility)
          * Linewidth: 4 (thick for emphasis)
          * Z-order: 4 (top layer, highest priority)
          * Model predicted buy but it was actually not a good opportunity

        - **Orange circles**: False negatives / missed opportunities (when y_true provided)
          * Color: 'orange' with 'darkorange' edge
          * Marker: 'o'
          * Size: 80 points
          * Alpha: 0.6 (semi-transparent)
          * Edge width: 1
          * Z-order: 2.5 (between true opportunities and predictions)
          * Model failed to predict an actual buy opportunity

        **Legend Counts:**

        Each legend entry includes the count of that marker type:
        - "True Buy Opportunities (n=X)": Number of ground truth positives
        - "Model Buy Signals (n=X)": Number of predicted positives
        - "False Positives (n=X)": Number of incorrect buy predictions
        - "Missed Opportunities (n=X)": Number of missed true opportunities

        These counts help quickly assess model behavior (e.g., too many false
        positives indicates low precision, many missed opportunities indicates
        low recall).

        **Interpreting Patterns:**

        - **Clusters of bright green**: Model is active in predicting
        - **Green on light green**: Successful predictions (true positives)
        - **Red X on light green**: Model predicted incorrectly on true opportunity
        - **Red X on blue**: Model predicted on non-opportunity (false positive)
        - **Orange circles**: Opportunities the model completely missed
        - **Periods with no markers**: Model predicts no action

        **Visual Design:**

        - Grid enabled with alpha=0.3 for easy time/price reading
        - X-axis rotated 45 degrees for date readability
        - Bold labels and title for clarity
        - Legend positioned automatically at best location
        - Z-order ensures important markers (errors) are visible on top

        See Also
        --------
        matplotlib.pyplot.plot : Plot lines
        matplotlib.pyplot.scatter : Plot scatter markers
        matplotlib.axes.Axes.legend : Add legend to axes

        Examples
        --------
        This method is called internally by signal():

        >>> # The price chart is automatically created
        >>> fig = df.rhoa.plots.signal(y_pred=predictions, y_true=targets)
        >>> # Bottom panel shows price with all signal markers

        """
        df = self._df

        # Plot stock price
        ax.plot(
            df[date_col], df[price_col],
            linewidth=2, color='#2E86AB',
            label=f'{price_col} Price', zorder=1
        )

        # If ground truth provided, overlay true opportunities
        if y_true is not None:
            true_buys_mask = y_true == 1
            true_buys_df = df[true_buys_mask]
            n_opportunities = true_buys_mask.sum()

            ax.scatter(
                true_buys_df[date_col], true_buys_df[price_col],
                color='lightgreen', s=50, alpha=0.4,
                label=f'True Buy Opportunities (n={n_opportunities})',
                zorder=2
            )

        # Overlay predicted buy signals
        pred_buys_mask = y_pred == 1
        pred_buys_df = df[pred_buys_mask]
        n_signals = pred_buys_mask.sum()

        ax.scatter(
            pred_buys_df[date_col], pred_buys_df[price_col],
            color='lime', s=120, edgecolors='darkgreen', linewidths=2,
            label=f'Model Buy Signals (n={n_signals})',
            zorder=3
        )

        # If ground truth provided, mark false positives and false negatives
        if y_true is not None:
            # False positives: predicted buy (1) but actually no-buy (0)
            fp_mask = (y_pred == 1) & (y_true == 0)
            fp_df = df[fp_mask]
            n_fp = fp_mask.sum()

            if n_fp > 0:
                ax.scatter(
                    fp_df[date_col], fp_df[price_col],
                    color='red', marker='x', s=300, linewidths=4,
                    label=f'False Positives (n={n_fp})',
                    zorder=4
                )

            # False negatives: predicted no-buy (0) but actually buy (1)
            fn_mask = (y_pred == 0) & (y_true == 1)
            fn_df = df[fn_mask]
            n_fn = fn_mask.sum()

            if n_fn > 0:
                ax.scatter(
                    fn_df[date_col], fn_df[price_col],
                    color='orange', marker='o', s=80, alpha=0.6,
                    edgecolors='darkorange', linewidths=1,
                    label=f'Missed Opportunities (n={n_fn})',
                    zorder=2.5
                )

        # Set labels and title
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Price', fontsize=12, fontweight='bold')

        if title:
            title_text = f'{title} - Price with Buy Signals'
        else:
            title_text = f'{price_col} Price with Buy Signals'

        if threshold is not None:
            title_text += f' (Threshold {threshold:.2f})'

        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=15)

        # Add legend and grid
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.tick_params(axis='x', rotation=45)

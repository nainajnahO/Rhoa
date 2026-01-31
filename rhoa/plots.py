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
from pandas.api.extensions import register_dataframe_accessor
from typing import Optional, Union, Tuple


@register_dataframe_accessor("plots")
class PlotsAccessor:
    """Accessor for DataFrame plotting methods for stock prediction visualization."""

    def __init__(self, pandas_obj):
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
        """Visualize stock predictions with buy signals overlaid on price chart.
        
        Creates a professional visualization showing model predictions on a stock
        price chart. Optionally includes a confusion matrix when ground truth
        labels are provided. This method follows the same high-quality styling
        as the rhoa library's other visualization methods.
        
        Args:
            y_pred (array-like): Binary predictions array (0 or 1) indicating
                predicted buy signals. Must have same length as DataFrame.
            y_true (array-like, optional): Ground truth binary labels (0 or 1).
                When provided, adds confusion matrix panel and marks false 
                positives/negatives on the chart. Defaults to None.
            date_col (str, optional): Name of the date column in DataFrame.
                Defaults to 'Date'.
            price_col (str, optional): Name of the price column to plot (typically
                'Close' price). Defaults to 'Close'.
            threshold (float, optional): The prediction threshold that was used
                to generate y_pred (for display in title). Defaults to None.
            title (str, optional): Custom title for the plot. If None, generates
                automatic title. Defaults to None.
            figsize (tuple, optional): Figure size as (width, height) in inches.
                Defaults to (18, 10).
            cmap (str, optional): Colormap for confusion matrix. Use 'Blues' for
                general purpose, 'Greens' for highlighting positive signals.
                Defaults to 'Blues'.
            save_path (str, optional): Path to save the figure. If None, figure
                is not saved to disk. Defaults to None.
            dpi (int, optional): Resolution for saved figure. Defaults to 300.
            show (bool, optional): Whether to display the plot. Defaults to True.
            **kwargs: Additional styling options passed to matplotlib.
        
        Returns:
            matplotlib.figure.Figure: The created figure object, which can be
                further customized or saved.
        
        Raises:
            ValueError: If required columns are missing from DataFrame.
            ValueError: If y_pred length doesn't match DataFrame length.
        
        Example:
            Visualize predictions with confusion matrix:
            
            >>> import pandas as pd
            >>> import rhoa
            >>> df = pd.read_csv('stock_data.csv')
            >>> df['Date'] = pd.to_datetime(df['Date'])
            >>> # Assume you have predictions and targets
            >>> fig = df.plots.signal(
            ...     y_pred=predictions,
            ...     y_true=targets,
            ...     threshold=0.67,
            ...     save_path='my_prediction.png'
            ... )
            
        Example:
            Visualize predictions only (no ground truth):
            
            >>> fig = df.plots.signal(
            ...     y_pred=predictions,
            ...     date_col='Date',
            ...     price_col='Close',
            ...     cmap='Greens'
            ... )
        
        Note:
            - Light green background dots: True buy opportunities (when y_true provided)
            - Bright green dots: Model predicted buy signals
            - Red X markers: False positive predictions (when y_true provided)
            - Orange circles: Missed opportunities / false negatives (when y_true provided)
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
        """Validate inputs for signal plotting method."""
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
        """Plot confusion matrix with custom annotations."""
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
        """Plot price chart with overlaid buy signals."""
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

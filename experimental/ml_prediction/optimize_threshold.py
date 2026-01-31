"""
Threshold optimization script to find the best prediction threshold for target precision.

This script:
1. Loads a trained model
2. Makes predictions on validation/test data
3. Tests multiple thresholds to find optimal precision-recall tradeoff
4. Identifies thresholds achieving 90%, 95%, and 100% precision
5. Plots precision-recall curves and threshold analysis
6. Saves results for comparison

Usage:
    python optimize_threshold.py --model_path saved_models/my_model --data_file data/combined_stocks.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import rhoa

from persistence import load_pipeline
from feature_engineering import create_features
from target_generation import generate_targets
import config


def calculate_metrics_at_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float
) -> Dict:
    """
    Calculate classification metrics at a specific threshold.
    
    Args:
        y_true: True labels
        probabilities: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary with metrics
    """
    predictions = (probabilities >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    
    # Metrics
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    
    # Total signals
    total_signals = tp + fp
    
    return {
        'threshold': threshold,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_signals': int(total_signals)
    }


def optimize_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    min_threshold: float = 0.50,
    max_threshold: float = 0.90,
    step: float = 0.01,
    target_precisions: List[float] = [0.90, 0.95, 1.00]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Test multiple thresholds and find optimal values for target precision levels.
    
    Args:
        y_true: True labels
        probabilities: Predicted probabilities
        min_threshold: Minimum threshold to test
        max_threshold: Maximum threshold to test
        step: Step size for threshold testing
        target_precisions: List of target precision levels to find
        
    Returns:
        Tuple of (results_df, target_thresholds_dict)
    """
    thresholds = np.arange(min_threshold, max_threshold + step, step)
    results = []
    
    print(f"Testing {len(thresholds)} thresholds from {min_threshold} to {max_threshold}...")
    
    for threshold in thresholds:
        metrics = calculate_metrics_at_threshold(y_true, probabilities, threshold)
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    # Find thresholds for target precision levels
    target_thresholds = {}
    for target_prec in target_precisions:
        # Find rows where precision >= target
        candidates = results_df[results_df['precision'] >= target_prec]
        
        if len(candidates) > 0:
            # Choose threshold with highest recall (most signals)
            best_idx = candidates['recall'].idxmax()
            best_row = candidates.loc[best_idx]
            target_thresholds[target_prec] = {
                'threshold': best_row['threshold'],
                'precision': best_row['precision'],
                'recall': best_row['recall'],
                'f1': best_row['f1'],
                'total_signals': best_row['total_signals'],
                'tp': best_row['tp'],
                'fp': best_row['fp']
            }
        else:
            target_thresholds[target_prec] = None
    
    return results_df, target_thresholds


def plot_threshold_analysis(
    results_df: pd.DataFrame,
    target_thresholds: Dict,
    save_path: str = None
):
    """
    Create visualization of threshold analysis.
    
    Args:
        results_df: DataFrame with metrics at different thresholds
        target_thresholds: Dictionary of target precision thresholds
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Precision-Recall vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(results_df['threshold'], results_df['precision'], 
             label='Precision', linewidth=2, color='blue')
    ax1.plot(results_df['threshold'], results_df['recall'], 
             label='Recall', linewidth=2, color='green')
    ax1.plot(results_df['threshold'], results_df['f1'], 
             label='F1-Score', linewidth=2, color='red', linestyle='--')
    
    # Mark target thresholds
    for target_prec, info in target_thresholds.items():
        if info:
            ax1.axvline(info['threshold'], color='gray', linestyle=':', alpha=0.5)
            ax1.text(info['threshold'], 0.05, f"{int(target_prec*100)}%", 
                    rotation=90, fontsize=9, alpha=0.7)
    
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Precision, Recall, F1 vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(results_df['threshold'].min(), results_df['threshold'].max())
    ax1.set_ylim(0, 1.05)
    
    # 2. Precision-Recall Curve
    ax2 = axes[0, 1]
    ax2.plot(results_df['recall'], results_df['precision'], 
             linewidth=2, color='purple')
    
    # Mark target precision levels
    for target_prec, info in target_thresholds.items():
        if info:
            ax2.scatter(info['recall'], info['precision'], 
                       s=100, color='red', zorder=5)
            ax2.text(info['recall'] + 0.01, info['precision'], 
                    f"{int(target_prec*100)}%\n(θ={info['threshold']:.2f})", 
                    fontsize=9, va='center')
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(0.5, results_df['recall'].max() * 1.1))
    ax2.set_ylim(0, 1.05)
    
    # 3. Total Signals vs Threshold
    ax3 = axes[1, 0]
    ax3.plot(results_df['threshold'], results_df['total_signals'], 
             linewidth=2, color='orange')
    ax3.fill_between(results_df['threshold'], results_df['total_signals'], 
                     alpha=0.3, color='orange')
    
    # Mark target thresholds
    for target_prec, info in target_thresholds.items():
        if info:
            ax3.axvline(info['threshold'], color='gray', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('Total Signals (TP + FP)', fontsize=12)
    ax3.set_title('Number of Signals vs Threshold', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(results_df['threshold'].min(), results_df['threshold'].max())
    
    # 4. True Positives vs False Positives
    ax4 = axes[1, 1]
    ax4.plot(results_df['threshold'], results_df['tp'], 
             label='True Positives', linewidth=2, color='green')
    ax4.plot(results_df['threshold'], results_df['fp'], 
             label='False Positives', linewidth=2, color='red')
    
    # Mark target thresholds
    for target_prec, info in target_thresholds.items():
        if info:
            ax4.axvline(info['threshold'], color='gray', linestyle=':', alpha=0.5)
    
    ax4.set_xlabel('Threshold', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('True Positives vs False Positives', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(results_df['threshold'].min(), results_df['threshold'].max())
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Threshold analysis plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_optimization_results(target_thresholds: Dict, y_true: np.ndarray):
    """
    Print summary of threshold optimization results.
    
    Args:
        target_thresholds: Dictionary of target precision thresholds
        y_true: True labels (for baseline stats)
    """
    total_positives = int(y_true.sum())
    total_samples = len(y_true)
    
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("="*80)
    print(f"Total samples: {total_samples}")
    print(f"Positive samples: {total_positives} ({total_positives/total_samples*100:.1f}%)")
    print()
    
    for target_prec in sorted(target_thresholds.keys(), reverse=True):
        info = target_thresholds[target_prec]
        
        print(f"{'─'*80}")
        print(f"TARGET PRECISION: {int(target_prec*100)}%")
        
        if info:
            print(f"  Threshold:       {info['threshold']:.3f}")
            print(f"  Actual Precision: {info['precision']:.1%}")
            print(f"  Recall:          {info['recall']:.1%}")
            print(f"  F1-Score:        {info['f1']:.3f}")
            print(f"  Total Signals:   {info['total_signals']}")
            print(f"  True Positives:  {info['tp']} / {total_positives}")
            print(f"  False Positives: {info['fp']}")
            
            if info['tp'] > 0:
                improvement = (info['tp'] / 3) if target_prec == 1.0 else None
                if improvement:
                    print(f"  Improvement:     {improvement:.1f}x more TP than baseline (3 TP)")
        else:
            print(f"  ⚠️  Could not achieve {int(target_prec*100)}% precision with any threshold")
        print()
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Optimize prediction threshold for target precision levels"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='saved_models/high_precision_model',
        help='Path to trained model directory'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to data CSV file (can be combined multi-stock data)'
    )
    parser.add_argument(
        '--min_threshold',
        type=float,
        default=0.50,
        help='Minimum threshold to test (default: 0.50)'
    )
    parser.add_argument(
        '--max_threshold',
        type=float,
        default=0.90,
        help='Maximum threshold to test (default: 0.90)'
    )
    parser.add_argument(
        '--step',
        type=float,
        default=0.01,
        help='Step size for threshold testing (default: 0.01)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='threshold_optimization',
        help='Output directory for results (default: threshold_optimization)'
    )
    parser.add_argument(
        '--use_test_split',
        action='store_true',
        help='Use last 15%% of data as test set (default: use all data)'
    )
    
    args = parser.parse_args()
    
    # Make paths relative to script location
    script_dir = Path(__file__).parent
    model_path = script_dir / args.model_path
    data_file = Path(args.data_file)
    if not data_file.is_absolute():
        data_file = script_dir / data_file
    output_dir = script_dir / args.output_dir
    
    print("="*80)
    print("THRESHOLD OPTIMIZATION")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Data:  {data_file}")
    print(f"Output: {output_dir}")
    print("="*80)
    print()
    
    # Load model
    print("### Loading Model ###")
    model_pipeline = load_pipeline(str(model_path))
    print(f"✓ Model loaded: {len(model_pipeline['feature_columns'])} features")
    print()
    
    # Load data
    print("### Loading Data ###")
    df = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df)} rows")
    
    # Optionally use only test split
    if args.use_test_split:
        test_start = int(len(df) * 0.85)
        df = df.iloc[test_start:].copy()
        print(f"✓ Using last 15% as test set: {len(df)} rows")
    print()
    
    # Generate features and targets
    print("### Generating Features ###")
    df_features = create_features(df)
    df_with_target = generate_targets(
        df_features,
        threshold_percent=config.RETURN_THRESHOLD_PERCENT,
        horizon_days=config.TARGET_HORIZON_DAYS
    )
    df_clean = df_with_target.dropna()
    print(f"✓ Generated {len(model_pipeline['feature_columns'])} features")
    print(f"✓ Clean samples: {len(df_clean)}")
    print()
    
    # Prepare data
    feature_columns = model_pipeline['feature_columns']
    X = df_clean[feature_columns].values
    y_true = df_clean['target'].values
    
    # Make predictions
    print("### Making Predictions ###")
    X_scaled = model_pipeline['scaler'].transform(X)
    probabilities = model_pipeline['model'].predict(X_scaled, verbose=0).flatten()
    print(f"✓ Generated predictions for {len(probabilities)} samples")
    print(f"  Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    print()
    
    # Optimize thresholds
    print("### Optimizing Thresholds ###")
    results_df, target_thresholds = optimize_thresholds(
        y_true,
        probabilities,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step,
        target_precisions=[0.90, 0.95, 1.00]
    )
    print(f"✓ Tested {len(results_df)} thresholds")
    print()
    
    # Print results
    print_optimization_results(target_thresholds, y_true)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_csv = output_dir / 'threshold_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\n✓ Detailed results saved to: {results_csv}")
    
    # Save target thresholds
    target_csv = output_dir / 'target_thresholds.csv'
    target_df = pd.DataFrame([
        {
            'target_precision': f"{int(k*100)}%",
            'threshold': v['threshold'] if v else None,
            'actual_precision': v['precision'] if v else None,
            'recall': v['recall'] if v else None,
            'f1': v['f1'] if v else None,
            'total_signals': v['total_signals'] if v else None,
            'tp': v['tp'] if v else None,
            'fp': v['fp'] if v else None,
        }
        for k, v in sorted(target_thresholds.items(), reverse=True)
    ])
    target_df.to_csv(target_csv, index=False)
    print(f"✓ Target thresholds saved to: {target_csv}")
    
    # Plot analysis
    print("\n### Creating Visualizations ###")
    plot_path = output_dir / 'threshold_analysis.png'
    plot_threshold_analysis(results_df, target_thresholds, save_path=str(plot_path))
    
    print("\n" + "="*80)
    print("✓ OPTIMIZATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()

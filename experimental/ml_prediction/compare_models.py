"""
Model comparison script to evaluate and compare different trained models.

This script:
1. Loads multiple trained models
2. Evaluates each on the same test dataset
3. Compares performance at multiple thresholds
4. Creates side-by-side visualizations
5. Generates comparison report

Usage:
    python compare_models.py \
        --models saved_models/high_precision_model saved_models/multi_stock_v1 \
        --labels "Baseline" "Multi-Stock" \
        --data_file data/combined_stocks.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import List, Dict

# Add parent directory to path for rhoa imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import rhoa

from persistence import load_pipeline
from feature_engineering import create_features
from target_generation import generate_targets
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import config


def evaluate_model_at_thresholds(
    model_pipeline: Dict,
    df: pd.DataFrame,
    thresholds: List[float] = [0.50, 0.60, 0.65, 0.67, 0.70]
) -> pd.DataFrame:
    """
    Evaluate a model at multiple thresholds.
    
    Args:
        model_pipeline: Loaded model pipeline
        df: DataFrame with OHLCV data
        thresholds: List of thresholds to test
        
    Returns:
        DataFrame with results for each threshold
    """
    # Generate features and target
    df_features = create_features(df)
    df_with_target = generate_targets(
        df_features,
        threshold_percent=config.RETURN_THRESHOLD_PERCENT,
        horizon_days=config.TARGET_HORIZON_DAYS
    )
    df_clean = df_with_target.dropna()
    
    # Prepare features
    feature_columns = model_pipeline['feature_columns']
    X = df_clean[feature_columns].values
    y_true = df_clean['target'].values
    
    # Scale and predict
    X_scaled = model_pipeline['scaler'].transform(X)
    probabilities = model_pipeline['model'].predict(X_scaled, verbose=0).flatten()
    
    # Evaluate at each threshold
    results = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Metrics
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'total_signals': int(tp + fp)
        })
    
    return pd.DataFrame(results)


def plot_model_comparison(
    all_results: Dict[str, pd.DataFrame],
    save_path: str = None
):
    """
    Create comparison plots for multiple models.
    
    Args:
        all_results: Dictionary mapping model name to results DataFrame
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color palette
    colors = sns.color_palette("husl", len(all_results))
    
    # Plot 1: Precision vs Threshold
    ax1 = axes[0, 0]
    for (model_name, results), color in zip(all_results.items(), colors):
        ax1.plot(results['threshold'], results['precision'], 
                marker='o', linewidth=2, markersize=6, label=model_name, color=color)
    
    ax1.axhline(0.90, color='red', linestyle='--', alpha=0.3, label='90% Target')
    ax1.axhline(1.00, color='green', linestyle='--', alpha=0.3, label='100% Target')
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Recall vs Threshold
    ax2 = axes[0, 1]
    for (model_name, results), color in zip(all_results.items(), colors):
        ax2.plot(results['threshold'], results['recall'], 
                marker='s', linewidth=2, markersize=6, label=model_name, color=color)
    
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Recall', fontsize=12)
    ax2.set_title('Recall vs Threshold', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(0.5, max([df['recall'].max() for df in all_results.values()]) * 1.1))
    
    # Plot 3: Precision-Recall Curve
    ax3 = axes[1, 0]
    for (model_name, results), color in zip(all_results.items(), colors):
        ax3.plot(results['recall'], results['precision'], 
                marker='o', linewidth=2, markersize=6, label=model_name, color=color)
    
    ax3.set_xlabel('Recall', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)
    
    # Plot 4: Total Signals vs Threshold
    ax4 = axes[1, 1]
    x = np.arange(len(results['threshold']))
    width = 0.8 / len(all_results)
    
    for i, ((model_name, results), color) in enumerate(zip(all_results.items(), colors)):
        offset = (i - len(all_results)/2 + 0.5) * width
        ax4.bar(x + offset, results['total_signals'], width, 
               label=model_name, color=color, alpha=0.7)
    
    ax4.set_xlabel('Threshold', fontsize=12)
    ax4.set_ylabel('Total Signals (TP + FP)', fontsize=12)
    ax4.set_title('Number of Signals vs Threshold', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{t:.2f}" for t in results['threshold']])
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrices(
    all_results: Dict[str, pd.DataFrame],
    threshold: float,
    save_path: str = None
):
    """
    Plot confusion matrices side-by-side for a specific threshold.
    
    Args:
        all_results: Dictionary mapping model name to results DataFrame
        threshold: Specific threshold to display
        save_path: Path to save plot
    """
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, results) in zip(axes, all_results.items()):
        # Get results for this threshold
        row = results[results['threshold'] == threshold].iloc[0]
        
        # Create confusion matrix
        cm = np.array([[row['tn'], row['fp']], 
                      [row['fn'], row['tp']]])
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
        
        # Labels
        classes = ['Negative (0)', 'Positive (1)']
        ax.set(xticks=np.arange(2),
               yticks=np.arange(2),
               xticklabels=classes,
               yticklabels=classes,
               ylabel='True label',
               xlabel='Predicted label')
        
        # Add counts
        thresh_val = cm_norm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                       ha="center", va="center",
                       color="white" if cm_norm[i, j] > thresh_val else "black",
                       fontsize=12)
        
        # Title
        ax.set_title(f'{model_name}\n(Threshold: {threshold:.2f})\n'
                    f'Precision: {row["precision"]:.1%}, Recall: {row["recall"]:.1%}',
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrices saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_comparison_table(all_results: Dict[str, pd.DataFrame], thresholds: List[float]):
    """
    Print comparison table for all models.
    
    Args:
        all_results: Dictionary mapping model name to results DataFrame
        thresholds: List of thresholds to display
    """
    print("\n" + "="*100)
    print("MODEL COMPARISON TABLE")
    print("="*100)
    
    for threshold in thresholds:
        print(f"\n{'─'*100}")
        print(f"THRESHOLD: {threshold:.2f}")
        print(f"{'─'*100}")
        
        # Table header
        print(f"{'Model':<25} | {'Precision':<10} | {'Recall':<10} | {'Signals':<8} | {'TP':<5} | {'FP':<5}")
        print("─" * 100)
        
        # Table rows
        for model_name, results in all_results.items():
            row = results[results['threshold'] == threshold].iloc[0]
            print(f"{model_name:<25} | {row['precision']:>9.1%} | {row['recall']:>9.1%} | "
                  f"{int(row['total_signals']):>8d} | {int(row['tp']):>5d} | {int(row['fp']):>5d}")
    
    print("\n" + "="*100)


def create_comparison_report(
    all_results: Dict[str, pd.DataFrame],
    output_path: str
):
    """
    Create a detailed comparison report.
    
    Args:
        all_results: Dictionary mapping model name to results DataFrame
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("MODEL COMPARISON REPORT\n")
        f.write("="*100 + "\n\n")
        
        # Summary section
        f.write("SUMMARY\n")
        f.write("-"*100 + "\n")
        f.write(f"Number of models compared: {len(all_results)}\n")
        f.write(f"Models: {', '.join(all_results.keys())}\n\n")
        
        # Best performance for each metric
        f.write("BEST PERFORMANCE BY METRIC\n")
        f.write("-"*100 + "\n")
        
        # Best precision at each level
        for target_prec in [0.90, 0.95, 1.00]:
            best_model = None
            best_recall = 0
            best_signals = 0
            
            for model_name, results in all_results.items():
                candidates = results[results['precision'] >= target_prec]
                if len(candidates) > 0:
                    max_recall = candidates['recall'].max()
                    if max_recall > best_recall:
                        best_recall = max_recall
                        best_model = model_name
                        best_signals = candidates.loc[candidates['recall'].idxmax(), 'total_signals']
            
            if best_model:
                f.write(f"\nBest at {int(target_prec*100)}% precision:\n")
                f.write(f"  Model: {best_model}\n")
                f.write(f"  Recall: {best_recall:.1%}\n")
                f.write(f"  Signals: {best_signals}\n")
            else:
                f.write(f"\nNo model achieved {int(target_prec*100)}% precision\n")
        
        # Detailed results
        f.write("\n\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*100 + "\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"\nModel: {model_name}\n")
            f.write("-"*100 + "\n")
            f.write(results.to_string(index=False))
            f.write("\n\n")
    
    print(f"✓ Comparison report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple trained models"
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='Paths to model directories to compare'
    )
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        default=None,
        help='Labels for each model (default: use directory names)'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to test data CSV'
    )
    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        default=[0.50, 0.60, 0.65, 0.67, 0.70],
        help='Thresholds to test (default: 0.50 0.60 0.65 0.67 0.70)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='model_comparison',
        help='Output directory for results (default: model_comparison)'
    )
    parser.add_argument(
        '--comparison_threshold',
        type=float,
        default=0.67,
        help='Threshold for confusion matrix comparison (default: 0.67)'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate labels if not provided
    if args.labels:
        if len(args.labels) != len(args.models):
            raise ValueError("Number of labels must match number of models")
        labels = args.labels
    else:
        labels = [Path(m).name for m in args.models]
    
    print("="*100)
    print("MODEL COMPARISON")
    print("="*100)
    print(f"Comparing {len(args.models)} models:")
    for label, model_path in zip(labels, args.models):
        print(f"  - {label}: {model_path}")
    print(f"Test data: {args.data_file}")
    print(f"Thresholds: {args.thresholds}")
    print("="*100)
    print()
    
    # Load test data
    print("### Loading Test Data ###")
    data_file = Path(args.data_file)
    if not data_file.is_absolute():
        data_file = script_dir / data_file
    df = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df)} rows")
    print()
    
    # Load and evaluate each model
    print("### Loading and Evaluating Models ###")
    all_results = {}
    
    for label, model_path in zip(labels, args.models):
        print(f"\nEvaluating: {label}")
        print("-" * 80)
        
        # Load model
        model_path_full = Path(model_path)
        if not model_path_full.is_absolute():
            model_path_full = script_dir / model_path
        
        model_pipeline = load_pipeline(str(model_path_full))
        print(f"✓ Model loaded: {len(model_pipeline['feature_columns'])} features")
        
        # Evaluate
        results = evaluate_model_at_thresholds(model_pipeline, df, thresholds=args.thresholds)
        all_results[label] = results
        print(f"✓ Evaluated at {len(results)} thresholds")
    
    print("\n" + "="*100)
    print("✓ All models evaluated")
    print("="*100)
    
    # Print comparison table
    print_comparison_table(all_results, args.thresholds)
    
    # Create visualizations
    print("\n### Creating Visualizations ###")
    
    # Comparison plots
    plot_path = output_dir / 'model_comparison.png'
    plot_model_comparison(all_results, save_path=str(plot_path))
    
    # Confusion matrices at specific threshold
    cm_path = output_dir / f'confusion_matrices_threshold_{args.comparison_threshold:.2f}.png'
    plot_confusion_matrices(all_results, args.comparison_threshold, save_path=str(cm_path))
    
    # Save detailed results
    print("\n### Saving Results ###")
    
    # Save combined results CSV
    for label, results in all_results.items():
        results_path = output_dir / f'{label.replace(" ", "_")}_results.csv'
        results.to_csv(results_path, index=False)
        print(f"✓ {label} results saved to: {results_path}")
    
    # Create comparison report
    report_path = output_dir / 'comparison_report.txt'
    create_comparison_report(all_results, str(report_path))
    
    print("\n" + "="*100)
    print("✓ COMPARISON COMPLETE!")
    print("="*100)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()

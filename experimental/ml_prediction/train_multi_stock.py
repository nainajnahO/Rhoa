"""
Complete training script for multi-stock stock prediction model.

This script:
1. Loads combined multi-stock data
2. Generates technical indicator features
3. Creates target labels
4. Splits data with multi-stock strategy
5. Trains neural network model
6. Saves trained model and artifacts
7. Evaluates on test set

Usage:
    python train_multi_stock.py --data_file data/combined_stocks.csv --model_name multi_stock_v1
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime

# Add parent directory to path for rhoa imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import rhoa

import config
from feature_engineering import create_features
from target_generation import generate_targets
from preprocessing import preprocess_pipeline, print_preprocessing_summary
from model import create_model
from training import train_model, plot_training_history, print_training_summary
from persistence import save_pipeline
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate model on test set at specific threshold.
    
    Args:
        model: Trained Keras model
        X_test: Test features (scaled)
        y_test: Test labels
        threshold: Classification threshold
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    probabilities = model.predict(X_test, verbose=0).flatten()
    predictions = (probabilities >= threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Metrics
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    
    return {
        'threshold': threshold,
        'confusion_matrix': cm,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'precision': precision,
        'recall': recall,
        'total_signals': int(tp + fp),
        'probabilities': probabilities
    }


def plot_confusion_matrix(cm, threshold, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        threshold: Threshold used
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    classes = ['Negative (0)', 'Positive (1)']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add counts to cells
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                   ha="center", va="center",
                   color="white" if cm_norm[i, j] > thresh else "black",
                   fontsize=14)
    
    ax.set_title(f'Confusion Matrix (Threshold: {threshold})', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_evaluation_results(eval_results):
    """
    Print evaluation results summary.
    
    Args:
        eval_results: Dictionary from evaluate_model
    """
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    print(f"Threshold: {eval_results['threshold']:.3f}")
    print()
    print(f"Confusion Matrix:")
    print(f"  TN: {eval_results['tn']:4d}  |  FP: {eval_results['fp']:4d}")
    print(f"  FN: {eval_results['fn']:4d}  |  TP: {eval_results['tp']:4d}")
    print()
    print(f"Metrics:")
    print(f"  Precision:     {eval_results['precision']:.1%}")
    print(f"  Recall:        {eval_results['recall']:.1%}")
    print(f"  Total Signals: {eval_results['total_signals']}")
    print(f"  Correct:       {eval_results['tp']} / {eval_results['total_signals']}")
    print(f"  Wrong:         {eval_results['fp']} / {eval_results['total_signals']}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Train stock prediction model on multi-stock data"
    )
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to combined stock data CSV (from aggregate_data.py)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='Name for saved model (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--holdout_stocks',
        type=str,
        nargs='+',
        default=None,
        help='Stock IDs to hold out entirely for testing (optional)'
    )
    parser.add_argument(
        '--eval_threshold',
        type=float,
        default=0.67,
        help='Threshold for evaluation (default: 0.67)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: from config)'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    data_file = Path(args.data_file)
    if not data_file.is_absolute():
        data_file = script_dir / data_file
    
    # Generate model name if not provided
    if args.model_name:
        model_name = args.model_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"multi_stock_model_{timestamp}"
    
    output_dir = script_dir / 'saved_models' / model_name
    
    print("="*80)
    print("MULTI-STOCK MODEL TRAINING")
    print("="*80)
    print(f"Data file: {data_file}")
    print(f"Model name: {model_name}")
    print(f"Output directory: {output_dir}")
    if args.holdout_stocks:
        print(f"Holdout stocks: {', '.join(args.holdout_stocks)}")
    print("="*80)
    print()
    
    # Load data
    print("### Step 1: Loading Data ###")
    df_raw = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df_raw)} rows")
    
    # Check for stock_id column
    if 'stock_id' in df_raw.columns:
        n_stocks = df_raw['stock_id'].nunique()
        print(f"✓ Found {n_stocks} unique stocks")
        stock_ids = df_raw['stock_id'].unique()
        for stock_id in stock_ids:
            count = len(df_raw[df_raw['stock_id'] == stock_id])
            print(f"  - {stock_id}: {count:,} rows")
    else:
        print("⚠️  No 'stock_id' column found - treating as single stock")
    print()
    
    # Generate features
    print("### Step 2: Generating Features ###")
    print("Creating technical indicators...")
    df_features = create_features(df_raw)
    print(f"✓ Generated features")
    print()
    
    # Generate targets
    print("### Step 3: Generating Targets ###")
    print(f"Target: >{config.RETURN_THRESHOLD_PERCENT}% return in {config.TARGET_HORIZON_DAYS} days")
    df_with_target = generate_targets(
        df_features,
        threshold_percent=config.RETURN_THRESHOLD_PERCENT,
        horizon_days=config.TARGET_HORIZON_DAYS
    )
    print(f"✓ Generated target labels")
    print()
    
    # Get feature columns (exclude non-feature columns)
    exclude_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target', 'stock_id', 
                   'future_return', 'future_close'}
    feature_columns = [col for col in df_with_target.columns if col not in exclude_cols]
    print(f"✓ Using {len(feature_columns)} features")
    print()
    
    # Preprocessing pipeline
    print("### Step 4: Preprocessing & Splitting ###")
    has_stock_col = 'stock_id' in df_with_target.columns
    
    # Override epochs if specified
    if args.epochs:
        config.TRAINING_CONFIG['epochs'] = args.epochs
        print(f"Using {args.epochs} training epochs")
    
    preprocessed = preprocess_pipeline(
        df_with_target,
        feature_columns=feature_columns,
        target_column='target',
        stock_col='stock_id' if has_stock_col else None,
        holdout_stocks=args.holdout_stocks
    )
    
    X_train = preprocessed['X_train'].values
    y_train = preprocessed['y_train'].values
    X_val = preprocessed['X_val'].values
    y_val = preprocessed['y_val'].values
    X_test = preprocessed['X_test'].values
    y_test = preprocessed['y_test'].values
    scaler = preprocessed['scaler']
    
    print_preprocessing_summary(preprocessed['stats'])
    print()
    
    # Create model
    print("### Step 5: Creating Model ###")
    model = create_model(input_dim=X_train.shape[1])
    model.summary()
    print()
    
    # Train model
    print("### Step 6: Training Model ###")
    model, history = train_model(
        X_train, y_train,
        X_val, y_val,
        model=model,
        config=config.TRAINING_CONFIG
    )
    print_training_summary(history)
    print()
    
    # Evaluate on test set
    print("### Step 7: Evaluating on Test Set ###")
    eval_results = evaluate_model(model, X_test, y_test, threshold=args.eval_threshold)
    print_evaluation_results(eval_results)
    print()
    
    # Save model and artifacts
    print("### Step 8: Saving Model ###")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model pipeline
    save_pipeline(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        model_dir=str(output_dir),
        model_name=model_name
    )
    print(f"✓ Model saved to: {output_dir}")
    
    # Save training history plot
    history_plot_path = output_dir / 'training_history.png'
    plot_training_history(history, save_path=str(history_plot_path))
    
    # Save confusion matrix plot
    cm_plot_path = output_dir / f'confusion_matrix_threshold_{args.eval_threshold}.png'
    plot_confusion_matrix(
        eval_results['confusion_matrix'],
        threshold=args.eval_threshold,
        save_path=str(cm_plot_path)
    )
    
    # Save evaluation results
    eval_results_path = output_dir / 'test_evaluation.txt'
    with open(eval_results_path, 'w') as f:
        f.write("TEST SET EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Threshold: {eval_results['threshold']:.3f}\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  TN: {eval_results['tn']:4d}  |  FP: {eval_results['fp']:4d}\n")
        f.write(f"  FN: {eval_results['fn']:4d}  |  TP: {eval_results['tp']:4d}\n\n")
        f.write(f"Metrics:\n")
        f.write(f"  Precision:     {eval_results['precision']:.1%}\n")
        f.write(f"  Recall:        {eval_results['recall']:.1%}\n")
        f.write(f"  Total Signals: {eval_results['total_signals']}\n")
        f.write(f"  Correct:       {eval_results['tp']} / {eval_results['total_signals']}\n")
        f.write(f"  Wrong:         {eval_results['fp']} / {eval_results['total_signals']}\n")
    
    print(f"✓ Evaluation results saved to: {eval_results_path}")
    print()
    
    print("="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Run threshold optimization:")
    print(f"     python optimize_threshold.py --model_path {output_dir} --data_file {data_file}")
    print(f"\n  2. Compare with baseline model:")
    print(f"     (Use evaluate_multi_threshold.py once created)")
    print()


if __name__ == '__main__':
    main()

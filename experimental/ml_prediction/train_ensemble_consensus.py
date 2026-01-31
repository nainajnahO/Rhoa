"""
Ensemble training with consensus voting for ultra-high precision.

This script:
1. Trains multiple models with different random seeds
2. Uses consensus voting (majority agreement) for predictions
3. Reduces false positives through democratic decision-making
4. Evaluates ensemble at different consensus thresholds

Key difference from previous ensemble approach:
- Uses VOTING (binary predictions), not averaging probabilities
- Requires N out of M models to agree before predicting positive
- More conservative, better for maintaining high precision

Usage:
    python train_ensemble_consensus.py --data_file data/combined_stocks.csv --n_models 5
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime
import json

# Add parent directory to path for rhoa imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import rhoa

import config
from feature_engineering import create_features
from target_generation import generate_targets
from preprocessing import preprocess_pipeline, print_preprocessing_summary
from model import create_model
from training import train_model, print_training_summary
from persistence import save_model as save_model_func
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import tensorflow as tf


def train_ensemble_models(
    X_train, y_train,
    X_val, y_val,
    n_models=5,
    base_seed=42
):
    """
    Train multiple models with different random seeds.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_models: Number of models to train
        base_seed: Base random seed
        
    Returns:
        List of trained models
    """
    models = []
    
    print(f"Training {n_models} models with different random seeds...")
    print("="*80)
    
    for i in range(n_models):
        seed = base_seed + i
        print(f"\n### Training Model {i+1}/{n_models} (seed={seed}) ###")
        
        # Set random seeds
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Create and train model
        model = create_model(input_dim=X_train.shape[1])
        model, history = train_model(
            X_train, y_train,
            X_val, y_val,
            model=model,
            config=config.TRAINING_CONFIG,
            verbose=0  # Less verbose for ensemble training
        )
        
        # Quick evaluation
        val_loss = history.history['val_loss'][-1]
        print(f"✓ Model {i+1} trained - Final val_loss: {val_loss:.4f}")
        
        models.append(model)
    
    print("\n" + "="*80)
    print(f"✓ All {n_models} models trained successfully")
    
    return models


def ensemble_predict_consensus(
    models,
    X,
    individual_threshold=0.5,
    consensus_threshold=0.6
):
    """
    Make ensemble predictions using consensus voting.
    
    Args:
        models: List of trained models
        X: Input features (scaled)
        individual_threshold: Threshold for each model's prediction
        consensus_threshold: Fraction of models that must agree (0.0-1.0)
        
    Returns:
        Tuple of (ensemble_predictions, individual_predictions, individual_probabilities)
    """
    n_models = len(models)
    n_samples = len(X)
    
    # Get predictions from each model
    individual_probs = np.zeros((n_samples, n_models))
    individual_preds = np.zeros((n_samples, n_models))
    
    for i, model in enumerate(models):
        probs = model.predict(X, verbose=0).flatten()
        individual_probs[:, i] = probs
        individual_preds[:, i] = (probs >= individual_threshold).astype(int)
    
    # Count votes (how many models predict positive)
    votes = individual_preds.sum(axis=1)
    
    # Require consensus threshold
    min_votes = int(np.ceil(n_models * consensus_threshold))
    ensemble_predictions = (votes >= min_votes).astype(int)
    
    return ensemble_predictions, individual_preds, individual_probs, votes


def evaluate_ensemble(
    models,
    X_test,
    y_test,
    individual_threshold=0.5,
    consensus_threshold=0.6
):
    """
    Evaluate ensemble on test set.
    
    Args:
        models: List of trained models
        X_test: Test features
        y_test: Test labels
        individual_threshold: Threshold for individual models
        consensus_threshold: Consensus threshold
        
    Returns:
        Dictionary with evaluation metrics
    """
    ensemble_preds, individual_preds, individual_probs, votes = ensemble_predict_consensus(
        models, X_test, individual_threshold, consensus_threshold
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, ensemble_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Metrics
    precision = precision_score(y_test, ensemble_preds, zero_division=0)
    recall = recall_score(y_test, ensemble_preds, zero_division=0)
    
    return {
        'consensus_threshold': consensus_threshold,
        'individual_threshold': individual_threshold,
        'n_models': len(models),
        'min_votes_required': int(np.ceil(len(models) * consensus_threshold)),
        'confusion_matrix': cm,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'precision': precision,
        'recall': recall,
        'total_signals': int(tp + fp),
        'votes_distribution': {
            str(i): int((votes == i).sum()) for i in range(len(models) + 1)
        }
    }


def test_consensus_thresholds(models, X_test, y_test, individual_threshold=0.5):
    """
    Test multiple consensus thresholds to find optimal balance.
    
    Args:
        models: List of trained models
        X_test: Test features
        y_test: Test labels
        individual_threshold: Threshold for individual model predictions
        
    Returns:
        DataFrame with results for different consensus thresholds
    """
    n_models = len(models)
    
    # Test different consensus thresholds
    consensus_thresholds = [i/n_models for i in range(1, n_models + 1)]
    results = []
    
    print(f"\nTesting {len(consensus_thresholds)} consensus thresholds...")
    
    for cons_thresh in consensus_thresholds:
        eval_results = evaluate_ensemble(
            models, X_test, y_test,
            individual_threshold=individual_threshold,
            consensus_threshold=cons_thresh
        )
        
        results.append({
            'consensus_threshold': cons_thresh,
            'min_votes': eval_results['min_votes_required'],
            'precision': eval_results['precision'],
            'recall': eval_results['recall'],
            'total_signals': eval_results['total_signals'],
            'tp': eval_results['tp'],
            'fp': eval_results['fp']
        })
    
    results_df = pd.DataFrame(results)
    return results_df


def plot_consensus_analysis(results_df, save_path=None):
    """
    Plot consensus threshold analysis.
    
    Args:
        results_df: DataFrame with consensus results
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Precision and Recall vs Consensus Threshold
    ax1 = axes[0]
    ax1.plot(results_df['min_votes'], results_df['precision'], 
             marker='o', linewidth=2, markersize=8, label='Precision', color='blue')
    ax1.plot(results_df['min_votes'], results_df['recall'], 
             marker='s', linewidth=2, markersize=8, label='Recall', color='green')
    
    # Mark 90% and 95% precision levels
    ax1.axhline(0.90, color='red', linestyle='--', alpha=0.5, label='90% Precision')
    ax1.axhline(0.95, color='orange', linestyle='--', alpha=0.5, label='95% Precision')
    ax1.axhline(1.00, color='purple', linestyle='--', alpha=0.5, label='100% Precision')
    
    ax1.set_xlabel('Minimum Votes Required (out of {})'.format(results_df['min_votes'].max()), fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Precision & Recall vs Consensus Threshold', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Total Signals vs Consensus Threshold
    ax2 = axes[1]
    ax2.bar(results_df['min_votes'], results_df['total_signals'], 
            color='orange', alpha=0.7, edgecolor='black')
    ax2.plot(results_df['min_votes'], results_df['tp'], 
             marker='o', linewidth=2, markersize=8, label='True Positives', color='green')
    ax2.plot(results_df['min_votes'], results_df['fp'], 
             marker='x', linewidth=2, markersize=8, label='False Positives', color='red')
    
    ax2.set_xlabel('Minimum Votes Required', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Signals Breakdown vs Consensus Threshold', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Consensus analysis plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_evaluation_results(eval_results):
    """Print ensemble evaluation results."""
    print("\n" + "="*80)
    print("ENSEMBLE EVALUATION (TEST SET)")
    print("="*80)
    print(f"Number of models:        {eval_results['n_models']}")
    print(f"Individual threshold:    {eval_results['individual_threshold']:.2f}")
    print(f"Consensus threshold:     {eval_results['consensus_threshold']:.0%}")
    print(f"Min votes required:      {eval_results['min_votes_required']} / {eval_results['n_models']}")
    print()
    print(f"Confusion Matrix:")
    print(f"  TN: {eval_results['tn']:4d}  |  FP: {eval_results['fp']:4d}")
    print(f"  FN: {eval_results['fn']:4d}  |  TP: {eval_results['tp']:4d}")
    print()
    print(f"Metrics:")
    print(f"  Precision:     {eval_results['precision']:.1%}")
    print(f"  Recall:        {eval_results['recall']:.1%}")
    print(f"  Total Signals: {eval_results['total_signals']}")
    print()
    print(f"Votes Distribution:")
    for votes, count in sorted(eval_results['votes_distribution'].items(), key=lambda x: int(x[0])):
        print(f"  {votes} votes: {count} samples")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Train ensemble model with consensus voting"
    )
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to combined stock data CSV'
    )
    parser.add_argument(
        '--n_models',
        type=int,
        default=5,
        help='Number of models in ensemble (default: 5)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='Name for saved ensemble (default: auto-generated)'
    )
    parser.add_argument(
        '--consensus_threshold',
        type=float,
        default=0.6,
        help='Consensus threshold (default: 0.6 = 60%% agreement)'
    )
    parser.add_argument(
        '--holdout_stocks',
        type=str,
        nargs='+',
        default=None,
        help='Stock IDs to hold out for testing'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    data_file = Path(args.data_file)
    if not data_file.is_absolute():
        data_file = script_dir / data_file
    
    # Generate model name
    if args.model_name:
        model_name = args.model_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"ensemble_{args.n_models}models_{timestamp}"
    
    output_dir = script_dir / 'saved_models' / model_name
    
    print("="*80)
    print("ENSEMBLE TRAINING (CONSENSUS VOTING)")
    print("="*80)
    print(f"Data file: {data_file}")
    print(f"Number of models: {args.n_models}")
    print(f"Consensus threshold: {args.consensus_threshold:.0%}")
    print(f"Model name: {model_name}")
    print("="*80)
    print()
    
    # Load and prepare data
    print("### Loading and Preparing Data ###")
    df_raw = pd.read_csv(data_file)
    df_features = create_features(df_raw)
    df_with_target = generate_targets(
        df_features,
        threshold_percent=config.RETURN_THRESHOLD_PERCENT,
        horizon_days=config.TARGET_HORIZON_DAYS
    )
    
    exclude_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target', 'stock_id',
                   'future_return', 'future_close'}
    feature_columns = [col for col in df_with_target.columns if col not in exclude_cols]
    
    preprocessed = preprocess_pipeline(
        df_with_target,
        feature_columns=feature_columns,
        target_column='target',
        stock_col='stock_id' if 'stock_id' in df_with_target.columns else None,
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
    
    # Train ensemble
    print("### Training Ensemble ###")
    models = train_ensemble_models(
        X_train, y_train,
        X_val, y_val,
        n_models=args.n_models
    )
    print()
    
    # Test different consensus thresholds
    print("### Testing Consensus Thresholds ###")
    results_df = test_consensus_thresholds(models, X_test, y_test)
    print("\nConsensus Threshold Results:")
    print(results_df.to_string(index=False))
    print()
    
    # Evaluate with specified consensus threshold
    print(f"### Evaluating at {args.consensus_threshold:.0%} Consensus ###")
    eval_results = evaluate_ensemble(
        models, X_test, y_test,
        consensus_threshold=args.consensus_threshold
    )
    print_evaluation_results(eval_results)
    print()
    
    # Save ensemble
    print("### Saving Ensemble ###")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual models
    for i, model in enumerate(models):
        model_path = output_dir / f'model_{i+1}.h5'
        model.save(str(model_path))
    
    # Save scaler
    import pickle
    scaler_path = output_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature columns
    features_path = output_dir / 'feature_columns.json'
    with open(features_path, 'w') as f:
        json.dump(feature_columns, f, indent=2)
    
    # Save ensemble config
    config_path = output_dir / 'ensemble_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'n_models': args.n_models,
            'consensus_threshold': args.consensus_threshold,
            'individual_threshold': 0.5,
            'num_features': len(feature_columns)
        }, f, indent=2)
    
    # Save consensus results
    results_path = output_dir / 'consensus_results.csv'
    results_df.to_csv(results_path, index=False)
    
    # Save evaluation results
    eval_path = output_dir / 'test_evaluation.json'
    with open(eval_path, 'w') as f:
        eval_save = {k: v for k, v in eval_results.items() 
                    if k not in ['confusion_matrix']}  # Can't serialize numpy array
        eval_save['confusion_matrix'] = eval_results['confusion_matrix'].tolist()
        json.dump(eval_save, f, indent=2)
    
    # Plot analysis
    plot_path = output_dir / 'consensus_analysis.png'
    plot_consensus_analysis(results_df, save_path=str(plot_path))
    
    print(f"✓ Ensemble saved to: {output_dir}")
    print()
    
    print("="*80)
    print("✓ ENSEMBLE TRAINING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()

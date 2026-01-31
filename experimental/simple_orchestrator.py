"""
Simple ML Stock Predictor Orchestrator - Complete Standalone Version

This script does everything:
1. Trains a model if it doesn't exist (first stock is holdout, rest are for training)
2. Makes predictions on the holdout stock
3. Applies pattern filtering
4. Creates side-by-side comparison plots

To use:
1. Edit STOCK_URLS (first stock will be holdout/test stock)
2. Edit other configuration below if needed
3. Run: python experimental/simple_orchestrator.py
"""

import os
import sys
from pathlib import Path

# Add paths for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir / 'ml_prediction'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from PIL import Image

# Import rhoa for plotting
import rhoa

# Import ML prediction modules
import config
from aggregate_data import aggregate_stocks_from_urls, load_single_stock
from feature_engineering import create_features
from target_generation import generate_targets
from preprocessing import preprocess_pipeline, print_preprocessing_summary
from model import create_model
from training import train_model, plot_training_history, print_training_summary
from persistence import save_pipeline, load_pipeline, predict_with_pipeline


# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

# Google Sheets URLs (CSV export format)
STOCK_URLS = {
    'EPRO-BB': 'https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=71372649',
    'LOOMIS': 'https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=911108469',
    'BEIA-B': 'https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=1487889979',
    'HUSQ-B': 'https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=531134335'
}

# Model configuration
TARGET_RETURN_PERCENT = 10.0
PREDICTION_THRESHOLD = 0.67  # Lowered from 0.7 for model without class weighting
MODEL_NAME = 'simple_model'

# Date span for plotting (set to None to use all available data)
START_DATE = None  # e.g., '2023-01-01' or None for earliest date
END_DATE = "2024-01-25"   # e.g., '2024-12-31' or None for latest date

# Pattern filtering configuration
WINDOW_DAYS = 14  # Look for patterns within 2 weeks
MIN_SIGNALS = 3   # Minimum signals required in window
MIN_SLOPE_ANGLE = 15  # Minimum angle in degrees for upward trend

# ============================================================================


def filter_signals_by_pattern(df, predictions, date_col='Date', price_col='Close',
                               window_days=14, min_signals=3, min_angle_degrees=45):
    """
    Filter buy signals based on pattern detection.
    
    Keep only signals that are part of a pattern where:
    - At least `min_signals` buy signals occur within `window_days`
    - The signals form an upward trend with slope angle >= `min_angle_degrees`
    """
    filtered_predictions = np.zeros_like(predictions)
    
    # Get indices where predictions == 1
    signal_indices = np.where(predictions == 1)[0]
    
    if len(signal_indices) == 0:
        return filtered_predictions, {'original': 0, 'filtered': 0, 'removed': 0}
    
    # Convert to DataFrame for easier manipulation
    df_signals = df.iloc[signal_indices].copy()
    df_signals['original_idx'] = signal_indices
    
    signals_kept = 0
    
    # For each buy signal, check if it's part of a valid pattern
    for i, row in df_signals.iterrows():
        current_date = row[date_col]
        current_price = row[price_col]
        current_idx = row['original_idx']
        
        # Find all signals within window_days before current signal (inclusive)
        window_start = current_date - pd.Timedelta(days=window_days)
        window_mask = (df_signals[date_col] >= window_start) & (df_signals[date_col] <= current_date)
        signals_in_window = df_signals[window_mask]
        
        # Check if we have enough signals
        if len(signals_in_window) >= min_signals:
            # Fit a line through the signals in the window
            dates_numeric = (signals_in_window[date_col] - signals_in_window[date_col].min()).dt.days.values
            prices = signals_in_window[price_col].values
            
            if len(dates_numeric) >= 2:
                # Fit linear regression
                X = dates_numeric.reshape(-1, 1)
                y = prices
                
                lr = LinearRegression()
                lr.fit(X, y)
                slope = lr.coef_[0]
                
                # Calculate angle in degrees
                angle_radians = np.arctan(slope)
                angle_degrees = np.degrees(angle_radians)
                
                # Check if slope meets threshold
                if angle_degrees >= min_angle_degrees:
                    filtered_predictions[current_idx] = 1
                    signals_kept += 1
    
    stats = {
        'original': len(signal_indices),
        'filtered': signals_kept,
        'removed': len(signal_indices) - signals_kept,
    }
    
    return filtered_predictions, stats


def calc_metrics(y_pred, y_true, mask):
    """Calculate precision, recall, and confusion matrix values."""
    y_p = y_pred[mask]
    y_t = y_true[mask].astype(int)
    
    tp = ((y_p == 1) & (y_t == 1)).sum()
    fp = ((y_p == 1) & (y_t == 0)).sum()
    fn = ((y_p == 0) & (y_t == 1)).sum()
    tn = ((y_p == 0) & (y_t == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': precision, 'recall': recall}


def create_side_by_side_plot(df_plot, predictions_original, predictions_filtered, 
                             y_true, stock_name, save_path, threshold):
    """Create a side-by-side comparison plot of original vs filtered predictions using rhoa."""
    
    # Create two separate plots using rhoa
    output_dir = Path(save_path).parent
    original_path = output_dir / 'temp_original.png'
    filtered_path = output_dir / 'temp_filtered.png'
    
    # Plot original predictions
    fig1 = df_plot.plots.signal(
        y_pred=predictions_original,
        y_true=y_true,
        date_col='Date',
        price_col='Close',
        threshold=threshold,
        title=f'{stock_name} - Original Predictions',
        cmap='Greens',
        save_path=str(original_path),
        show=False
    )
    plt.close(fig1)
    
    # Plot filtered predictions
    fig2 = df_plot.plots.signal(
        y_pred=predictions_filtered,
        y_true=y_true,
        date_col='Date',
        price_col='Close',
        threshold=threshold,
        title=f'{stock_name} - Pattern Filtered Predictions',
        cmap='Greens',
        save_path=str(filtered_path),
        show=False
    )
    plt.close(fig2)
    
    # Load both images and combine side-by-side
    img1 = Image.open(original_path)
    img2 = Image.open(filtered_path)
    
    # Create combined image
    total_width = img1.width + img2.width
    max_height = max(img1.height, img2.height)
    
    combined = Image.new('RGB', (total_width, max_height), 'white')
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))
    
    # Save combined image
    combined.save(save_path, dpi=(300, 300))
    
    # Clean up temporary files
    original_path.unlink()
    filtered_path.unlink()


def main():
    print("=" * 80)
    print("SIMPLE ML STOCK PREDICTOR ORCHESTRATOR")
    print("Side-by-Side Comparison: Original vs Pattern-Filtered")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Target return: {TARGET_RETURN_PERCENT}%")
    print(f"  - Prediction threshold: {PREDICTION_THRESHOLD}")
    print(f"  - Pattern window: {WINDOW_DAYS} days")
    print(f"  - Min signals: {MIN_SIGNALS}")
    print(f"  - Min slope angle: {MIN_SLOPE_ANGLE}°")
    print("=" * 80)
    print()
    
    # Override config
    config.RETURN_THRESHOLD_PERCENT = TARGET_RETURN_PERCENT
    
    # Determine holdout stock
    stock_ids = list(STOCK_URLS.keys())
    holdout_stock = stock_ids[0]
    training_stock_urls = {k: v for k, v in STOCK_URLS.items() if k != holdout_stock}
    
    print(f"Holdout stock (test): {holdout_stock}")
    print(f"Training stocks: {', '.join(training_stock_urls.keys())}")
    print()
    
    # Set up paths
    ml_dir = script_dir / 'ml_prediction'
    model_dir = ml_dir / 'saved_models' / MODEL_NAME
    output_plot_path = script_dir / f'{holdout_stock}_comparison.png'
    
    # Check if model exists
    model_exists = (model_dir / 'model.h5').exists()
    
    if model_exists:
        print(f"✓ Found existing model at: {model_dir}")
        print("  Loading saved model...")
        print()
        
        pipeline = load_pipeline(str(model_dir))
        feature_columns = pipeline['feature_columns']
    else:
        print("=" * 80)
        print("MODEL TRAINING (First Time Setup)")
        print("=" * 80)
        print()
        
        print("### Training Step 1: Loading Training Data ###")
        df_training = aggregate_stocks_from_urls(
            stock_urls=training_stock_urls,
            output_file=None,
            min_rows_per_stock=250
        )
        print()
        
        print("### Training Step 2: Generating Features ###")
        print("Creating 63 technical indicators...")
        df_features = create_features(df_training)
        print(f"✓ Generated features for {len(df_features)} rows")
        print()
        
        print("### Training Step 3: Generating Target Labels ###")
        print(f"Target: >{TARGET_RETURN_PERCENT}% return in {config.TARGET_HORIZON_DAYS} days")
        df_with_target = generate_targets(
            df_features,
            threshold_percent=TARGET_RETURN_PERCENT,
            horizon_days=config.TARGET_HORIZON_DAYS
        )
        
        valid_targets = df_with_target['target'].dropna()
        positive_pct = (valid_targets == 1).sum() / len(valid_targets) * 100
        print(f"✓ Generated targets: {positive_pct:.1f}% positive class")
        print()
        
        print("### Training Step 4: Preprocessing & Splitting ###")
        exclude_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target', 'stock_id',
                       'future_return', 'future_close', 'forward_return'}
        feature_columns = [col for col in df_with_target.columns if col not in exclude_cols]
        print(f"Using {len(feature_columns)} features")
        
        preprocessed = preprocess_pipeline(
            df_with_target,
            feature_columns=feature_columns,
            target_column='target',
            stock_col='stock_id' if 'stock_id' in df_with_target.columns else None,
            holdout_stocks=None
        )
        
        X_train = preprocessed['X_train'].values
        y_train = preprocessed['y_train'].values
        X_val = preprocessed['X_val'].values
        y_val = preprocessed['y_val'].values
        scaler = preprocessed['scaler']
        
        print_preprocessing_summary(preprocessed['stats'])
        print()
        
        print("### Training Step 5: Creating & Training Model ###")
        model = create_model(input_dim=X_train.shape[1])
        print(f"Model architecture: {config.MODEL_CONFIG['layers']} layers")
        print()
        
        model, history = train_model(
            X_train, y_train,
            X_val, y_val,
            model=model,
            config=config.TRAINING_CONFIG
        )
        print_training_summary(history)
        print()
        
        print("### Training Step 6: Saving Model ###")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        save_pipeline(
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            base_path=str(model_dir)
        )
        
        history_plot_path = model_dir / 'training_history.png'
        plot_training_history(history, save_path=str(history_plot_path))
        print(f"✓ Model saved to: {model_dir}")
        print()
        
        # Create pipeline dict for use below
        pipeline = {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns
        }
    
    print("=" * 80)
    print("PREDICTION & COMPARISON")
    print("=" * 80)
    print()
    
    # Load and prepare data
    print("### Prediction Step 1: Loading and Preparing Data ###")
    df_holdout = load_single_stock(STOCK_URLS[holdout_stock], stock_id=holdout_stock)
    print(f"✓ Loaded {len(df_holdout)} rows")
    
    df_holdout_features = create_features(df_holdout)
    df_holdout_with_targets = generate_targets(
        df_holdout_features,
        threshold_percent=TARGET_RETURN_PERCENT,
        horizon_days=config.TARGET_HORIZON_DAYS
    )
    
    # Prepare features
    target_related = {'forward_return', 'target', 'future_return', 'future_close'}
    actual_features = [col for col in feature_columns if col not in target_related]
    df_holdout_clean = df_holdout_with_targets.dropna(subset=actual_features)
    
    X_holdout = df_holdout_clean[actual_features].copy()
    if 'forward_return' in feature_columns:
        X_holdout['forward_return'] = 0.0
    X_holdout = X_holdout[feature_columns]
    
    print(f"✓ Prepared {len(df_holdout_clean)} rows for prediction")
    print()
    
    # Make original predictions
    print("### Prediction Step 2: Making Original Predictions ###")
    predictions_original, probabilities = predict_with_pipeline(
        pipeline,
        X_holdout,
        threshold=PREDICTION_THRESHOLD
    )
    
    n_signals_original = predictions_original.sum()
    print(f"✓ Generated {n_signals_original} original buy signals")
    print(f"  Signal rate: {n_signals_original/len(predictions_original)*100:.1f}%")
    print()
    
    # Apply pattern filtering
    print("### Prediction Step 3: Applying Pattern Filter ###")
    predictions_filtered, filter_stats = filter_signals_by_pattern(
        df=df_holdout_clean,
        predictions=predictions_original,
        date_col='Date',
        price_col='Close',
        window_days=WINDOW_DAYS,
        min_signals=MIN_SIGNALS,
        min_angle_degrees=MIN_SLOPE_ANGLE
    )
    
    print(f"✓ Pattern filtering complete:")
    print(f"  - Original signals: {filter_stats['original']}")
    print(f"  - Filtered signals: {filter_stats['filtered']}")
    print(f"  - Removed signals: {filter_stats['removed']}")
    if filter_stats['original'] > 0:
        print(f"  - Reduction: {filter_stats['removed']/filter_stats['original']*100:.1f}%")
    print()
    
    # Calculate metrics
    print("### Prediction Step 4: Performance Comparison ###")
    y_true = df_holdout_clean['target'].values
    y_true_available = ~np.isnan(y_true)
    
    metrics_original = calc_metrics(predictions_original, y_true, y_true_available)
    metrics_filtered = calc_metrics(predictions_filtered, y_true, y_true_available)
    
    print("\nOriginal Predictions:")
    print(f"  True Positives:  {metrics_original['tp']}")
    print(f"  False Positives: {metrics_original['fp']}")
    print(f"  Precision: {metrics_original['precision']:.1%}")
    print(f"  Recall: {metrics_original['recall']:.1%}")
    
    print("\nFiltered Predictions:")
    print(f"  True Positives:  {metrics_filtered['tp']}")
    print(f"  False Positives: {metrics_filtered['fp']}")
    print(f"  Precision: {metrics_filtered['precision']:.1%}")
    print(f"  Recall: {metrics_filtered['recall']:.1%}")
    
    if metrics_original['precision'] > 0:
        precision_change = (metrics_filtered['precision'] - metrics_original['precision']) / metrics_original['precision'] * 100
        print(f"\n✓ Precision change: {precision_change:+.1f}%")
    print()
    
    # Prepare data for plotting
    print("### Prediction Step 5: Creating Side-by-Side Visualization ###")
    
    df_plot = df_holdout_clean[['Date', 'Close']].copy()
    df_plot_predictions_original = predictions_original.copy()
    df_plot_predictions_filtered = predictions_filtered.copy()
    df_plot_y_true = y_true.copy()
    
    # Apply date filtering if specified
    if START_DATE is not None or END_DATE is not None:
        date_mask = pd.Series([True] * len(df_plot), index=df_plot.index)
        
        if START_DATE is not None:
            start_dt = pd.to_datetime(START_DATE)
            date_mask &= (df_plot['Date'] >= start_dt)
            print(f"  Filtering from: {START_DATE}")
        
        if END_DATE is not None:
            end_dt = pd.to_datetime(END_DATE)
            date_mask &= (df_plot['Date'] <= end_dt)
            print(f"  Filtering to: {END_DATE}")
        
        df_plot = df_plot[date_mask].copy()
        df_plot_predictions_original = df_plot_predictions_original[date_mask]
        df_plot_predictions_filtered = df_plot_predictions_filtered[date_mask]
        df_plot_y_true = df_plot_y_true[date_mask]
        
        print(f"  Filtered to {len(df_plot)} rows for plotting")
    
    # Create side-by-side plot
    create_side_by_side_plot(
        df_plot=df_plot,
        predictions_original=df_plot_predictions_original,
        predictions_filtered=df_plot_predictions_filtered,
        y_true=df_plot_y_true,
        stock_name=holdout_stock,
        save_path=str(output_plot_path),
        threshold=PREDICTION_THRESHOLD
    )
    
    print(f"\n✓ Comparison plot saved to: {output_plot_path}")
    print()
    
    print("=" * 80)
    print("✓ ORCHESTRATION COMPLETE!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Stock: {holdout_stock}")
    print(f"  - Original signals: {n_signals_original}")
    print(f"  - Filtered signals: {filter_stats['filtered']}")
    print(f"  - Original precision: {metrics_original['precision']:.1%}")
    print(f"  - Filtered precision: {metrics_filtered['precision']:.1%}")
    print(f"  - Comparison plot: {output_plot_path}")
    print()


if __name__ == '__main__':
    main()

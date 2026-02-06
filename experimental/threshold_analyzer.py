"""
Threshold Analyzer

Loads cached predictions from multi_stock_orchestrator and applies different
ensemble thresholds (5%, 10%, 15%) without retraining models.

This allows fast experimentation with different thresholds.

Usage:
    python threshold_analyzer.py
"""

import sys
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = script_dir / 'outputs'
PREDICTIONS_CACHE_DIR = OUTPUT_DIR / 'cache' / 'predictions'
PLOTS_DIR = OUTPUT_DIR / 'plots'
SUMMARIES_DIR = OUTPUT_DIR / 'summaries'

TARGET_PERCENT = 10.0  # Which target threshold to analyze (7 or 10)
THRESHOLDS = [5, 10, 15]  # Top percentages to test

def load_ensemble_weights():
    """Load pre-calculated precision-squared weights"""
    weights_file = script_dir / 'ensemble_weights.csv'
    
    if not weights_file.exists():
        print("❌ ERROR: ensemble_weights.csv not found!")
        sys.exit(1)
    
    df_weights = pd.read_csv(weights_file)
    weights = dict(zip(df_weights['model'], df_weights['weight']))
    return weights

def load_all_cached_predictions(target_percent):
    """Load all cached predictions for a specific target threshold"""
    if not PREDICTIONS_CACHE_DIR.exists():
        print("❌ ERROR: predictions_cache directory not found!")
        print("   Please run multi_stock_orchestrator.py first to generate predictions.")
        sys.exit(1)
    
    # Look for cache files with the specified target percentage
    pattern = f'*_target{int(target_percent)}pct_predictions.pkl'
    cache_files = list(PREDICTIONS_CACHE_DIR.glob(pattern))
    
    if len(cache_files) == 0:
        print(f"❌ ERROR: No cached predictions found for {target_percent}% target!")
        print(f"   Pattern searched: {pattern}")
        print("   Please run multi_stock_orchestrator.py first.")
        sys.exit(1)
    
    cached_data = {}
    for cache_file in cache_files:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            cached_data[data['symbol']] = data
    
    return cached_data

def generate_ensemble_predictions(all_probabilities, weights, n_future, top_percent):
    """Generate ensemble predictions at specified threshold"""
    # Calculate weighted scores
    weighted_scores = np.zeros(n_future)
    for model_name, probs in all_probabilities.items():
        weight = weights.get(model_name, 0)
        weighted_scores += probs * weight
    
    # Apply threshold
    threshold_idx = int(n_future * (1 - top_percent/100))
    threshold_idx = max(0, min(threshold_idx, n_future - 1))
    
    sorted_scores = np.sort(weighted_scores)
    threshold = sorted_scores[threshold_idx] if threshold_idx < n_future else 0
    
    predictions = (weighted_scores >= threshold).astype(int)
    
    return predictions, weighted_scores

def has_signal_in_last_5_days(predictions, dates):
    """Check if any of the last 5 days have a signal"""
    if len(predictions) < 5:
        check_range = len(predictions)
    else:
        check_range = 5
    
    last_predictions = predictions[-check_range:]
    last_dates = dates[-check_range:]
    
    signal_indices = [i for i, pred in enumerate(last_predictions) if pred == 1]
    
    if signal_indices:
        signal_dates = [last_dates[i] for i in signal_indices]
        return True, signal_dates
    
    return False, []

def create_signal_plot(df_clean, df_future, predictions, stock_name, output_filename, target_percent, ensemble_percent):
    """Create plot showing future signals for a stock"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Historical price (black line)
    ax.plot(df_clean['Date'], df_clean['Close'], 
            color='black', linewidth=1.5, label='Historical', zorder=1)
    
    # Orange separator line
    separator = df_future['Date'].min()
    ax.axvline(x=separator, color='orange', linestyle='--', 
               linewidth=2, label='Today', zorder=2)
    
    # Future price (gray dashed line)
    ax.plot(df_future['Date'], df_future['Close'], 
            color='gray', linewidth=1.5, linestyle='--', 
            label='Future', alpha=0.6, zorder=1)
    
    # Blue star signals
    signal_days = df_future[predictions == 1].copy()
    if len(signal_days) > 0:
        ax.scatter(signal_days['Date'], signal_days['Close'],
                  color='blue', marker='*', s=300, alpha=0.9,
                  label=f'Buy Signals (n={len(signal_days)})', 
                  zorder=4, edgecolors='darkblue', linewidths=1.5)
    
    # Highlight last 5 days with vertical shading
    if len(df_future) >= 5:
        last_5_dates = df_future['Date'].iloc[-5:]
        ax.axvspan(last_5_dates.min(), last_5_dates.max(), 
                   alpha=0.1, color='green', label='This Week')
    
    # Title and labels
    ax.set_title(f'{stock_name} - Target {target_percent}% | Ensemble {ensemble_percent}% Future Signals', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Price', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save to organized folder structure
    target_dir = PLOTS_DIR / f'target{int(target_percent)}pct'
    ensemble_dir = target_dir / f'ensemble{int(ensemble_percent)}pct'
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    output_path = ensemble_dir / output_filename
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    print("\n" + "="*80)
    print("THRESHOLD ANALYZER - Fast Threshold Experimentation")
    print("="*80 + "\n")
    
    # Load weights
    weights = load_ensemble_weights()
    print(f"✓ Loaded weights for {len(weights)} models\n")
    
    # Load cached predictions for the specified target percentage
    print(f"Loading cached predictions for {TARGET_PERCENT}% target...")
    cached_data = load_all_cached_predictions(TARGET_PERCENT)
    print(f"✓ Loaded predictions for {len(cached_data)} stocks\n")
    
    # Process each threshold
    for top_percent in THRESHOLDS:
        print("="*80)
        print(f"ANALYZING THRESHOLD: TOP {top_percent}%")
        print("="*80 + "\n")
        
        results = []
        plots_generated = 0
        
        for symbol, data in cached_data.items():
            print(f"[{symbol}] Applying {top_percent}% threshold...")
            
            try:
                all_probs = data['all_probabilities']
                df_clean = data['df_clean']
                df_future = data['df_future']
                
                # Generate ensemble predictions with this threshold
                predictions, scores = generate_ensemble_predictions(
                    all_probs, weights, len(df_future), top_percent
                )
                
                # Check for signals in last 5 days
                has_signal, signal_dates = has_signal_in_last_5_days(
                    predictions, df_future['Date'].values
                )
                
                if has_signal:
                    # Generate plot
                    stock_name = symbol.replace('.ST', '').replace('.', '_')
                    plot_filename = f"{stock_name}.png"
                    create_signal_plot(df_clean, df_future, predictions, stock_name, plot_filename, TARGET_PERCENT, top_percent)
                    print(f"  ✓ SIGNAL - Generated plots/target{int(TARGET_PERCENT)}pct/ensemble{top_percent}pct/{plot_filename}")
                    status = "SIGNAL"
                    plots_generated += 1
                else:
                    print(f"  No signals in last 5 days")
                    status = "No signal"
                
                # Log result
                results.append({
                    'symbol': symbol,
                    'threshold': top_percent,
                    'status': status,
                    'signals_in_future': predictions.sum(),
                    'signals_last_5_days': len(signal_dates) if has_signal else 0,
                    'signal_dates': ','.join([str(d)[:10] for d in signal_dates]) if has_signal else ''
                })
                
            except Exception as e:
                print(f"  ✗ ERROR: {str(e)[:80]}")
                results.append({
                    'symbol': symbol,
                    'threshold': top_percent,
                    'status': f"ERROR",
                    'signals_in_future': 0,
                    'signals_last_5_days': 0,
                    'signal_dates': str(e)[:50]
                })
        
        # Save summary for this threshold
        df_results = pd.DataFrame(results)
        summary_path = SUMMARIES_DIR / f'target{int(TARGET_PERCENT)}pct_ensemble{top_percent}pct.csv'
        df_results.to_csv(summary_path, index=False)
        
        stocks_with_signals = df_results[df_results['status'] == 'SIGNAL']
        
        print(f"\n{'='*80}")
        print(f"THRESHOLD {top_percent}% SUMMARY")
        print(f"{'='*80}")
        print(f"Stocks analyzed:                  {len(cached_data)}")
        print(f"Stocks with signals (last 5 days): {len(stocks_with_signals)}")
        print(f"Plots generated:                  {plots_generated}")
        print(f"Summary saved:                    {summary_path.name}")
        
        if len(stocks_with_signals) > 0:
            print(f"\nStocks with signals:")
            for _, row in stocks_with_signals.head(10).iterrows():
                print(f"  {row['symbol']:20s} - {row['signals_last_5_days']} signals")
            if len(stocks_with_signals) > 10:
                print(f"  ... and {len(stocks_with_signals) - 10} more")
        
        print()
    
    print("="*80)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAnalyzed target: {TARGET_PERCENT}% return threshold")
    print(f"\nGenerated files in outputs/ directory:")
    for thresh in THRESHOLDS:
        print(f"  summaries/target{int(TARGET_PERCENT)}pct_ensemble{thresh}pct.csv")
        print(f"  plots/target{int(TARGET_PERCENT)}pct/ensemble{thresh}pct/*.png")
    print()
    print("💡 Change TARGET_PERCENT to 7.0 to analyze the 7% target predictions")
    print()

if __name__ == '__main__':
    main()

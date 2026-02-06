"""PassiveAggressive Stock Predictor - Fixed Implementation"""
import sys
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import rhoa
import logisticregression_predictor as helper

STOCK_URL = helper.STOCK_URL
OMX30_URL = helper.OMX30_URL
STOCK_NAME = helper.STOCK_NAME
TARGET_RETURN_PERCENT = helper.TARGET_RETURN_PERCENT
PREDICTION_THRESHOLD = helper.PREDICTION_THRESHOLD
TARGET_HORIZON_DAYS = helper.TARGET_HORIZON_DAYS
N_SPLITS = helper.N_SPLITS
MIN_TRAIN_SIZE = helper.MIN_TRAIN_SIZE

PASSIVEAGGRESSIVE_PARAMS = {
    'C': 1.0,
    'max_iter': 1000,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

OUTPUT_PLOT_PATH = 'predictions_passiveaggressive.png'
OUTPUT_METRICS_PATH = 'walk_forward_metrics_passiveaggressive.csv'

def walk_forward_validation(df, features, n_splits, min_train_size):
    total_rows = len(df)
    results = []
    initial_train_idx = int(total_rows * min_train_size)
    test_size_per_split = (total_rows - initial_train_idx) // n_splits
    
    print(f"\n{'='*80}\nWALK-FORWARD VALIDATION: {n_splits} splits\n{'='*80}\n")
    
    for split_num in range(n_splits):
        print(f"\n{'='*60}\nSPLIT {split_num + 1}/{n_splits}\n{'='*60}")
        
        train_end_idx = initial_train_idx + (split_num * test_size_per_split)
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + test_size_per_split, total_rows)
        
        df_train = df.iloc[:train_end_idx].copy()
        df_test = df.iloc[test_start_idx:test_end_idx].copy()
        
        print(f"Training: {df_train['Date'].min().date()} to {df_train['Date'].max().date()} ({len(df_train)} rows)")
        print(f"Testing:  {df_test['Date'].min().date()} to {df_test['Date'].max().date()} ({len(df_test)} rows)")
        
        if len(df_test) < 10:
            continue
        
        X_train = df_train[features].values
        y_train = df_train['target'].values
        X_test = df_test[features].values
        y_test = df_test['target'].values
        
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            continue
        
        print(f"\nScaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training PassiveAggressive model...")
        model = PassiveAggressiveClassifier(**PASSIVEAGGRESSIVE_PARAMS)
        model.fit(X_train_scaled, y_train)
        
        # PA doesn't have predict_proba, use decision_function
        decision_function = model.decision_function(X_test_scaled)
        y_pred_proba = 1 / (1 + np.exp(-decision_function))
        y_pred = (y_pred_proba >= PREDICTION_THRESHOLD).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0
        
        n_signals = y_pred.sum()
        
        results.append({
            'split': split_num + 1,
            'train_start': df_train['Date'].min(),
            'train_end': df_train['Date'].max(),
            'test_start': df_test['Date'].min(),
            'test_end': df_test['Date'].max(),
            'n_train': len(df_train),
            'n_test': len(df_test),
            'n_signals': n_signals,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'model': model,
            'scaler': scaler,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'df_test': df_test
        })
        
        print(f"\n📊 Results: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    return results

def main():
    print("="*80 + "\nPASSIVEAGGRESSIVE STOCK PREDICTOR\n" + "="*80)
    
    features_file = script_dir / 'selected_features.txt'
    with open(features_file, 'r') as f:
        selected_features = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(selected_features)} features\n")
    
    print("### Loading Data ###")
    df_stock = helper.load_stock_data(STOCK_URL, stock_id=STOCK_NAME)
    df_omx30 = helper.load_stock_data(OMX30_URL, stock_id='OMX30')
    
    print("\n### Feature Engineering ###")
    df_features = helper.create_features(df_stock, index_df=df_omx30)
    
    print("\n### Target Generation ###")
    df_with_target, df_future = helper.generate_targets(df_features, TARGET_RETURN_PERCENT, TARGET_HORIZON_DAYS)
    
    print("\n### Feature Selection ###")
    available_features = [f for f in selected_features if f in df_with_target.columns]
    df_clean = df_with_target[['Date', 'Close', 'target'] + available_features].copy().dropna()
    df_future_clean = df_future[['Date', 'Close'] + available_features].copy()
    print(f"Clean data: {len(df_clean)} rows")
    
    results = walk_forward_validation(df=df_clean, features=available_features, n_splits=N_SPLITS, min_train_size=MIN_TRAIN_SIZE)
    
    if len(results) == 0:
        return
    
    print(f"\n{'='*80}\nAGGREGATE RESULTS\n{'='*80}")
    avg_precision = np.mean([r['precision'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    avg_auc = np.mean([r['auc'] for r in results])
    
    print(f"\nAvg Precision: {avg_precision:.4f}")
    print(f"Avg Recall:    {avg_recall:.4f}")
    print(f"Avg F1:        {avg_f1:.4f}")
    print(f"Avg AUC:       {avg_auc:.4f}")
    
    metrics_df = pd.DataFrame([{'split': r['split'], 'precision': r['precision'], 'recall': r['recall'], 'f1': r['f1'], 'auc': r['auc']} for r in results])
    metrics_path = script_dir / OUTPUT_METRICS_PATH
    metrics_df.to_csv(metrics_path, index=False)
    
    # Future predictions
    X_all = df_clean[available_features].values
    y_all = df_clean['target'].values
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)
    final_model = PassiveAggressiveClassifier(**PASSIVEAGGRESSIVE_PARAMS)
    final_model.fit(X_all_scaled, y_all)
    
    df_future_for_pred = df_future_clean.dropna()
    if len(df_future_for_pred) > 0:
        X_future = df_future_for_pred[available_features].values
        X_future_scaled = final_scaler.transform(X_future)
        decision_function = final_model.decision_function(X_future_scaled)
        y_future_proba = 1 / (1 + np.exp(-decision_function))
        y_future_pred = (y_future_proba >= PREDICTION_THRESHOLD).astype(int)
        print(f"\n✅ Future: {y_future_pred.sum()} signals")
    else:
        df_future_for_pred = None
        y_future_pred = None
    
    # Plot
    df_plot_list = [r['df_test'][['Date', 'Close']].copy().assign(y_pred=r['y_pred'], y_test=r['y_test']) for r in results]
    df_historical = pd.concat(df_plot_list, ignore_index=True)
    
    output_path = script_dir / OUTPUT_PLOT_PATH
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df_historical['Date'], df_historical['Close'], color='black', linewidth=1.5, label='Historical Price', zorder=1)
    
    buy_signals = df_historical[df_historical['y_pred'] == 1]
    correct_signals = buy_signals[buy_signals['y_test'] == 1]
    incorrect_signals = buy_signals[buy_signals['y_test'] == 0]
    
    if len(correct_signals) > 0:
        ax.scatter(correct_signals['Date'], correct_signals['Close'], color='green', marker='^', s=150, alpha=0.8,
                  label=f'Correct (n={len(correct_signals)})', zorder=3, edgecolors='darkgreen', linewidths=1.5)
    if len(incorrect_signals) > 0:
        ax.scatter(incorrect_signals['Date'], incorrect_signals['Close'], color='red', marker='v', s=150, alpha=0.8,
                  label=f'Incorrect (n={len(incorrect_signals)})', zorder=3, edgecolors='darkred', linewidths=1.5)
    
    if df_future_for_pred is not None and len(df_future_for_pred) > 0:
        ax.axvline(x=df_future_for_pred['Date'].min(), color='orange', linestyle='--', linewidth=2, label='Validation | Future', zorder=2)
        ax.plot(df_future_for_pred['Date'], df_future_for_pred['Close'], color='gray', linewidth=1.5, linestyle='--', label='Future', zorder=1, alpha=0.6)
        
        future_signals = df_future_for_pred[y_future_pred == 1]
        if len(future_signals) > 0:
            ax.scatter(future_signals['Date'], future_signals['Close'], color='blue', marker='*', s=300, alpha=0.9,
                      label=f'Future (n={len(future_signals)})', zorder=4, edgecolors='darkblue', linewidths=1.5)
    
    ax.set_title(f'{STOCK_NAME} - PassiveAggressive Buy Signals', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nVisualization: {output_path}")
    print("="*80 + "\nCOMPLETE! ✅\n" + "="*80)

if __name__ == '__main__':
    main()

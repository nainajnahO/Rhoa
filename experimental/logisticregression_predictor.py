"""
Logistic Regression Stock Predictor - Fixed Implementation

Fast linear baseline model with proper time series validation.

Usage:
    python experimental/logisticregression_predictor.py
"""

import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import rhoa

# Configuration
STOCK_URL = 'https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=1940537480'
OMX30_URL = 'https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=1961884031'
STOCK_NAME = 'TEST'

TARGET_RETURN_PERCENT = 10.0
PREDICTION_THRESHOLD = 0.5
TARGET_HORIZON_DAYS = 42

N_SPLITS = 5
MIN_TRAIN_SIZE = 0.50

LOGISTICREGRESSION_PARAMS = {
    'C': 1.0,
    'penalty': 'l2',
    'solver': 'lbfgs',
    'class_weight': 'balanced',
    'max_iter': 1000,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': 0
}

OUTPUT_PLOT_PATH = 'predictions_logisticregression.png'
OUTPUT_METRICS_PATH = 'walk_forward_metrics_logisticregression.csv'
START_DATE = None
END_DATE = None


def load_stock_data(url, stock_id=None):
    print(f"Loading data from Google Sheets...")
    df = pd.read_csv(url)
    
    column_mapping = {
        'date': 'Date', 'DATE': 'Date',
        'open': 'Open', 'OPEN': 'Open',
        'high': 'High', 'HIGH': 'High',
        'low': 'Low', 'LOW': 'Low',
        'close': 'Close', 'CLOSE': 'Close', 'Close*': 'Close', 'Adj Close': 'Close',
        'volume': 'Volume', 'VOLUME': 'Volume', 'Vol.': 'Volume',
    }
    df = df.rename(columns=column_mapping)
    
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if missing_cols := [col for col in required_cols if col not in df.columns]:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df[required_cols].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    df = df.drop_duplicates(subset=['Date'], keep='last')
    
    if stock_id:
        df['stock_id'] = stock_id
    
    print(f"  Loaded {len(df)} rows")
    return df


def create_features(df, index_df=None):
    print("Generating features...")
    df = df.copy()
    
    df['return_1d'] = df['Close'].pct_change() * 100
    df['return_5d'] = df['Close'].pct_change(5) * 100
    df['intraday_change'] = ((df['Close'] - df['Open']) / df['Open']) * 100
    df['high_low_range'] = ((df['High'] - df['Low']) / df['Close']) * 100
    df['close_position'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'])).fillna(0.5)
    df['close_low_diff'] = df['Close'] - df['Low']
    df['high_close_diff'] = df['High'] - df['Close']
    df['price_diff_10d'] = df['Close'].diff(10)
    
    df['high_52w'] = df['High'].rolling(window=252).max()
    df['dist_from_52w_high'] = ((df['Close'] - df['high_52w']) / df['high_52w']) * 100
    
    df['sma_20'] = df['Close'].indicators.sma(window_size=20)
    df['close_to_sma_20'] = ((df['Close'] - df['sma_20']) / df['sma_20']) * 100
    
    df['ewma_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['ewma_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['ewma_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['close_to_ewma_5'] = ((df['Close'] - df['ewma_5']) / df['ewma_5']) * 100
    df['close_to_ewma_10'] = ((df['Close'] - df['ewma_10']) / df['ewma_10']) * 100
    df['close_to_ewma_50'] = ((df['Close'] - df['ewma_50']) / df['ewma_50']) * 100
    
    df['rsi'] = df['Close'].indicators.rsi(window_size=14)
    
    adx_result = df['Close'].indicators.adx(high=df['High'], low=df['Low'], window_size=14)
    df['adx'] = adx_result['ADX']
    df['adx_plus_di'] = adx_result['+DI']
    df['adx_minus_di'] = adx_result['-DI']
    df['adx_di_diff'] = df['adx_plus_di'] - df['adx_minus_di']
    
    stoch_result = df['Close'].indicators.stochastic(high=df['High'], low=df['Low'], k_window=14, d_window=3)
    df['stoch_k'] = stoch_result['%K']
    df['stoch_d'] = stoch_result['%D']
    
    df['atr'] = df['Close'].indicators.atr(high=df['High'], low=df['Low'], window_size=14)
    df['atr_percent'] = (df['atr'] / df['Close']) * 100
    df['ewmstd_10'] = df['Close'].ewm(span=10).std()
    
    macd_result = df['Close'].indicators.macd(short_window=12, long_window=26, signal_window=9)
    df['macd_histogram'] = macd_result['histogram']
    
    df['psar'] = df['Close'].indicators.parabolic_sar(high=df['High'], low=df['Low'], af_start=0.02, af_increment=0.02, af_maximum=0.2)
    df['close_to_psar'] = ((df['Close'] - df['psar']) / df['psar']) * 100
    
    df['volume_change'] = df['Volume'].pct_change() * 100
    df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
    df['price_volume'] = df['return_1d'] * np.sign(df['volume_change'])
    df['volume_acceleration'] = df['volume_change'].diff()
    
    df['price_direction'] = np.where(df['Close'] > df['Close'].shift(1), 1, 0)
    df['consecutive_up'] = (df['price_direction'].groupby((df['price_direction'] != df['price_direction'].shift()).cumsum()).cumsum() * df['price_direction']).astype(float)
    df['consecutive_down'] = ((1 - df['price_direction']).groupby(((1 - df['price_direction']) != (1 - df['price_direction']).shift()).cumsum()).cumsum() * (1 - df['price_direction'])).astype(float)
    
    df['momentum_acceleration'] = df['return_1d'].diff()
    df['return_skew_20d'] = df['return_1d'].rolling(window=20).skew()
    
    df['return_1d_lag_2'] = df['return_1d'].shift(2)
    df['return_1d_lag_3'] = df['return_1d'].shift(3)
    df['return_1d_lag_5'] = df['return_1d'].shift(5)
    df['rsi_lag_2'] = df['rsi'].shift(2)
    
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    
    if index_df is not None:
        print("  Adding market correlation features...")
        index_df = index_df.copy()
        index_df = index_df.rename(columns={'Close': 'OMX30_Close'})
        index_df['OMX30_return_1d'] = index_df['OMX30_Close'].pct_change() * 100
        index_df['OMX30_return_5d'] = index_df['OMX30_Close'].pct_change(5) * 100
        
        df['DateOnly'] = df['Date'].dt.normalize()
        index_df['DateOnly'] = index_df['Date'].dt.normalize()
        
        rows_before_merge = len(df)
        df = df.merge(index_df[['DateOnly', 'OMX30_Close', 'OMX30_return_1d', 'OMX30_return_5d']], on='DateOnly', how='inner')
        df = df.drop('DateOnly', axis=1)
        rows_after_merge = len(df)
        
        if rows_before_merge > rows_after_merge:
            print(f"  Removed {rows_before_merge - rows_after_merge} days with missing OMX30 data")
        
        df['relative_to_OMX30_5d'] = df['return_5d'] - df['OMX30_return_5d']
        df['corr_to_OMX30_20d'] = df['return_1d'].rolling(window=20).corr(df['OMX30_return_1d'])
        df['corr_to_OMX30_60d'] = df['return_1d'].rolling(window=60).corr(df['OMX30_return_1d'])
        
        def calculate_beta(stock_returns, market_returns):
            if len(stock_returns) < 20:
                return np.nan
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            return covariance / market_variance if market_variance > 0 else np.nan
        
        df['beta_to_OMX30'] = df['return_1d'].rolling(window=60).apply(
            lambda x: calculate_beta(x, df.loc[x.index, 'OMX30_return_1d']), raw=False
        )
    
    print(f"  Generated {len(df.columns)} total columns")
    return df


def generate_targets(df, threshold_percent, horizon_days):
    print(f"Generating target labels (>{threshold_percent}% return in {horizon_days} days)...")
    df = df.copy()
    
    df['future_close'] = df['Close'].shift(-horizon_days)
    df['forward_return'] = ((df['future_close'] - df['Close']) / df['Close']) * 100
    df['target'] = (df['forward_return'] >= threshold_percent).astype(int)
    
    rows_before = len(df)
    df_future = df.iloc[-horizon_days:].copy()
    df_with_targets = df.iloc[:-horizon_days].copy()
    
    print(f"  Removed last {horizon_days} rows (unknowable targets)")
    print(f"  Rows: {rows_before} -> {len(df_with_targets)} (training)")
    print(f"  Saved {len(df_future)} rows for future predictions")
    
    valid_targets = df_with_targets['target'].dropna()
    if len(valid_targets) > 0:
        positive_pct = (valid_targets == 1).sum() / len(valid_targets) * 100
        print(f"  Positive class: {positive_pct:.1f}%")
    
    return df_with_targets, df_future


def walk_forward_validation(df, features, n_splits, min_train_size):
    total_rows = len(df)
    results = []
    
    initial_train_idx = int(total_rows * min_train_size)
    remaining_rows = total_rows - initial_train_idx
    test_size_per_split = remaining_rows // n_splits
    
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD VALIDATION: {n_splits} splits")
    print(f"  Total rows: {total_rows}")
    print(f"  Initial training: {initial_train_idx} rows ({min_train_size:.0%})")
    print(f"  Test size per split: ~{test_size_per_split} rows")
    print(f"{'='*80}\n")
    
    for split_num in range(n_splits):
        print(f"\n{'='*60}")
        print(f"SPLIT {split_num + 1}/{n_splits}")
        print(f"{'='*60}")
        
        train_end_idx = initial_train_idx + (split_num * test_size_per_split)
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + test_size_per_split, total_rows)
        
        df_train = df.iloc[:train_end_idx].copy()
        df_test = df.iloc[test_start_idx:test_end_idx].copy()
        
        train_start_date = df_train['Date'].min()
        train_end_date = df_train['Date'].max()
        test_start_date = df_test['Date'].min()
        test_end_date = df_test['Date'].max()
        
        print(f"Training period: {train_start_date.date()} to {train_end_date.date()} ({len(df_train)} rows)")
        print(f"Testing period:  {test_start_date.date()} to {test_end_date.date()} ({len(df_test)} rows)")
        
        if len(df_test) < 10:
            print("⚠️  Test set too small, skipping split")
            continue
        
        X_train = df_train[features].values
        y_train = df_train['target'].values
        X_test = df_test[features].values
        y_test = df_test['target'].values
        
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            print("⚠️  NaN values detected, skipping split")
            continue
        
        print(f"\nScaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training LogisticRegression model...")
        model = LogisticRegression(**LOGISTICREGRESSION_PARAMS)
        model.fit(X_train_scaled, y_train)
        
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= PREDICTION_THRESHOLD).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0
        
        n_signals = y_pred.sum()
        n_correct_signals = tp
        signal_accuracy = n_correct_signals / n_signals if n_signals > 0 else 0.0
        
        result = {
            'split': split_num + 1,
            'train_start': train_start_date,
            'train_end': train_end_date,
            'test_start': test_start_date,
            'test_end': test_end_date,
            'n_train': len(df_train),
            'n_test': len(df_test),
            'n_signals': n_signals,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'signal_accuracy': signal_accuracy,
            'model': model,
            'scaler': scaler,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'df_test': df_test
        }
        results.append(result)
        
        print(f"\n📊 Split {split_num + 1} Results:")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        print(f"  Buy Signals: {n_signals} ({n_signals/len(df_test)*100:.1f}% of test set)")
        print(f"  Signal Accuracy: {signal_accuracy:.1%} ({n_correct_signals}/{n_signals} signals correct)")
    
    return results


def main():
    print("="*80)
    print("LOGISTIC REGRESSION STOCK PREDICTOR - FIXED IMPLEMENTATION")
    print("="*80)
    print(f"Configuration:")
    print(f"  Stock: {STOCK_NAME}")
    print(f"  Target: >{TARGET_RETURN_PERCENT}% return in {TARGET_HORIZON_DAYS} days")
    print(f"  Walk-forward splits: {N_SPLITS}")
    print("="*80)
    print()
    
    features_file = script_dir / 'selected_features.txt'
    with open(features_file, 'r') as f:
        selected_features = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(selected_features)} selected features\n")
    
    print("### Step 1: Loading Data ###")
    df_stock = load_stock_data(STOCK_URL, stock_id=STOCK_NAME)
    df_omx30 = load_stock_data(OMX30_URL, stock_id='OMX30')
    print()
    
    print("### Step 2: Feature Engineering ###")
    df_features = create_features(df_stock, index_df=df_omx30)
    print()
    
    print("### Step 3: Target Generation (ANTI-LEAKAGE) ###")
    df_with_target, df_future = generate_targets(df_features, TARGET_RETURN_PERCENT, TARGET_HORIZON_DAYS)
    print()
    
    print("### Step 4: Feature Selection ###")
    available_features = [f for f in selected_features if f in df_with_target.columns]
    print(f"Using {len(available_features)} features")
    
    df_clean = df_with_target[['Date', 'Close', 'target'] + available_features].copy()
    df_future_clean = df_future[['Date', 'Close'] + available_features].copy()
    
    rows_before = len(df_clean)
    df_clean = df_clean.dropna()
    rows_after = len(df_clean)
    print(f"Removed NaN rows: {rows_before} -> {rows_after}\n")
    
    print("### Step 5: Walk-Forward Validation ###")
    results = walk_forward_validation(df=df_clean, features=available_features, n_splits=N_SPLITS, min_train_size=MIN_TRAIN_SIZE)
    
    if len(results) == 0:
        print("❌ No valid splits. Exiting.")
        return
    
    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS")
    print(f"{'='*80}")
    
    avg_precision = np.mean([r['precision'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    avg_auc = np.mean([r['auc'] for r in results])
    
    print(f"\nAverage Metrics:")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall:    {avg_recall:.4f}")
    print(f"  F1-Score:  {avg_f1:.4f}")
    print(f"  AUC:       {avg_auc:.4f}\n")
    
    metrics_df = pd.DataFrame([{
        'split': r['split'],
        'precision': r['precision'],
        'recall': r['recall'],
        'f1': r['f1'],
        'auc': r['auc']
    } for r in results])
    
    metrics_path = script_dir / OUTPUT_METRICS_PATH
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")
    
    # Future predictions
    X_all = df_clean[available_features].values
    y_all = df_clean['target'].values
    
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)
    
    final_model = LogisticRegression(**LOGISTICREGRESSION_PARAMS)
    final_model.fit(X_all_scaled, y_all)
    
    df_future_for_pred = df_future_clean.dropna()
    
    if len(df_future_for_pred) > 0:
        X_future = df_future_for_pred[available_features].values
        X_future_scaled = final_scaler.transform(X_future)
        y_future_proba = final_model.predict_proba(X_future_scaled)[:, 1]
        y_future_pred = (y_future_proba >= PREDICTION_THRESHOLD).astype(int)
        
        print(f"\n✅ Future predictions: {y_future_pred.sum()} buy signals")
    else:
        df_future_for_pred = None
        y_future_pred = None
    
    # Visualization
    df_plot_list = []
    for r in results:
        df_test = r['df_test'][['Date', 'Close']].copy()
        df_test['y_pred'] = r['y_pred']
        df_test['y_test'] = r['y_test']
        df_plot_list.append(df_test)
    
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
        separator_date = df_future_for_pred['Date'].min()
        ax.axvline(x=separator_date, color='orange', linestyle='--', linewidth=2, label='Validation | Future', zorder=2)
        ax.plot(df_future_for_pred['Date'], df_future_for_pred['Close'], color='gray', linewidth=1.5, linestyle='--', label='Future Price', zorder=1, alpha=0.6)
        
        future_signals = df_future_for_pred[y_future_pred == 1]
        if len(future_signals) > 0:
            ax.scatter(future_signals['Date'], future_signals['Close'], color='blue', marker='*', s=300, alpha=0.9,
                      label=f'Future (n={len(future_signals)})', zorder=4, edgecolors='darkblue', linewidths=1.5)
    
    ax.set_title(f'{STOCK_NAME} - Logistic Regression Buy Signals', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Visualization: {output_path}")
    print("="*80)
    print("COMPLETE! ✅")
    print("="*80)


if __name__ == '__main__':
    main()

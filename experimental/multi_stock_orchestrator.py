"""
Multi-Stock Orchestrator with Yahoo Finance

Trains models on multiple Stockholm stocks and caches predictions for later analysis.

Features:
- Data caching for fast re-runs (Yahoo Finance)
- Trains all 29 ML models on each stock
- Caches predictions for both 7% and 10% return targets
- Use threshold_analyzer.py to analyze signals with different ensemble thresholds

Usage:
    1. Ensure stocks.csv exists with stock symbols
    2. Run: python experimental/multi_stock_orchestrator.py
    3. Use threshold_analyzer.py to analyze cached predictions
"""

import sys
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import helpers
import logisticregression_predictor as helper
from yahoo_finance_loader import fetch_stock_data, get_cache_info

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import ensemble components
from ensemble_voting_predictor import MODELS, create_sequences

# Configuration
TARGET_RETURN_THRESHOLDS = [7.0, 10.0]  # Train on both 7% and 10% targets
TARGET_HORIZON_DAYS = 42
TIMESTEPS = 5

# Output directories
OUTPUT_DIR = script_dir / 'outputs'
PREDICTIONS_CACHE_DIR = OUTPUT_DIR / 'cache' / 'predictions'
SUMMARIES_DIR = OUTPUT_DIR / 'summaries'

# Create directories
PREDICTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiting (not needed for Yahoo Finance, but keeping for compatibility)
last_api_call_time = 0
RATE_LIMIT_DELAY = 0  # No rate limiting needed for Yahoo Finance

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_stock_list(filepath='stocks.csv'):
    """Load list of Stockholm stock symbols from CSV or TXT file"""
    stocks_file = script_dir / filepath
    if not stocks_file.exists():
        # Try fallback to stockholm_stocks.txt
        fallback_file = script_dir / 'stockholm_stocks.txt'
        if fallback_file.exists():
            stocks_file = fallback_file
            filepath = 'stockholm_stocks.txt'
        else:
            print(f"❌ ERROR: {filepath} not found!")
            print("   Please create stocks.csv or stockholm_stocks.txt with stock symbols")
            sys.exit(1)
    
    # Check if it's a CSV file
    if filepath.endswith('.csv'):
        df = pd.read_csv(stocks_file)
        # Try to find the ticker column (case insensitive)
        ticker_col = None
        for col in df.columns:
            if col.lower() in ['ticker', 'symbol', 'stock', 'ticker symbol']:
                ticker_col = col
                break
        
        if ticker_col is None:
            # If no obvious ticker column, use first column
            ticker_col = df.columns[0]
        
        stocks = df[ticker_col].dropna().str.strip().tolist()
    else:
        # Text file format (one symbol per line)
        with open(stocks_file, 'r') as f:
            stocks = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    return stocks

def rate_limited_fetch(symbol):
    """Fetch stock data (no rate limiting needed for Yahoo Finance)"""
    # Fetch data (no rate limiting needed for Yahoo Finance)
    data = fetch_stock_data(symbol)
    return data

# ============================================================================
# MODEL TRAINING AND PREDICTION
# ============================================================================

def train_and_predict_single_model(model_name, model_type, script_name, 
                                    X_train_scaled, y_train, X_future_scaled, n_features):
    """Train a single model and get predictions on future data"""
    try:
        # Import model module
        module = __import__(script_name)
        
        if model_type == 'keras_seq':
            # Create 3D sequences for RNN/CNN models
            X_train_seq = create_sequences(X_train_scaled, TIMESTEPS)
            y_train_seq = y_train[TIMESTEPS-1:]
            X_future_seq = create_sequences(X_future_scaled, TIMESTEPS)
            
            model = module.create_model(n_features)
            model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0)
            probs = model.predict(X_future_seq, verbose=0).flatten()
            
            # Pad first timesteps-1 entries
            full_probs = np.zeros(len(X_future_scaled))
            full_probs[TIMESTEPS-1:] = probs
            
        elif model_type == 'keras':
            # Deep MLP - 2D input
            model = module.create_model(n_features)
            model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
            full_probs = model.predict(X_future_scaled, verbose=0).flatten()
            
        else:  # sklearn models
            # Instantiate the correct sklearn model
            if 'xgboost' in script_name:
                import xgboost as xgb
                model = xgb.XGBClassifier(**module.XGBOOST_PARAMS)
            elif 'lightgbm' in script_name:
                import lightgbm as lgb
                model = lgb.LGBMClassifier(**module.LIGHTGBM_PARAMS)
            elif 'catboost' in script_name:
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(**module.CATBOOST_PARAMS)
            elif 'randomforest' in script_name:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**module.RANDOMFOREST_PARAMS)
            elif 'adaboost' in script_name:
                from sklearn.ensemble import AdaBoostClassifier
                model = AdaBoostClassifier(**module.ADABOOST_PARAMS)
            elif 'gradientboosting' in script_name:
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(**module.GRADIENTBOOSTING_PARAMS)
            elif 'extratrees' in script_name:
                from sklearn.ensemble import ExtraTreesClassifier
                model = ExtraTreesClassifier(**module.EXTRATREES_PARAMS)
            elif 'bagging' in script_name:
                from sklearn.ensemble import BaggingClassifier
                model = BaggingClassifier(**module.BAGGING_PARAMS)
            elif 'mlp_predictor' in script_name and 'deepmlp' not in script_name:
                from sklearn.neural_network import MLPClassifier
                model = MLPClassifier(**module.MLP_PARAMS)
            elif 'svm' in script_name:
                from sklearn.svm import SVC
                model = SVC(**module.SVM_PARAMS)
            elif 'logisticregression' in script_name:
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(**module.LOGISTICREGRESSION_PARAMS)
            elif 'naivebayes' in script_name:
                from sklearn.naive_bayes import GaussianNB
                model = GaussianNB(**module.NAIVEBAYES_PARAMS)
            elif 'ridgeclassifier' in script_name:
                from sklearn.linear_model import RidgeClassifier
                model = RidgeClassifier(**module.RIDGECLASSIFIER_PARAMS)
            elif 'sgdclassifier' in script_name:
                from sklearn.linear_model import SGDClassifier
                model = SGDClassifier(**module.SGDCLASSIFIER_PARAMS)
            elif 'lda' in script_name:
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                model = LinearDiscriminantAnalysis(**module.LDA_PARAMS)
            elif 'qda' in script_name:
                from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
                model = QuadraticDiscriminantAnalysis(**module.QDA_PARAMS)
            elif 'perceptron' in script_name:
                from sklearn.linear_model import Perceptron
                model = Perceptron(**module.PERCEPTRON_PARAMS)
            elif 'passiveaggressive' in script_name:
                from sklearn.linear_model import PassiveAggressiveClassifier
                model = PassiveAggressiveClassifier(**module.PASSIVEAGGRESSIVE_PARAMS)
            elif 'histgradientboosting' in script_name:
                from sklearn.ensemble import HistGradientBoostingClassifier
                model = HistGradientBoostingClassifier(**module.HISTGRADIENTBOOSTING_PARAMS)
            elif 'knn' in script_name:
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier(**module.KNN_PARAMS)
            elif 'decisiontree' in script_name:
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier(**module.DECISIONTREE_PARAMS)
            else:
                return None
            
            # Train and predict
            model.fit(X_train_scaled, y_train)
            
            if hasattr(model, 'predict_proba'):
                full_probs = model.predict_proba(X_future_scaled)[:, 1]
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function(X_future_scaled)
                full_probs = 1 / (1 + np.exp(-decision))
            else:
                full_probs = model.predict(X_future_scaled).astype(float)
        
        return full_probs
        
    except Exception as e:
        # Silent failure - just return None
        return None

def train_all_models_and_predict(df_clean, df_future, features):
    """Train all 29 models on historical data and predict on future data"""
    # Prepare training data
    X_train = df_clean[features].values
    y_train = df_clean['target'].values
    X_future = df_future[features].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_future_scaled = scaler.transform(X_future)
    
    n_features = X_train.shape[1]
    
    # Collect predictions from all models
    all_probabilities = {}
    
    for model_name, _, model_type, script_name in MODELS:
        probs = train_and_predict_single_model(
            model_name, model_type, script_name,
            X_train_scaled, y_train, X_future_scaled, n_features
        )
        
        if probs is not None:
            all_probabilities[model_name] = probs
    
    return all_probabilities

def save_predictions_cache(symbol, target_percent, all_probabilities, df_clean, df_future):
    """Save predictions and data to cache for later threshold analysis"""
    cache_file = PREDICTIONS_CACHE_DIR / f"{symbol.replace('.', '_').replace('-', '_')}_target{int(target_percent)}pct_predictions.pkl"
    
    cache_data = {
        'symbol': symbol,
        'target_percent': target_percent,
        'all_probabilities': all_probabilities,
        'df_clean': df_clean[['Date', 'Close']].copy(),
        'df_future': df_future[['Date', 'Close']].copy(),
        'timestamp': pd.Timestamp.now()
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

def load_predictions_cache(symbol, target_percent):
    """Load cached predictions if available"""
    cache_file = PREDICTIONS_CACHE_DIR / f"{symbol.replace('.', '_').replace('-', '_')}_target{int(target_percent)}pct_predictions.pkl"
    
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

# ============================================================================
# ENSEMBLE PREDICTION
# ============================================================================

# Ensemble prediction, signal detection, and plotting removed
# Use threshold_analyzer.py for all analysis and visualization

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("MULTI-STOCK PREDICTION CACHING")
    print("="*80 + "\n")
    
    print("This script trains all 29 models and caches predictions.")
    print("Use threshold_analyzer.py afterwards to analyze signals.\n")
    
    # Load stock list
    stocks = load_stock_list()
    print(f"Loaded {len(stocks)} stocks from stocks.csv\n")
    
    # Load selected features
    features_file = script_dir / 'selected_features.txt'
    with open(features_file, 'r') as f:
        selected_features = [line.strip() for line in f if line.strip()]
    print(f"Using {len(selected_features)} features\n")
    
    # Cache info
    print(f"Cache status: {get_cache_info()}\n")
    
    # Fetch OMX30 once
    print("="*80)
    print("FETCHING OMX30 INDEX")
    print("="*80)
    df_omx30 = None
    for omx_symbol in ['^OMX', 'OMXS30.ST', 'OMX.ST']:
        try:
            df_omx30 = rate_limited_fetch(omx_symbol)
            print(f"✓ Successfully fetched {omx_symbol}")
            print(f"  Rows: {len(df_omx30)}, Date range: {df_omx30['Date'].min()} to {df_omx30['Date'].max()}\n")
            break
        except Exception as e:
            print(f"  ✗ Failed to fetch {omx_symbol}: {str(e)[:80]}")
    
    if df_omx30 is None:
        print("⚠️  WARNING: Could not fetch OMX30. Proceeding without market correlation features.\n")
    
    # Results tracking
    results = []
    
    # Process each stock
    print("="*80)
    print("PROCESSING STOCKS")
    print("="*80 + "\n")
    
    for i, symbol in enumerate(stocks, 1):
        print(f"[{i}/{len(stocks)}] {symbol}")
        
        try:
            # Fetch stock data (with caching and rate limiting)
            df_stock = rate_limited_fetch(symbol)
            print(f"  ✓ Data: {len(df_stock)} rows")
            
            # Generate features (once for both target thresholds)
            df_features = helper.create_features(df_stock, index_df=df_omx30)
            
            # Filter to available features
            available_features = [f for f in selected_features if f in df_features.columns]
            
            # Process each target threshold
            for target_percent in TARGET_RETURN_THRESHOLDS:
                print(f"\n  --- Training with {target_percent}% target ---")
                
                # Generate targets
                df_with_target, df_future = helper.generate_targets(df_features, target_percent, TARGET_HORIZON_DAYS)
            
                # Clean data
                df_clean = df_with_target[['Date', 'Close', 'target'] + available_features].copy()
                df_future_clean = df_future[['Date', 'Close'] + available_features].copy()
                
                # Handle inf values (replace with NaN)
                df_clean[available_features] = df_clean[available_features].replace([np.inf, -np.inf], np.nan)
                df_future_clean[available_features] = df_future_clean[available_features].replace([np.inf, -np.inf], np.nan)
                
                # Drop rows with NaN
                rows_before = len(df_clean)
                df_clean = df_clean.dropna()
                rows_after = len(df_clean)
                
                df_future_clean = df_future_clean.dropna()
                
                if rows_before - rows_after > 0:
                    print(f"  Cleaned {rows_before - rows_after} rows with inf/nan values")
                
                if len(df_clean) < 100:
                    print(f"  ⚠️  Insufficient data after cleaning: {len(df_clean)} rows - skipping {target_percent}% target")
                    continue
                
                print(f"  ✓ Features: {len(available_features)}, Train: {len(df_clean)}, Future: {len(df_future_clean)}")
                
                # Train models and predict
                print(f"  Training 29 models...")
                all_probs = train_all_models_and_predict(df_clean, df_future_clean, available_features)
                print(f"  ✓ Models trained: {len(all_probs)}/29")
                
                # Save predictions to cache (with target_percent in filename)
                save_predictions_cache(symbol, target_percent, all_probs, df_clean, df_future_clean)
                print(f"  ✓ Predictions cached for {target_percent}% target")
            
            # Record successful caching
            results.append({
                'symbol': symbol,
                'status': 'Success',
                'data_rows': len(df_stock),
                'features_used': len(available_features),
                'targets_cached': '7% and 10%'
            })
            
        except Exception as e:
            error_msg = str(e)[:80]
            print(f"  ✗ ERROR: {error_msg}")
            results.append({
                'symbol': symbol,
                'status': 'ERROR',
                'data_rows': 0,
                'features_used': 0,
                'targets_cached': error_msg
            })
        
        print()
    
    # Save summary
    df_summary = pd.DataFrame(results)
    summary_path = SUMMARIES_DIR / 'multi_stock_results.csv'
    df_summary.to_csv(summary_path, index=False)
    
    # Print final summary
    stocks_success = df_summary[df_summary['status'] == 'Success']
    stocks_with_errors = df_summary[df_summary['status'] == 'ERROR']
    
    print("="*80)
    print("PREDICTION CACHING COMPLETE")
    print("="*80)
    print(f"Total stocks processed:           {len(stocks)}")
    print(f"Successfully cached:              {len(stocks_success)}")
    print(f"Errors:                           {len(stocks_with_errors)}")
    print(f"\nSummary saved:                    outputs/summaries/{summary_path.name}")
    print()
    
    print("💡 Next step: Run threshold_analyzer.py to analyze signals")
    print("   - Applies ensemble voting with precision-squared weights")
    print("   - Tests 5%, 10%, 15% ensemble thresholds")
    print("   - Generates plots for stocks with signals")
    print("   - Can switch between 7% and 10% target analysis instantly")
    print()
    
    print(f"Cache status: {get_cache_info()}")
    print("="*80)

if __name__ == '__main__':
    main()

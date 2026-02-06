"""
Stock Feasibility Orchestrator

Trains all 29 ML models on a stock and predicts future buy signals 
for the next 42 days to determine purchase feasibility.

Usage:
    1. Edit STOCK_URL and STOCK_NAME variables below
    2. Run: python experimental/orchestrator.py
    3. Review the 3 generated plots (5%, 10%, 15% thresholds)
"""

import sys
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import helpers
import logisticregression_predictor as helper

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================================
# CONFIGURATION - EDIT THESE FOR DIFFERENT STOCKS
# ============================================================================
STOCK_URL = 'https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=1940537480'
OMX30_URL = 'https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=1961884031'
STOCK_NAME = 'HEXA-B'

# Fixed parameters
TARGET_RETURN_PERCENT = 10.0
TARGET_HORIZON_DAYS = 42
TIMESTEPS = 5

# Import model list and create_sequences
from ensemble_voting_predictor import MODELS, create_sequences

# ============================================================================
# LOAD WEIGHTS
# ============================================================================
def load_ensemble_weights():
    """Load pre-calculated precision-squared weights"""
    weights_file = script_dir / 'ensemble_weights.csv'
    
    if not weights_file.exists():
        print("❌ ERROR: ensemble_weights.csv not found!")
        print("   Please run ensemble_voting_predictor.py first to generate weights.")
        sys.exit(1)
    
    df_weights = pd.read_csv(weights_file)
    weights = dict(zip(df_weights['model'], df_weights['weight']))
    
    print(f"✓ Loaded weights for {len(weights)} models")
    print(f"  Total weight: {sum(weights.values()):.6f}\n")
    
    return weights

# ============================================================================
# DATA LOADING
# ============================================================================
def load_and_prepare_data():
    """Load stock data, generate features and targets"""
    print("="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)
    print(f"Stock: {STOCK_NAME}")
    print(f"Target: >{TARGET_RETURN_PERCENT}% return in {TARGET_HORIZON_DAYS} days\n")
    
    # Load data
    print("Loading stock and index data...")
    df_stock = helper.load_stock_data(STOCK_URL, stock_id=STOCK_NAME)
    df_omx30 = helper.load_stock_data(OMX30_URL, stock_id='OMX30')
    print(f"  Stock: {len(df_stock)} rows")
    print(f"  OMX30: {len(df_omx30)} rows\n")
    
    # Generate features
    print("Generating features...")
    df_features = helper.create_features(df_stock, index_df=df_omx30)
    print(f"  Features: {len(df_features.columns)} columns\n")
    
    # Generate targets
    print("Generating targets (ANTI-LEAKAGE)...")
    df_with_target, df_future = helper.generate_targets(df_features, TARGET_RETURN_PERCENT, TARGET_HORIZON_DAYS)
    print(f"  Historical data: {len(df_with_target)} rows (with targets)")
    print(f"  Future data: {len(df_future)} rows (no targets, for prediction)\n")
    
    # Load selected features
    features_file = script_dir / 'selected_features.txt'
    with open(features_file, 'r') as f:
        selected_features = [line.strip() for line in f if line.strip()]
    
    available_features = [f for f in selected_features if f in df_with_target.columns]
    print(f"Using {len(available_features)} features\n")
    
    # Prepare clean dataframes
    df_clean = df_with_target[['Date', 'Close', 'target'] + available_features].copy().dropna()
    df_future_clean = df_future[['Date', 'Close'] + available_features].copy().dropna()
    
    print(f"Clean data: {len(df_clean)} historical + {len(df_future_clean)} future rows")
    
    return df_clean, df_future_clean, available_features

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
        print(f"  ✗ {model_name}: {str(e)[:60]}")
        return None

def train_all_models_and_predict(df_clean, df_future_clean, features, weights):
    """Train all 29 models on historical data and predict on future data"""
    print("\n" + "="*80)
    print("TRAINING ALL MODELS AND PREDICTING FUTURE")
    print("="*80)
    print(f"Training on: {len(df_clean)} historical days")
    print(f"Predicting:  {len(df_future_clean)} future days\n")
    
    # Prepare training data
    X_train = df_clean[features].values
    y_train = df_clean['target'].values
    X_future = df_future_clean[features].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_future_scaled = scaler.transform(X_future)
    
    n_features = X_train.shape[1]
    
    # Collect predictions from all models
    all_probabilities = {}
    success_count = 0
    
    print("Training models...")
    for model_name, _, model_type, script_name in MODELS:
        probs = train_and_predict_single_model(
            model_name, model_type, script_name,
            X_train_scaled, y_train, X_future_scaled, n_features
        )
        
        if probs is not None:
            all_probabilities[model_name] = probs
            success_count += 1
            print(f"  ✓ {model_name}")
    
    print(f"\n✓ Successfully trained {success_count}/{len(MODELS)} models")
    
    if success_count < 20:
        print(f"⚠️  WARNING: Only {success_count} models succeeded. Results may be less reliable.")
    
    return all_probabilities

# ============================================================================
# ENSEMBLE PREDICTION
# ============================================================================
def generate_ensemble_predictions(all_probabilities, weights, df_future_clean, thresholds=[5, 10, 15]):
    """Generate ensemble predictions at different thresholds"""
    print("\n" + "="*80)
    print("GENERATING ENSEMBLE PREDICTIONS")
    print("="*80)
    print(f"Thresholds: {thresholds}%\n")
    
    n_future = len(df_future_clean)
    
    # Calculate weighted scores
    weighted_scores = np.zeros(n_future)
    for model_name, probs in all_probabilities.items():
        weight = weights.get(model_name, 0)
        weighted_scores += probs * weight
    
    # Generate predictions for each threshold
    results = {}
    for top_percent in thresholds:
        threshold_idx = int(n_future * (1 - top_percent/100))
        threshold_idx = max(0, min(threshold_idx, n_future - 1))
        
        sorted_scores = np.sort(weighted_scores)
        threshold = sorted_scores[threshold_idx] if threshold_idx < n_future else 0
        
        predictions = (weighted_scores >= threshold).astype(int)
        n_signals = predictions.sum()
        
        results[top_percent] = {
            'predictions': predictions,
            'scores': weighted_scores,
            'threshold': threshold,
            'n_signals': n_signals
        }
        
        print(f"  Top {top_percent:2d}%: {n_signals:2d} signals (threshold={threshold:.4f})")
    
    return results

# ============================================================================
# PLOTTING
# ============================================================================
def plot_future_predictions(df_clean, df_future_clean, results, stock_name):
    """Create 3 plots for different thresholds"""
    print("\n" + "="*80)
    print("CREATING PREDICTION PLOTS")
    print("="*80 + "\n")
    
    for top_percent, data in sorted(results.items()):
        predictions = data['predictions']
        n_signals = data['n_signals']
        
        output_path = script_dir / f'predictions_orchestrator_{top_percent}pct.png'
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot historical price (black line)
        ax.plot(df_clean['Date'], df_clean['Close'], 
                color='black', linewidth=1.5, label='Historical Price', zorder=1)
        
        # Orange dotted separator line
        separator_date = df_future_clean['Date'].min()
        ax.axvline(x=separator_date, color='orange', linestyle='--', 
                   linewidth=2, label='Historical | Future', zorder=2)
        
        # Future price (gray dashed line)
        ax.plot(df_future_clean['Date'], df_future_clean['Close'], 
                color='gray', linewidth=1.5, linestyle='--', 
                label='Future Price', alpha=0.6, zorder=1)
        
        # Blue stars for future signals
        future_signals = df_future_clean[predictions == 1]
        if n_signals > 0:
            ax.scatter(future_signals['Date'], future_signals['Close'],
                      color='blue', marker='*', s=300, alpha=0.9,
                      label=f'Future Buy Signals (n={n_signals})', 
                      zorder=4, edgecolors='darkblue', linewidths=1.5)
        
        # Title and labels
        title = f'{stock_name} - Ensemble Top {top_percent}% Future Buy Signals\n{n_signals} signals in next {len(df_future_clean)} days'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Saved: {output_path.name}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*80)
    print("STOCK FEASIBILITY ORCHESTRATOR")
    print("="*80 + "\n")
    
    # Load weights
    weights = load_ensemble_weights()
    
    # Load and prepare data
    df_clean, df_future_clean, features = load_and_prepare_data()
    
    # Train models and predict
    all_probabilities = train_all_models_and_predict(df_clean, df_future_clean, features, weights)
    
    # Generate ensemble predictions
    results = generate_ensemble_predictions(all_probabilities, weights, df_future_clean, 
                                           thresholds=[5, 10, 15])
    
    # Create plots
    plot_future_predictions(df_clean, df_future_clean, results, STOCK_NAME)
    
    # Final summary
    print("\n" + "="*80)
    print("ORCHESTRATOR COMPLETE ✅")
    print("="*80)
    print(f"Stock: {STOCK_NAME}")
    print(f"Models trained: {len(all_probabilities)}/{len(MODELS)}")
    print(f"Future period: {df_future_clean['Date'].min()} to {df_future_clean['Date'].max()} ({len(df_future_clean)} days)")
    print()
    print("Signals Generated:")
    for top_percent in [5, 10, 15]:
        n_signals = results[top_percent]['n_signals']
        print(f"  Top {top_percent:2d}%: {n_signals:2d} signals")
    print()
    print("Plots saved:")
    print("  - predictions_orchestrator_5pct.png")
    print("  - predictions_orchestrator_10pct.png")
    print("  - predictions_orchestrator_15pct.png")
    print("="*80)
    print()
    print("💡 USAGE: Higher threshold (5%) = more selective, fewer but higher-confidence signals")
    print("💡        Lower threshold (15%) = more signals, lower confidence per signal")
    print()

if __name__ == '__main__':
    main()

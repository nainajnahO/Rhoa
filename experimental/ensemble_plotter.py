"""
Ensemble Prediction Plotter

Creates individual prediction plots for ensemble at different thresholds,
matching the style of individual model plots.
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

import logisticregression_predictor as helper

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configuration
STOCK_URL = helper.STOCK_URL
OMX30_URL = helper.OMX30_URL
STOCK_NAME = helper.STOCK_NAME
TARGET_RETURN_PERCENT = helper.TARGET_RETURN_PERCENT
TARGET_HORIZON_DAYS = helper.TARGET_HORIZON_DAYS
PREDICTION_THRESHOLD = 0.5
TIMESTEPS = 5

# Import model info
from ensemble_voting_predictor import MODELS, calculate_model_weights, create_sequences

def load_all_data():
    """Load complete dataset"""
    print("Loading data...")
    df_stock = helper.load_stock_data(STOCK_URL, stock_id=STOCK_NAME)
    df_omx30 = helper.load_stock_data(OMX30_URL, stock_id='OMX30')
    df_features = helper.create_features(df_stock, index_df=df_omx30)
    df_with_target, df_future = helper.generate_targets(df_features, TARGET_RETURN_PERCENT, TARGET_HORIZON_DAYS)
    
    # Load features
    features_file = script_dir / 'selected_features.txt'
    with open(features_file, 'r') as f:
        selected_features = [line.strip() for line in f if line.strip()]
    
    available_features = [f for f in selected_features if f in df_with_target.columns]
    df_clean = df_with_target[['Date', 'Close', 'target'] + available_features].copy().dropna()
    df_future_clean = df_future[['Date', 'Close'] + available_features].copy()
    
    return df_clean, df_future_clean, available_features

def train_and_predict_model(model_name, model_type, script_name, X_train, y_train, X_val):
    """Train a single model and get predictions"""
    try:
        module = __import__(script_name)
        
        if model_type == 'keras_seq':
            X_train_seq = create_sequences(X_train, TIMESTEPS)
            y_train_seq = y_train[TIMESTEPS-1:]
            X_val_seq = create_sequences(X_val, TIMESTEPS)
            
            model = module.create_model(X_train.shape[1])
            model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0)
            probs = model.predict(X_val_seq, verbose=0).flatten()
            
            full_probs = np.zeros(len(X_val))
            full_probs[TIMESTEPS-1:] = probs
            
        elif model_type == 'keras':
            model = module.create_model(X_train.shape[1])
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            full_probs = model.predict(X_val, verbose=0).flatten()
            
        else:  # sklearn models
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
            elif 'knn' in script_name:
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier(**module.KNN_PARAMS)
            elif 'decisiontree' in script_name:
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier(**module.DECISIONTREE_PARAMS)
            else:
                return None
            
            model.fit(X_train, y_train)
            
            if hasattr(model, 'predict_proba'):
                full_probs = model.predict_proba(X_val)[:, 1]
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function(X_val)
                full_probs = 1 / (1 + np.exp(-decision))
            else:
                full_probs = model.predict(X_val).astype(float)
        
        return full_probs
    except Exception as e:
        print(f"  ✗ {model_name}: {str(e)[:60]}")
        return None

def generate_ensemble_predictions(df_clean, df_future_clean, features, top_percent_list=[5, 10, 15]):
    """Generate ensemble predictions at different thresholds"""
    print("\n" + "="*80)
    print(f"GENERATING ENSEMBLE PREDICTIONS FOR THRESHOLDS: {top_percent_list}")
    print("="*80)
    
    # Calculate weights
    weights, _ = calculate_model_weights()
    
    # Use last 20% for validation
    val_start_idx = int(len(df_clean) * 0.8)
    train_start_idx = int(len(df_clean) * 0.5)
    
    df_train = df_clean.iloc[train_start_idx:val_start_idx].copy().reset_index(drop=True)
    df_val = df_clean.iloc[val_start_idx:].copy().reset_index(drop=True)
    
    print(f"\nTraining set: {len(df_train)} days")
    print(f"Validation set: {len(df_val)} days")
    
    # Train all models on training data
    X_train = df_train[features].values
    y_train = df_train['target'].values
    X_val = df_val[features].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Collect predictions
    print("\nCollecting model predictions...")
    all_probabilities = {}
    
    for model_name, _, model_type, script_name in MODELS:
        probs = train_and_predict_model(model_name, model_type, script_name, 
                                        X_train_scaled, y_train, X_val_scaled)
        if probs is not None:
            all_probabilities[model_name] = probs
            print(f"  ✓ {model_name}")
    
    print(f"\nCollected {len(all_probabilities)}/29 models")
    
    # Generate predictions at different thresholds
    results = {}
    for top_percent in top_percent_list:
        print(f"\nGenerating predictions for Top {top_percent}%...")
        
        # Calculate weighted scores
        weighted_scores = np.zeros(len(X_val))
        for model_name, probs in all_probabilities.items():
            weight = weights.get(model_name, 0)
            weighted_scores += probs * weight
        
        # Select top X%
        threshold_idx = int(len(X_val) * (1 - top_percent/100))
        sorted_scores = np.sort(weighted_scores)
        threshold = sorted_scores[threshold_idx] if threshold_idx < len(X_val) else 0
        
        predictions = (weighted_scores >= threshold).astype(int)
        
        results[top_percent] = {
            'df_val': df_val,
            'predictions': predictions,
            'scores': weighted_scores,
            'threshold': threshold
        }
        
        n_signals = predictions.sum()
        y_true = df_val['target'].values
        tp = ((predictions == 1) & (y_true == 1)).sum()
        fp = ((predictions == 1) & (y_true == 0)).sum()
        
        print(f"  Signals: {n_signals}, TP: {tp}, FP: {fp}")
    
    return results

def plot_ensemble_predictions(results, stock_name):
    """Create plots for each threshold"""
    print("\n" + "="*80)
    print("CREATING PREDICTION PLOTS")
    print("="*80)
    
    for top_percent, data in sorted(results.items()):
        df_val = data['df_val']
        predictions = data['predictions']
        y_true = df_val['target'].values
        
        output_path = script_dir / f'predictions_ensemble_{top_percent}pct.png'
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot price
        ax.plot(df_val['Date'], df_val['Close'], color='black', linewidth=1.5, 
                label='Price', zorder=1)
        
        # Get signal indices
        signal_indices = predictions == 1
        correct = (predictions == 1) & (y_true == 1)
        incorrect = (predictions == 1) & (y_true == 0)
        
        # Plot correct signals
        if correct.sum() > 0:
            ax.scatter(df_val.loc[correct, 'Date'], df_val.loc[correct, 'Close'],
                      color='green', marker='^', s=150, alpha=0.8,
                      label=f'Correct (n={correct.sum()})', zorder=3, 
                      edgecolors='darkgreen', linewidths=1.5)
        
        # Plot incorrect signals
        if incorrect.sum() > 0:
            ax.scatter(df_val.loc[incorrect, 'Date'], df_val.loc[incorrect, 'Close'],
                      color='red', marker='v', s=150, alpha=0.8,
                      label=f'Incorrect (n={incorrect.sum()})', zorder=3,
                      edgecolors='darkred', linewidths=1.5)
        
        # Calculate metrics
        tp = correct.sum()
        fp = incorrect.sum()
        fn = ((predictions == 0) & (y_true == 1)).sum()
        tn = ((predictions == 0) & (y_true == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        ax.set_title(f'{stock_name} - Ensemble Top {top_percent}% Buy Signals\n'
                    f'Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f} '
                    f'(TP={tp}, FP={fp}, FN={fn})',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✓ Saved: {output_path}")
        print(f"  Top {top_percent}%: Precision={precision:.4f}, Recall={recall:.4f}, "
              f"F1={f1:.4f}, Signals={signal_indices.sum()}")

def main():
    print("="*80)
    print("ENSEMBLE PREDICTION PLOTTER")
    print("="*80)
    
    # Load data
    df_clean, df_future_clean, features = load_all_data()
    
    # Generate predictions
    results = generate_ensemble_predictions(df_clean, df_future_clean, features, 
                                           top_percent_list=[5, 10, 15])
    
    # Create plots
    plot_ensemble_predictions(results, STOCK_NAME)
    
    print("\n" + "="*80)
    print("COMPLETE! ✅")
    print("="*80)
    print("\nGenerated files:")
    print("  - predictions_ensemble_5pct.png")
    print("  - predictions_ensemble_10pct.png")
    print("  - predictions_ensemble_15pct.png")

if __name__ == '__main__':
    main()

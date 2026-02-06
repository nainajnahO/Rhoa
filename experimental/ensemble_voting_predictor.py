"""
Weighted Ensemble Voting Predictor

Combines all 29 models using precision-squared weighted voting.
Predicts buy signals for top X% days by weighted vote score.

Goal: Maximize TP while minimizing FP
"""

import sys
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import for data loading
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

# Ensemble Configuration
TOP_PERCENT = 10  # Predict buy for top 10% of days by weighted score
TIMESTEPS = 5  # For sequence models

# All 29 models with their metric files and types
MODELS = [
    # Ensemble Methods
    ('XGBoost', 'walk_forward_metrics.csv', 'sklearn', 'xgboost_predictor'),
    ('LightGBM', 'walk_forward_metrics_lgbm.csv', 'sklearn', 'lightgbm_predictor'),
    ('CatBoost', 'walk_forward_metrics_catboost.csv', 'sklearn', 'catboost_predictor'),
    ('RandomForest', 'walk_forward_metrics_randomforest.csv', 'sklearn', 'randomforest_predictor'),
    ('AdaBoost', 'walk_forward_metrics_adaboost.csv', 'sklearn', 'adaboost_predictor'),
    ('GradientBoosting', 'walk_forward_metrics_gradientboosting.csv', 'sklearn', 'gradientboosting_predictor'),
    ('ExtraTrees', 'walk_forward_metrics_extratrees.csv', 'sklearn', 'extratrees_predictor'),
    ('Bagging', 'walk_forward_metrics_bagging.csv', 'sklearn', 'bagging_predictor'),
    # Neural Networks
    ('MLP', 'walk_forward_metrics_mlp.csv', 'sklearn', 'mlp_predictor'),
    ('DeepMLP', 'walk_forward_metrics_deepmlp.csv', 'keras', 'deepmlp_predictor'),
    ('LSTM', 'walk_forward_metrics_lstm.csv', 'keras_seq', 'lstm_predictor'),
    ('GRU', 'walk_forward_metrics_gru.csv', 'keras_seq', 'gru_predictor'),
    ('BiLSTM', 'walk_forward_metrics_bilstm.csv', 'keras_seq', 'bilstm_predictor'),
    ('SimpleRNN', 'walk_forward_metrics_simplernn.csv', 'keras_seq', 'simplernn_predictor'),
    ('CNN1D', 'walk_forward_metrics_cnn1d.csv', 'keras_seq', 'cnn1d_predictor'),
    ('CNN-LSTM', 'walk_forward_metrics_cnnlstm.csv', 'keras_seq', 'cnnlstm_predictor'),
    ('Attention-LSTM', 'walk_forward_metrics_attentionlstm.csv', 'keras_seq', 'attentionlstm_predictor'),
    # Traditional ML
    ('SVM', 'walk_forward_metrics_svm.csv', 'sklearn', 'svm_predictor'),
    ('LogisticRegression', 'walk_forward_metrics_logisticregression.csv', 'sklearn', 'logisticregression_predictor'),
    ('NaiveBayes', 'walk_forward_metrics_naivebayes.csv', 'sklearn', 'naivebayes_predictor'),
    ('RidgeClassifier', 'walk_forward_metrics_ridgeclassifier.csv', 'sklearn', 'ridgeclassifier_predictor'),
    ('SGDClassifier', 'walk_forward_metrics_sgdclassifier.csv', 'sklearn', 'sgdclassifier_predictor'),
    ('LDA', 'walk_forward_metrics_lda.csv', 'sklearn', 'lda_predictor'),
    ('QDA', 'walk_forward_metrics_qda.csv', 'sklearn', 'qda_predictor'),
    ('Perceptron', 'walk_forward_metrics_perceptron.csv', 'sklearn', 'perceptron_predictor'),
    ('PassiveAggressive', 'walk_forward_metrics_passiveaggressive.csv', 'sklearn', 'passiveaggressive_predictor'),
    # Specialized
    ('HistGradientBoosting', 'walk_forward_metrics_histgradientboosting.csv', 'sklearn', 'histgradientboosting_predictor'),
    # Instance-Based
    ('KNN', 'walk_forward_metrics_knn.csv', 'sklearn', 'knn_predictor'),
    # Baseline
    ('DecisionTree', 'walk_forward_metrics_decisiontree.csv', 'sklearn', 'decisiontree_predictor'),
]

def calculate_model_weights():
    """Calculate precision^2 weights for each model from their metrics"""
    print("\n" + "="*80)
    print("CALCULATING MODEL WEIGHTS (Precision^2)")
    print("="*80 + "\n")
    
    weights = {}
    precisions = {}
    
    for model_name, csv_file, _, _ in MODELS:
        try:
            df = pd.read_csv(script_dir / csv_file)
            avg_precision = df['precision'].mean()
            precisions[model_name] = avg_precision
            weights[model_name] = avg_precision ** 2
            print(f"{model_name:25s} Precision={avg_precision:.4f}  Weight={avg_precision**2:.6f}")
        except FileNotFoundError:
            print(f"{model_name:25s} MISSING METRICS FILE")
            weights[model_name] = 0
            precisions[model_name] = 0
    
    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    
    print(f"\nTotal normalized weight: {sum(weights.values()):.6f}")
    
    # Sort by weight
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Most Influential Models:")
    for model_name, weight in sorted_weights[:10]:
        print(f"  {model_name:25s} {weight:.4f} ({weight*100:.1f}%)")
    
    return weights, precisions

def load_model_modules():
    """Load model parameter modules"""
    print("\n" + "="*80)
    print("LOADING MODEL PARAMETERS")
    print("="*80 + "\n")
    
    loaded_modules = {}
    
    for model_name, _, model_type, script_name in MODELS:
        try:
            # Import the module dynamically
            module = __import__(script_name)
            loaded_modules[model_name] = {
                'module': module,
                'type': model_type
            }
            print(f"✓ {model_name:25s} parameters loaded")
        except Exception as e:
            print(f"✗ {model_name:25s} failed: {str(e)[:50]}")
    
    print(f"\nSuccessfully loaded {len(loaded_modules)}/{len(MODELS)} modules")
    return loaded_modules

def create_sequences(X, timesteps):
    """Create sliding window sequences for RNN/CNN models"""
    n_samples = len(X) - timesteps + 1
    X_seq = np.zeros((n_samples, timesteps, X.shape[1]))
    for i in range(n_samples):
        X_seq[i] = X[i:i+timesteps]
    return X_seq

def collect_model_predictions(validation_data, features, loaded_modules):
    """Run all models on validation data and collect predictions"""
    print("\n" + "="*80)
    print("COLLECTING MODEL PREDICTIONS ON VALIDATION SET")
    print("="*80 + "\n")
    
    X_val = validation_data[features].values
    y_val = validation_data['target'].values
    dates_val = validation_data['Date'].values
    
    all_predictions = {}
    all_probabilities = {}
    
    for model_name, _, model_type, script_name in MODELS:
        if model_name not in loaded_modules:
            print(f"⊘ {model_name:25s} not loaded, skipping")
            continue
        
        try:
            module = loaded_modules[model_name]['module']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_val)
            
            # Handle sequence models
            if model_type == 'keras_seq':
                X_seq = create_sequences(X_scaled, TIMESTEPS)
                y_seq = y_val[TIMESTEPS-1:]
                dates_seq = dates_val[TIMESTEPS-1:]
                
                # Train model
                model = module.create_model(X_scaled.shape[1])
                model.fit(X_seq, y_seq, epochs=50, batch_size=32, verbose=0)
                
                # Predict
                probs = model.predict(X_seq, verbose=0).flatten()
                preds = (probs >= PREDICTION_THRESHOLD).astype(int)
                
                # Pad to match original length
                full_probs = np.zeros(len(X_val))
                full_preds = np.zeros(len(X_val))
                full_probs[TIMESTEPS-1:] = probs
                full_preds[TIMESTEPS-1:] = preds
                
            elif model_type == 'keras':
                # Keras non-sequence models
                model = module.create_model(X_scaled.shape[1])
                model.fit(X_scaled, y_val, epochs=50, batch_size=32, verbose=0)
                full_probs = model.predict(X_scaled, verbose=0).flatten()
                full_preds = (full_probs >= PREDICTION_THRESHOLD).astype(int)
                
            else:  # sklearn models
                # Import and train sklearn model based on script
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
                    print(f"⊘ {model_name:25s} unknown sklearn model")
                    continue
                
                model.fit(X_scaled, y_val)
                
                # Get probabilities
                if hasattr(model, 'predict_proba'):
                    full_probs = model.predict_proba(X_scaled)[:, 1]
                elif hasattr(model, 'decision_function'):
                    decision = model.decision_function(X_scaled)
                    full_probs = 1 / (1 + np.exp(-decision))
                else:
                    full_probs = model.predict(X_scaled).astype(float)
                
                full_preds = (full_probs >= PREDICTION_THRESHOLD).astype(int)
            
            all_predictions[model_name] = full_preds
            all_probabilities[model_name] = full_probs
            n_signals = full_preds.sum()
            print(f"✓ {model_name:25s} {n_signals:4d} signals")
            
        except Exception as e:
            print(f"✗ {model_name:25s} failed: {str(e)[:60]}")
            all_predictions[model_name] = np.zeros(len(X_val))
            all_probabilities[model_name] = np.zeros(len(X_val))
    
    print(f"\nCollected predictions from {len(all_predictions)} models")
    return all_predictions, all_probabilities, y_val, dates_val

def ensemble_predict(all_probabilities, weights, top_percent=10):
    """Calculate weighted vote scores and predict top X% days"""
    n_days = len(next(iter(all_probabilities.values())))
    
    # Calculate weighted vote score for each day
    weighted_scores = np.zeros(n_days)
    for model_name, probs in all_probabilities.items():
        weight = weights.get(model_name, 0)
        weighted_scores += probs * weight
    
    # Rank days by score, predict top X%
    threshold_idx = int(n_days * (1 - top_percent/100))
    if threshold_idx < n_days:
        sorted_scores = np.sort(weighted_scores)
        threshold = sorted_scores[threshold_idx]
    else:
        threshold = weighted_scores.min() - 1
    
    predictions = (weighted_scores >= threshold).astype(int)
    return predictions, weighted_scores, threshold

def evaluate_ensemble(predictions, y_true):
    """Evaluate ensemble performance"""
    cm = confusion_matrix(y_true, predictions)
    tn, fp, fn, tp = cm.ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, average='binary', zero_division=0)
    
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_signals': predictions.sum()
    }

def threshold_analysis(all_probabilities, weights, y_true):
    """Test multiple top_percent thresholds"""
    print("\n" + "="*80)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*80 + "\n")
    
    thresholds = [5, 10, 15, 20, 25, 30]
    results = []
    
    print(f"{'Top %':>8} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Signals':>10} {'TP':>6} {'FP':>6}")
    print("-" * 80)
    
    for top_pct in thresholds:
        preds, scores, thresh = ensemble_predict(all_probabilities, weights, top_pct)
        metrics = evaluate_ensemble(preds, y_true)
        
        results.append({
            'top_percent': top_pct,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'signals': metrics['n_signals'],
            'tp': metrics['tp'],
            'fp': metrics['fp']
        })
        
        print(f"{top_pct:>8}% {metrics['precision']:>12.4f} {metrics['recall']:>12.4f} "
              f"{metrics['f1']:>12.4f} {metrics['n_signals']:>10} {metrics['tp']:>6} {metrics['fp']:>6}")
    
    return pd.DataFrame(results)

def main():
    print("="*80)
    print("WEIGHTED ENSEMBLE VOTING PREDICTOR")
    print("="*80)
    
    # Calculate model weights
    weights, precisions = calculate_model_weights()
    
    # Save weights
    weights_df = pd.DataFrame([
        {'model': k, 'precision': precisions[k], 'precision_squared': precisions[k]**2, 'weight': v}
        for k, v in sorted(weights.items(), key=lambda x: x[1], reverse=True)
    ])
    weights_path = script_dir / 'ensemble_weights.csv'
    weights_df.to_csv(weights_path, index=False)
    print(f"\n✓ Weights saved to: {weights_path}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING VALIDATION DATA")
    print("="*80 + "\n")
    
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
    
    # Use last 20% as validation set
    val_start_idx = int(len(df_clean) * 0.8)
    df_validation = df_clean.iloc[val_start_idx:].copy().reset_index(drop=True)
    
    print(f"Validation set: {len(df_validation)} days")
    print(f"Date range: {df_validation['Date'].min().date()} to {df_validation['Date'].max().date()}")
    print(f"Positive class: {df_validation['target'].sum()} days ({df_validation['target'].mean()*100:.1f}%)")
    
    # Load model modules
    loaded_modules = load_model_modules()
    
    # Collect predictions
    all_predictions, all_probabilities, y_true, dates = collect_model_predictions(
        df_validation, available_features, loaded_modules
    )
    
    # Ensemble prediction
    print("\n" + "="*80)
    print(f"ENSEMBLE PREDICTION (Top {TOP_PERCENT}%)")
    print("="*80 + "\n")
    
    ensemble_preds, ensemble_scores, threshold = ensemble_predict(
        all_probabilities, weights, TOP_PERCENT
    )
    
    ensemble_metrics = evaluate_ensemble(ensemble_preds, y_true)
    
    print(f"Weighted vote threshold: {threshold:.4f}")
    print(f"Signals: {ensemble_metrics['n_signals']}")
    print(f"Precision: {ensemble_metrics['precision']:.4f}")
    print(f"Recall: {ensemble_metrics['recall']:.4f}")
    print(f"F1: {ensemble_metrics['f1']:.4f}")
    print(f"TP: {ensemble_metrics['tp']}, FP: {ensemble_metrics['fp']}, FN: {ensemble_metrics['fn']}, TN: {ensemble_metrics['tn']}")
    
    # Threshold analysis
    threshold_df = threshold_analysis(all_probabilities, weights, y_true)
    threshold_path = script_dir / 'ensemble_threshold_analysis.csv'
    threshold_df.to_csv(threshold_path, index=False)
    print(f"\n✓ Threshold analysis saved to: {threshold_path}")
    
    # Signal breakdown
    signal_indices = np.where(ensemble_preds == 1)[0]
    signal_data = []
    
    for idx in signal_indices:
        voting_models = []
        for model_name, probs in all_probabilities.items():
            if probs[idx] >= PREDICTION_THRESHOLD:
                voting_models.append({
                    'model': model_name,
                    'prob': probs[idx],
                    'weight': weights.get(model_name, 0)
                })
        
        voting_models = sorted(voting_models, key=lambda x: x['weight'], reverse=True)
        
        signal_data.append({
            'date': dates[idx],
            'weighted_score': ensemble_scores[idx],
            'actual': y_true[idx],
            'n_voters': len(voting_models),
            'top_voter': voting_models[0]['model'] if voting_models else 'None'
        })
    
    signals_df = pd.DataFrame(signal_data)
    signals_path = script_dir / 'ensemble_signals.csv'
    signals_df.to_csv(signals_path, index=False)
    print(f"✓ Signal details saved to: {signals_path}")
    
    print("\n" + "="*80)
    print("ENSEMBLE COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()

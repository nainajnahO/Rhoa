# Experimental Stock Predictors - FIXED VERSIONS
#
# This module contains corrected implementations of ML models
# for stock prediction with proper time series validation to prevent data leakage.
#
# Available Models (29 Total):
#
# Ensemble Methods (8):
# - XGBoost (xgboost_predictor.py) - Stable, widely adopted gradient boosting
# - LightGBM (lightgbm_predictor.py) - Faster, more efficient gradient boosting
# - CatBoost (catboost_predictor.py) - Superior categorical feature handling
# - RandomForest (randomforest_predictor.py) - Robust ensemble, less prone to overfitting
# - AdaBoost (adaboost_predictor.py) - Adaptive boosting, focuses on difficult examples
# - GradientBoosting (gradientboosting_predictor.py) - Classic sklearn gradient boosting
# - ExtraTrees (extratrees_predictor.py) - Extra randomization, reduces overfitting
# - Bagging (bagging_predictor.py) - Bootstrap aggregating with decision trees
#
# Neural Networks (9):
# - MLP (mlp_predictor.py) - Multi-layer perceptron (sklearn)
# - Deep MLP (deepmlp_predictor.py) - Deep feedforward network (Keras/TF)
# - LSTM (lstm_predictor.py) - Long short-term memory (Keras/TF)
# - GRU (gru_predictor.py) - Gated recurrent unit (Keras/TF)
# - BiLSTM (bilstm_predictor.py) - Bidirectional LSTM (Keras/TF)
# - SimpleRNN (simplernn_predictor.py) - Basic recurrent network (Keras/TF)
# - CNN1D (cnn1d_predictor.py) - 1D convolutional network (Keras/TF)
# - CNN-LSTM (cnnlstm_predictor.py) - Hybrid CNN+LSTM (Keras/TF)
# - Attention-LSTM (attentionlstm_predictor.py) - LSTM with attention (Keras/TF)
#
# Traditional ML (9):
# - SVM (svm_predictor.py) - Support vector machine, effective in high-dimensional spaces
# - Logistic Regression (logisticregression_predictor.py) - Fast linear baseline
# - Naive Bayes (naivebayes_predictor.py) - Probabilistic, independent features
# - Ridge Classifier (ridgeclassifier_predictor.py) - L2 regularized linear model
# - SGD Classifier (sgdclassifier_predictor.py) - Stochastic gradient descent
# - LDA (lda_predictor.py) - Linear discriminant analysis
# - QDA (qda_predictor.py) - Quadratic discriminant analysis
# - Perceptron (perceptron_predictor.py) - Single-layer neural network
# - Passive Aggressive (passiveaggressive_predictor.py) - Online learning
#
# Specialized (1):
# - HistGradientBoosting (histgradientboosting_predictor.py) - Fast histogram-based boosting
#
# Instance-Based (1):
# - KNN (knn_predictor.py) - K-nearest neighbors, local pattern recognition
#
# Baseline (1):
# - Decision Tree (decisiontree_predictor.py) - Interpretable single tree baseline
#
# Key Features:
# - No data leakage in target generation (last 42 days removed before splitting)
# - Walk-forward validation with 5 expanding windows
# - Future predictions on latest 42 days (orange dotted line separator)
# - Custom visualization with triangles (validated) and blue stars (future)
# - 50 optimized technical indicators
# - Feature scaling for neural networks and distance-based models
# - Sequence generation for recurrent and convolutional models (timesteps=5)

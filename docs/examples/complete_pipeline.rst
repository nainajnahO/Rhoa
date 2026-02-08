Complete ML Pipeline
====================

End-to-end examples showing how to build complete machine learning pipelines with Rhoa.

.. _pipeline-ml:

Basic Pipeline
--------------

A complete workflow from data to predictions.

Step-by-Step Pipeline
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import rhoa
   from rhoa.targets import generate_target_combinations
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import classification_report, confusion_matrix

   # Step 1: Load data
   df = pd.read_csv('stock_prices.csv')
   print(f"Data shape: {df.shape}")

   # Step 2: Generate technical indicator features
   close = df['Close']
   high = df['High']
   low = df['Low']

   df['SMA_20'] = close.indicators.sma(20)
   df['SMA_50'] = close.indicators.sma(50)
   df['RSI_14'] = close.indicators.rsi(14)
   df['ATR_14'] = close.indicators.atr(high, low, 14)

   macd = close.indicators.macd()
   df['MACD'] = macd['macd']
   df['MACD_Signal'] = macd['signal']

   # Step 3: Generate optimized targets
   targets, meta = generate_target_combinations(
       df,
       mode='auto',
       target_class_balance=0.4
   )

   print(f"\nTarget parameters for method 7:")
   print(f"  Period: {meta['method_7']['period']}")
   print(f"  Threshold: {meta['method_7']['threshold']}%")

   # Step 4: Prepare ML dataset
   ml_df = pd.concat([df, targets], axis=1).dropna()

   feature_cols = ['SMA_20', 'SMA_50', 'RSI_14', 'ATR_14', 'MACD', 'MACD_Signal']
   X = ml_df[feature_cols]
   y = ml_df['Target_7']

   print(f"\nML dataset shape: {X.shape}")
   print(f"Positive class: {y.sum()} ({y.mean()*100:.1f}%)")

   # Step 5: Train-test split (time-based)
   split_idx = int(len(X) * 0.8)
   X_train, X_test = X[:split_idx], X[split_idx:]
   y_train, y_test = y[:split_idx], y[split_idx:]

   # Step 6: Train model
   model = RandomForestClassifier(
       n_estimators=100,
       max_depth=10,
       random_state=42
   )
   model.fit(X_train, y_train)

   # Step 7: Evaluate
   y_pred = model.predict(X_test)
   print("\nClassification Report:")
   print(classification_report(y_test, y_pred))

   # Step 8: Feature importance
   importance = pd.DataFrame({
       'feature': feature_cols,
       'importance': model.feature_importances_
   }).sort_values('importance', ascending=False)
   print("\nFeature Importance:")
   print(importance)

Advanced Pipeline
-----------------

More sophisticated pipeline with additional features and validation.

Feature Engineering Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import numpy as np
   import rhoa
   from rhoa.targets import generate_target_combinations

   def engineer_features(df):
       """Create comprehensive feature set."""
       close = df['Close']
       high = df['High']
       low = df['Low']
       volume = df['Volume']

       # Price-based features
       df['Returns'] = close.pct_change()
       df['Log_Returns'] = np.log(close / close.shift(1))

       # Moving averages
       df['SMA_10'] = close.indicators.sma(10)
       df['SMA_20'] = close.indicators.sma(20)
       df['SMA_50'] = close.indicators.sma(50)
       df['SMA_200'] = close.indicators.sma(200)

       # MA relationships
       df['SMA_10_20_ratio'] = df['SMA_10'] / df['SMA_20']
       df['SMA_50_200_ratio'] = df['SMA_50'] / df['SMA_200']
       df['Price_SMA20_ratio'] = close / df['SMA_20']

       # Momentum indicators
       df['RSI_14'] = close.indicators.rsi(14)
       df['RSI_28'] = close.indicators.rsi(28)

       macd = close.indicators.macd()
       df['MACD'] = macd['macd']
       df['MACD_Signal'] = macd['signal']
       df['MACD_Hist'] = macd['histogram']

       # Volatility
       df['ATR_14'] = close.indicators.atr(high, low, 14)
       df['ATR_28'] = close.indicators.atr(high, low, 28)

       bb = close.indicators.bollinger_bands(20, 2.0)
       df['BB_Upper'] = bb['upper_band']
       df['BB_Lower'] = bb['lower_band']
       df['BB_Width'] = (bb['upper_band'] - bb['lower_band']) / bb['middle_band']
       df['BB_Position'] = (close - bb['lower_band']) / (bb['upper_band'] - bb['lower_band'])

       # Trend strength
       adx = close.indicators.adx(high, low, 14)
       df['ADX'] = adx['ADX']
       df['Plus_DI'] = adx['+DI']
       df['Minus_DI'] = adx['-DI']

       # Volume features
       df['Volume_SMA'] = volume.rolling(20).mean()
       df['Volume_Ratio'] = volume / df['Volume_SMA']

       # Lagged features
       for lag in [1, 2, 3, 5]:
           df[f'Returns_Lag{lag}'] = df['Returns'].shift(lag)
           df[f'RSI_Lag{lag}'] = df['RSI_14'].shift(lag)

       return df

   # Use it
   df = pd.read_csv('prices.csv')
   df = engineer_features(df)

   print(f"Total features created: {len(df.columns)}")

Cross-Validation Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import TimeSeriesSplit
   from sklearn.metrics import precision_score, recall_score, f1_score

   def time_series_cv(X, y, model, n_splits=5):
       """Perform time-series cross-validation."""
       tscv = TimeSeriesSplit(n_splits=n_splits)

       scores = {
           'precision': [],
           'recall': [],
           'f1': [],
           'accuracy': []
       }

       for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
           # Split data
           X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
           y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

           # Train
           model.fit(X_train, y_train)

           # Predict
           y_pred = model.predict(X_val)

           # Score
           scores['precision'].append(precision_score(y_val, y_pred))
           scores['recall'].append(recall_score(y_val, y_pred))
           scores['f1'].append(f1_score(y_val, y_pred))
           scores['accuracy'].append((y_pred == y_val).mean())

           print(f"Fold {fold}:")
           print(f"  Precision: {scores['precision'][-1]:.3f}")
           print(f"  Recall: {scores['recall'][-1]:.3f}")
           print(f"  F1: {scores['f1'][-1]:.3f}")

       # Average scores
       print("\nAverage Scores:")
       for metric, values in scores.items():
           print(f"  {metric}: {np.mean(values):.3f} (+/- {np.std(values):.3f})")

       return scores

   # Use it
   from sklearn.ensemble import RandomForestClassifier

   model = RandomForestClassifier(n_estimators=100, random_state=42)
   scores = time_series_cv(X, y, model, n_splits=5)

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import RandomizedSearchCV
   from sklearn.ensemble import RandomForestClassifier
   from scipy.stats import randint

   # Define parameter space
   param_dist = {
       'n_estimators': randint(50, 200),
       'max_depth': randint(5, 20),
       'min_samples_split': randint(2, 20),
       'min_samples_leaf': randint(1, 10),
       'max_features': ['sqrt', 'log2']
   }

   # Time series split for CV
   tscv = TimeSeriesSplit(n_splits=3)

   # Random search
   rf = RandomForestClassifier(random_state=42)
   search = RandomizedSearchCV(
       rf,
       param_distributions=param_dist,
       n_iter=20,
       cv=tscv,
       scoring='f1',
       n_jobs=-1,
       random_state=42
   )

   # Fit
   search.fit(X_train, y_train)

   print("Best parameters:")
   print(search.best_params_)
   print(f"\nBest F1 score: {search.best_score_:.3f}")

   # Use best model
   best_model = search.best_estimator_
   y_pred = best_model.predict(X_test)

Multiple Model Comparison
--------------------------

Compare different algorithms.

Model Comparison Function
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
   from sklearn.linear_model import LogisticRegression
   from sklearn.svm import SVC
   from sklearn.metrics import classification_report, roc_auc_score

   def compare_models(X_train, X_test, y_train, y_test):
       """Compare multiple models."""
       models = {
           'Logistic Regression': LogisticRegression(max_iter=1000),
           'Random Forest': RandomForestClassifier(n_estimators=100),
           'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
           'SVM': SVC(probability=True, kernel='rbf')
       }

       results = []

       for name, model in models.items():
           print(f"\n{'='*50}")
           print(f"Training {name}")
           print('='*50)

           # Train
           model.fit(X_train, y_train)

           # Predict
           y_pred = model.predict(X_test)
           y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

           # Metrics
           accuracy = (y_pred == y_test).mean()
           precision = precision_score(y_test, y_pred)
           recall = recall_score(y_test, y_pred)
           f1 = f1_score(y_test, y_pred)
           auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

           results.append({
               'Model': name,
               'Accuracy': accuracy,
               'Precision': precision,
               'Recall': recall,
               'F1': f1,
               'AUC': auc
           })

           print(f"Accuracy: {accuracy:.3f}")
           print(f"Precision: {precision:.3f}")
           print(f"Recall: {recall:.3f}")
           print(f"F1: {f1:.3f}")
           if auc:
               print(f"AUC: {auc:.3f}")

       # Summary
       results_df = pd.DataFrame(results)
       print("\n" + "="*50)
       print("SUMMARY")
       print("="*50)
       print(results_df.to_string(index=False))

       return results_df

   # Use it
   results = compare_models(X_train, X_test, y_train, y_test)

Production Pipeline
-------------------

A production-ready pipeline with proper structure.

Pipeline Class
~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import numpy as np
   import pickle
   from sklearn.ensemble import RandomForestClassifier
   from rhoa.targets import generate_target_combinations

   class StockPredictionPipeline:
       """Production ML pipeline for stock prediction."""

       def __init__(self, model=None, target_meta=None):
           self.model = model or RandomForestClassifier(n_estimators=100)
           self.target_meta = target_meta
           self.feature_cols = None

       def create_features(self, df):
           """Create feature set."""
           close = df['Close']
           high = df['High']
           low = df['Low']

           features = df.copy()

           # Technical indicators
           features['SMA_20'] = close.indicators.sma(20)
           features['SMA_50'] = close.indicators.sma(50)
           features['RSI_14'] = close.indicators.rsi(14)
           features['ATR_14'] = close.indicators.atr(high, low, 14)

           macd = close.indicators.macd()
           features['MACD'] = macd['macd']
           features['MACD_Signal'] = macd['signal']

           bb = close.indicators.bollinger_bands(20, 2.0)
           features['BB_Width'] = (bb['upper_band'] - bb['lower_band']) / bb['middle_band']

           # Returns
           features['Returns'] = close.pct_change()

           return features

       def fit(self, df, target_balance=0.4):
           """Train the pipeline."""
           print("Creating features...")
           df_features = self.create_features(df)

           print("Generating targets...")
           targets, self.target_meta = generate_target_combinations(
               df,
               mode='auto',
               target_class_balance=target_balance
           )

           # Prepare data
           ml_df = pd.concat([df_features, targets], axis=1).dropna()

           self.feature_cols = ['SMA_20', 'SMA_50', 'RSI_14', 'ATR_14',
                                'MACD', 'MACD_Signal', 'BB_Width', 'Returns']

           X = ml_df[self.feature_cols]
           y = ml_df['Target_7']

           print(f"Training on {len(X)} samples...")
           self.model.fit(X, y)

           print("Training complete!")
           return self

       def predict(self, df):
           """Make predictions on new data."""
           if self.feature_cols is None:
               raise ValueError("Pipeline not trained. Call fit() first.")

           # Create features
           df_features = self.create_features(df)

           # Extract feature columns
           X = df_features[self.feature_cols].dropna()

           # Predict
           predictions = self.model.predict(X)
           probabilities = self.model.predict_proba(X)

           return pd.DataFrame({
               'prediction': predictions,
               'probability': probabilities[:, 1]
           }, index=X.index)

       def save(self, path):
           """Save pipeline to disk."""
           with open(path, 'wb') as f:
               pickle.dump(self, f)

       @classmethod
       def load(cls, path):
           """Load pipeline from disk."""
           with open(path, 'rb') as f:
               return pickle.load(f)

   # Usage
   # Training
   df_train = pd.read_csv('train_data.csv')
   pipeline = StockPredictionPipeline()
   pipeline.fit(df_train)
   pipeline.save('model.pkl')

   # Prediction
   df_new = pd.read_csv('new_data.csv')
   pipeline = StockPredictionPipeline.load('model.pkl')
   predictions = pipeline.predict(df_new)
   print(predictions)

Backtesting Framework
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def backtest_strategy(df, predictions, initial_capital=100000):
       """Simple backtesting framework."""
       capital = initial_capital
       position = 0
       trades = []
       portfolio_value = []

       for idx in predictions.index:
           current_price = df.loc[idx, 'Close']
           pred = predictions.loc[idx, 'prediction']
           prob = predictions.loc[idx, 'probability']

           # Entry: buy signal with high confidence
           if pred == 1 and prob > 0.7 and position == 0:
               # Buy
               position = capital / current_price
               entry_price = current_price
               capital = 0
               trades.append({
                   'date': idx,
                   'type': 'BUY',
                   'price': current_price,
                   'position': position
               })

           # Exit: sell signal or stop loss
           elif position > 0:
               # Simple exit: hold for N days or price drops 5%
               if current_price < entry_price * 0.95 or len(trades) >= 5:
                   # Sell
                   capital = position * current_price
                   trades.append({
                       'date': idx,
                       'type': 'SELL',
                       'price': current_price,
                       'pnl': capital - initial_capital
                   })
                   position = 0

           # Track portfolio value
           total_value = capital + (position * current_price if position > 0 else 0)
           portfolio_value.append(total_value)

       # Results
       final_value = capital + (position * df.loc[predictions.index[-1], 'Close'] if position > 0 else 0)
       returns = (final_value - initial_capital) / initial_capital * 100

       print(f"Initial Capital: ${initial_capital:,.2f}")
       print(f"Final Value: ${final_value:,.2f}")
       print(f"Total Return: {returns:.2f}%")
       print(f"Number of Trades: {len([t for t in trades if t['type'] == 'BUY'])}")

       return trades, portfolio_value

   # Use it
   trades, portfolio = backtest_strategy(df, predictions)

Complete Example
----------------

Putting it all together.

.. code-block:: python

   import pandas as pd
   import rhoa
   from rhoa.targets import generate_target_combinations
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import classification_report

   # 1. Load and prepare data
   df = pd.read_csv('stock_prices.csv')
   print(f"Loaded {len(df)} records")

   # 2. Engineer features
   close = df['Close']
   high = df['High']
   low = df['Low']

   df['SMA_20'] = close.indicators.sma(20)
   df['SMA_50'] = close.indicators.sma(50)
   df['RSI_14'] = close.indicators.rsi(14)
   df['ATR_14'] = close.indicators.atr(high, low, 14)

   macd = close.indicators.macd()
   df['MACD'] = macd['macd']
   df['MACD_Signal'] = macd['signal']

   # 3. Generate targets
   targets, meta = generate_target_combinations(df, mode='auto', target_class_balance=0.4)

   # 4. Prepare ML data
   ml_df = pd.concat([df, targets], axis=1).dropna()
   feature_cols = ['SMA_20', 'SMA_50', 'RSI_14', 'ATR_14', 'MACD', 'MACD_Signal']
   X = ml_df[feature_cols]
   y = ml_df['Target_7']

   # 5. Time-based split
   split_idx = int(len(X) * 0.8)
   X_train, X_test = X[:split_idx], X[split_idx:]
   y_train, y_test = y[:split_idx], y[split_idx:]

   # 6. Train model
   model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
   model.fit(X_train, y_train)

   # 7. Evaluate
   y_pred = model.predict(X_test)
   print(classification_report(y_test, y_pred))

   # 8. Visualize predictions
   fig = ml_df.iloc[split_idx:].plots.signal(
       y_pred=y_pred,
       y_true=y_test,
       date_col='Date',
       price_col='Close',
       title='Model Predictions'
   )

   print("Pipeline complete!")

Next Steps
----------

You now have complete working examples of ML pipelines with Rhoa. For more details:

- :doc:`/api/index` - API reference
- :doc:`/user_guide/index` - Conceptual guides
- :doc:`/faq` - Troubleshooting

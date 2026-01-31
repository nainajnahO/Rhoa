# Improvements Guide: Increasing True Positives (90-100% Precision)

## 🎯 Objective

Improve the model to capture **more true positives** while maintaining **90-100% precision**.

Current baseline: **3 TP, 0 FP (100% precision, 2.3% recall)** at threshold 0.67

## ✅ What Has Been Implemented

All the infrastructure is now in place to train improved models with multi-stock data:

### 1. Data Aggregation Pipeline
**File**: `aggregate_data.py`

- Combines multiple stock CSV files into a single training dataset
- Standardizes formats (handles various CSV column naming conventions)
- Adds stock identifier for tracking
- Validates data quality
- **Usage**:
  ```bash
  python aggregate_data.py --input_dir data/stocks --output_file data/combined_stocks.csv
  ```

### 2. Multi-Stock Training Support
**File**: `preprocessing.py` (updated)

- New function: `time_series_split_multi_stock()`
- Splits each stock chronologically (prevents data leakage)
- Optional: Hold out entire stocks for testing (cross-stock generalization)
- Maintains class balance across splits

### 3. Threshold Optimization
**File**: `optimize_threshold.py`

- Tests thresholds from 0.50 to 0.90 (configurable)
- Finds optimal thresholds for 90%, 95%, 100% precision
- Creates precision-recall curves
- Shows tradeoffs between precision and recall
- **Usage**:
  ```bash
  python optimize_threshold.py --model_path saved_models/my_model --data_file data/combined_stocks.csv
  ```

### 4. Multi-Stock Training Script
**File**: `train_multi_stock.py`

- Complete training pipeline for multi-stock data
- Uses same architecture and features as baseline
- Supports holdout stocks for testing
- Generates comprehensive evaluation reports
- **Usage**:
  ```bash
  python train_multi_stock.py --data_file data/combined_stocks.csv --model_name multi_stock_v1
  ```

### 5. Ensemble with Consensus Voting
**File**: `train_ensemble_consensus.py`

- Trains multiple models with different random seeds
- Uses consensus voting (not probability averaging)
- Requires N out of M models to agree before predicting positive
- More conservative, better for high precision
- **Usage**:
  ```bash
  python train_ensemble_consensus.py --data_file data/combined_stocks.csv --n_models 5
  ```

### 6. Model Comparison Tool
**File**: `compare_models.py`

- Evaluates multiple models on same test data
- Compares at multiple thresholds
- Creates side-by-side visualizations
- Generates detailed comparison reports
- **Usage**:
  ```bash
  python compare_models.py \
      --models saved_models/high_precision_model saved_models/multi_stock_v1 \
      --labels "Baseline" "Multi-Stock" \
      --data_file data/combined_stocks.csv
  ```

## 📊 Expected Improvements

With more diverse training data (5-10+ stocks), we expect:

| Approach | Precision | Recall | Signals (per 130+ cases) | Improvement |
|----------|-----------|--------|---------------------------|-------------|
| **Current Baseline** | 100% | 2.3% | 3 | - |
| Multi-Stock (θ=0.65) | 95-100% | 5-8% | 6-10 | **2-3x more TP** |
| Multi-Stock (θ=0.63) | 90-95% | 8-15% | 10-20 | **3-7x more TP** |
| Ensemble (60% consensus) | 95-100% | 10-20% | 13-26 | **4-9x more TP** |

## 🚀 Step-by-Step Workflow

### Step 1: Prepare Stock Data

You need to provide **5-10 additional stock CSV files** (more is better).

**Required format** (same as your current test data):
```csv
Date,Open,High,Low,Close,Volume
1/2/2015 17:30:00,71.43,71.68,71.05,71.33,611903
1/5/2015 17:30:00,71.2,71.28,70.6,70.75,410586
...
```

**File structure**:
```
experimental/ml_prediction/
└── data/
    └── stocks/
        ├── AAPL.csv
        ├── MSFT.csv
        ├── GOOGL.csv
        ├── TSLA.csv
        └── ... (more stocks)
```

**Tips**:
- Each stock should have 250+ rows minimum
- Longer history is better (ideally 2000+ rows like your current data)
- Mix of different sectors is ideal
- Files can have various column naming (Open/OPEN/open, etc.) - script handles this

### Step 2: Aggregate Data

Combine all stock files into a single training dataset:

```bash
cd experimental/ml_prediction
python aggregate_data.py --input_dir data/stocks --output_file data/combined_stocks.csv
```

This will:
- Load and validate all stock CSVs
- Standardize formats
- Add stock identifiers
- Save combined dataset

**Expected output**:
```
Found 7 CSV files in data/stocks
Loading AAPL... ✓ 2617 rows (2015-01-02 to 2025-07-16)
Loading MSFT... ✓ 2500 rows (2015-01-02 to 2025-06-30)
...
Total stocks loaded: 7
Total rows: 18,000
```

### Step 3: Train Multi-Stock Model

Train a new model on the combined dataset:

```bash
python train_multi_stock.py \
    --data_file data/combined_stocks.csv \
    --model_name multi_stock_v1 \
    --eval_threshold 0.67
```

**Optional**: Hold out specific stocks for testing:
```bash
python train_multi_stock.py \
    --data_file data/combined_stocks.csv \
    --model_name multi_stock_v1 \
    --holdout_stocks AAPL MSFT
```

This will:
- Generate 63 technical indicators for all stocks
- Split data with multi-stock strategy
- Train neural network (same architecture as baseline)
- Evaluate on test set
- Save model to `saved_models/multi_stock_v1/`

### Step 4: Optimize Threshold

Find the best threshold for your precision target:

```bash
python optimize_threshold.py \
    --model_path saved_models/multi_stock_v1 \
    --data_file data/combined_stocks.csv \
    --min_threshold 0.50 \
    --max_threshold 0.80
```

This will:
- Test 31 thresholds (0.50 to 0.80)
- Identify thresholds achieving 90%, 95%, 100% precision
- Create precision-recall curves
- Save results to `threshold_optimization/`

**Look for**:
- Threshold with 90-95% precision and maximum recall
- Compare with baseline (3 TP at 100% precision)

### Step 5: Compare with Baseline

Compare the new model against the original:

```bash
python compare_models.py \
    --models saved_models/high_precision_model saved_models/multi_stock_v1 \
    --labels "Baseline (Single Stock)" "Multi-Stock" \
    --data_file data/combined_stocks.csv \
    --thresholds 0.50 0.60 0.63 0.65 0.67 0.70
```

This will:
- Evaluate both models at multiple thresholds
- Create side-by-side comparison plots
- Generate detailed report
- Save to `model_comparison/`

### Step 6 (Optional): Train Ensemble

If single model doesn't achieve desired recall at 90%+ precision, train an ensemble:

```bash
python train_ensemble_consensus.py \
    --data_file data/combined_stocks.csv \
    --n_models 5 \
    --consensus_threshold 0.6 \
    --model_name ensemble_5models_v1
```

This will:
- Train 5 models with different random seeds
- Test consensus voting (require 3/5 models to agree = 60%)
- Evaluate at different consensus levels
- Save ensemble to `saved_models/ensemble_5models_v1/`

**Then compare all three**:
```bash
python compare_models.py \
    --models saved_models/high_precision_model \
             saved_models/multi_stock_v1 \
             saved_models/ensemble_5models_v1 \
    --labels "Baseline" "Multi-Stock" "Ensemble (5 models)" \
    --data_file data/combined_stocks.csv
```

## 🎯 Success Criteria

The improvement is successful if:

1. **Precision ≥ 90%** on held-out test data
2. **Recall > 2.3%** (current baseline)
3. **At least 10-20 signals** on test set (vs. current 3)
4. **Works across different stocks** (not just the training stock)

## 📝 Notes

- All scripts preserve the original 63 feature set
- Same neural network architecture [128, 64, 32]
- Balanced class weights (same training strategy)
- Walk-forward validation ensures no data leakage

## 🔄 Reverting if Needed

If improvements don't work as expected:

```bash
git revert HEAD  # Reverts to baseline commit
```

The baseline model (100% precision, 3 TP) is preserved in:
- `saved_models/high_precision_model/`
- Git commit: `c78e2f5` - "Add high-precision stock prediction model (100% precision baseline)"

## 📬 Next Steps

1. **Provide stock CSV files** in `data/stocks/` directory
2. Run the workflow steps above
3. Evaluate if improvements meet success criteria
4. If successful, use the new model at optimized threshold
5. If not successful, try ensemble or revert to baseline

---

**Questions or issues?** Check the individual script help:
```bash
python <script_name>.py --help
```

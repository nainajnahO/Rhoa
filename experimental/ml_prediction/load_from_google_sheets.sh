#!/bin/bash
# Example script to load stocks from Google Sheets and train model
#
# Usage: ./load_from_google_sheets.sh

# Set your Google Sheets URLs here
# Each line: STOCK_ID=URL
STOCK_URLS=(
    "AAPL=https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=0"
    "MSFT=https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=1"
    "GOOGL=https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=2"
)

echo "==================================================================="
echo "Loading stocks from Google Sheets and training model"
echo "==================================================================="

# Step 1: Aggregate data
echo ""
echo "Step 1: Aggregating stock data..."
python aggregate_data.py \
    --google_sheets "${STOCK_URLS[@]}" \
    --output_file data/combined_stocks.csv

if [ $? -ne 0 ]; then
    echo "Error: Failed to aggregate data"
    exit 1
fi

# Step 2: Train model
echo ""
echo "Step 2: Training multi-stock model..."
python train_multi_stock.py \
    --data_file data/combined_stocks.csv \
    --model_name multi_stock_google_sheets \
    --eval_threshold 0.67

if [ $? -ne 0 ]; then
    echo "Error: Failed to train model"
    exit 1
fi

# Step 3: Optimize threshold
echo ""
echo "Step 3: Optimizing threshold..."
python optimize_threshold.py \
    --model_path saved_models/multi_stock_google_sheets \
    --data_file data/combined_stocks.csv

if [ $? -ne 0 ]; then
    echo "Error: Failed to optimize threshold"
    exit 1
fi

# Step 4: Compare with baseline
echo ""
echo "Step 4: Comparing with baseline..."
python compare_models.py \
    --models saved_models/high_precision_model saved_models/multi_stock_google_sheets \
    --labels "Baseline" "Multi-Stock (Google Sheets)" \
    --data_file data/combined_stocks.csv

if [ $? -ne 0 ]; then
    echo "Error: Failed to compare models"
    exit 1
fi

echo ""
echo "==================================================================="
echo "✓ Complete! Check the output directories for results:"
echo "  - Model: saved_models/multi_stock_google_sheets/"
echo "  - Threshold optimization: threshold_optimization/"
echo "  - Model comparison: model_comparison/"
echo "==================================================================="

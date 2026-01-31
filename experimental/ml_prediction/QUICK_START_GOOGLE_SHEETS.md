# Quick Start: Using Google Sheets

## Your Google Sheet Example

Based on your URL, here's how to use it:

```python
# Your Google Sheet URL
url = "https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=0"
```

## If You Have Multiple Stocks in Different Tabs

If your Google Sheet has multiple tabs (one per stock), use different `gid` values:

```bash
python aggregate_data.py --google_sheets \
    "AAPL=https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=0" \
    "MSFT=https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=1" \
    "GOOGL=https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=2" \
    --output_file data/combined_stocks.csv
```

**Where**:
- `gid=0` = First tab
- `gid=1` = Second tab
- `gid=2` = Third tab
- etc.

## If All Your Stocks Are in One Sheet

If you only have one stock per Google Sheet, or want to use one tab for now:

```bash
python aggregate_data.py --google_sheets \
    "STOCK1=https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=0" \
    --output_file data/combined_stocks.csv
```

## Complete Workflow Example

Here's a complete example using your Google Sheets:

```bash
cd experimental/ml_prediction

# Step 1: Aggregate stocks from Google Sheets
python aggregate_data.py --google_sheets \
    "STOCK1=https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=0" \
    "STOCK2=https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=1" \
    "STOCK3=https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=2" \
    --output_file data/combined_stocks.csv

# Step 2: Train model
python train_multi_stock.py \
    --data_file data/combined_stocks.csv \
    --model_name multi_stock_v1

# Step 3: Optimize threshold
python optimize_threshold.py \
    --model_path saved_models/multi_stock_v1 \
    --data_file data/combined_stocks.csv

# Step 4: Compare with baseline
python compare_models.py \
    --models saved_models/high_precision_model saved_models/multi_stock_v1 \
    --labels "Baseline" "Multi-Stock" \
    --data_file data/combined_stocks.csv
```

## Using the Convenience Script

Edit `load_from_google_sheets.sh` with your URLs:

```bash
STOCK_URLS=(
    "STOCK1=https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=0"
    "STOCK2=https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=1"
    "STOCK3=https://docs.google.com/spreadsheets/d/1To-jrrwNBP1XbC4g7LOFdruBKAOdfe-zW3dJrBEkjC4/export?format=csv&gid=2"
)
```

Then run:
```bash
./load_from_google_sheets.sh
```

This will automatically:
1. ✓ Load and aggregate stocks
2. ✓ Train the model
3. ✓ Optimize threshold
4. ✓ Compare with baseline

## Data Format Required

Your Google Sheet tabs should have these columns:
```csv
Date,Open,High,Low,Close,Volume
1/2/2015 17:30:00,71.43,71.68,71.05,71.33,611903
1/5/2015 17:30:00,71.2,71.28,70.6,70.75,410586
...
```

## Tips

1. **Multiple Stocks**: Use 5-10 different stocks for best results
2. **Tab Organization**: One stock per tab makes management easier
3. **Naming**: Use meaningful stock IDs (AAPL, MSFT, GOOGL, etc.)
4. **Updates**: Just update the Google Sheet - no need to re-download files
5. **Sharing**: Make sure the sheet is publicly accessible or has link sharing enabled

## Troubleshooting

**Error: "Missing required columns"**
- Make sure your sheet has: Date, Open, High, Low, Close, Volume
- Column names are case-insensitive (date/Date/DATE all work)

**Error: "403 Forbidden"**
- Your sheet may be private
- Right-click sheet → Share → Change to "Anyone with the link"

**Error: "Only X rows (minimum 250)"**
- Each stock needs at least 250 rows of data
- More data is better (aim for 2000+ rows)

## Next Steps

After training:
1. Check `threshold_optimization/` for best threshold recommendations
2. Review `model_comparison/` to see improvement over baseline
3. If precision ≥ 90% and more TPs captured → Success! 🎉
4. If not satisfactory, try ensemble training or add more stocks

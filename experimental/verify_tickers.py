"""
Verify Stock Tickers

Quickly verifies all stock tickers in stocks.csv by pinging Yahoo Finance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from yahoo_finance_loader import fetch_stock_data

print("="*80)
print("VERIFYING STOCK TICKERS")
print("="*80)
print()

# Load stocks
stocks_file = Path(__file__).parent / 'stocks.csv'
df_stocks = pd.read_csv(stocks_file)

print(f"Found {len(df_stocks)} stocks to verify\n")

valid = []
invalid = []

for idx, row in df_stocks.iterrows():
    symbol = row['Ticker']
    name = row['Stock Name']
    
    print(f"[{idx+1}/{len(df_stocks)}] {symbol:20s} {name:30s} ... ", end='', flush=True)
    
    try:
        df = fetch_stock_data(symbol)
        if len(df) > 100:  # Must have at least 100 days of data
            print(f"✓ Valid ({len(df)} rows)")
            valid.append((symbol, name, len(df)))
        else:
            print(f"✗ Insufficient data ({len(df)} rows)")
            invalid.append((symbol, name, f"Only {len(df)} rows"))
    except Exception as e:
        error_msg = str(e)[:50]
        print(f"✗ Error: {error_msg}")
        invalid.append((symbol, name, error_msg))

print()
print("="*80)
print("VERIFICATION SUMMARY")
print("="*80)
print(f"Total stocks:   {len(df_stocks)}")
print(f"Valid tickers:  {len(valid)}")
print(f"Invalid tickers: {len(invalid)}")
print()

if invalid:
    print("Invalid Tickers:")
    print("-" * 80)
    for symbol, name, error in invalid:
        print(f"  {symbol:20s} {name:30s} - {error}")
    print()

print("="*80)
print(f"Result: {len(valid)}/{len(df_stocks)} tickers are valid")
print("="*80)

"""
Diagnose Infinity Value Issue

Investigates why Yahoo Finance data creates inf values during feature generation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from yahoo_finance_loader import fetch_stock_data
import logisticregression_predictor as helper

print("="*80)
print("DIAGNOSING INFINITY VALUE ISSUE")
print("="*80)
print()

# Fetch sample data
print("Step 1: Fetching HEXA-B.ST from Yahoo Finance...")
df_stock = fetch_stock_data('HEXA-B.ST')
print(f"✓ Fetched {len(df_stock)} rows")
print(f"  Date range: {df_stock['Date'].min()} to {df_stock['Date'].max()}")
print(f"  Columns: {list(df_stock.columns)}")
print()

# Show raw data sample
print("Step 2: Raw data sample (first 10 rows):")
print(df_stock.head(10))
print()

print("Step 3: Raw data statistics:")
print(df_stock[['Open', 'High', 'Low', 'Close', 'Volume']].describe())
print()

# Check for problematic values in raw data
print("Step 4: Checking for problematic values in raw data:")
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    zeros = (df_stock[col] == 0).sum()
    negatives = (df_stock[col] < 0).sum()
    nans = df_stock[col].isna().sum()
    infs = np.isinf(df_stock[col]).sum()
    
    if zeros > 0 or negatives > 0 or nans > 0 or infs > 0:
        print(f"  {col}: zeros={zeros}, negatives={negatives}, nans={nans}, infs={infs}")

print()

# Fetch OMX30
print("Step 5: Fetching OMX30 index...")
df_omx30 = fetch_stock_data('^OMX')
print(f"✓ Fetched {len(df_omx30)} rows")
print(f"  Date range: {df_omx30['Date'].min()} to {df_omx30['Date'].max()}")
print()

print("Step 6: OMX30 data sample (first 10 rows):")
print(df_omx30.head(10))
print()

# Generate features and track where inf appears
print("Step 7: Generating features...")
df_features = helper.create_features(df_stock, index_df=df_omx30)
print(f"✓ Generated {len(df_features.columns)} columns")
print()

# Check each column for inf values
print("Step 8: Checking which features have infinity values:")
inf_columns = []
for col in df_features.columns:
    if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
        inf_count = np.isinf(df_features[col]).sum()
        if inf_count > 0:
            inf_columns.append((col, inf_count))
            print(f"  {col}: {inf_count} inf values ({inf_count/len(df_features)*100:.1f}%)")

print()

if len(inf_columns) > 0:
    print(f"Step 9: Investigating the first feature with inf values...")
    problem_col = inf_columns[0][0]
    print(f"  Feature: {problem_col}")
    
    # Find rows with inf
    inf_mask = np.isinf(df_features[problem_col])
    inf_rows = df_features[inf_mask].head(5)
    
    print(f"\n  First 5 rows with inf in {problem_col}:")
    print(inf_rows[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', problem_col]])
    
    print(f"\n  Corresponding raw data:")
    for date in inf_rows['Date'].head(5):
        idx = df_stock[df_stock['Date'] == date].index
        if len(idx) > 0:
            print(f"\n    Date: {date}")
            print(f"    {df_stock.loc[idx[0], ['Open', 'High', 'Low', 'Close', 'Volume']].to_dict()}")
    
    print()
    
    # Check surrounding rows
    print(f"Step 10: Checking context around first inf value...")
    first_inf_idx = df_features[inf_mask].index[0]
    context_start = max(0, first_inf_idx - 5)
    context_end = min(len(df_features), first_inf_idx + 5)
    
    print(f"\n  Rows {context_start} to {context_end} (inf at row {first_inf_idx}):")
    context_df = df_features.iloc[context_start:context_end][['Date', 'Close', problem_col]]
    context_df['is_inf'] = np.isinf(context_df[problem_col])
    print(context_df)
    
else:
    print("No infinity values found! The issue must be elsewhere.")

print()
print("="*80)
print("COMPARISON: GOOGLE SHEETS FORMAT VS YAHOO FINANCE FORMAT")
print("="*80)
print()

print("Google Sheets expected format:")
print("  - Date, Open, High, Low, Close, Volume")
print("  - All values > 0")
print("  - No NaN, no inf")
print()

print("Yahoo Finance actual format:")
print(f"  - Columns: {list(df_stock.columns)}")
print(f"  - Date type: {df_stock['Date'].dtype}")
print(f"  - Close type: {df_stock['Close'].dtype}")
print(f"  - Min Close: {df_stock['Close'].min():.2f}")
print(f"  - Max Close: {df_stock['Close'].max():.2f}")
print(f"  - Any zeros in Close: {(df_stock['Close'] == 0).sum()}")
print(f"  - Any NaN in Close: {df_stock['Close'].isna().sum()}")
print()

print("="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)

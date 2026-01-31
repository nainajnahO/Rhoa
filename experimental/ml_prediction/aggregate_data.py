"""
Data aggregation script to combine multiple stock CSV files into a single training dataset.

This script:
1. Loads multiple stock CSV files from a specified directory
2. Standardizes the format (Date, OHLCV columns)
3. Adds stock identifier for tracking
4. Combines into a single dataset for training
5. Validates data quality and temporal ordering

Usage:
    python aggregate_data.py --input_dir data/stocks --output_file data/combined_stocks.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import argparse


def load_single_stock(filepath: str, stock_id: Optional[str] = None) -> pd.DataFrame:
    """
    Load a single stock CSV file and standardize the format.
    
    Args:
        filepath: Path to the CSV file
        stock_id: Stock identifier (if None, uses filename without extension)
        
    Returns:
        DataFrame with standardized columns and stock_id
    """
    # Read CSV
    df = pd.read_csv(filepath)
    
    # Standardize column names (handle various formats)
    column_mapping = {
        'date': 'Date',
        'DATE': 'Date',
        'open': 'Open',
        'OPEN': 'Open',
        'high': 'High',
        'HIGH': 'High',
        'low': 'Low',
        'LOW': 'Low',
        'close': 'Close',
        'CLOSE': 'Close',
        'Close*': 'Close',
        'Adj Close': 'Close',
        'volume': 'Volume',
        'VOLUME': 'Volume',
        'Vol.': 'Volume',
    }
    
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {filepath}: {missing_cols}")
    
    # Keep only required columns
    df = df[required_cols].copy()
    
    # Parse date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['Date'])
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Add stock identifier
    if stock_id is None:
        stock_id = Path(filepath).stem
    df['stock_id'] = stock_id
    
    # Convert numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with NaN in OHLCV
    df = df.dropna(subset=numeric_cols)
    
    # Basic validation
    if len(df) < 100:
        print(f"Warning: {stock_id} has only {len(df)} valid rows after cleaning")
    
    # Check for price anomalies (e.g., all zeros, negative prices)
    if (df['Close'] <= 0).any():
        print(f"Warning: {stock_id} has invalid prices (<=0)")
        df = df[df['Close'] > 0]
    
    return df


def aggregate_stocks(
    input_dir: str,
    output_file: Optional[str] = None,
    file_pattern: str = "*.csv",
    min_rows_per_stock: int = 250
) -> pd.DataFrame:
    """
    Aggregate multiple stock CSV files into a single dataset.
    
    Args:
        input_dir: Directory containing stock CSV files
        output_file: Optional path to save combined dataset
        file_pattern: Pattern to match CSV files (default: "*.csv")
        min_rows_per_stock: Minimum rows required per stock (default: 250)
        
    Returns:
        Combined DataFrame with all stocks
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all CSV files
    csv_files = list(input_path.glob(file_pattern))
    
    if not csv_files:
        raise ValueError(f"No CSV files found matching pattern '{file_pattern}' in {input_dir}")
    
    print(f"Found {len(csv_files)} CSV files in {input_dir}")
    print("-" * 80)
    
    # Load all stocks
    all_stocks = []
    skipped_stocks = []
    
    for csv_file in csv_files:
        stock_id = csv_file.stem
        try:
            print(f"Loading {stock_id}...", end=" ")
            df = load_single_stock(str(csv_file), stock_id=stock_id)
            
            if len(df) < min_rows_per_stock:
                print(f"SKIP (only {len(df)} rows, minimum {min_rows_per_stock})")
                skipped_stocks.append(stock_id)
                continue
            
            all_stocks.append(df)
            date_range = f"{df['Date'].min().date()} to {df['Date'].max().date()}"
            print(f"✓ {len(df)} rows ({date_range})")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            skipped_stocks.append(stock_id)
    
    if not all_stocks:
        raise ValueError("No valid stocks loaded. Check your data files.")
    
    # Combine all stocks
    combined_df = pd.concat(all_stocks, ignore_index=True)
    
    # Sort by stock_id and Date to maintain temporal order within each stock
    combined_df = combined_df.sort_values(['stock_id', 'Date']).reset_index(drop=True)
    
    print("-" * 80)
    print(f"\n📊 Aggregation Summary:")
    print(f"  Total stocks loaded: {len(all_stocks)}")
    print(f"  Stocks skipped: {len(skipped_stocks)}")
    if skipped_stocks:
        print(f"    Skipped: {', '.join(skipped_stocks)}")
    print(f"  Total rows: {len(combined_df):,}")
    print(f"  Date range: {combined_df['Date'].min().date()} to {combined_df['Date'].max().date()}")
    print(f"\n  Rows per stock:")
    for stock_id in combined_df['stock_id'].unique():
        count = len(combined_df[combined_df['stock_id'] == stock_id])
        print(f"    {stock_id}: {count:,}")
    
    # Save if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved combined dataset to: {output_file}")
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple stock CSV files into a single training dataset"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/stocks',
        help='Directory containing stock CSV files (default: data/stocks)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='data/combined_stocks.csv',
        help='Output file for combined dataset (default: data/combined_stocks.csv)'
    )
    parser.add_argument(
        '--file_pattern',
        type=str,
        default='*.csv',
        help='Pattern to match CSV files (default: *.csv)'
    )
    parser.add_argument(
        '--min_rows',
        type=int,
        default=250,
        help='Minimum rows per stock to include (default: 250)'
    )
    
    args = parser.parse_args()
    
    # Make paths relative to script location
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir
    output_file = script_dir / args.output_file
    
    print("="*80)
    print("STOCK DATA AGGREGATION")
    print("="*80)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"File pattern: {args.file_pattern}")
    print(f"Minimum rows per stock: {args.min_rows}")
    print("="*80)
    print()
    
    try:
        combined_df = aggregate_stocks(
            input_dir=str(input_dir),
            output_file=str(output_file),
            file_pattern=args.file_pattern,
            min_rows_per_stock=args.min_rows
        )
        
        print("\n" + "="*80)
        print("✓ AGGREGATION COMPLETE!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

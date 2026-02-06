"""
Yahoo Finance Data Loader

Fetches stock data from Yahoo Finance with intelligent caching.
Free, fast, and supports Stockholm Exchange stocks.

Features:
- No API key required
- Fast data fetching (~0.1s per stock)
- Automatic caching to disk
- Support for Stockholm stocks (.ST suffix)
- No rate limits
"""

import yfinance as yf
import pandas as pd
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Cache configuration
OUTPUT_DIR = Path(__file__).parent / 'outputs'
CACHE_DIR = OUTPUT_DIR / 'cache' / 'yahoo_finance'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def fetch_stock_data(symbol, force_refresh=False):
    """
    Fetch stock data from Yahoo Finance with caching
    
    Args:
        symbol: Stock symbol (e.g., 'HEXA-B.ST' for Stockholm stocks)
        force_refresh: If True, bypass cache and fetch fresh data
        
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
        
    Raises:
        Exception: If data fetch fails or data is invalid
    """
    cache_file = CACHE_DIR / f"{symbol.replace('.', '_').replace('-', '_')}.csv"
    
    # Check cache
    if cache_file.exists() and not force_refresh:
        try:
            df = pd.read_csv(cache_file)
            df['Date'] = pd.to_datetime(df['Date'])
            print(f"  Loading {symbol} from cache...")
            return df
        except Exception as e:
            print(f"  Cache read error for {symbol}: {e}")
            # Continue to fetch from Yahoo Finance
    
    # Fetch from Yahoo Finance
    print(f"  Fetching {symbol} from Yahoo Finance...")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max", start="2015-01-01")
        
        if df.empty:
            raise Exception(f"No data available for {symbol}")
        
        # Reset index to have Date as a column
        df = df.reset_index()
        
        # Ensure Date column is present
        if 'Date' not in df.columns:
            raise Exception(f"Date column missing for {symbol}")
        
        # Select and rename columns to match expected format
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Convert Date to datetime and remove timezone info for consistency
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # Filter from 2015 onwards
        df = df[df['Date'] >= '2015-01-01']
        
        if len(df) == 0:
            raise Exception(f"No data available from 2015 onwards for {symbol}")
        
        # Cache the data
        df.to_csv(cache_file, index=False)
        
        return df
        
    except Exception as e:
        raise Exception(f"Failed to fetch {symbol}: {str(e)}")

def clear_cache():
    """Delete all cached files"""
    if CACHE_DIR.exists():
        for cache_file in CACHE_DIR.glob('*.csv'):
            cache_file.unlink()
        print(f"Cleared {CACHE_DIR}")

def get_cache_info():
    """Get information about cached files"""
    if not CACHE_DIR.exists():
        return "No cache directory"
    
    cache_files = list(CACHE_DIR.glob('*.csv'))
    if not cache_files:
        return "Cache is empty"
    
    total_size = sum(f.stat().st_size for f in cache_files)
    return f"{len(cache_files)} cached stocks, {total_size / 1024 / 1024:.1f} MB"

if __name__ == '__main__':
    # Test the loader
    print("Testing Yahoo Finance Loader")
    print("="*80)
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Cache status: {get_cache_info()}")
    print()
    
    # Test with Stockholm stocks
    try:
        test_symbol = 'HEXA-B.ST'
        print(f"Testing with {test_symbol}...")
        df = fetch_stock_data(test_symbol)
        print(f"✓ Successfully fetched {len(df)} rows")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        print(f"\nCache status: {get_cache_info()}")
    except Exception as e:
        print(f"✗ Error: {e}")

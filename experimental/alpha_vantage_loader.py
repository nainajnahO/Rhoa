"""
Alpha Vantage Data Loader

Fetches stock data from Alpha Vantage API with intelligent caching
to minimize API calls and respect rate limits.

Features:
- Automatic caching to disk (JSON format)
- Rate limit protection
- Data parsing and normalization
- Date filtering (2015 onwards)
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path

# API Configuration
API_KEY = '63X34508IYCBBIAV'
CACHE_DIR = Path(__file__).parent / 'alpha_vantage_cache'
CACHE_DIR.mkdir(exist_ok=True)

def fetch_stock_data(symbol, force_refresh=False):
    """
    Fetch stock data from Alpha Vantage with caching
    
    Args:
        symbol: Stock symbol (e.g., 'HEXA-B.STO' for Stockholm stocks)
        force_refresh: If True, bypass cache and fetch fresh data
        
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
        
    Raises:
        Exception: If API call fails or data is invalid
    """
    cache_file = CACHE_DIR / f"{symbol.replace('.', '_').replace('-', '_')}.json"
    
    # Check cache
    if cache_file.exists() and not force_refresh:
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            print(f"  Loading {symbol} from cache...")
            return parse_alpha_vantage_data(data, symbol)
        except Exception as e:
            print(f"  Cache read error for {symbol}: {e}")
            # Continue to fetch from API
    
    # Fetch from API
    print(f"  Fetching {symbol} from Alpha Vantage API...")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={API_KEY}"
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"API error: HTTP {response.status_code}")
        
        data = response.json()
        
        # Check for error messages
        if "Error Message" in data:
            raise Exception(f"Invalid symbol: {symbol}")
        if "Note" in data:  # Rate limit message
            raise Exception("Rate limit reached - please wait and try again")
        if "Information" in data:  # API key issue
            raise Exception(f"API issue: {data['Information']}")
        
        # Cache the response
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        
        return parse_alpha_vantage_data(data, symbol)
        
    except requests.exceptions.Timeout:
        raise Exception("Request timeout")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")

def parse_alpha_vantage_data(data, symbol):
    """
    Parse Alpha Vantage JSON response to pandas DataFrame
    
    Args:
        data: JSON response from Alpha Vantage
        symbol: Stock symbol (for error messages)
        
    Returns:
        DataFrame with normalized columns and date filtering
    """
    # Try different time series keys (API can return different formats)
    time_series = None
    for key in ["Time Series (Daily)", "Time Series (Adjusted)"]:
        if key in data:
            time_series = data[key]
            break
    
    if not time_series:
        raise Exception(f"No time series data found for {symbol}")
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Rename columns (handle both adjusted and non-adjusted formats)
    column_mapping = {}
    for col in df.columns:
        if 'open' in col.lower():
            column_mapping[col] = 'Open'
        elif 'high' in col.lower():
            column_mapping[col] = 'High'
        elif 'low' in col.lower():
            column_mapping[col] = 'Low'
        elif 'close' in col.lower() and 'adjusted' not in col.lower():
            column_mapping[col] = 'Close'
        elif 'volume' in col.lower():
            column_mapping[col] = 'Volume'
    
    df = df.rename(columns=column_mapping)
    
    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise Exception(f"Missing columns: {missing_cols}")
    
    # Convert to numeric
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter from 2015 onwards
    df = df[df.index >= '2015-01-01']
    
    if len(df) == 0:
        raise Exception(f"No data available from 2015 onwards for {symbol}")
    
    # Reset index to have Date column
    df = df.reset_index()
    df.rename(columns={'index': 'Date'}, inplace=True)
    
    return df[required_cols[:4] + ['Volume', 'Date']].rename(columns={'Date': 'Date'})[[
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
    ]]

def clear_cache():
    """Delete all cached files"""
    if CACHE_DIR.exists():
        for cache_file in CACHE_DIR.glob('*.json'):
            cache_file.unlink()
        print(f"Cleared {CACHE_DIR}")

def get_cache_info():
    """Get information about cached files"""
    if not CACHE_DIR.exists():
        return "No cache directory"
    
    cache_files = list(CACHE_DIR.glob('*.json'))
    if not cache_files:
        return "Cache is empty"
    
    total_size = sum(f.stat().st_size for f in cache_files)
    return f"{len(cache_files)} cached stocks, {total_size / 1024 / 1024:.1f} MB"

if __name__ == '__main__':
    # Test the loader
    print("Testing Alpha Vantage Loader")
    print("="*80)
    print(f"API Key: {API_KEY[:10]}...")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Cache status: {get_cache_info()}")
    print()
    
    # Test with a sample stock
    try:
        test_symbol = 'IBM'  # Use US stock for testing
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

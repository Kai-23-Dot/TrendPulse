"""
Script to fetch all US stock tickers from public sources.
Creates a local database (CSV) that the app loads at startup.

Run this once to generate the database:
    python3 fetch_tickers.py
"""

import pandas as pd
import requests
from pathlib import Path

def fetch_nasdaq_tickers():
    """Fetch all NASDAQ-listed companies."""
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            tickers = [t.strip() for t in response.text.split('\n') if t.strip()]
            return tickers
    except:
        pass
    return []

def fetch_nyse_tickers():
    """Fetch all NYSE-listed companies."""
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.txt"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            tickers = [t.strip() for t in response.text.split('\n') if t.strip()]
            return tickers
    except:
        pass
    return []

def fetch_amex_tickers():
    """Fetch all AMEX-listed companies."""
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex/amex_tickers.txt"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            tickers = [t.strip() for t in response.text.split('\n') if t.strip()]
            return tickers
    except:
        pass
    return []

def main():
    print("Fetching US stock tickers...")
    
    nasdaq = fetch_nasdaq_tickers()
    print(f"  NASDAQ: {len(nasdaq)} tickers")
    
    nyse = fetch_nyse_tickers()
    print(f"  NYSE: {len(nyse)} tickers")
    
    amex = fetch_amex_tickers()
    print(f"  AMEX: {len(amex)} tickers")
    
    # Combine and deduplicate
    all_tickers = sorted(set(nasdaq + nyse + amex))
    
    # Remove invalid tickers (those with special characters)
    valid_tickers = [t for t in all_tickers if t.isalpha() or (t.replace('.', '').replace('-', '').isalnum())]
    
    print(f"\nTotal unique tickers: {len(valid_tickers)}")
    
    # Save to file
    output_path = Path("data/us_tickers.txt")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(valid_tickers))
    
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()

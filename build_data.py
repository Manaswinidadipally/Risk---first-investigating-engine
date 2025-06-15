import yfinance as yf
import pandas as pd
from datetime import datetime

def get_sp500_tickers():
    """Get S&P 500 tickers from Wikipedia"""
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    tickers = table[0]['Symbol'].tolist()
    tickers = [t.replace('.', '-') for t in tickers]  # fix BRK.B, etc.
    return tickers

def download_price_data(tickers, start="2022-01-01", end="2024-12-31"):
    """Download adjusted close prices"""
    all_data = yf.download(tickers, start=start, end=end, progress=True, group_by='ticker', auto_adjust=True)

    price_df = pd.DataFrame(index=all_data.index)

    for ticker in tickers:
        try:
            price_df[ticker] = all_data[ticker]['Close']
        except Exception:
            continue  # skip if data not available

    return price_df

if __name__ == "__main__":
    tickers = get_sp500_tickers()
    print(f"Downloading data for {len(tickers)} S&P 500 stocks...")

    df = download_price_data(tickers)
    print("Downloaded shape:", df.shape)

    df.to_csv("SP500_2022_2024.csv")
    print("✅ Saved to SP500_2022_2024.csv")

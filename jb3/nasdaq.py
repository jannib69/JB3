import pandas as pd
import requests
import time
import random
from datetime import datetime

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")  # ignore all warnings

def download_ticker_data(
    tickers: list[str] | str,
    start: str = "1999-01-01",
    end: str | None = None,
    limit: int = 9999,
    sleep_range: tuple[float, float] = (0.1, 0.3),
    pivot_col: str = None,
    clean: bool = False,
) -> pd.DataFrame:
    """
    Download historical NASDAQ stock data.

    Args:
        tickers (list[str] | str): Stock tickers (e.g., "AAPL" or ["AAPL","MSFT"])
        start (str): Start date in 'YYYY-MM-DD'. Default '1999-01-01'.
        end (str | None): End date in 'YYYY-MM-DD'. Default today.
        limit (int): Maximum rows per ticker. Default 9999.
        sleep_range (tuple): Random sleep range in seconds between requests. Default (0.1,0.3)
        pivot_cols (list[str]): Columns to pivot on. Default [].
        clean (bool): Convert numeric columns and set index. Default False.

    Returns:
        pd.DataFrame: Combined historical data. Pivoted if pivot_cols given.
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/"
    }

    all_data = []

    if isinstance(tickers, str):
        tickers = [tickers]

    for ticker in tqdm(tickers, desc="Downloading NASDAQ data"):
        url = (
            f"https://api.nasdaq.com/api/quote/{ticker}/historical?"
            f"assetclass=stocks&fromdate={start}&todate={end}&limit={limit}"
        )

        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()

            rows = data.get("data", {}).get("tradesTable", {}).get("rows", [])
            if not rows:
                continue

            df = pd.DataFrame(rows)
            df["Ticker"] = ticker
            all_data.append(df)

        except Exception as e:
            print(f"Warning: could not fetch data for {ticker}: {e}")

        time.sleep(random.uniform(*sleep_range))

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)

    if clean:
        for col in ["close", "volume", "open", "high", "low"]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].str.replace(",", "").str.replace("$", ""),
                    errors="coerce"
                )
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    if pivot_col is not None:
        if clean:
            df["date"] = df.index

        df =df.pivot(index="date", columns="Ticker", values=pivot_col)

    df.sort_index(inplace=True)

    return df


import requests
import pandas as pd


def get_nasdaq_screener(
        limit: int = 25,
        exchange: str = None,
        marketcap: str = None,
        recommendation: str = None,
        sector: str = None,
        region: str = None,
        country: str | None = None

) -> pd.DataFrame:
    """
    NASDAQ screener data

    Exchange: NASDAQ, Global Select, Global Market, Capital Market, ADR, NYSE, AMEX
    Market Cap: Mega, Large, Medium, Small, Micro, Nano
    Analyst Rating: Strong Buy, Hold, Buy, Sell, Strong Sell
    Sector: Technology, Telecommunications, Healthcare, Financials, Real Estate,
            Consumer Discretionary, Consumer Staples, Industrials, Basic Materials, Energy, Utilities
    Region: Africa, Asia, Australia and South Pacific, Caribbean, Europe, Middle East, North America, South America
    Country: United States, Israel, Malaysia, United Kingdom, Netherlands, China, Canada,
             South Korea, Guernsey, Singapore, Taiwan, Switzerland, Belgium, Cayman Islands,
             Norway, Brazil, Ireland, Australia
    """
    url = "https://api.nasdaq.com/api/screener/stocks?"
    params = {"tableonly": "false", "limit": limit}

    if exchange: params["exchange"] = exchange
    if marketcap: params["marketcap"] = marketcap
    if recommendation: params["recommendation"] = recommendation
    if sector: params["sector"] = sector
    if region: params["region"] = region
    if country: params["country"] = country  # Lahko več držav: "United States|China"

    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "origin": "https://www.nasdaq.com",
        "priority": "u=1, i",
        "referer": "https://www.nasdaq.com/",
        "sec-ch-ua": '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/143.0.0.0 Safari/537.36"
        )
    }

    r = requests.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    rows = data.get("data", {}).get("table", {}).get("rows", [])
    df = pd.DataFrame(rows)
    return df
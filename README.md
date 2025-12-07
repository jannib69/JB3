# JB3

JB3 is a Python package providing unified access to LJSE, NASDAQ, and FRED (Federal Reserve Economic Data).
It enables unified downloading, cleaning, transforming, and pivoting of financial and macroeconomic time-series data
for analysis, trading strategies, or machine-learning pipelines.

## Features

### NASDAQ & LJSE Stock Market Data
- Download historical NASDAQ and LJSE stock data
- Clean and normalize numeric columns (open, high, low, close, volume)
- Pivot data by ticker for analysis-ready DataFrames
- Screener with flexible filtering:
  - Exchange
  - Market cap
  - Analyst rating
  - Sector
  - Region
  - Country

### FRED Macroeconomic Data
- Search FRED series by keyword (e.g., GDP, CPI, M2, UNRATE)
- Download multiple macroeconomic series at once
- Automatic detection of the lowest valid FRED frequency (Daily → Weekly → Monthly → Quarterly → Annual)
- Optional server-side unit transformations:
  - pch = percent change
  - pc1 = percent change from year ago (YoY)
  - chg = change from previous period
  - log = natural log
- Clean, aligned DataFrames with datetime index
- Built-in retry logic for stable API access

## Installation

pip install jb3

## Usage Examples

### 1. Download NASDAQ Historical Data

from jb3 import nasdaq

df = nasdaq.download_ticker_data("AAPL", clean=True, pivot_col="close")
print(df.head())

### 2. NASDAQ Screener Example

from jb3 import nasdaq

screener = nasdaq.get_nasdaq_screener(
    exchange="NASDAQ",
    marketcap="Large",
    sector="Technology",
    limit=50
)

print(screener.head())

### 3. FRED Example — GDP & CPI (auto frequency detection)

from jb3 import fred

df = fred.get_fred_series(
    ["GDP", "CPIAUCSL"],
    api_key="YOUR_FRED_API_KEY"
)
print(df.tail())

### 4. FRED Example — YoY Inflation (pc1 transformation)

df = fred.get_fred_series(
    "CPIAUCSL",
    api_key="YOUR_FRED_API_KEY",
    units="pc1"
)
print(df.head())

### 5. Search FRED Tickers (GDP-related)

tickers = fred.get_fred_tickers("GDP", api_key="YOUR_FRED_API_KEY")
print(tickers[["id", "title", "frequency"]].head())
  
## LJSE Documentation

https://pypi.org/project/LJSE/

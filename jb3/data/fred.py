import pandas as pd
import requests
import time
import random
from datetime import datetime
from tqdm import tqdm
from requests.adapters import HTTPAdapter, Retry


# ---------------------------------------------------------
#  METADATA -> detect lowest frequency automatically
# ---------------------------------------------------------

FREQ_MAP = {
    "Daily": "d",
    "Weekly": "w",
    "Biweekly": "bw",
    "Monthly": "m",
    "Quarterly": "q",
    "Semiannual": "sa",
    "Annual": "a"
}

FREQ_ORDER = ["d", "w", "bw", "m", "q", "sa", "a"]  # from lowest to highest granularity


def fred_get_metadata(series_id: str, api_key: str) -> dict | None:
    """Fetch metadata for a FRED series."""
    url = (
        "https://api.stlouisfed.org/fred/series"
        f"?series_id={series_id}&api_key={api_key}&file_type=json"
    )
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None

    data = r.json().get("seriess", [])
    return data[0] if data else None


def _fred_detect_lowest_freq(series_id: str, api_key: str) -> str | None:
    """Return the lowest available frequency for a series."""
    meta = fred_get_metadata(series_id, api_key)
    if meta is None:
        return None

    long_freq = meta.get("frequency", None)
    if long_freq not in FREQ_MAP:
        return None

    short_freq = FREQ_MAP[long_freq]

    # The series only has ONE frequency in FRED, so the shortest available = its own
    return short_freq


# ---------------------------------------------------------
#  MAIN FUNCTION — now WITHOUT frequency parameter
# ---------------------------------------------------------

def get_fred_series(
        series_ids: list[str] | str,
        api_key: str,
        start: str = "1950-01-01",
        end: str | None = None,
        units: str | None = None,
        sleep_range: tuple[float, float] = (0.1, 0.3),
        clean: bool = True
) -> pd.DataFrame:
    """
    Download multiple FRED time series with:
    - automatic lowest-frequency detection (Daily → Weekly → Monthly → Quarterly → Annual)
    - optional FRED unit transformation (pch, pc1, log,...)
    - cleaned and aligned numeric time series

    FRED automatically supports only ONE frequency per series, so
    this function uses the NATIVELY LOWEST frequency available.

    PARAMETERS
    ----------
    series_ids : list[str] or str
        FRED series IDs ("GDP", "CPIAUCSL", etc.)
    api_key : str
        Your FRED API key.
    start : str
        Start date.
    end : str or None
        End date, default today.
    units : str or None
        FRED transformations, e.g.:
        "pc1" = percent change YoY
        "pch" = percent change
        "log" = natural log
        "chg" = change from previous period
    sleep_range : tuple
        Random delay between API calls.
    clean : bool
        Clean numeric values and use datetime index.

    RETURNS
    -------
    pd.DataFrame
        Combined time series aligned by date.

    EXAMPLES
    --------
    df = get_fred_series(["GDP", "CPIAUCSL"], api_key)

    df = get_fred_series("CPIAUCSL", api_key, units="pc1")  # YoY percent change
    """

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    if isinstance(series_ids, str):
        series_ids = [series_ids]

    base_url = "https://api.stlouisfed.org/fred/series/observations"

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    all_data = []

    for sid in tqdm(series_ids, desc="Downloading FRED series"):

        # ------------------------------------------------------
        # AUTO-DETECT TRUE LOWEST FREQUENCY
        # ------------------------------------------------------
        freq = _fred_detect_lowest_freq(sid, api_key)
        if freq is None:
            print(f"Skipping {sid}: no frequency found.")
            continue

        # Apply units if provided
        unit_str = f"&units={units}" if units else ""

        url = (
            f"{base_url}?series_id={sid}&api_key={api_key}"
            f"&file_type=json&frequency={freq}"
            f"{unit_str}"
            f"&observation_start={start}&observation_end={end}"
        )

        try:
            r = session.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()

            rows = data.get("observations", [])
            if not rows:
                print(f"No observations for {sid}, skipping.")
                continue

            df = pd.DataFrame(rows)

            if clean:
                df["date"] = pd.to_datetime(df["date"])
                df[sid] = pd.to_numeric(df["value"], errors="coerce")
                df = df[["date", sid]]
                df.set_index("date", inplace=True)

            all_data.append(df)

        except Exception as e:
            print(f"Warning: could not fetch {sid}: {e}")

        time.sleep(random.uniform(*sleep_range))

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, axis=1)
    df.sort_index(inplace=True)
    return df


# ---------------------------------------------------------
# GET FRED TICKERS: Search series by keyword
# ---------------------------------------------------------

def get_fred_tickers(search_text: str, api_key: str, limit: int = 1000) -> pd.DataFrame:
    """
    Search FRED series by keyword.
    Returns metadata such as:
    - id (series_id)
    - title
    - frequency
    - units
    - seasonal_adjustment
    - notes
    """

    url = "https://api.stlouisfed.org/fred/series/search"

    params = dict(
        search_text=search_text,
        api_key=api_key,
        file_type="json",
        limit=limit
    )

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    r = session.get(url, params=params, timeout=10)
    r.raise_for_status()

    data = r.json().get("seriess", [])
    return pd.DataFrame(data)
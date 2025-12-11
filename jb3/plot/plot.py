import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_returns(
    df: pd.DataFrame,
    start: str = None,
    end: str = None,
    avg: bool = False,
    figsize=(12, 6)
):
    """
    Plot cumulative returns for a time series of stock prices.

    Parameters
    ----------
    df : pd.DataFrame
        Time-series price data. Index must be DatetimeIndex. Columns = tickers.
    start : str or None
        Start date (YYYY-MM-DD). If None, uses the earliest date in the index.
    end : str or None
        End date. If None, uses the latest date in the index.
    avg : bool
        If True, also plot the average cumulative return across all tickers.
    figsize : tuple
        Size of the figure.
    """

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    # Slice range
    if start is None:
        start = df.index.min()
    if end is None:
        end = df.index.max()

    df_slice = df.loc[start:end]

    # Compute cumulative returns: (price / first_price) - 1
    cumulative = df_slice / df_slice.iloc[0] - 1

    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)

    # Plot cumulative returns
    for col in cumulative.columns:
        plt.plot(cumulative.index, cumulative[col], label=f"{col} cumulative", alpha=0.8)

    # Optional: plot average cumulative return
    if avg:
        avg_series = cumulative.mean(axis=1)
        plt.plot(
            avg_series.index,
            avg_series,
            label="Average cumulative return",
            linewidth=2.5,
            color="black"
        )

    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_returns_heatmap(series: pd.Series, freq: str = "M",
                         figsize=(14, 6), cmap="RdYlGn"):
    """
    Heatmap returns for frequencies: D, W, M, Q, A.

    Rules:
    - D, W → no annotation on heatmap (too many cells)
    - M, Q, A → annotated values
    """

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have DatetimeIndex")

    freq = freq.upper()
    if freq not in ["D", "W", "M", "Q", "A"]:
        raise ValueError("freq must be in {D, W, M, Q, A}")

    # Compute returns
    resampled = series.resample(freq).last()
    returns = resampled.pct_change() * 100
    returns = returns.dropna()

    idx = returns.index
    years = idx.year

    # Define period per frequency
    if freq == "D":
        periods = idx.day      # 1–31 (for readability)
    elif freq == "W":
        periods = idx.isocalendar().week.astype(int)
    elif freq == "M":
        periods = idx.month
    elif freq == "Q":
        periods = idx.quarter
    elif freq == "A":
        periods = np.ones(len(idx), dtype=int)

    df = pd.DataFrame({
        "year": years,
        "period": periods,
        "return": returns.values
    })

    pivot = df.pivot_table(
        index="year",
        columns="period",
        values="return",
        aggfunc="mean"
    ).sort_index(axis=1)

    # Decide annotation
    use_annot = False if freq.startswith("W") or freq == "D" else True

    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot,
        cmap=cmap,
        center=0,
        annot=use_annot,
        fmt=".1f",
        linewidths=0.2,
        linecolor="gray"
    )

    plt.title(f"Returns Heatmap ({freq}-frequency)")
    plt.xlabel("Period")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.show()

    return pivot

def plot_average_returns_by_period(
    series: pd.Series,
    period: str = "weekday",
    figsize=(12, 5),
    color="steelblue"
):
    """
    Plot the average percentage returns of a price time series grouped by a chosen calendar period.

    Parameters
    ----------
    series : pd.Series
        Price time series indexed with a DatetimeIndex. Values must be numeric.

    period : str, optional
        Time grouping period. Supported:
            - "weekday" : Average returns by day of the week (Mon–Fri)
            - "month"   : Average returns by month of year (Jan–Dec)
            - "quarter" : Average returns by quarter (Q1–Q4)
            - "day"     : Average returns by day of month (1–31)
            - "week"    : Average returns by ISO week number (1–53)

    figsize : tuple, optional
        Matplotlib figure size.

    color : str, optional
        Bar color.

    Returns
    -------
    pandas.Series
        A Series of grouped average returns in percentage units, with cleaned labels.

    """

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have DatetimeIndex")

    series = series.sort_index()

    # compute returns with correct alignment
    ret = series.pct_change().dropna()

    idx = ret.index
    period = period.lower()

    # ---- GROUP ASSIGNMENT ----
    if period == "weekday":
        groups = idx.weekday                     # 0–6
        order = [0,1,2,3,4]                      # Mon–Fri
        labels = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        mapping = dict(zip(order, labels))

    elif period == "month":
        groups = idx.month
        order = sorted(groups.unique())
        mapping = {m: idx[idx.month == m].month_name().iloc[0] for m in order}

    elif period == "quarter":
        groups = idx.quarter
        order = sorted(groups.unique())
        mapping = {q: f"Q{q}" for q in order}

    elif period == "day":
        groups = idx.day
        order = sorted(groups.unique())
        mapping = {d: str(d) for d in order}

    elif period == "week":
        groups = idx.isocalendar().week.astype(int)
        order = sorted(groups.unique())
        mapping = {w: f"W{w}" for w in order}

    else:
        raise ValueError("Invalid period")

    # ---- BUILD ALIGNED DATAFRAME ----
    df = pd.DataFrame({"grp": groups, "ret": ret})
    df = df[df["grp"].isin(order)]

    avg = df.groupby("grp")["ret"].mean().loc[order] * 100

    avg.index = [mapping[g] for g in avg.index]

    # ---- PLOT ----
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)
    sns.barplot(x=avg.index, y=avg.values, color=color, edgecolor="black")

    plt.title(f"Average Returns by {period.capitalize()}")
    plt.ylabel("Avg Return (%)")
    plt.xlabel(period.capitalize())
    plt.axhline(0, color="black", linewidth=1)
    plt.tight_layout()
    plt.show()

    return avg
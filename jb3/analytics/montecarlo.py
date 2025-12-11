import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mc_bootstrap_simulation(
    series: pd.Series,
    n_sim: int = 500,
    horizon: int = 20,
    use_log_returns: bool = True,
    figsize=(14, 8),
    n_paths_to_show: int = 60,
    theme: str = "dark",
):
    # -------------------------------
    # PLOT THEME
    # -------------------------------
    if theme == "dark":
        plt.style.use("dark_background")
        hist_color = "white"
        median_color = "#4f7fff"
        interval_color = "#1f3b73"
        sim_color = "gray"
    else:
        plt.style.use("default")
        hist_color = "black"
        median_color = "blue"
        interval_color = "lightblue"
        sim_color = "gray"

    # -------------------------------
    # CLEAN SERIES
    # -------------------------------
    series = series.dropna().astype(float)
    last_price = series.iloc[-1]
    last_idx = series.index[-1]

    # -------------------------------
    # RETURNS (no drift added!)
    # -------------------------------
    if use_log_returns:
        r = np.log(series / series.shift(1)).dropna()
        innovations = r.values  # already contains empirical drift
    else:
        r = series.pct_change().dropna()
        innovations = r.values

    # -------------------------------
    # BOOTSTRAP SAMPLE RETURNS
    # -------------------------------
    rng = np.random.default_rng()
    sampled = rng.choice(innovations, size=(horizon, n_sim), replace=True)

    # -------------------------------
    # PRICE SIMULATION
    # -------------------------------
    prices = np.zeros_like(sampled)
    if use_log_returns:
        prices[0] = last_price * np.exp(sampled[0])
    else:
        prices[0] = last_price * (1 + sampled[0])

    for t in range(1, horizon):
        if use_log_returns:
            prices[t] = prices[t-1] * np.exp(sampled[t])
        else:
            prices[t] = prices[t-1] * (1 + sampled[t])

    # -------------------------------
    # INDEX SETUP
    # -------------------------------
    future_idx = pd.date_range(start=last_idx, periods=horizon+1, freq="D")

    sims_price = pd.DataFrame(prices, index=future_idx[1:])
    sims_price.loc[future_idx[0]] = last_price
    sims_price = sims_price.sort_index()

    # -------------------------------
    # CUMULATIVE RETURNS SIMULATION
    # -------------------------------
    if use_log_returns:
        sims_cum = np.exp(np.log(sims_price / last_price)) - 1
    else:
        sims_cum = (sims_price / last_price) - 1

    sims_cum = pd.DataFrame(sims_cum)

    # -------------------------------
    # HISTORY WINDOW
    # -------------------------------
    if len(series) > horizon:
        history = series.iloc[-horizon:]
    else:
        history = series

    # historical cumulative return
    if use_log_returns:
        hr = np.log(history / history.shift(1)).dropna()
        hist_cum = np.exp(hr.cumsum()) - 1
    else:
        hr = history.pct_change().dropna()
        hist_cum = (1 + hr).cumprod() - 1

    # -------------------------------
    # PLOT
    # -------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    # -------------------------------
    # PRICE SIMULATION PLOT (TOP)
    # -------------------------------
    ax1.plot(history.index, history.values, lw=2.3, color=hist_color, label="History")

    show = min(n_paths_to_show, n_sim)
    ax1.plot(sims_price.index, sims_price.iloc[:, :show], alpha=0.18, color=sim_color)

    p5 = sims_price.quantile(0.05, axis=1)
    p95 = sims_price.quantile(0.95, axis=1)
    median = sims_price.median(axis=1)

    ax1.fill_between(sims_price.index, p5, p95, color=interval_color, alpha=0.30)
    ax1.plot(median.index, median.values, lw=2.2, color=median_color, label="Median")

    ax1.axvline(last_idx, lw=1.3, color=hist_color, alpha=0.7)
    ax1.set_title("Monte Carlo Simulation – Prices ({})".format("Log Returns" if use_log_returns else "Simple Returns"))
    ax1.legend()

    # -------------------------------
    # CUMULATIVE RETURN PLOT (BOTTOM)
    # -------------------------------
    ax2.plot(hist_cum.index, hist_cum.values, lw=2.0, color=hist_color, label="Historical CumReturn")

    ax2.plot(sims_cum.index, sims_cum.iloc[:, :show], alpha=0.20, color=sim_color)

    cum_p5 = sims_cum.quantile(0.05, axis=1)
    cum_p95 = sims_cum.quantile(0.95, axis=1)
    cum_median = sims_cum.median(axis=1)

    ax2.fill_between(sims_cum.index, cum_p5, cum_p95, color=interval_color, alpha=0.30)
    ax2.plot(cum_median.index, cum_median.values, lw=2.0, color=median_color, label="Median CumReturn")

    ax2.axvline(last_idx, lw=1.3, color=hist_color, alpha=0.7)
    ax2.set_title("Monte Carlo Simulation – Cumulative Returns")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return sims_price, sims_cum
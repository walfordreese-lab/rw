# -*- coding: utf-8 -*-
"""
First Red Day -- Feature Study
================================
For every raw FRD signal in the universe (no candle filter, no stop, no squeeze),
extract 6 signal-day features and measure the next-day short return.

Goal: understand WHICH characteristics of the FRD setup predict next-day
follow-through, so future filter improvements are data-driven.

Features measured on the FRD bar (index i):
  1. gain_3d          -- 3-day rolling gain leading into the FRD
  2. streak           -- consecutive green closes immediately before FRD
  3. frd_decline      -- FRD day's close-to-close decline vs prior close
  4. gap_open_pct     -- FRD open vs prior close (negative = gap down)
  5. frd_close_pct    -- close position within FRD day's high-low range (0=low, 1=high)
  6. vol_ratio        -- FRD volume vs average of prior 3 days

Target (short direction, positive = profit):
  next_ret = (open[t] - close[t]) / open[t]   where t = trade day = FRD + 1

Output:
  console  -- full correlation table ranked by |r|
  PNG      -- 2x2 panel: correlation bar chart + scatter plots for top 3 features
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Configuration ─────────────────────────────────────────────────────────────
TICKERS = [
    # Original list
    "BBIG", "MULN", "FFIE", "NKLA", "ATER", "CLOV", "EXPR", "PROG",
    "SNDL", "TELL", "MMAT", "ILUS", "MOXC", "HCDI", "GOVX", "VINC",
    "SHOT", "MRIN", "DPRO", "GFAI", "BFRI", "AGRI", "WILL", "AREB",
    "NISN", "RELI", "BKSY", "VMAR", "AEYE", "MNMD",
    # Expanded universe
    "AMC", "GME", "BBBY", "SPRT", "IRNT", "OPAD", "BKKT", "PHUN",
    "ANY", "MEGL", "WORX", "JUPW", "CIFS", "PROP", "ESSC", "GREE",
    "AULT", "HTCR", "PNTM", "MARK", "BGFV", "REED", "CRTX", "VERB",
    "ZKIN", "TANH", "UONE", "DARE", "CUEN", "EZFL", "MFON", "FPAY",
    "GTII", "GORO", "VVPR", "INPX",
]
START_DATE    = "2022-01-01"
END_DATE      = "2024-12-31"
RELAXED_MODE  = True

if RELAXED_MODE:
    MIN_PRICE     = 1.0
    MAX_PRICE     = 15.0
    MIN_PUMP_GAIN = 0.40
    MIN_AVG_VOL   = 500_000
else:
    MIN_PRICE     = 3.0
    MAX_PRICE     = 10.0
    MIN_PUMP_GAIN = 0.75
    MIN_AVG_VOL   = 1_000_000

ROLL_WIN   = 3
MIN_GREENS = 2

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Human-readable labels for each feature (used in charts and tables)
FEATURE_LABELS = {
    "gain_3d":       "3-Day Prior Gain",
    "streak":        "Consec. Green Days",
    "frd_decline":   "FRD Day Decline",
    "gap_open_pct":  "Gap at Open (vs prior close)",
    "frd_close_pct": "Close Position in Range",
    "vol_ratio":     "Volume Ratio (vs 3d avg)",
}
FEATURES = list(FEATURE_LABELS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 -- DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
def fetch_data(tickers: list, start: str, end: str) -> dict:
    """Download adjusted daily OHLCV for each ticker via yfinance."""
    data = {}
    min_bars = ROLL_WIN + MIN_GREENS + 2
    for ticker in tickers:
        print(f"  Fetching {ticker} ...", end=" ", flush=True)
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < min_bars:
                print("(skipped)")
                continue
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            data[ticker] = df
            print(f"OK  ({len(df)} bars)")
        except Exception as exc:
            print(f"ERROR: {exc}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 -- FEATURE EXTRACTION
# Pull all FRD signals (no candle filter) and compute the 6 features
# plus the next-day short return target.
# ─────────────────────────────────────────────────────────────────────────────
def _build_streak(is_green: pd.Series) -> pd.Series:
    """Consecutive-green-close streak ending at each bar."""
    counts, c = [], 0
    for flag in is_green:
        c = c + 1 if flag else 0
        counts.append(c)
    return pd.Series(counts, index=is_green.index)


def extract_features(data: dict) -> pd.DataFrame:
    """
    Iterate over every qualifying FRD signal and compute all 6 features
    plus the short-direction next-day return.

    No candle filter, no risk rules -- we want the full raw population
    so correlations reflect the signal itself, not survivorship of filters.

    Feature definitions
    -------------------
    gain_3d       : close[i] / close[i-3] - 1
                    How large was the preceding pump?

    streak        : streak[i-1]
                    How many green days ran into the FRD?

    frd_decline   : (prior_close - close[i]) / prior_close
                    How hard did the stock fall on the FRD (close-to-close)?

    gap_open_pct  : (open[i] - prior_close) / prior_close
                    Did the FRD open below (negative) or above (positive)
                    the prior close? A gap-down open suggests sellers are
                    already in control before the regular session.

    frd_close_pct : (close[i] - low[i]) / (high[i] - low[i])
                    0 = closed at the day's low (maximum intraday weakness),
                    1 = closed at the day's high (intraday strength despite
                    being nominally red).

    vol_ratio     : volume[i] / mean(volume[i-3 : i])
                    Is participation on the FRD elevated vs the prior run?
                    > 1 means distribution volume; < 1 means quiet fade.
    """
    records = []

    for ticker, df in data.items():
        close  = df["Close"].squeeze()
        open_  = df["Open"].squeeze()
        high_  = df["High"].squeeze()
        low_   = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        n   = len(df)
        idx = df.index

        daily_chg = close.pct_change()
        is_green  = daily_chg > 0
        streak    = _build_streak(is_green)

        roll_gain = close.pct_change(ROLL_WIN)
        roll_vol  = volume.rolling(ROLL_WIN).mean()  # used for universe filter

        for i in range(ROLL_WIN + MIN_GREENS, n - 1):
            frd_close = float(close.iloc[i])

            # ── Universe filters (same as backtest, no candle filter) ──────
            if not (MIN_PRICE <= frd_close <= MAX_PRICE):
                continue
            if float(roll_gain.iloc[i]) < MIN_PUMP_GAIN:
                continue
            if float(roll_vol.iloc[i]) < MIN_AVG_VOL:
                continue

            # ── FRD signal ────────────────────────────────────────────────
            if int(streak.iloc[i - 1]) < MIN_GREENS:
                continue
            if not (frd_close < float(close.iloc[i - 1])):
                continue

            frd_open    = float(open_.iloc[i])
            frd_high    = float(high_.iloc[i])
            frd_low     = float(low_.iloc[i])
            frd_vol     = float(volume.iloc[i])
            prior_close = float(close.iloc[i - 1])

            # ── Feature 1: 3-day prior gain ───────────────────────────────
            gain_3d = float(roll_gain.iloc[i])

            # ── Feature 2: streak ─────────────────────────────────────────
            streak_n = int(streak.iloc[i - 1])

            # ── Feature 3: FRD decline (close-to-close) ───────────────────
            frd_decline = (prior_close - frd_close) / prior_close \
                          if prior_close > 0 else 0.0

            # ── Feature 4: gap at open vs prior close ─────────────────────
            # Negative = gap down (bearish open), Positive = gap up
            gap_open_pct = (frd_open - prior_close) / prior_close \
                           if prior_close > 0 else 0.0

            # ── Feature 5: close position within day's range ─────────────
            frd_range = frd_high - frd_low
            frd_close_pct = (frd_close - frd_low) / frd_range \
                            if frd_range > 0 else 0.5

            # ── Feature 6: volume ratio vs prior 3-day average ───────────
            prior_vol_avg = float(volume.iloc[max(0, i - ROLL_WIN): i].mean())
            vol_ratio = frd_vol / prior_vol_avg if prior_vol_avg > 0 else 1.0

            # ── Target: next-day short return (open -> close, short) ──────
            t     = i + 1
            entry = float(open_.iloc[t])
            exit_ = float(close.iloc[t])
            if entry <= 0:
                continue
            next_ret = (entry - exit_) / entry   # positive = stock fell = profit

            records.append({
                "Ticker":        ticker,
                "Signal_Date":   idx[i],
                "Trade_Date":    idx[t],
                # raw values for reference
                "Entry_Open":    round(entry, 4),
                "Exit_Close":    round(exit_, 4),
                # features
                "gain_3d":       round(gain_3d,       4),
                "streak":        streak_n,
                "frd_decline":   round(frd_decline,   4),
                "gap_open_pct":  round(gap_open_pct,  4),
                "frd_close_pct": round(frd_close_pct, 4),
                "vol_ratio":     round(vol_ratio,      3),
                # target
                "next_ret":      round(next_ret,       4),
            })

    return pd.DataFrame(records).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 -- CORRELATION ANALYSIS
# Rank every feature by Pearson r with next-day short return.
# ─────────────────────────────────────────────────────────────────────────────
def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation between each feature and next_ret.
    Returns a DataFrame sorted by |r| descending.

    Pearson r interpretation:
      +1 = perfect positive linear relationship
      -1 = perfect negative linear relationship
       0 = no linear relationship

    For short follow-through:
      Positive r means higher feature value --> larger next-day gain for the short.
      Negative r means higher feature value --> smaller (or negative) gain for short.
    """
    rows = []
    for feat in FEATURES:
        pair = df[[feat, "next_ret"]].dropna()
        n    = len(pair)
        if n < 3:
            r = float("nan")
        else:
            r = float(pair[feat].corr(pair["next_ret"]))

        # Directional interpretation
        if np.isnan(r):
            direction = "n/a"
        elif r > 0.1:
            direction = "(+) higher -> more follow-through"
        elif r < -0.1:
            direction = "(-) lower  -> more follow-through"
        else:
            direction = "  ~ weak / no clear direction"

        rows.append({
            "Feature":     feat,
            "Label":       FEATURE_LABELS[feat],
            "Pearson_r":   round(r, 4),
            "Abs_r":       round(abs(r), 4),
            "N":           n,
            "Direction":   direction,
        })

    result = pd.DataFrame(rows).sort_values("Abs_r", ascending=False)
    result.reset_index(drop=True, inplace=True)
    return result


def print_correlation_table(corr_df: pd.DataFrame) -> None:
    """Print the correlation table to stdout."""
    wide = "=" * 78
    print(f"\n{wide}")
    print("  FEATURE CORRELATION WITH NEXT-DAY SHORT RETURN  (ranked by |r|)")
    print(wide)
    print(f"  {'#':<3}  {'Feature':<17}  {'Label':<30}  {'r':>7}  {'|r|':>6}  N")
    print(f"  {'-'*3}  {'-'*17}  {'-'*30}  {'-'*7}  {'-'*6}  {'-'*4}")
    for _, row in corr_df.iterrows():
        print(f"  {int(_)+1:<3}  {row['Feature']:<17}  {row['Label']:<30}  "
              f"{row['Pearson_r']:>+7.4f}  {row['Abs_r']:>6.4f}  {row['N']}")
    print(wide)
    print()
    print("  Directional notes:")
    for _, row in corr_df.iterrows():
        print(f"    {row['Feature']:<17}  {row['Direction']}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 -- VISUALIZATION
# 2x2 panel: correlation bar chart (top-left) + scatter for top-3 features.
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_study(df: pd.DataFrame,
                       corr_df: pd.DataFrame,
                       output_dir: str) -> None:
    """
    Save a 2x2 figure:
      [top-left]    Horizontal bar chart -- all 6 features ranked by Pearson r
      [top-right]   Scatter: best-correlated feature vs next_ret
      [bottom-left] Scatter: 2nd feature vs next_ret
      [bottom-right]Scatter: 3rd feature vs next_ret

    Each scatter includes:
      - Coloured dots (green = profitable short, red = loss)
      - Ticker + date labels for each point
      - OLS regression line
      - r value in the title
    """
    top3 = corr_df["Feature"].tolist()[:3]
    n_trades = len(df)

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── Panel A: correlation bar chart ────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 0])

    features_ordered = corr_df["Feature"].tolist()
    r_values         = corr_df["Pearson_r"].tolist()
    labels_short     = [FEATURE_LABELS[f] for f in features_ordered]
    bar_colors       = ["#2980b9" if r >= 0 else "#c0392b" for r in r_values]

    bars = ax_bar.barh(range(len(features_ordered)), r_values,
                       color=bar_colors, alpha=0.85, height=0.6)
    ax_bar.set_yticks(range(len(features_ordered)))
    ax_bar.set_yticklabels(labels_short, fontsize=9)
    ax_bar.axvline(0, color="black", linewidth=0.8)
    ax_bar.set_xlabel("Pearson r  with next-day short return", fontsize=9)
    ax_bar.set_title(f"All Features -- Correlation Ranking\n(n={n_trades} trades)",
                     fontsize=10, fontweight="bold")
    ax_bar.grid(True, axis="x", alpha=0.3)

    # Annotate bars with r value
    for bar, r_val in zip(bars, r_values):
        x_pos = r_val + (0.005 if r_val >= 0 else -0.005)
        ha    = "left" if r_val >= 0 else "right"
        ax_bar.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{r_val:+.3f}", va="center", ha=ha, fontsize=8)

    # ── Panels B, C, D: scatter plots for top-3 features ─────────────────
    scatter_positions = [(0, 1), (1, 0), (1, 1)]

    for panel_idx, feat in enumerate(top3):
        row_pos, col_pos = scatter_positions[panel_idx]
        ax = fig.add_subplot(gs[row_pos, col_pos])

        sub    = df[["Ticker", "Trade_Date", feat, "next_ret"]].dropna()
        x      = sub[feat].values.astype(float)
        y      = sub["next_ret"].values.astype(float) * 100   # convert to %
        tickers = sub["Ticker"].values
        dates   = sub["Trade_Date"].values

        r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")

        # Scatter dots coloured by outcome
        dot_colors = ["#27ae60" if v > 0 else "#e74c3c" for v in y]
        ax.scatter(x, y, c=dot_colors, s=80, alpha=0.85,
                   edgecolors="white", linewidth=0.5, zorder=3)

        # Ticker + date labels on each point (small, offset)
        for xi, yi, tkr, dt in zip(x, y, tickers, dates):
            date_str = pd.Timestamp(dt).strftime("%m/%d/%y")
            ax.annotate(
                f"{tkr}\n{date_str}",
                xy=(xi, yi),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6.5,
                color="#333333",
                zorder=4,
            )

        # OLS regression line
        if len(x) > 2 and not np.isnan(r):
            m, b   = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, m * x_line + b,
                    color="navy", linewidth=1.5, linestyle="--",
                    alpha=0.8, zorder=2, label=f"OLS  (slope={m:+.2f})")
            ax.legend(fontsize=7, loc="best")

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_xlabel(FEATURE_LABELS[feat], fontsize=9)
        ax.set_ylabel("Next-Day Short Return (%)", fontsize=9)
        ax.set_title(
            f"#{panel_idx + 1}: {FEATURE_LABELS[feat]}\nr = {r:+.4f}",
            fontsize=10, fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"FRD Feature Study  --  {n_trades} signals, 2022-2024\n"
        f"Universe: {MIN_PRICE}$-${MAX_PRICE} | "
        f"{ROLL_WIN}d gain >{MIN_PUMP_GAIN:.0%} | vol >{MIN_AVG_VOL:,}",
        fontsize=12, y=1.01,
    )

    out_path = os.path.join(output_dir, "feature_study.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature study chart saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("+" + "=" * 52 + "+")
    print("|       First Red Day -- Feature Study             |")
    print("+" + "=" * 52 + "+")
    print(f"  Universe : {len(TICKERS)} tickers | {START_DATE} -> {END_DATE}")
    print(f"  Filters  : ${MIN_PRICE}-${MAX_PRICE} | "
          f"{ROLL_WIN}d gain >{MIN_PUMP_GAIN:.0%} | vol >{MIN_AVG_VOL:,}")
    print("  No candle filter -- capturing full raw signal population\n")

    print(">> [1/4] Fetching OHLCV data ...")
    data = fetch_data(TICKERS, START_DATE, END_DATE)
    print(f"\n  Loaded {len(data)}/{len(TICKERS)} tickers.\n")

    print(">> [2/4] Extracting features from FRD signals ...")
    df = extract_features(data)
    print(f"  Extracted {len(df)} signals with features.\n")

    if df.empty:
        print("No signals found. Try adjusting universe filters.")
        return

    # Print the raw signal table
    print("  Signal inventory:")
    print(f"  {'#':<3}  {'Ticker':<6}  {'Signal':>10}  {'Trade':>10}  "
          f"{'gain3d':>7}  {'streak':>6}  {'frd_dec':>7}  "
          f"{'gap':>7}  {'cls_pct':>7}  {'vol_r':>6}  {'next_ret':>9}")
    print("  " + "-" * 98)
    for i, row in df.iterrows():
        sig_d = pd.Timestamp(row["Signal_Date"]).strftime("%Y-%m-%d")
        trd_d = pd.Timestamp(row["Trade_Date"]).strftime("%Y-%m-%d")
        outcome = "WIN " if row["next_ret"] > 0 else "LOSS"
        print(f"  {i:<3}  {row['Ticker']:<6}  {sig_d}  {trd_d}  "
              f"{row['gain_3d']:>+6.1%}  {row['streak']:>6}  "
              f"{row['frd_decline']:>+6.1%}  {row['gap_open_pct']:>+6.1%}  "
              f"{row['frd_close_pct']:>7.3f}  {row['vol_ratio']:>6.2f}  "
              f"{row['next_ret']:>+7.1%}  {outcome}")
    print()

    print(">> [3/4] Computing feature correlations ...")
    corr_df = compute_correlations(df)
    print_correlation_table(corr_df)

    print(">> [4/4] Generating feature study chart ...")
    plot_feature_study(df, corr_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()

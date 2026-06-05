# -*- coding: utf-8 -*-
"""
First Red Day (FRD) Short Strategy Backtest
============================================
Universe : Low-float pump stocks (user-supplied list)
Period   : 2022-01-01 -> 2024-12-31
Direction: Short -- enter at next-day open, exit at same-day close
Profit % = (Entry_Open - Exit_Close) / Entry_Open

NOTE ON FILTER CALIBRATION
---------------------------
The strict parameters (75% 3-day gain + $3-$10 + 1M vol) are correct for
the target setup but yield very few signals on this specific ticker list
because most of these stocks had their peak pumps before 2022, are now
delisted, or trade mostly below $3.

Set RELAXED_MODE = True to use looser thresholds (40% gain / 500K vol) that
still capture the same FRD dynamic and produce a meaningful sample size.
Edit the CONFIG block to dial in whatever thresholds suit your research.
"""

import os
import sys
# Force UTF-8 output on Windows terminals that default to cp1252
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")      # non-interactive backend -- works without a display
import matplotlib.pyplot as plt

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
START_DATE = "2022-01-01"
END_DATE   = "2024-12-31"

# ── Filter thresholds ─────────────────────────────────────────────────────────
# Toggle RELAXED_MODE = True to widen filters and generate a larger sample.
# The strategy logic is identical in both modes -- only the entry criteria change.
RELAXED_MODE = True

if RELAXED_MODE:
    # Looser thresholds -- still requires a meaningful pump + liquidity
    MIN_PRICE     = 1.0          # allow sub-$3 micro-caps
    MAX_PRICE     = 15.0
    MIN_PUMP_GAIN = 0.40         # 40% 3-day gain
    MIN_AVG_VOL   = 500_000      # 500k avg volume
else:
    # Original strict thresholds as specified
    MIN_PRICE     = 3.0
    MAX_PRICE     = 10.0
    MIN_PUMP_GAIN = 0.75         # 75% 3-day gain
    MIN_AVG_VOL   = 1_000_000    # 1M avg volume

ROLL_WIN    = 3   # rolling window in days for gain/volume measurement
MIN_GREENS  = 2   # consecutive green closes required immediately before FRD

# ── Risk management rules ─────────────────────────────────────────────────────
STOP_LOSS_PCT    = 0.12  # exit short if price rises 12% above entry on trade day
SQUEEZE_LOOKBACK = 30    # skip ticker for this many calendar days after a loss

# ── Candle quality filter ─────────────────────────────────────────────────────
# FRD candle must close in the LOWER 33% of its own high-low range.
FRD_CLOSE_RANGE_MAX = 0.33

# ── Feature-study-motivated filters (applied in S4 comparison run) ────────────
# Derived from frd_feature_study.py: gain_3d is the strongest predictor (r=+0.52),
# vol_ratio has negative correlation (lower = better), and streak > 4 hurts.
FEAT_MIN_GAIN    = 1.00   # require >= 100% 3-day prior gain (was 40%)
FEAT_MAX_STREAK  = 4      # cap consecutive green days at 4 (longer = harder squeeze)
FEAT_VOL_FADE    = 0.80   # FRD volume must be < 80% of prior 3-day avg (quiet distribution)

# All file output (PNG) lands beside this script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 -- DATA FETCHING
# Download adjusted daily OHLCV for the full date range via yfinance.
# Each ticker is fetched individually to avoid MultiIndex column issues.
# ─────────────────────────────────────────────────────────────────────────────
def fetch_data(tickers: list, start: str, end: str) -> dict:
    """
    Download adjusted daily OHLCV for each ticker.

    Returns
    -------
    dict  {ticker_str -> pd.DataFrame}
        Tickers with no data or fewer than (ROLL_WIN + MIN_GREENS + 2) bars
        are skipped so downstream code never receives an empty frame.
    """
    data = {}
    min_bars = ROLL_WIN + MIN_GREENS + 2

    for ticker in tickers:
        print(f"  Fetching {ticker} ...", end=" ", flush=True)
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,   # prices adjusted for splits/dividends
            )

            # Newer yfinance (>=0.2.x) may return MultiIndex columns even for a
            # single ticker -- flatten to plain column names.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < min_bars:
                print("(skipped -- insufficient data)")
                continue

            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            data[ticker] = df
            print(f"OK  ({len(df)} bars)")

        except Exception as exc:
            print(f"ERROR: {exc}")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 -- SIGNAL DETECTION
# Identify every First Red Day (FRD) setup that passes all universe filters.
# ─────────────────────────────────────────────────────────────────────────────
def _build_streak(is_green: pd.Series) -> pd.Series:
    """
    Return the length of the consecutive-green-close streak ending at each bar.
    Resets to 0 on any flat or red close.
    """
    counts, current = [], 0
    for flag in is_green:
        current = current + 1 if flag else 0
        counts.append(current)
    return pd.Series(counts, index=is_green.index)


def detect_signals(data: dict,
                   use_candle_filter: bool = False,
                   gain_override: float = None,
                   max_streak: int = None,
                   vol_fade_max: float = None,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Scan every ticker for FRD setups that pass the universe filters.

    Core universe filters (always applied):
      1. Close price in [MIN_PRICE, MAX_PRICE]
      2. 3-day rolling gain >= gain_override (or MIN_PUMP_GAIN if None)
      3. Mean daily volume over ROLL_WIN days >= MIN_AVG_VOL

    Signal conditions (always applied):
      - streak at bar i-1 >= MIN_GREENS
      - close[i] < close[i-1]  (first red close)

    Optional filters (pass the relevant argument to enable):
      use_candle_filter : FRD close must be in lower FRD_CLOSE_RANGE_MAX of range
      gain_override     : override the minimum 3-day gain threshold
      max_streak        : discard signals where streak > max_streak
                          (feature study showed longer streaks hurt short follow-through)
      vol_fade_max      : FRD volume / prior-3d-avg-volume must be <= vol_fade_max
                          (feature study: lower relative FRD volume = better follow-through)

    Trade execution:
      - Enter at bar i+1 open, exit at bar i+1 close (no overnight hold)
    """
    records = []

    # Resolve effective gain threshold
    eff_min_gain = gain_override if gain_override is not None else MIN_PUMP_GAIN

    if verbose:
        extras = ""
        if max_streak  is not None: extras += f"  maxStrk"
        if vol_fade_max is not None: extras += f"  VolFade"
        if use_candle_filter:        extras += f"  Cndl"
        print(f"\n  {'Ticker':<6}  {'Price':>7}  {'Gain':>6}  {'Vol':>6}  "
              f"{'All3':>5}  {'FRD':>5}{extras}  note")
        print("  " + "-" * (65 + len(extras) * 2))

    for ticker, df in data.items():
        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze()
        open_  = df["Open"].squeeze()
        high_  = df["High"].squeeze()
        low_   = df["Low"].squeeze()

        n   = len(df)
        idx = df.index

        daily_chg = close.pct_change()
        is_green  = daily_chg > 0
        streak    = _build_streak(is_green)

        roll_gain = close.pct_change(ROLL_WIN)
        avg_vol   = volume.rolling(ROLL_WIN).mean()

        n_price = n_gain = n_vol = n_all3 = n_frd = n_extra = 0

        for i in range(ROLL_WIN + MIN_GREENS, n - 1):
            frd_close = float(close.iloc[i])

            # -- Filter 1: price range --
            price_ok = MIN_PRICE <= frd_close <= MAX_PRICE
            if price_ok: n_price += 1

            # -- Filter 2: 3-day pump gain (uses effective threshold) --
            gain_ok = float(roll_gain.iloc[i]) >= eff_min_gain
            if gain_ok: n_gain += 1

            # -- Filter 3: average volume --
            vol_ok = float(avg_vol.iloc[i]) >= MIN_AVG_VOL
            if vol_ok: n_vol += 1

            if not (price_ok and gain_ok and vol_ok):
                continue
            n_all3 += 1

            # -- Signal: prior streak + first red close --
            cur_streak = int(streak.iloc[i - 1])
            if cur_streak < MIN_GREENS:
                continue
            if not (frd_close < float(close.iloc[i - 1])):
                continue
            n_frd += 1

            # -- Optional: max streak cap --
            if max_streak is not None and cur_streak > max_streak:
                continue

            # -- Optional: volume fade filter --
            if vol_fade_max is not None:
                prior_avg = float(volume.iloc[max(0, i - ROLL_WIN): i].mean())
                frd_vol   = float(volume.iloc[i])
                if prior_avg > 0 and (frd_vol / prior_avg) > vol_fade_max:
                    continue

            # -- Optional: candle close-in-range filter --
            frd_high  = float(high_.iloc[i])
            frd_low   = float(low_.iloc[i])
            frd_range = frd_high - frd_low
            if use_candle_filter:
                if frd_range > 0:
                    if (frd_close - frd_low) / frd_range > FRD_CLOSE_RANGE_MAX:
                        continue
                else:
                    continue

            n_extra += 1
            t     = i + 1
            entry = float(open_.iloc[t])
            if entry <= 0:
                continue

            close_pct_val = round((frd_close - frd_low) / frd_range, 3) \
                            if frd_range > 0 else None

            records.append({
                "Ticker":         ticker,
                "Signal_Date":    idx[i],
                "Trade_Date":     idx[t],
                "Signal_Close":   round(frd_close, 4),
                "FRD_Close_Pct":  close_pct_val,
                "Entry_Open":     round(entry, 4),
                "Trade_High":     round(float(high_.iloc[t]), 4),
                "Exit_Close":     round(float(close.iloc[t]), 4),
                "Roll3_Gain":     round(float(roll_gain.iloc[i]), 4),
                "Avg_Vol":        int(avg_vol.iloc[i]),
                "Streak":         int(streak.iloc[i - 1]),
            })

        if verbose:
            note = ""
            if close.max() < MIN_PRICE:
                note = "all bars below min price"
            elif close.min() > MAX_PRICE:
                note = "all bars above max price"
            elif roll_gain.max() < eff_min_gain:
                note = f"max 3d gain = {roll_gain.max():.0%} < {eff_min_gain:.0%}"
            elif avg_vol.max() < MIN_AVG_VOL:
                note = f"max avg vol = {avg_vol.max()/1e6:.1f}M < {MIN_AVG_VOL/1e6:.1f}M"
            extra_str = f"  {n_extra:>4}" if (max_streak or vol_fade_max or use_candle_filter) else ""
            print(f"  {ticker:<6}  {n_price:>7}  {n_gain:>6}  {n_vol:>6}  "
                  f"{n_all3:>5}  {n_frd:>5}{extra_str}  {note}")

    if verbose:
        print()

    signals = pd.DataFrame(records)
    return signals.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 -- TRADE SIMULATION
# Applies two risk rules on top of the raw signals, in chronological order:
#
#   1. Hard stop loss (STOP_LOSS_PCT): if the trade-day high >= entry * (1 + stop),
#      the short is covered at the stop price instead of the close.
#      Worst-case loss is capped at -STOP_LOSS_PCT regardless of how far it runs.
#
#   2. Squeeze filter (SQUEEZE_LOOKBACK days): if a ticker produced a losing FRD
#      trade within the last SQUEEZE_LOOKBACK calendar days, skip the next signal
#      on that ticker. This avoids re-shorting into an ongoing squeeze.
#
# Both rules are applied in a single forward pass so the squeeze filter always
# reflects actual outcomes (including stop-adjusted P&L) from prior trades.
# ─────────────────────────────────────────────────────────────────────────────
def simulate_trades(signals: pd.DataFrame,
                    use_stop: bool = True,
                    use_squeeze: bool = True) -> pd.DataFrame:
    """
    Process signals chronologically, optionally applying stop loss and squeeze filter.

    use_stop    : cap the short loss at STOP_LOSS_PCT if intraday high hits the level
    use_squeeze : skip a ticker for SQUEEZE_LOOKBACK days after any losing trade

    Returns a DataFrame of executed trades only (squeeze-filtered signals are
    excluded).  Added columns vs raw signals:
      Actual_Exit  -- price at which position was closed
      Stop_Hit     -- True if the stop triggered before the close
      PnL_Pct      -- (Entry_Open - Actual_Exit) / Entry_Open  [+ = profit]
      PnL_Pts      -- Entry_Open - Actual_Exit  ($ per share)
      Is_Win       -- PnL_Pct > 0
    """
    if signals.empty:
        return signals.copy()

    # Process in signal-date order so the squeeze filter is causal
    ordered = signals.sort_values("Signal_Date").reset_index(drop=True)

    executed = []
    last_loss_date: dict = {}   # ticker -> Trade_Date of most recent loss

    for _, row in ordered.iterrows():
        tkr      = row["Ticker"]
        sig_date = row["Signal_Date"]

        # ── Squeeze filter ────────────────────────────────────────────────
        if use_squeeze and tkr in last_loss_date:
            days_since = (sig_date - last_loss_date[tkr]).days
            if days_since <= SQUEEZE_LOOKBACK:
                continue   # still in the squeeze window -- skip

        # ── Exit price (stop or close) ────────────────────────────────────
        entry   = row["Entry_Open"]
        stop_px = entry * (1.0 + STOP_LOSS_PCT)

        if use_stop and row["Trade_High"] >= stop_px:
            actual_exit = stop_px
            stop_hit    = True
        else:
            actual_exit = row["Exit_Close"]
            stop_hit    = False

        pnl_pct = (entry - actual_exit) / entry
        pnl_pts = entry - actual_exit
        is_win  = pnl_pct > 0

        if use_squeeze and not is_win:
            last_loss_date[tkr] = row["Trade_Date"]

        rec = row.to_dict()
        rec.update({
            "Actual_Exit": round(actual_exit, 4),
            "Stop_Hit":    stop_hit,
            "PnL_Pct":     pnl_pct,
            "PnL_Pts":     pnl_pts,
            "Is_Win":      is_win,
        })
        executed.append(rec)

    trades = pd.DataFrame(executed)
    trades.sort_values("Trade_Date", inplace=True)
    trades.reset_index(drop=True, inplace=True)
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 -- PERFORMANCE REPORTING
# Print summary table to stdout and save equity curve as equity_curve.png.
# ─────────────────────────────────────────────────────────────────────────────
def _metrics(trades: pd.DataFrame) -> dict:
    """Extract scalar performance metrics from a trades DataFrame."""
    if trades.empty:
        return dict(n=0, win_rate=0, avg_win=0, avg_loss=0,
                    expectancy=0, best=0, worst=0, n_stopped=0)
    wins   = trades[trades["Is_Win"]]
    losses = trades[~trades["Is_Win"]]
    n      = len(trades)
    wr     = len(wins) / n
    aw     = wins["PnL_Pct"].mean()   if not wins.empty   else 0.0
    al     = losses["PnL_Pct"].mean() if not losses.empty else 0.0
    return dict(
        n          = n,
        win_rate   = wr,
        avg_win    = aw,
        avg_loss   = al,
        expectancy = wr * aw + (1 - wr) * al,
        best       = trades["PnL_Pct"].max(),
        worst      = trades["PnL_Pct"].min(),
        n_stopped  = int(trades["Stop_Hit"].sum()) if "Stop_Hit" in trades.columns else 0,
    )


def print_comparison(runs: list) -> None:
    """
    Print a side-by-side comparison table for multiple backtest runs.

    Parameters
    ----------
    runs : list of (label_str, trades_DataFrame) tuples
    """
    metrics = [(label, _metrics(t)) for label, t in runs]

    col_w = 22
    sep   = "+" + ("-" * 20) + ("+" + "-" * col_w) * len(metrics) + "+"

    print("\n" + "=" * (20 + (col_w + 1) * len(metrics) + 1))
    print("  RUN COMPARISON")
    print("=" * (20 + (col_w + 1) * len(metrics) + 1))

    # Header row
    header = f"  {'Metric':<18}"
    for label, _ in metrics:
        header += f"  {label:^{col_w-2}}"
    print(header)
    print("-" * (20 + (col_w + 1) * len(metrics)))

    rows = [
        ("Trades executed",  lambda m: str(m["n"])),
        ("Stops triggered",  lambda m: str(m["n_stopped"])),
        ("Win rate",         lambda m: f"{m['win_rate']:.1%}"),
        ("Avg win",          lambda m: f"{m['avg_win']:+.2%}"),
        ("Avg loss",         lambda m: f"{m['avg_loss']:+.2%}"),
        ("Expectancy/trade", lambda m: f"{m['expectancy']:+.2%}"),
        ("Best trade",       lambda m: f"{m['best']:+.2%}"),
        ("Worst trade",      lambda m: f"{m['worst']:+.2%}"),
    ]

    for row_label, fmt in rows:
        line = f"  {row_label:<18}"
        for _, m in metrics:
            line += f"  {fmt(m):^{col_w-2}}"
        print(line)

    print("=" * (20 + (col_w + 1) * len(metrics) + 1))
    print()


def report_performance(trades: pd.DataFrame, output_dir: str) -> None:
    """
    Compute and print performance metrics, then save a two-panel equity chart.

    Metrics: win rate, avg win %, avg loss %, expectancy, total trades,
             best trade, worst trade.

    Equity curve: cumulative sum of PnL_Pct (equal-weight, one share per trade).
    """
    if trades.empty:
        print("\nNo qualifying trades found.")
        print("Hint: set RELAXED_MODE = True at the top of the script, or")
        print("      lower MIN_PUMP_GAIN / MIN_AVG_VOL / widen MIN/MAX_PRICE.")
        return

    wins   = trades[trades["Is_Win"]]
    losses = trades[~trades["Is_Win"]]
    n      = len(trades)

    win_rate   = len(wins) / n
    avg_win    = wins["PnL_Pct"].mean()   if not wins.empty   else 0.0
    avg_loss   = losses["PnL_Pct"].mean() if not losses.empty else 0.0
    # avg_loss is already negative, so this nets out naturally
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    best_row  = trades.loc[trades["PnL_Pct"].idxmax()]
    worst_row = trades.loc[trades["PnL_Pct"].idxmin()]

    mode_tag   = "RELAXED" if RELAXED_MODE else "STRICT"
    n_stopped  = int(trades["Stop_Hit"].sum()) if "Stop_Hit" in trades.columns else 0

    # -- Console summary -------------------------------------------------------
    wide = "=" * 56
    thin = "-" * 56
    print(f"\n{wide}")
    print(f"  FIRST RED DAY SHORT STRATEGY - BACKTEST RESULTS [{mode_tag}]")
    print(f"  Period     : {START_DATE}  ->  {END_DATE}")
    print(f"  Filters    : ${MIN_PRICE}-${MAX_PRICE} | "
          f"{ROLL_WIN}d gain >{MIN_PUMP_GAIN:.0%} | "
          f"vol >{MIN_AVG_VOL:,}")
    print(f"  Stop Loss  : {STOP_LOSS_PCT:.0%} hard stop on trade day")
    print(f"  Squeeze    : skip ticker for {SQUEEZE_LOOKBACK}d after any loss")
    print(wide)
    print(f"  Total Trades   : {n}  ({n_stopped} stopped out)")
    print(f"  Win Rate       : {win_rate:.1%}   ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg Win        : {avg_win:+.2%}")
    print(f"  Avg Loss       : {avg_loss:+.2%}")
    print(f"  Expectancy     : {expectancy:+.2%}  per trade")
    print(thin)
    print(f"  Best  Trade    : {best_row['PnL_Pct']:+.2%}"
          f"   ({best_row['Ticker']} | {best_row['Trade_Date'].date()})")
    print(f"  Worst Trade    : {worst_row['PnL_Pct']:+.2%}"
          f"   ({worst_row['Ticker']} | {worst_row['Trade_Date'].date()})")
    print(f"{wide}\n")

    # Per-ticker breakdown
    stop_col = "Stop_Hit" in trades.columns
    print("  Per-Ticker Breakdown:")
    hdr = (f"  {'Ticker':<7}  {'N':>4}  {'Win%':>6}  "
           f"{'AvgPnL':>8}  {'Best':>8}  {'Worst':>9}  {'Stops':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tkr, grp in trades.groupby("Ticker"):
        wr    = grp["Is_Win"].mean()
        avg   = grp["PnL_Pct"].mean()
        best  = grp["PnL_Pct"].max()
        wst   = grp["PnL_Pct"].min()
        stops = int(grp["Stop_Hit"].sum()) if stop_col else 0
        print(f"  {tkr:<7}  {len(grp):>4}  {wr:>5.1%}  {avg:>+7.2%}  "
              f"{best:>+7.2%}  {wst:>+8.2%}  {stops:>6}")
    print()

    # -- Build and save equity curve chart ------------------------------------
    trades = trades.copy()
    trades["Cum_PnL_Pct"] = trades["PnL_Pct"].cumsum() * 100   # convert to %

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(
        f"First Red Day Short Strategy - Equity Curve (2022-2024) [{mode_tag}]\n"
        f"Win Rate {win_rate:.1%}  |  Expectancy {expectancy:+.2%}/trade  |  "
        f"{n} trades  |  ${MIN_PRICE}-${MAX_PRICE}, "
        f"{ROLL_WIN}d gain >{MIN_PUMP_GAIN:.0%}, vol >{MIN_AVG_VOL:,}",
        fontsize=11,
    )

    # Top panel -- cumulative return
    cum = trades["Cum_PnL_Pct"]
    ax1.plot(trades.index, cum, color="steelblue", linewidth=1.5,
             label="Cumulative Return")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.fill_between(trades.index, cum, 0,
                     where=(cum >= 0), alpha=0.15, color="green",
                     label="Profit zone")
    ax1.fill_between(trades.index, cum, 0,
                     where=(cum <  0), alpha=0.15, color="red",
                     label="Loss zone")
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.set_xticks(trades.index)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom panel -- per-trade P&L bars
    bar_colors = ["#27ae60" if w else "#e74c3c" for w in trades["Is_Win"]]
    ax2.bar(trades.index, trades["PnL_Pct"] * 100,
            color=bar_colors, alpha=0.8, width=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Trade P&L (%)")
    ax2.set_xlabel("Trade # (chronological)")
    ax2.set_xticks(trades.index)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "equity_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Equity curve saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    mode_label = "RELAXED" if RELAXED_MODE else "STRICT"
    print("+" + "=" * 58 + "+")
    print("|       First Red Day Short Strategy Backtest          |")
    print("+" + "=" * 58 + "+")
    print(f"  Universe  : {len(TICKERS)} tickers  |  {START_DATE} -> {END_DATE}")
    print(f"  Mode      : {mode_label}")
    print(f"  Base filters : ${MIN_PRICE}-${MAX_PRICE} | "
          f"{ROLL_WIN}d gain >{MIN_PUMP_GAIN:.0%} | vol >{MIN_AVG_VOL:,}")
    print(f"  Feature flts : gain >{FEAT_MIN_GAIN:.0%} | "
          f"streak <=  {FEAT_MAX_STREAK} | vol_ratio <={FEAT_VOL_FADE:.0%}\n")

    # Step 1 -- fetch data once; both scenarios share it
    print(">> [1/4] Fetching OHLCV data ...")
    data = fetch_data(TICKERS, START_DATE, END_DATE)
    print(f"\n  Loaded {len(data)}/{len(TICKERS)} tickers.\n")

    # Step 2 -- detect signals for both filter regimes (silently for S1)
    print(">> [2/4] Scanning for First Red Day signals ...")

    # S1 baseline: original relaxed filters, no extra signal rules
    signals_s1 = detect_signals(data, verbose=False)

    # S4 feature-based: 100% gain threshold + streak cap + vol fade
    print(f"  Feature-filter funnel  "
          f"(gain >{FEAT_MIN_GAIN:.0%} | streak<={FEAT_MAX_STREAK} | vol<={FEAT_VOL_FADE:.0%})")
    signals_s4 = detect_signals(
        data,
        gain_override  = FEAT_MIN_GAIN,
        max_streak     = FEAT_MAX_STREAK,
        vol_fade_max   = FEAT_VOL_FADE,
        verbose        = True,
    )

    print(f"  S1 baseline signals    : {len(signals_s1)}")
    print(f"  S4 feature-filter sigs : {len(signals_s4)}\n")

    # Step 3 -- simulate
    print(">> [3/4] Simulating trades ...")

    # S1: no risk rules -- pure baseline to show the raw edge (or lack of it)
    t1 = simulate_trades(signals_s1, use_stop=False, use_squeeze=False)

    # S4: feature filters + stop + squeeze
    t4 = simulate_trades(signals_s4, use_stop=True, use_squeeze=True)
    sq4 = len(signals_s4) - len(t4)

    print(f"  S1 (original baseline) : {len(t1)} trades executed")
    print(f"  S4 (feature filters)   : {len(t4)} trades executed  "
          f"({sq4} skipped by squeeze)\n")

    # Step 4 -- comparison then full detail report for S4
    print(">> [4/4] Generating performance report ...")
    print_comparison([
        ("S1: Baseline",        t1),
        ("S4: Feature Filters", t4),
    ])
    report_performance(t4, OUTPUT_DIR)


if __name__ == "__main__":
    main()

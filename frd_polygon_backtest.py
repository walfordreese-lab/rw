# -*- coding: utf-8 -*-
"""
frd_polygon_backtest.py
=======================
First Red Day (FRD) Short Strategy — Polygon.io Universe

Discovers qualifying tickers from Polygon (including delisted names), then
runs the FRD feature-filter strategy on two independent windows and reports
both results side by side.

  Training set (in-sample)  : 2022-01-01 -> 2024-12-31
  Test set (out-of-sample)  : 2025-01-01 -> 2025-12-31

Universe criteria (applied during universe scan):
  - Price between $3 and $10 at time of pump
  - 3-day rolling gain >= 100%
  - 3-day average volume >= 1,000,000 shares/day

FRD signal filters (same in both periods):
  - At least 2 consecutive green closes immediately before the FRD candle
  - Streak <= 4 consecutive green days (longer streaks = harder squeeze)
  - FRD-day volume < 80% of the prior 3-day average (quiet distribution)

Risk rules (reset between periods -- no leakage from train to test):
  - 12% hard stop loss: if intraday high >= entry * 1.12, exit at stop
  - 30-day squeeze filter: skip a ticker for 30 calendar days after a loss

Execution model: enter at next open, exit at same-day close (no overnight hold).
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from polygon_fetcher import scan_and_fetch

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_START = "2022-01-01"
TRAIN_END   = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = "2025-12-31"

PRICE_MIN   = 3.0
PRICE_MAX   = 10.0
GAIN_3D_MIN = 1.00    # 100% minimum 3-day gain
VOL_MIN     = 1_000_000

ROLL_WIN        = 3
MIN_GREENS      = 2   # minimum consecutive green days before FRD candle
MAX_STREAK      = 4   # cap: more than this many greens = skip
VOL_FADE_MAX    = 0.80  # FRD volume must be < 80% of prior 3-day avg

STOP_LOSS_PCT    = 0.12   # 12% hard stop
SQUEEZE_LOOKBACK = 30     # calendar days to skip a ticker after a loss

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Signal detection ───────────────────────────────────────────────────────────

def _build_streak(is_green: pd.Series) -> pd.Series:
    """Consecutive green-close streak length ending at each bar (resets on red)."""
    counts, cur = [], 0
    for flag in is_green:
        cur = cur + 1 if flag else 0
        counts.append(cur)
    return pd.Series(counts, index=is_green.index)


def detect_signals(data: dict, start: str, end: str) -> pd.DataFrame:
    """
    Scan qualifying tickers for FRD setups within [start, end].

    For each bar i (signal date) inside the window:
      Universe filters:  price in [PRICE_MIN, PRICE_MAX]
                         3-day rolling gain >= GAIN_3D_MIN
                         3-day avg volume >= VOL_MIN
      Signal conditions: prior streak >= MIN_GREENS
                         close[i] < close[i-1]  (first red close)
      Feature filters:   streak <= MAX_STREAK
                         volume[i] / mean(volume[i-3:i]) <= VOL_FADE_MAX

    Trade execution: enter at open[i+1], exit at close[i+1].
    """
    ts_start = pd.Timestamp(start)
    ts_end   = pd.Timestamp(end)
    records  = []

    for ticker, df_full in data.items():
        # Include a generous lead-in so rolling windows are valid at ts_start
        df = df_full[
            (df_full.index >= ts_start - pd.Timedelta(days=30))
            & (df_full.index <= ts_end + pd.Timedelta(days=5))
        ].copy()

        n = len(df)
        if n < ROLL_WIN + MIN_GREENS + 2:
            continue

        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze()
        open_  = df["Open"].squeeze()
        high_  = df["High"].squeeze()
        low_   = df["Low"].squeeze()
        idx    = df.index

        daily_chg = close.pct_change()
        is_green  = daily_chg > 0
        streak    = _build_streak(is_green)
        roll_gain = close.pct_change(ROLL_WIN)
        avg_vol   = volume.rolling(ROLL_WIN).mean()

        for i in range(ROLL_WIN + MIN_GREENS, n - 1):
            signal_date = idx[i]
            # Only emit signals inside the requested window
            if not (ts_start <= signal_date <= ts_end):
                continue

            frd_close = float(close.iloc[i])

            # Universe filters
            if not (PRICE_MIN <= frd_close <= PRICE_MAX):
                continue
            if float(roll_gain.iloc[i]) < GAIN_3D_MIN:
                continue
            if float(avg_vol.iloc[i]) < VOL_MIN:
                continue

            # Signal: prior green streak + first red close
            cur_streak = int(streak.iloc[i - 1])
            if cur_streak < MIN_GREENS:
                continue
            if frd_close >= float(close.iloc[i - 1]):
                continue  # not a red close

            # Feature filters
            if cur_streak > MAX_STREAK:
                continue
            prior_3d_vol = float(volume.iloc[max(0, i - ROLL_WIN):i].mean())
            frd_vol      = float(volume.iloc[i])
            if prior_3d_vol > 0 and (frd_vol / prior_3d_vol) > VOL_FADE_MAX:
                continue

            # Trade bar
            t     = i + 1
            entry = float(open_.iloc[t])
            if entry <= 0:
                continue

            frd_high  = float(high_.iloc[i])
            frd_low   = float(low_.iloc[i])
            frd_range = frd_high - frd_low
            close_pct = (
                round((frd_close - frd_low) / frd_range, 3) if frd_range > 0 else None
            )

            records.append({
                "Ticker":        ticker,
                "Signal_Date":   signal_date,
                "Trade_Date":    idx[t],
                "Signal_Close":  round(frd_close, 4),
                "FRD_Close_Pct": close_pct,
                "Entry_Open":    round(entry, 4),
                "Trade_High":    round(float(high_.iloc[t]), 4),
                "Exit_Close":    round(float(close.iloc[t]), 4),
                "Roll3_Gain":    round(float(roll_gain.iloc[i]), 4),
                "Avg_Vol":       int(avg_vol.iloc[i]),
                "Streak":        cur_streak,
                "FRD_Vol_Ratio": round(frd_vol / prior_3d_vol, 3) if prior_3d_vol > 0 else None,
            })

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).reset_index(drop=True)


# ── Trade simulation ───────────────────────────────────────────────────────────

def simulate_trades(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Execute signals chronologically with stop-loss and squeeze filter.
    The squeeze filter state is local to this call (no cross-period leakage).

    Added columns vs raw signals:
        Actual_Exit  -- price the position was closed at
        Stop_Hit     -- True if intraday high triggered the hard stop
        PnL_Pct      -- (Entry_Open - Actual_Exit) / Entry_Open  (+  = profit for short)
        PnL_Pts      -- Entry_Open - Actual_Exit  ($ per share)
        Is_Win       -- PnL_Pct > 0
    """
    if signals.empty:
        return pd.DataFrame()

    ordered          = signals.sort_values("Signal_Date").reset_index(drop=True)
    executed         = []
    last_loss_date: dict = {}

    for _, row in ordered.iterrows():
        tkr      = row["Ticker"]
        sig_date = row["Signal_Date"]

        # Squeeze filter
        if tkr in last_loss_date:
            if (sig_date - last_loss_date[tkr]).days <= SQUEEZE_LOOKBACK:
                continue

        entry   = row["Entry_Open"]
        stop_px = entry * (1.0 + STOP_LOSS_PCT)

        if row["Trade_High"] >= stop_px:
            actual_exit = stop_px
            stop_hit    = True
        else:
            actual_exit = row["Exit_Close"]
            stop_hit    = False

        pnl_pct = (entry - actual_exit) / entry
        is_win  = pnl_pct > 0

        if not is_win:
            last_loss_date[tkr] = row["Trade_Date"]

        rec = row.to_dict()
        rec.update({
            "Actual_Exit": round(actual_exit, 4),
            "Stop_Hit":    stop_hit,
            "PnL_Pct":     pnl_pct,
            "PnL_Pts":     entry - actual_exit,
            "Is_Win":      is_win,
        })
        executed.append(rec)

    if not executed:
        return pd.DataFrame()
    trades = pd.DataFrame(executed)
    trades.sort_values("Trade_Date", inplace=True)
    return trades.reset_index(drop=True)


# ── Metrics & reporting ────────────────────────────────────────────────────────

def _metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return dict(n=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                    expectancy=0.0, best=0.0, worst=0.0, n_stopped=0)
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
        n_stopped  = int(trades["Stop_Hit"].sum()),
    )


def print_side_by_side(
    train_trades: pd.DataFrame,
    test_trades: pd.DataFrame,
    n_sig_train: int,
    n_sig_test: int,
) -> None:
    m_tr = _metrics(train_trades)
    m_te = _metrics(test_trades)

    col_w  = 26
    total_w = 22 + col_w * 2 + 4

    tr_lbl = f"TRAIN 2022-2024  (n={m_tr['n']})"
    te_lbl = f"TEST  2025 OOS   (n={m_te['n']})"

    print("\n" + "=" * total_w)
    print("  FRD SHORT STRATEGY — POLYGON UNIVERSE")
    print("  Filters: $3-$10 | 3d gain >=100% | avg vol >=1M")
    print("           streak <=4 | FRD vol <=80% of run avg | 12% stop | 30d squeeze")
    print("=" * total_w)

    header = f"  {'Metric':<20}  {tr_lbl:^{col_w}}  {te_lbl:^{col_w}}"
    print(header)
    print("-" * total_w)

    rows = [
        ("Signals found",    lambda m, ns: str(ns)),
        ("Squeeze-skipped",  lambda m, ns: str(ns - m["n"])),
        ("Trades executed",  lambda m, ns: str(m["n"])),
        ("Stops triggered",  lambda m, ns: str(m["n_stopped"])),
        ("Win rate",         lambda m, ns: f"{m['win_rate']:.1%}"),
        ("Avg win",          lambda m, ns: f"{m['avg_win']:+.2%}"),
        ("Avg loss",         lambda m, ns: f"{m['avg_loss']:+.2%}"),
        ("Expectancy/trade", lambda m, ns: f"{m['expectancy']:+.2%}"),
        ("Best trade",       lambda m, ns: f"{m['best']:+.2%}"),
        ("Worst trade",      lambda m, ns: f"{m['worst']:+.2%}"),
    ]

    for row_lbl, fmt in rows:
        val_tr = fmt(m_tr, n_sig_train)
        val_te = fmt(m_te, n_sig_test)
        print(f"  {row_lbl:<20}  {val_tr:^{col_w}}  {val_te:^{col_w}}")

    print("=" * total_w + "\n")


def print_ticker_breakdown(trades: pd.DataFrame, label: str) -> None:
    if trades.empty:
        print(f"  No executed trades in {label}.\n")
        return

    print(f"\n  Per-Ticker Breakdown — {label}")
    hdr = (
        f"  {'Ticker':<9}  {'N':>4}  {'Win%':>6}  "
        f"{'AvgPnL':>8}  {'Best':>8}  {'Worst':>9}  {'Stops':>6}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for tkr, grp in trades.groupby("Ticker"):
        wr    = grp["Is_Win"].mean()
        avg   = grp["PnL_Pct"].mean()
        best  = grp["PnL_Pct"].max()
        worst = grp["PnL_Pct"].min()
        stops = int(grp["Stop_Hit"].sum())
        print(
            f"  {tkr:<9}  {len(grp):>4}  {wr:>5.1%}  {avg:>+7.2%}  "
            f"{best:>+7.2%}  {worst:>+8.2%}  {stops:>6}"
        )
    print()


def save_equity_chart(
    train_trades: pd.DataFrame,
    test_trades: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "First Red Day Short Strategy — Polygon Universe\n"
        "Train 2022-2024  vs  Out-of-Sample 2025",
        fontsize=13,
    )

    for col, (trades, label) in enumerate(
        [(train_trades, "Train 2022-2024"), (test_trades, "Test 2025 OOS")]
    ):
        ax_eq  = axes[0][col]
        ax_bar = axes[1][col]

        if trades.empty:
            for ax in (ax_eq, ax_bar):
                ax.text(0.5, 0.5, "No trades", ha="center", va="center",
                        transform=ax.transAxes, fontsize=13, color="gray")
                ax.set_xticks([])
            ax_eq.set_title(label)
            continue

        t   = trades.copy().reset_index(drop=True)
        m   = _metrics(t)
        t["Cum_PnL"] = t["PnL_Pct"].cumsum() * 100

        ax_eq.set_title(
            f"{label}\n"
            f"WR={m['win_rate']:.1%}  E={m['expectancy']:+.2%}/trade  n={m['n']}"
        )
        ax_eq.plot(t.index, t["Cum_PnL"], color="steelblue", linewidth=1.5)
        ax_eq.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_eq.fill_between(t.index, t["Cum_PnL"], 0,
                           where=(t["Cum_PnL"] >= 0), alpha=0.15, color="green")
        ax_eq.fill_between(t.index, t["Cum_PnL"], 0,
                           where=(t["Cum_PnL"] <  0), alpha=0.15, color="red")
        ax_eq.set_ylabel("Cumulative Return (%)")
        ax_eq.grid(True, alpha=0.3)

        colors = ["#27ae60" if w else "#e74c3c" for w in t["Is_Win"]]
        ax_bar.bar(t.index, t["PnL_Pct"] * 100, color=colors, alpha=0.8, width=0.7)
        ax_bar.axhline(0, color="black", linewidth=0.8)
        ax_bar.set_ylabel("Trade P&L (%)")
        ax_bar.set_xlabel("Trade # (chronological)")
        ax_bar.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "frd_polygon_equity.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Equity chart saved -> {out}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("+" + "=" * 62 + "+")
    print("|    FRD Short Strategy — Polygon.io Universe (2022-2025)    |")
    print("+" + "=" * 62 + "+")
    print(f"  Universe : ${PRICE_MIN}-${PRICE_MAX}  |  3d gain >={GAIN_3D_MIN:.0%}  "
          f"|  avg vol >={VOL_MIN:,}")
    print(f"  FRD flts : streak <={MAX_STREAK}  |  FRD vol <={VOL_FADE_MAX:.0%} of run avg")
    print(f"  Risk     : {STOP_LOSS_PCT:.0%} hard stop  |  {SQUEEZE_LOOKBACK}d squeeze "
          f"(reset between periods)\n")

    # ── Step 1: discover universe ─────────────────────────────────────────────
    print(">> [1/4] Scanning Polygon for qualifying tickers (2022-2025) ...")
    data = scan_and_fetch(
        start       = TRAIN_START,
        end         = TEST_END,
        price_min   = PRICE_MIN,
        price_max   = PRICE_MAX,
        gain_3d_min = GAIN_3D_MIN,
        vol_min     = VOL_MIN,
        lookback_days = 12,
    )
    print(f"  Universe: {len(data)} qualifying tickers.\n")

    if not data:
        print("  No tickers found — check API key and network.")
        return

    # ── Step 2: detect signals for each period ────────────────────────────────
    print(">> [2/4] Detecting FRD signals ...")
    sig_train = detect_signals(data, TRAIN_START, TRAIN_END)
    sig_test  = detect_signals(data, TEST_START,  TEST_END)

    n_sig_train = len(sig_train)
    n_sig_test  = len(sig_test)
    print(f"  Training signals  (2022-2024) : {n_sig_train}")
    print(f"  Test signals      (2025)      : {n_sig_test}\n")

    # ── Step 3: simulate trades (squeeze state does NOT cross period boundary) ─
    print(">> [3/4] Simulating trades ...")
    trades_train = simulate_trades(sig_train)
    trades_test  = simulate_trades(sig_test)

    n_tr = len(trades_train)
    n_te = len(trades_test)
    print(f"  Train trades : {n_tr}  "
          f"({n_sig_train - n_tr} skipped by squeeze)")
    print(f"  Test  trades : {n_te}  "
          f"({n_sig_test  - n_te} skipped by squeeze)\n")

    # ── Step 4: report ────────────────────────────────────────────────────────
    print(">> [4/4] Generating performance reports ...")
    print_side_by_side(trades_train, trades_test, n_sig_train, n_sig_test)
    print_ticker_breakdown(trades_train, "Train 2022-2024")
    print_ticker_breakdown(trades_test,  "Test  2025 OOS")
    save_equity_chart(trades_train, trades_test)

    # Save raw trade logs for further analysis
    if not trades_train.empty:
        p = os.path.join(OUTPUT_DIR, "trades_train_2022_2024.csv")
        trades_train.to_csv(p, index=False)
        print(f"  Train trade log -> {p}")

    if not trades_test.empty:
        p = os.path.join(OUTPUT_DIR, "trades_test_2025.csv")
        trades_test.to_csv(p, index=False)
        print(f"  Test  trade log -> {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()

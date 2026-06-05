# -*- coding: utf-8 -*-
"""
frd_trash_backtest.py
=====================
Backtest the fundamental trash score as a filter on FRD short signals.

For every historical FRD trade produced by the feature-filter run, fetches
Polygon quarterly financials filed BEFORE the signal date (point-in-time safe),
computes a trash score, then compares High Trash (>= TRASH_THRESHOLD) vs
Low Trash performance side by side.

Usage
-----
    python frd_trash_backtest.py
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time as _time
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests

from polygon_fetcher import API_KEY, BASE_URL
from frd_backtest import (
    TICKERS, START_DATE, END_DATE,
    fetch_data, detect_signals, simulate_trades,
    _metrics, print_comparison,
    FEAT_MIN_GAIN, FEAT_MAX_STREAK, FEAT_VOL_FADE,
    MIN_PUMP_GAIN, MIN_AVG_VOL,
    OUTPUT_DIR,
)

TRASH_THRESHOLD = 7

# ── Polygon helper ─────────────────────────────────────────────────────────────

def _get(path, params=None, retries=3):
    url = BASE_URL + path
    p = {"apiKey": API_KEY, **(params or {})}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=p, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                _time.sleep(2 ** (attempt + 2))
                continue
            return {}
        except Exception:
            _time.sleep(2)
    return {}


# ── Fetch historical financials ────────────────────────────────────────────────

def fetch_all_financials(tickers: list) -> dict:
    """
    Fetch all available quarterly filings for each ticker from Polygon.

    Returns {ticker: [filings sorted by filing_date descending]}
    Each filing: {"filing_date": date, "financials": dict}
    """
    result = {}
    for ticker in tickers:
        print(f"  {ticker} ...", end=" ", flush=True)
        filings = []
        body = _get(
            "/vX/reference/financials",
            {"ticker": ticker, "timeframe": "quarterly",
             "limit": 20, "order": "desc"},
        )
        for item in body.get("results", []):
            fd = item.get("filing_date") or item.get("period_of_report_date")
            if not fd:
                continue
            try:
                filing_date = date.fromisoformat(fd)
            except Exception:
                continue
            filings.append({
                "filing_date": filing_date,
                "financials":  item.get("financials", {}),
            })
        filings.sort(key=lambda x: x["filing_date"], reverse=True)
        result[ticker] = filings
        print(f"{len(filings)} filings")
        _time.sleep(0.25)
    return result


# ── Point-in-time trash scoring ────────────────────────────────────────────────

def _extract(fin: dict) -> tuple:
    """Return (revenue, net_income, cash, shares) from a Polygon financials dict."""
    ic = fin.get("income_statement", {})
    bs = fin.get("balance_sheet", {})

    revenue = 0.0
    for k in ("revenues", "revenue", "net_revenues"):
        if k in ic:
            revenue = float(ic[k].get("value") or 0)
            break

    net_income = 0.0
    for k in ("net_income_loss", "net_income_loss_attributable_to_parent"):
        if k in ic:
            net_income = float(ic[k].get("value") or 0)
            break

    cash = 0.0
    for k in ("cash_and_cash_equivalents_including_discontinued_operations",
              "cash_and_cash_equivalents", "cash_and_equivalents"):
        if k in bs:
            cash = float(bs[k].get("value") or 0)
            break

    shares = 0.0
    for k in ("diluted_average_shares", "basic_average_shares"):
        if k in ic:
            shares = float(ic[k].get("value") or 0)
            break
    if not shares:
        if "common_stock_shares_outstanding" in bs:
            shares = float(bs["common_stock_shares_outstanding"].get("value") or 0)

    return revenue, net_income, cash, shares


def score_signal(
    ticker: str,
    signal_date: pd.Timestamp,
    price_at_signal: float,
    df_price: pd.DataFrame,
    filings_by_ticker: dict,
) -> tuple[int, dict]:
    """
    Compute a point-in-time trash score for one historical signal.
    Only uses filings filed strictly before signal_date (no lookahead).

    Returns (score 0-10, breakdown_dict).
    """
    score = 0
    bd = {k: False for k in (
        "no_revenue", "burning_excess", "short_runway",
        "recent_dilution", "small_cap", "sub2_history",
    )}

    filings = [f for f in filings_by_ticker.get(ticker, [])
               if f["filing_date"] < signal_date.date()]
    if not filings:
        return 0, bd

    revenue, net_income, cash, shares = _extract(filings[0]["financials"])
    market_cap = shares * price_at_signal if shares > 0 else 0.0

    if 0 < market_cap < 50_000_000:
        score += 1
        bd["small_cap"] = True

    if revenue == 0.0:
        score += 3
        bd["no_revenue"] = True

    if net_income < 0 and market_cap > 0:
        if abs(net_income) * 4 / market_cap > 0.50:
            score += 2
            bd["burning_excess"] = True

    quarterly_burn = abs(net_income) if net_income < 0 else 0.0
    if quarterly_burn > 0 and cash > 0:
        if (cash / quarterly_burn) * 3.0 < 6.0:
            score += 2
            bd["short_runway"] = True

    if len(filings) >= 2:
        # Compare most-recent filing vs up to 4 quarters back
        _, _, _, shares_old = _extract(filings[min(len(filings) - 1, 3)]["financials"])
        if shares > 0 and shares_old > 0:
            if (shares - shares_old) / shares_old > 0.15:
                score += 1
                bd["recent_dilution"] = True

    # Sub-$2 in prior 12 months — use yfinance data already in memory
    yr_ago = signal_date - timedelta(days=365)
    hist = df_price.loc[yr_ago:signal_date, "Low"].squeeze()
    if not hist.empty and (hist < 2.0).any():
        score += 1
        bd["sub2_history"] = True

    return min(score, 10), bd


# ── Output helpers ─────────────────────────────────────────────────────────────

def print_score_distribution(trades: pd.DataFrame) -> None:
    print("\n  Trash Score Distribution:")
    print(f"  {'Score':>6}  {'Trades':>7}  {'WinRate':>8}  {'AvgPnL':>8}  {'Expectancy':>11}")
    print("  " + "-" * 48)
    for s in sorted(trades["Trash_Score"].unique()):
        g = trades[trades["Trash_Score"] == s]
        wr  = g["Is_Win"].mean()
        avg = g["PnL_Pct"].mean()
        wins   = g[g["Is_Win"]]
        losses = g[~g["Is_Win"]]
        aw = wins["PnL_Pct"].mean()   if not wins.empty   else 0.0
        al = losses["PnL_Pct"].mean() if not losses.empty else 0.0
        exp = wr * aw + (1 - wr) * al
        tag = " ⚠️ HIGH" if s >= TRASH_THRESHOLD else ""
        print(f"  {s:>6}  {len(g):>7}  {wr:>7.1%}  {avg:>+7.2%}  {exp:>+10.2%}{tag}")
    print()


def save_charts(trades: pd.DataFrame) -> None:
    high = trades[trades["Trash_Score"] >= TRASH_THRESHOLD].copy()
    low  = trades[trades["Trash_Score"] <  TRASH_THRESHOLD].copy()

    high["Cum"] = high["PnL_Pct"].cumsum() * 100
    low["Cum"]  = low["PnL_Pct"].cumsum()  * 100

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"FRD Trash Score Backtest  ({START_DATE} — {END_DATE})\n"
        f"Feature filters: gain>{FEAT_MIN_GAIN:.0%} | streak<={FEAT_MAX_STREAK} | vol<={FEAT_VOL_FADE:.0%}  "
        f"|  High Trash = score >= {TRASH_THRESHOLD}",
        fontsize=11,
    )

    # ── Top-left: equity curves ───────────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_title("Equity Curve — High vs Low Trash")
    if not high.empty:
        ax.plot(range(len(high)), high["Cum"].values,
                color="#c0392b", linewidth=1.8, label=f"High Trash ≥{TRASH_THRESHOLD} (n={len(high)})")
    if not low.empty:
        ax.plot(range(len(low)), low["Cum"].values,
                color="#27ae60", linewidth=1.8, label=f"Low Trash <{TRASH_THRESHOLD} (n={len(low)})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_xlabel("Trade #")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Top-right: win rate by score ──────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_title("Win Rate by Trash Score")
    scores = sorted(trades["Trash_Score"].unique())
    wr_by_score = [trades[trades["Trash_Score"] == s]["Is_Win"].mean() for s in scores]
    n_by_score  = [len(trades[trades["Trash_Score"] == s]) for s in scores]
    bar_colors  = ["#c0392b" if s >= TRASH_THRESHOLD else "#27ae60" for s in scores]
    bars = ax.bar(scores, [w * 100 for w in wr_by_score], color=bar_colors, alpha=0.8)
    for bar, n in zip(bars, n_by_score):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.axhline(50, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Trash Score")
    ax.set_ylabel("Win Rate (%)")
    ax.set_xticks(scores)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Bottom-left: avg PnL by score ─────────────────────────────────────────
    ax = axes[1, 0]
    ax.set_title("Avg PnL % by Trash Score")
    avg_pnl = [trades[trades["Trash_Score"] == s]["PnL_Pct"].mean() * 100 for s in scores]
    bar_colors2 = ["#c0392b" if s >= TRASH_THRESHOLD else "#27ae60" for s in scores]
    ax.bar(scores, avg_pnl, color=bar_colors2, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Trash Score")
    ax.set_ylabel("Avg PnL (%)")
    ax.set_xticks(scores)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Bottom-right: per-trade scatter coloured by score ─────────────────────
    ax = axes[1, 1]
    ax.set_title("Per-Trade PnL coloured by Trash Score")
    scatter_colors = ["#c0392b" if s >= TRASH_THRESHOLD else "#27ae60"
                      for s in trades["Trash_Score"]]
    ax.scatter(range(len(trades)), trades["PnL_Pct"] * 100,
               c=scatter_colors, alpha=0.6, s=30)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Trade # (chronological)")
    ax.set_ylabel("PnL (%)")
    ax.grid(True, alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#c0392b", label=f"High Trash ≥{TRASH_THRESHOLD}"),
        Patch(color="#27ae60", label=f"Low Trash <{TRASH_THRESHOLD}"),
    ], fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "frd_trash_backtest.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def score_trades(trades: pd.DataFrame, data: dict, filings: dict) -> pd.DataFrame:
    trades = trades.copy()
    scores, zero_data = [], 0
    for _, row in trades.iterrows():
        tkr = row["Ticker"]
        s, _ = score_signal(
            tkr, row["Signal_Date"], row["Signal_Close"], data[tkr], filings,
        )
        scores.append(s)
        if s == 0:
            zero_data += 1
    trades["Trash_Score"] = scores
    if zero_data:
        print(f"  Note: {zero_data} trades scored 0 "
              f"(no Polygon filings before signal date).")
    return trades


def main():
    print("+" + "=" * 62 + "+")
    print("|         FRD Trash Score Backtest — Polygon.io          |")
    print("+" + "=" * 62 + "+")
    print(f"  Period     : {START_DATE}  ->  {END_DATE}")
    print(f"  High Trash : score >= {TRASH_THRESHOLD}")
    print()

    # ── 1. Price data ─────────────────────────────────────────────────────────
    print(">> [1/5] Fetching OHLCV data (yfinance) ...")
    data = fetch_data(TICKERS, START_DATE, END_DATE)
    print(f"\n  Loaded {len(data)}/{len(TICKERS)} tickers.\n")

    # ── 2. Detect signals — run BOTH filter regimes ───────────────────────────
    print(">> [2/5] Detecting FRD signals ...")

    # Relaxed: wider filters, bigger sample
    signals_r = detect_signals(data, verbose=False)
    # Feature: strict filters used by the main backtest
    signals_f = detect_signals(
        data,
        gain_override = FEAT_MIN_GAIN,
        max_streak    = FEAT_MAX_STREAK,
        vol_fade_max  = FEAT_VOL_FADE,
        verbose       = False,
    )
    print(f"  Relaxed signals : {len(signals_r)}")
    print(f"  Feature signals : {len(signals_f)}\n")

    # Pick whichever gives enough trades; prefer relaxed for sample size
    if len(signals_r) >= 10:
        signals = signals_r
        label = f"Relaxed (gain>{MIN_PUMP_GAIN:.0%}, vol>{MIN_AVG_VOL/1e6:.1f}M)"
    elif len(signals_f) >= 5:
        signals = signals_f
        label = f"Feature (gain>{FEAT_MIN_GAIN:.0%}, streak<={FEAT_MAX_STREAK})"
    else:
        # Merge both for maximum coverage
        signals = pd.concat([signals_r, signals_f]).drop_duplicates(
            subset=["Ticker", "Signal_Date"]
        ).reset_index(drop=True)
        label = "Combined (relaxed + feature)"

    print(f"  Using: {label}  ({len(signals)} signals)\n")

    if signals.empty:
        print("  No signals — nothing to score.")
        return

    # ── 3. Simulate trades ────────────────────────────────────────────────────
    print(">> [3/5] Simulating trades ...")
    trades = simulate_trades(signals, use_stop=True, use_squeeze=True)
    print(f"  {len(trades)} trades executed "
          f"({len(signals) - len(trades)} skipped by squeeze filter).\n")

    # ── 4. Fetch Polygon fundamentals ─────────────────────────────────────────
    unique_tickers = sorted(trades["Ticker"].unique())
    print(f">> [4/5] Fetching Polygon quarterly financials "
          f"for {len(unique_tickers)} tickers ...")
    filings = fetch_all_financials(unique_tickers)
    print()

    # ── 5. Score trades ───────────────────────────────────────────────────────
    print(">> [5/5] Scoring trades ...")
    trades = score_trades(trades, data, filings)
    print()

    # ── Results ───────────────────────────────────────────────────────────────
    high = trades[trades["Trash_Score"] >= TRASH_THRESHOLD]
    low  = trades[trades["Trash_Score"] <  TRASH_THRESHOLD]

    print_score_distribution(trades)

    print_comparison([
        (f"High Trash ≥{TRASH_THRESHOLD}", high),
        (f"Low Trash <{TRASH_THRESHOLD}",  low),
        ("All Trades",                      trades),
    ])

    # Full trade list
    print("  All trades with trash scores:")
    print(f"  {'Ticker':<7}  {'Signal Date':>12}  {'PnL':>7}  {'Win':>4}  {'Trash':>6}")
    print("  " + "-" * 44)
    for _, r in trades.sort_values("Signal_Date").iterrows():
        warn = " ⚠️" if r["Trash_Score"] >= TRASH_THRESHOLD else ""
        print(f"  {r['Ticker']:<7}  {str(r['Signal_Date'].date()):>12}  "
              f"{r['PnL_Pct']:>+6.1%}  {'W' if r['Is_Win'] else 'L':>4}  "
              f"{r['Trash_Score']:>4}/10{warn}")
    print()

    save_charts(trades)


if __name__ == "__main__":
    main()

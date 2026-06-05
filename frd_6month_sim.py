# -*- coding: utf-8 -*-
"""
frd_6month_sim.py
=================
Replay the full FRD scanner logic over the last 6 months and show every
alert that would have fired.

For each trading day in the simulation window the script:
  1. Builds the watchlist exactly as the live scanner does (7-day lookback,
     all scanner universe filters including vol-ratio).
  2. Checks if that day was an FRD for any watchlist ticker:
       close < prev_close  AND  close < high * (1 - HOD_FADE_PCT)
  3. Simulates the trade: short at next-day open, cover at next-day close
     (or at the 12% stop if next-day high hits it first).
  4. Fetches current fundamental trash scores for every alert ticker.

Grouped daily data is pulled from Polygon and cached in poly_cache/,
so re-runs are fast once the data is downloaded.

Usage
-----
    python frd_6month_sim.py
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time as _time
from datetime import date, timedelta
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from polygon_fetcher import fetch_grouped_day, _bdays, API_KEY, BASE_URL
import requests

# ── Scanner parameters (must match frd_intraday_scanner.py) ───────────────────
PRICE_MIN         = 3.0
PRICE_MAX         = 10.0
GAIN_3D_MIN       = 1.00
VOL_MIN           = 1_000_000
MAX_STREAK        = 4
VOL_RATIO_LOW     = 0.10
VOL_RATIO_HIGH    = 0.50
HOD_FADE_PCT      = 0.03
STOP_PCT          = 0.12
LOOKBACK_DAYS     = 7
TRASH_THRESHOLD   = 7

SIM_MONTHS        = 6
OUTPUT_DIR        = os.path.dirname(os.path.abspath(__file__))


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


# ── Fundamental trash score (current, not point-in-time) ──────────────────────

def fetch_trash_score(ticker: str) -> tuple[int, dict]:
    score = 0
    bd = {k: False for k in (
        "no_revenue", "burning_excess", "short_runway",
        "recent_dilution", "small_cap", "sub2_history",
    )}
    try:
        ref = _get(f"/v3/reference/tickers/{ticker}")
        info = ref.get("results", {})
        market_cap = float(info.get("market_cap") or 0)
        if 0 < market_cap < 50_000_000:
            score += 1
            bd["small_cap"] = True

        fin = _get("/vX/reference/financials",
                   {"ticker": ticker, "timeframe": "quarterly",
                    "limit": 4, "order": "desc"})
        results = fin.get("results", [])
        revenue = net_income = cash = shares_recent = shares_old = 0.0

        if results:
            ic = results[0].get("financials", {}).get("income_statement", {})
            bs = results[0].get("financials", {}).get("balance_sheet", {})
            for k in ("revenues", "revenue", "net_revenues"):
                if k in ic:
                    revenue = float(ic[k].get("value") or 0); break
            for k in ("net_income_loss", "net_income_loss_attributable_to_parent"):
                if k in ic:
                    net_income = float(ic[k].get("value") or 0); break
            for k in ("cash_and_cash_equivalents_including_discontinued_operations",
                      "cash_and_cash_equivalents", "cash_and_equivalents"):
                if k in bs:
                    cash = float(bs[k].get("value") or 0); break
            for k in ("diluted_average_shares", "basic_average_shares"):
                if k in ic:
                    shares_recent = float(ic[k].get("value") or 0); break
            if not shares_recent and "common_stock_shares_outstanding" in bs:
                shares_recent = float(bs["common_stock_shares_outstanding"].get("value") or 0)

        if revenue == 0.0:
            score += 3; bd["no_revenue"] = True
        if net_income < 0 and market_cap > 0:
            if abs(net_income) * 4 / market_cap > 0.50:
                score += 2; bd["burning_excess"] = True
        quarterly_burn = abs(net_income) if net_income < 0 else 0.0
        if quarterly_burn > 0 and cash > 0:
            if (cash / quarterly_burn) * 3.0 < 6.0:
                score += 2; bd["short_runway"] = True

        if len(results) >= 2:
            ic_old = results[-1].get("financials", {}).get("income_statement", {})
            bs_old = results[-1].get("financials", {}).get("balance_sheet", {})
            for k in ("diluted_average_shares", "basic_average_shares"):
                if k in ic_old:
                    shares_old = float(ic_old[k].get("value") or 0); break
            if not shares_old and "common_stock_shares_outstanding" in bs_old:
                shares_old = float(bs_old["common_stock_shares_outstanding"].get("value") or 0)
            if shares_recent > 0 and shares_old > 0:
                if (shares_recent - shares_old) / shares_old > 0.15:
                    score += 1; bd["recent_dilution"] = True

        yr_ago = (date.today() - timedelta(days=365)).isoformat()
        hist = _get(f"/v2/aggs/ticker/{ticker}/range/1/day/{yr_ago}/{date.today().isoformat()}",
                    {"adjusted": "true", "limit": 300})
        for bar in hist.get("results", []):
            if float(bar.get("l", 999)) < 2.0:
                score += 1; bd["sub2_history"] = True; break
    except Exception:
        pass
    return min(score, 10), bd


# ── Universe helpers ───────────────────────────────────────────────────────────

def _streak(closes):
    count = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] > closes[i - 1]:
            count += 1
        else:
            break
    return count


def build_watchlist(ticker_bars: dict) -> dict:
    """
    Apply scanner universe filters to a dict of {ticker: [bars]}.
    Each bar is (open, high, low, close, volume).
    Returns {ticker: meta} matching build_pump_universe() in the scanner.
    """
    universe = {}
    for ticker, bars in ticker_bars.items():
        if len(bars) < 4:
            continue
        closes = [b[3] for b in bars]
        vols   = [b[4] for b in bars]
        last_close = closes[-1]

        if not (PRICE_MIN <= last_close <= PRICE_MAX):
            continue
        base = closes[-4]
        if base <= 0:
            continue
        if (last_close - base) / base < GAIN_3D_MIN:
            continue
        run_avg_vol = sum(vols[-3:]) / 3
        if run_avg_vol < VOL_MIN:
            continue
        s = _streak(closes)
        if s > MAX_STREAK:
            continue
        prior_avg = sum(vols[-4:-1]) / 3 if len(vols) >= 4 else 0.0
        vol_ratio = vols[-1] / prior_avg if prior_avg > 0 else 0.0
        if not (vol_ratio < VOL_RATIO_LOW or vol_ratio >= VOL_RATIO_HIGH):
            continue

        universe[ticker] = {
            "prev_close":  last_close,
            "roll3_gain":  (last_close - base) / base,
            "run_avg_vol": run_avg_vol,
            "streak":      s,
            "vol_ratio":   vol_ratio,
        }
    return universe


# ── Main simulation ────────────────────────────────────────────────────────────

def main():
    today     = date.today()
    sim_end   = today - timedelta(days=1)           # yesterday (last complete day)
    sim_start = today - timedelta(days=SIM_MONTHS * 31)

    # Extra days at the start so the first sim day has a full 7-day lookback
    data_start = sim_start - timedelta(days=LOOKBACK_DAYS * 3)

    print("+" + "=" * 66 + "+")
    print("|        FRD Scanner 6-Month Simulation — Polygon.io          |")
    print("+" + "=" * 66 + "+")
    print(f"  Simulation : {sim_start}  ->  {sim_end}")
    print(f"  Filters    : ${PRICE_MIN}-${PRICE_MAX}  |  3d gain >={GAIN_3D_MIN:.0%}"
          f"  |  avg vol >={VOL_MIN/1e6:.0f}M  |  streak <={MAX_STREAK}")
    print(f"  Vol ratio  : < {VOL_RATIO_LOW}  OR  >= {VOL_RATIO_HIGH}")
    print(f"  FRD signal : close < prev_close  AND  >={HOD_FADE_PCT:.0%} off day-high  AND  close < VWAP")
    print(f"  Stop loss  : {STOP_PCT:.0%} above entry (next-day open)")
    print()

    # ── Load all grouped daily data ───────────────────────────────────────────
    all_days = _bdays(data_start.isoformat(), sim_end.isoformat())
    sim_days = _bdays(sim_start.isoformat(), sim_end.isoformat())

    print(f">> [1/4] Loading {len(all_days)} days of grouped daily data ...")
    print(f"  ({data_start} -> {sim_end}, cached in poly_cache/)\n")

    daily_frames: dict[str, pd.DataFrame] = {}
    for i, day in enumerate(all_days):
        if i % 20 == 0:
            print(f"  {i}/{len(all_days)} days loaded ...", flush=True)
        daily_frames[day] = fetch_grouped_day(day)

    print(f"  {len(all_days)} days loaded.\n")

    # Build ticker_bars_by_day: for quick rolling-window access
    # {ticker: {day: (open, high, low, close, volume)}}
    print(">> [2/4] Indexing bars by ticker ...")
    bars_by_ticker: dict[str, dict[str, tuple]] = defaultdict(dict)
    for day, df in daily_frames.items():
        for row in df.itertuples(index=False):
            if not (0.5 <= row.close <= 50.0 and row.volume >= 100_000):
                continue
            vwap = float(row.vwap) if hasattr(row, "vwap") and pd.notna(row.vwap) else None
            bars_by_ticker[row.ticker][day] = (
                float(row.open), float(row.high), float(row.low),
                float(row.close), float(row.volume), vwap,
            )
    print(f"  {len(bars_by_ticker):,} tickers indexed.\n")

    # ── Day-by-day simulation ─────────────────────────────────────────────────
    print(">> [3/4] Simulating scanner day by day ...")
    alerts = []
    watchlist_sizes = []

    for sim_idx, sim_day in enumerate(sim_days):
        day_pos = all_days.index(sim_day) if sim_day in all_days else -1
        if day_pos < LOOKBACK_DAYS:
            continue

        # The 7 trading days ending the day before sim_day
        lookback_window = all_days[day_pos - LOOKBACK_DAYS: day_pos]

        # Build rolling bars for each ticker over the lookback window
        ticker_bars: dict[str, list] = defaultdict(list)
        for day in lookback_window:
            for ticker, day_bars in bars_by_ticker.items():
                if day in day_bars:
                    ticker_bars[ticker].append(day_bars[day])

        # Build watchlist
        watchlist = build_watchlist(ticker_bars)
        watchlist_sizes.append(len(watchlist))

        # Check FRD on sim_day
        for ticker, meta in watchlist.items():
            bar = bars_by_ticker[ticker].get(sim_day)
            if bar is None:
                continue
            o, h, l, c, v, vwap = bar
            prev_close  = meta["prev_close"]
            gone_red    = c < prev_close
            fading_hod  = c < h * (1.0 - HOD_FADE_PCT)
            below_vwap  = (vwap is not None) and (c < vwap)

            if gone_red and fading_hod and below_vwap:
                alerts.append({
                    "Alert_Date":  sim_day,
                    "Ticker":      ticker,
                    "Prev_Close":  round(prev_close, 2),
                    "Alert_Close": round(c, 2),
                    "Day_High":    round(h, 2),
                    "VWAP":        round(vwap, 2),
                    "Pct_vs_Prev": round((c - prev_close) / prev_close, 4),
                    "Pct_off_HOD": round((h - c) / h, 4),
                    "Pct_vs_VWAP": round((c - vwap) / vwap, 4),
                    "Roll3_Gain":  round(meta["roll3_gain"], 4),
                    "Vol_Ratio":   round(meta["vol_ratio"], 2),
                    "Streak":      meta["streak"],
                })

    print(f"  Simulated {len(sim_days)} trading days.")
    print(f"  Avg watchlist size: {sum(watchlist_sizes)/max(len(watchlist_sizes),1):.1f} tickers/day")
    print(f"  Total FRD alerts  : {len(alerts)}\n")

    if not alerts:
        print("  No alerts in this period.")
        return

    alerts_df = pd.DataFrame(alerts)

    # ── Trade simulation ──────────────────────────────────────────────────────
    # Entry = next trading day's open, Exit = next trading day's close
    # Stop = entry * (1 + STOP_PCT) if next day's high reaches it
    entries, exits, stop_hits, pnl_pcts = [], [], [], []
    for _, row in alerts_df.iterrows():
        ticker   = row["Ticker"]
        alert_d  = row["Alert_Date"]
        if alert_d not in all_days:
            entries.append(None); exits.append(None)
            stop_hits.append(False); pnl_pcts.append(None)
            continue
        idx = all_days.index(alert_d)
        if idx + 1 >= len(all_days):
            entries.append(None); exits.append(None)
            stop_hits.append(False); pnl_pcts.append(None)
            continue
        next_day = all_days[idx + 1]
        bar = bars_by_ticker[ticker].get(next_day)
        if bar is None:
            entries.append(None); exits.append(None)
            stop_hits.append(False); pnl_pcts.append(None)
            continue
        o, h, l, c, v, _vwap = bar
        entry   = o
        stop_px = entry * (1.0 + STOP_PCT)
        if h >= stop_px:
            actual_exit = stop_px
            stop_hit    = True
        else:
            actual_exit = c
            stop_hit    = False
        pnl = (entry - actual_exit) / entry
        entries.append(round(entry, 2))
        exits.append(round(actual_exit, 2))
        stop_hits.append(stop_hit)
        pnl_pcts.append(round(pnl, 4))

    alerts_df["Entry_Open"] = entries
    alerts_df["Exit"]       = exits
    alerts_df["Stop_Hit"]   = stop_hits
    alerts_df["PnL_Pct"]    = pnl_pcts
    alerts_df["Is_Win"]     = alerts_df["PnL_Pct"].apply(
        lambda x: x > 0 if x is not None else None
    )

    # ── Fetch trash scores ────────────────────────────────────────────────────
    unique_tickers = sorted(alerts_df["Ticker"].unique())
    print(f">> [4/4] Fetching trash scores for {len(unique_tickers)} alert tickers ...")
    trash_scores = {}
    for tkr in unique_tickers:
        try:
            ts, _ = fetch_trash_score(tkr)
        except Exception:
            ts = 0
        trash_scores[tkr] = ts
        print(f"  {tkr}: {ts}/10")
        _time.sleep(0.25)

    alerts_df["Trash_Score"] = alerts_df["Ticker"].map(trash_scores).fillna(0).astype(int)
    print()

    # Drop alerts with no next-day data (end of dataset)
    tradeable = alerts_df.dropna(subset=["PnL_Pct"]).copy()

    # ── Print results ─────────────────────────────────────────────────────────
    w = 118
    print("=" * w)
    print("  ALERTS THAT WOULD HAVE FIRED (last 6 months, with VWAP filter)")
    print("=" * w)
    print(f"  {'Date':<12}  {'Ticker':<7}  {'PrevC':>6}  {'Close':>6}  "
          f"{'VWAP':>6}  {'High':>6}  {'vsPrev':>7}  {'offHOD':>7}  {'vsVWAP':>7}  "
          f"{'3dGain':>7}  {'Streak':>6}  "
          f"{'Entry':>6}  {'Exit':>6}  {'PnL':>7}  {'Trash':>7}")
    print("  " + "-" * (w - 2))

    for _, r in alerts_df.sort_values("Alert_Date").iterrows():
        ts   = int(r["Trash_Score"])
        warn = "⚠️ " if ts >= TRASH_THRESHOLD else "   "
        pnl_str   = f"{r['PnL_Pct']:>+6.1%}" if pd.notna(r["PnL_Pct"])    else "   n/a"
        entry_str = f"${r['Entry_Open']:>5.2f}" if pd.notna(r["Entry_Open"]) else "   n/a"
        exit_str  = f"${r['Exit']:>5.2f}"       if pd.notna(r["Exit"])       else "   n/a"
        stop_tag  = "S" if r["Stop_Hit"] else " "
        date_str  = r["Alert_Date"].isoformat() if hasattr(r["Alert_Date"], "isoformat") \
                    else str(r["Alert_Date"])[:10]
        print(
            f"  {date_str:<12}  {r['Ticker']:<7}  "
            f"${r['Prev_Close']:>5.2f}  ${r['Alert_Close']:>5.2f}  "
            f"${r['VWAP']:>5.2f}  ${r['Day_High']:>5.2f}  "
            f"{r['Pct_vs_Prev']:>+6.1%}  {r['Pct_off_HOD']:>6.1%}  {r['Pct_vs_VWAP']:>+6.1%}  "
            f"{r['Roll3_Gain']:>6.0%}  {r['Streak']:>6}  "
            f"{entry_str}  {exit_str}  {pnl_str}{stop_tag}  "
            f"{warn}{ts:>2}/10"
        )

    # ── Performance summary ───────────────────────────────────────────────────
    print()
    if tradeable.empty:
        print("  No tradeable alerts (no next-day data available).")
        return

    wins   = tradeable[tradeable["Is_Win"] == True]
    losses = tradeable[tradeable["Is_Win"] == False]
    n      = len(tradeable)
    wr     = len(wins) / n
    aw     = wins["PnL_Pct"].mean()   if not wins.empty   else 0.0
    al     = losses["PnL_Pct"].mean() if not losses.empty else 0.0
    exp    = wr * aw + (1 - wr) * al
    stops  = int(tradeable["Stop_Hit"].sum())
    h_tr   = tradeable[tradeable["Trash_Score"] >= TRASH_THRESHOLD]
    l_tr   = tradeable[tradeable["Trash_Score"] <  TRASH_THRESHOLD]

    print("=" * 60)
    print("  PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"  Alerts fired       : {len(alerts_df)}")
    print(f"  Tradeable alerts   : {n}  ({stops} stopped out)")
    print(f"  Win rate           : {wr:.1%}  ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg win            : {aw:+.2%}")
    print(f"  Avg loss           : {al:+.2%}")
    print(f"  Expectancy/trade   : {exp:+.2%}")
    print(f"  Best trade         : {tradeable['PnL_Pct'].max():+.2%}"
          f"  ({tradeable.loc[tradeable['PnL_Pct'].idxmax(), 'Ticker']})")
    print(f"  Worst trade        : {tradeable['PnL_Pct'].min():+.2%}"
          f"  ({tradeable.loc[tradeable['PnL_Pct'].idxmin(), 'Ticker']})")
    print("-" * 60)

    def _grp_stats(grp, label):
        if grp.empty:
            print(f"  {label:<25}: no trades")
            return
        w2 = grp[grp["Is_Win"] == True]
        l2 = grp[grp["Is_Win"] == False]
        wr2 = len(w2) / len(grp)
        aw2 = w2["PnL_Pct"].mean() if not w2.empty else 0.0
        al2 = l2["PnL_Pct"].mean() if not l2.empty else 0.0
        print(f"  {label:<25}: n={len(grp):>3}  wr={wr2:.1%}  "
              f"exp={wr2*aw2+(1-wr2)*al2:+.2%}  avg={grp['PnL_Pct'].mean():+.2%}")

    _grp_stats(h_tr, f"High Trash (>={TRASH_THRESHOLD})")
    _grp_stats(l_tr, f"Low Trash (<{TRASH_THRESHOLD})")
    print("=" * 60)

    # Monthly breakdown
    tradeable = tradeable.copy()
    tradeable["Month"] = pd.to_datetime(tradeable["Alert_Date"]).dt.strftime("%Y-%m")
    print("\n  Monthly breakdown:")
    print(f"  {'Month':<9}  {'N':>4}  {'Win%':>6}  {'AvgPnL':>8}  {'Expectancy':>11}")
    print("  " + "-" * 42)
    for month, grp in sorted(tradeable.groupby("Month")):
        w3 = grp[grp["Is_Win"] == True]
        l3 = grp[grp["Is_Win"] == False]
        wr3 = len(w3) / len(grp)
        aw3 = w3["PnL_Pct"].mean() if not w3.empty else 0.0
        al3 = l3["PnL_Pct"].mean() if not l3.empty else 0.0
        print(f"  {month:<9}  {len(grp):>4}  {wr3:>5.1%}  "
              f"{grp['PnL_Pct'].mean():>+7.2%}  "
              f"{wr3*aw3+(1-wr3)*al3:>+10.2%}")

    # ── Chart ─────────────────────────────────────────────────────────────────
    tradeable_sorted = tradeable.sort_values("Alert_Date").reset_index(drop=True)
    tradeable_sorted["Cum_PnL"] = tradeable_sorted["PnL_Pct"].cumsum() * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"FRD Scanner 6-Month Simulation  ({sim_start} — {sim_end})\n"
        f"Filters: ${PRICE_MIN}-${PRICE_MAX} | 3d gain >={GAIN_3D_MIN:.0%} | "
        f"avg vol >={VOL_MIN/1e6:.0f}M | streak <={MAX_STREAK} | "
        f"vol ratio <{VOL_RATIO_LOW} or >={VOL_RATIO_HIGH}\n"
        f"{n} trades  |  Win rate {wr:.1%}  |  Expectancy {exp:+.2%}/trade",
        fontsize=10,
    )

    cum = tradeable_sorted["Cum_PnL"]
    ax1.plot(tradeable_sorted.index, cum, color="steelblue", linewidth=1.8)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.fill_between(tradeable_sorted.index, cum, 0,
                     where=(cum >= 0), alpha=0.15, color="green")
    ax1.fill_between(tradeable_sorted.index, cum, 0,
                     where=(cum < 0),  alpha=0.15, color="red")

    # Mark high-trash alerts
    ht = tradeable_sorted[tradeable_sorted["Trash_Score"] >= TRASH_THRESHOLD]
    if not ht.empty:
        ax1.scatter(ht.index, ht["Cum_PnL"], color="#c0392b", zorder=5,
                    s=60, label=f"High Trash ≥{TRASH_THRESHOLD}", marker="^")

    ax1.set_ylabel("Cumulative Return (%)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    bar_colors = []
    for _, r in tradeable_sorted.iterrows():
        if r["Stop_Hit"]:
            bar_colors.append("#e67e22")
        elif r["Is_Win"]:
            bar_colors.append("#27ae60")
        else:
            bar_colors.append("#e74c3c")

    ax2.bar(tradeable_sorted.index, tradeable_sorted["PnL_Pct"] * 100,
            color=bar_colors, alpha=0.85, width=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Trade P&L (%)")
    ax2.set_xlabel("Trade # (chronological)")
    ax2.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(color="#27ae60", label="Win"),
        Patch(color="#e74c3c", label="Loss"),
        Patch(color="#e67e22", label="Stopped out"),
    ], fontsize=8, loc="upper right")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "frd_6month_sim.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Chart saved -> {out}")


if __name__ == "__main__":
    main()

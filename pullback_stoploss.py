#!/usr/bin/env python3
"""
pullback_stoploss.py
====================
Stop loss + profit target optimization on the best pullback setup:
  mid/small cap | dn8-10% | above 21MA | first green candle | 20d max hold

Stop levels tested  : none, 5%, 7%, 8%, 10%, 12%, 15%
Profit targets tested: none, 10%, 15%, 20%, 25%
Full grid           : 7 stops x 5 targets = 35 combinations

Same-day conflict rule: if both stop and target are reached on the same bar,
assumes stop was hit first (conservative / realistic worst-case).
"""

import sys, io, pickle, warnings
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "poly_cache"
sys.path.insert(0, str(BASE_DIR))
from polygon_fetcher import _bdays
from etf_filter import get_etf_set, is_etf

DATA_START    = date(2022, 1, 1)
SIM_START     = date(2023, 1, 1)
SIM_END       = date(2025, 6, 30)
HOLD_DAYS     = 20
COOLDOWN_DAYS = 20

LARGE_MIN = 50_000_000
MID_MIN   =  5_000_000
SMALL_MIN =    500_000

STOP_LEVELS   = [None, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15]   # None = no stop
TARGET_LEVELS = [None, 0.10, 0.15, 0.20, 0.25]                # None = no target


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════════

def _rolling_mean(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    cs = np.cumsum(arr)
    out[n - 1:] = (cs[n - 1:] - np.concatenate([[0], cs[:-n]])) / n
    return out

def _rolling_max(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(n - 1, len(arr)):
        out[i] = arr[i - n + 1: i + 1].max()
    return out


def load_all_bars():
    end_buf = SIM_END + timedelta(days=HOLD_DAYS * 3)
    days_list = _bdays(DATA_START.isoformat(), end_buf.isoformat())
    bars: dict[str, dict[date, tuple]] = defaultdict(dict)
    loaded = []
    print(f"  Loading pkl: {DATA_START} -> {end_buf} ...", flush=True)
    for day in days_list:
        path = CACHE_DIR / f"grouped_{day}.pkl"
        if not path.exists():
            continue
        loaded.append(day)
        with open(path, "rb") as f:
            df = pickle.load(f)
        for row in df.itertuples(index=False):
            c, v = float(row.close), float(row.volume)
            if c <= 0 or v < 10_000:
                continue
            bars[row.ticker][day] = (
                float(row.open), float(row.high), float(row.low), c, float(row.volume)
            )
    print(f"  {len(loaded)} days, {len(bars):,} tickers.", flush=True)
    return bars, sorted(loaded)


def classify_universe(bars, all_days):
    sim_days = [d for d in all_days if SIM_START <= d <= SIM_END]
    ticker_dvol: dict[str, list[float]] = defaultdict(list)
    for day in sim_days:
        for tkr, dmap in bars.items():
            if day in dmap:
                c, v = dmap[day][3], dmap[day][4]
                ticker_dvol[tkr].append(c * v)
    umap: dict[str, str] = {}
    for tkr, vals in ticker_dvol.items():
        if len(vals) < 50:
            umap[tkr] = "other"; continue
        avg = float(np.mean(vals))
        if avg >= LARGE_MIN:
            umap[tkr] = "large"
        elif avg >= MID_MIN:
            umap[tkr] = "mid"
        elif avg >= SMALL_MIN:
            umap[tkr] = "small"
        else:
            umap[tkr] = "other"
    return umap


# ════════════════════════════════════════════════════════════════════════════════
# SIGNAL COLLECTION  — stores full forward OHLC for stop/target simulation
# ════════════════════════════════════════════════════════════════════════════════

def collect_signals(bars, all_days, umap):
    """
    Collect dn8-10% + above-21MA + green-candle signals for mid/small tickers.
    For each signal, stores:
      entry_open : D+1 open  (our entry price)
      d1_low/high: D+1 intraday range  (stop/target may hit on entry day)
      fwd_l/h   : lists of HOLD_DAYS daily lows/highs (D+2 .. D+HOLD+1)
      fwd_c_20  : close at end of HOLD_DAYS (D+HOLD+1)
      pb_dn10   : True if signal is also dn10 (subset of dn8)
    """
    all_days_idx = {d: i for i, d in enumerate(all_days)}
    all_days_set = set(all_days)
    sim_set      = {d for d in all_days if SIM_START <= d <= SIM_END}

    mid_small = [t for t, u in umap.items() if u in ("mid", "small")]
    records = []

    for ti, ticker in enumerate(mid_small):
        if (ti + 1) % 1000 == 0:
            print(f"  {ti+1:,}/{len(mid_small):,} tickers, {len(records):,} signals ...", flush=True)

        dmap = bars.get(ticker, {})
        tdates = sorted(d for d in dmap if d in all_days_set)
        if len(tdates) < 252:
            continue

        closes = np.array([dmap[d][3] for d in tdates])
        vols   = np.array([dmap[d][4] for d in tdates])

        ma21   = _rolling_mean(closes, 21)
        ma200  = _rolling_mean(closes, 200)
        avol   = _rolling_mean(vols, 20)
        high20 = _rolling_max(closes, 20)

        cooldown_end: date | None = None

        for i, sig_day in enumerate(tdates):
            if sig_day not in sim_set:
                continue
            if cooldown_end is not None and sig_day <= cooldown_end:
                continue
            if np.isnan(ma200[i]) or np.isnan(avol[i]) or np.isnan(high20[i]):
                continue

            c = closes[i]
            h20 = high20[i]
            if c <= 0 or h20 <= 0:
                continue

            # Pullback check
            pct_from_high = (h20 - c) / h20
            dn8  = pct_from_high >= 0.08
            if not dn8:
                continue
            dn10 = pct_from_high >= 0.10

            # Above 21MA quality filter
            if np.isnan(ma21[i]) or c < ma21[i]:
                continue

            # D+1 data
            d0_idx = all_days_idx.get(sig_day, -1)
            if d0_idx < 0 or d0_idx + 1 >= len(all_days):
                continue
            d1_day = all_days[d0_idx + 1]
            if d1_day not in dmap:
                continue

            d1o, d1h, d1l, d1c, _ = dmap[d1_day]
            if d1c <= d1o:          # must be green candle
                continue

            # Forward HOLD_DAYS of OHLC (D+2 .. D+HOLD+1)
            fwd_l, fwd_h, fwd_c = [], [], []
            for fd in range(1, HOLD_DAYS + 1):
                fwd_idx = d0_idx + fd + 1
                if fwd_idx >= len(all_days):
                    fwd_l.append(np.nan); fwd_h.append(np.nan); fwd_c.append(np.nan)
                    continue
                fday = all_days[fwd_idx]
                if fday in dmap:
                    fo, fh, fl, fc, _ = dmap[fday]
                    fwd_l.append(fl); fwd_h.append(fh); fwd_c.append(fc)
                else:
                    fwd_l.append(np.nan); fwd_h.append(np.nan); fwd_c.append(np.nan)

            # Require the time-exit close to be valid
            if len(fwd_c) < HOLD_DAYS or np.isnan(fwd_c[-1]):
                continue

            records.append({
                "ticker":      ticker,
                "signal_date": sig_day,
                "entry_open":  d1o,
                "d1_low":      d1l,
                "d1_high":     d1h,
                "fwd_l":       fwd_l,
                "fwd_h":       fwd_h,
                "fwd_c_last":  fwd_c[-1],
                "pb_dn10":     dn10,
            })
            cooldown_end = all_days[min(d0_idx + COOLDOWN_DAYS, len(all_days) - 1)]

    print(f"  Done. {len(records):,} signals collected.", flush=True)
    return records


# ════════════════════════════════════════════════════════════════════════════════
# VECTORIZED SIMULATION
# ════════════════════════════════════════════════════════════════════════════════

def build_arrays(sigs):
    """
    Convert list of signal dicts into numpy arrays for vectorized simulation.
    all_lows/highs have shape (n_sigs, HOLD_DAYS+1):
      col 0 = D+1 (entry day intraday range)
      col 1..HOLD_DAYS = D+2..D+HOLD+1
    """
    n = len(sigs)
    entries  = np.array([s["entry_open"] for s in sigs])
    fwd_last = np.array([s["fwd_c_last"] for s in sigs])

    all_lows  = np.full((n, HOLD_DAYS + 1), np.nan)
    all_highs = np.full((n, HOLD_DAYS + 1), np.nan)

    for i, s in enumerate(sigs):
        all_lows[i, 0]  = s["d1_low"]
        all_highs[i, 0] = s["d1_high"]
        for d, (fl, fh) in enumerate(zip(s["fwd_l"], s["fwd_h"]), start=1):
            all_lows[i, d]  = fl
            all_highs[i, d] = fh

    return entries, all_lows, all_highs, fwd_last


def simulate(entries, all_lows, all_highs, fwd_last, stop_pct, target_pct):
    """
    Vectorized simulation for one (stop_pct, target_pct) combination.
    Returns (pnl_arr, stopped_arr, targeted_arr) — all shape (n_valid,).
    """
    n = len(entries)
    INF = 1e10

    stop_px   = entries * (1.0 - stop_pct)   if stop_pct   else np.full(n, -INF)
    target_px = entries * (1.0 + target_pct) if target_pct else np.full(n,  INF)

    exited   = np.zeros(n, dtype=bool)
    exit_pnl = np.full(n, np.nan)
    stopped  = np.zeros(n, dtype=bool)
    targeted = np.zeros(n, dtype=bool)

    for d in range(HOLD_DAYS + 1):
        lows  = all_lows[:, d]
        highs = all_highs[:, d]
        valid_bar = ~np.isnan(lows) & ~np.isnan(highs)
        active = ~exited & valid_bar

        hit_stop   = active & (lows  <= stop_px)
        hit_target = active & (highs >= target_px)

        both       = hit_stop & hit_target
        stop_only  = hit_stop & ~hit_target
        tgt_only   = hit_target & ~hit_stop

        # Same-day conflict: stop assumed first
        exit_pnl[both]  = (stop_px[both]   - entries[both])  / entries[both]
        stopped[both]   = True;  exited[both] = True

        exit_pnl[stop_only] = (stop_px[stop_only] - entries[stop_only]) / entries[stop_only]
        stopped[stop_only]  = True;  exited[stop_only] = True

        exit_pnl[tgt_only] = (target_px[tgt_only] - entries[tgt_only]) / entries[tgt_only]
        targeted[tgt_only]  = True;  exited[tgt_only] = True

    # Time exit for any still open
    not_exited = ~exited & ~np.isnan(fwd_last)
    exit_pnl[not_exited] = (fwd_last[not_exited] - entries[not_exited]) / entries[not_exited]

    valid = ~np.isnan(exit_pnl)
    return exit_pnl[valid], stopped[valid], targeted[valid]


# ════════════════════════════════════════════════════════════════════════════════
# STATS + REPORTING
# ════════════════════════════════════════════════════════════════════════════════

def stats(pnl, stopped, targeted):
    n = len(pnl)
    if n == 0:
        return {}
    wins = pnl[pnl > 0];  losses = pnl[pnl <= 0]
    return {
        "n":        n,
        "wr":       float((pnl > 0).mean()),
        "exp":      float(pnl.mean()),
        "avg_w":    float(wins.mean())    if len(wins)   else 0.0,
        "avg_l":    float(losses.mean())  if len(losses) else 0.0,
        "stop_r":   float(stopped.mean()),
        "tgt_r":    float(targeted.mean()),
        "time_r":   float((~stopped & ~targeted).mean()),
    }


def stop_label(s):
    return "none" if s is None else f"{s:.0%}"

def tgt_label(t):
    return "none" if t is None else f"{t:.0%}"


def print_stop_table(rows, title):
    """Print the stop-only table (no target)."""
    W = 88
    print(f"\n{'='*W}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*W}", flush=True)
    print(f"  {'Stop':<8}  {'n':>6}  {'WR':>5}  {'AvgW':>7}  {'AvgL':>7}  {'Exp':>7}  {'StopR':>6}  {'TimeR':>6}", flush=True)
    print(f"  {'-'*(W-2)}", flush=True)
    for r in rows:
        print(f"  {r['stop']:<8}  {r['n']:>6}  {r['wr']:>4.0%}  "
              f"{r['avg_w']:>+6.1%}  {r['avg_l']:>+6.1%}  {r['exp']:>+6.2%}  "
              f"{r['stop_r']:>5.0%}  {r['time_r']:>5.0%}", flush=True)
    print(f"{'='*W}", flush=True)


def print_grid_table(rows, title, top_n=35):
    """Print the full stop x target grid sorted by exp."""
    rows = sorted(rows, key=lambda r: r["exp"], reverse=True)[:top_n]
    W = 100
    print(f"\n{'='*W}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*W}", flush=True)
    print(f"  {'Stop':<8}  {'Target':<8}  {'n':>6}  {'WR':>5}  {'AvgW':>7}  {'AvgL':>7}  "
          f"{'Exp':>7}  {'StopR':>6}  {'TgtR':>6}  {'TimeR':>6}", flush=True)
    print(f"  {'-'*(W-2)}", flush=True)
    for r in rows:
        print(f"  {r['stop']:<8}  {r['target']:<8}  {r['n']:>6}  {r['wr']:>4.0%}  "
              f"{r['avg_w']:>+6.1%}  {r['avg_l']:>+6.1%}  {r['exp']:>+6.2%}  "
              f"{r['stop_r']:>5.0%}  {r['tgt_r']:>5.0%}  {r['time_r']:>5.0%}", flush=True)
    print(f"{'='*W}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    SEP = "=" * 80
    print(SEP, flush=True)
    print("  Stop Loss + Profit Target Optimization", flush=True)
    print("  Setup: mid/small | dn8-10% | above 21MA | green candle | 20d max hold", flush=True)
    print(SEP, flush=True)
    print(flush=True)

    print("[1/4] Loading bars ...", flush=True)
    bars, all_days = load_all_bars()
    print(flush=True)

    print("[2/4] Classifying universe ...", flush=True)
    umap = classify_universe(bars, all_days)
    counts = {k: sum(1 for v in umap.values() if v == k) for k in ("large","mid","small","other")}
    print(f"  large={counts['large']:,}  mid={counts['mid']:,}  small={counts['small']:,}  other={counts['other']:,}", flush=True)
    etf_set = get_etf_set()
    before = len(umap)
    umap = {t: u for t, u in umap.items() if not is_etf(t, etf_set)}
    print(f"  ETF filter: {before - len(umap):,} tickers removed, {len(umap):,} remain.", flush=True)
    print(flush=True)

    print("[3/4] Scanning for dn8-10% + above-21MA + green-candle signals ...", flush=True)
    all_sigs = collect_signals(bars, all_days, umap)
    print(flush=True)

    if not all_sigs:
        print("  No signals found.", flush=True)
        return

    # Split into dn8 (includes dn10) and dn10 subsets
    dn8_sigs  = all_sigs
    dn10_sigs = [s for s in all_sigs if s["pb_dn10"]]
    print(f"  dn8 signals:  {len(dn8_sigs):,}", flush=True)
    print(f"  dn10 signals: {len(dn10_sigs):,}", flush=True)
    print(flush=True)

    print("[4/4] Running stop loss × profit target grid ...", flush=True)

    results = {}
    for label, sigs in [("dn8+above21MA", dn8_sigs), ("dn10+above21MA", dn10_sigs)]:
        entries, all_lows, all_highs, fwd_last = build_arrays(sigs)
        grid_rows = []
        stop_rows = []

        total = len(STOP_LEVELS) * len(TARGET_LEVELS)
        done = 0
        for stop_pct in STOP_LEVELS:
            for target_pct in TARGET_LEVELS:
                done += 1
                pnl, stopped_arr, targeted_arr = simulate(
                    entries, all_lows, all_highs, fwd_last, stop_pct, target_pct
                )
                s = stats(pnl, stopped_arr, targeted_arr)
                s["stop"]   = stop_label(stop_pct)
                s["target"] = tgt_label(target_pct)
                grid_rows.append(s)

                # Also track stop-only rows (no target)
                if target_pct is None:
                    stop_rows.append(s)

        results[label] = {"grid": grid_rows, "stop": stop_rows}
        print(f"  {label}: {total} combinations simulated.", flush=True)

    print(flush=True)

    # ── Report: Stop-only tables ────────────────────────────────────────────
    print_stop_table(results["dn8+above21MA"]["stop"],
                     "STOP LOSS ONLY — dn8+above21MA (no profit target)")
    print_stop_table(results["dn10+above21MA"]["stop"],
                     "STOP LOSS ONLY — dn10+above21MA (no profit target)")

    # ── Report: Full grid ───────────────────────────────────────────────────
    print_grid_table(results["dn8+above21MA"]["grid"],
                     "FULL GRID (stop x target), sorted by exp — dn8+above21MA")
    print_grid_table(results["dn10+above21MA"]["grid"],
                     "FULL GRID (stop x target), sorted by exp — dn10+above21MA")

    # ── Best combination summary ─────────────────────────────────────────────
    print(f"\n{'='*80}", flush=True)
    print("  BEST COMBINATIONS SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    for label in ["dn8+above21MA", "dn10+above21MA"]:
        grid = results[label]["grid"]
        best = max(grid, key=lambda r: r["exp"])
        best_no_tgt = max(results[label]["stop"], key=lambda r: r["exp"])
        print(f"\n  {label}:", flush=True)
        print(f"    Best overall  : stop={best['stop']:<6}  target={best['target']:<6}  "
              f"n={best['n']:,}  WR={best['wr']:.0%}  Exp={best['exp']:+.2%}  "
              f"StopR={best['stop_r']:.0%}  TgtR={best['tgt_r']:.0%}", flush=True)
        print(f"    Best stop-only: stop={best_no_tgt['stop']:<6}  "
              f"n={best_no_tgt['n']:,}  WR={best_no_tgt['wr']:.0%}  "
              f"Exp={best_no_tgt['exp']:+.2%}  StopR={best_no_tgt['stop_r']:.0%}", flush=True)
    print(flush=True)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    for label, data in results.items():
        fname = f"stoploss_grid_{label.replace('+','_')}.csv"
        pd.DataFrame(data["grid"]).to_csv(BASE_DIR / fname, index=False)
        print(f"  Saved {fname}", flush=True)

    print(flush=True)


if __name__ == "__main__":
    main()

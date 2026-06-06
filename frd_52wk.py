#!/usr/bin/env python3
"""
frd_52wk.py
===========
Strategy G analysis — two questions:
  1. Repeat offenders: tickers with 2+ signals in the 6-month window
  2. 52-week high filter: signal close must be >= 50% below the 52-week high
"""
import sys, io, pickle, time as _time
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "poly_cache"
HIST_DIR  = CACHE_DIR / "ticker_history"
HIST_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from polygon_fetcher import API_KEY, BASE_URL, _bdays

# ── Pool parameters (must match frd_final.py exactly) ───────────────────────
BASE_MIN_PRICE   = 2.0
BASE_MAX_PRICE   = 25.0
BASE_MIN_AVG_VOL = 300_000
BASE_MIN_3D_GAIN = 0.40
BASE_MAX_STREAK  = 6
BASE_HOD_FADE    = 0.03
LOOKBACK         = 20
SIM_DAYS         = 190

# Strategy G tight filters
G_STREAK    = 1
G_HOD       = 0.12
G_DOWN      = -0.10
G_GAIN      = 0.75
G_VOLR      = 0.30
G_STOP      = 0.15

# 52-week high filter
HIST_FROM        = "2024-06-01"   # need a full year back from Dec 2025 signals
MIN_52WK_DISC    = 0.50           # close must be >= 50% below 52-week high


# ── Polygon helper ────────────────────────────────────────────────────────────

def _get(path, params=None, retries=3):
    url = BASE_URL + path
    p   = {"apiKey": API_KEY, **(params or {})}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=p, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                _time.sleep(2 ** (attempt + 2))
                continue
            return {}
        except Exception:
            _time.sleep(2)
    return {}


def fetch_ticker_52wk(ticker):
    """Per-ticker daily bars from HIST_FROM to today, cached as {ticker}_52wk.pkl."""
    cache = HIST_DIR / f"{ticker}_52wk.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    today_str = date.today().isoformat()
    body = _get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{HIST_FROM}/{today_str}",
        {"adjusted": "true", "limit": 1000},
    )
    rows = []
    for b in body.get("results", []):
        rows.append({
            "date":  date.fromtimestamp(b["t"] // 1000),
            "high":  float(b["h"]),
            "close": float(b["c"]),
        })
    df = (pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
          if rows else pd.DataFrame(columns=["date", "high", "close"]))
    with open(cache, "wb") as f:
        pickle.dump(df, f)
    return df


def high_52wk_at(hist: pd.DataFrame, signal_date: date):
    """Max high in the 252 trading bars on or before signal_date, or None."""
    if hist.empty:
        return None
    prior = hist[hist["date"] <= signal_date]
    if prior.empty:
        return None
    return float(prior.tail(252)["high"].max())


# ── Bar loading (same logic as frd_final.py) ─────────────────────────────────

def load_bars():
    today = date.today()
    start = today - timedelta(days=SIM_DAYS + LOOKBACK + 15)
    days  = _bdays(start, today)
    bars  = defaultdict(dict)
    loaded = []
    for day in days:
        path = CACHE_DIR / f"grouped_{day}.pkl"
        if not path.exists():
            continue
        loaded.append(day)
        with open(path, "rb") as f:
            df = pickle.load(f)
        for row in df.itertuples(index=False):
            c = float(row.close)
            if not (BASE_MIN_PRICE <= c <= BASE_MAX_PRICE):
                continue
            vwap = float(row.vwap) if hasattr(row, "vwap") and pd.notna(row.vwap) else None
            bars[row.ticker][day] = (
                float(row.open), float(row.high), float(row.low),
                c, float(row.volume), vwap,
            )
    print(f"  {len(loaded)} days, {len(bars):,} tickers.", flush=True)
    return bars, loaded


def build_pool(bars, all_days):
    sim_start = LOOKBACK + 3
    day_set   = set(all_days)
    records   = []
    for ticker, dmap in bars.items():
        tdates = sorted(d for d in dmap if d in day_set)
        if len(tdates) < sim_start + 2:
            continue
        for di in range(sim_start, len(tdates)):
            sim_day = tdates[di]
            idx_all = all_days.index(sim_day) if sim_day in day_set else -1
            if idx_all < 0 or idx_all + 1 >= len(all_days):
                continue
            o, h, l, c, v, vwap = dmap[sim_day]
            prec       = tdates[:di]
            prev_close = dmap[prec[-1]][3]
            if c >= prev_close:
                continue
            pct_off_hod = (h - c) / h
            if pct_off_hod < BASE_HOD_FADE:
                continue
            vol_hist = [dmap[d][4] for d in prec[-LOOKBACK:]]
            avg_vol  = np.mean(vol_hist)
            if avg_vol < BASE_MIN_AVG_VOL:
                continue
            if len(prec) < 3:
                continue
            roll3 = (c - dmap[prec[-3]][3]) / dmap[prec[-3]][3]
            if roll3 < BASE_MIN_3D_GAIN:
                continue
            streak = 0
            for k in range(len(prec) - 1, max(len(prec) - 8, -1), -1):
                if k == 0:
                    break
                if dmap[prec[k]][3] > dmap[prec[k - 1]][3]:
                    streak += 1
                else:
                    break
            if streak > BASE_MAX_STREAK:
                continue
            vol_ratio   = v / avg_vol if avg_vol > 0 else 0.0
            pct_vs_prev = (c - prev_close) / prev_close
            next_day    = all_days[idx_all + 1]
            if next_day not in dmap:
                continue
            nd_o, nd_h, nd_l, nd_c, nd_v, _ = dmap[next_day]
            records.append(dict(
                ticker=ticker, date=sim_day, close=c, high=h,
                prev_close=prev_close,
                pct_off_hod=pct_off_hod, pct_vs_prev=pct_vs_prev,
                streak=streak, roll3_gain=roll3, vol_ratio=vol_ratio,
                nd_open=nd_o, nd_high=nd_h, nd_close=nd_c,
            ))
    return pd.DataFrame(records)


# ── Trade simulation ──────────────────────────────────────────────────────────

def sim(sigs, stop_pct=G_STOP):
    if sigs.empty:
        return dict(n=0, wr=0, exp=0, avg_w=0, avg_l=0, stop_r=0)
    s   = sigs.copy().sort_values("date").reset_index(drop=True)
    e   = s["nd_open"].values.astype(float)
    nh  = s["nd_high"].values.astype(float)
    nc  = s["nd_close"].values.astype(float)
    sp  = e * (1.0 + stop_pct)
    st  = nh >= sp
    ex  = np.where(st, sp, nc)
    pnl = (e - ex) / e
    wins = pnl > 0
    return dict(
        n=len(pnl), wr=wins.mean(),
        avg_w=pnl[wins].mean() if wins.any() else 0.0,
        avg_l=pnl[~wins].mean() if (~wins).any() else 0.0,
        exp=pnl.mean(), stop_r=st.mean(),
    )


def full_trades(label, sigs, col_52wk=False, stop_pct=G_STOP):
    bar = "=" * 72
    r   = sim(sigs, stop_pct)
    print(f"\n{bar}")
    print(f"  {label}")
    if r["n"] == 0:
        print("  No trades.")
        print(bar)
        return
    print(f"  n={r['n']}  WR={r['wr']:.0%}  AvgW={r['avg_w']:+.1%}  "
          f"AvgL={r['avg_l']:+.1%}  Exp={r['exp']:+.1%}  StopR={r['stop_r']:.0%}")
    print(bar)

    s   = sigs.copy().sort_values("date").reset_index(drop=True)
    e   = s["nd_open"].values.astype(float)
    nh  = s["nd_high"].values.astype(float)
    nc  = s["nd_close"].values.astype(float)
    sp  = e * (1.0 + stop_pct)
    st  = nh >= sp
    ex  = np.where(st, sp, nc)
    pnl = (e - ex) / e

    hdr = (f"  {'Date':>10}  {'Ticker':>6}  {'HOD%':>5}  {'Prev%':>6}  "
           f"{'3dG%':>5}  {'Str':>3}  {'VolR':>5}")
    if col_52wk:
        hdr += f"  {'52wkH':>7}  {'Disc':>6}"
    hdr += f"  {'Entry':>7}  {'Exit':>7}  {'PnL':>7}  WL"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for idx in range(len(s)):
        row  = s.iloc[idx]
        p    = pnl[idx]
        wl   = "W" if p > 0 else ("L*" if st[idx] else "L")
        line = (f"  {str(row['date']):>10}  {row['ticker']:>6}  "
                f"{row['pct_off_hod']:>4.0%}  {row['pct_vs_prev']:>+5.0%}  "
                f"{row['roll3_gain']:>4.0%}  {int(row['streak']):>3}  "
                f"{row['vol_ratio']:>5.2f}")
        if col_52wk:
            h52  = row.get("high_52wk") if hasattr(row, "get") else row["high_52wk"]
            disc = row.get("disc_52wk") if hasattr(row, "get") else row["disc_52wk"]
            h52s = f"${h52:>6.2f}" if pd.notna(h52) else "    N/A"
            ds   = f"{disc:>5.0%}"  if pd.notna(disc) else "   N/A"
            line += f"  {h52s}  {ds}"
        line += (f"  ${e[idx]:>6.2f}  ${ex[idx]:>6.2f}  {p:>+6.1%}  {wl}")
        print(line)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72, flush=True)
    print("  Strategy G — Repeat Offenders + 52-Week High Filter", flush=True)
    print("=" * 72, flush=True)

    print("\nLoading grouped daily bars ...", flush=True)
    bars, all_days = load_bars()
    pool = build_pool(bars, all_days)
    print(f"Base pool: {len(pool):,} signals\n", flush=True)

    # Apply Strategy G filters
    G = pool[
        (pool["streak"]      <= G_STREAK) &
        (pool["pct_off_hod"] >= G_HOD)    &
        (pool["pct_vs_prev"] <= G_DOWN)   &
        (pool["roll3_gain"]  >= G_GAIN)   &
        (pool["vol_ratio"]   >= G_VOLR)
    ].copy().sort_values("date").reset_index(drop=True)

    print(f"Strategy G baseline: {len(G)} signals, "
          f"{G['ticker'].nunique()} unique tickers\n", flush=True)

    # ── 1. Repeat offenders ───────────────────────────────────────────────────
    bar = "=" * 72
    print(bar, flush=True)
    print("  REPEAT OFFENDERS (2+ signals in 6-month window)", flush=True)
    print(bar, flush=True)

    counts  = G["ticker"].value_counts()
    repeats = counts[counts >= 2].sort_values(ascending=False)

    if repeats.empty:
        print("  None found.\n", flush=True)
    else:
        print(f"  {'Ticker':<8}  {'Signals':>7}  Dates", flush=True)
        print("  " + "-" * 52, flush=True)
        for tkr, cnt in repeats.items():
            dates_str = "  ".join(
                str(d) for d in sorted(G.loc[G["ticker"] == tkr, "date"])
            )
            print(f"  {tkr:<8}  {cnt:>7}      {dates_str}", flush=True)
        print(flush=True)

        repeat_sigs = G[G["ticker"].isin(repeats.index)]
        full_trades("Repeat offenders only — backtest stats", repeat_sigs)

    # ── 2. 52-week high filter ────────────────────────────────────────────────
    unique_tickers = G["ticker"].unique()
    print(f"\n{bar}", flush=True)
    print(f"  Fetching 52-week price history for {len(unique_tickers)} tickers "
          f"(from {HIST_FROM}) ...", flush=True)
    print(bar, flush=True)

    hist_map = {}
    for tkr in sorted(unique_tickers):
        hist = fetch_ticker_52wk(tkr)
        hist_map[tkr] = hist
        print(f"  {tkr}: {len(hist)} bars", flush=True)
        _time.sleep(0.12)

    # Compute 52-week high and discount for each signal
    G = G.copy()
    G["high_52wk"] = [
        high_52wk_at(hist_map.get(r["ticker"], pd.DataFrame()), r["date"])
        for _, r in G.iterrows()
    ]
    G["disc_52wk"] = G.apply(
        lambda r: (r["high_52wk"] - r["close"]) / r["high_52wk"]
        if pd.notna(r["high_52wk"]) and r["high_52wk"] > 0 else float("nan"),
        axis=1,
    )

    # Disposition table
    print(f"\n  Signal disposition vs 52-week high:", flush=True)
    print(f"  {'Date':>10}  {'Ticker':>6}  {'Close':>7}  {'52wkH':>8}  "
          f"{'Discount':>9}  {'>=50%?':>7}", flush=True)
    print("  " + "-" * 62, flush=True)
    for _, row in G.iterrows():
        disc = row["disc_52wk"]
        h52  = row["high_52wk"]
        if pd.isna(disc):
            disc_s = "    N/A"
            pass_s = "   N/A"
        else:
            disc_s = f"{disc:>7.1%}"
            pass_s = "   YES" if disc >= MIN_52WK_DISC else "    no"
        print(f"  {str(row['date']):>10}  {row['ticker']:>6}  "
              f"${row['close']:>6.2f}  ${h52:>7.2f}  {disc_s}  {pass_s}", flush=True)

    # Baseline (with 52wk columns populated)
    full_trades("Strategy G BASELINE", G, col_52wk=True)

    # With 52-week high filter
    G_52 = G[G["disc_52wk"] >= MIN_52WK_DISC]
    full_trades(
        f"Strategy G + close >= {MIN_52WK_DISC:.0%} below 52-week high",
        G_52, col_52wk=True,
    )

    # Comparison table
    r_base = sim(G)
    r_52   = sim(G_52)
    print(f"{'='*72}", flush=True)
    print(f"  FILTER COMPARISON", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  {'Variant':<42}  {'N':>4}  {'WR':>5}  {'AvgW':>6}  "
          f"{'AvgL':>6}  {'Exp':>6}  {'StpR':>5}", flush=True)
    print("  " + "-" * 72, flush=True)
    for label, r in [("G baseline", r_base), (f"G + >= {MIN_52WK_DISC:.0%} below 52wk high", r_52)]:
        if r["n"] == 0:
            print(f"  {label:<42}  {'--':>4}", flush=True)
        else:
            print(f"  {label:<42}  {r['n']:>4}  {r['wr']:>5.0%}  {r['avg_w']:>+6.1%}  "
                  f"{r['avg_l']:>+6.1%}  {r['exp']:>+6.1%}  {r['stop_r']:>5.0%}", flush=True)
    print(flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()

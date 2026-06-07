#!/usr/bin/env python3
"""
pullback_quality_filters.py
============================
Tests universe quality filters on the dn8 / dn10 + above-21MA + green-candle
+ 20d time-exit baseline (mid/small cap, 2023-2025).

Filters tested (individually):
  Q1  Market cap > $500M    (Polygon reference, fetched at cache time)
  Q2  Market cap > $1B
  Q3  Entry price > $10     (D+1 open, point-in-time)
  Q4  Entry price > $20
  Q5  Avg daily $ vol > $5M (20-bar avg at signal date, point-in-time)
  Q6  Avg daily $ vol > $10M
  Q7  S&P 500 constituent   (Wikipedia list, current membership)
  Q8  Russell 1000 approx   (iShares IWB holdings; fallback: mktcap > $2B)

Reports: n, WR, AvgW, AvgL, Exp, vs-baseline, coverage% for each filter × {dn8, dn10}.
Note: market cap is sourced from Polygon reference API at fetch time (not
      historical), and S&P 500 / Russell 1000 use current membership, so
      results carry mild survivorship bias for those two filters.
"""

import sys, io, pickle, warnings, time, requests
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "poly_cache"
FUND_DIR  = BASE_DIR / "fundamentals_cache"
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

INDEX_CACHE = FUND_DIR / "index_members.pkl"


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════════

def _rolling_mean(arr, n):
    out = np.full(len(arr), np.nan)
    cs = np.cumsum(arr)
    out[n - 1:] = (cs[n - 1:] - np.concatenate([[0], cs[:-n]])) / n
    return out

def _rolling_max(arr, n):
    out = np.full(len(arr), np.nan)
    for i in range(n - 1, len(arr)):
        out[i] = arr[i - n + 1: i + 1].max()
    return out


def load_all_bars():
    end_buf = SIM_END + timedelta(days=HOLD_DAYS * 3)
    days_list = _bdays(DATA_START.isoformat(), end_buf.isoformat())
    bars = defaultdict(dict)
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
    ticker_dvol = defaultdict(list)
    for day in sim_days:
        for tkr, dmap in bars.items():
            if day in dmap:
                c, v = dmap[day][3], dmap[day][4]
                ticker_dvol[tkr].append(c * v)
    umap = {}
    for tkr, vals in ticker_dvol.items():
        if len(vals) < 50:
            umap[tkr] = "other"; continue
        avg = float(np.mean(vals))
        if avg >= LARGE_MIN:    umap[tkr] = "large"
        elif avg >= MID_MIN:    umap[tkr] = "mid"
        elif avg >= SMALL_MIN:  umap[tkr] = "small"
        else:                   umap[tkr] = "other"
    return umap


# ════════════════════════════════════════════════════════════════════════════════
# MARKET CAP LOADING
# ════════════════════════════════════════════════════════════════════════════════

def load_market_caps(tickers) -> dict:
    """Load cached market_cap (in dollars) from fundamentals_cache per ticker."""
    caps = {}
    for t in tickers:
        path = FUND_DIR / f"{t}.pkl"
        if not path.exists():
            continue
        try:
            d = pickle.load(open(path, "rb"))
            mc = d.get("market_cap")
            if mc and mc > 0:
                caps[t] = float(mc)
        except Exception:
            pass
    return caps


# ════════════════════════════════════════════════════════════════════════════════
# INDEX MEMBERSHIP
# ════════════════════════════════════════════════════════════════════════════════

def _fetch_sp500() -> set:
    """Fetch current S&P 500 constituents from Wikipedia."""
    print("  Fetching S&P 500 from Wikipedia ...", flush=True)
    try:
        from io import StringIO
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=30,
        )
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text), attrs={"id": "constituents"})
        tickers = set(
            tables[0]["Symbol"].str.strip().str.replace(".", "-", regex=False).tolist()
        )
        print(f"  S&P 500: {len(tickers)} tickers.", flush=True)
        return tickers
    except Exception as exc:
        print(f"  WARNING: S&P 500 fetch failed ({exc}). Using empty set.", flush=True)
        return set()


def _build_russell1000_proxy(market_caps: dict) -> set:
    """
    Russell 1000 proxy: top 1000 tickers by market cap from fundamentals cache.
    Polygon's reference API does not support sorting by market_cap, so we use
    the market_cap values already fetched into fundamentals_cache/.
    """
    sorted_by_cap = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
    r1000 = {t for t, _ in sorted_by_cap[:1000]}
    print(f"  Russell 1000 proxy (top 1000 by cached mktcap): {len(r1000)} tickers.", flush=True)
    return r1000


def load_index_members(market_caps: dict, refresh: bool = False):
    """Return (sp500_set, r1000_set). Cached in fundamentals_cache/index_members.pkl."""
    if not refresh and INDEX_CACHE.exists():
        with open(INDEX_CACHE, "rb") as f:
            d = pickle.load(f)
        print(f"  Index cache loaded: S&P500={len(d['sp500']):,}  R1000={len(d['r1000']):,}", flush=True)
        return d["sp500"], d["r1000"]

    sp500 = _fetch_sp500()
    r1000 = _build_russell1000_proxy(market_caps)

    FUND_DIR.mkdir(exist_ok=True)
    with open(INDEX_CACHE, "wb") as f:
        pickle.dump({"sp500": sp500, "r1000": r1000}, f, protocol=4)
    return sp500, r1000


# ════════════════════════════════════════════════════════════════════════════════
# SIGNAL COLLECTION — stores quality metadata per signal
# ════════════════════════════════════════════════════════════════════════════════

def collect_signals(bars, all_days, umap, market_caps):
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
        avol20 = _rolling_mean(vols,   20)
        high20 = _rolling_max(closes,  20)
        # 20-day dollar volume rolling mean
        dvol20 = _rolling_mean(closes * vols, 20)

        mktcap = market_caps.get(ticker)   # may be None

        cooldown_end = None

        for i, sig_day in enumerate(tdates):
            if sig_day not in sim_set:
                continue
            if cooldown_end is not None and sig_day <= cooldown_end:
                continue
            if np.isnan(ma200[i]) or np.isnan(avol20[i]) or np.isnan(high20[i]):
                continue

            c   = closes[i]
            h20 = high20[i]
            if c <= 0 or h20 <= 0:
                continue

            pct_off = (h20 - c) / h20
            dn8  = pct_off >= 0.08
            if not dn8:
                continue
            dn10 = pct_off >= 0.10

            if np.isnan(ma21[i]) or c < ma21[i]:
                continue

            d0_idx = all_days_idx.get(sig_day, -1)
            if d0_idx < 0 or d0_idx + 1 >= len(all_days):
                continue
            d1_day = all_days[d0_idx + 1]
            if d1_day not in dmap:
                continue

            d1o, _, _, d1c, _ = dmap[d1_day]
            if d1c <= d1o:
                continue

            # 20-day forward time exit
            fwd_idx = d0_idx + HOLD_DAYS + 1
            if fwd_idx >= len(all_days):
                continue
            fday = all_days[fwd_idx]
            if fday not in dmap:
                continue
            exit_close = dmap[fday][3]
            if np.isnan(exit_close) or exit_close <= 0:
                continue

            pnl = (exit_close - d1o) / d1o

            records.append({
                "ticker":      ticker,
                "signal_date": sig_day,
                "entry_open":  d1o,
                "avg_dvol":    float(dvol20[i]) if not np.isnan(dvol20[i]) else 0.0,
                "mktcap":      mktcap,
                "pb_dn10":     dn10,
                "pnl":         pnl,
            })
            cooldown_end = all_days[min(d0_idx + COOLDOWN_DAYS, len(all_days) - 1)]

    print(f"  Done. {len(records):,} signals collected.", flush=True)
    return records


# ════════════════════════════════════════════════════════════════════════════════
# STATS + REPORTING
# ════════════════════════════════════════════════════════════════════════════════

def stats(pnl_arr):
    n = len(pnl_arr)
    if n == 0:
        return {"n": 0, "wr": 0, "exp": 0, "avg_w": 0, "avg_l": 0}
    wins   = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr <= 0]
    return {
        "n":     n,
        "wr":    float((pnl_arr > 0).mean()),
        "exp":   float(pnl_arr.mean()),
        "avg_w": float(wins.mean())   if len(wins)   else 0.0,
        "avg_l": float(losses.mean()) if len(losses) else 0.0,
    }


def run_filters(sigs, sp500, r1000, label):
    """Run all quality filters on a list of signal dicts, print comparison table."""
    total = len(sigs)
    if total == 0:
        print(f"  No signals for {label}.", flush=True)
        return

    # Base PnL array (baseline = all signals)
    base_pnl  = np.array([s["pnl"] for s in sigs])
    base_stat = stats(base_pnl)

    # Define filters
    filters = [
        ("Baseline",            lambda s: True),
        ("Mkt cap > $500M",     lambda s: s["mktcap"] is not None and s["mktcap"] >= 500e6),
        ("Mkt cap > $1B",       lambda s: s["mktcap"] is not None and s["mktcap"] >= 1e9),
        ("Price > $10",         lambda s: s["entry_open"] >= 10),
        ("Price > $20",         lambda s: s["entry_open"] >= 20),
        ("Avg $vol > $5M",      lambda s: s["avg_dvol"] >= 5e6),
        ("Avg $vol > $10M",     lambda s: s["avg_dvol"] >= 10e6),
        ("S&P 500",             lambda s: s["ticker"] in sp500),
        ("Russell 1000",        lambda s: s["ticker"] in r1000),
    ]

    rows = []
    for fname, ftest in filters:
        subset  = [s for s in sigs if ftest(s)]
        pnl_arr = np.array([s["pnl"] for s in subset])
        st      = stats(pnl_arr)
        coverage = len(subset) / total if total else 0
        vs_base  = st["exp"] - base_stat["exp"] if st["n"] > 0 else float("nan")
        rows.append({
            "filter":   fname,
            "n":        st["n"],
            "wr":       st["wr"],
            "avg_w":    st["avg_w"],
            "avg_l":    st["avg_l"],
            "exp":      st["exp"],
            "vs_base":  vs_base,
            "coverage": coverage,
        })

    # Print table
    W = 108
    print(f"\n{'='*W}", flush=True)
    print(f"  UNIVERSE QUALITY FILTER STUDY — {label}", flush=True)
    print(f"{'='*W}", flush=True)
    print(
        f"  {'Filter':<22}  {'n':>6}  {'WR':>5}  {'AvgW':>7}  {'AvgL':>7}  "
        f"{'Exp':>7}  {'vs Base':>8}  {'Coverage':>9}",
        flush=True,
    )
    print(f"  {'-'*(W-2)}", flush=True)
    for r in rows:
        vs = f"{r['vs_base']:>+7.2%}" if not np.isnan(r["vs_base"]) else "      —"
        print(
            f"  {r['filter']:<22}  {r['n']:>6}  {r['wr']:>4.0%}  "
            f"{r['avg_w']:>+6.1%}  {r['avg_l']:>+6.1%}  {r['exp']:>+6.2%}  "
            f"{vs}  {r['coverage']:>8.0%}",
            flush=True,
        )
    print(f"{'='*W}", flush=True)

    return rows


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    SEP = "=" * 80
    print(SEP, flush=True)
    print("  Universe Quality Filter Study", flush=True)
    print("  Base: mid/small | dn8/dn10 | above-21MA | green candle | 20d time exit", flush=True)
    print(SEP, flush=True)
    print(flush=True)

    print("[1/6] Loading bars ...", flush=True)
    bars, all_days = load_all_bars()
    print(flush=True)

    print("[2/6] Classifying universe + ETF filter ...", flush=True)
    umap = classify_universe(bars, all_days)
    counts = {k: sum(1 for v in umap.values() if v == k)
              for k in ("large", "mid", "small", "other")}
    print(f"  large={counts['large']:,}  mid={counts['mid']:,}  "
          f"small={counts['small']:,}  other={counts['other']:,}", flush=True)
    etf_set = get_etf_set()
    before  = len(umap)
    umap    = {t: u for t, u in umap.items() if not is_etf(t, etf_set)}
    print(f"  ETF filter: {before - len(umap):,} removed, {len(umap):,} remain.", flush=True)
    print(flush=True)

    print("[3/6] Loading market caps from fundamentals cache ...", flush=True)
    all_tickers = list(umap.keys())
    market_caps = load_market_caps(all_tickers)
    covered     = sum(1 for t in all_tickers if t in market_caps)
    print(f"  Market cap data found for {covered:,} / {len(all_tickers):,} tickers.", flush=True)
    print(flush=True)

    print("[4/6] Loading index constituent lists ...", flush=True)
    sp500, r1000 = load_index_members(market_caps)
    print(flush=True)

    print("[5/6] Collecting signals ...", flush=True)
    all_sigs  = collect_signals(bars, all_days, umap, market_caps)
    dn8_sigs  = all_sigs
    dn10_sigs = [s for s in all_sigs if s["pb_dn10"]]
    print(f"  dn8:  {len(dn8_sigs):,}  |  dn10: {len(dn10_sigs):,}", flush=True)
    print(flush=True)

    print("[6/6] Running quality filter grid ...", flush=True)

    rows_dn8  = run_filters(dn8_sigs,  sp500, r1000, "dn8+above21MA")
    rows_dn10 = run_filters(dn10_sigs, sp500, r1000, "dn10+above21MA")

    # Summary: best filter per setup
    print(f"\n{SEP}", flush=True)
    print("  BEST QUALITY FILTERS (by expectancy)", flush=True)
    print(SEP, flush=True)
    for label, rows in [("dn8+above21MA", rows_dn8), ("dn10+above21MA", rows_dn10)]:
        if not rows:
            continue
        non_base = [r for r in rows if r["filter"] != "Baseline" and r["n"] >= 50]
        if not non_base:
            continue
        best = max(non_base, key=lambda r: r["exp"])
        print(f"  {label}:  {best['filter']:<22}  n={best['n']:,}  "
              f"WR={best['wr']:.0%}  Exp={best['exp']:+.2%}  "
              f"vs baseline={best['vs_base']:+.2%}  Coverage={best['coverage']:.0%}", flush=True)
    print(SEP, flush=True)

    # Save to CSV
    if rows_dn8:
        pd.DataFrame(rows_dn8).to_csv("quality_filters_dn8.csv", index=False)
        pd.DataFrame(rows_dn10).to_csv("quality_filters_dn10.csv", index=False)
        print("  Saved: quality_filters_dn8.csv, quality_filters_dn10.csv", flush=True)


if __name__ == "__main__":
    main()

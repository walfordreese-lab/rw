#!/usr/bin/env python3
"""
Strategy G v2 — adds 200-day MA and minimum-float filters.

New rules on top of Strategy G baseline:
  1. Close on signal day must be BELOW the 200-day simple moving average.
  2. Shares outstanding (proxy for float) must be >= MIN_FLOAT shares.

Per-ticker price history is fetched once from Polygon and cached locally.
Shares outstanding is fetched from the Polygon reference API.
"""
import sys, io, pickle, time, json
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

import requests
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR   = Path(__file__).parent
CACHE_DIR  = BASE_DIR / "poly_cache"
EXTRA_DIR  = BASE_DIR / "poly_cache" / "ticker_history"
EXTRA_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE_DIR))
from polygon_fetcher import _bdays, API_KEY, BASE_URL, _get

# ── new filter constants ───────────────────────────────────────────────────────
MA_PERIOD      = 200          # 200-day SMA
HIST_FROM      = "2025-01-01" # far enough back for 200d MA at Dec 2025 signals
MIN_FLOAT      = 10_000_000   # minimum shares outstanding (float proxy)

# ── base pool (same as frd_final.py) ──────────────────────────────────────────
BASE_MIN_PRICE   = 2.0
BASE_MAX_PRICE   = 25.0
BASE_MIN_AVG_VOL = 300_000
BASE_MIN_3D_GAIN = 0.40
BASE_MAX_STREAK  = 6
BASE_HOD_FADE    = 0.03
LOOKBACK         = 20
SIM_DAYS         = 190

# ── Strategy G filters ─────────────────────────────────────────────────────────
G_STREAK_MAX  = 1
G_HOD_MIN     = 0.12
G_PREV_MIN    = -0.10
G_GAIN_MIN    = 0.75
G_VOLR_MIN    = 0.30
STOP_PCT      = 0.15


# ── helpers ───────────────────────────────────────────────────────────────────

def load_bars():
    today = date.today()
    start = today - timedelta(days=SIM_DAYS + LOOKBACK + 15)
    days  = _bdays(start, today)
    bars  = defaultdict(dict)
    loaded = []
    for i, day in enumerate(days):
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
    print(f"  {len(loaded)} days, {len(bars)} tickers.", flush=True)
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
                if dmap[prec[k]][3] > dmap[prec[k-1]][3]:
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
                ticker=ticker, date=sim_day, close=c,
                pct_off_hod=pct_off_hod, pct_vs_prev=pct_vs_prev,
                streak=streak, roll3_gain=roll3, vol_ratio=vol_ratio,
                nd_open=nd_o, nd_high=nd_h, nd_close=nd_c,
            ))
    return pd.DataFrame(records)


def apply_g(sigs):
    return sigs[
        (sigs["streak"]      <= G_STREAK_MAX) &
        (sigs["pct_off_hod"] >= G_HOD_MIN)    &
        (sigs["pct_vs_prev"] <= G_PREV_MIN)   &
        (sigs["roll3_gain"]  >= G_GAIN_MIN)   &
        (sigs["vol_ratio"]   >= G_VOLR_MIN)
    ].copy()


# ── Polygon per-ticker price history ─────────────────────────────────────────

def fetch_ticker_history(ticker: str) -> pd.Series:
    """Return a date-indexed Series of daily close prices from HIST_FROM to today."""
    cache_path = EXTRA_DIR / f"{ticker}_hist.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    today_str = date.today().isoformat()
    url  = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{HIST_FROM}/{today_str}"
    p    = {"adjusted": "true", "limit": 50000, "apiKey": API_KEY}
    try:
        r = requests.get(url, params=p, timeout=30)
        time.sleep(0.15)
        if r.status_code != 200:
            print(f"    {ticker}: HTTP {r.status_code}", flush=True)
            return pd.Series(dtype=float)
        data = r.json().get("results", [])
    except Exception as e:
        print(f"    {ticker}: error {e}", flush=True)
        return pd.Series(dtype=float)

    if not data:
        with open(cache_path, "wb") as f:
            pickle.dump(pd.Series(dtype=float), f)
        return pd.Series(dtype=float)

    rows = [(date.fromtimestamp(bar["t"] / 1000), bar["c"]) for bar in data]
    s = pd.Series({d: c for d, c in rows}, name=ticker)
    s.index = pd.to_datetime(s.index)
    with open(cache_path, "wb") as f:
        pickle.dump(s, f)
    return s


def ma200_at(hist: pd.Series, signal_date) -> float | None:
    """Return 200-day SMA of close prices ending at signal_date, or None."""
    ts = pd.Timestamp(signal_date)
    past = hist[hist.index <= ts]
    if len(past) < MA_PERIOD:
        return None
    return float(past.iloc[-MA_PERIOD:].mean())


# ── Polygon reference — shares outstanding ────────────────────────────────────

_shares_cache: dict = {}

def fetch_shares_outstanding(ticker: str) -> int | None:
    """Return share_class_shares_outstanding from Polygon reference API, or None."""
    if ticker in _shares_cache:
        return _shares_cache[ticker]

    cache_path = EXTRA_DIR / f"{ticker}_ref.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            val = pickle.load(f)
        _shares_cache[ticker] = val
        return val

    data = _get(f"/v3/reference/tickers/{ticker}")
    time.sleep(0.15)
    res  = data.get("results", {})
    val  = res.get("share_class_shares_outstanding") or res.get("weighted_shares_outstanding")
    val  = int(val) if val else None

    _shares_cache[ticker] = val
    with open(cache_path, "wb") as f:
        pickle.dump(val, f)
    return val


# ── trade simulation ──────────────────────────────────────────────────────────

def sim(sigs, stop_pct=STOP_PCT):
    if len(sigs) == 0:
        return dict(n=0, wr=0, exp=0, avg_w=0, avg_l=0, stop_r=0)
    e   = sigs["nd_open"].values.astype(float)
    nh  = sigs["nd_high"].values.astype(float)
    nc  = sigs["nd_close"].values.astype(float)
    sp  = e * (1.0 + stop_pct)
    stp = nh >= sp
    ex  = np.where(stp, sp, nc)
    pnl = (e - ex) / e
    n, wins = len(pnl), int((pnl > 0).sum())
    return dict(n=n, wr=wins/n if n else 0,
                exp=float(pnl.mean()) if n else 0,
                avg_w=float(pnl[pnl>0].mean()) if wins else 0,
                avg_l=float(pnl[pnl<=0].mean()) if n-wins else 0,
                stop_r=float(stp.mean()))


def print_trades(label, sigs):
    if sigs.empty:
        print(f"\n  {label}: no trades", flush=True)
        return
    # Sort first so that array positions match display order
    s   = sigs.copy().sort_values("date").reset_index(drop=True)
    e   = s["nd_open"].values.astype(float)
    nh  = s["nd_high"].values.astype(float)
    nc  = s["nd_close"].values.astype(float)
    sp  = e * (1.0 + STOP_PCT)
    stp = nh >= sp
    ex  = np.where(stp, sp, nc)
    pnl = (e - ex) / e

    s["entry"] = np.round(e, 2)
    s["exit"]  = np.round(ex, 2)
    s["pnl"]   = np.round(pnl * 100, 1)
    s["W/L"]   = np.where(pnl > 0, "W", "L")
    s["STOP"]  = np.where(stp, "*", " ")

    st = sim(sigs)
    print(f"\n{'='*72}", flush=True)
    print(f"  {label}", flush=True)
    print(f"  n={st['n']}  WR={st['wr']:.0%}  AvgW={st['avg_w']:+.1%}  "
          f"AvgL={st['avg_l']:+.1%}  Exp={st['exp']:+.1%}  StopR={st['stop_r']:.0%}", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  {'Date':>12}  {'Ticker':>6}  {'HOD%':>5}  {'Prev%':>6}  "
          f"{'3dG%':>5}  {'Str':>3}  {'VolR':>5}  "
          f"{'MA200':>8}  {'Close':>6}  "
          f"{'Float':>10}  {'Entry':>6}  {'Exit':>6}  {'PnL':>6}  WL", flush=True)
    print(f"  {'-'*90}", flush=True)

    for _, r in s.iterrows():
        ma_str    = f"{r['ma200']:.2f}" if pd.notna(r.get("ma200")) else "   n/a"
        float_str = f"{int(r['shares_out'])/1e6:.1f}M" if pd.notna(r.get("shares_out")) and r["shares_out"] else "  n/a"
        blow_str  = "<MA" if r.get("below_ma200") else "   "
        print(f"  {str(r['date']):>12}  {r['ticker']:>6}  "
              f"{r['pct_off_hod']:>5.0%}  {r['pct_vs_prev']:>6.0%}  "
              f"{r['roll3_gain']:>5.0%}  {int(r['streak']):>3}  {r['vol_ratio']:>5.2f}  "
              f"{ma_str:>8}  {r['close']:>6.2f}  "
              f"{float_str:>10}  {r['entry']:>6.2f}  {r['exit']:>6.2f}  "
              f"{r['pnl']:>+6.1f}%  {r['W/L']}{r['STOP']}", flush=True)


def main():
    print("=" * 72, flush=True)
    print("  Strategy G v2 — with 200d MA + Float filters", flush=True)
    print(f"  200d MA period : {MA_PERIOD} days  |  Min float : {MIN_FLOAT/1e6:.0f}M shares", flush=True)
    print("=" * 72, flush=True)

    # ── 1. Build signal pool ──────────────────────────────────────────────────
    print("\nLoading grouped daily bars…", flush=True)
    bars, all_days = load_bars()
    sigs   = build_pool(bars, all_days)
    g_base = apply_g(sigs)
    print(f"\nStrategy G baseline: {len(g_base)} signals, "
          f"{g_base['ticker'].nunique()} unique tickers", flush=True)

    # ── 2. Fetch per-ticker price history for 200d MA ─────────────────────────
    tickers = sorted(g_base["ticker"].unique())
    print(f"\nFetching 200d price history for {len(tickers)} tickers (HIST_FROM={HIST_FROM})…", flush=True)
    hist_by_ticker: dict[str, pd.Series] = {}
    for tkr in tickers:
        hist_by_ticker[tkr] = fetch_ticker_history(tkr)
        cached = len(hist_by_ticker[tkr])
        print(f"  {tkr}: {cached} bars", flush=True)

    # ── 3. Fetch shares outstanding ───────────────────────────────────────────
    print(f"\nFetching shares outstanding for {len(tickers)} tickers…", flush=True)
    shares_by_ticker: dict[str, int | None] = {}
    for tkr in tickers:
        shares_by_ticker[tkr] = fetch_shares_outstanding(tkr)
        s = shares_by_ticker[tkr]
        s_str = f"{s/1e6:.1f}M" if s else "n/a"
        flag  = "  OK" if s and s >= MIN_FLOAT else "  FAIL (<10M)" if s else "  FAIL (no data)"
        print(f"  {tkr}: {s_str}{flag}", flush=True)

    # ── 4. Add new columns to signal df ──────────────────────────────────────
    def row_ma200(r):
        hist = hist_by_ticker.get(r["ticker"], pd.Series(dtype=float))
        return ma200_at(hist, r["date"])

    g_base["ma200"]      = g_base.apply(row_ma200, axis=1)
    g_base["below_ma200"]= g_base["ma200"].notna() & (g_base["close"] < g_base["ma200"])
    g_base["shares_out"] = g_base["ticker"].map(shares_by_ticker)
    g_base["float_ok"]   = g_base["shares_out"].apply(
        lambda x: x is not None and x >= MIN_FLOAT
    )

    # ── 5. Print the full G baseline with new column values ──────────────────
    print_trades("Strategy G BASELINE (no new filters)", g_base)

    # ── 6. Apply 200d MA filter ───────────────────────────────────────────────
    g_ma = g_base[g_base["below_ma200"]].copy()
    print_trades("Strategy G + below 200d MA", g_ma)

    # ── 7. Apply float filter ─────────────────────────────────────────────────
    g_float = g_base[g_base["float_ok"]].copy()
    print_trades("Strategy G + float >= 10M", g_float)

    # ── 8. Both filters combined ──────────────────────────────────────────────
    g_both = g_base[g_base["below_ma200"] & g_base["float_ok"]].copy()
    print_trades("Strategy G + BOTH filters (below 200d MA AND float >= 10M)", g_both)

    # ── 9. Summary table ──────────────────────────────────────────────────────
    variants = [
        ("G baseline",          g_base),
        ("G + below 200d MA",   g_ma),
        ("G + float >= 10M",    g_float),
        ("G + both filters",    g_both),
    ]
    print("\n\n" + "="*72, flush=True)
    print("  FILTER COMPARISON", flush=True)
    print("="*72, flush=True)
    print(f"  {'Variant':<28}  {'N':>4}  {'WR':>6}  {'AvgW':>7}  {'AvgL':>7}  {'Exp':>7}  {'StpR':>6}", flush=True)
    print(f"  {'-'*70}", flush=True)
    for label, sg in variants:
        st = sim(sg)
        if st["n"] == 0:
            print(f"  {label:<28}  {'0':>4}  {'--':>6}", flush=True)
            continue
        print(f"  {label:<28}  {st['n']:>4}  {st['wr']:>6.0%}  {st['avg_w']:>7.1%}  "
              f"{st['avg_l']:>7.1%}  {st['exp']:>7.1%}  {st['stop_r']:>6.0%}", flush=True)

    # ── 10. Which signals were filtered out and why ───────────────────────────
    print("\n\n  Signal disposition:", flush=True)
    print(f"  {'Date':>12}  {'Ticker':>6}  {'below_MA':>9}  {'float_ok':>9}  {'passes':>7}", flush=True)
    print(f"  {'-'*50}", flush=True)
    for _, r in g_base.sort_values("date").iterrows():
        bma   = "YES" if r["below_ma200"] else "no "
        flok  = "YES" if r["float_ok"]   else "no "
        both  = "KEEP" if r["below_ma200"] and r["float_ok"] else "drop"
        so_str = f"{r['shares_out']/1e6:.1f}M" if r["shares_out"] else "n/a"
        ma_str = f"{r['ma200']:.2f}" if pd.notna(r.get("ma200")) else "n/a"
        print(f"  {str(r['date']):>12}  {r['ticker']:>6}  "
              f"{'close='+str(round(r['close'],2))+' MA='+ma_str:>20}  "
              f"{so_str:>9}  "
              f"bMA={bma}  flt={flok}  -> {both}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

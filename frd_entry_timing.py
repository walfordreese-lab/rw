#!/usr/bin/env python3
"""
frd_entry_timing.py
===================
FRD Short Entry Timing Study — Strategy G universe (same 7 filters)

Tests 6 entry timing variants on the next-day (D+1) trade using 5-min bars.

Variant  Entry condition
───────────────────────────────────────────────────────────────────────────
V1       Short at market open (no waiting, entry = D+1 open)
V2       First red 5-min candle opening before 9:45 AM ET
V3       First red 5-min candle opening before 10:00 AM ET
V4       First red 5-min candle opening before 10:30 AM ET
T1       HOD frozen after first 90 min (9:30-11:00); first red 5-min after
T2       No new high after 10 AM; first 5-min close below the 10 AM candle

All variants share:
  - 15% hard stop above entry
  - EOD exit (last available 5-min bar close)
  - Same Strategy G universe (7 daily-bar filters)

Reports per variant:
  n | no-trigger | win rate | expectancy | avg hold | stop-out rate | avg entry vs HOD%
"""
import sys
import io
import pickle
import time as _time
from datetime import date, timedelta, datetime, time as dt_time
from pathlib import Path
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR     = Path(__file__).parent
CACHE_DIR    = BASE_DIR / "poly_cache"
INTRADAY_DIR = BASE_DIR / "intraday_cache"

sys.path.insert(0, str(BASE_DIR))
from polygon_fetcher import _bdays, API_KEY, BASE_URL, fetch_grouped_day

# ── Strategy G parameters ─────────────────────────────────────────────────────
STOP_PCT       = 0.15
BASE_MIN_PRICE = 2.0
BASE_MAX_PRICE = 25.0
BASE_MIN_VOL   = 300_000
BASE_HOD_FADE  = 0.03
LOOKBACK       = 20
SIM_DAYS       = 190

G_STREAK = 1
G_HOD    = 0.12
G_PREV   = -0.10
G_GAIN   = 0.75
G_VOLR   = 0.30

ET      = ZoneInfo("America/New_York")
T_OPEN  = dt_time(9, 30)
T_9_45  = dt_time(9, 45)
T_10    = dt_time(10, 0)
T_10_30 = dt_time(10, 30)
T_11    = dt_time(11, 0)
T_CLOSE = dt_time(16, 0)

REQUEST_DELAY = 0.13


# ── Daily bars: load from cache, fetch from Polygon if missing ────────────────

def load_bars():
    today = date.today()
    start = today - timedelta(days=SIM_DAYS + LOOKBACK + 15)
    all_days = _bdays(start, today)

    CACHE_DIR.mkdir(exist_ok=True)
    bars    = defaultdict(dict)
    loaded  = []
    missing = [d for d in all_days if not (CACHE_DIR / f"grouped_{d}.pkl").exists()]

    if missing:
        print(f"  Fetching {len(missing)} uncached days from Polygon …", flush=True)

    for i, day in enumerate(all_days):
        if i % 30 == 0:
            print(f"    {i}/{len(all_days)}  {day}", flush=True)
        path = CACHE_DIR / f"grouped_{day}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                df = pickle.load(fh)
        else:
            df = fetch_grouped_day(day)

        if df.empty:
            continue
        loaded.append(day)
        for row in df.itertuples(index=False):
            c = float(row.close)
            if not (BASE_MIN_PRICE <= c <= BASE_MAX_PRICE):
                continue
            vwap = (float(row.vwap)
                    if hasattr(row, "vwap") and pd.notna(row.vwap) else None)
            bars[row.ticker][day] = (
                float(row.open), float(row.high), float(row.low),
                c, float(row.volume), vwap,
            )

    print(f"  Loaded {len(loaded)} days, {len(bars)} tickers", flush=True)
    return bars, all_days


# ── Build Strategy G signal pool (identical to frd_final.py) ─────────────────

def build_signals(bars, all_days):
    day_set   = set(all_days)
    sim_start = LOOKBACK + 3
    records   = []

    for ticker, dmap in bars.items():
        tdates = sorted(d for d in dmap if d in day_set)
        if len(tdates) < sim_start + 2:
            continue

        for di in range(sim_start, len(tdates)):
            d       = tdates[di]
            idx_all = all_days.index(d) if d in day_set else -1
            if idx_all < 0 or idx_all + 1 >= len(all_days):
                continue

            o, h, l, c, v, _ = dmap[d]
            prec               = tdates[:di]
            prev_c             = dmap[prec[-1]][3]
            if c >= prev_c:
                continue

            pct_off_hod = (h - c) / h
            if pct_off_hod < BASE_HOD_FADE:
                continue

            vol_hist = [dmap[x][4] for x in prec[-LOOKBACK:]]
            avg_vol  = np.mean(vol_hist)
            if avg_vol < BASE_MIN_VOL:
                continue

            if len(prec) < 3:
                continue
            roll3 = (c - dmap[prec[-3]][3]) / dmap[prec[-3]][3]
            if roll3 < 0.40:
                continue

            streak = 0
            for k in range(len(prec) - 1, max(len(prec) - 8, -1), -1):
                if k == 0:
                    break
                if dmap[prec[k]][3] > dmap[prec[k - 1]][3]:
                    streak += 1
                else:
                    break

            vol_ratio   = v / avg_vol if avg_vol > 0 else 0.0
            pct_vs_prev = (c - prev_c) / prev_c

            # Strategy G: all 7 filters
            if not (streak      <= G_STREAK and
                    pct_off_hod >= G_HOD    and
                    pct_vs_prev <= G_PREV   and
                    roll3       >= G_GAIN   and
                    vol_ratio   >= G_VOLR):
                continue

            nd = all_days[idx_all + 1]
            if nd not in dmap:
                continue
            nd_o, nd_h, nd_l, nd_c, *_ = dmap[nd]

            records.append(dict(
                ticker=ticker, date=d, trade_date=nd,
                pct_off_hod=pct_off_hod, pct_vs_prev=pct_vs_prev,
                streak=streak, roll3_gain=roll3, vol_ratio=vol_ratio,
                nd_open=nd_o, nd_high=nd_h, nd_close=nd_c,
            ))

    return pd.DataFrame(records)


# ── 5-min bar fetching (cached) ───────────────────────────────────────────────

def _api_get(path, params=None):
    url = BASE_URL + path
    p   = {"apiKey": API_KEY, **(params or {})}
    for attempt in range(4):
        try:
            r = requests.get(url, params=p, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = 2 ** (attempt + 2)
                print(f"  [rate-limit] sleeping {wait}s …", flush=True)
                _time.sleep(wait)
            elif r.status_code in (500, 502, 503, 504):
                _time.sleep(3)
            else:
                return {}
        except Exception:
            _time.sleep(2)
    return {}


def fetch_5m(ticker: str, day: date) -> list:
    """Return market-hours 5-min bars for (ticker, day), from cache or Polygon."""
    INTRADAY_DIR.mkdir(exist_ok=True)
    cache = INTRADAY_DIR / f"5m_{ticker}_{day}.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    ds   = day.isoformat()
    body = _api_get(
        f"/v2/aggs/ticker/{ticker}/range/5/minute/{ds}/{ds}",
        {"adjusted": "true", "limit": 200, "sort": "asc"},
    )
    _time.sleep(REQUEST_DELAY)

    bars = []
    for b in body.get("results", []):
        ts = datetime.fromtimestamp(b["t"] / 1000, tz=ET)
        if ts.time() < T_OPEN or ts.time() >= T_CLOSE:
            continue
        bars.append({
            "ts": ts,
            "o":  float(b["o"]),
            "h":  float(b["h"]),
            "l":  float(b["l"]),
            "c":  float(b["c"]),
            "v":  float(b.get("v", 0)),
        })

    with open(cache, "wb") as fh:
        pickle.dump(bars, fh, protocol=4)
    return bars


# ── Trade simulation ──────────────────────────────────────────────────────────

def simulate(bars: list, entry_idx: int, entry: float,
             nd_high: float, nd_close: float, entry_ts: datetime) -> dict:
    """
    Simulate a short trade.

    entry_idx < 0  — entry at open; check all bars for stop touch.
    entry_idx >= 0 — entry at close of bar[entry_idx]; check bar[entry_idx+1:] onward.

    Stop: entry * (1 + STOP_PCT).
    Exit: EOD (last 5-min bar close) unless stopped earlier.
    """
    stop  = entry * (1.0 + STOP_PCT)
    after = bars if entry_idx < 0 else bars[entry_idx + 1:]

    stopped = False
    exit_px = bars[-1]["c"] if bars else nd_close
    exit_ts = bars[-1]["ts"] if bars else None

    for bar in after:
        if bar["h"] >= stop:
            stopped = True
            exit_px = stop
            exit_ts = bar["ts"]
            break

    pnl = (entry - exit_px) / entry

    if exit_ts and entry_ts:
        hold = max(5.0, (exit_ts - entry_ts).total_seconds() / 60)
    else:
        hold = 390.0

    # How far below the day's final HOD did we enter?  (lower = closer to top)
    vs_hod = (nd_high - entry) / nd_high if nd_high > 0 else 0.0

    return dict(entry=entry, exit=exit_px, pnl=pnl,
                stopped=stopped, hold_min=hold, vs_hod=vs_hod)


# ── Entry variant functions ───────────────────────────────────────────────────

def _first_red(bars: list, before: dt_time):
    """Return (index, bar) of the first red candle whose open time < `before`."""
    for i, b in enumerate(bars):
        if b["ts"].time() >= before:
            return None, None
        if b["c"] < b["o"]:
            return i, b
    return None, None


def v1_open(bars, nd_open, nd_high, nd_close):
    """V1 — Short at market open (entry = daily nd_open)."""
    if not bars:
        return None
    return simulate(bars, -1, nd_open, nd_high, nd_close, bars[0]["ts"])


def v2_945(bars, nd_open, nd_high, nd_close):
    """V2 — First red 5-min candle opening before 9:45 AM ET."""
    idx, b = _first_red(bars, T_9_45)
    if b is None:
        return None
    return simulate(bars, idx, b["c"], nd_high, nd_close, b["ts"])


def v3_1000(bars, nd_open, nd_high, nd_close):
    """V3 — First red 5-min candle opening before 10:00 AM ET."""
    idx, b = _first_red(bars, T_10)
    if b is None:
        return None
    return simulate(bars, idx, b["c"], nd_high, nd_close, b["ts"])


def v4_1030(bars, nd_open, nd_high, nd_close):
    """V4 — First red 5-min candle opening before 10:30 AM ET."""
    idx, b = _first_red(bars, T_10_30)
    if b is None:
        return None
    return simulate(bars, idx, b["c"], nd_high, nd_close, b["ts"])


def t1_hod_fade(bars, nd_open, nd_high, nd_close):
    """T1 — HOD set in first 90 min (bars with ts <= 11:00); first red 5-min after."""
    hod_bars = [b for b in bars if b["ts"].time() <= T_11]
    if not hod_bars:
        return None
    hod = max(b["h"] for b in hod_bars)

    for i, b in enumerate(bars):
        if b["ts"].time() <= T_11:
            continue
        if b["h"] > hod:
            return None          # new high post-window → pattern invalid
        if b["c"] < b["o"]:     # first red candle after 11:00
            return simulate(bars, i, b["c"], nd_high, nd_close, b["ts"])
    return None


def t2_failed_push(bars, nd_open, nd_high, nd_close):
    """T2 — No new high after 10 AM; first 5-min close below the 10 AM candle."""
    up_to_10 = [b for b in bars if b["ts"].time() <= T_10]
    if not up_to_10:
        return None
    hod10   = max(b["h"] for b in up_to_10)
    ref_bar = next((b for b in up_to_10 if b["ts"].time() == T_10), up_to_10[-1])
    ref_cls = ref_bar["c"]

    for i, b in enumerate(bars):
        if b["ts"].time() <= T_10:
            continue
        if b["h"] > hod10:
            return None          # new high after 10 AM → pattern invalid
        if b["c"] < ref_cls:    # first close below the 10 AM candle
            return simulate(bars, i, b["c"], nd_high, nd_close, b["ts"])
    return None


# ── Variant registry ──────────────────────────────────────────────────────────

VARIANTS = [
    ("V1  Short at Open          ", v1_open),
    ("V2  First Red < 9:45 AM    ", v2_945),
    ("V3  First Red < 10:00 AM   ", v3_1000),
    ("V4  First Red < 10:30 AM   ", v4_1030),
    ("T1  HOD Fade (after 11 AM) ", t1_hod_fade),
    ("T2  Failed Push (10 AM ref)", t2_failed_push),
]


# ── Reporting helpers ─────────────────────────────────────────────────────────

def _stats(trades):
    pnl  = np.array([t["pnl"]     for t in trades])
    hold = np.array([t["hold_min"] for t in trades])
    stp  = np.array([t["stopped"]  for t in trades])
    hod  = np.array([t["vs_hod"]   for t in trades])
    n    = len(trades)
    wins = int((pnl > 0).sum())
    return dict(
        n=n, wr=wins / n if n else 0,
        exp=float(pnl.mean()) if n else 0,
        avg_w=float(pnl[pnl > 0].mean()) if wins else 0,
        avg_l=float(pnl[pnl <= 0].mean()) if n - wins else 0,
        hold_h=float(hold.mean()) / 60,
        stop_r=float(stp.mean()),
        vs_hod=float(hod.mean()),
    )


def print_summary(results):
    W = 30
    print(f"\n{'='*76}", flush=True)
    print(f"  ENTRY TIMING SUMMARY  (stop {STOP_PCT:.0%}, exit EOD)", flush=True)
    print(f"{'='*76}", flush=True)
    hdr = (f"  {'Variant':<{W}}  {'n':>4}  {'NTrig':>5}  "
           f"{'WR':>5}  {'Exp':>6}  {'AvgW':>6}  {'AvgL':>6}  "
           f"{'Hold':>5}  {'StpR':>5}  {'VsHOD':>6}")
    print(hdr, flush=True)
    print(f"  {'-'*74}", flush=True)

    for label, (trades, no_trig) in results.items():
        if not trades:
            row = (f"  {label:<{W}}  {'0':>4}  {no_trig:>5}  "
                   f"{'—':>5}  {'—':>6}  {'—':>6}  {'—':>6}  "
                   f"{'—':>5}  {'—':>5}  {'—':>6}")
            print(row, flush=True)
            continue

        s = _stats(trades)
        row = (f"  {label:<{W}}  {s['n']:>4}  {no_trig:>5}  "
               f"{s['wr']:>5.0%}  {s['exp']:>+6.1%}  "
               f"{s['avg_w']:>+6.1%}  {s['avg_l']:>+6.1%}  "
               f"{s['hold_h']:>4.1f}h  {s['stop_r']:>5.0%}  "
               f"{s['vs_hod']:>6.1%}")
        print(row, flush=True)

    print(flush=True)


def print_detail(results):
    print(f"\n{'='*76}", flush=True)
    print(f"  PER-TRADE DETAIL", flush=True)
    print(f"{'='*76}", flush=True)

    for label, (trades, no_trig) in results.items():
        label_clean = label.strip()
        print(f"\n  ── {label_clean}  "
              f"(n={len(trades)}, no-trigger={no_trig}) ──", flush=True)
        if not trades:
            continue

        print(f"  {'Date':>12}  {'Ticker':>6}  {'Entry':>7}  {'Exit':>7}  "
              f"{'PnL':>7}  {'Hold':>6}  {'VsHOD':>6}  W/L  Stp", flush=True)
        print(f"  {'-'*65}", flush=True)

        for t in sorted(trades, key=lambda x: str(x.get("date", ""))):
            wl  = "W" if t["pnl"] > 0 else "L"
            stp = "*" if t["stopped"] else " "
            print(f"  {str(t['date']):>12}  {t['ticker']:>6}  "
                  f"{t['entry']:>7.2f}  {t['exit']:>7.2f}  "
                  f"{t['pnl']:>+7.1%}  {t['hold_min']/60:>5.1f}h  "
                  f"{t['vs_hod']:>6.1%}  {wl}    {stp}", flush=True)

        if trades:
            s = _stats(trades)
            print(f"  Summary: WR={s['wr']:.0%}  Exp={s['exp']:+.1%}  "
                  f"AvgW={s['avg_w']:+.1%}  AvgL={s['avg_l']:+.1%}  "
                  f"StpR={s['stop_r']:.0%}  AvgVsHOD={s['vs_hod']:.1%}",
                  flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 76, flush=True)
    print("  FRD Short Entry Timing Study — Strategy G Universe (7 filters)", flush=True)
    print(f"  Stop: {STOP_PCT:.0%}  |  Exit: EOD (last 5-min bar close)", flush=True)
    print("=" * 76, flush=True)

    # ── Load daily bars ───────────────────────────────────────────────────────
    print("\n[1/3] Loading daily bars …", flush=True)
    bars, all_days = load_bars()

    # ── Build Strategy G signals ──────────────────────────────────────────────
    print("\n[2/3] Building Strategy G signals …", flush=True)
    sigs = build_signals(bars, all_days)
    print(f"  Strategy G signals: {len(sigs)}", flush=True)

    if sigs.empty:
        print("  No signals found — check that Polygon data was fetched "
              "and the date range covers the study period.", flush=True)
        return

    print("\n  Signal list:", flush=True)
    print(f"  {'Date':>12}  {'Ticker':>6}  {'TradeDate':>12}  "
          f"{'HOD%':>5}  {'Prev%':>6}  {'3dG%':>5}  {'Str':>3}  {'VolR':>5}",
          flush=True)
    print(f"  {'-'*60}", flush=True)
    for _, r in sigs.sort_values("date").iterrows():
        print(f"  {str(r['date']):>12}  {r['ticker']:>6}  "
              f"{str(r['trade_date']):>12}  "
              f"{r['pct_off_hod']:>5.0%}  {r['pct_vs_prev']:>6.0%}  "
              f"{r['roll3_gain']:>5.0%}  {int(r['streak']):>3}  "
              f"{r['vol_ratio']:>5.2f}", flush=True)

    # ── Fetch 5-min bars for each trade day ───────────────────────────────────
    print(f"\n[3/3] Fetching 5-min intraday bars for {len(sigs)} trade days …",
          flush=True)
    intraday: dict = {}
    for _, row in sigs.iterrows():
        key = (row["ticker"], row["trade_date"])
        if key not in intraday:
            bars5 = fetch_5m(row["ticker"], row["trade_date"])
            intraday[key] = bars5
            src = "cache" if (INTRADAY_DIR / f"5m_{row['ticker']}_{row['trade_date']}.pkl").exists() else "API"
            print(f"  {row['ticker']:>6}  {row['trade_date']}  "
                  f"{len(bars5):>3} bars  [{src}]", flush=True)

    # ── Run all variants ──────────────────────────────────────────────────────
    results: dict[str, tuple[list, int]] = {}

    for label, fn in VARIANTS:
        trades: list = []
        no_trig = 0
        for _, row in sigs.iterrows():
            b5 = intraday.get((row["ticker"], row["trade_date"]), [])
            if not b5:
                no_trig += 1
                continue
            t = fn(b5, row["nd_open"], row["nd_high"], row["nd_close"])
            if t is None:
                no_trig += 1
            else:
                t.update(ticker=row["ticker"], date=row["date"],
                         trade_date=row["trade_date"])
                trades.append(t)
        results[label] = (trades, no_trig)

    # ── Print results ─────────────────────────────────────────────────────────
    print_summary(results)
    print_detail(results)

    # ── Also print daily-bar baseline for comparison ──────────────────────────
    print(f"\n{'='*76}", flush=True)
    print(f"  DAILY-BAR BASELINE (V1 equivalent, nd_open entry, nd_close exit)", flush=True)
    print(f"{'='*76}", flush=True)
    e  = sigs["nd_open"].values.astype(float)
    nh = sigs["nd_high"].values.astype(float)
    nc = sigs["nd_close"].values.astype(float)
    sp = e * (1.0 + STOP_PCT)
    stopped_d = nh >= sp
    ex_d  = np.where(stopped_d, sp, nc)
    pnl_d = (e - ex_d) / e
    n_d   = len(pnl_d)
    print(f"  n={n_d}  WR={(pnl_d>0).mean():.0%}  "
          f"Exp={pnl_d.mean():+.1%}  "
          f"AvgW={pnl_d[pnl_d>0].mean():+.1%}  "
          f"AvgL={pnl_d[pnl_d<=0].mean():+.1%}  "
          f"StpR={stopped_d.mean():.0%}", flush=True)
    print("  (daily-bar baseline uses nd_close as exit; "
          "5-min variants use last 5-min bar close)", flush=True)


if __name__ == "__main__":
    main()

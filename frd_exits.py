#!/usr/bin/env python3
"""
frd_exits.py — Strategy G exit-strategy optimization
Compares 6 exit approaches on the same 12-signal Strategy G baseline.

Exit strategies tested (entry = next-day open, hard stop = 15% above entry):
  1. EOD D+1       — baseline; exit at close of entry day
  2. Hold 2 days   — exit at D+2 close; stop applies both days
  3. Hold 3 days   — exit at D+3 close; stop applies all three days
  4. D+2 open      — exit at D+2 open if profitable; else fall back to D+1 EOD
  5. Trailing 8%   — trailing stop 8% above intraday low (D+1 only, OHLCV approx)
  6. Target 15%    — 15% profit target; else EOD D+1
"""
import sys, io, pickle
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "poly_cache"
sys.path.insert(0, str(BASE_DIR))
from polygon_fetcher import _bdays

# Pool build parameters (identical to frd_final.py)
BASE_MIN_PRICE   = 2.0
BASE_MAX_PRICE   = 25.0
BASE_MIN_AVG_VOL = 300_000
BASE_MIN_3D_GAIN = 0.40
BASE_MAX_STREAK  = 6
BASE_HOD_FADE    = 0.03
LOOKBACK         = 20
SIM_DAYS         = 190

# Strategy G tight filters
G_STREAK  = 1
G_HOD     = 0.12
G_DOWN    = -0.10
G_GAIN    = 0.75
G_VOLR    = 0.30

# Exit parameters
HARD_STOP  = 0.15   # 15% hard stop above entry for all strategies
TARGET_PCT = 0.15   # 15% profit target (strategy 6)
TRAIL_PCT  = 0.08   # 8% trailing stop above intraday low (strategy 5)


# ── Data loading ─────────────────────────────────────────────────────────────

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
    """Same logic as frd_final.py but also captures D+2 and D+3 bars."""
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

            # D+1 (entry day)
            nd1 = all_days[idx_all + 1]
            if nd1 not in dmap:
                continue
            nd1_o, nd1_h, nd1_l, nd1_c, *_ = dmap[nd1]

            def _day(offset):
                """Return (o,h,l,c) for all_days[idx_all+offset], NaN if missing."""
                if idx_all + offset >= len(all_days):
                    return (np.nan,) * 4
                d = all_days[idx_all + offset]
                if d not in dmap:
                    return (np.nan,) * 4
                bv = dmap[d]
                return bv[0], bv[1], bv[2], bv[3]

            nd2_o, nd2_h, nd2_l, nd2_c = _day(2)
            nd3_o, nd3_h, nd3_l, nd3_c = _day(3)

            records.append(dict(
                ticker=ticker, date=sim_day, close=c, high=h,
                prev_close=prev_close,
                pct_off_hod=pct_off_hod, pct_vs_prev=pct_vs_prev,
                streak=streak, roll3_gain=roll3, vol_ratio=vol_ratio,
                nd_open=nd1_o, nd_high=nd1_h, nd_low=nd1_l, nd_close=nd1_c,
                nd2_open=nd2_o, nd2_high=nd2_h, nd2_low=nd2_l, nd2_close=nd2_c,
                nd3_open=nd3_o, nd3_high=nd3_h, nd3_low=nd3_l, nd3_close=nd3_c,
            ))

    return pd.DataFrame(records)


# ── Exit strategy simulators ─────────────────────────────────────────────────
# All return (exit_px, pnl, flag) arrays where flag is True if hard-stopped.
# OHLCV assumption for intraday sequence (short position):
#   - Hard stop hits BEFORE target/trail if both would trigger on same day.
#   - When trailing stop is live, the intraday sequence assumed is:
#     open → low (favorable move first) → high (bounce to stop or close)

def _exit1(s):
    """1. EOD D+1 (baseline)."""
    e   = s["nd_open"].values.astype(float)
    nh  = s["nd_high"].values.astype(float)
    nc  = s["nd_close"].values.astype(float)
    sp  = e * (1 + HARD_STOP)
    hit = nh >= sp
    ex  = np.where(hit, sp, nc)
    return ex, (e - ex) / e, hit


def _exit2(s):
    """2. Hold 2 days — hard stop applies on D+1 and D+2."""
    e    = s["nd_open"].values.astype(float)
    nh1  = s["nd_high"].values.astype(float)
    nh2  = s["nd2_high"].values.astype(float)
    nc2  = s["nd2_close"].values.astype(float)
    nc1  = s["nd_close"].values.astype(float)
    sp   = e * (1 + HARD_STOP)
    hit1 = nh1 >= sp
    # If D+2 data missing, fall back to D+1 close
    d2_ok  = ~np.isnan(nh2)
    hit2   = d2_ok & ~hit1 & (nh2 >= sp)
    hit    = hit1 | hit2
    # Exit: stop on D+1, stop on D+2, or D+2 close (or D+1 close if D+2 missing)
    ex = np.where(hit1, sp,
         np.where(hit2, sp,
         np.where(d2_ok, nc2, nc1)))
    return ex, (e - ex) / e, hit


def _exit3(s):
    """3. Hold 3 days — hard stop applies on D+1, D+2, D+3."""
    e    = s["nd_open"].values.astype(float)
    nh1  = s["nd_high"].values.astype(float)
    nh2  = s["nd2_high"].values.astype(float)
    nh3  = s["nd3_high"].values.astype(float)
    nc3  = s["nd3_close"].values.astype(float)
    nc2  = s["nd2_close"].values.astype(float)
    nc1  = s["nd_close"].values.astype(float)
    sp   = e * (1 + HARD_STOP)
    hit1 = nh1 >= sp
    d2_ok = ~np.isnan(nh2)
    d3_ok = ~np.isnan(nh3)
    hit2  = d2_ok & ~hit1 & (nh2 >= sp)
    hit3  = d3_ok & ~hit1 & ~hit2 & (nh3 >= sp)
    hit   = hit1 | hit2 | hit3
    ex = np.where(hit1, sp,
         np.where(hit2, sp,
         np.where(hit3, sp,
         np.where(d3_ok, nc3,
         np.where(d2_ok, nc2, nc1)))))
    return ex, (e - ex) / e, hit


def _exit4(s):
    """4. Exit at D+2 open if profitable for short; else fall back to D+1 EOD."""
    e    = s["nd_open"].values.astype(float)
    nh1  = s["nd_high"].values.astype(float)
    nc1  = s["nd_close"].values.astype(float)
    no2  = s["nd2_open"].values.astype(float)
    sp   = e * (1 + HARD_STOP)
    hit1 = nh1 >= sp
    d2_ok       = ~np.isnan(no2)
    profitable  = d2_ok & ~hit1 & (no2 < e)       # D+2 open is below entry
    gap_stopped = d2_ok & ~hit1 & ~profitable & (no2 >= sp)
    ex = np.where(hit1,        sp,
         np.where(gap_stopped, sp,
         np.where(profitable,  no2,
                               nc1)))              # fallback: D+1 EOD
    hit = hit1 | gap_stopped
    return ex, (e - ex) / e, hit


def _exit5(s):
    """5. Trailing stop 8% above intraday low (OHLCV approx, D+1 only).

    Assumed intraday path (short):
      - Open → Low first (favorable move) → High (potential bounce)
    If hard stop (15%) is breached on the way up, exit at hard stop.
    Otherwise, trailing stop = nd_low * (1 + TRAIL_PCT); if nd_high >= trail
    stop, exit at trail stop.  Else exit at nd_close.
    """
    e    = s["nd_open"].values.astype(float)
    nh   = s["nd_high"].values.astype(float)
    nl   = s["nd_low"].values.astype(float)
    nc   = s["nd_close"].values.astype(float)
    sp   = e * (1 + HARD_STOP)
    ts   = nl * (1 + TRAIL_PCT)   # trailing stop level (only meaningful if nl < e)
    hit_hard  = nh >= sp
    moved_fav = nl < e                          # stock moved in our favor
    hit_trail = moved_fav & ~hit_hard & (nh >= ts)
    ex = np.where(hit_hard,  sp,
         np.where(hit_trail, ts,
                             nc))
    return ex, (e - ex) / e, hit_hard


def _exit6(s):
    """6. 15% profit target or EOD D+1, whichever comes first.

    Assumed intraday path: hard stop check before target
    (conservative — hard stop wins on days where both could trigger).
    """
    e    = s["nd_open"].values.astype(float)
    nh   = s["nd_high"].values.astype(float)
    nl   = s["nd_low"].values.astype(float)
    nc   = s["nd_close"].values.astype(float)
    sp   = e * (1 + HARD_STOP)
    tgt  = e * (1 - TARGET_PCT)
    hit_stop   = nh >= sp
    hit_target = ~hit_stop & (nl <= tgt)
    ex = np.where(hit_stop,   sp,
         np.where(hit_target, tgt,
                              nc))
    return ex, (e - ex) / e, hit_stop


# ── Display ────────────────────────────────────────────────────────────────────

EXITS = [
    ("1. EOD D+1 (baseline)",                    _exit1),
    ("2. Hold 2 days",                            _exit2),
    ("3. Hold 3 days",                            _exit3),
    ("4. D+2 open if profitable, else D+1 EOD",  _exit4),
    ("5. Trailing stop 8% above low (D+1)",       _exit5),
    ("6. Target 15% gain or EOD D+1",             _exit6),
]


def stats(pnl, stopped):
    n = len(pnl)
    wins = pnl > 0
    return dict(
        n=n, wr=wins.mean(),
        avg_w=pnl[wins].mean() if wins.any() else 0.0,
        avg_l=pnl[~wins].mean() if (~wins).any() else 0.0,
        exp=pnl.mean(), stop_r=stopped.mean(),
    )


def print_trade_table(label, sigs, ex_fn):
    s   = sigs.copy().sort_values("date").reset_index(drop=True)
    ex, pnl, stopped = ex_fn(s)
    r   = stats(pnl, stopped)
    bar = "=" * 80

    print(f"\n{bar}", flush=True)
    print(f"  {label}", flush=True)
    print(f"  n={r['n']}  WR={r['wr']:.0%}  AvgW={r['avg_w']:+.1%}  "
          f"AvgL={r['avg_l']:+.1%}  Exp={r['exp']:+.1%}  StopR={r['stop_r']:.0%}",
          flush=True)
    print(bar, flush=True)

    entry = s["nd_open"].values.astype(float)
    print(f"  {'Date':>10}  {'Ticker':>6}  {'HOD%':>5}  {'Prev%':>6}  "
          f"{'Str':>3}  {'Entry':>7}  {'Exit':>7}  {'PnL':>7}  WL", flush=True)
    print("  " + "-" * 68, flush=True)
    for i in range(len(s)):
        row = s.iloc[i]
        p   = pnl[i]
        wl  = "W" if p > 0 else ("L*" if stopped[i] else "L")
        print(
            f"  {str(row['date']):>10}  {row['ticker']:>6}  "
            f"{row['pct_off_hod']:>4.0%}  {row['pct_vs_prev']:>+5.0%}  "
            f"{int(row['streak']):>3}  "
            f"${entry[i]:>6.2f}  ${ex[i]:>6.2f}  {p:>+6.1%}  {wl}",
            flush=True,
        )
    print(flush=True)
    return r


def main():
    print("=" * 80, flush=True)
    print("  Strategy G — Exit Strategy Optimization", flush=True)
    print("=" * 80, flush=True)

    print("\nLoading bars ...", flush=True)
    bars, all_days = load_bars()
    pool = build_pool(bars, all_days)
    print(f"Base pool: {len(pool):,} signals\n", flush=True)

    G = pool[
        (pool["streak"]      <= G_STREAK) &
        (pool["pct_off_hod"] >= G_HOD)    &
        (pool["pct_vs_prev"] <= G_DOWN)   &
        (pool["roll3_gain"]  >= G_GAIN)   &
        (pool["vol_ratio"]   >= G_VOLR)
    ].copy().sort_values("date").reset_index(drop=True)

    print(f"Strategy G baseline: {len(G)} signals\n", flush=True)

    # Per-strategy trade tables
    all_stats = []
    for label, fn in EXITS:
        r = print_trade_table(label, G, fn)
        all_stats.append((label, r))

    # Summary comparison
    bar = "=" * 80
    print(f"\n{bar}", flush=True)
    print("  SUMMARY — all exit strategies", flush=True)
    print(bar, flush=True)
    print(f"  {'Strategy':<44}  {'N':>4}  {'WR':>5}  {'AvgW':>6}  "
          f"{'AvgL':>6}  {'Exp':>6}  {'StpR':>5}", flush=True)
    print("  " + "-" * 78, flush=True)
    for label, r in all_stats:
        print(
            f"  {label:<44}  {r['n']:>4}  {r['wr']:>5.0%}  {r['avg_w']:>+6.1%}  "
            f"{r['avg_l']:>+6.1%}  {r['exp']:>+6.1%}  {r['stop_r']:>5.0%}",
            flush=True,
        )
    print(flush=True)
    print("Notes:", flush=True)
    print("  - Hard stop (15% above entry) is active in all strategies.", flush=True)
    print("  - D+2/D+3 OHLCV data approximates multi-day path with daily bars.", flush=True)
    print("  - Strategy 5 (trailing): sequence assumed open→low→high within D+1.", flush=True)
    print("  - Strategy 6 (target): hard stop assumed to hit before target on conflict.", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

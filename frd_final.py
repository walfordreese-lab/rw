#!/usr/bin/env python3
"""
FRD Final Strategy Validation
Stress-tests Strategy G and searches for marginal improvements.
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

BASE_MIN_PRICE   = 2.0
BASE_MAX_PRICE   = 25.0
BASE_MIN_AVG_VOL = 300_000
BASE_MIN_3D_GAIN = 0.40
BASE_MAX_STREAK  = 6
BASE_HOD_FADE    = 0.03
LOOKBACK         = 20
SIM_DAYS         = 190


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
            prec = tdates[:di]
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
            pct_vs_vwap = (c - vwap) / vwap if vwap is not None else np.nan
            day_range   = h - l
            close_pos   = (c - l) / day_range if day_range > 0 else 0.5
            next_day    = all_days[idx_all + 1]
            if next_day not in dmap:
                continue
            nd_o, nd_h, nd_l, nd_c, nd_v, _ = dmap[next_day]
            records.append(dict(
                ticker=ticker, date=sim_day, close=c, high=h, low=l,
                prev_close=prev_close,
                pct_off_hod=pct_off_hod, pct_vs_prev=pct_vs_prev,
                pct_vs_vwap=pct_vs_vwap, close_pos=close_pos,
                streak=streak, roll3_gain=roll3, vol_ratio=vol_ratio,
                nd_open=nd_o, nd_high=nd_h, nd_close=nd_c,
            ))
    return pd.DataFrame(records)


def sim(sigs, stop_pct):
    if len(sigs) == 0:
        return dict(n=0, wr=0, exp=0, avg_w=0, avg_l=0, stop_r=0)
    e  = sigs["nd_open"].values.astype(float)
    nh = sigs["nd_high"].values.astype(float)
    nc = sigs["nd_close"].values.astype(float)
    sp = e * (1.0 + stop_pct)
    stopped = nh >= sp
    ex  = np.where(stopped, sp, nc)
    pnl = (e - ex) / e
    n, wins = len(pnl), int((pnl > 0).sum())
    return dict(n=n, wr=wins/n if n else 0,
                exp=float(pnl.mean()) if n else 0,
                avg_w=float(pnl[pnl>0].mean()) if wins else 0,
                avg_l=float(pnl[pnl<=0].mean()) if n-wins else 0,
                stop_r=float(stopped.mean()))


def show(label, sigs, stop_pct):
    st = sim(sigs, stop_pct)
    print(f"  {label:<55}  n={st['n']:>3}  WR={st['wr']:>5.1%}  "
          f"AvgW={st['avg_w']:>6.1%}  AvgL={st['avg_l']:>6.1%}  "
          f"Exp={st['exp']:>6.1%}  StpR={st['stop_r']:>5.1%}", flush=True)


def monthly(label, sigs, stop_pct):
    if sigs.empty:
        return
    e  = sigs["nd_open"].values.astype(float)
    nh = sigs["nd_high"].values.astype(float)
    nc = sigs["nd_close"].values.astype(float)
    sp = e * (1.0 + stop_pct)
    stopped = nh >= sp
    ex  = np.where(stopped, sp, nc)
    pnl = (e - ex) / e
    s = sigs.copy()
    s["pnl"]   = pnl
    s["month"] = pd.to_datetime(s["date"]).dt.strftime("%Y-%m")
    grp = s.groupby("month").apply(
        lambda g: pd.Series({"n": len(g), "wr": (g["pnl"]>0).mean(), "exp": g["pnl"].mean()}),
        include_groups=False
    )
    print(f"\n  Monthly breakdown — {label}:")
    print(f"  {'Month':>8}  {'N':>4}  {'WR':>6}  {'Exp':>7}")
    for mo, row in grp.iterrows():
        marker = "  <-- loss" if row["exp"] < 0 else ""
        print(f"  {mo:>8}  {int(row['n']):>4}  {row['wr']:>6.0%}  {row['exp']:>7.1%}{marker}")
    print(f"  {'TOTAL':>8}  {grp['n'].sum():>4.0f}  "
          f"{(sigs['pnl'] if 'pnl' in sigs else pd.Series(pnl) > 0).mean():>6.0%}  {float(np.mean(pnl)):>7.1%}",
          flush=True)


def full_trades(label, sigs, stop_pct):
    if sigs.empty:
        print(f"  {label}: no trades", flush=True)
        return
    e  = sigs["nd_open"].values.astype(float)
    nh = sigs["nd_high"].values.astype(float)
    nc = sigs["nd_close"].values.astype(float)
    sp = e * (1.0 + stop_pct)
    stopped = nh >= sp
    ex  = np.where(stopped, sp, nc)
    pnl = (e - ex) / e
    s = sigs.copy().sort_values("date")
    s["entry"]   = np.round(e, 2)
    s["exit"]    = np.round(ex, 2)
    s["pnl_pct"] = np.round(pnl*100, 1)
    s["W/L"]     = np.where(pnl > 0, "W", "L")
    s["STOP"]    = np.where(stopped, "*", "")
    print(f"\n  All trades — {label}:", flush=True)
    print(f"  {'Date':>12}  {'Ticker':>6}  {'HOD%':>5}  {'Prev%':>6}  "
          f"{'3dG%':>5}  {'Str':>3}  {'VolR':>5}  "
          f"{'Entry':>6}  {'Exit':>6}  {'PnL':>6}  W/L", flush=True)
    print(f"  {'-'*85}", flush=True)
    for _, r in s.iterrows():
        print(f"  {str(r['date']):>12}  {r['ticker']:>6}  "
              f"{r['pct_off_hod']:>5.0%}  {r['pct_vs_prev']:>6.0%}  "
              f"{r['roll3_gain']:>5.0%}  {int(r['streak']):>3}  {r['vol_ratio']:>5.2f}  "
              f"{r['entry']:>6.2f}  {r['exit']:>6.2f}  "
              f"{r['pnl_pct']:>+6.1f}%  {r['W/L']}{r['STOP']}", flush=True)
    st = sim(sigs, stop_pct)
    print(f"  Summary: n={st['n']}, WR={st['wr']:.0%}, "
          f"AvgW={st['avg_w']:+.1%}, AvgL={st['avg_l']:+.1%}, "
          f"Exp={st['exp']:+.1%}, StopRate={st['stop_r']:.0%}", flush=True)


def main():
    print("=" * 72, flush=True)
    print("  FRD Final Strategy Validation", flush=True)
    print("=" * 72, flush=True)

    print("\nLoading data…", flush=True)
    bars, all_days = load_bars()
    sigs = build_pool(bars, all_days)
    print(f"\nBase pool: {len(sigs)} signals", flush=True)

    # ── Core: Strategy G (the winner) ────────────────────────────────────────
    G = sigs[
        (sigs["streak"]       <= 1)  &
        (sigs["pct_off_hod"]  >= 0.12) &
        (sigs["pct_vs_prev"]  <= -0.10) &
        (sigs["roll3_gain"]   >= 0.75) &
        (sigs["vol_ratio"]    >= 0.30)
    ]
    print("\n", flush=True)
    full_trades("Strategy G — streak<=1, HOD>=12%, down>=10%, 3dG>=75%, volR>=0.3 | Stop 15%", G, 0.15)

    # Monthly consistency
    Gm = G.copy()
    e  = Gm["nd_open"].values.astype(float)
    nh = Gm["nd_high"].values.astype(float)
    nc = Gm["nd_close"].values.astype(float)
    sp = e * 1.15
    stopped = nh >= sp
    ex  = np.where(stopped, sp, nc)
    Gm["pnl"] = (e - ex) / e
    Gm["month"] = pd.to_datetime(Gm["date"]).dt.strftime("%Y-%m")
    grp = Gm.groupby("month").apply(
        lambda g: pd.Series({"n": len(g), "wr": (g["pnl"]>0).mean(), "exp": g["pnl"].mean()}),
        include_groups=False
    )
    print(f"\n  Monthly (Strategy G, 15% stop):", flush=True)
    print(f"  {'Month':>8}  {'N':>4}  {'WR':>6}  {'Exp':>7}", flush=True)
    cum = 0.0
    for mo, row in grp.iterrows():
        cum += row["exp"] * row["n"]
        marker = "  <-- losing month" if row["exp"] < 0 else ""
        print(f"  {mo:>8}  {int(row['n']):>4}  {row['wr']:>6.0%}  {row['exp']:>7.1%}{marker}")
    all_pnl = Gm["pnl"].values
    print(f"  {'TOTAL':>8}  {len(all_pnl):>4}  "
          f"{(all_pnl>0).mean():>6.0%}  {float(all_pnl.mean()):>7.1%}", flush=True)

    # ── Sensitivity: vary the 3d gain threshold ───────────────────────────────
    print("\n\n--- Sensitivity: 3-day gain threshold (all other G filters fixed) ---", flush=True)
    print(f"  {'3dGain min':<14}", end="", flush=True)
    for s in [0.08, 0.12, 0.15]:
        print(f"  [stop {s:.0%}]  n    WR    Exp", end="", flush=True)
    print(flush=True)
    for mg in [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]:
        sub = sigs[
            (sigs["streak"]       <= 1)  &
            (sigs["pct_off_hod"]  >= 0.12) &
            (sigs["pct_vs_prev"]  <= -0.10) &
            (sigs["roll3_gain"]   >= mg) &
            (sigs["vol_ratio"]    >= 0.30)
        ]
        print(f"  >= {mg:.0%}  ({len(sub):>3})", end="", flush=True)
        for stp in [0.08, 0.12, 0.15]:
            st = sim(sub, stp)
            print(f"  [{stp:.0%}]  {st['n']:>2}  {st['wr']:>5.0%}  {st['exp']:>+5.1%}", end="", flush=True)
        print(flush=True)

    # ── Sensitivity: vary the HOD fade threshold ─────────────────────────────
    print("\n--- Sensitivity: HOD fade threshold (all other G filters fixed) ---", flush=True)
    for mh in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
        sub = sigs[
            (sigs["streak"]       <= 1)  &
            (sigs["pct_off_hod"]  >= mh) &
            (sigs["pct_vs_prev"]  <= -0.10) &
            (sigs["roll3_gain"]   >= 0.75) &
            (sigs["vol_ratio"]    >= 0.30)
        ]
        st = sim(sub, 0.15)
        print(f"  HOD >= {mh:.0%}  n={st['n']:>3}  WR={st['wr']:>5.0%}  "
              f"AvgW={st['avg_w']:>+6.1%}  AvgL={st['avg_l']:>+6.1%}  "
              f"Exp={st['exp']:>+6.1%}  StpR={st['stop_r']:>5.0%}", flush=True)

    # ── Sensitivity: vary the prev-close drop ────────────────────────────────
    print("\n--- Sensitivity: prev-close drop threshold (all other G filters fixed) ---", flush=True)
    for mp in [-0.03, -0.05, -0.07, -0.10, -0.12, -0.15, -0.20]:
        sub = sigs[
            (sigs["streak"]       <= 1)  &
            (sigs["pct_off_hod"]  >= 0.12) &
            (sigs["pct_vs_prev"]  <= mp) &
            (sigs["roll3_gain"]   >= 0.75) &
            (sigs["vol_ratio"]    >= 0.30)
        ]
        st = sim(sub, 0.15)
        print(f"  down >= {abs(mp):.0%}  n={st['n']:>3}  WR={st['wr']:>5.0%}  "
              f"AvgW={st['avg_w']:>+6.1%}  AvgL={st['avg_l']:>+6.1%}  "
              f"Exp={st['exp']:>+6.1%}  StpR={st['stop_r']:>5.0%}", flush=True)

    # ── Sensitivity: vary streak threshold ───────────────────────────────────
    print("\n--- Sensitivity: max streak (all other G filters fixed) ---", flush=True)
    for ms in [0, 1, 2, 3, 4]:
        sub = sigs[
            (sigs["streak"]       <= ms) &
            (sigs["pct_off_hod"]  >= 0.12) &
            (sigs["pct_vs_prev"]  <= -0.10) &
            (sigs["roll3_gain"]   >= 0.75) &
            (sigs["vol_ratio"]    >= 0.30)
        ]
        st = sim(sub, 0.15)
        print(f"  streak <= {ms}  n={st['n']:>3}  WR={st['wr']:>5.0%}  "
              f"AvgW={st['avg_w']:>+6.1%}  AvgL={st['avg_l']:>+6.1%}  "
              f"Exp={st['exp']:>+6.1%}  StpR={st['stop_r']:>5.0%}", flush=True)

    # ── Sensitivity: volume ratio ─────────────────────────────────────────────
    print("\n--- Sensitivity: vol ratio min (all other G filters fixed) ---", flush=True)
    for mv in [0.0, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
        sub = sigs[
            (sigs["streak"]       <= 1)  &
            (sigs["pct_off_hod"]  >= 0.12) &
            (sigs["pct_vs_prev"]  <= -0.10) &
            (sigs["roll3_gain"]   >= 0.75) &
            (sigs["vol_ratio"]    >= mv)
        ]
        st = sim(sub, 0.15)
        print(f"  volR >= {mv:.1f}  n={st['n']:>3}  WR={st['wr']:>5.0%}  "
              f"AvgW={st['avg_w']:>+6.1%}  AvgL={st['avg_l']:>+6.1%}  "
              f"Exp={st['exp']:>+6.1%}  StpR={st['stop_r']:>5.0%}", flush=True)

    # ── Best-guess refined combo ──────────────────────────────────────────────
    # Based on sensitivities, try the tightest version that preserves n>=8
    print("\n\n=== REFINED COMBINATIONS ===", flush=True)
    combos = [
        # label, streak_max, hod_min, prev_min, gain_min, vr_min, stop
        ("G  (baseline)",            1, 0.12, -0.10, 0.75, 0.30, 0.15),
        ("G+ (HOD>=15%)",            1, 0.15, -0.10, 0.75, 0.30, 0.15),
        ("G+ (down>=12%)",           1, 0.12, -0.12, 0.75, 0.30, 0.15),
        ("G+ (HOD>=15%+down>=12%)",  1, 0.15, -0.12, 0.75, 0.30, 0.15),
        ("G+ (3dG>=100%)",           1, 0.12, -0.10, 1.00, 0.30, 0.15),
        ("G+ (3dG>=125%)",           1, 0.12, -0.10, 1.25, 0.30, 0.15),
        ("G+ (volR>=0.5)",           1, 0.12, -0.10, 0.75, 0.50, 0.15),
        ("G+ (volR>=1.0)",           1, 0.12, -0.10, 0.75, 1.00, 0.15),
        ("G  20% stop",              1, 0.12, -0.10, 0.75, 0.30, 0.20),
        ("G  streak<=2",             2, 0.12, -0.10, 0.75, 0.30, 0.15),
        ("G  streak==0",             0, 0.12, -0.10, 0.75, 0.30, 0.15),
    ]
    print(f"  {'Combo':<32}  {'N':>4}  {'WR':>6}  {'AvgW':>7}  {'AvgL':>7}  {'Exp':>7}  {'StpR':>6}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for label, ms, mh, mp, mg, mv, stp in combos:
        sub = sigs[
            (sigs["streak"]       <= ms) &
            (sigs["pct_off_hod"]  >= mh) &
            (sigs["pct_vs_prev"]  <= mp) &
            (sigs["roll3_gain"]   >= mg) &
            (sigs["vol_ratio"]    >= mv)
        ]
        st = sim(sub, stp)
        print(f"  {label:<32}  {st['n']:>4}  {st['wr']:>6.0%}  {st['avg_w']:>7.1%}  "
              f"{st['avg_l']:>7.1%}  {st['exp']:>7.1%}  {st['stop_r']:>6.0%}", flush=True)

    # ── Final recommendation ──────────────────────────────────────────────────
    print("\n\n" + "="*72, flush=True)
    print("  FINAL RECOMMENDATION", flush=True)
    print("="*72, flush=True)
    Gbest = sigs[
        (sigs["streak"]       <= 1) &
        (sigs["pct_off_hod"]  >= 0.15) &
        (sigs["pct_vs_prev"]  <= -0.10) &
        (sigs["roll3_gain"]   >= 0.75) &
        (sigs["vol_ratio"]    >= 0.30)
    ]
    full_trades("Strategy G+ | streak<=1 | HOD>=15% | down>=10% | 3dG>=75% | volR>=0.3 | Stop 15%", Gbest, 0.15)
    print(flush=True)
    print("  === PARAMETERS TO CODE INTO SCANNER ===", flush=True)
    print("  Universe  : price $2-$25, avg vol >= 300K, 3d gain >= 75%", flush=True)
    print("  Signal    : streak <= 1  (at most 1 consecutive green close)", flush=True)
    print("              close < prev_close  (must close red)", flush=True)
    print("              (high - close) / high >= 15%  (HOD fade >= 15%)", flush=True)
    print("              (close - prev_close) / prev_close <= -10%  (down >= 10%)", flush=True)
    print("              volume / 20d_avg_vol >= 0.3  (vol ratio >= 0.3)", flush=True)
    print("  Trade     : SHORT at next-day open", flush=True)
    print("  Stop loss : 15% above entry (hard)", flush=True)
    print("  Exit      : EOD close (next day)", flush=True)
    print(flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

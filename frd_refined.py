#!/usr/bin/env python3
"""
FRD Refined Strategy Analysis
Deep-dives the top patterns found by frd_strategy_search.py.
Tests: failed-bounce, continuation, gap-fail, HOD-crush variants.
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

# ── universe (loose base) ─────────────────────────────────────────────────────
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


def build_signal_pool(bars, all_days):
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

            vol_ratio    = v / avg_vol if avg_vol > 0 else 0.0
            pct_vs_prev  = (c - prev_close) / prev_close
            pct_vs_vwap  = (c - vwap) / vwap if vwap is not None else np.nan
            day_range    = h - l
            close_pos    = (c - l) / day_range if day_range > 0 else 0.5
            open_vs_prev = (o - prev_close) / prev_close
            close_vs_open= (c - o) / o
            prev_red     = (prev_close < dmap[prec[-2]][3]) if len(prec) >= 2 else False
            # prev prev close (2 days ago)
            prev2_close  = dmap[prec[-2]][3] if len(prec) >= 2 else prev_close

            # Gap-up fail: opened above prev_close but closed below → strong intraday reversal
            gap_up_fail  = (o > prev_close) and (c < prev_close)

            next_day = all_days[idx_all + 1]
            if next_day not in dmap:
                continue
            nd_o, nd_h, nd_l, nd_c, nd_v, _ = dmap[next_day]

            records.append(dict(
                ticker=ticker, date=sim_day,
                prev_close=prev_close, open=o, high=h, low=l, close=c,
                pct_off_hod=pct_off_hod,
                pct_vs_prev=pct_vs_prev,
                pct_vs_vwap=pct_vs_vwap,
                close_pos=close_pos,
                open_vs_prev=open_vs_prev,
                close_vs_open=close_vs_open,
                streak=streak,
                roll3_gain=roll3,
                vol_ratio=vol_ratio,
                prev_day_red=prev_red,
                gap_up_fail=gap_up_fail,
                nd_open=nd_o, nd_high=nd_h, nd_close=nd_c,
            ))

    return pd.DataFrame(records)


def sim(sigs, stop_pct):
    if len(sigs) == 0:
        return dict(n=0, wr=0, exp=0, avg_w=0, avg_l=0, stop_r=0, pnl=[])
    entry   = sigs["nd_open"].values.astype(float)
    nd_high = sigs["nd_high"].values.astype(float)
    nd_cls  = sigs["nd_close"].values.astype(float)
    stop_px = entry * (1.0 + stop_pct)
    stopped = nd_high >= stop_px
    exit_px = np.where(stopped, stop_px, nd_cls)
    pnl     = (entry - exit_px) / entry
    n, wins = len(pnl), int((pnl > 0).sum())
    return dict(
        n=n, wr=wins/n if n else 0,
        exp=float(pnl.mean()) if n else 0,
        avg_w=float(pnl[pnl>0].mean()) if wins else 0,
        avg_l=float(pnl[pnl<=0].mean()) if n-wins else 0,
        stop_r=float(stopped.mean()) if n else 0,
        pnl=pnl,
    )


def show_trades(label, sigs, stop_pct):
    if len(sigs) == 0:
        print(f"\n  {label}: no signals", flush=True)
        return
    entry   = sigs["nd_open"].values.astype(float)
    nd_high = sigs["nd_high"].values.astype(float)
    nd_cls  = sigs["nd_close"].values.astype(float)
    stop_px = entry * (1.0 + stop_pct)
    stopped = nd_high >= stop_px
    exit_px = np.where(stopped, stop_px, nd_cls)
    pnl     = (entry - exit_px) / entry

    s = sigs.copy()
    s["entry"]   = np.round(entry, 2)
    s["exit"]    = np.round(exit_px, 2)
    s["pnl"]     = np.round(pnl * 100, 1)
    s["stop"]    = stopped
    s["W/L"]     = np.where(pnl > 0, "W", "L")

    st = sim(sigs, stop_pct)
    n = st["n"]
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  n={n}  WR={st['wr']:.0%}  AvgW={st['avg_w']:.1%}  AvgL={st['avg_l']:.1%}  Exp={st['exp']:.1%}  StopR={st['stop_r']:.0%}  Stop={stop_pct:.0%}")
    print(f"{'='*70}")
    cols = ["date","ticker","pct_off_hod","pct_vs_prev","close_pos",
            "roll3_gain","streak","vol_ratio","entry","exit","pnl","W/L","stop"]
    sub = s[cols].sort_values("date")
    # format
    def fmt_pct(v): return f"{v*100:.0f}%"
    sub = sub.copy()
    sub["pct_off_hod"] = sub["pct_off_hod"].map(lambda x: f"{x:.0%}")
    sub["pct_vs_prev"] = sub["pct_vs_prev"].map(lambda x: f"{x:.0%}")
    sub["roll3_gain"]  = sub["roll3_gain"].map(lambda x: f"{x:.0%}")
    sub["vol_ratio"]   = sub["vol_ratio"].map(lambda x: f"{x:.2f}")
    sub["close_pos"]   = sub["close_pos"].map(lambda x: f"{x:.2f}")
    sub["pnl"]         = sub["pnl"].map(lambda x: f"{x:+.1f}%")
    sub["streak"]      = sub["streak"].astype(int)
    sub["stop"]        = sub["stop"].map(lambda x: "STOP" if x else "   ")
    print(sub.to_string(index=False), flush=True)


def show_monthly(label, sigs, stop_pct):
    if len(sigs) == 0:
        return
    entry   = sigs["nd_open"].values.astype(float)
    nd_high = sigs["nd_high"].values.astype(float)
    nd_cls  = sigs["nd_close"].values.astype(float)
    stop_px = entry * (1.0 + stop_pct)
    stopped = nd_high >= stop_px
    exit_px = np.where(stopped, stop_px, nd_cls)
    pnl     = (entry - exit_px) / entry

    s = sigs.copy()
    s["pnl"]   = pnl
    s["month"] = pd.to_datetime(s["date"]).dt.strftime("%Y-%m")
    grp = s.groupby("month").apply(
        lambda g: pd.Series({
            "n":   len(g),
            "wr":  (g["pnl"] > 0).mean(),
            "exp": g["pnl"].mean(),
        }), include_groups=False
    )
    print(f"\n  Monthly breakdown for: {label}")
    print(f"  {'Month':>8}  {'N':>4}  {'WR':>6}  {'Exp':>7}")
    for mo, row in grp.iterrows():
        print(f"  {mo:>8}  {int(row['n']):>4}  {row['wr']:>6.0%}  {row['exp']:>7.1%}")


def multi_stop_sweep(label, sigs):
    stops = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    print(f"\n  Stop-sweep for: {label}  (n base={len(sigs)})")
    print(f"  {'Stop':>6}  {'N':>4}  {'WR':>6}  {'AvgW':>7}  {'AvgL':>7}  {'Exp':>7}  {'StpR':>6}")
    for s in stops:
        st = sim(sigs, s)
        if st["n"] == 0:
            continue
        print(f"  {s:>6.0%}  {st['n']:>4}  {st['wr']:>6.0%}  {st['avg_w']:>7.1%}  "
              f"{st['avg_l']:>7.1%}  {st['exp']:>7.1%}  {st['stop_r']:>6.0%}")


def main():
    print("=" * 72, flush=True)
    print("  FRD Refined Strategy Analysis", flush=True)
    print("=" * 72, flush=True)

    print("\nLoading data…", flush=True)
    bars, all_days = load_bars()
    print("Building signal pool…", flush=True)
    sigs = build_signal_pool(bars, all_days)
    print(f"\nBase pool: {len(sigs)} signals, {sigs['ticker'].nunique()} tickers", flush=True)

    # ── Strategy A: "Failed Bounce" ───────────────────────────────────────────
    # streak <= 1: stock had at most 1 green day before today's red
    # down >= 10% from prev close
    # 3d gain >= 75%, vol_ratio >= 0.3
    fa = sigs[
        (sigs["streak"]       <= 1)  &
        (sigs["pct_vs_prev"]  <= -0.10) &
        (sigs["roll3_gain"]   >= 0.75)  &
        (sigs["vol_ratio"]    >= 0.30)
    ]
    show_trades("Strategy A: Failed Bounce  (streak<=1, down>=10%, 3dG>=75%, volR>=0.3)", fa, 0.15)
    multi_stop_sweep("A: Failed Bounce", fa)
    show_monthly("A: Failed Bounce (stop=15%)", fa, 0.15)

    # ── Strategy B: Failed Bounce + streak == 1 only (true bounce day) ───────
    fb = sigs[
        (sigs["streak"]       == 1)  &
        (sigs["pct_vs_prev"]  <= -0.10) &
        (sigs["roll3_gain"]   >= 0.75)  &
        (sigs["vol_ratio"]    >= 0.30)
    ]
    show_trades("Strategy B: True Failed Bounce  (streak==1, down>=10%, 3dG>=75%, volR>=0.3)", fb, 0.15)
    multi_stop_sweep("B: True Failed Bounce", fb)

    # ── Strategy C: HOD Crush (large fade, strongly red) ─────────────────────
    fc = sigs[
        (sigs["pct_off_hod"]  >= 0.15) &
        (sigs["pct_vs_prev"]  <= -0.10) &
        (sigs["roll3_gain"]   >= 0.75)  &
        (sigs["vol_ratio"]    >= 0.30)
    ]
    show_trades("Strategy C: HOD Crush  (HOD-fade>=15%, down>=10%, 3dG>=75%, volR>=0.3)", fc, 0.15)
    multi_stop_sweep("C: HOD Crush", fc)

    # ── Strategy D: Gap-Up Fail ───────────────────────────────────────────────
    fd = sigs[
        (sigs["gap_up_fail"]  == True) &
        (sigs["roll3_gain"]   >= 0.75)  &
        (sigs["vol_ratio"]    >= 0.30)
    ]
    show_trades("Strategy D: Gap-Up Fail  (opened above prev, closed red, 3dG>=75%, volR>=0.3)", fd, 0.15)
    multi_stop_sweep("D: Gap-Up Fail", fd)

    # ── Strategy E: Continuation Short (prev day also red) ───────────────────
    fe = sigs[
        (sigs["streak"]       == 0)  &
        (sigs["pct_vs_prev"]  <= -0.08) &
        (sigs["roll3_gain"]   >= 0.75)  &
        (sigs["vol_ratio"]    >= 0.30)
    ]
    show_trades("Strategy E: Continuation Short  (streak==0, down>=8%, 3dG>=75%, volR>=0.3)", fe, 0.15)
    multi_stop_sweep("E: Continuation Short", fe)

    # ── Strategy F: Best combo from search + fix (HOD>=12%, down>=10%, 3dG>=150%) ─
    ff = sigs[
        (sigs["pct_off_hod"]  >= 0.12) &
        (sigs["pct_vs_prev"]  <= -0.10) &
        (sigs["roll3_gain"]   >= 1.50)  &
        (sigs["vol_ratio"]    >= 0.30)
    ]
    show_trades("Strategy F: HOD+Crash+HighGain  (HOD>=12%, down>=10%, 3dG>=150%, volR>=0.3)", ff, 0.15)
    multi_stop_sweep("F: HOD+Crash+HighGain", ff)

    # ── Strategy G: Ultra-strict (A + HOD >= 12%) ────────────────────────────
    fg = sigs[
        (sigs["streak"]       <= 1)  &
        (sigs["pct_off_hod"]  >= 0.12) &
        (sigs["pct_vs_prev"]  <= -0.10) &
        (sigs["roll3_gain"]   >= 0.75)  &
        (sigs["vol_ratio"]    >= 0.30)
    ]
    show_trades("Strategy G: Ultra-strict  (streak<=1, HOD>=12%, down>=10%, 3dG>=75%, volR>=0.3)", fg, 0.15)
    multi_stop_sweep("G: Ultra-strict", fg)

    # ── Strategy H: Relax to any stop, show close_pos filter ────────────────
    # Close must be in bottom 20% of day's range (near the lows)
    fh = sigs[
        (sigs["streak"]       <= 2)  &
        (sigs["close_pos"]    <= 0.20) &
        (sigs["pct_vs_prev"]  <= -0.05) &
        (sigs["roll3_gain"]   >= 0.75)  &
        (sigs["vol_ratio"]    >= 0.30)
    ]
    show_trades("Strategy H: Close-near-low  (streak<=2, clsPos<=20%, down>=5%, 3dG>=75%, volR>=0.3)", fh, 0.12)
    multi_stop_sweep("H: Close-near-low", fh)

    # ── Summary comparison ────────────────────────────────────────────────────
    strategies = [
        ("A: Failed Bounce",      fa, 0.15),
        ("B: True Bounce-Fail",   fb, 0.15),
        ("C: HOD Crush",          fc, 0.15),
        ("D: Gap-Up Fail",        fd, 0.15),
        ("E: Continuation Short", fe, 0.15),
        ("F: HOD+Crash+HighGain", ff, 0.15),
        ("G: Ultra-strict",       fg, 0.15),
        ("H: Close-near-low",     fh, 0.12),
    ]

    print("\n\n" + "=" * 72, flush=True)
    print("  STRATEGY COMPARISON SUMMARY", flush=True)
    print("=" * 72, flush=True)
    print(f"  {'Strategy':<26}  {'N':>4}  {'WR':>6}  {'AvgW':>7}  {'AvgL':>7}  {'Exp':>7}  {'StpR':>6}  {'Stop':>5}")
    print(f"  {'-'*70}")
    for name, sg, stp in strategies:
        st = sim(sg, stp)
        if st["n"] == 0:
            print(f"  {name:<26}  {'  0':>4}  {'  ---':>6}")
            continue
        print(f"  {name:<26}  {st['n']:>4}  {st['wr']:>6.0%}  {st['avg_w']:>7.1%}  "
              f"{st['avg_l']:>7.1%}  {st['exp']:>7.1%}  {st['stop_r']:>6.0%}  {stp:>5.0%}")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

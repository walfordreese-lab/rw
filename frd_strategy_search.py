#!/usr/bin/env python3
"""
FRD Strategy Parameter Search
Generates a broad signal pool then grid-searches filter / exit combinations
over the cached 6-month Polygon data.  Reports all positive-expectancy combos
with >= MIN_TRADES trades, ranked by expectancy * sqrt(n).
"""
import sys, pickle, itertools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "poly_cache"
sys.path.insert(0, str(BASE_DIR))
from polygon_fetcher import _bdays

# ── base (loose) universe filters ────────────────────────────────────────────
BASE_MIN_PRICE    = 2.0
BASE_MAX_PRICE    = 25.0
BASE_MIN_AVG_VOL  = 300_000
BASE_MIN_3D_GAIN  = 0.40   # 40% 3-day gain
BASE_MAX_STREAK   = 6
BASE_HOD_FADE     = 0.03   # at least 3% off HOD
LOOKBACK          = 20
SIM_PERIOD_DAYS   = 190    # ~6 months + buffer
MIN_TRADES        = 5      # minimum trades for a combo to be reported

# ── grid search parameters ────────────────────────────────────────────────────
GRID = dict(
    min_pct_off_hod = [0.03, 0.05, 0.08, 0.12, 0.18, 0.25],
    # how far below prev_close the close must be (any, 5%, 10%, 20%)
    min_pct_vs_prev = [-0.99, -0.05, -0.10, -0.20],
    # minimum % below VWAP (None = no VWAP req)
    min_vwap_gap    = [None, 0.03, 0.07, 0.12, 0.20],
    # close position in day range (0=at low, 1=at high); max allowed
    max_close_pos   = [0.20, 0.35, 0.50, 1.0],
    # streak of green days before today (<=)
    max_streak      = [1, 2, 3, 4],
    # minimum 3-day gain required
    min_3d_gain     = [0.75, 1.0, 1.5, 2.0],
    # minimum vol ratio (today / 20d avg)
    min_vol_ratio   = [0.0, 0.3, 0.7, 1.5],
    # stop loss %
    stop_pct        = [0.06, 0.08, 0.10, 0.12, 0.15],
)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_bars():
    today = date.today()
    start = today - timedelta(days=SIM_PERIOD_DAYS + LOOKBACK + 15)
    days  = _bdays(start, today)

    print(f"Loading {len(days)} trading days from cache…", flush=True)
    bars   = defaultdict(dict)   # ticker -> {day: (o,h,l,c,v,vwap)}
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
        if i % 30 == 0:
            print(f"  {i}/{len(days)} days…", flush=True)

    print(f"  {len(loaded)} days, {len(bars)} tickers.", flush=True)
    return bars, loaded


def build_signal_pool(bars, all_days):
    """Return DataFrame of every potential signal with all features."""
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

            # ── hard base filters ─────────────────────────────────────────
            if c >= prev_close:                         # must close red
                continue
            pct_off_hod = (h - c) / h
            if pct_off_hod < BASE_HOD_FADE:            # at least 3% off HOD
                continue

            # avg vol
            vol_hist = [dmap[d][4] for d in prec[-LOOKBACK:]]
            avg_vol  = np.mean(vol_hist)
            if avg_vol < BASE_MIN_AVG_VOL:
                continue

            # 3-day gain
            if len(prec) < 3:
                continue
            roll3 = (c - dmap[prec[-3]][3]) / dmap[prec[-3]][3]
            if roll3 < BASE_MIN_3D_GAIN:
                continue

            # streak (consecutive green closes before today)
            streak = 0
            for k in range(len(prec) - 1, max(len(prec) - 7, -1), -1):
                if k == 0:
                    break
                if dmap[prec[k]][3] > dmap[prec[k-1]][3]:
                    streak += 1
                else:
                    break
            if streak > BASE_MAX_STREAK:
                continue

            # ── derived features ──────────────────────────────────────────
            vol_ratio   = v / avg_vol if avg_vol > 0 else 0.0
            pct_vs_prev = (c - prev_close) / prev_close
            pct_vs_vwap = (c - vwap) / vwap if vwap is not None else np.nan
            day_range   = h - l
            close_pos   = (c - l) / day_range if day_range > 0 else 0.5
            open_vs_prev = (o - prev_close) / prev_close   # gap direction at open
            close_vs_open = (c - o) / o                    # intraday direction
            roll5 = (c - dmap[prec[-5]][3]) / dmap[prec[-5]][3] if len(prec) >= 5 else roll3

            # previous day was red?
            prev_red = (prev_close < dmap[prec[-2]][3]) if len(prec) >= 2 else False

            # next-day bar for simulation
            next_day = all_days[idx_all + 1]
            if next_day not in dmap:
                continue
            nd_o, nd_h, nd_l, nd_c, nd_v, _ = dmap[next_day]

            records.append({
                "ticker":        ticker,
                "date":          sim_day,
                # signal features
                "pct_off_hod":   pct_off_hod,
                "pct_vs_prev":   pct_vs_prev,
                "pct_vs_vwap":   pct_vs_vwap,
                "close_pos":     close_pos,
                "streak":        streak,
                "roll3_gain":    roll3,
                "roll5_gain":    roll5,
                "vol_ratio":     vol_ratio,
                "open_vs_prev":  open_vs_prev,
                "close_vs_open": close_vs_open,
                "prev_day_red":  prev_red,
                # next-day trade data
                "nd_open":       nd_o,
                "nd_high":       nd_h,
                "nd_close":      nd_c,
            })

    df = pd.DataFrame(records)
    return df


def simulate(sigs: pd.DataFrame, stop_pct: float):
    """Vectorised trade simulation; returns stats dict or None."""
    if len(sigs) < MIN_TRADES:
        return None
    entry   = sigs["nd_open"].values.astype(float)
    nd_high = sigs["nd_high"].values.astype(float)
    nd_cls  = sigs["nd_close"].values.astype(float)

    stop_px = entry * (1.0 + stop_pct)
    stopped = nd_high >= stop_px
    exit_px = np.where(stopped, stop_px, nd_cls)
    pnl     = (entry - exit_px) / entry

    n       = len(pnl)
    wins    = int((pnl > 0).sum())
    exp     = float(pnl.mean())
    avg_w   = float(pnl[pnl > 0].mean()) if wins > 0 else 0.0
    avg_l   = float(pnl[pnl <= 0].mean()) if (n - wins) > 0 else 0.0
    return dict(n=n, win_rate=wins/n, avg_win=avg_w, avg_loss=avg_l,
                expectancy=exp, stop_rate=float(stopped.mean()))


def grid_search(sigs: pd.DataFrame):
    G = GRID
    combos = list(itertools.product(
        G["min_pct_off_hod"], G["min_pct_vs_prev"], G["min_vwap_gap"],
        G["max_close_pos"],   G["max_streak"],        G["min_3d_gain"],
        G["min_vol_ratio"],   G["stop_pct"],
    ))
    total = len(combos)
    print(f"\nTesting {total:,} combinations…", flush=True)

    # Pre-extract numpy arrays for speed
    hod_arr   = sigs["pct_off_hod"].values
    prev_arr  = sigs["pct_vs_prev"].values
    vwap_arr  = sigs["pct_vs_vwap"].values
    cp_arr    = sigs["close_pos"].values
    str_arr   = sigs["streak"].values.astype(int)
    g3_arr    = sigs["roll3_gain"].values
    vr_arr    = sigs["vol_ratio"].values
    vwap_ok   = ~np.isnan(vwap_arr)

    results = []
    for i, (mh, mp, mv, mcp, ms, mg, mvr, stp) in enumerate(combos):
        if i % 100_000 == 0 and i > 0:
            print(f"  {i:,}/{total:,}…", flush=True)

        mask = (
            (hod_arr >= mh) &
            (prev_arr <= mp) &
            (cp_arr  <= mcp) &
            (str_arr <= ms) &
            (g3_arr  >= mg) &
            (vr_arr  >= mvr)
        )
        if mv is not None:
            mask = mask & vwap_ok & (vwap_arr <= -mv)

        idx   = np.where(mask)[0]
        stats = simulate(sigs.iloc[idx], stp)
        if stats is None:
            continue
        stats.update(min_pct_off_hod=mh, min_pct_vs_prev=mp, min_vwap_gap=mv,
                     max_close_pos=mcp, max_streak=ms, min_3d_gain=mg,
                     min_vol_ratio=mvr, stop_pct=stp)
        results.append(stats)

    return pd.DataFrame(results) if results else pd.DataFrame()


def main():
    print("=" * 64, flush=True)
    print("  FRD Strategy Parameter Search", flush=True)
    print("=" * 64, flush=True)

    bars, all_days = load_bars()
    print(f"\nBuilding signal pool…", flush=True)
    sigs = build_signal_pool(bars, all_days)

    if sigs.empty:
        print("No signals found — check cache.", flush=True)
        return

    print(f"\nSignal pool: {len(sigs)} signals, {sigs['ticker'].nunique()} tickers", flush=True)
    print(f"Date range : {sigs['date'].min()} -> {sigs['date'].max()}", flush=True)

    # Feature distribution
    def pct(col, q):
        return f"{sigs[col].quantile(q):.0%}"
    print(f"\nFeature distributions (25/50/75 pct):")
    print(f"  pct_off_hod  : {pct('pct_off_hod',.25)} / {pct('pct_off_hod',.50)} / {pct('pct_off_hod',.75)}")
    print(f"  pct_vs_prev  : {pct('pct_vs_prev',.25)} / {pct('pct_vs_prev',.50)} / {pct('pct_vs_prev',.75)}")
    print(f"  roll3_gain   : {pct('roll3_gain',.25)} / {pct('roll3_gain',.50)} / {pct('roll3_gain',.75)}")
    print(f"  vol_ratio    : {sigs['vol_ratio'].quantile(.25):.2f} / {sigs['vol_ratio'].quantile(.50):.2f} / {sigs['vol_ratio'].quantile(.75):.2f}")
    print(f"  close_pos    : {sigs['close_pos'].quantile(.25):.2f} / {sigs['close_pos'].quantile(.50):.2f} / {sigs['close_pos'].quantile(.75):.2f}")
    vwap_cov = sigs["pct_vs_vwap"].notna().mean()
    print(f"  vwap coverage: {vwap_cov:.0%}")
    print(f"  streak dist  : {dict(sigs['streak'].value_counts().sort_index())}", flush=True)

    results = grid_search(sigs)

    if results.empty:
        print("No valid combos.", flush=True)
        return

    # ── report positive-expectancy combos ────────────────────────────────────
    pos = results[(results["expectancy"] > 0) & (results["n"] >= MIN_TRADES)].copy()
    print(f"\n{len(pos)} positive-expectancy combos (n≥{MIN_TRADES})", flush=True)

    if pos.empty:
        neg = results[results["n"] >= MIN_TRADES]
        print("None found.  Showing best 20 by expectancy:", flush=True)
        best = neg.nlargest(20, "expectancy")
    else:
        pos["score"] = pos["expectancy"] * np.sqrt(pos["n"])
        best = pos.nlargest(30, "score")

    hdr = (f"{'HOD':>6} {'Prev':>6} {'VWAP':>6} {'ClsP':>5} "
           f"{'Stk':>4} {'3dG':>5} {'VolR':>5} {'Stop':>5} "
           f"{'N':>4} {'WR':>6} {'AvgW':>7} {'AvgL':>7} {'Exp':>7} {'StpR':>5}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for _, r in best.iterrows():
        vg  = f"{r['min_vwap_gap']:.0%}" if r["min_vwap_gap"] else "  --"
        mp  = f"{r['min_pct_vs_prev']:.0%}" if r["min_pct_vs_prev"] > -0.9 else " any"
        print(f"{r['min_pct_off_hod']:>6.0%} {mp:>6} {vg:>6} {r['max_close_pos']:>5.2f} "
              f"{int(r['max_streak']):>4} {r['min_3d_gain']:>5.0%} {r['min_vol_ratio']:>5.2f} {r['stop_pct']:>5.0%} "
              f"{int(r['n']):>4} {r['win_rate']:>6.1%} {r['avg_win']:>7.2%} {r['avg_loss']:>7.2%} "
              f"{r['expectancy']:>7.2%} {r['stop_rate']:>5.1%}")

    # ── also show raw signal breakdown for best combo ─────────────────────────
    if not pos.empty:
        top = best.iloc[0]
        print(f"\n--- Best combo detail ---", flush=True)
        mask = (
            (sigs["pct_off_hod"] >= top["min_pct_off_hod"]) &
            (sigs["pct_vs_prev"] <= top["min_pct_vs_prev"]) &
            (sigs["close_pos"]   <= top["max_close_pos"]) &
            (sigs["streak"]      <= int(top["max_streak"])) &
            (sigs["roll3_gain"]  >= top["min_3d_gain"]) &
            (sigs["vol_ratio"]   >= top["min_vol_ratio"])
        )
        if top["min_vwap_gap"] is not None:
            mask = mask & sigs["pct_vs_vwap"].notna() & (sigs["pct_vs_vwap"] <= -top["min_vwap_gap"])
        sub = sigs[mask].copy()
        entry   = sub["nd_open"].values
        nd_high = sub["nd_high"].values
        nd_cls  = sub["nd_close"].values
        stop_px = entry * (1.0 + top["stop_pct"])
        stopped = nd_high >= stop_px
        exit_px = np.where(stopped, stop_px, nd_cls)
        sub["pnl"]     = (entry - exit_px) / entry
        sub["stopped"] = stopped
        sub["entry"]   = entry
        sub["exit"]    = exit_px
        cols = ["date","ticker","pct_off_hod","pct_vs_prev","close_pos",
                "roll3_gain","vol_ratio","streak","entry","exit","pnl","stopped"]
        print(sub[cols].sort_values("date").to_string(index=False))

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
frd_2year_backtest.py
=====================
Strategy G — 2-year out-of-sample backtest  (June 2023 – June 2025)

Filters (7):
  1. streak      <= 1   (at most 1 consecutive green close before signal day)
  2. HOD fade    >= 12% (close at least 12% below day high)
  3. vs prev     <= -10% (close at least 10% below prior close)
  4. 3-day gain  >= 75% (3-bar return from 3 days prior)
  5. vol ratio   >= 0.30 (today vol / 20d avg vol)
  6. price       $2 – $25
  7. avg vol     >= 300 K (20-day average daily volume)

Exit 4:
  - Enter short at D+1 open
  - Hard stop: D+1 high >= entry * 1.15 → cover at stop price (D+1)
  - If not stopped: D+1 close < entry (profitable) → hold overnight → cover at D+2 open
  - Else: cover at D+1 close (EOD D+1)
"""

import sys
import io
import pickle
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

# ── Strategy G parameters ──────────────────────────────────────────────────────
PRICE_MIN      = 2.0
PRICE_MAX      = 25.0
AVG_VOL_MIN    = 300_000
GAIN_3D_MIN    = 0.75
MAX_STREAK     = 1
HOD_FADE_MIN   = 0.12
DOWN_PCT_MIN   = 0.10
VOL_RATIO_MIN  = 0.30
STOP_PCT       = 0.15
LOOKBACK_VOL   = 20   # days for avg-vol computation

# ── Sim window ─────────────────────────────────────────────────────────────────
SIM_START = date(2023, 6, 1)
SIM_END   = date(2025, 6, 30)


def load_bars(sim_start: date, sim_end: date):
    """Load all pkl files covering the window + a generous lookback buffer."""
    data_start = sim_start - timedelta(days=LOOKBACK_VOL * 2 + 10)
    days = _bdays(data_start.isoformat(), sim_end.isoformat())
    bars: dict[str, dict[date, tuple]] = defaultdict(dict)
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
            if not (0.50 <= c <= 50.0):
                continue
            v = float(row.volume)
            if v < 50_000:
                continue
            bars[row.ticker][day] = (
                float(row.open), float(row.high), float(row.low),
                c, v,
            )
    print(f"  Loaded {len(loaded)} trading days, {len(bars):,} tickers.", flush=True)
    return bars, loaded


def build_signals(bars: dict, all_days: list[date], sim_start: date, sim_end: date):
    """Apply Strategy G filters and collect qualifying signals."""
    sim_day_set = set(d for d in all_days if sim_start <= d <= sim_end)
    records = []

    for ticker, dmap in bars.items():
        tdates = sorted(dmap.keys())
        if len(tdates) < LOOKBACK_VOL + 4:
            continue
        date_idx = {d: i for i, d in enumerate(tdates)}

        for di in range(LOOKBACK_VOL + 3, len(tdates)):
            sim_day = tdates[di]
            if sim_day not in sim_day_set:
                continue

            o, h, l, c, v = dmap[sim_day]

            # ── Price range ────────────────────────────────────────────────────
            if not (PRICE_MIN <= c <= PRICE_MAX):
                continue

            # ── Need prior data for lookbacks ─────────────────────────────────
            prec = tdates[:di]
            if len(prec) < LOOKBACK_VOL + 3:
                continue

            prev_close = dmap[prec[-1]][3]

            # ── Must close red ────────────────────────────────────────────────
            if c >= prev_close:
                continue

            # ── HOD fade >=12% ────────────────────────────────────────────────
            pct_off_hod = (h - c) / h
            if pct_off_hod < HOD_FADE_MIN:
                continue

            # ── Down vs prev_close >=10% ─────────────────────────────────────
            pct_vs_prev = (c - prev_close) / prev_close
            if pct_vs_prev > -DOWN_PCT_MIN:
                continue

            # ── 3-day gain >=75% (close 3 bars ago) ─────────────────────────
            base_close = dmap[prec[-3]][3]
            roll3_gain = (c - base_close) / base_close
            if roll3_gain < GAIN_3D_MIN:
                continue

            # ── 20-day avg vol >=300K ─────────────────────────────────────────
            vols_20 = [dmap[d][4] for d in prec[-LOOKBACK_VOL:]]
            avg_vol = float(np.mean(vols_20))
            if avg_vol < AVG_VOL_MIN:
                continue

            # ── Vol ratio >=0.30 ─────────────────────────────────────────────
            vol_ratio = v / avg_vol if avg_vol > 0 else 0.0
            if vol_ratio < VOL_RATIO_MIN:
                continue

            # ── Streak <=1 ────────────────────────────────────────────────────
            streak = 0
            for k in range(len(prec) - 1, max(len(prec) - 8, -1), -1):
                if k == 0:
                    break
                if dmap[prec[k]][3] > dmap[prec[k - 1]][3]:
                    streak += 1
                else:
                    break
            if streak > MAX_STREAK:
                continue

            # ── Need D+1 and D+2 bars (for Exit 4) ───────────────────────────
            all_sorted = sorted(dmap.keys())
            all_idx    = {d: i for i, d in enumerate(all_sorted)}
            d0_idx = all_idx.get(sim_day, -1)
            if d0_idx < 0 or d0_idx + 1 >= len(all_sorted):
                continue
            d1_day = all_sorted[d0_idx + 1]
            d2_day = all_sorted[d0_idx + 2] if d0_idx + 2 < len(all_sorted) else None

            if d1_day not in dmap:
                continue
            d1o, d1h, d1l, d1c, _ = dmap[d1_day]
            d2_bar = dmap.get(d2_day) if d2_day else None

            records.append(dict(
                ticker=ticker,
                signal_date=sim_day,
                close=c, high=h, low=l,
                prev_close=prev_close,
                pct_off_hod=pct_off_hod,
                pct_vs_prev=pct_vs_prev,
                roll3_gain=roll3_gain,
                avg_vol=avg_vol,
                vol_ratio=vol_ratio,
                streak=streak,
                d1_open=d1o, d1_high=d1h, d1_low=d1l, d1_close=d1c,
                d2_open=d2_bar[0] if d2_bar else np.nan,
            ))

    return pd.DataFrame(records)


def apply_exit4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exit 4 logic:
      - Enter short at D+1 open
      - Stop: if D+1 high >= entry * 1.15, cover at stop_price
      - Else if D+1 close < entry (profitable): cover at D+2 open
      - Else: cover at D+1 close (EOD D+1)
    """
    entries  = df["d1_open"].values.astype(float)
    d1_highs = df["d1_high"].values.astype(float)
    d1_close = df["d1_close"].values.astype(float)
    d2_open  = df["d2_open"].values.astype(float)

    stop_price = entries * (1.0 + STOP_PCT)
    stopped    = d1_highs >= stop_price

    exits = np.empty(len(entries))
    exit_type = []

    for i in range(len(entries)):
        if stopped[i]:
            exits[i] = stop_price[i]
            exit_type.append("stop")
        elif d1_close[i] < entries[i]:
            # profitable at EOD D+1 — hold overnight if D+2 open exists
            d2o = d2_open[i]
            if not np.isnan(d2o):
                exits[i] = d2o
                exit_type.append("d2_open")
            else:
                exits[i] = d1_close[i]
                exit_type.append("d1_eod_no_d2")
        else:
            exits[i] = d1_close[i]
            exit_type.append("d1_eod")

    pnl = (entries - exits) / entries

    df = df.copy()
    df["entry"]     = np.round(entries, 2)
    df["exit"]      = np.round(exits, 2)
    df["exit_type"] = exit_type
    df["stopped"]   = stopped
    df["pnl"]       = pnl
    df["win"]       = pnl > 0
    return df


def stats(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return dict(n=0, wr=0, exp=0, avg_w=0, avg_l=0, stop_r=0)
    pnl = df["pnl"].values
    wins = df[df["win"]]["pnl"]
    loss = df[~df["win"]]["pnl"]
    n = len(pnl)
    return dict(
        n=n,
        wr=float((pnl > 0).mean()),
        exp=float(pnl.mean()),
        avg_w=float(wins.mean()) if len(wins) else 0.0,
        avg_l=float(loss.mean()) if len(loss) else 0.0,
        stop_r=float(df["stopped"].mean()),
    )


def main():
    print("=" * 72, flush=True)
    print("  Strategy G — 2-Year Backtest  (June 2023 - June 2025)", flush=True)
    print("=" * 72, flush=True)
    print(f"  Filters: streak<={MAX_STREAK}, HOD>={HOD_FADE_MIN:.0%}, "
          f"down>={DOWN_PCT_MIN:.0%}, 3dGain>={GAIN_3D_MIN:.0%}, "
          f"volRatio>={VOL_RATIO_MIN}, price ${PRICE_MIN}-${PRICE_MAX}, "
          f"avgVol>={AVG_VOL_MIN/1e3:.0f}K", flush=True)
    print(f"  Exit 4: D+1 stop {STOP_PCT:.0%} | if profitable EOD D+1 -> D+2 open | else D+1 EOD",
          flush=True)
    print(flush=True)

    print("Loading cached daily bars ...", flush=True)
    bars, all_days = load_bars(SIM_START, SIM_END)
    print(flush=True)

    print("Scanning for Strategy G signals ...", flush=True)
    sigs = build_signals(bars, all_days, SIM_START, SIM_END)
    print(f"  Raw signals: {len(sigs)}", flush=True)
    print(flush=True)

    if sigs.empty:
        print("  No signals found in window.", flush=True)
        return

    sigs = apply_exit4(sigs).sort_values("signal_date").reset_index(drop=True)

    # Drop rows where D+2 open was unavailable (end-of-dataset edge case, minimal)
    tradeable = sigs.dropna(subset=["exit"]).copy()
    print(f"  Tradeable trades: {len(tradeable)}", flush=True)
    print(flush=True)

    # ── Trade-by-trade list ────────────────────────────────────────────────────
    w = 110
    print("=" * w, flush=True)
    print("  ALL TRADES", flush=True)
    print("=" * w, flush=True)
    print(f"  {'Signal Date':<12}  {'Ticker':<6}  {'HOD%':>5}  {'Prev%':>6}  "
          f"{'3dG%':>5}  {'Str':>3}  {'VolR':>5}  "
          f"{'Entry':>6}  {'Exit':>6}  {'PnL':>7}  {'ExitType'}", flush=True)
    print(f"  {'-'*102}", flush=True)
    for _, r in tradeable.iterrows():
        wl_tag = "W" if r["win"] else "L"
        st_tag = "*" if r["stopped"] else " "
        print(
            f"  {str(r['signal_date']):<12}  {r['ticker']:<6}  "
            f"{r['pct_off_hod']:>5.0%}  {r['pct_vs_prev']:>6.0%}  "
            f"{r['roll3_gain']:>5.0%}  {int(r['streak']):>3}  {r['vol_ratio']:>5.2f}  "
            f"${r['entry']:>5.2f}  ${r['exit']:>5.2f}  "
            f"{r['pnl']*100:>+6.1f}%  {wl_tag}{st_tag} {r['exit_type']}",
            flush=True,
        )

    # ── Overall performance ────────────────────────────────────────────────────
    st = stats(tradeable)
    print(flush=True)
    print("=" * 60, flush=True)
    print("  OVERALL PERFORMANCE  (2-year, Jun 2023 - Jun 2025)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Signals found      : {len(sigs)}", flush=True)
    print(f"  Tradeable trades   : {st['n']}", flush=True)
    print(f"  Win rate           : {st['wr']:.1%}  "
          f"({int(tradeable['win'].sum())}W / {int((~tradeable['win']).sum())}L)", flush=True)
    print(f"  Avg win            : {st['avg_w']:+.2%}", flush=True)
    print(f"  Avg loss           : {st['avg_l']:+.2%}", flush=True)
    print(f"  Expectancy / trade : {st['exp']:+.2%}", flush=True)
    print(f"  Stop-out rate      : {st['stop_r']:.1%}", flush=True)
    if st['n'] > 0:
        best_i  = tradeable["pnl"].idxmax()
        worst_i = tradeable["pnl"].idxmin()
        print(f"  Best trade         : {tradeable.loc[best_i,'pnl']:+.2%}"
              f"  ({tradeable.loc[best_i,'ticker']} {tradeable.loc[best_i,'signal_date']})", flush=True)
        print(f"  Worst trade        : {tradeable.loc[worst_i,'pnl']:+.2%}"
              f"  ({tradeable.loc[worst_i,'ticker']} {tradeable.loc[worst_i,'signal_date']})", flush=True)

    # Exit type breakdown
    print(flush=True)
    print("  Exit type breakdown:", flush=True)
    for etype, grp in tradeable.groupby("exit_type"):
        s2 = stats(grp)
        print(f"    {etype:<20}: n={s2['n']:>3}  WR={s2['wr']:.0%}  Exp={s2['exp']:+.2%}", flush=True)

    # ── Monthly breakdown ──────────────────────────────────────────────────────
    tradeable["month"] = pd.to_datetime(tradeable["signal_date"]).dt.strftime("%Y-%m")
    print(flush=True)
    print("=" * 60, flush=True)
    print("  MONTHLY BREAKDOWN", flush=True)
    print("=" * 60, flush=True)
    print(f"  {'Month':<9}  {'N':>4}  {'Win%':>6}  {'AvgPnL':>8}  {'Expectancy':>11}  {'Stops':>6}", flush=True)
    print(f"  {'-'*52}", flush=True)
    for month, grp in sorted(tradeable.groupby("month")):
        s2 = stats(grp)
        marker = "  <-- loss" if s2["exp"] < 0 else ""
        print(f"  {month:<9}  {s2['n']:>4}  {s2['wr']:>5.0%}  "
              f"{grp['pnl'].mean():>+7.2%}  {s2['exp']:>+10.2%}  "
              f"{int(grp['stopped'].sum()):>6}{marker}", flush=True)
    print(f"  {'TOTAL':<9}  {st['n']:>4}  {st['wr']:>5.0%}  "
          f"{st['exp']:>+7.2%}  {st['exp']:>+10.2%}  "
          f"{int(tradeable['stopped'].sum()):>6}", flush=True)

    # ── Comparison table vs known 6-month baseline ────────────────────────────
    print(flush=True)
    print("=" * 70, flush=True)
    print("  COMPARISON: 2-Year vs 6-Month Baseline", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Metric':<22}  {'6-Month Baseline':>20}  {'2-Year Result':>20}", flush=True)
    print(f"  {'-'*64}", flush=True)
    baselines = [
        ("Trades (n)",       "12",          f"{st['n']}"),
        ("Win rate",         "75% – 83%",   f"{st['wr']:.1%}"),
        ("Avg win",          "N/A",         f"{st['avg_w']:+.2%}"),
        ("Avg loss",         "N/A",         f"{st['avg_l']:+.2%}"),
        ("Expectancy/trade", "+8.0% – +9.8%", f"{st['exp']:+.2%}"),
        ("Stop-out rate",    "N/A",         f"{st['stop_r']:.1%}"),
    ]
    for label, b6, b2 in baselines:
        print(f"  {label:<22}  {b6:>20}  {b2:>20}", flush=True)

    # Yearly sub-breakdown for the 2-year window
    print(flush=True)
    print("  Year-by-year sub-split:", flush=True)
    tradeable["year"] = pd.to_datetime(tradeable["signal_date"]).dt.year
    for yr, grp in sorted(tradeable.groupby("year")):
        s2 = stats(grp)
        print(f"    {yr}: n={s2['n']:>3}  WR={s2['wr']:.0%}  "
              f"Exp={s2['exp']:+.2%}  AvgW={s2['avg_w']:+.2%}  AvgL={s2['avg_l']:+.2%}", flush=True)

    print(flush=True)
    print("  Done.", flush=True)


if __name__ == "__main__":
    main()

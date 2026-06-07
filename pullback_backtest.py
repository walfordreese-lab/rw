#!/usr/bin/env python3
"""
pullback_backtest.py
====================
Full grid search: long pullback strategy on daily Polygon data (Jan 2023 – Jun 2025).

Universes   : large-cap (avg daily $ vol > $50M), mid-cap ($5M–$50M), small-cap ($500K–$5M)
Quality     : none | above 21MA | above 50MA | above 200MA | RS>SPY-63d | above 50MA+RS
Pullbacks   : dn5% dn8% dn10% dn15% from 20d-high | touch 21MA | touch 50MA | touch 200MA
Triggers    : immediate (D+1 open) | confirm green (D+1 close>open) | vol expand (vol>1.5x avg)
Hold periods: 2 / 5 / 10 / 20 days
Exit rules  : time-only | target 10% or time | target 15% or time | trail-stop 5%

Scoring: expectancy × sqrt(n). Top 10 combos with n >= 30.
"""

import sys, io, pickle, warnings
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "poly_cache"
sys.path.insert(0, str(BASE_DIR))
from polygon_fetcher import _bdays

# ── Window ─────────────────────────────────────────────────────────────────────
DATA_START  = date(2022, 1, 1)   # warm-up for 200d MA and RS
SIM_START   = date(2023, 1, 1)   # first signal allowed
SIM_END     = date(2025, 6, 30)  # last signal allowed
FWD_DAYS    = 22                  # precompute up to 22 trading days forward

# ── Universe thresholds (avg daily dollar volume over data period) ─────────────
LARGE_DVOL_MIN  = 50_000_000
MID_DVOL_MIN    =  5_000_000
MID_DVOL_MAX    = 50_000_000
SMALL_DVOL_MIN  =    500_000
SMALL_DVOL_MAX  =  5_000_000

# ── Signal cooldown: skip ticker for N days after a signal fires ───────────────
COOLDOWN_DAYS = 20

PRINT_SEP = "=" * 80


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_all_bars():
    """Load grouped daily pkl files from DATA_START to SIM_END + FWD_DAYS buffer."""
    end_buf = SIM_END + timedelta(days=FWD_DAYS * 2)
    all_days_list = _bdays(DATA_START.isoformat(), end_buf.isoformat())
    bars: dict[str, dict[date, tuple]] = defaultdict(dict)
    loaded = []
    print(f"  Loading pkl cache: {DATA_START} -> {end_buf} ...", flush=True)
    for i, day in enumerate(all_days_list):
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
            vwap = float(row.vwap) if hasattr(row, "vwap") and pd.notna(row.vwap) else c
            bars[row.ticker][day] = (
                float(row.open), float(row.high), float(row.low), c, v, vwap
            )
    print(f"  {len(loaded)} days loaded, {len(bars):,} tickers.", flush=True)
    return bars, sorted(loaded)


# ══════════════════════════════════════════════════════════════════════════════
# 2. UNIVERSE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_universe(bars: dict, all_days: list[date]) -> dict[str, str]:
    """
    Returns {ticker: 'large'|'mid'|'small'|'other'} based on average daily
    dollar volume computed over the full data period.
    """
    sim_days = [d for d in all_days if SIM_START <= d <= SIM_END]
    ticker_dvol: dict[str, list[float]] = defaultdict(list)
    for day in sim_days:
        for tkr, dmap in bars.items():
            if day in dmap:
                o, h, l, c, v, vw = dmap[day]
                ticker_dvol[tkr].append(c * v)

    classification: dict[str, str] = {}
    for tkr, dvols in ticker_dvol.items():
        if len(dvols) < 50:      # need decent history
            classification[tkr] = "other"
            continue
        avg = float(np.mean(dvols))
        if avg >= LARGE_DVOL_MIN:
            classification[tkr] = "large"
        elif avg >= MID_DVOL_MIN:
            classification[tkr] = "mid"
        elif avg >= SMALL_DVOL_MIN:
            classification[tkr] = "small"
        else:
            classification[tkr] = "other"
    counts = {k: sum(1 for v in classification.values() if v == k)
              for k in ("large", "mid", "small", "other")}
    print(f"  Universe: large={counts['large']}, mid={counts['mid']}, "
          f"small={counts['small']}, other={counts['other']}", flush=True)
    return classification


# ══════════════════════════════════════════════════════════════════════════════
# 3. INDICATOR + SIGNAL COMPUTATION PER TICKER
# ══════════════════════════════════════════════════════════════════════════════

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


def compute_signals_for_ticker(
    ticker: str,
    dmap: dict[date, tuple],
    all_days: list[date],
    spy_returns: dict[date, float],
) -> list[dict]:
    """
    Compute all pullback signals for one ticker. Returns list of signal dicts,
    each with precomputed forward returns and regime flags.
    """
    # Build aligned arrays
    tdates = sorted(d for d in dmap if d in set(all_days))
    if len(tdates) < 252:
        return []
    n = len(tdates)
    closes = np.array([dmap[d][3] for d in tdates])
    highs  = np.array([dmap[d][1] for d in tdates])
    lows   = np.array([dmap[d][2] for d in tdates])
    opens  = np.array([dmap[d][0] for d in tdates])
    vols   = np.array([dmap[d][4] for d in tdates])
    vwaps  = np.array([dmap[d][5] for d in tdates])

    # Indicators
    ma21  = _rolling_mean(closes, 21)
    ma50  = _rolling_mean(closes, 50)
    ma200 = _rolling_mean(closes, 200)
    avol  = _rolling_mean(vols, 20)          # 20d avg volume
    high20 = _rolling_max(closes, 20)        # 20d rolling high (for % pullback)

    # RS vs SPY: 63d % return vs SPY 63d % return
    spy_r63 = np.array([spy_returns.get(d, np.nan) for d in tdates])
    stk_r63 = np.full(n, np.nan)
    for i in range(63, n):
        stk_r63[i] = (closes[i] - closes[i - 63]) / closes[i - 63]

    # Find sim day indices
    sim_day_set = set(d for d in tdates if SIM_START <= d <= SIM_END)

    # Build all_days index for forward lookup
    all_days_set  = set(all_days)
    all_days_idx  = {d: i for i, d in enumerate(all_days)}

    results: list[dict] = []
    cooldown_end: date | None = None

    for i, sig_day in enumerate(tdates):
        if sig_day not in sim_day_set:
            continue
        if cooldown_end is not None and sig_day <= cooldown_end:
            continue
        if np.isnan(ma200[i]) or np.isnan(avol[i]) or np.isnan(high20[i]):
            continue

        c, h, l, o, v, vw = closes[i], highs[i], lows[i], opens[i], vols[i], vwaps[i]
        prev_c = closes[i - 1] if i > 0 else c
        ema21  = ma21[i];  ema50 = ma50[i];  ema200 = ma200[i]
        vol_avg = avol[i]
        h20    = high20[i]   # highest close of past 20 days (include today? use [i-1..i-20])

        # Pullback detections (boolean per definition)
        pct_from_high = (h20 - c) / h20 if h20 > 0 else 0.0

        dn5  = pct_from_high >= 0.05
        dn8  = pct_from_high >= 0.08
        dn10 = pct_from_high >= 0.10
        dn15 = pct_from_high >= 0.15

        # MA touches: price crossed into / through the MA from above
        prev_ma21  = ma21[i - 1]  if i > 0 else np.nan
        prev_ma50  = ma50[i - 1]  if i > 0 else np.nan
        prev_ma200 = ma200[i - 1] if i > 0 else np.nan

        touch21  = (not np.isnan(prev_ma21)  and prev_c >= prev_ma21  and c <= ema21  * 1.02)
        touch50  = (not np.isnan(prev_ma50)  and prev_c >= prev_ma50  and c <= ema50  * 1.02)
        touch200 = (not np.isnan(prev_ma200) and prev_c >= prev_ma200 and c <= ema200 * 1.02)

        # VWAP reclaim
        prev_vw = vwaps[i - 1] if i > 0 else vw
        vwap_reclaim = (closes[i - 1] < prev_vw) and (c >= vw)

        # Quality flags (evaluated on signal day)
        above_21ma  = c >= ema21
        above_50ma  = c >= ema50
        above_200ma = c >= ema200
        rs_pos_63   = (not np.isnan(stk_r63[i]) and not np.isnan(spy_r63[i])
                       and stk_r63[i] > spy_r63[i])

        # Any pullback trigger must fire for this day to be a candidate
        any_pullback = dn5 or dn8 or dn10 or dn15 or touch21 or touch50 or touch200 or vwap_reclaim
        if not any_pullback:
            continue

        # Need D+1 day data for entry
        d0_all_idx = all_days_idx.get(sig_day, -1)
        if d0_all_idx < 0 or d0_all_idx + 1 >= len(all_days):
            continue
        d1_day = all_days[d0_all_idx + 1]
        if d1_day not in dmap:
            continue
        d1o, d1h, d1l, d1c, d1v, _ = dmap[d1_day]

        # Trigger flags (evaluated on D+1)
        trig_green   = d1c > d1o                           # first green candle
        trig_volexp  = (vol_avg > 0 and d1v > 1.5 * vol_avg)  # volume expansion
        trig_gap_up  = d1o > h                             # gap above signal day high

        # Precompute forward bars (D+1 through D+FWD_DAYS)
        fwd_o, fwd_h, fwd_l, fwd_c = [], [], [], []
        for fd in range(1, FWD_DAYS + 1):
            fwd_all_idx = d0_all_idx + fd + 1   # D+(fd+1) is fd-th forward day from entry
            if fwd_all_idx >= len(all_days):
                break
            fday = all_days[fwd_all_idx]
            if fday in dmap:
                fo, fh, fl, fc, fv, _ = dmap[fday]
                fwd_o.append(fo); fwd_h.append(fh)
                fwd_l.append(fl); fwd_c.append(fc)
            else:
                fwd_o.append(np.nan); fwd_h.append(np.nan)
                fwd_l.append(np.nan); fwd_c.append(np.nan)

        if len(fwd_c) < 2:
            continue

        results.append({
            # Identity
            "ticker":       ticker,
            "signal_date":  sig_day,
            # Entry data (at D+1 open)
            "entry_open":   d1o,
            "d1_high":      d1h,
            "d1_low":       d1l,
            "d1_close":     d1c,
            "d1_vol":       d1v,
            "sig_vol_avg":  vol_avg,
            # Trigger booleans
            "trig_green":   trig_green,
            "trig_volexp":  trig_volexp,
            "trig_gapup":   trig_gap_up,
            # Pullback type booleans
            "pb_dn5":       dn5,
            "pb_dn8":       dn8,
            "pb_dn10":      dn10,
            "pb_dn15":      dn15,
            "pb_touch21":   touch21,
            "pb_touch50":   touch50,
            "pb_touch200":  touch200,
            "pb_vwap":      vwap_reclaim,
            # Quality booleans
            "q_above21":    above_21ma,
            "q_above50":    above_50ma,
            "q_above200":   above_200ma,
            "q_rs":         rs_pos_63,
            # Forward bar arrays (lists)
            "fwd_h":        fwd_h,
            "fwd_l":        fwd_l,
            "fwd_c":        fwd_c,
        })

        cooldown_end = all_days[min(d0_all_idx + COOLDOWN_DAYS, len(all_days) - 1)]

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. FORWARD RETURN / EXIT SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate_exit(entry: float, fwd_h: list, fwd_l: list, fwd_c: list,
                  hold_days: int, exit_rule: str) -> float | None:
    """
    Given entry price and forward OHLC lists, simulate the chosen exit rule.
    Returns PnL % (positive = profit) or None if data insufficient.
    """
    days_avail = min(hold_days, len(fwd_c))
    if days_avail == 0 or entry <= 0:
        return None

    if exit_rule == "time":
        # Exit at close of hold_days-th forward day
        idx = days_avail - 1
        if np.isnan(fwd_c[idx]):
            return None
        return (fwd_c[idx] - entry) / entry

    elif exit_rule.startswith("target"):
        # target10 or target15: exit if high reaches target, else time exit
        target_pct = 0.10 if exit_rule == "target10" else 0.15
        target_px  = entry * (1 + target_pct)
        for d in range(days_avail):
            fh = fwd_h[d]
            if np.isnan(fh):
                continue
            if fh >= target_px:
                return target_pct   # hit target this day
        # time exit
        idx = days_avail - 1
        return (fwd_c[idx] - entry) / entry if not np.isnan(fwd_c[idx]) else None

    elif exit_rule == "trail5":
        # 5% trailing stop from running max high; else time exit
        peak = entry
        trail_stop_pct = 0.05
        for d in range(days_avail):
            fh = fwd_h[d]
            fl = fwd_l[d]
            fc = fwd_c[d]
            if np.isnan(fh) or np.isnan(fl):
                continue
            if fh > peak:
                peak = fh
            stop_px = peak * (1 - trail_stop_pct)
            if fl <= stop_px:
                return (stop_px - entry) / entry  # stopped out
        idx = days_avail - 1
        return (fwd_c[idx] - entry) / entry if not np.isnan(fwd_c[idx]) else None

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 5. GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

UNIVERSES   = ["large", "mid", "small"]
QUALITY_MAP = {
    "none":         lambda r: True,
    "above21":      lambda r: r["q_above21"],
    "above50":      lambda r: r["q_above50"],
    "above200":     lambda r: r["q_above200"],
    "rs_pos":       lambda r: r["q_rs"],
    "above50+rs":   lambda r: r["q_above50"] and r["q_rs"],
    "above200+rs":  lambda r: r["q_above200"] and r["q_rs"],
}
PULLBACK_MAP = {
    "dn5":      lambda r: r["pb_dn5"],
    "dn8":      lambda r: r["pb_dn8"],
    "dn10":     lambda r: r["pb_dn10"],
    "dn15":     lambda r: r["pb_dn15"],
    "touch21":  lambda r: r["pb_touch21"],
    "touch50":  lambda r: r["pb_touch50"],
    "touch200": lambda r: r["pb_touch200"],
    "vwap":     lambda r: r["pb_vwap"],
}
TRIGGER_MAP = {
    "immed":    lambda r: True,                   # immediate: always enter
    "green":    lambda r: r["trig_green"],
    "volexp":   lambda r: r["trig_volexp"],
    "gapup":    lambda r: r["trig_gapup"],
}
HOLD_DAYS   = [2, 5, 10, 20]
EXIT_RULES  = ["time", "target10", "target15", "trail5"]
MIN_TRADES  = 30


def run_grid(signals_by_universe: dict[str, list[dict]]) -> pd.DataFrame:
    """
    For every combination of (universe, quality, pullback, trigger, hold, exit),
    filter signals and compute stats.
    """
    rows = []
    total_combos = (len(UNIVERSES) * len(QUALITY_MAP) * len(PULLBACK_MAP) *
                    len(TRIGGER_MAP) * len(HOLD_DAYS) * len(EXIT_RULES))
    done = 0

    for universe in UNIVERSES:
        sigs = signals_by_universe.get(universe, [])
        if not sigs:
            done += len(QUALITY_MAP) * len(PULLBACK_MAP) * len(TRIGGER_MAP) * len(HOLD_DAYS) * len(EXIT_RULES)
            continue

        for qname, qfn in QUALITY_MAP.items():
            for pbname, pbfn in PULLBACK_MAP.items():
                # Pre-filter by quality + pullback (expensive loop once, reused across triggers/holds/exits)
                qpb_sigs = [r for r in sigs if qfn(r) and pbfn(r)]

                for trname, trfn in TRIGGER_MAP.items():
                    tr_sigs = [r for r in qpb_sigs if trfn(r)]

                    for hold in HOLD_DAYS:
                        for exit_rule in EXIT_RULES:
                            done += 1
                            if done % 500 == 0:
                                print(f"  {done}/{total_combos} combos ...", flush=True)

                            if len(tr_sigs) < MIN_TRADES:
                                continue

                            pnls = []
                            for r in tr_sigs:
                                pnl = simulate_exit(
                                    r["entry_open"],
                                    r["fwd_h"], r["fwd_l"], r["fwd_c"],
                                    hold, exit_rule,
                                )
                                if pnl is not None:
                                    pnls.append(pnl)

                            if len(pnls) < MIN_TRADES:
                                continue

                            arr = np.array(pnls)
                            n   = len(arr)
                            wr  = float((arr > 0).mean())
                            exp = float(arr.mean())
                            w   = arr[arr > 0]
                            l   = arr[arr <= 0]
                            rows.append({
                                "universe":  universe,
                                "quality":   qname,
                                "pullback":  pbname,
                                "trigger":   trname,
                                "hold":      hold,
                                "exit":      exit_rule,
                                "n":         n,
                                "wr":        wr,
                                "exp":       exp,
                                "avg_w":     float(w.mean()) if len(w) else 0.0,
                                "avg_l":     float(l.mean()) if len(l) else 0.0,
                                "score":     exp * np.sqrt(n),
                            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(PRINT_SEP, flush=True)
    print("  Long Pullback Backtest — Full Grid Search", flush=True)
    print(f"  Signals: {SIM_START} -> {SIM_END} | Forward: {FWD_DAYS} days", flush=True)
    print(PRINT_SEP, flush=True)
    print(flush=True)

    # ── Load ────────────────────────────────────────────────────────────────
    print("[1/5] Loading bars ...", flush=True)
    bars, all_days = load_all_bars()
    print(flush=True)

    # ── Universe classification ─────────────────────────────────────────────
    print("[2/5] Classifying universe ...", flush=True)
    universe_map = classify_universe(bars, all_days)
    print(flush=True)

    # ── SPY reference: 63d rolling return ────────────────────────────────────
    print("[3/5] Building SPY reference ...", flush=True)
    spy_bars = bars.get("SPY", {})
    spy_dates = sorted(spy_bars.keys())
    spy_closes = {d: spy_bars[d][3] for d in spy_dates}
    spy_r63: dict[date, float] = {}
    for i, d in enumerate(spy_dates):
        if i >= 63:
            c_now  = spy_closes[d]
            c_past = spy_closes[spy_dates[i - 63]]
            spy_r63[d] = (c_now - c_past) / c_past
    print(f"  SPY: {len(spy_dates)} days, RS reference built.", flush=True)
    print(flush=True)

    # ── Signal collection ────────────────────────────────────────────────────
    print("[4/5] Scanning for pullback signals ...", flush=True)
    signals_by_universe: dict[str, list[dict]] = {"large": [], "mid": [], "small": []}
    total_sigs = 0
    for ti, (ticker, dmap) in enumerate(bars.items()):
        uclass = universe_map.get(ticker, "other")
        if uclass not in signals_by_universe:
            continue
        sigs = compute_signals_for_ticker(ticker, dmap, all_days, spy_r63)
        if sigs:
            signals_by_universe[uclass].extend(sigs)
            total_sigs += len(sigs)
        if (ti + 1) % 500 == 0:
            print(f"  {ti+1:,} tickers processed, {total_sigs:,} signals so far ...", flush=True)

    for uclass, sigs in signals_by_universe.items():
        print(f"  {uclass}: {len(sigs):,} signals", flush=True)
    print(flush=True)

    # ── Grid search ──────────────────────────────────────────────────────────
    print("[5/5] Running grid search ...", flush=True)
    results = run_grid(signals_by_universe)
    print(f"  {len(results):,} combinations with n >= {MIN_TRADES}", flush=True)
    print(flush=True)

    if results.empty:
        print("  No qualifying combinations found.", flush=True)
        return

    results = results.sort_values("score", ascending=False).reset_index(drop=True)

    # ── Report: Top 10 by score ────────────────────────────────────────────
    W = 110
    print(PRINT_SEP, flush=True)
    print("  TOP 10 COMBINATIONS  (ranked by expectancy x sqrt(n), n >= 30)", flush=True)
    print(PRINT_SEP, flush=True)
    print(f"  {'Rank':<5}  {'Universe':<7}  {'Quality':<12}  {'Pullback':<9}  "
          f"{'Trigger':<8}  {'Hold':>4}  {'Exit':<10}  "
          f"{'n':>5}  {'WR':>6}  {'AvgW':>7}  {'AvgL':>7}  {'Exp':>7}  {'Score':>8}", flush=True)
    print(f"  {'-'*105}", flush=True)

    top10 = results.head(10)
    for rank, (_, r) in enumerate(top10.iterrows(), 1):
        print(
            f"  {rank:<5}  {r['universe']:<7}  {r['quality']:<12}  {r['pullback']:<9}  "
            f"{r['trigger']:<8}  {int(r['hold']):>4}d  {r['exit']:<10}  "
            f"{int(r['n']):>5}  {r['wr']:>5.0%}  {r['avg_w']:>+6.1%}  "
            f"{r['avg_l']:>+6.1%}  {r['exp']:>+6.2%}  {r['score']:>8.2f}",
            flush=True,
        )

    # ── Summary stats by dimension ─────────────────────────────────────────
    print(flush=True)
    print("=" * 60, flush=True)
    print("  AVERAGE EXPECTANCY BY DIMENSION", flush=True)
    print("=" * 60, flush=True)
    for dim in ["universe", "quality", "pullback", "trigger", "hold", "exit"]:
        grp = results.groupby(dim)["exp"].mean().sort_values(ascending=False)
        print(f"\n  By {dim}:", flush=True)
        for val, exp in grp.items():
            cnt = int((results[dim] == val).sum())
            print(f"    {str(val):<16}  exp={exp:>+.3%}  n_combos={cnt}", flush=True)

    # ── Universe breakdown of top 50 ──────────────────────────────────────
    print(flush=True)
    print("=" * 60, flush=True)
    print("  TOP 50 FULL TABLE  (score-ranked)", flush=True)
    print("=" * 60, flush=True)
    print(f"  {'#':<4}  {'Univ':<6}  {'Quality':<12}  {'Pullback':<9}  "
          f"{'Trig':<7}  {'Hold':>4}  {'Exit':<10}  "
          f"{'n':>5}  {'WR':>5}  {'AvgW':>7}  {'AvgL':>7}  {'Exp':>7}  {'Score':>8}", flush=True)
    print(f"  {'-'*100}", flush=True)
    for rank, (_, r) in enumerate(results.head(50).iterrows(), 1):
        print(
            f"  {rank:<4}  {r['universe']:<6}  {r['quality']:<12}  {r['pullback']:<9}  "
            f"{r['trigger']:<7}  {int(r['hold']):>4}d  {r['exit']:<10}  "
            f"{int(r['n']):>5}  {r['wr']:>4.0%}  {r['avg_w']:>+6.1%}  "
            f"{r['avg_l']:>+6.1%}  {r['exp']:>+6.2%}  {r['score']:>8.2f}",
            flush=True,
        )

    # ── Save to CSV ────────────────────────────────────────────────────────
    out_csv = BASE_DIR / "pullback_grid_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"\n  Full results saved -> {out_csv}", flush=True)
    print(flush=True)
    print("  Done.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
pullback_alt_exits.py
=====================
Tests time-based and condition-based exit strategies on the top pullback setup:
  mid/small | dn8-10% | above 21MA | first green candle | up to 20d hold

Exits compared:
  1. Baseline        : hold 20 trading days, exit at close
  2. No gain by day 3: if close at day 3 <= entry price, exit at day-3 close; else day 20
  3. No gain by day 5: if close at day 5 <= entry price, exit at day-5 close; else day 20
  4. Break 50MA      : if stock close drops below 50MA after entry, exit at that close
  5. SPY drop >3%    : if SPY cumulative return from entry day's close falls below -3%,
                       exit at stock's close that same day

Combinations (earliest trigger wins):
  6. Day3 cutoff OR 50MA break
  7. Day5 cutoff OR SPY drop >3%
  8. Day3 cutoff OR SPY drop >3%
"""

import sys, io, pickle, warnings
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "poly_cache"
sys.path.insert(0, str(BASE_DIR))
from polygon_fetcher import _bdays

DATA_START    = date(2022, 1, 1)
SIM_START     = date(2023, 1, 1)
SIM_END       = date(2025, 6, 30)
HOLD_DAYS     = 20
COOLDOWN_DAYS = 20

LARGE_MIN = 50_000_000
MID_MIN   =  5_000_000
SMALL_MIN =    500_000

SPY_DROP_PCT = 0.03   # SPY exit threshold


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════════

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


def load_all_bars():
    end_buf = SIM_END + timedelta(days=HOLD_DAYS * 3)
    days_list = _bdays(DATA_START.isoformat(), end_buf.isoformat())
    bars: dict[str, dict[date, tuple]] = defaultdict(dict)
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
    ticker_dvol: dict[str, list[float]] = defaultdict(list)
    for day in sim_days:
        for tkr, dmap in bars.items():
            if day in dmap:
                c, v = dmap[day][3], dmap[day][4]
                ticker_dvol[tkr].append(c * v)
    umap: dict[str, str] = {}
    for tkr, vals in ticker_dvol.items():
        if len(vals) < 50:
            umap[tkr] = "other"; continue
        avg = float(np.mean(vals))
        if avg >= LARGE_MIN:       umap[tkr] = "large"
        elif avg >= MID_MIN:       umap[tkr] = "mid"
        elif avg >= SMALL_MIN:     umap[tkr] = "small"
        else:                      umap[tkr] = "other"
    return umap


# ════════════════════════════════════════════════════════════════════════════════
# SIGNAL COLLECTION  — includes forward 50MA and SPY closes
# ════════════════════════════════════════════════════════════════════════════════

def collect_signals(bars, all_days, umap):
    """
    Collect dn8-10% + above-21MA + green-candle signals (mid+small).
    For each signal, stores:
      entry_open  : D+1 open  (our entry)
      fwd_c       : 20 forward closes D+2..D+21
      fwd_ma50    : 50MA at each forward day  (point-in-time, computed from full history)
      spy_ref     : SPY close at D+1  (cumulative reference for SPY exit)
      spy_fwd     : SPY closes D+2..D+21
      pb_dn10     : True if also dn10 signal
    """
    # SPY reference closes (for all days)
    spy_dmap = bars.get("SPY", {})
    spy_closes: dict[date, float] = {d: v[3] for d, v in spy_dmap.items()}

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
        ma50   = _rolling_mean(closes, 50)
        ma200  = _rolling_mean(closes, 200)
        avol   = _rolling_mean(vols, 20)
        high20 = _rolling_max(closes, 20)

        # Index for forward-day 50MA lookup
        tdate_to_idx = {d: i for i, d in enumerate(tdates)}

        cooldown_end: date | None = None

        for i, sig_day in enumerate(tdates):
            if sig_day not in sim_set:
                continue
            if cooldown_end is not None and sig_day <= cooldown_end:
                continue
            if np.isnan(ma200[i]) or np.isnan(avol[i]) or np.isnan(high20[i]):
                continue

            c = closes[i]
            h20 = high20[i]
            if c <= 0 or h20 <= 0:
                continue

            pct_from_high = (h20 - c) / h20
            if pct_from_high < 0.08:
                continue
            dn10 = pct_from_high >= 0.10

            if np.isnan(ma21[i]) or c < ma21[i]:
                continue

            d0_idx = all_days_idx.get(sig_day, -1)
            if d0_idx < 0 or d0_idx + 1 >= len(all_days):
                continue
            d1_day = all_days[d0_idx + 1]
            if d1_day not in dmap:
                continue

            d1o, d1h, d1l, d1c, _ = dmap[d1_day]
            if d1c <= d1o:
                continue   # must be green candle

            spy_ref = spy_closes.get(d1_day, np.nan)

            # Collect forward data: D+2..D+21
            fwd_c, fwd_ma50, spy_fwd = [], [], []
            for fd in range(1, HOLD_DAYS + 1):
                fwd_all_idx = d0_idx + fd + 1
                if fwd_all_idx >= len(all_days):
                    fwd_c.append(np.nan); fwd_ma50.append(np.nan); spy_fwd.append(np.nan)
                    continue
                fday = all_days[fwd_all_idx]
                # Stock close
                fwd_c.append(dmap[fday][3] if fday in dmap else np.nan)
                # 50MA at forward day (point-in-time)
                fi = tdate_to_idx.get(fday, -1)
                fwd_ma50.append(float(ma50[fi]) if fi >= 0 and not np.isnan(ma50[fi]) else np.nan)
                # SPY close
                spy_fwd.append(spy_closes.get(fday, np.nan))

            # Skip if 20th day data is missing (needed for baseline)
            if len(fwd_c) < HOLD_DAYS or np.isnan(fwd_c[-1]):
                continue

            records.append({
                "ticker":      ticker,
                "signal_date": sig_day,
                "entry_open":  d1o,
                "fwd_c":       fwd_c,
                "fwd_ma50":    fwd_ma50,
                "spy_ref":     spy_ref,
                "spy_fwd":     spy_fwd,
                "pb_dn10":     dn10,
            })
            cooldown_end = all_days[min(d0_idx + COOLDOWN_DAYS, len(all_days) - 1)]

    print(f"  Done. {len(records):,} signals.", flush=True)
    return records


# ════════════════════════════════════════════════════════════════════════════════
# EXIT SIMULATION  — per-signal, returns dict of (pnl, early_exit, hold_days)
# ════════════════════════════════════════════════════════════════════════════════

def _time_exit(entry, fwd_c):
    fc = fwd_c[-1]
    return {"pnl": (fc - entry) / entry, "early": False, "hold_days": HOLD_DAYS}


def simulate_exits(sig: dict) -> dict:
    """
    Compute all 8 exit results for a single signal.
    Each result: {"pnl": float, "early": bool, "hold_days": int, "reason": str}
    """
    entry   = sig["entry_open"]
    fwd_c   = sig["fwd_c"]
    fwd_ma50 = sig["fwd_ma50"]
    spy_ref  = sig["spy_ref"]
    spy_fwd  = sig["spy_fwd"]

    base = _time_exit(entry, fwd_c)
    base["reason"] = "time"

    # ── Day 3 cutoff (check fwd_c[1] = D+3 close = 3rd day including entry day) ──
    d3 = dict(base); d3["reason"] = "time"
    if len(fwd_c) > 1 and not np.isnan(fwd_c[1]) and fwd_c[1] <= entry:
        d3 = {"pnl": (fwd_c[1] - entry) / entry, "early": True,
              "hold_days": 3, "reason": "day3"}

    # ── Day 5 cutoff (fwd_c[3] = D+5 = 5th day) ─────────────────────────────
    d5 = dict(base); d5["reason"] = "time"
    if len(fwd_c) > 3 and not np.isnan(fwd_c[3]) and fwd_c[3] <= entry:
        d5 = {"pnl": (fwd_c[3] - entry) / entry, "early": True,
              "hold_days": 5, "reason": "day5"}

    # ── Break 50MA ────────────────────────────────────────────────────────────
    ma50_r = dict(base); ma50_r["reason"] = "time"
    for d, (fc, fm) in enumerate(zip(fwd_c, fwd_ma50)):
        if not np.isnan(fc) and not np.isnan(fm) and fc < fm:
            ma50_r = {"pnl": (fc - entry) / entry, "early": True,
                      "hold_days": d + 2, "reason": "ma50"}
            break

    # ── SPY drop >3% from D+1 close ──────────────────────────────────────────
    spy_r = dict(base); spy_r["reason"] = "time"
    if not np.isnan(spy_ref) and spy_ref > 0:
        for d, (fc, spy) in enumerate(zip(fwd_c, spy_fwd)):
            if not np.isnan(spy) and spy / spy_ref - 1 <= -SPY_DROP_PCT:
                pnl = (fc - entry) / entry if not np.isnan(fc) else np.nan
                spy_r = {"pnl": pnl, "early": True,
                         "hold_days": d + 2, "reason": "spy"}
                break

    # ── Combinations: whichever fires first ──────────────────────────────────
    def combo(*sources):
        early = [s for s in sources if s["early"]]
        if not early:
            return dict(base) | {"reason": "time"}
        best = min(early, key=lambda s: s["hold_days"])
        return best

    d3_ma50  = combo(d3, ma50_r)
    d5_spy   = combo(d5, spy_r)
    d3_spy   = combo(d3, spy_r)

    return {
        "baseline":  base,
        "day3":      d3,
        "day5":      d5,
        "ma50":      ma50_r,
        "spy":       spy_r,
        "d3_ma50":   d3_ma50,
        "d5_spy":    d5_spy,
        "d3_spy":    d3_spy,
    }


# ════════════════════════════════════════════════════════════════════════════════
# STATS + REPORTING
# ════════════════════════════════════════════════════════════════════════════════

EXIT_ORDER = ["baseline", "day3", "day5", "ma50", "spy", "d3_ma50", "d5_spy", "d3_spy"]
EXIT_LABELS = {
    "baseline": "Baseline (20d time exit)",
    "day3":     "No gain by day 3",
    "day5":     "No gain by day 5",
    "ma50":     "Break 50MA",
    "spy":      "SPY drop >3%",
    "d3_ma50":  "Day3 cutoff OR 50MA break",
    "d5_spy":   "Day5 cutoff OR SPY drop",
    "d3_spy":   "Day3 cutoff OR SPY drop",
}


def aggregate(results_list: list[dict]) -> dict:
    """Aggregate per-signal exit results into stats per exit type."""
    agg: dict[str, dict] = {k: {"pnls": [], "hold_days": [], "early_count": 0, "reasons": []}
                             for k in EXIT_ORDER}
    for res in results_list:
        for key in EXIT_ORDER:
            r = res[key]
            if r["pnl"] is None or np.isnan(r["pnl"]):
                continue
            agg[key]["pnls"].append(r["pnl"])
            agg[key]["hold_days"].append(r["hold_days"])
            if r["early"]:
                agg[key]["early_count"] += 1
            agg[key]["reasons"].append(r["reason"])
    return agg


def compute_stats(agg: dict, key: str) -> dict:
    a = agg[key]
    pnls = np.array(a["pnls"])
    if len(pnls) == 0:
        return {}
    w = pnls[pnls > 0]; l = pnls[pnls <= 0]
    n = len(pnls)
    return {
        "n":           n,
        "wr":          float((pnls > 0).mean()),
        "exp":         float(pnls.mean()),
        "avg_w":       float(w.mean()) if len(w) else 0.0,
        "avg_l":       float(l.mean()) if len(l) else 0.0,
        "early_pct":   a["early_count"] / n,
        "avg_hold":    float(np.mean(a["hold_days"])),
        "reason_dist": {r: a["reasons"].count(r) / n for r in set(a["reasons"])},
    }


def print_comparison(sigs: list[dict], label: str):
    print(f"\nRunning exits for {label} ({len(sigs):,} signals) ...", flush=True)
    all_results = [simulate_exits(s) for s in sigs]
    agg = aggregate(all_results)
    base_exp = compute_stats(agg, "baseline")["exp"]

    W = 106
    print(f"\n{'='*W}", flush=True)
    print(f"  CONDITIONAL EXIT COMPARISON — {label}", flush=True)
    print(f"{'='*W}", flush=True)
    print(f"  {'Exit Strategy':<32}  {'n':>6}  {'WR':>5}  {'AvgW':>7}  {'AvgL':>7}  "
          f"{'Exp':>7}  {'vs Base':>7}  {'Early%':>6}  {'AvgHold':>7}", flush=True)
    print(f"  {'-'*(W-2)}", flush=True)

    rows = []
    for key in EXIT_ORDER:
        s = compute_stats(agg, key)
        if not s:
            continue
        delta = s["exp"] - base_exp
        rows.append((key, s, delta))

    # Sort: baseline first, then others by exp descending
    rows_sorted = [rows[0]] + sorted(rows[1:], key=lambda x: x[1]["exp"], reverse=True)

    for key, s, delta in rows_sorted:
        delta_str = f"{delta:+.2%}" if key != "baseline" else "—"
        print(f"  {EXIT_LABELS[key]:<32}  {s['n']:>6}  {s['wr']:>4.0%}  "
              f"{s['avg_w']:>+6.1%}  {s['avg_l']:>+6.1%}  {s['exp']:>+6.2%}  "
              f"{delta_str:>7}  {s['early_pct']:>5.0%}  {s['avg_hold']:>7.1f}d",
              flush=True)

    print(f"{'='*W}", flush=True)

    # Exit reason breakdown
    print(f"\n  Exit reason breakdown:", flush=True)
    print(f"  {'Strategy':<32}  {'time':>6}  {'day3':>6}  {'day5':>6}  "
          f"{'ma50':>6}  {'spy':>6}", flush=True)
    print(f"  {'-'*65}", flush=True)
    for key, s, _ in rows_sorted:
        rd = s["reason_dist"]
        def pct(reason):
            v = rd.get(reason, 0)
            return f"{v:.0%}" if v > 0 else "    -"
        print(f"  {EXIT_LABELS[key]:<32}  {pct('time'):>6}  {pct('day3'):>6}  "
              f"{pct('day5'):>6}  {pct('ma50'):>6}  {pct('spy'):>6}", flush=True)

    print(flush=True)

    # Return top result for summary
    return max(rows, key=lambda x: x[1]["exp"])


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    SEP = "=" * 80
    print(SEP, flush=True)
    print("  Pullback Conditional Exit Study", flush=True)
    print("  Setup: mid/small | dn8-10% | above 21MA | green candle | max 20d hold", flush=True)
    print(SEP, flush=True)
    print(flush=True)

    print("[1/4] Loading bars ...", flush=True)
    bars, all_days = load_all_bars()
    print(flush=True)

    print("[2/4] Classifying universe ...", flush=True)
    umap = classify_universe(bars, all_days)
    counts = {k: sum(1 for v in umap.values() if v == k)
              for k in ("large","mid","small","other")}
    print(f"  large={counts['large']:,}  mid={counts['mid']:,}  "
          f"small={counts['small']:,}  other={counts['other']:,}", flush=True)
    print(flush=True)

    print("[3/4] Scanning for signals (dn8-10%, above-21MA, green, mid+small) ...", flush=True)
    all_sigs  = collect_signals(bars, all_days, umap)
    dn8_sigs  = all_sigs
    dn10_sigs = [s for s in all_sigs if s["pb_dn10"]]
    print(f"  dn8: {len(dn8_sigs):,}  |  dn10: {len(dn10_sigs):,}", flush=True)
    print(flush=True)

    print("[4/4] Simulating all exit strategies ...", flush=True)

    best_dn8  = print_comparison(dn8_sigs,  "dn8+above21MA")
    best_dn10 = print_comparison(dn10_sigs, "dn10+above21MA")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}", flush=True)
    print("  BEST EXIT STRATEGY SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    for label, (key, s, delta) in [("dn8+above21MA", best_dn8),
                                    ("dn10+above21MA", best_dn10)]:
        print(f"\n  {label}:", flush=True)
        print(f"    Best exit  : {EXIT_LABELS[key]}", flush=True)
        print(f"    n={s['n']:,}  WR={s['wr']:.0%}  AvgW={s['avg_w']:+.1%}  "
              f"AvgL={s['avg_l']:+.1%}  Exp={s['exp']:+.2%}  "
              f"vs baseline={delta:+.2%}  Early={s['early_pct']:.0%}  "
              f"AvgHold={s['avg_hold']:.1f}d", flush=True)

    print(flush=True)

    # ── Save results ─────────────────────────────────────────────────────────
    def to_df(sigs, label):
        all_res = [simulate_exits(s) for s in sigs]
        agg = aggregate(all_res)
        rows = []
        for key in EXIT_ORDER:
            st = compute_stats(agg, key)
            if st:
                rows.append({"exit": EXIT_LABELS[key], **st})
        return pd.DataFrame(rows)

    df8  = to_df(dn8_sigs,  "dn8")
    df10 = to_df(dn10_sigs, "dn10")
    df8.drop(columns=["reason_dist"], errors="ignore").to_csv(
        BASE_DIR / "alt_exits_dn8.csv", index=False)
    df10.drop(columns=["reason_dist"], errors="ignore").to_csv(
        BASE_DIR / "alt_exits_dn10.csv", index=False)
    print("  Saved: alt_exits_dn8.csv, alt_exits_dn10.csv", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()

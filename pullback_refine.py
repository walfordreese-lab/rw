#!/usr/bin/env python3
"""
pullback_refine.py
==================
Two focused studies from the pullback grid search:

Study 1 — Top setup refinement
  Base: mid/small, dn8/dn10, first-green-close, 20d hold
  Add: above-21MA quality filter
  Compare: time exit vs 5% trailing stop
  Show: 8 combos side-by-side + monthly breakdown for best combo

Study 2 — Capitulation deep dive
  Setup: dn15% + volume expansion (vol > 1.5x 20d avg)
  All universes, all holds (2/5/10/20d), all exits (time/trail5/target15)
  Monthly breakdown + trade list for best variant
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

DATA_START   = date(2022, 1, 1)
SIM_START    = date(2023, 1, 1)
SIM_END      = date(2025, 6, 30)
FWD_DAYS     = 22
COOLDOWN     = 20

LARGE_MIN  = 50_000_000
MID_MIN    =  5_000_000
MID_MAX    = 50_000_000
SMALL_MIN  =    500_000
SMALL_MAX  =  5_000_000

SEP = "=" * 80


# ── helpers ────────────────────────────────────────────────────────────────────

def _rmean(arr, n):
    out = np.full(len(arr), np.nan)
    cs  = np.cumsum(arr)
    out[n-1:] = (cs[n-1:] - np.concatenate([[0], cs[:-n]])) / n
    return out

def _rmax(arr, n):
    out = np.full(len(arr), np.nan)
    for i in range(n-1, len(arr)):
        out[i] = arr[i-n+1:i+1].max()
    return out


# ── data loading ───────────────────────────────────────────────────────────────

def load_bars():
    end_buf  = SIM_END + timedelta(days=FWD_DAYS * 2)
    all_days = _bdays(DATA_START.isoformat(), end_buf.isoformat())
    bars: dict[str, dict] = defaultdict(dict)
    loaded = []
    print(f"  Loading pkl cache {DATA_START} -> {end_buf} ...", flush=True)
    for day in all_days:
        p = CACHE_DIR / f"grouped_{day}.pkl"
        if not p.exists(): continue
        loaded.append(day)
        with open(p, "rb") as f: df = pickle.load(f)
        for row in df.itertuples(index=False):
            c, v = float(row.close), float(row.volume)
            if c <= 0 or v < 10_000: continue
            vw = float(row.vwap) if hasattr(row,"vwap") and pd.notna(row.vwap) else c
            bars[row.ticker][day] = (float(row.open), float(row.high),
                                     float(row.low), c, v, vw)
    print(f"  {len(loaded)} days, {len(bars):,} tickers.", flush=True)
    return bars, sorted(loaded)


def classify(bars, all_days):
    sim_days = [d for d in all_days if SIM_START <= d <= SIM_END]
    dvols: dict[str, list] = defaultdict(list)
    for day in sim_days:
        for tkr, dmap in bars.items():
            if day in dmap:
                c, v = dmap[day][3], dmap[day][4]
                dvols[tkr].append(c * v)
    result = {}
    for tkr, dv in dvols.items():
        if len(dv) < 50:
            result[tkr] = "other"; continue
        avg = float(np.mean(dv))
        if   avg >= LARGE_MIN:               result[tkr] = "large"
        elif MID_MIN   <= avg < MID_MAX:     result[tkr] = "mid"
        elif SMALL_MIN <= avg < SMALL_MAX:   result[tkr] = "small"
        else:                                result[tkr] = "other"
    return result


def spy_ref(bars, all_days):
    spy = bars.get("SPY", {})
    spy_dates = sorted(spy.keys())
    r63 = {}
    for i, d in enumerate(spy_dates):
        if i >= 63:
            r63[d] = (spy[d][3] - spy[spy_dates[i-63]][3]) / spy[spy_dates[i-63]][3]
    return r63


# ── signal scanner ─────────────────────────────────────────────────────────────

def scan_ticker(ticker, dmap, all_days, spy_r63):
    tdates = sorted(d for d in dmap if d in set(all_days))
    if len(tdates) < 252: return []
    n = len(tdates)
    closes = np.array([dmap[d][3] for d in tdates])
    highs  = np.array([dmap[d][1] for d in tdates])
    lows   = np.array([dmap[d][2] for d in tdates])
    opens  = np.array([dmap[d][0] for d in tdates])
    vols   = np.array([dmap[d][4] for d in tdates])
    vwaps  = np.array([dmap[d][5] for d in tdates])

    ma21   = _rmean(closes, 21)
    ma50   = _rmean(closes, 50)
    ma200  = _rmean(closes, 200)
    avol   = _rmean(vols, 20)
    high20 = _rmax(closes, 20)

    spy_arr  = np.array([spy_r63.get(d, np.nan) for d in tdates])
    stk_r63  = np.full(n, np.nan)
    for i in range(63, n):
        stk_r63[i] = (closes[i] - closes[i-63]) / closes[i-63]

    sim_set   = {d for d in tdates if SIM_START <= d <= SIM_END}
    all_idx   = {d: i for i, d in enumerate(all_days)}
    ad_sorted = all_days

    results = []
    cooldown_end = None

    for i, sd in enumerate(tdates):
        if sd not in sim_set: continue
        if cooldown_end and sd <= cooldown_end: continue
        if any(np.isnan(x) for x in [ma21[i], ma50[i], ma200[i], avol[i], high20[i]]): continue

        c = closes[i]; h = highs[i]; v = vols[i]; vw = vwaps[i]
        av = avol[i]; h20 = high20[i]
        pct_from_h20 = (h20 - c) / h20 if h20 > 0 else 0.0

        dn5  = pct_from_h20 >= 0.05
        dn8  = pct_from_h20 >= 0.08
        dn10 = pct_from_h20 >= 0.10
        dn15 = pct_from_h20 >= 0.15

        if not (dn5 or dn8 or dn10 or dn15): continue

        # D+1 lookup
        d0i = all_idx.get(sd, -1)
        if d0i < 0 or d0i+1 >= len(ad_sorted): continue
        d1d = ad_sorted[d0i+1]
        if d1d not in dmap: continue
        d1o, d1h, d1l, d1c, d1v, _ = dmap[d1d]

        # Triggers
        trig_green  = d1c > d1o
        trig_volexp = (av > 0 and d1v > 1.5 * av)

        # Quality
        q_above21  = c >= ma21[i]
        q_above50  = c >= ma50[i]
        q_above200 = c >= ma200[i]
        q_rs       = (not np.isnan(stk_r63[i]) and not np.isnan(spy_arr[i])
                      and stk_r63[i] > spy_arr[i])

        # Forward bars
        fwd_h, fwd_l, fwd_c = [], [], []
        for fd in range(1, FWD_DAYS+1):
            fi = d0i + fd + 1
            if fi >= len(ad_sorted): break
            fd_day = ad_sorted[fi]
            if fd_day in dmap:
                fo, fh, fl, fc, fv, _ = dmap[fd_day]
                fwd_h.append(fh); fwd_l.append(fl); fwd_c.append(fc)
            else:
                fwd_h.append(np.nan); fwd_l.append(np.nan); fwd_c.append(np.nan)

        if len(fwd_c) < 2: continue

        results.append({
            "ticker": ticker, "signal_date": sd,
            "pct_from_high": pct_from_h20,
            "entry": d1o, "d1_close": d1c, "d1_vol": d1v, "sig_avol": av,
            "trig_green": trig_green, "trig_volexp": trig_volexp,
            "pb_dn5": dn5, "pb_dn8": dn8, "pb_dn10": dn10, "pb_dn15": dn15,
            "q_above21": q_above21, "q_above50": q_above50,
            "q_above200": q_above200, "q_rs": q_rs,
            "fwd_h": fwd_h, "fwd_l": fwd_l, "fwd_c": fwd_c,
        })
        cooldown_end = ad_sorted[min(d0i + COOLDOWN, len(ad_sorted)-1)]

    return results


# ── exit simulation ────────────────────────────────────────────────────────────

def sim_exit(entry, fwd_h, fwd_l, fwd_c, hold, rule):
    days = min(hold, len(fwd_c))
    if days == 0 or entry <= 0: return None
    if rule == "time":
        idx = days - 1
        return (fwd_c[idx] - entry) / entry if not np.isnan(fwd_c[idx]) else None
    elif rule in ("target10", "target15"):
        tgt = 0.10 if rule == "target10" else 0.15
        tgt_px = entry * (1 + tgt)
        for d in range(days):
            if not np.isnan(fwd_h[d]) and fwd_h[d] >= tgt_px:
                return tgt
        idx = days - 1
        return (fwd_c[idx] - entry) / entry if not np.isnan(fwd_c[idx]) else None
    elif rule == "trail5":
        peak = entry
        for d in range(days):
            fh, fl, fc = fwd_h[d], fwd_l[d], fwd_c[d]
            if np.isnan(fh): continue
            if fh > peak: peak = fh
            stop = peak * 0.95
            if fl <= stop: return (stop - entry) / entry
        idx = days - 1
        return (fwd_c[idx] - entry) / entry if not np.isnan(fwd_c[idx]) else None
    return None


def calc_stats(pnls):
    if not pnls: return {}
    arr = np.array(pnls)
    w = arr[arr > 0]; l = arr[arr <= 0]
    return {
        "n":      len(arr),
        "wr":     float((arr>0).mean()),
        "exp":    float(arr.mean()),
        "avg_w":  float(w.mean()) if len(w) else 0.0,
        "avg_l":  float(l.mean()) if len(l) else 0.0,
        "median": float(np.median(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
        "score":  float(arr.mean() * np.sqrt(len(arr))),
    }

def print_row(label, st, w=42):
    if not st or st.get("n",0) == 0:
        print(f"  {label:<{w}}  n=    0", flush=True); return
    print(f"  {label:<{w}}  n={st['n']:>6}  WR={st['wr']:>4.0%}  "
          f"AvgW={st['avg_w']:>+6.1%}  AvgL={st['avg_l']:>+6.1%}  "
          f"Exp={st['exp']:>+6.2%}  Score={st['score']:>7.2f}", flush=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print(SEP, flush=True)
    print("  Pullback Refinement — Two Focused Studies", flush=True)
    print(SEP, flush=True)
    print(flush=True)

    print("[1/4] Loading bars ...", flush=True)
    bars, all_days = load_bars()
    print(flush=True)

    print("[2/4] Classifying universe + SPY reference ...", flush=True)
    umap   = classify(bars, all_days)
    spy_r63 = spy_ref(bars, all_days)
    counts = {k: sum(1 for v in umap.values() if v==k) for k in ("large","mid","small")}
    print(f"  large={counts['large']}, mid={counts['mid']}, small={counts['small']}", flush=True)
    print(flush=True)

    print("[3/4] Scanning signals ...", flush=True)
    sigs_by_u: dict[str, list] = {"large":[], "mid":[], "small":[]}
    for ti, (tkr, dmap) in enumerate(bars.items()):
        u = umap.get(tkr, "other")
        if u not in sigs_by_u: continue
        sigs_by_u[u].extend(scan_ticker(tkr, dmap, all_days, spy_r63))
        if (ti+1) % 2000 == 0:
            total = sum(len(v) for v in sigs_by_u.values())
            print(f"  {ti+1:,} tickers, {total:,} signals ...", flush=True)

    for u, sigs in sigs_by_u.items():
        print(f"  {u}: {len(sigs):,} signals", flush=True)
    all_sigs = sigs_by_u["large"] + sigs_by_u["mid"] + sigs_by_u["small"]
    mid_small = sigs_by_u["mid"] + sigs_by_u["small"]
    print(flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # STUDY 1 — Top setup refinement
    # ════════════════════════════════════════════════════════════════════════
    print(SEP, flush=True)
    print("  STUDY 1 — Top Setup Refinement", flush=True)
    print("  Base: mid/small + dn8-10% + first-green-close + 20d hold", flush=True)
    print("  Variable: quality filter (none vs above-21MA) x exit (time vs trail5)", flush=True)
    print(SEP, flush=True)
    print(flush=True)
    print(f"  {'Combo':<52}  {'n':>6}  {'WR':>5}  {'AvgW':>7}  {'AvgL':>7}  {'Exp':>7}  {'Score':>7}", flush=True)
    print(f"  {'-'*97}", flush=True)

    study1_best = None
    study1_best_score = -999

    for pb_name, pb_col in [("dn8", "pb_dn8"), ("dn10", "pb_dn10")]:
        for q_name, q_fn in [("no-filter", lambda r: True),
                               ("above-21MA", lambda r: r["q_above21"])]:
            for exit_rule in ["time", "trail5"]:
                subset = [r for r in mid_small
                          if r[pb_col] and r["trig_green"] and q_fn(r)]
                pnls = [p for r in subset
                        for p in [sim_exit(r["entry"], r["fwd_h"], r["fwd_l"],
                                           r["fwd_c"], 20, exit_rule)]
                        if p is not None]
                st = calc_stats(pnls)
                label = f"mid/small | {pb_name} | green | 20d | {exit_rule:<8} | {q_name}"
                print_row(label, st, w=55)
                if st and st["score"] > study1_best_score:
                    study1_best_score = st["score"]
                    study1_best = (subset, exit_rule, 20, label)
        print(flush=True)

    # Hold period sensitivity for best quality filter (above-21MA, dn8)
    print(flush=True)
    print("  Hold period sensitivity — mid/small | dn8 | green | above-21MA:", flush=True)
    print(f"  {'Hold':<10}  {'Exit':<12}  {'n':>6}  {'WR':>5}  {'AvgW':>7}  {'AvgL':>7}  {'Exp':>7}  {'Score':>7}", flush=True)
    print(f"  {'-'*75}", flush=True)
    subset_21 = [r for r in mid_small if r["pb_dn8"] and r["trig_green"] and r["q_above21"]]
    for hold in [2, 5, 10, 20]:
        for exit_rule in ["time", "trail5", "target15"]:
            pnls = [p for r in subset_21
                    for p in [sim_exit(r["entry"], r["fwd_h"], r["fwd_l"],
                                       r["fwd_c"], hold, exit_rule)]
                    if p is not None]
            st = calc_stats(pnls)
            if st:
                print(f"  {hold:>2}d         {exit_rule:<12}  {st['n']:>6}  {st['wr']:>4.0%}  "
                      f"{st['avg_w']:>+6.1%}  {st['avg_l']:>+6.1%}  "
                      f"{st['exp']:>+6.2%}  {st['score']:>7.2f}", flush=True)
        print(flush=True)

    # Monthly breakdown for best combo (dn8 | above-21MA | 20d | time)
    best_subset_s1 = [r for r in mid_small if r["pb_dn8"] and r["trig_green"] and r["q_above21"]]
    best_exit_s1   = "time"
    pnl_records_s1 = []
    for r in best_subset_s1:
        p = sim_exit(r["entry"], r["fwd_h"], r["fwd_l"], r["fwd_c"], 20, best_exit_s1)
        if p is not None:
            pnl_records_s1.append({"month": str(r["signal_date"])[:7], "pnl": p,
                                    "ticker": r["ticker"], "date": r["signal_date"],
                                    "universe": "mid" if r in sigs_by_u["mid"] else "small"})

    df_s1 = pd.DataFrame(pnl_records_s1)
    print(flush=True)
    print("  Monthly breakdown — mid/small | dn8 | above-21MA | green | 20d | time:", flush=True)
    print(f"  {'Month':<9}  {'N':>5}  {'WR':>5}  {'AvgW':>7}  {'AvgL':>7}  {'Exp':>7}  {'StdDev':>8}", flush=True)
    print(f"  {'-'*60}", flush=True)
    for month, grp in sorted(df_s1.groupby("month")):
        arr = grp["pnl"].values
        w=arr[arr>0]; l=arr[arr<=0]
        wr=float((arr>0).mean()); exp=float(arr.mean())
        aw=float(w.mean()) if len(w) else 0.0
        al=float(l.mean()) if len(l) else 0.0
        sd=float(arr.std()) if len(arr)>1 else 0.0
        mk = "  <-- loss" if exp < 0 else ""
        print(f"  {month:<9}  {len(arr):>5}  {wr:>4.0%}  {aw:>+6.1%}  "
              f"{al:>+6.1%}  {exp:>+6.2%}  {sd:>7.2%}{mk}", flush=True)
    all_pnl_s1 = df_s1["pnl"].values
    wr_t = float((all_pnl_s1>0).mean()); exp_t = float(all_pnl_s1.mean())
    w_t=all_pnl_s1[all_pnl_s1>0]; l_t=all_pnl_s1[all_pnl_s1<=0]
    print(f"  {'TOTAL':<9}  {len(all_pnl_s1):>5}  {wr_t:>4.0%}  "
          f"{float(w_t.mean()) if len(w_t) else 0:>+6.1%}  "
          f"{float(l_t.mean()) if len(l_t) else 0:>+6.1%}  "
          f"{exp_t:>+6.2%}  {float(all_pnl_s1.std()):>7.2%}", flush=True)

    # Return distribution
    print(flush=True)
    print("  Return distribution — Study 1 best combo:", flush=True)
    bins = [(-1,-0.20),(-0.20,-0.10),(-0.10,-0.05),(-0.05,0),
            (0,0.05),(0.05,0.10),(0.10,0.20),(0.20,0.50),(0.50,2.0)]
    for lo, hi in bins:
        cnt = int(((all_pnl_s1 >= lo) & (all_pnl_s1 < hi)).sum())
        bar = "#" * (cnt // max(1, len(all_pnl_s1)//100))
        pct = cnt / len(all_pnl_s1) * 100
        print(f"  [{lo:>+6.0%}, {hi:>+6.0%})  {cnt:>5}  ({pct:4.1f}%)  {bar}", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # STUDY 2 — Capitulation deep dive
    # ════════════════════════════════════════════════════════════════════════
    print(flush=True)
    print(SEP, flush=True)
    print("  STUDY 2 — Capitulation Setup Deep Dive", flush=True)
    print("  Setup: dn15% from 20d-high + volume expansion (vol > 1.5x 20d avg)", flush=True)
    print(SEP, flush=True)

    cap_all   = [r for r in all_sigs    if r["pb_dn15"] and r["trig_volexp"]]
    cap_large = [r for r in sigs_by_u["large"] if r["pb_dn15"] and r["trig_volexp"]]
    cap_mid   = [r for r in sigs_by_u["mid"]   if r["pb_dn15"] and r["trig_volexp"]]
    cap_small = [r for r in sigs_by_u["small"]  if r["pb_dn15"] and r["trig_volexp"]]

    print(flush=True)
    print(f"  Total capitulation signals: {len(cap_all):,}  "
          f"(large={len(cap_large)}, mid={len(cap_mid)}, small={len(cap_small)})", flush=True)
    print(flush=True)

    # Grid over all combos
    print("  Full grid — capitulation (dn15 + volexp):", flush=True)
    print(f"  {'Universe':<8}  {'Quality':<12}  {'Hold':>4}  {'Exit':<10}  "
          f"{'n':>5}  {'WR':>5}  {'AvgW':>7}  {'AvgL':>7}  {'Exp':>8}  {'Score':>7}", flush=True)
    print(f"  {'-'*93}", flush=True)

    cap_results = []
    for u_name, u_sigs in [("all", cap_all), ("large", cap_large),
                             ("mid", cap_mid), ("small", cap_small)]:
        for q_name, q_fn in [("none", lambda r: True),
                               ("above21", lambda r: r["q_above21"]),
                               ("above50", lambda r: r["q_above50"])]:
            for hold in [2, 5, 10, 20]:
                for exit_rule in ["time", "trail5", "target15"]:
                    subset = [r for r in u_sigs if q_fn(r)]
                    pnls = [p for r in subset
                            for p in [sim_exit(r["entry"], r["fwd_h"], r["fwd_l"],
                                               r["fwd_c"], hold, exit_rule)]
                            if p is not None]
                    st = calc_stats(pnls)
                    if not st or st["n"] < 20: continue
                    cap_results.append({
                        "universe": u_name, "quality": q_name,
                        "hold": hold, "exit": exit_rule, **st
                    })
                    print(f"  {u_name:<8}  {q_name:<12}  {hold:>4}d  {exit_rule:<10}  "
                          f"{st['n']:>5}  {st['wr']:>4.0%}  "
                          f"{st['avg_w']:>+6.1%}  {st['avg_l']:>+6.1%}  "
                          f"{st['exp']:>+7.2%}  {st['score']:>7.2f}", flush=True)

    # Rank capitulation combos by score
    if cap_results:
        df_cap = pd.DataFrame(cap_results).sort_values("score", ascending=False)
        print(flush=True)
        print("  Capitulation combos ranked by score:", flush=True)
        print(f"  {'#':<4}  {'Universe':<8}  {'Quality':<10}  {'Hold':>4}  {'Exit':<10}  "
              f"{'n':>5}  {'WR':>5}  {'Exp':>8}  {'Score':>7}", flush=True)
        print(f"  {'-'*75}", flush=True)
        for rank, (_, r) in enumerate(df_cap.iterrows(), 1):
            print(f"  {rank:<4}  {r['universe']:<8}  {r['quality']:<10}  "
                  f"{int(r['hold']):>4}d  {r['exit']:<10}  "
                  f"{int(r['n']):>5}  {r['wr']:>4.0%}  "
                  f"{r['exp']:>+7.2%}  {r['score']:>7.2f}", flush=True)
        print(flush=True)

        # Best capitulation combo — monthly breakdown + trade list
        best_cap = df_cap.iloc[0]
        print(f"  Best capitulation combo: {best_cap['universe']} | "
              f"{best_cap['quality']} | {int(best_cap['hold'])}d | {best_cap['exit']}", flush=True)
        print(flush=True)

        if best_cap["universe"] == "all":
            cap_best_pool = cap_all
        elif best_cap["universe"] == "large":
            cap_best_pool = cap_large
        elif best_cap["universe"] == "mid":
            cap_best_pool = cap_mid
        else:
            cap_best_pool = cap_small

        q_fn_best = (lambda r: r["q_above21"] if best_cap["quality"] == "above21"
                     else (lambda r: r["q_above50"]) if best_cap["quality"] == "above50"
                     else lambda r: True)
        cap_best_sigs = [r for r in cap_best_pool if q_fn_best(r)]

        cap_pnl_recs = []
        for r in cap_best_sigs:
            p = sim_exit(r["entry"], r["fwd_h"], r["fwd_l"], r["fwd_c"],
                         int(best_cap["hold"]), best_cap["exit"])
            if p is not None:
                cap_pnl_recs.append({
                    "ticker": r["ticker"],
                    "signal_date": r["signal_date"],
                    "month": str(r["signal_date"])[:7],
                    "entry": r["entry"],
                    "pct_from_high": r["pct_from_high"],
                    "pnl": p,
                    "win": p > 0,
                })

        df_cap_best = pd.DataFrame(cap_pnl_recs).sort_values("signal_date")

        # Full trade list
        print("  All trades:", flush=True)
        print(f"  {'Date':<12}  {'Ticker':<7}  {'Pull%':>6}  {'Entry':>7}  {'PnL':>7}  W/L", flush=True)
        print(f"  {'-'*53}", flush=True)
        for _, r in df_cap_best.iterrows():
            wl = "W" if r["win"] else "L"
            print(f"  {str(r['signal_date']):<12}  {r['ticker']:<7}  "
                  f"{r['pct_from_high']:>5.1%}  ${r['entry']:>6.2f}  "
                  f"{r['pnl']*100:>+6.1f}%  {wl}", flush=True)

        # Monthly breakdown
        print(flush=True)
        print("  Monthly breakdown (best cap combo):", flush=True)
        print(f"  {'Month':<9}  {'N':>4}  {'WR':>5}  {'AvgW':>7}  {'AvgL':>7}  {'Exp':>7}  {'StdDev':>8}", flush=True)
        print(f"  {'-'*58}", flush=True)
        for month, grp in sorted(df_cap_best.groupby("month")):
            arr = grp["pnl"].values
            w=arr[arr>0]; l=arr[arr<=0]
            wr=float((arr>0).mean()); exp=float(arr.mean())
            sd=float(arr.std()) if len(arr)>1 else 0.0
            mk = "  <-- loss" if exp < 0 else ""
            print(f"  {month:<9}  {len(arr):>4}  {wr:>4.0%}  "
                  f"{float(w.mean()) if len(w) else 0:>+6.1%}  "
                  f"{float(l.mean()) if len(l) else 0:>+6.1%}  "
                  f"{exp:>+6.2%}  {sd:>7.2%}{mk}", flush=True)
        all_pnl_cap = df_cap_best["pnl"].values
        w_c=all_pnl_cap[all_pnl_cap>0]; l_c=all_pnl_cap[all_pnl_cap<=0]
        print(f"  {'TOTAL':<9}  {len(all_pnl_cap):>4}  "
              f"{float((all_pnl_cap>0).mean()):>4.0%}  "
              f"{float(w_c.mean()) if len(w_c) else 0:>+6.1%}  "
              f"{float(l_c.mean()) if len(l_c) else 0:>+6.1%}  "
              f"{float(all_pnl_cap.mean()):>+6.2%}  "
              f"{float(all_pnl_cap.std()):>7.2%}", flush=True)

        # Distribution
        print(flush=True)
        print("  Return distribution — best capitulation combo:", flush=True)
        bins = [(-1,-0.20),(-0.20,-0.10),(-0.10,-0.05),(-0.05,0),
                (0,0.05),(0.05,0.10),(0.10,0.20),(0.20,0.50),(0.50,2.0)]
        for lo, hi in bins:
            cnt = int(((all_pnl_cap >= lo) & (all_pnl_cap < hi)).sum())
            bar = "#" * (cnt // max(1, len(all_pnl_cap)//50))
            pct = cnt / len(all_pnl_cap) * 100
            print(f"  [{lo:>+6.0%}, {hi:>+6.0%})  {cnt:>4}  ({pct:4.1f}%)  {bar}", flush=True)

        # Save cap trades
        df_cap_best.to_csv(BASE_DIR / "cap_trades_best.csv", index=False)

    # Save study 1 trades
    df_s1.to_csv(BASE_DIR / "study1_trades.csv", index=False)
    print(flush=True)
    print("  Saved: study1_trades.csv, cap_trades_best.csv", flush=True)
    print(flush=True)
    print("  Done.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
frd_regime_filter.py
====================
Tests regime filters against the 2-year Strategy G signal set (Jun 2023 – Jun 2025).

Market data (IWM, XBI, SPY, VIX) fetched via yfinance.
Signal logic is identical to frd_2year_backtest.py.

Regime filters tested (individually then in combination):
  1.  IWM < 50d MA
  2.  IWM < 200d MA
  3.  XBI < 50d MA
  4.  XBI < 200d MA
  5.  SPY down > 1% on signal day
  6.  SPY < 50d MA
  7.  VIX > 20
  8.  VIX > 25
  9.  IWM down on signal day
  10. XBI down > 2% on signal day
"""

import sys
import io
import pickle
import warnings
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
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
LOOKBACK_VOL   = 20

SIM_START = date(2023, 6, 1)
SIM_END   = date(2025, 6, 30)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_bars():
    data_start = SIM_START - timedelta(days=LOOKBACK_VOL * 2 + 10)
    days = _bdays(data_start.isoformat(), SIM_END.isoformat())
    bars: dict[str, dict] = defaultdict(dict)
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
            if not (0.50 <= c <= 50.0):
                continue
            if float(row.volume) < 50_000:
                continue
            bars[row.ticker][day] = (
                float(row.open), float(row.high), float(row.low),
                c, float(row.volume),
            )
    print(f"  Loaded {len(loaded)} trading days, {len(bars):,} tickers.", flush=True)
    return bars, loaded


def fetch_market_data():
    """Fetch IWM, XBI, SPY, VIX daily data for MA calculation (extra buffer for 200d MA)."""
    start_str = (SIM_START - timedelta(days=300)).strftime("%Y-%m-%d")
    end_str   = (SIM_END   + timedelta(days=5)).strftime("%Y-%m-%d")
    print(f"  Downloading IWM, XBI, SPY, ^VIX  ({start_str} -> {end_str}) ...", flush=True)
    raw = yf.download(
        ["IWM", "XBI", "SPY", "^VIX"],
        start=start_str,
        end=end_str,
        auto_adjust=True,
        progress=False,
    )
    close = raw["Close"].copy()
    open_ = raw["Open"].copy() if "Open" in raw else None

    market: dict[str, pd.DataFrame] = {}
    for ticker in ["IWM", "XBI", "SPY", "^VIX"]:
        key = "VIX" if ticker == "^VIX" else ticker
        c   = close[ticker].dropna()
        df  = pd.DataFrame({"close": c})
        df["ma50"]  = df["close"].rolling(50).mean()
        df["ma200"] = df["close"].rolling(200).mean()
        if open_ is not None and ticker in open_.columns:
            df["open"] = open_[ticker]
        df["pct_chg"] = df["close"].pct_change()
        df.index = pd.to_datetime(df.index).normalize()
        market[key] = df
        print(f"    {key}: {len(df)} bars, MA50 ready from {df['ma50'].first_valid_index().date()}", flush=True)
    return market


def get_regime_row(market: dict, key: str, sig_date: date) -> pd.Series | None:
    ts = pd.Timestamp(sig_date)
    df = market.get(key)
    if df is None:
        return None
    if ts in df.index:
        return df.loc[ts]
    # fall back to most recent prior row
    prior = df[df.index < ts]
    if prior.empty:
        return None
    return prior.iloc[-1]


# ── Signal building ────────────────────────────────────────────────────────────

def build_signals(bars, all_days):
    sim_day_set = {d for d in all_days if SIM_START <= d <= SIM_END}
    records = []
    for ticker, dmap in bars.items():
        tdates = sorted(dmap.keys())
        if len(tdates) < LOOKBACK_VOL + 4:
            continue
        for di in range(LOOKBACK_VOL + 3, len(tdates)):
            sim_day = tdates[di]
            if sim_day not in sim_day_set:
                continue
            o, h, l, c, v = dmap[sim_day]
            if not (PRICE_MIN <= c <= PRICE_MAX):
                continue
            prec = tdates[:di]
            if len(prec) < LOOKBACK_VOL + 3:
                continue
            prev_close = dmap[prec[-1]][3]
            if c >= prev_close:
                continue
            pct_off_hod = (h - c) / h
            if pct_off_hod < HOD_FADE_MIN:
                continue
            pct_vs_prev = (c - prev_close) / prev_close
            if pct_vs_prev > -DOWN_PCT_MIN:
                continue
            base_close = dmap[prec[-3]][3]
            roll3_gain = (c - base_close) / base_close
            if roll3_gain < GAIN_3D_MIN:
                continue
            vols_20 = [dmap[d][4] for d in prec[-LOOKBACK_VOL:]]
            avg_vol = float(np.mean(vols_20))
            if avg_vol < AVG_VOL_MIN:
                continue
            vol_ratio = v / avg_vol if avg_vol > 0 else 0.0
            if vol_ratio < VOL_RATIO_MIN:
                continue
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
            all_sorted = sorted(dmap.keys())
            d0_idx = all_sorted.index(sim_day)
            if d0_idx + 1 >= len(all_sorted):
                continue
            d1_day = all_sorted[d0_idx + 1]
            d2_day = all_sorted[d0_idx + 2] if d0_idx + 2 < len(all_sorted) else None
            if d1_day not in dmap:
                continue
            d1o, d1h, d1l, d1c, _ = dmap[d1_day]
            d2_bar = dmap.get(d2_day) if d2_day else None
            records.append(dict(
                ticker=ticker, signal_date=sim_day,
                pct_off_hod=pct_off_hod, pct_vs_prev=pct_vs_prev,
                roll3_gain=roll3_gain, streak=streak, vol_ratio=vol_ratio,
                d1_open=d1o, d1_high=d1h, d1_close=d1c,
                d2_open=d2_bar[0] if d2_bar else np.nan,
            ))
    return pd.DataFrame(records)


def apply_exit4(df: pd.DataFrame) -> pd.DataFrame:
    entries  = df["d1_open"].values.astype(float)
    d1_highs = df["d1_high"].values.astype(float)
    d1_close = df["d1_close"].values.astype(float)
    d2_open  = df["d2_open"].values.astype(float)
    stop_px  = entries * (1.0 + STOP_PCT)
    stopped  = d1_highs >= stop_px
    exits    = np.where(
        stopped, stop_px,
        np.where(d1_close < entries, d2_open, d1_close)
    )
    pnl = (entries - exits) / entries
    df = df.copy()
    df["entry"]   = entries
    df["exit"]    = exits
    df["stopped"] = stopped
    df["pnl"]     = pnl
    df["win"]     = pnl > 0
    return df


def stats(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return dict(n=0, wr=np.nan, exp=np.nan, avg_w=np.nan, avg_l=np.nan, stop_r=np.nan)
    pnl  = df["pnl"].values
    wins = pnl[pnl > 0]
    loss = pnl[pnl <= 0]
    return dict(
        n=len(pnl),
        wr=float((pnl > 0).mean()),
        exp=float(pnl.mean()),
        avg_w=float(wins.mean()) if len(wins) else 0.0,
        avg_l=float(loss.mean()) if len(loss) else 0.0,
        stop_r=float(df["stopped"].mean()),
    )


def print_stats_row(label: str, st: dict, n_base: int):
    if st["n"] == 0:
        print(f"  {label:<40}  n=  0  (no signals pass)", flush=True)
        return
    pct = st["n"] / n_base * 100
    print(
        f"  {label:<40}  n={st['n']:>3} ({pct:>4.0f}%)  "
        f"WR={st['wr']:>5.0%}  "
        f"AvgW={st['avg_w']:>+7.2%}  "
        f"AvgL={st['avg_l']:>+7.2%}  "
        f"Exp={st['exp']:>+7.2%}  "
        f"StpR={st['stop_r']:>5.0%}",
        flush=True,
    )


# ── Regime flag builder ────────────────────────────────────────────────────────

def attach_regime_flags(sigs: pd.DataFrame, market: dict) -> pd.DataFrame:
    """Add a boolean column for each regime condition to the signals df."""
    cols = {
        "iwm_lt_ma50":    [],
        "iwm_lt_ma200":   [],
        "xbi_lt_ma50":    [],
        "xbi_lt_ma200":   [],
        "spy_dn_1pct":    [],
        "spy_lt_ma50":    [],
        "vix_gt_20":      [],
        "vix_gt_25":      [],
        "iwm_dn":         [],
        "xbi_dn_2pct":    [],
    }
    for _, row in sigs.iterrows():
        sd = row["signal_date"]
        iwm = get_regime_row(market, "IWM", sd)
        xbi = get_regime_row(market, "XBI", sd)
        spy = get_regime_row(market, "SPY", sd)
        vix = get_regime_row(market, "VIX", sd)

        def safe(series, attr, default=np.nan):
            if series is None:
                return default
            v = series.get(attr, np.nan)
            return float(v) if pd.notna(v) else default

        iwm_c   = safe(iwm, "close")
        iwm_m50 = safe(iwm, "ma50")
        iwm_m200= safe(iwm, "ma200")
        iwm_pct = safe(iwm, "pct_chg")
        xbi_c   = safe(xbi, "close")
        xbi_m50 = safe(xbi, "ma50")
        xbi_m200= safe(xbi, "ma200")
        xbi_pct = safe(xbi, "pct_chg")
        spy_c   = safe(spy, "close")
        spy_m50 = safe(spy, "ma50")
        spy_pct = safe(spy, "pct_chg")
        vix_c   = safe(vix, "close")

        cols["iwm_lt_ma50"].append(  iwm_c   < iwm_m50  if not np.isnan(iwm_c + iwm_m50)   else False)
        cols["iwm_lt_ma200"].append( iwm_c   < iwm_m200 if not np.isnan(iwm_c + iwm_m200)  else False)
        cols["xbi_lt_ma50"].append(  xbi_c   < xbi_m50  if not np.isnan(xbi_c + xbi_m50)   else False)
        cols["xbi_lt_ma200"].append( xbi_c   < xbi_m200 if not np.isnan(xbi_c + xbi_m200)  else False)
        cols["spy_dn_1pct"].append(  spy_pct < -0.01    if not np.isnan(spy_pct)            else False)
        cols["spy_lt_ma50"].append(  spy_c   < spy_m50  if not np.isnan(spy_c + spy_m50)    else False)
        cols["vix_gt_20"].append(    vix_c   > 20.0     if not np.isnan(vix_c)              else False)
        cols["vix_gt_25"].append(    vix_c   > 25.0     if not np.isnan(vix_c)              else False)
        cols["iwm_dn"].append(       iwm_pct < 0.0      if not np.isnan(iwm_pct)            else False)
        cols["xbi_dn_2pct"].append(  xbi_pct < -0.02   if not np.isnan(xbi_pct)            else False)

    for col, vals in cols.items():
        sigs = sigs.copy()
        sigs[col] = vals
    return sigs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80, flush=True)
    print("  Strategy G — Regime Filter Analysis  (Jun 2023 – Jun 2025)", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)

    print("[1/4] Loading daily bars ...", flush=True)
    bars, all_days = load_bars()
    print(flush=True)

    print("[2/4] Fetching market regime data (yfinance) ...", flush=True)
    market = fetch_market_data()
    print(flush=True)

    print("[3/4] Building Strategy G signals ...", flush=True)
    sigs = build_signals(bars, all_days)
    sigs = apply_exit4(sigs).sort_values("signal_date").reset_index(drop=True)
    print(f"  Total signals: {len(sigs)}", flush=True)
    print(flush=True)

    print("[4/4] Attaching regime flags ...", flush=True)
    sigs = attach_regime_flags(sigs, market)
    print(flush=True)

    n_base = len(sigs)
    base_st = stats(sigs)

    # ── Print full signal list with regime state ───────────────────────────────
    print("=" * 130, flush=True)
    print("  ALL SIGNALS WITH REGIME STATE", flush=True)
    print("=" * 130, flush=True)
    hdr = (f"  {'Date':<12}  {'Tkr':<5}  {'PnL':>7}  {'W':<2}  "
           f"{'I<50':>5}  {'I<200':>5}  {'X<50':>5}  {'X<200':>5}  "
           f"{'S>-1%':>5}  {'S<50':>5}  {'V>20':>5}  {'V>25':>5}  "
           f"{'Idn':>4}  {'X>-2%':>5}")
    print(hdr, flush=True)
    print(f"  {'-'*125}", flush=True)
    for _, r in sigs.iterrows():
        def b(col): return "Y" if r[col] else "."
        wl = "W" if r["win"] else "L"
        st_mark = "*" if r["stopped"] else " "
        print(
            f"  {str(r['signal_date']):<12}  {r['ticker']:<5}  "
            f"{r['pnl']*100:>+6.1f}%  {wl}{st_mark}  "
            f"  {b('iwm_lt_ma50'):>3}    {b('iwm_lt_ma200'):>3}    {b('xbi_lt_ma50'):>3}    {b('xbi_lt_ma200'):>3}    "
            f"  {b('spy_dn_1pct'):>3}    {b('spy_lt_ma50'):>3}    {b('vix_gt_20'):>3}    {b('vix_gt_25'):>3}    "
            f"  {b('iwm_dn'):>2}    {b('xbi_dn_2pct'):>3}",
            flush=True,
        )

    # ── Individual filter results ──────────────────────────────────────────────
    print(flush=True)
    print("=" * 100, flush=True)
    print("  INDIVIDUAL REGIME FILTERS", flush=True)
    print("=" * 100, flush=True)
    print(f"  {'Filter':<40}  {'n (% of 32)':>12}  {'WR':>6}  {'AvgW':>8}  {'AvgL':>8}  {'Exp':>8}  {'StpR':>6}", flush=True)
    print(f"  {'-'*95}", flush=True)

    print_stats_row("BASELINE (no filter)", base_st, n_base)
    print(f"  {'-'*95}", flush=True)

    individual_filters = [
        ("1. IWM < 50d MA",           "iwm_lt_ma50"),
        ("2. IWM < 200d MA",          "iwm_lt_ma200"),
        ("3. XBI < 50d MA",           "xbi_lt_ma50"),
        ("4. XBI < 200d MA",          "xbi_lt_ma200"),
        ("5. SPY down >1% on sig day","spy_dn_1pct"),
        ("6. SPY < 50d MA",           "spy_lt_ma50"),
        ("7. VIX > 20",               "vix_gt_20"),
        ("8. VIX > 25",               "vix_gt_25"),
        ("9. IWM down on sig day",    "iwm_dn"),
        ("10. XBI down >2% on sig day","xbi_dn_2pct"),
    ]
    for label, col in individual_filters:
        sub = sigs[sigs[col]]
        print_stats_row(label, stats(sub), n_base)

    # ── Combination filter results ─────────────────────────────────────────────
    print(flush=True)
    print("=" * 100, flush=True)
    print("  COMBINATION REGIME FILTERS", flush=True)
    print("=" * 100, flush=True)
    print(f"  {'Combination':<40}  {'n (% of 32)':>12}  {'WR':>6}  {'AvgW':>8}  {'AvgL':>8}  {'Exp':>8}  {'StpR':>6}", flush=True)
    print(f"  {'-'*95}", flush=True)

    combos = [
        # Two-way AND combos
        ("IWM<50MA AND VIX>20",
         lambda r: r["iwm_lt_ma50"] and r["vix_gt_20"]),
        ("IWM<50MA AND XBI<50MA",
         lambda r: r["iwm_lt_ma50"] and r["xbi_lt_ma50"]),
        ("IWM<50MA AND SPY<50MA",
         lambda r: r["iwm_lt_ma50"] and r["spy_lt_ma50"]),
        ("XBI<50MA AND VIX>20",
         lambda r: r["xbi_lt_ma50"] and r["vix_gt_20"]),
        ("XBI<50MA AND SPY<50MA",
         lambda r: r["xbi_lt_ma50"] and r["spy_lt_ma50"]),
        ("SPY<50MA AND VIX>20",
         lambda r: r["spy_lt_ma50"] and r["vix_gt_20"]),
        ("VIX>20 AND XBI dn>2%",
         lambda r: r["vix_gt_20"] and r["xbi_dn_2pct"]),
        ("VIX>25 AND IWM<50MA",
         lambda r: r["vix_gt_25"] and r["iwm_lt_ma50"]),
        ("IWM dn AND XBI dn>2%",
         lambda r: r["iwm_dn"] and r["xbi_dn_2pct"]),
        ("SPY dn>1% AND VIX>20",
         lambda r: r["spy_dn_1pct"] and r["vix_gt_20"]),
        ("IWM<200MA AND XBI<200MA",
         lambda r: r["iwm_lt_ma200"] and r["xbi_lt_ma200"]),
        ("IWM<200MA AND VIX>20",
         lambda r: r["iwm_lt_ma200"] and r["vix_gt_20"]),

        # Two-way OR combos
        ("IWM<50MA OR XBI<50MA",
         lambda r: r["iwm_lt_ma50"] or r["xbi_lt_ma50"]),
        ("IWM<200MA OR XBI<200MA",
         lambda r: r["iwm_lt_ma200"] or r["xbi_lt_ma200"]),
        ("VIX>20 OR SPY<50MA",
         lambda r: r["vix_gt_20"] or r["spy_lt_ma50"]),

        # Three-way combos
        ("IWM<50MA AND XBI<50MA AND VIX>20",
         lambda r: r["iwm_lt_ma50"] and r["xbi_lt_ma50"] and r["vix_gt_20"]),
        ("IWM<50MA AND XBI<50MA AND SPY<50MA",
         lambda r: r["iwm_lt_ma50"] and r["xbi_lt_ma50"] and r["spy_lt_ma50"]),
        ("XBI<50MA AND VIX>20 AND IWM dn",
         lambda r: r["xbi_lt_ma50"] and r["vix_gt_20"] and r["iwm_dn"]),
        ("SPY<50MA AND VIX>25 AND IWM<200MA",
         lambda r: r["spy_lt_ma50"] and r["vix_gt_25"] and r["iwm_lt_ma200"]),
        ("VIX>20 AND (IWM<50MA OR XBI<50MA)",
         lambda r: r["vix_gt_20"] and (r["iwm_lt_ma50"] or r["xbi_lt_ma50"])),
        ("IWM<50MA AND XBI dn>2% AND VIX>20",
         lambda r: r["iwm_lt_ma50"] and r["xbi_dn_2pct"] and r["vix_gt_20"]),
        ("SPY dn>1% AND XBI dn>2%",
         lambda r: r["spy_dn_1pct"] and r["xbi_dn_2pct"]),
        ("XBI<200MA AND VIX>20",
         lambda r: r["xbi_lt_ma200"] and r["vix_gt_20"]),
    ]

    for label, fn in combos:
        mask = sigs.apply(fn, axis=1)
        sub  = sigs[mask]
        print_stats_row(label, stats(sub), n_base)

    # ── Best filters ranked by expectancy ──────────────────────────────────────
    print(flush=True)
    print("=" * 100, flush=True)
    print("  RANKED BY EXPECTANCY  (n >= 5 only)", flush=True)
    print("=" * 100, flush=True)
    print(f"  {'Filter':<40}  {'n':>4}  {'WR':>6}  {'AvgW':>8}  {'AvgL':>8}  {'Exp':>8}  {'StpR':>6}", flush=True)
    print(f"  {'-'*90}", flush=True)

    all_results = []

    # individual
    for label, col in individual_filters:
        sub = sigs[sigs[col]]
        st  = stats(sub)
        if st["n"] >= 5:
            all_results.append((label, st))

    # combos
    for label, fn in combos:
        mask = sigs.apply(fn, axis=1)
        sub  = sigs[mask]
        st   = stats(sub)
        if st["n"] >= 5:
            all_results.append((label, st))

    # sort by expectancy descending
    all_results.sort(key=lambda x: x[1]["exp"], reverse=True)
    for label, st in all_results:
        print(
            f"  {label:<40}  {st['n']:>4}  {st['wr']:>6.0%}  "
            f"{st['avg_w']:>+8.2%}  {st['avg_l']:>+8.2%}  "
            f"{st['exp']:>+8.2%}  {st['stop_r']:>6.0%}",
            flush=True,
        )

    # ── Best filter deep-dive ──────────────────────────────────────────────────
    if all_results:
        best_label, best_st = all_results[0]
        print(flush=True)
        print("=" * 80, flush=True)
        print(f"  BEST FILTER DEEP-DIVE: {best_label}", flush=True)
        print("=" * 80, flush=True)
        # Rebuild mask for best
        # Find the matching label
        best_sub = None
        for label, col in individual_filters:
            if label == best_label:
                best_sub = sigs[sigs[col]]
                break
        if best_sub is None:
            for label, fn in combos:
                if label == best_label:
                    best_sub = sigs[sigs.apply(fn, axis=1)]
                    break
        if best_sub is not None:
            best_sub = best_sub.sort_values("signal_date")
            print(f"  {'Date':<12}  {'Ticker':<6}  {'PnL':>7}  {'ExitType'}", flush=True)
            print(f"  {'-'*45}", flush=True)
            for _, r in best_sub.iterrows():
                stopped_mark = dmap_exit_type(r)
                wl = "W" if r["win"] else "L"
                st_tag = "*" if r["stopped"] else " "
                print(f"  {str(r['signal_date']):<12}  {r['ticker']:<6}  "
                      f"{r['pnl']*100:>+6.1f}%  {wl}{st_tag}", flush=True)
            print(flush=True)

            # Monthly for best
            best_sub = best_sub.copy()
            best_sub["month"] = pd.to_datetime(best_sub["signal_date"]).dt.strftime("%Y-%m")
            print(f"  Monthly breakdown:", flush=True)
            print(f"  {'Month':<9}  {'N':>4}  {'WR':>6}  {'Exp':>8}  {'StpR':>6}", flush=True)
            print(f"  {'-'*42}", flush=True)
            for month, grp in sorted(best_sub.groupby("month")):
                s2 = stats(grp)
                mk = "  <-- loss" if s2["exp"] < 0 else ""
                print(f"  {month:<9}  {s2['n']:>4}  {s2['wr']:>5.0%}  "
                      f"{s2['exp']:>+7.2%}  {s2['stop_r']:>5.0%}{mk}", flush=True)

    print(flush=True)
    print("  Done.", flush=True)


def dmap_exit_type(r):
    if r["stopped"]:
        return "stop"
    if r["pnl"] > 0:
        return "d2_open"
    return "d1_eod"


if __name__ == "__main__":
    main()

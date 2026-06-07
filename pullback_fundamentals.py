#!/usr/bin/env python3
"""
pullback_fundamentals.py
=======================
Tests 8 fundamental quality filters on the dn8-10% / green-candle / 20d-hold setup.

Filters (individually + combinations of 2-3):
  F1  Revenue growth YoY > 0         (TTM vs prior-year TTM, or MRQ vs Q-4)
  F2  Positive net income OR EPS improving YoY
  F3  Gross margin > 30%             (most recent quarter filed before signal)
  F4  Debt/Equity < 1.0              (most recent quarter)
  F5  Market cap > $500M             (Polygon /v3/reference/tickers)
  F6  Avg daily $ vol > $10M         (proxy for institutional access)
  F7  Not biotech/pharma + zero rev  (SIC code + revenue check)
  F8  Entry price > $5               (from signal bar data)

Data: Polygon grouped daily bars (poly_cache/) + Polygon financials API
      Fundamental data cached per-ticker in fundamentals_cache/
Point-in-time: uses most recent quarterly filing dated before the signal date.
"""

import sys, io, pickle, time, warnings
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import logging
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Suppress 404/rate-limit warnings from polygon_fetcher during bulk fetch
logging.disable(logging.WARNING)

BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "poly_cache"
FUND_DIR  = BASE_DIR / "fundamentals_cache"
FUND_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from polygon_fetcher import _bdays, _get
from etf_filter import get_etf_set, is_etf

# ── Backtest window ────────────────────────────────────────────────────────────
DATA_START    = date(2022, 1, 1)
SIM_START     = date(2023, 1, 1)
SIM_END       = date(2025, 6, 30)
FWD_DAYS      = 22
COOLDOWN_DAYS = 20

# ── Universe thresholds ────────────────────────────────────────────────────────
LARGE_MIN = 50_000_000
MID_MIN   =  5_000_000
MID_MAX   = 50_000_000
SMALL_MIN =    500_000
SMALL_MAX =  5_000_000

# ── Fundamental filter thresholds ─────────────────────────────────────────────
MCAP_MIN  = 500_000_000   # F5
DVOL_MIN  =  10_000_000   # F6
GM_MIN    = 0.30          # F3
DE_MAX    = 1.0           # F4
PRICE_MIN = 5.0           # F8

# SIC codes for bio/pharma companies (used in F7)
BIOPHARMA_SICS = set(
    [str(c) for c in range(2830, 2837)] +   # pharmaceutical manufacturing
    [str(c) for c in range(8731, 8735)] +   # commercial R&D labs
    ["8099", "8011", "8049"]                # health services
)

REQUEST_DELAY = 0.13   # ~7.7 calls/sec


# ════════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
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
    end_buf = SIM_END + timedelta(days=FWD_DAYS * 2)
    all_days_list = _bdays(DATA_START.isoformat(), end_buf.isoformat())
    bars: dict[str, dict[date, tuple]] = defaultdict(dict)
    loaded = []
    print(f"  Loading pkl: {DATA_START} -> {end_buf} ...", flush=True)
    for day in all_days_list:
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
            bars[row.ticker][day] = (float(row.open), float(row.high), float(row.low), c, v, vwap)
    print(f"  {len(loaded)} days, {len(bars):,} tickers.", flush=True)
    return bars, sorted(loaded)


def classify_universe(bars, all_days):
    """Returns (universe_map, dvol_avg_map)."""
    sim_days = [d for d in all_days if SIM_START <= d <= SIM_END]
    ticker_dvols: dict[str, list[float]] = defaultdict(list)
    for day in sim_days:
        for tkr, dmap in bars.items():
            if day in dmap:
                c, v = dmap[day][3], dmap[day][4]
                ticker_dvols[tkr].append(c * v)

    umap: dict[str, str] = {}
    dvol_avg: dict[str, float] = {}
    for tkr, vals in ticker_dvols.items():
        avg = float(np.mean(vals)) if vals else 0.0
        dvol_avg[tkr] = avg
        if len(vals) < 50:
            umap[tkr] = "other"
        elif avg >= LARGE_MIN:
            umap[tkr] = "large"
        elif avg >= MID_MIN:
            umap[tkr] = "mid"
        elif avg >= SMALL_MIN:
            umap[tkr] = "small"
        else:
            umap[tkr] = "other"
    return umap, dvol_avg


def build_spy_r63(bars, all_days):
    spy = bars.get("SPY", {})
    spy_dates = sorted(spy.keys())
    spy_closes = {d: spy[d][3] for d in spy_dates}
    r63: dict[date, float] = {}
    for i, d in enumerate(spy_dates):
        if i >= 63:
            r63[d] = (spy_closes[d] - spy_closes[spy_dates[i - 63]]) / spy_closes[spy_dates[i - 63]]
    return r63


# ════════════════════════════════════════════════════════════════════════════════
# 2. SIGNAL COLLECTION  (dn8-10%, green candle, mid+small only)
# ════════════════════════════════════════════════════════════════════════════════

def collect_signals(bars, all_days, umap, spy_r63):
    """
    Scan mid+small tickers for dn8/dn10 signals with first-green-candle trigger.
    Precomputes the 20d time-exit PnL for each qualifying signal.
    Returns a DataFrame of signal records.
    """
    all_days_idx  = {d: i for i, d in enumerate(all_days)}
    all_days_set  = set(all_days)
    sim_set       = {d for d in all_days if SIM_START <= d <= SIM_END}

    mid_small_tickers = [t for t, u in umap.items() if u in ("mid", "small")]
    records = []

    for ti, ticker in enumerate(mid_small_tickers):
        if (ti + 1) % 1000 == 0:
            print(f"  {ti+1:,}/{len(mid_small_tickers):,} tickers, {len(records):,} signals ...", flush=True)

        dmap = bars.get(ticker, {})
        tdates = sorted(d for d in dmap if d in all_days_set)
        if len(tdates) < 252:
            continue

        n = len(tdates)
        closes = np.array([dmap[d][3] for d in tdates])
        opens  = np.array([dmap[d][0] for d in tdates])
        vols   = np.array([dmap[d][4] for d in tdates])

        ma21   = _rolling_mean(closes, 21)
        ma200  = _rolling_mean(closes, 200)
        avol   = _rolling_mean(vols, 20)
        high20 = _rolling_max(closes, 20)

        stk_r63 = np.full(n, np.nan)
        for i in range(63, n):
            stk_r63[i] = (closes[i] - closes[i - 63]) / closes[i - 63]

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
            dn8  = pct_from_high >= 0.08
            dn10 = pct_from_high >= 0.10

            if not dn8:
                continue

            # Need D+1 bar for entry trigger
            d0_idx = all_days_idx.get(sig_day, -1)
            if d0_idx < 0 or d0_idx + 1 >= len(all_days):
                continue
            d1_day = all_days[d0_idx + 1]
            if d1_day not in dmap:
                continue

            d1o, d1h, d1l, d1c, d1v, _ = dmap[d1_day]
            if d1c <= d1o:          # not green — skip (this study is green-only)
                continue

            # Need 20 forward days for exit
            fwd_c_vals = []
            for fd in range(1, 21):
                fwd_idx = d0_idx + fd + 1
                if fwd_idx >= len(all_days):
                    fwd_c_vals.append(np.nan)
                    continue
                fday = all_days[fwd_idx]
                fwd_c_vals.append(dmap[fday][3] if fday in dmap else np.nan)

            if len(fwd_c_vals) < 20 or np.isnan(fwd_c_vals[19]):
                continue

            pnl_20d = (fwd_c_vals[19] - d1o) / d1o

            # Quality flags
            q_above21 = bool(not np.isnan(ma21[i]) and c >= ma21[i])
            rs_val = stk_r63[i]; spy_val = spy_r63.get(sig_day, np.nan)
            q_rs = bool(not np.isnan(rs_val) and not np.isnan(spy_val) and rs_val > spy_val)

            records.append({
                "ticker":      ticker,
                "signal_date": sig_day,
                "universe":    umap[ticker],
                "entry_open":  d1o,
                "pb_dn8":      True,
                "pb_dn10":     dn10,
                "q_above21":   q_above21,
                "q_rs":        q_rs,
                "pnl_20d":     pnl_20d,
            })

            cooldown_end = all_days[min(d0_idx + COOLDOWN_DAYS, len(all_days) - 1)]

    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════════════
# 3. FUNDAMENTAL FETCHING + CACHING
# ════════════════════════════════════════════════════════════════════════════════

def _val(item):
    """Extract .value from a Polygon financial line-item dict."""
    if isinstance(item, dict):
        return item.get("value")
    return None


def fetch_ticker_fundamentals(ticker: str) -> dict:
    """
    Fetch and cache fundamental data for one ticker.
    Returns dict with keys: market_cap, sic_code, quarters (list newest-first).
    Each quarter: {filing_date, revenue, gross_profit, net_income, eps, liabilities, equity}
    """
    cache_path = FUND_DIR / f"{ticker}.pkl"
    if cache_path.exists():
        try:
            return pickle.load(open(cache_path, "rb"))
        except Exception:
            pass   # corrupt cache: re-fetch

    data: dict = {"market_cap": None, "sic_code": None, "quarters": []}

    # Reference data: market cap + SIC code
    try:
        resp = _get(f"/v3/reference/tickers/{ticker}")
        time.sleep(REQUEST_DELAY)
        res = resp.get("results") or {}
        data["market_cap"] = res.get("market_cap")
        sic = res.get("sic_code")
        data["sic_code"] = str(sic) if sic else None
    except Exception:
        time.sleep(REQUEST_DELAY)

    # Quarterly financials — fetch 20 quarters (~5 years) to cover all signal dates
    try:
        resp = _get("/vX/reference/financials", {
            "ticker": ticker,
            "timeframe": "quarterly",
            "limit": 20,
            "order": "desc",
        })
        time.sleep(REQUEST_DELAY)
        for q in resp.get("results", []):
            fd_str = q.get("filing_date")
            if not fd_str:
                continue
            try:
                filing_dt = date.fromisoformat(fd_str)
            except ValueError:
                continue

            inc = q.get("financials", {}).get("income_statement", {})
            bal = q.get("financials", {}).get("balance_sheet", {})

            rev  = _val(inc.get("revenues"))
            gp   = _val(inc.get("gross_profit"))
            ni   = _val(inc.get("net_income_loss"))
            eps  = (_val(inc.get("diluted_earnings_per_share"))
                    or _val(inc.get("basic_earnings_per_share")))

            # Balance sheet: try several common field names
            liab = (  _val(bal.get("liabilities"))
                   or _val(bal.get("total_liabilities"))
                   or _val(bal.get("current_and_noncurrent_liabilities")))
            eq   = (  _val(bal.get("equity"))
                   or _val(bal.get("stockholders_equity"))
                   or _val(bal.get("equity_attributable_to_parent")))

            data["quarters"].append({
                "filing_date":  filing_dt,
                "revenue":      rev,
                "gross_profit": gp,
                "net_income":   ni,
                "eps":          eps,
                "liabilities":  liab,
                "equity":       eq,
            })
    except Exception:
        time.sleep(REQUEST_DELAY)

    # Sort newest-first
    data["quarters"].sort(key=lambda q: q["filing_date"], reverse=True)

    try:
        pickle.dump(data, open(cache_path, "wb"), protocol=4)
    except Exception:
        pass

    return data


def get_qtrs_before(quarters: list, sig_date: date) -> list:
    """Return quarters filed on or before sig_date, newest first."""
    return [q for q in quarters if q["filing_date"] <= sig_date]


# ════════════════════════════════════════════════════════════════════════════════
# 4. FUNDAMENTAL FLAG COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════

def compute_flags(ticker: str, sig_date: date, entry_price: float,
                  fund_data: dict, dvol_avg: float) -> dict:
    """
    Compute F1-F8 boolean flags for one signal.
    Returns True (pass), False (fail), or None (insufficient data).
    """
    fd = fund_data.get(ticker) or {"market_cap": None, "sic_code": None, "quarters": []}
    qtrs = get_qtrs_before(fd.get("quarters", []), sig_date)

    # Extract non-None sequences (newest first)
    def _seq(key):
        return [q[key] for q in qtrs if q.get(key) is not None]

    revs  = _seq("revenue")
    gps   = _seq("gross_profit")
    nis   = _seq("net_income")
    epss  = _seq("eps")
    liabs = _seq("liabilities")
    eqs   = _seq("equity")

    # ── F1: Revenue growth YoY ─────────────────────────────────────────────────
    f1: bool | None = None
    if len(revs) >= 4:
        if len(revs) >= 8:
            # TTM vs prior-year TTM (more robust, handles seasonality)
            ttm_cur  = sum(revs[:4])
            ttm_prev = sum(revs[4:8])
            f1 = bool(ttm_prev > 0 and ttm_cur > ttm_prev)
        else:
            # MRQ vs same quarter prior year
            f1 = bool(revs[3] > 0 and revs[0] > revs[3])

    # ── F2: Positive net income OR improving EPS YoY ──────────────────────────
    f2: bool | None = None
    if nis:
        pos_ni = nis[0] > 0
        improving_eps = bool(len(epss) >= 4 and epss[0] > epss[3])
        f2 = bool(pos_ni or improving_eps)

    # ── F3: Gross margin > 30% ─────────────────────────────────────────────────
    f3: bool | None = None
    if revs and gps:
        rev0 = revs[0]
        gp0  = gps[0]
        if rev0 and rev0 > 0:
            f3 = bool(gp0 / rev0 >= GM_MIN)

    # ── F4: Debt/Equity < 1.0 ─────────────────────────────────────────────────
    f4: bool | None = None
    if liabs and eqs:
        eq0 = eqs[0]
        if eq0 and eq0 > 0:
            f4 = bool(liabs[0] / eq0 < DE_MAX)

    # ── F5: Market cap > $500M ─────────────────────────────────────────────────
    mcap = fd.get("market_cap")
    f5: bool | None = bool(mcap >= MCAP_MIN) if mcap is not None else None

    # ── F6: Avg daily $ vol > $10M (institutional access proxy) ───────────────
    f6 = bool(dvol_avg >= DVOL_MIN)

    # ── F7: Not biotech/pharma with zero revenue ───────────────────────────────
    sic = fd.get("sic_code") or ""
    is_biopharma  = sic in BIOPHARMA_SICS
    has_revenue   = bool(revs and revs[0] is not None and revs[0] >= 1_000_000)
    f7 = bool(not is_biopharma or has_revenue)

    # ── F8: Entry price > $5 ──────────────────────────────────────────────────
    f8 = bool(entry_price >= PRICE_MIN)

    return {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5, "f6": f6, "f7": f7, "f8": f8}


# ════════════════════════════════════════════════════════════════════════════════
# 5. STATS + GRID
# ════════════════════════════════════════════════════════════════════════════════

FILTER_LONG = {
    "f1": "Revenue growth YoY",
    "f2": "Pos EPS / Improving EPS",
    "f3": "Gross margin >30%",
    "f4": "D/E <1.0",
    "f5": "Mkt cap >$500M",
    "f6": "Avg $ vol >$10M",
    "f7": "Not bio/pharma+0 rev",
    "f8": "Price >$5",
}
FILTER_SHORT = {
    "f1": "RevGrowth",
    "f2": "PosEPS",
    "f3": "GrossM>30",
    "f4": "DE<1",
    "f5": "MCap>500M",
    "f6": "DVol>10M",
    "f7": "NoBioPharma",
    "f8": "Price>5",
}
FILTER_KEYS = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]


def _stats(pnls):
    if not pnls:
        return dict(n=0, wr=np.nan, exp=np.nan, avg_w=np.nan, avg_l=np.nan)
    arr = np.array(pnls)
    w = arr[arr > 0]; l = arr[arr <= 0]
    return dict(
        n=len(arr),
        wr=float((arr > 0).mean()),
        exp=float(arr.mean()),
        avg_w=float(w.mean()) if len(w) else 0.0,
        avg_l=float(l.mean()) if len(l) else 0.0,
    )


def _print_row(label, s, base_n, width=48):
    n_str  = f"{s['n']:>6}" if s['n'] > 0 else "     -"
    pct_str = f"{s['n']/base_n:.0%}" if base_n and s['n'] > 0 else "   -"
    wr_str  = f"{s['wr']:>5.0%}" if not np.isnan(s.get('wr', np.nan)) and s['n'] else "    -"
    aw_str  = f"{s['avg_w']:>+6.1%}" if not np.isnan(s.get('avg_w', np.nan)) and s['n'] else "     -"
    al_str  = f"{s['avg_l']:>+6.1%}" if not np.isnan(s.get('avg_l', np.nan)) and s['n'] else "     -"
    ex_str  = f"{s['exp']:>+6.2%}" if not np.isnan(s.get('exp', np.nan)) and s['n'] else "     -"
    label_trunc = label[:width]
    print(f"  {label_trunc:<{width}}  {n_str}  {pct_str:>5}  {wr_str}  {aw_str}  {al_str}  {ex_str}", flush=True)


def run_filter_grid(df: pd.DataFrame, pb_col: str, label: str):
    """Test all individual + pair + triple combinations for one pullback level."""
    base_df  = df[df[pb_col]].copy()
    base_s   = _stats(base_df["pnl_20d"].dropna().tolist())
    base_n   = base_s["n"]

    HDR = 48
    W   = HDR + 50
    print(f"\n{'='*W}", flush=True)
    print(f"  {label}  (mid+small, green candle, 20d hold, time exit)", flush=True)
    print(f"{'='*W}", flush=True)
    print(f"  {'Filter':<{HDR}}  {'n':>6}  {'%Base':>5}  {'WR':>5}  {'AvgW':>6}  {'AvgL':>6}  {'Exp':>7}", flush=True)
    print(f"  {'-'*(W-2)}", flush=True)
    _print_row("BASELINE (no filter)", base_s, base_n, HDR)
    print(f"  {'-'*(W-2)}", flush=True)

    # Individual filters
    ind_rows = []
    for k in FILTER_KEYS:
        sub = base_df[base_df[k] == True]
        s = _stats(sub["pnl_20d"].dropna().tolist())
        ind_rows.append((FILTER_LONG[k], s))
    ind_rows.sort(key=lambda x: x[1]["exp"] if x[1]["n"] else -99, reverse=True)
    print(f"  -- Individual filters --", flush=True)
    for name, s in ind_rows:
        _print_row(name, s, base_n, HDR)

    # Pairs: top 15 by exp with n >= 10
    pair_rows = []
    for a, b in combinations(FILTER_KEYS, 2):
        sub = base_df[(base_df[a] == True) & (base_df[b] == True)]
        s = _stats(sub["pnl_20d"].dropna().tolist())
        if s["n"] >= 10:
            name = f"{FILTER_SHORT[a]} + {FILTER_SHORT[b]}"
            pair_rows.append((name, s))
    pair_rows.sort(key=lambda x: x[1]["exp"] if x[1]["n"] else -99, reverse=True)
    print(f"  -- Top pairs (n>=10) --", flush=True)
    for name, s in pair_rows[:15]:
        _print_row(name, s, base_n, HDR)

    # Triples: top 10 by exp with n >= 10
    triple_rows = []
    for a, b, c in combinations(FILTER_KEYS, 3):
        sub = base_df[(base_df[a] == True) & (base_df[b] == True) & (base_df[c] == True)]
        s = _stats(sub["pnl_20d"].dropna().tolist())
        if s["n"] >= 10:
            name = f"{FILTER_SHORT[a]}+{FILTER_SHORT[b]}+{FILTER_SHORT[c]}"
            triple_rows.append((name, s))
    triple_rows.sort(key=lambda x: x[1]["exp"] if x[1]["n"] else -99, reverse=True)
    print(f"  -- Top triples (n>=10) --", flush=True)
    for name, s in triple_rows[:10]:
        _print_row(name, s, base_n, HDR)

    print(f"{'='*W}", flush=True)

    # Build full result DataFrame for CSV
    all_rows = [{"filter": "BASELINE", "type": "baseline",
                 "n": base_s["n"], "wr": base_s["wr"],
                 "exp": base_s["exp"], "avg_w": base_s["avg_w"], "avg_l": base_s["avg_l"]}]
    for k in FILTER_KEYS:
        sub = base_df[base_df[k] == True]
        s = _stats(sub["pnl_20d"].dropna().tolist())
        all_rows.append({"filter": FILTER_LONG[k], "type": "individual",
                         "n": s["n"], "wr": s["wr"], "exp": s["exp"],
                         "avg_w": s["avg_w"], "avg_l": s["avg_l"]})
    for a, b in combinations(FILTER_KEYS, 2):
        sub = base_df[(base_df[a] == True) & (base_df[b] == True)]
        s = _stats(sub["pnl_20d"].dropna().tolist())
        if s["n"] >= 5:
            all_rows.append({"filter": f"{FILTER_SHORT[a]}+{FILTER_SHORT[b]}", "type": "pair",
                             "n": s["n"], "wr": s["wr"], "exp": s["exp"],
                             "avg_w": s["avg_w"], "avg_l": s["avg_l"]})
    for a, b, c in combinations(FILTER_KEYS, 3):
        sub = base_df[(base_df[a] == True) & (base_df[b] == True) & (base_df[c] == True)]
        s = _stats(sub["pnl_20d"].dropna().tolist())
        if s["n"] >= 5:
            all_rows.append({"filter": f"{FILTER_SHORT[a]}+{FILTER_SHORT[b]}+{FILTER_SHORT[c]}",
                             "type": "triple",
                             "n": s["n"], "wr": s["wr"], "exp": s["exp"],
                             "avg_w": s["avg_w"], "avg_l": s["avg_l"]})
    return pd.DataFrame(all_rows).sort_values("exp", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    SEP = "=" * 80
    print(SEP, flush=True)
    print("  Pullback Fundamental Quality Layer", flush=True)
    print("  Base: mid+small | dn8-10% | green candle | 20d hold | time exit", flush=True)
    print(SEP, flush=True)
    print(flush=True)

    # ── Load bars ────────────────────────────────────────────────────────────
    print("[1/6] Loading bars ...", flush=True)
    bars, all_days = load_all_bars()
    print(flush=True)

    # ── Classify universe ────────────────────────────────────────────────────
    print("[2/6] Classifying universe + computing avg dollar volume ...", flush=True)
    umap, dvol_avg = classify_universe(bars, all_days)
    counts = {k: sum(1 for v in umap.values() if v == k) for k in ("large", "mid", "small", "other")}
    print(f"  large={counts['large']:,}  mid={counts['mid']:,}  small={counts['small']:,}  other={counts['other']:,}", flush=True)
    etf_set = get_etf_set()
    before = len(umap)
    umap = {t: u for t, u in umap.items() if not is_etf(t, etf_set)}
    dvol_avg = {t: v for t, v in dvol_avg.items() if t in umap}
    print(f"  ETF filter: {before - len(umap):,} tickers removed, {len(umap):,} remain.", flush=True)
    print(flush=True)

    # ── SPY RS reference ─────────────────────────────────────────────────────
    spy_r63 = build_spy_r63(bars, all_days)
    print(f"  SPY RS reference built ({len(spy_r63)} days).", flush=True)
    print(flush=True)

    # ── Collect signals ──────────────────────────────────────────────────────
    print("[3/6] Scanning for dn8/dn10 green-candle signals (mid+small) ...", flush=True)
    df_sigs = collect_signals(bars, all_days, umap, spy_r63)
    n_dn8  = int(df_sigs["pb_dn8"].sum())
    n_dn10 = int(df_sigs["pb_dn10"].sum())
    n_tkrs = df_sigs["ticker"].nunique()
    print(f"  Signals collected: {len(df_sigs):,} total | dn8={n_dn8:,} | dn10={n_dn10:,}", flush=True)
    print(f"  Unique tickers:    {n_tkrs:,}", flush=True)
    print(flush=True)

    # ── Fetch fundamentals ───────────────────────────────────────────────────
    unique_tickers = sorted(df_sigs["ticker"].unique())
    n_total  = len(unique_tickers)
    n_cached = sum(1 for t in unique_tickers if (FUND_DIR / f"{t}.pkl").exists())
    n_needed = n_total - n_cached
    print(f"[4/6] Fetching fundamentals for {n_total:,} tickers "
          f"({n_cached} cached, {n_needed} to fetch) ...", flush=True)
    if n_needed > 0:
        est_min = n_needed * 2 * REQUEST_DELAY / 60
        print(f"  Est. time for API calls: ~{est_min:.0f} min", flush=True)

    fund_data: dict[str, dict] = {}
    for i, tkr in enumerate(unique_tickers):
        if (i + 1) % 500 == 0:
            pct = (i + 1) / n_total
            print(f"  {i+1}/{n_total} ({pct:.0%}) ...", flush=True)
        fund_data[tkr] = fetch_ticker_fundamentals(tkr)

    print(f"  Done. {len(fund_data):,} tickers loaded.", flush=True)
    print(flush=True)

    # ── Attach fundamental flags ─────────────────────────────────────────────
    print("[5/6] Computing fundamental flags ...", flush=True)
    flag_records = []
    for _, row in df_sigs.iterrows():
        flags = compute_flags(
            row["ticker"], row["signal_date"], row["entry_open"],
            fund_data, dvol_avg.get(row["ticker"], 0.0)
        )
        flag_records.append(flags)

    flags_df = pd.DataFrame(flag_records, index=df_sigs.index)
    df = pd.concat([df_sigs, flags_df], axis=1)

    # Coverage stats (how many signals have data for each filter)
    print(f"\n  Filter coverage and pass rates (out of {len(df):,} total signals):", flush=True)
    for k in FILTER_KEYS:
        has_data = df[k].notna().sum()
        passing  = (df[k] == True).sum()
        pct_of_all = passing / len(df) if len(df) else 0
        print(f"    {FILTER_LONG[k]:<30}  data={has_data:>6,}  pass={passing:>6,}  ({pct_of_all:.0%} of all sigs)", flush=True)
    print(flush=True)

    # ── Run grid ─────────────────────────────────────────────────────────────
    print("[6/6] Running filter combination grid ...", flush=True)
    dn8_results  = run_filter_grid(df, "pb_dn8",  "DN8  FUNDAMENTAL FILTER COMPARISON")
    dn10_results = run_filter_grid(df, "pb_dn10", "DN10 FUNDAMENTAL FILTER COMPARISON")

    # Save CSV
    dn8_out  = BASE_DIR / "fund_filter_dn8.csv"
    dn10_out = BASE_DIR / "fund_filter_dn10.csv"
    dn8_results.to_csv(dn8_out,  index=False)
    dn10_results.to_csv(dn10_out, index=False)
    print(f"\n  Saved: {dn8_out.name}  ({len(dn8_results)} rows)", flush=True)
    print(f"  Saved: {dn10_out.name}  ({len(dn10_results)} rows)", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()

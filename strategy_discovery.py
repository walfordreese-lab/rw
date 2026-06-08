# -*- coding: utf-8 -*-
"""
strategy_discovery.py
Autonomous discovery of best trading strategies on mid/large cap US stocks.
Universe : avg daily dollar volume > $10M
In-sample : 2022-01-03 -> 2025-11-28
OOS       : 2025-12-01 -> 2026-06-05  (~6 months)
"""

import os, sys, pickle, warnings, re
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CACHE_DIR      = r"C:\Users\reese\frd_backtest\poly_cache"
OUTPUT_DIR     = r"C:\Users\reese\frd_backtest"
IS_START       = "2022-01-03"
IS_END         = "2025-11-28"
OOS_START      = "2025-12-01"
OOS_END        = "2026-06-05"
MIN_DV         = 10_000_000   # $10M avg daily dollar volume
COST_RT        = 0.0010       # 0.10% round-trip slippage
MIN_TRADES     = 50
WARMUP_DAYS    = 380          # calendar days before IS_START for feature warmup

# ── Data Loading ───────────────────────────────────────────────────────────────

def load_universe(cache_dir, warmup_start, oos_end, min_dv=MIN_DV):
    date_re = re.compile(r"grouped_(\d{4}-\d{2}-\d{2})\.pkl$")
    all_pkls = sorted([
        (m.group(1), os.path.join(cache_dir, f))
        for f in os.listdir(cache_dir)
        if (m := date_re.match(f)) and warmup_start <= m.group(1) <= oos_end
    ])
    print(f"  {len(all_pkls)} trading days  ({all_pkls[0][0]} -> {all_pkls[-1][0]})")

    # Pass 1: avg dollar volume per ticker
    print("  Pass 1: avg dollar volume scan ...")
    dv_sum, dv_cnt = defaultdict(float), defaultdict(int)
    for date_str, pkl_path in all_pkls:
        try:
            with open(pkl_path, "rb") as f:
                df = pickle.load(f)
            if df.empty: continue
            dv = df["close"] * df["volume"]
            for tkr, v in zip(df["ticker"], dv):
                dv_sum[tkr] += v
                dv_cnt[tkr] += 1
        except Exception:
            continue

    qualifying = {t for t, s in dv_sum.items()
                  if dv_cnt[t] >= 120 and s / dv_cnt[t] >= min_dv}
    print(f"  {len(qualifying):,} tickers pass avg DV > ${min_dv/1e6:.0f}M  (n_days >= 120)")

    # Pass 2: per-ticker OHLCV rows
    print("  Pass 2: building per-ticker frames ...")
    rows: dict[str, list] = {t: [] for t in qualifying}
    for date_str, pkl_path in all_pkls:
        try:
            with open(pkl_path, "rb") as f:
                df = pickle.load(f)
            if df.empty: continue
            sub = df[df["ticker"].isin(qualifying)]
            ts  = pd.Timestamp(date_str)
            for r in sub.itertuples(index=False):
                rows[r.ticker].append((ts, r.open, r.high, r.low, r.close, r.volume,
                                       r.vwap if hasattr(r, "vwap") else np.nan))
        except Exception:
            continue

    universe = {}
    for tkr, data in rows.items():
        if len(data) < 200: continue
        df = pd.DataFrame(data, columns=["Date","Open","High","Low","Close","Volume","VWAP"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        universe[tkr] = df
    print(f"  {len(universe):,} tickers with >= 200 bars.")
    return universe


# ── Feature helpers ────────────────────────────────────────────────────────────

def rsi(series, n=14):
    delta  = series.diff()
    gain   = delta.clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    loss   = (-delta.clip(upper=0)).ewm(com=n-1, min_periods=n).mean()
    rs     = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def atr(df, n=14):
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift(1)).abs()
    lpc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def add_features(df):
    c = df["Close"]; v = df["Volume"]
    df["ret1"]   = c.pct_change(1)
    df["ret5"]   = c.pct_change(5)
    df["ret10"]  = c.pct_change(10)
    df["ret21"]  = c.pct_change(21)
    df["ret63"]  = c.pct_change(63)
    df["ret126"] = c.pct_change(126)
    df["ret252"] = c.pct_change(252)
    df["rsi14"]  = rsi(c, 14)
    df["ema10"]  = c.ewm(span=10, adjust=False).mean()
    df["ema30"]  = c.ewm(span=30, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    df["sma20"]  = c.rolling(20).mean()
    df["sma50"]  = c.rolling(50).mean()
    df["high52"] = c.rolling(252).max()
    df["low52"]  = c.rolling(252).min()
    df["vol20"]  = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol20"]
    df["atr14"]  = atr(df, 14)
    df["gap"]    = df["Open"] / c.shift(1) - 1
    df["close_pct_ma20"] = c / df["sma20"] - 1
    df["close_pct_hi52"] = c / df["high52"] - 1
    df["realized_vol20"] = df["ret1"].rolling(20).std() * np.sqrt(252)
    df["dollar_vol"] = c * v
    return df


# ── Generic backtester ─────────────────────────────────────────────────────────

def run_backtest(signals_df, cost_rt=COST_RT):
    """
    signals_df columns: ticker, entry_date, exit_date, direction(+1/-1),
                        entry_price, exit_price
    Returns trades DataFrame with PnL columns added.
    """
    if signals_df.empty:
        return pd.DataFrame()
    df = signals_df.copy()
    df["gross_pnl"] = (df["exit_price"] - df["entry_price"]) / df["entry_price"] * df["direction"]
    df["net_pnl"]   = df["gross_pnl"] - cost_rt
    df["is_win"]    = df["net_pnl"] > 0
    return df


def _daily_portfolio_returns(trades):
    """
    Build a daily equal-weight portfolio return series.

    For each entry_date, the 'daily batch return' is the mean net_pnl of all
    trades entered that day (realized at exit). Compounding these gives a
    realistic equity curve: invest equally in each new batch, reinvest proceeds.
    """
    if trades.empty:
        return pd.Series(dtype=float, name="port_ret")
    daily = (trades.set_index("entry_date")["net_pnl"]
             .groupby(level=0).mean()
             .sort_index())
    daily.name = "port_ret"
    return daily


def metrics(trades, label=""):
    if trades is None or trades.empty:
        return {"label": label, "n": 0}
    p   = trades["net_pnl"]
    n   = len(p)
    wr  = (p > 0).mean()
    avg = p.mean()
    aw  = p[p > 0].mean() if (p > 0).any() else 0.0
    al  = p[p <= 0].mean() if (p <= 0).any() else 0.0
    pf  = (p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9)) if (p <= 0).any() else float("inf")

    # Portfolio-level equity curve (daily avg return per entry-date batch)
    daily = _daily_portfolio_returns(trades)
    eq    = (1 + daily).cumprod()
    dd    = eq / eq.cummax() - 1
    mdd   = dd.min()
    total_ret = float(eq.iloc[-1] - 1)

    span_days = (daily.index.max() - daily.index.min()).days
    years = max(span_days / 365.25, 0.05)
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if total_ret > -1 else -1.0

    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    neg    = daily[daily < 0]
    sortino_den = neg.std() if len(neg) > 1 else 0.0
    sortino = (daily.mean() / sortino_den * np.sqrt(252)) if sortino_den > 0 else float("inf")
    calmar  = ann_ret / abs(mdd) if mdd != 0 else float("inf")

    return dict(
        label=label, n=n, win_rate=wr, avg_pnl=avg,
        avg_win=aw, avg_loss=al, profit_factor=pf,
        total_ret=total_ret, ann_ret=ann_ret,
        sharpe=sharpe, sortino=sortino,
        max_dd=mdd, calmar=calmar,
        best=p.max(), worst=p.min(),
        n_positive_days=int((daily > 0).sum()),
        n_negative_days=int((daily < 0).sum()),
    )


def monthly_breakdown(trades):
    if trades.empty: return pd.DataFrame()
    daily = _daily_portfolio_returns(trades)
    if daily.empty: return pd.DataFrame()
    daily.index = pd.to_datetime(daily.index)
    monthly_ret = daily.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_cnt = (trades.set_index("entry_date")["net_pnl"]
                   .groupby(level=0).count()
                   .resample("ME").sum())
    mb = pd.DataFrame({"total_ret": monthly_ret, "n_trades": monthly_cnt})
    mb.index = mb.index.to_period("M")
    mb["cum_ret"] = (1 + mb["total_ret"]).cumprod() - 1
    return mb


def regime_analysis(trades, spy_df):
    """Tag each trade with SPY regime (bull/bear/neutral)."""
    if trades.empty or spy_df is None: return {}
    spy_ret63 = spy_df["Close"].pct_change(63)
    t = trades.copy()
    regimes = spy_ret63.reindex(t["entry_date"], method="ffill")
    t["regime"] = pd.cut(regimes.values, bins=[-np.inf, -0.05, 0.05, np.inf],
                         labels=["bear","neutral","bull"])
    out = {}
    for reg, grp in t.groupby("regime", observed=True):
        out[reg] = {"n": len(grp), "win_rate": (grp["net_pnl"]>0).mean(),
                    "avg_pnl": grp["net_pnl"].mean(), "total_ret": grp["net_pnl"].sum()}
    return out


# ── Strategy implementations ───────────────────────────────────────────────────
# All strategies: signal at close of day t -> enter open of t+1 -> exit close of t+N

def strategy_momentum_xsec(universe, start, end, lookback=63, skip=5, hold=21, top_pct=0.20):
    """Cross-sectional price momentum. Buy top X% stocks by lookback return."""
    ts, te = pd.Timestamp(start), pd.Timestamp(end)
    # Build close matrix
    closes = {}
    for tkr, df in universe.items():
        sub = df["Close"]
        if len(sub) >= lookback + hold + 10:
            closes[tkr] = sub
    if not closes: return pd.DataFrame()
    close_mat = pd.DataFrame(closes).sort_index()

    # Rebalance on ~monthly frequency (every 21 days)
    rebal_dates = close_mat[(close_mat.index >= ts) & (close_mat.index <= te)].index[::hold]

    records = []
    all_dates = close_mat.index.tolist()

    for sig_date in rebal_dates:
        idx = close_mat.index.get_loc(sig_date)
        if idx < lookback + skip + 1 or idx + hold >= len(all_dates):
            continue
        ret = (close_mat.iloc[idx - skip] / close_mat.iloc[idx - lookback] - 1)
        ret = ret.dropna()
        if len(ret) < 20: continue

        threshold = ret.quantile(1 - top_pct)
        longs = ret[ret >= threshold].index.tolist()

        entry_idx = idx + 1
        exit_idx  = min(idx + hold, len(all_dates) - 1)
        entry_date = all_dates[entry_idx]
        exit_date  = all_dates[exit_idx]

        for tkr in longs:
            ep = close_mat.loc[entry_date, tkr] if entry_date in close_mat.index else np.nan
            xp = close_mat.loc[exit_date, tkr]  if exit_date  in close_mat.index else np.nan
            if np.isnan(ep) or np.isnan(xp) or ep <= 0: continue
            records.append(dict(ticker=tkr, entry_date=entry_date, exit_date=exit_date,
                                direction=1, entry_price=ep, exit_price=xp))
    return pd.DataFrame(records)


def strategy_rsi_mean_rev(universe, start, end, rsi_thresh=30, hold=5, exit_rsi=50):
    """Buy oversold (RSI < rsi_thresh), exit after hold days or RSI recovery."""
    ts, te = pd.Timestamp(start), pd.Timestamp(end)
    records = []
    for tkr, df in universe.items():
        sub = df[(df.index >= ts - pd.Timedelta(days=60)) & (df.index <= te + pd.Timedelta(days=hold+5))].copy()
        if len(sub) < 30: continue
        sub = add_features(sub) if "rsi14" not in sub.columns else sub
        sub2 = sub[(sub.index >= ts) & (sub.index <= te)]
        all_idx = sub.index.tolist()
        for i, (sig_date, row) in enumerate(sub2.iterrows()):
            if pd.isna(row.get("rsi14", np.nan)): continue
            if row["rsi14"] >= rsi_thresh: continue
            # find entry bar in full sub
            full_i = sub.index.get_loc(sig_date)
            if full_i + 1 >= len(all_idx): continue
            entry_date = all_idx[full_i + 1]
            entry_price = sub.loc[entry_date, "Open"] if entry_date in sub.index else np.nan
            if np.isnan(entry_price) or entry_price <= 0: continue
            # find exit: hold days or rsi recovery
            exit_date  = None; exit_price = None
            for j in range(1, hold + 1):
                if full_i + j >= len(all_idx): break
                d  = all_idx[full_i + j]
                rx = sub.loc[d, "rsi14"] if "rsi14" in sub.columns else np.nan
                if j == hold or (not pd.isna(rx) and rx >= exit_rsi):
                    exit_date  = d
                    exit_price = sub.loc[d, "Close"]
                    break
            if exit_date is None or exit_price is None: continue
            records.append(dict(ticker=tkr, entry_date=entry_date, exit_date=exit_date,
                                direction=1, entry_price=entry_price, exit_price=exit_price))
    return pd.DataFrame(records)


def strategy_volume_breakout(universe, start, end, hi_window=20, vol_mult=2.0, hold=10, stop_pct=0.04):
    """Buy when close > N-day high with volume > vol_mult * avg. Exit after hold or stop."""
    ts, te = pd.Timestamp(start), pd.Timestamp(end)
    records = []
    for tkr, df in universe.items():
        sub = df[(df.index >= ts - pd.Timedelta(days=60)) & (df.index <= te + pd.Timedelta(days=hold+3))].copy()
        if len(sub) < hi_window + 5: continue
        c, v = sub["Close"], sub["Volume"]
        rolling_hi = c.shift(1).rolling(hi_window).max()
        vol_avg    = v.shift(1).rolling(hi_window).mean()
        sub2 = sub[(sub.index >= ts) & (sub.index <= te)]
        all_idx = sub.index.tolist()
        for sig_date, row in sub2.iterrows():
            full_i = sub.index.get_loc(sig_date)
            if full_i + 1 >= len(all_idx): continue
            rhi = rolling_hi.iloc[full_i]
            vav = vol_avg.iloc[full_i]
            if pd.isna(rhi) or pd.isna(vav) or vav == 0: continue
            if row["Close"] <= rhi: continue
            if row["Volume"] < vol_mult * vav: continue
            entry_date  = all_idx[full_i + 1]
            entry_price = sub.loc[entry_date, "Open"]
            if entry_price <= 0: continue
            stop_price  = entry_price * (1 - stop_pct)
            exit_date = None; exit_price = None
            for j in range(1, hold + 1):
                if full_i + 1 + j >= len(all_idx): break
                d    = all_idx[full_i + 1 + j]
                dlow = sub.loc[d, "Low"]
                if dlow <= stop_price:
                    exit_date  = d; exit_price = stop_price; break
                if j == hold:
                    exit_date  = d; exit_price = sub.loc[d, "Close"]; break
            if exit_date is None: continue
            records.append(dict(ticker=tkr, entry_date=entry_date, exit_date=exit_date,
                                direction=1, entry_price=entry_price, exit_price=exit_price))
    return pd.DataFrame(records)


def strategy_gap_reversal(universe, start, end, gap_min=0.015, gap_max=0.08):
    """Fade opening gaps: if gap-up > gap_min, short at open, exit at close."""
    ts, te = pd.Timestamp(start), pd.Timestamp(end)
    records = []
    for tkr, df in universe.items():
        sub = df[(df.index >= ts) & (df.index <= te)].copy()
        if len(sub) < 5: continue
        gap = sub["Open"] / sub["Close"].shift(1) - 1
        for sig_date in sub.index:
            g = gap.loc[sig_date]
            if pd.isna(g): continue
            if not (gap_min <= abs(g) <= gap_max): continue
            direction  = -1 if g > 0 else 1   # short gap-up, long gap-down
            entry_price = sub.loc[sig_date, "Open"]
            exit_price  = sub.loc[sig_date, "Close"]
            if entry_price <= 0: continue
            records.append(dict(ticker=tkr, entry_date=sig_date, exit_date=sig_date,
                                direction=direction, entry_price=entry_price, exit_price=exit_price))
    return pd.DataFrame(records)


def strategy_52w_high_proximity(universe, start, end, prox=0.05, hold=10, stop_pct=0.05):
    """Buy stocks within prox% of 52-week high, exit after hold or stop."""
    ts, te = pd.Timestamp(start), pd.Timestamp(end)
    records = []
    for tkr, df in universe.items():
        sub = df[(df.index >= ts - pd.Timedelta(days=420)) & (df.index <= te + pd.Timedelta(days=hold+3))].copy()
        if len(sub) < 260: continue
        c = sub["Close"]
        hi52 = c.shift(1).rolling(252).max()
        sub2 = sub[(sub.index >= ts) & (sub.index <= te)]
        all_idx = sub.index.tolist()
        for sig_date, row in sub2.iterrows():
            full_i = sub.index.get_loc(sig_date)
            if full_i + 1 >= len(all_idx): continue
            h52 = hi52.iloc[full_i]
            if pd.isna(h52) or h52 <= 0: continue
            pct_from_hi = (h52 - row["Close"]) / h52
            if pct_from_hi > prox or pct_from_hi < 0: continue  # too far OR above
            # require positive 21-day momentum
            if full_i < 21: continue
            mom21 = row["Close"] / sub["Close"].iloc[full_i - 21] - 1
            if mom21 <= 0: continue
            entry_date  = all_idx[full_i + 1]
            entry_price = sub.loc[entry_date, "Open"]
            if entry_price <= 0: continue
            stop_price  = entry_price * (1 - stop_pct)
            exit_date = None; exit_price = None
            for j in range(1, hold + 1):
                if full_i + 1 + j >= len(all_idx): break
                d    = all_idx[full_i + 1 + j]
                dlow = sub.loc[d, "Low"]
                if dlow <= stop_price:
                    exit_date = d; exit_price = stop_price; break
                if j == hold:
                    exit_date = d; exit_price = sub.loc[d, "Close"]; break
            if exit_date is None: continue
            records.append(dict(ticker=tkr, entry_date=entry_date, exit_date=exit_date,
                                direction=1, entry_price=entry_price, exit_price=exit_price))
    return pd.DataFrame(records)


def strategy_ema_trend(universe, start, end, fast=10, slow=30, hold=15, stop_pct=0.06):
    """Buy on EMA(fast) crossing above EMA(slow), exit after hold days or stop."""
    ts, te = pd.Timestamp(start), pd.Timestamp(end)
    records = []
    for tkr, df in universe.items():
        sub = df[(df.index >= ts - pd.Timedelta(days=90)) & (df.index <= te + pd.Timedelta(days=hold+3))].copy()
        if len(sub) < slow + 10: continue
        c = sub["Close"]
        ema_f = c.ewm(span=fast, adjust=False).mean()
        ema_s = c.ewm(span=slow, adjust=False).mean()
        cross_up = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
        sub2 = sub[(sub.index >= ts) & (sub.index <= te)]
        all_idx = sub.index.tolist()
        for sig_date in sub2.index[cross_up.reindex(sub2.index, fill_value=False)]:
            full_i = sub.index.get_loc(sig_date)
            if full_i + 1 >= len(all_idx): continue
            entry_date  = all_idx[full_i + 1]
            entry_price = sub.loc[entry_date, "Open"]
            if entry_price <= 0: continue
            stop_price  = entry_price * (1 - stop_pct)
            exit_date = None; exit_price = None
            for j in range(1, hold + 1):
                if full_i + 1 + j >= len(all_idx): break
                d    = all_idx[full_i + 1 + j]
                dlow = sub.loc[d, "Low"]
                if dlow <= stop_price:
                    exit_date = d; exit_price = stop_price; break
                if j == hold:
                    exit_date = d; exit_price = sub.loc[d, "Close"]; break
            if exit_date is None: continue
            records.append(dict(ticker=tkr, entry_date=entry_date, exit_date=exit_date,
                                direction=1, entry_price=entry_price, exit_price=exit_price))
    return pd.DataFrame(records)


def strategy_low_vol_factor(universe, start, end, vol_pct=0.33, hold=21):
    """Buy lowest-volatility tercile stocks, monthly rebalance."""
    ts, te = pd.Timestamp(start), pd.Timestamp(end)
    closes = {tkr: df["Close"] for tkr, df in universe.items() if len(df) >= 250}
    if not closes: return pd.DataFrame()
    close_mat = pd.DataFrame(closes).sort_index()
    vol_mat   = close_mat.pct_change().rolling(20).std()

    rebal_dates = close_mat[(close_mat.index >= ts) & (close_mat.index <= te)].index[::hold]
    all_dates   = close_mat.index.tolist()
    records     = []

    for sig_date in rebal_dates:
        idx = close_mat.index.get_loc(sig_date)
        if idx + hold >= len(all_dates): continue
        v_row = vol_mat.iloc[idx].dropna()
        if len(v_row) < 20: continue
        threshold = v_row.quantile(vol_pct)
        longs = v_row[v_row <= threshold].index.tolist()
        entry_idx  = idx + 1
        exit_idx   = min(idx + hold, len(all_dates) - 1)
        entry_date = all_dates[entry_idx]
        exit_date  = all_dates[exit_idx]
        for tkr in longs:
            ep = close_mat.loc[entry_date, tkr] if entry_date in close_mat.index else np.nan
            xp = close_mat.loc[exit_date, tkr]  if exit_date  in close_mat.index else np.nan
            if np.isnan(ep) or np.isnan(xp) or ep <= 0: continue
            records.append(dict(ticker=tkr, entry_date=entry_date, exit_date=exit_date,
                                direction=1, entry_price=ep, exit_price=xp))
    return pd.DataFrame(records)


# ── Strategy grid ──────────────────────────────────────────────────────────────

STRATEGY_GRID = [
    # (name, fn, kwargs)
    ("Momentum_63_21_top20",   strategy_momentum_xsec,   dict(lookback=63, skip=5, hold=21, top_pct=0.20)),
    ("Momentum_126_21_top20",  strategy_momentum_xsec,   dict(lookback=126, skip=5, hold=21, top_pct=0.20)),
    ("Momentum_63_10_top15",   strategy_momentum_xsec,   dict(lookback=63, skip=5, hold=10, top_pct=0.15)),
    ("RSI_MeanRev_30_5d",      strategy_rsi_mean_rev,    dict(rsi_thresh=30, hold=5, exit_rsi=50)),
    ("RSI_MeanRev_35_3d",      strategy_rsi_mean_rev,    dict(rsi_thresh=35, hold=3, exit_rsi=55)),
    ("VolBreakout_20_2x_10d",  strategy_volume_breakout, dict(hi_window=20, vol_mult=2.0, hold=10, stop_pct=0.04)),
    ("VolBreakout_20_1p5x_5d", strategy_volume_breakout, dict(hi_window=20, vol_mult=1.5, hold=5, stop_pct=0.03)),
    ("GapRev_1p5_8pct",        strategy_gap_reversal,    dict(gap_min=0.015, gap_max=0.08)),
    ("GapRev_2_5pct",          strategy_gap_reversal,    dict(gap_min=0.020, gap_max=0.05)),
    ("Hi52W_5pct_10d",         strategy_52w_high_proximity, dict(prox=0.05, hold=10, stop_pct=0.05)),
    ("Hi52W_3pct_10d",         strategy_52w_high_proximity, dict(prox=0.03, hold=10, stop_pct=0.05)),
    ("EMA_10_30_15d",          strategy_ema_trend,       dict(fast=10, slow=30, hold=15, stop_pct=0.06)),
    ("EMA_10_50_20d",          strategy_ema_trend,       dict(fast=10, slow=50, hold=20, stop_pct=0.06)),
    ("LowVol_33pct_21d",       strategy_low_vol_factor,  dict(vol_pct=0.33, hold=21)),
]


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_metrics(m, prefix=""):
    if m["n"] == 0:
        print(f"{prefix}No trades")
        return
    print(f"{prefix}Trades       : {m['n']}")
    print(f"{prefix}Win rate     : {m['win_rate']:.1%}")
    print(f"{prefix}Avg PnL/trade: {m['avg_pnl']:+.3%}")
    print(f"{prefix}Avg win      : {m['avg_win']:+.3%}   Avg loss: {m['avg_loss']:+.3%}")
    print(f"{prefix}Profit factor: {m['profit_factor']:.2f}")
    print(f"{prefix}Total return : {m['total_ret']:+.2%}")
    print(f"{prefix}Ann. return  : {m['ann_ret']:+.2%}")
    print(f"{prefix}Sharpe       : {m['sharpe']:.3f}")
    print(f"{prefix}Sortino      : {m['sortino']:.3f}")
    print(f"{prefix}Max drawdown : {m['max_dd']:.2%}")
    print(f"{prefix}Calmar       : {m['calmar']:.3f}")
    print(f"{prefix}Best/Worst   : {m['best']:+.2%} / {m['worst']:+.2%}")


def print_monthly(mb, prefix=""):
    if mb.empty: return
    print(f"\n{prefix}Monthly Breakdown:")
    print(f"  {'Month':<10}  {'Return':>8}  {'Trades':>7}  {'Cum':>8}")
    for ym, row in mb.iterrows():
        print(f"  {str(ym):<10}  {row['total_ret']:>+7.2%}  {int(row['n_trades']):>7}  {row['cum_ret']:>+7.2%}")


def print_regime(regime, prefix=""):
    if not regime: return
    print(f"\n{prefix}Regime Sensitivity:")
    for reg, r in regime.items():
        print(f"  {reg:<8} | n={r['n']:>4} | WR={r['win_rate']:.1%} | avg={r['avg_pnl']:+.3%} | total={r['total_ret']:+.2%}")


def save_equity_curves(top3_results, filename="strategy_discovery_equity.png"):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Top 3 Strategies — In-Sample vs OOS Equity Curves", fontsize=13)
    for row_idx, (name, is_trades, oos_trades, is_m, oos_m) in enumerate(top3_results):
        for col_idx, (trades, label) in enumerate([(is_trades, "In-Sample"), (oos_trades, "OOS")]):
            ax = axes[row_idx][col_idx]
            if trades is None or trades.empty:
                ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes)
            else:
                daily = _daily_portfolio_returns(trades)
                eq    = (1 + daily).cumprod() - 1
                ax.plot(range(len(eq)), eq.values * 100, color="steelblue", linewidth=1.2)
                ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
                ax.fill_between(range(len(eq)), eq.values * 100, 0,
                                where=(eq.values >= 0), alpha=0.15, color="green")
                ax.fill_between(range(len(eq)), eq.values * 100, 0,
                                where=(eq.values < 0), alpha=0.15, color="red")
                ax.grid(True, alpha=0.3)
                ax.set_ylabel("Cum. Portfolio Return (%)")
            m = is_m if col_idx == 0 else oos_m
            title_str = f"{name}\n{label}  Sharpe={m.get('sharpe',0):.2f}  WR={m.get('win_rate',0):.1%}  n={m.get('n',0)}"
            ax.set_title(title_str, fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Equity chart -> {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  AUTONOMOUS STRATEGY DISCOVERY — MID/LARGE CAP US STOCKS")
    print(f"  Universe  : avg daily dollar volume > ${MIN_DV/1e6:.0f}M")
    print(f"  In-sample : {IS_START}  ->  {IS_END}")
    print(f"  OOS       : {OOS_START}  ->  {OOS_END}")
    print("=" * 70)

    warmup_start = (pd.Timestamp(IS_START) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    print(f"\n[1/4] Loading universe (warmup from {warmup_start}) ...")
    universe = load_universe(CACHE_DIR, warmup_start, OOS_END, min_dv=MIN_DV)

    # Get SPY for regime analysis
    spy_df = universe.get("SPY") if "SPY" in universe else universe.get("spy")

    print(f"\n[2/4] Pre-computing features for {len(universe)} tickers ...")
    for tkr in list(universe.keys()):
        try:
            universe[tkr] = add_features(universe[tkr])
        except Exception:
            pass

    print(f"\n[3/4] Running {len(STRATEGY_GRID)} strategy variants in-sample ...")
    is_results = []
    for name, fn, kwargs in STRATEGY_GRID:
        print(f"  {name:<35} ... ", end="", flush=True)
        try:
            sig = fn(universe, IS_START, IS_END, **kwargs)
            if sig.empty:
                print("0 trades — skip")
                continue
            trades = run_backtest(sig)
            m = metrics(trades, name)
            n = m["n"]
            if n < MIN_TRADES:
                print(f"{n} trades — below minimum, skip")
                continue
            print(f"{n} trades | IS Sharpe={m['sharpe']:.3f} | WR={m['win_rate']:.1%}")
            is_results.append((name, fn, kwargs, trades, m))
        except Exception as e:
            print(f"ERROR: {e}")

    if not is_results:
        print("No strategies passed minimum trade count. Exiting.")
        return

    is_results.sort(key=lambda x: x[4]["sharpe"], reverse=True)
    print(f"\n  Top in-sample strategies by Sharpe:")
    for rank, (name, _, _, _, m) in enumerate(is_results[:6], 1):
        print(f"  {rank}. {name:<35} Sharpe={m['sharpe']:.3f}  WR={m['win_rate']:.1%}  n={m['n']}")

    print(f"\n[4/4] Validating top strategies OOS ({OOS_START} -> {OOS_END}) ...")
    top3_results = []
    for name, fn, kwargs, is_trades, is_m in is_results[:6]:
        print(f"\n  {name}")
        try:
            oos_sig    = fn(universe, OOS_START, OOS_END, **kwargs)
            oos_trades = run_backtest(oos_sig) if not oos_sig.empty else pd.DataFrame()
            oos_m      = metrics(oos_trades, name + "_OOS")
            print(f"    OOS  n={oos_m['n']}  Sharpe={oos_m.get('sharpe',0):.3f}  WR={oos_m.get('win_rate',0):.1%}")
            top3_results.append((name, is_trades, oos_trades, is_m, oos_m))
        except Exception as e:
            print(f"    OOS error: {e}")
            top3_results.append((name, is_trades, pd.DataFrame(), is_m, {"n":0}))

    # Rank by OOS Sharpe; fallback to IS if OOS has too few trades
    def rank_score(r):
        _, _, oos_t, is_m, oos_m = r
        oos_sharpe = oos_m.get("sharpe", -99) if oos_m.get("n", 0) >= 10 else -99
        return oos_sharpe

    top3_results.sort(key=rank_score, reverse=True)
    top3 = top3_results[:3]

    # ── Full report ──────────────────────────────────────────────────────────

    sep = "=" * 70
    print(f"\n\n{sep}")
    print("  FINAL REPORT: TOP 3 STRATEGIES")
    print(sep)

    for rank, (name, is_trades, oos_trades, is_m, oos_m) in enumerate(top3, 1):
        print(f"\n{'─'*70}")
        print(f"  STRATEGY #{rank}: {name}")
        print(f"{'─'*70}")

        print(f"\n  [IN-SAMPLE: {IS_START} -> {IS_END}]")
        print_metrics(is_m, prefix="    ")

        is_mb = monthly_breakdown(is_trades)
        print_monthly(is_mb, prefix="  ")

        is_reg = regime_analysis(is_trades, spy_df)
        print_regime(is_reg, prefix="  ")

        print(f"\n  [OUT-OF-SAMPLE: {OOS_START} -> {OOS_END}]")
        print_metrics(oos_m, prefix="    ")

        oos_mb = monthly_breakdown(oos_trades)
        print_monthly(oos_mb, prefix="  ")

        oos_reg = regime_analysis(oos_trades, spy_df)
        print_regime(oos_reg, prefix="  ")

        # IS vs OOS consistency check
        is_sharpe  = is_m.get("sharpe", 0)
        oos_sharpe = oos_m.get("sharpe", 0)
        degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe) * 100 if is_sharpe != 0 else 0
        print(f"\n  IS Sharpe: {is_sharpe:.3f}  |  OOS Sharpe: {oos_sharpe:.3f}  |  Degradation: {degradation:.1f}%")

    print(f"\n{sep}")
    save_equity_curves(top3)
    print("\nDone.")


if __name__ == "__main__":
    main()

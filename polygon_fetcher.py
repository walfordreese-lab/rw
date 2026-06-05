# -*- coding: utf-8 -*-
"""
polygon_fetcher.py
==================
Discovers the FRD backtest universe via Polygon.io grouped daily bars,
then returns per-ticker OHLCV DataFrames ready for signal detection.

Algorithm
---------
1. Iterate every US trading day in [start, end] (with a short lookback prefix).
   Pull grouped daily bars for each day (one API call/day, cached as Parquet).
2. Accumulate per-ticker bar lists.  Broad pre-filter keeps only tickers that
   ever traded between $0.50-$50 with volume > 100 K (eliminates most micro-OTC
   noise while preserving delisted names that had their run in range).
3. Apply strict rolling 3-day criteria to find tickers that at any point met:
   price $3-$10, 3-day gain >= 100%, 3-day average volume >= 1 M.
4. Return qualifying tickers with their full OHLCV history.

Caching
-------
poly_cache/grouped_YYYY-MM-DD.parquet  -- one file per trading day
Re-runs skip already-cached files, so subsequent runs are very fast.
"""

import os
import sys
import time
import logging
from datetime import date
from typing import Dict, List, Optional

import pickle

import requests
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY   = "U8xSGmMGDkx1gyq2i9zH1d48Zd5iW6D1"
BASE_URL  = "https://api.polygon.io"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(SCRIPT_DIR, "poly_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Broad pre-filter applied during scan to keep memory manageable
PRE_PRICE_MIN = 0.50
PRE_PRICE_MAX = 50.0
PRE_VOL_MIN   = 100_000

REQUEST_DELAY = 0.13   # ~7 calls/sec  (well inside Polygon's limits on paid plans)
MAX_RETRIES   = 4

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _get(path: str, params: dict | None = None) -> dict:
    """Single GET with exponential-backoff retry on 429 / 5xx."""
    url = BASE_URL + path
    p   = {"apiKey": API_KEY, **(params or {})}
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=p, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = 2 ** (attempt + 2)   # 4s, 8s, 16s …
                print(f"  [rate-limit] sleeping {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504):
                time.sleep(3)
                continue
            _log.warning("HTTP %d  %s", r.status_code, path)
            return {}
        except requests.RequestException as exc:
            _log.warning("request error (attempt %d): %s", attempt + 1, exc)
            time.sleep(2)
    return {}


# ── Date helpers ───────────────────────────────────────────────────────────────

def _bdays(start: str, end: str) -> List[date]:
    """Return US business days (Mon-Fri) between start and end inclusive."""
    return [d.date() for d in pd.bdate_range(start=start, end=end)]


# ── Grouped daily cache ────────────────────────────────────────────────────────

def _gd_cache_path(day: date) -> str:
    return os.path.join(CACHE_DIR, f"grouped_{day.isoformat()}.pkl")


def fetch_grouped_day(day: date) -> pd.DataFrame:
    """
    Fetch (or load from cache) the Polygon grouped daily bars for one date.

    Returns a DataFrame with columns:
        ticker | open | high | low | close | volume | vwap
    Rows with volume == 0 or close == 0 are dropped.
    vwap may be NaN if the API omits the field on some dates.
    """
    cache = _gd_cache_path(day)
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    body = _get(
        f"/v2/aggs/grouped/locale/us/market/stocks/{day.isoformat()}",
        {"adjusted": "true", "include_otc": "false"},
    )
    time.sleep(REQUEST_DELAY)

    def _save(df: pd.DataFrame) -> pd.DataFrame:
        with open(cache, "wb") as fh:
            pickle.dump(df, fh, protocol=4)
        return df

    results = body.get("results", [])
    if not results:
        empty = pd.DataFrame(
            columns=["ticker", "open", "high", "low", "close", "volume", "vwap"]
        )
        return _save(empty)

    raw = pd.DataFrame(results)
    col_map = {
        "T":  "ticker",
        "o":  "open",
        "h":  "high",
        "l":  "low",
        "c":  "close",
        "v":  "volume",
        "vw": "vwap",
    }
    # Keep only columns the API actually returned
    rename = {k: v for k, v in col_map.items() if k in raw.columns}
    df = raw[list(rename)].rename(columns=rename).copy()

    # Ensure required columns exist even if API omits vwap
    for col in ("ticker", "open", "high", "low", "close", "volume"):
        if col not in df.columns:
            _log.warning("grouped day %s missing column '%s'", day, col)
            empty = pd.DataFrame(
                columns=["ticker", "open", "high", "low", "close", "volume", "vwap"]
            )
            return _save(empty)
    if "vwap" not in df.columns:
        df["vwap"] = float("nan")

    df = df[(df["volume"] > 0) & (df["close"] > 0)].copy()
    df.reset_index(drop=True, inplace=True)
    return _save(df)


# ── Main entry point ───────────────────────────────────────────────────────────

def scan_and_fetch(
    start: str,
    end: str,
    price_min: float = 3.0,
    price_max: float = 10.0,
    gain_3d_min: float = 1.00,
    vol_min: float = 1_000_000,
    lookback_days: int = 12,
) -> Dict[str, pd.DataFrame]:
    """
    Scan Polygon grouped daily bars to find every qualifying ticker, then
    return their full OHLCV history ready for FRD signal detection.

    Parameters
    ----------
    start, end      : 'YYYY-MM-DD' backtest window (signals detected here)
    price_min/max   : closing price filter during the pump
    gain_3d_min     : minimum 3-bar rolling gain (1.0 == 100%)
    vol_min         : minimum 3-bar average daily volume
    lookback_days   : calendar days before `start` to include in returned
                      DataFrames so the backtest engine can compute streaks
                      and rolling gains for signals near the window open

    Returns
    -------
    Dict[str, pd.DataFrame]
        Keys are ticker symbols (including delisted names preserved by Polygon).
        Each DataFrame is indexed by date with columns:
            Open | High | Low | Close | Volume
    """
    # Pull enough days before `start` to support the rolling windows
    lk_start = (
        pd.Timestamp(start) - pd.Timedelta(days=lookback_days + 7)
    ).strftime("%Y-%m-%d")

    days = _bdays(lk_start, end)
    total = len(days)

    print(
        f"\n  Scanning grouped daily bars: {lk_start} -> {end}  ({total} trading days)"
    )
    print("  (cached days load instantly; uncached days = 1 API call each)\n")

    # ── Pass 1: accumulate per-ticker bar lists ────────────────────────────────
    ticker_rows: Dict[str, List] = {}

    for i, day in enumerate(days):
        if i % 100 == 0 or i == total - 1:
            print(f"    {i + 1:>4}/{total}  {day}", flush=True)

        gd = fetch_grouped_day(day)

        # Broad pre-filter: keep tickers whose close is in PRE_PRICE range with
        # enough volume to be liquid.  This reduces memory ~10x vs keeping all.
        gd = gd[
            (gd["close"] >= PRE_PRICE_MIN)
            & (gd["close"] <= PRE_PRICE_MAX)
            & (gd["volume"] >= PRE_VOL_MIN)
        ]

        for row in gd.itertuples(index=False):
            tkr = row.ticker
            if tkr not in ticker_rows:
                ticker_rows[tkr] = []
            ticker_rows[tkr].append((
                day,
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume),
            ))

    print(f"\n  {len(ticker_rows):,} unique tickers tracked after broad pre-filter.")
    print("  Applying strict universe criteria (rolling 3-day window) ...")

    # ── Pass 2: apply rolling 3-day filter ────────────────────────────────────
    qualifying: Dict[str, pd.DataFrame] = {}

    for ticker, rows in ticker_rows.items():
        if len(rows) < 4:
            continue

        df = pd.DataFrame(
            rows, columns=["date", "Open", "High", "Low", "Close", "Volume"]
        )
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        close     = df["Close"]
        vol       = df["Volume"]
        roll_gain = close.pct_change(3)
        avg_vol   = vol.rolling(3).mean()

        mask = (
            (close >= price_min)
            & (close <= price_max)
            & (roll_gain >= gain_3d_min)
            & (avg_vol >= vol_min)
        )

        if mask.any():
            qualifying[ticker] = df

    print(f"  {len(qualifying):,} tickers satisfy all universe criteria.\n")
    return qualifying


# ── Optional standalone test ───────────────────────────────────────────────────
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Running quick universe scan: 2024-01-01 -> 2024-03-31")
    result = scan_and_fetch("2024-01-01", "2024-03-31")
    print(f"\nFound {len(result)} qualifying tickers:")
    for tkr, df in sorted(result.items()):
        print(f"  {tkr:<8}  {len(df)} bars  "
              f"price range ${df['Close'].min():.2f}-${df['Close'].max():.2f}")

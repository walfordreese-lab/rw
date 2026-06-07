"""
etf_filter.py
=============
ETF / leveraged-product exclusion filter for pullback scanner and backtests.

Primary:  Polygon reference API — fetches all tickers typed ETF, ETV, ETN, ETS.
Fallback: symbol-level regex for obvious leveraged/inverse patterns (2X, 3X, -1X …).

Cache: fundamentals_cache/etf_tickers.pkl  (auto-built on first use; delete to refresh)
"""

import pickle, re, time, requests
from pathlib import Path

BASE_DIR   = Path(__file__).parent
CACHE_FILE = BASE_DIR / "fundamentals_cache" / "etf_tickers.pkl"
API_KEY    = "U8xSGmMGDkx1gyq2i9zH1d48Zd5iW6D1"
BASE_URL   = "https://api.polygon.io"

# Polygon security type codes that are NOT individual stocks
_ETF_TYPES = ("ETF", "ETV", "ETN", "ETS")

# Ticker-symbol patterns that flag leveraged/inverse products
_SYMBOL_RE = re.compile(r"(\d+[Xx]|[Xx]\d+|-[123][Xx])", re.IGNORECASE)

# Name keywords indicating leveraged/inverse ETFs (used when name is available)
_NAME_KEYWORDS = frozenset([
    "ultra", "ultrapro", "short", "bear", "inverse",
    "2x", "3x", "-1x", "-2x", "-3x",
    "leveraged", "direxion", "proshares",
])


def _fetch_etf_set() -> set:
    """Paginate Polygon reference endpoint for all non-stock security types."""
    etf_tickers: set = set()
    for t_type in _ETF_TYPES:
        url: str | None = f"{BASE_URL}/v3/reference/tickers"
        params = {
            "type":   t_type,
            "market": "stocks",
            "limit":  1000,
            "apiKey": API_KEY,
        }
        page = 0
        while url:
            try:
                if page == 0:
                    r = requests.get(url, params=params, timeout=30)
                else:
                    # next_url already contains all params including apiKey
                    r = requests.get(url, timeout=30)
                data = r.json()
            except Exception as exc:
                print(f"  [etf_filter] request error ({t_type} page {page}): {exc}", flush=True)
                break
            for item in data.get("results", []):
                etf_tickers.add(item["ticker"])
            next_url = data.get("next_url")
            url = next_url if next_url else None
            page += 1
            time.sleep(0.15)
        print(f"  [etf_filter] {t_type}: {page} pages fetched", flush=True)
    return etf_tickers


def get_etf_set(refresh: bool = False) -> frozenset:
    """
    Return frozenset of known ETF/ETV/ETN tickers from Polygon.
    Loads from cache on subsequent calls; fetches from API on first run.
    Pass refresh=True to force a re-fetch from the API.
    """
    if not refresh and CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            return frozenset(pickle.load(f))

    print("  [etf_filter] Fetching ETF ticker list from Polygon API ...", flush=True)
    etf_set = _fetch_etf_set()
    CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(etf_set, f, protocol=4)
    print(f"  [etf_filter] {len(etf_set):,} ETF/ETV/ETN tickers cached.", flush=True)
    return frozenset(etf_set)


def is_etf(ticker: str, etf_set: frozenset | None = None, name: str | None = None) -> bool:
    """
    Return True if the ticker appears to be an ETF or leveraged/inverse product.

    Checks (in order):
      1. Polygon reference type set (primary, most reliable)
      2. Symbol-level regex  (catches tickers like ERX2X, etc.)
      3. Name keyword match  (if name is provided)
    """
    if etf_set is not None and ticker in etf_set:
        return True
    if _SYMBOL_RE.search(ticker):
        return True
    if name:
        nl = name.lower()
        if any(kw in nl for kw in _NAME_KEYWORDS):
            return True
    return False


def filter_tickers(tickers, etf_set: frozenset | None = None) -> set:
    """Return subset of tickers that are NOT ETFs."""
    return {t for t in tickers if not is_etf(t, etf_set)}


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    etf_set = get_etf_set(refresh=True)
    print(f"\nTotal ETF/ETV/ETN tickers in Polygon reference: {len(etf_set):,}")
    sample = sorted(etf_set)[:20]
    print(f"Sample: {sample}")

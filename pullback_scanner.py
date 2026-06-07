#!/usr/bin/env python3
"""
pullback_scanner.py
===================
End-of-day pullback scanner. Run after market close (4:30 PM ET or later).

Finds US equities meeting all of:
  - Avg daily $ vol $500K–$50M  (mid/small cap proxy)
  - Close above 21-day MA
  - Close down 8%+ from 20-day rolling high (close-to-close)
  - Today's candle closed green (close > open)

Outputs:
  - Console table sorted by % off 20-day high (deepest pullback first)
  - scanner_results/YYYY-MM-DD.csv
  - HTML email to RECIPIENT with full list and next-day entry note

Usage:
  python pullback_scanner.py                       # scan most recent trading day
  python pullback_scanner.py --date 2025-06-04     # scan a specific date

Email setup (one-time):
  1. Go to myaccount.google.com → Security → 2-Step Verification → App passwords
  2. Create app password for "Mail" / "Windows Computer"
  3. Set environment variable before running:
       set GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
     OR create a file named .env in this directory containing:
       GMAIL_APP_PASSWORD=xxxxxxxxxxxx

Historical edge (2023-2025 backtest, n=8,434):
  dn10%+ | above 21MA | green candle | 20d hold → +7.53% avg expectancy
"""

import sys, io, os, pickle, argparse, smtplib, warnings
from datetime import date, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR    = Path(__file__).parent
CACHE_DIR   = BASE_DIR / "poly_cache"
RESULTS_DIR = BASE_DIR / "scanner_results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from polygon_fetcher import _bdays, fetch_grouped_day
from etf_filter import get_etf_set, is_etf

# ── Load .env if present ────────────────────────────────────────────────────────
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Email config ────────────────────────────────────────────────────────────────
GMAIL_FROM        = os.environ.get("GMAIL_FROM", "walfordreese@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
RECIPIENT         = "walfordreese@gmail.com"

# ── Scanner thresholds ──────────────────────────────────────────────────────────
DVOL_MIN         = 500_000       # $500K min avg daily $ vol
DVOL_MAX         = 50_000_000    # $50M  max avg daily $ vol
PCT_OFF_HIGH_MIN = 0.10          # 10% minimum pullback from 20d high
MA_PERIOD        = 21            # MA used for quality filter
HIGH_PERIOD      = 20            # rolling high lookback
AVOL_PERIOD      = 20            # bars for avg volume calculation
DVOL_PERIOD      = 20            # bars for dollar volume classification
HISTORY_DAYS     = 90            # calendar days of history to load
MIN_BARS         = 25            # minimum history bars required per ticker


# ════════════════════════════════════════════════════════════════════════════════
# DATE HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def prev_business_days(ref_date: date, n: int) -> list[date]:
    """Return the n business days immediately before ref_date (not including it)."""
    start = ref_date - timedelta(days=n * 2 + 10)
    all_days = _bdays(start.isoformat(), (ref_date - timedelta(days=1)).isoformat())
    return sorted(all_days)[-n:]


def next_business_day(d: date) -> date:
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def find_scan_date(requested: date | None) -> date:
    """
    Return the scan date to use.
    If requested is provided, use that.
    Otherwise walk back from today to find the most recent day with cached data.
    """
    if requested:
        return requested
    today = date.today()
    for delta in range(7):
        candidate = today - timedelta(days=delta)
        if candidate.weekday() >= 5:      # skip weekends
            continue
        path = CACHE_DIR / f"grouped_{candidate}.pkl"
        if path.exists():
            return candidate
    return today


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════════

def _rolling_mean(arr: np.ndarray, n: int) -> float | None:
    """Return rolling mean of last n values, or None if insufficient data."""
    if len(arr) < n:
        return None
    return float(np.mean(arr[-n:]))


def _rolling_max(arr: np.ndarray, n: int) -> float | None:
    if len(arr) < n:
        return None
    return float(np.max(arr[-n:]))


def load_scan_data(scan_date: date) -> pd.DataFrame:
    """
    Fetch grouped daily bars for scan_date.
    Returns filtered DataFrame (green candles, volume > 0, close > 0).
    """
    print(f"  Fetching grouped bars for {scan_date} ...", flush=True)
    df = fetch_grouped_day(scan_date)
    if df.empty:
        print(f"  WARNING: No data for {scan_date}. Market may not be closed yet "
              f"or this is a holiday.", flush=True)
        return df

    # Ensure vwap column exists
    if "vwap" not in df.columns:
        df["vwap"] = df["close"]

    # Green candle pre-filter (close > open)
    df = df[
        (df["close"] > df["open"]) &
        (df["close"] > 0) &
        (df["volume"] > 50_000)
    ].copy()

    print(f"  {len(df):,} tickers with green candles today.", flush=True)
    return df


def load_history(active_tickers: set, history_days: list[date]) -> dict:
    """
    Load pkl cache for history_days, keeping only active_tickers.
    Returns bars[ticker][day] = (open, high, low, close, volume).
    """
    bars: dict[str, dict[date, tuple]] = defaultdict(dict)
    for day in history_days:
        path = CACHE_DIR / f"grouped_{day}.pkl"
        if not path.exists():
            continue
        with open(path, "rb") as f:
            df = pickle.load(f)
        for row in df.itertuples(index=False):
            if row.ticker not in active_tickers:
                continue
            c = float(row.close)
            v = float(row.volume)
            if c <= 0 or v < 1_000:
                continue
            bars[row.ticker][day] = (
                float(row.open), float(row.high), float(row.low), c, v
            )
    return bars


# ════════════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════

def compute_signals(today_df: pd.DataFrame, bars: dict,
                    scan_date: date) -> list[dict]:
    """
    For each ticker in today_df, check all pullback criteria using bar history.
    Returns list of signal dicts for tickers that pass all filters.
    """
    signals = []

    for row in today_df.itertuples(index=False):
        ticker = row.ticker
        today_close = float(row.close)
        today_open  = float(row.open)
        today_high  = float(row.high)
        today_low   = float(row.low)
        today_vol   = float(row.volume)

        hist = bars.get(ticker, {})
        hist_days = sorted(hist.keys())

        if len(hist_days) < MIN_BARS:
            continue

        # Build arrays (historical only, not including today)
        hist_closes = np.array([hist[d][3] for d in hist_days])
        hist_vols   = np.array([hist[d][4] for d in hist_days])

        # All closes including today (for MA and high calculations)
        all_closes = np.append(hist_closes, today_close)
        all_vols   = np.append(hist_vols, today_vol)

        n = len(all_closes)

        # ── 21-day MA (includes today) ────────────────────────────────────────
        ma21 = _rolling_mean(all_closes, MA_PERIOD)
        if ma21 is None:
            continue

        # ── Above 21MA filter ─────────────────────────────────────────────────
        if today_close < ma21:
            continue

        # ── 20-day rolling high (includes today) ─────────────────────────────
        high20 = _rolling_max(all_closes, HIGH_PERIOD)
        if high20 is None or high20 <= 0:
            continue

        # ── Pullback % ────────────────────────────────────────────────────────
        pct_off_high = (high20 - today_close) / high20
        if pct_off_high < PCT_OFF_HIGH_MIN:
            continue

        # ── Avg dollar volume (last DVOL_PERIOD days, history only) ───────────
        dvol_period = min(DVOL_PERIOD, len(hist_closes))
        if dvol_period < 5:
            continue
        avg_dvol = float(np.mean(
            hist_closes[-dvol_period:] * hist_vols[-dvol_period:]
        ))
        if not (DVOL_MIN <= avg_dvol <= DVOL_MAX):
            continue

        # ── Volume ratio (today vs 20d avg) ──────────────────────────────────
        avol_period = min(AVOL_PERIOD, len(hist_vols))
        avg_vol = float(np.mean(hist_vols[-avol_period:]))
        vol_ratio = today_vol / avg_vol if avg_vol > 0 else np.nan

        signals.append({
            "ticker":       ticker,
            "price":        today_close,
            "open":         today_open,
            "pct_off_high": pct_off_high,
            "high_20d":     high20,
            "ma21":         ma21,
            "vol_today":    today_vol,
            "vol_ratio":    vol_ratio,
            "avg_dvol":     avg_dvol,
        })

    # Sort by pct_off_high descending (deepest pullback first)
    signals.sort(key=lambda s: s["pct_off_high"], reverse=True)
    return signals


# ════════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ════════════════════════════════════════════════════════════════════════════════

def dvol_str(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    return f"${v/1_000:.0f}K"


def fmt_signals_table(signals: list[dict]) -> str:
    if not signals:
        return "  No signals today.\n"
    lines = []
    header = (f"  {'Rank':<4}  {'Ticker':<8}  {'Price':>7}  {'%OffHigh':>9}  "
              f"{'20dHigh':>8}  {'VolRatio':>8}  {'21MA':>7}  {'AvgDVol':>8}")
    sep = "  " + "-" * (len(header) - 2)
    lines.append(header)
    lines.append(sep)
    for rank, s in enumerate(signals, 1):
        lines.append(
            f"  {rank:<4}  {s['ticker']:<8}  ${s['price']:>6.2f}  "
            f"{-s['pct_off_high']:>+8.1%}  ${s['high_20d']:>7.2f}  "
            f"{s['vol_ratio']:>6.1f}x  ${s['ma21']:>6.2f}  {dvol_str(s['avg_dvol']):>8}"
        )
    return "\n".join(lines)


def fmt_email_html(signals: list[dict], scan_date: date, next_day: date) -> str:
    n = len(signals)
    if n == 0:
        body_content = "<p><em>No signals found today.</em></p>"
    else:
        rows_html = ""
        for rank, s in enumerate(signals, 1):
            bg = "#f9f9f9" if rank % 2 == 0 else "#ffffff"
            rows_html += (
                f'<tr style="background:{bg}">'
                f"<td align='center'>{rank}</td>"
                f"<td><strong>{s['ticker']}</strong></td>"
                f"<td align='right'>${s['price']:.2f}</td>"
                f"<td align='right' style='color:#c0392b'>{-s['pct_off_high']:+.1%}</td>"
                f"<td align='right'>${s['high_20d']:.2f}</td>"
                f"<td align='right'>{s['vol_ratio']:.1f}x</td>"
                f"<td align='right'>${s['ma21']:.2f}</td>"
                f"<td align='right'>{dvol_str(s['avg_dvol'])}</td>"
                f"</tr>\n"
            )
        body_content = f"""
        <table border="0" cellpadding="8" cellspacing="0"
               style="border-collapse:collapse;font-family:'Courier New',monospace;
                      font-size:13px;width:100%;max-width:820px">
          <thead>
            <tr style="background:#1a252f;color:white;text-align:center">
              <th>Rank</th><th>Ticker</th><th>Price</th><th>% Off High</th>
              <th>20d High</th><th>Vol Ratio</th><th>21MA</th><th>Avg $ Vol</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family:Arial,sans-serif;color:#222;max-width:900px;margin:0 auto;padding:20px">
  <h2 style="color:#1a252f;border-bottom:2px solid #1a252f;padding-bottom:8px">
    Pullback Scanner &mdash; {scan_date.strftime("%B %d, %Y")}
  </h2>
  <p style="margin:4px 0">
    <strong>{n} signal{'s' if n != 1 else ''}</strong> found &nbsp;&bull;&nbsp;
    <strong>Entry:</strong> Open on {next_day.strftime("%A, %B %d")} &nbsp;&bull;&nbsp;
    <strong>Hold:</strong> 20 trading days (time exit)
  </p>
  <p style="margin:4px 0;color:#555;font-size:13px">
    Filter: mid/small cap ($500K&ndash;$50M avg $ vol) &bull; above 21MA &bull;
    10%+ off 20d high &bull; green candle
  </p>
  <p style="margin:4px 0;color:#27ae60;font-size:13px">
    <strong>Historical edge (2023&ndash;2025):</strong>
    dn10%+ + above-21MA + green candle + 20d hold &rarr;
    <strong>+7.53% avg expectancy</strong> &nbsp;(n=8,434 trades)
  </p>
  <br>
  {body_content}
  <br>
  <p style="color:#888;font-size:11px;border-top:1px solid #ddd;padding-top:8px">
    Generated by pullback_scanner.py &mdash;
    Entry is at next-day open; confirm liquidity before entering.
    Past backtest results do not guarantee future performance.
  </p>
</body>
</html>"""


def fmt_email_text(signals: list[dict], scan_date: date, next_day: date) -> str:
    n = len(signals)
    header = (f"Pullback Scanner — {scan_date}\n"
              f"{n} signal{'s' if n!=1 else ''} found\n"
              f"Entry: Open on {next_day}\n"
              f"Strategy: mid/small | 10%+ off 20d high | above 21MA | green | 20d hold\n"
              f"Historical edge: +7.53% avg expectancy (2023-2025, n=8,434)\n"
              f"\n{fmt_signals_table(signals)}\n")
    return header


# ════════════════════════════════════════════════════════════════════════════════
# EMAIL
# ════════════════════════════════════════════════════════════════════════════════

def send_email(subject: str, html_body: str, text_body: str,
               recipients: list[str]) -> bool:
    if not GMAIL_APP_PASSWORD:
        print("  [email] GMAIL_APP_PASSWORD not set. Skipping email.", flush=True)
        print("  [email] Set env var GMAIL_APP_PASSWORD or add to .env file.", flush=True)
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_FROM
        msg["To"]      = ", ".join(recipients)
        msg.attach(MIMEText(text_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html",  "utf-8"))

        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(GMAIL_FROM, GMAIL_APP_PASSWORD)
            smtp.sendmail(GMAIL_FROM, recipients, msg.as_string())

        print(f"  [email] Sent to {recipients}", flush=True)
        return True
    except Exception as exc:
        print(f"  [email] FAILED: {exc}", flush=True)
        return False


# ════════════════════════════════════════════════════════════════════════════════
# CSV SAVE
# ════════════════════════════════════════════════════════════════════════════════

def save_csv(signals: list[dict], scan_date: date) -> Path:
    out_path = RESULTS_DIR / f"{scan_date}.csv"
    if not signals:
        pd.DataFrame().to_csv(out_path, index=False)
        return out_path

    rows = []
    for rank, s in enumerate(signals, 1):
        rows.append({
            "rank":         rank,
            "ticker":       s["ticker"],
            "scan_date":    str(scan_date),
            "price":        round(s["price"], 2),
            "pct_off_high": round(-s["pct_off_high"] * 100, 2),
            "high_20d":     round(s["high_20d"], 2),
            "ma21":         round(s["ma21"], 2),
            "vol_today":    int(s["vol_today"]),
            "vol_ratio":    round(s["vol_ratio"], 2),
            "avg_dvol":     int(s["avg_dvol"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return out_path


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="End-of-day pullback scanner (mid/small | dn8%+ | above-21MA | green)"
    )
    parser.add_argument(
        "--date", metavar="YYYY-MM-DD", default=None,
        help="Scan date (default: most recent cached trading day)"
    )
    args = parser.parse_args()
    requested = date.fromisoformat(args.date) if args.date else None

    SEP = "=" * 72
    print(SEP, flush=True)
    print("  Pullback Scanner — End-of-Day", flush=True)
    print(SEP, flush=True)
    print(flush=True)

    # ── Determine scan date ─────────────────────────────────────────────────
    scan_date = find_scan_date(requested)
    next_day  = next_business_day(scan_date)
    print(f"  Scan date : {scan_date}", flush=True)
    print(f"  Entry day : {next_day} (next business day open)", flush=True)
    print(flush=True)

    # ── Load today's grouped bars ────────────────────────────────────────────
    today_df = load_scan_data(scan_date)
    if today_df.empty:
        print("  No data available. Exiting.", flush=True)
        return

    # ── ETF exclusion ────────────────────────────────────────────────────────
    etf_set = get_etf_set()
    before_etf = len(today_df)
    today_df = today_df[~today_df["ticker"].apply(lambda t: is_etf(t, etf_set))].copy()
    etf_removed = before_etf - len(today_df)
    if etf_removed:
        print(f"  {etf_removed} ETF/leveraged tickers excluded.", flush=True)

    active_tickers = set(today_df["ticker"].tolist())
    print(f"  {len(active_tickers):,} active tickers (green candles) to evaluate.", flush=True)

    # ── Load price history ───────────────────────────────────────────────────
    end_hist = scan_date - timedelta(days=1)
    start_hist = scan_date - timedelta(days=HISTORY_DAYS)
    history_days = _bdays(start_hist.isoformat(), end_hist.isoformat())

    cached = sum(1 for d in history_days
                 if (CACHE_DIR / f"grouped_{d}.pkl").exists())
    print(f"  Loading {cached} days of price history ...", flush=True)
    bars = load_history(active_tickers, history_days)
    print(f"  History loaded for {len(bars):,} tickers.", flush=True)
    print(flush=True)

    # ── Compute signals ──────────────────────────────────────────────────────
    print("  Applying filters ...", flush=True)
    signals = compute_signals(today_df, bars, scan_date)
    print(f"  {len(signals)} signal(s) passed all filters.", flush=True)
    print(flush=True)

    # ── Console output ───────────────────────────────────────────────────────
    print(SEP, flush=True)
    print(f"  PULLBACK SIGNALS — {scan_date}  ({len(signals)} found)", flush=True)
    print(f"  Entry: Open on {next_day}", flush=True)
    print(SEP, flush=True)
    if signals:
        print(fmt_signals_table(signals), flush=True)
    else:
        print("  No signals today.", flush=True)
    print(SEP, flush=True)
    print(flush=True)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = save_csv(signals, scan_date)
    print(f"  CSV saved: {csv_path}", flush=True)

    # ── Send email ───────────────────────────────────────────────────────────
    n = len(signals)
    subject = (f"Pullback Scanner: {n} signal{'s' if n!=1 else ''} — {scan_date} "
               f"(entry {next_day})")
    html  = fmt_email_html(signals, scan_date, next_day)
    text  = fmt_email_text(signals, scan_date, next_day)
    send_email(subject, html, text, [RECIPIENT])

    print(flush=True)
    print(f"  Done. {len(signals)} signal(s) on {scan_date}.", flush=True)
    print(SEP, flush=True)


if __name__ == "__main__":
    main()

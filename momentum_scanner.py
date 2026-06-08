#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
momentum_scanner.py
===================
Monthly end-of-day momentum scanner.  Best run on the first trading day of
each month (or any rebalance day you choose).

Strategy: Cross-sectional 6-month momentum — top 20% of universe ranked by
126-day return (skipping most recent 5 days to avoid short-term reversal).

Backtest edge (strategy_discovery.py, Jan 2022 – Jun 2026):
  In-sample  (2022-2025) : Sharpe 5.01 | WR 52.6% | Ann. return +21.6% | MaxDD -21.4%
  Out-of-sample (H1 2026): Sharpe 15.4  | WR 54.1% | Total return +23.6% | MaxDD  -0.4%

Universe:
  - Avg daily dollar volume > $10 M (mid/large cap)
  - ETFs, leveraged, and inverse products excluded
  - Min 130 bars of history required

Output:
  - Console table sorted by momentum score
  - scanner_results/momentum/momentum_YYYY-MM-DD.csv
  - HTML email to walfordreese@gmail.com  subject [MOMENTUM-SCAN]

Usage:
  python momentum_scanner.py               # scan most recent trading day
  python momentum_scanner.py --date 2026-06-02
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

BASE_DIR     = Path(__file__).parent
CACHE_DIR    = BASE_DIR / "poly_cache"
FUND_DIR     = BASE_DIR / "fundamentals_cache"
RESULTS_DIR  = BASE_DIR / "scanner_results" / "momentum"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from polygon_fetcher import _bdays, fetch_grouped_day
from etf_filter import get_etf_set, is_etf

# ── Load .env ─────────────────────────────────────────────────────────────────
_env = BASE_DIR / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

GMAIL_FROM         = os.environ.get("GMAIL_FROM", "walfordreese@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
RECIPIENT          = "walfordreese@gmail.com"

# ── Strategy parameters ────────────────────────────────────────────────────────
MIN_DV          = 10_000_000     # $10M avg daily dollar volume
MIN_BARS        = 130            # minimum history bars
LOOKBACK        = 126            # 6-month momentum window
SKIP            = 5              # skip most recent N days (reversal buffer)
RETURN_3M       = 63             # 3-month return window
RETURN_1M       = 21             # 1-month return window
TOP_PCT         = 0.20           # select top 20% by momentum
DV_PERIOD       = 20             # days for avg dollar-vol calculation
HISTORY_CAL_DAYS = 220           # calendar days of history to load (covers ~150 bdays)

# ── Position sizing (informational only for momentum) ─────────────────────────
PORTFOLIO_SIZE  = 60_000
RISK_PER_TRADE  = 600            # 1% of portfolio


def _bdays_list(ref_date: date, n_cal_days: int) -> list[date]:
    start = ref_date - timedelta(days=n_cal_days)
    return [d.date() if hasattr(d, "date") else d
            for d in pd.bdate_range(start.isoformat(), (ref_date - timedelta(days=1)).isoformat())]


def next_bday(d: date) -> date:
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def find_scan_date(requested: date | None) -> date:
    if requested:
        return requested
    today = date.today()
    for delta in range(10):
        c = today - timedelta(days=delta)
        if c.weekday() >= 5:
            continue
        if (CACHE_DIR / f"grouped_{c}.pkl").exists():
            return c
    return today


def is_rebalance_day(scan_date: date) -> bool:
    """True if scan_date is the first or second business day of a month."""
    first = date(scan_date.year, scan_date.month, 1)
    bdays_in_month = [d.date() if hasattr(d, "date") else d
                      for d in pd.bdate_range(first.isoformat(),
                                              scan_date.isoformat())]
    return len(bdays_in_month) <= 2


# ── Market cap lookup (best-effort from fundamentals_cache) ───────────────────
def load_market_caps() -> dict[str, float]:
    caps = {}
    if not FUND_DIR.exists():
        return caps
    for path in FUND_DIR.glob("*.pkl"):
        if path.stem in ("etf_tickers", "index_members"):
            continue
        try:
            d = pickle.load(open(path, "rb"))
            mc = d.get("market_cap")
            if mc and float(mc) > 0:
                caps[path.stem] = float(mc)
        except Exception:
            pass
    return caps


# ── Data loading ───────────────────────────────────────────────────────────────
def load_history(tickers: set, hist_days: list[date]) -> dict[str, dict]:
    """
    Load pkl bars for hist_days, keeping only tickers in the given set.
    Returns bars[ticker][day] = (close, volume)
    """
    bars: dict[str, dict] = defaultdict(dict)
    for day in hist_days:
        path = CACHE_DIR / f"grouped_{day}.pkl"
        if not path.exists():
            continue
        with open(path, "rb") as f:
            df = pickle.load(f)
        for row in df.itertuples(index=False):
            if row.ticker not in tickers:
                continue
            c, v = float(row.close), float(row.volume)
            if c > 0 and v > 0:
                bars[row.ticker][day] = (c, v)
    return bars


# ── Signal computation ─────────────────────────────────────────────────────────
def compute_signals(today_df: pd.DataFrame, bars: dict,
                    mkt_caps: dict) -> list[dict]:
    signals = []
    for row in today_df.itertuples(index=False):
        tkr        = row.ticker
        today_close = float(row.close)
        today_vol   = float(row.volume)

        hist      = bars.get(tkr, {})
        hist_days = sorted(hist.keys())
        n_bars    = len(hist_days)

        if n_bars < MIN_BARS:
            continue

        closes = np.array([hist[d][0] for d in hist_days])
        vols   = np.array([hist[d][1] for d in hist_days])

        # All closes including today
        all_closes = np.append(closes, today_close)
        all_vols   = np.append(vols, today_vol)
        n_all      = len(all_closes)

        # ── Average dollar volume (last DV_PERIOD days, excluding today) ──────
        dv_n = min(DV_PERIOD, len(closes))
        avg_dv = float(np.mean(closes[-dv_n:] * vols[-dv_n:])) if dv_n >= 5 else 0
        if avg_dv < MIN_DV:
            continue

        # ── Momentum score: 126-day return skipping last 5 days ──────────────
        #   base index: n_all - 1 = today's index
        #   recent end: (today - SKIP) = index n_all - 1 - SKIP
        #   base start: (today - SKIP - LOOKBACK) = index n_all - 1 - SKIP - LOOKBACK
        idx_recent = n_all - 1 - SKIP
        idx_base   = n_all - 1 - SKIP - LOOKBACK
        if idx_base < 0 or idx_recent < 0:
            continue

        c_recent = float(all_closes[idx_recent])
        c_base   = float(all_closes[idx_base])
        if c_base <= 0:
            continue
        mom_6m = c_recent / c_base - 1

        # ── 3-month return ────────────────────────────────────────────────────
        idx_3m = n_all - 1 - RETURN_3M
        mom_3m = (float(all_closes[idx_3m] and today_close / all_closes[idx_3m] - 1)
                  if idx_3m >= 0 and all_closes[idx_3m] > 0 else None)

        # ── 1-month return ────────────────────────────────────────────────────
        idx_1m = n_all - 1 - RETURN_1M
        mom_1m = (today_close / float(all_closes[idx_1m]) - 1
                  if idx_1m >= 0 and all_closes[idx_1m] > 0 else None)

        # ── Avg volume (last 20 days) ─────────────────────────────────────────
        avg_vol = float(np.mean(vols[-min(20, len(vols)):]))

        # ── Market cap ────────────────────────────────────────────────────────
        mkt_cap = mkt_caps.get(tkr)

        signals.append(dict(
            ticker    = tkr,
            price     = today_close,
            mom_6m    = mom_6m,
            mom_3m    = mom_3m,
            mom_1m    = mom_1m,
            avg_vol   = avg_vol,
            avg_dv    = avg_dv,
            mkt_cap   = mkt_cap,
        ))

    if not signals:
        return []

    # Rank by 6-month momentum, select top 20%
    signals.sort(key=lambda s: s["mom_6m"], reverse=True)
    cutoff = max(1, int(len(signals) * TOP_PCT))
    return signals[:cutoff]


# ── Formatting ─────────────────────────────────────────────────────────────────
def _mc_str(v) -> str:
    if v is None:
        return "N/A"
    v = float(v)
    if v >= 1e9:  return f"${v/1e9:.1f}B"
    if v >= 1e6:  return f"${v/1e6:.0f}M"
    return f"${v/1e3:.0f}K"

def _dv_str(v: float) -> str:
    if v >= 1e9:  return f"${v/1e9:.2f}B"
    if v >= 1e6:  return f"${v/1e6:.1f}M"
    return f"${v/1e3:.0f}K"

def _pct(v) -> str:
    return f"{v:+.1%}" if v is not None else "N/A"


def fmt_console(signals: list[dict]) -> str:
    if not signals:
        return "  No signals found.\n"
    hdr = (f"  {'Rank':<4}  {'Ticker':<8}  {'Price':>7}  "
           f"{'6M Ret':>8}  {'3M Ret':>8}  {'1M Ret':>8}  "
           f"{'AvgVol':>10}  {'AvgDVol':>9}  {'MktCap':>9}")
    sep = "  " + "-" * (len(hdr) - 2)
    lines = [hdr, sep]
    for rank, s in enumerate(signals, 1):
        lines.append(
            f"  {rank:<4}  {s['ticker']:<8}  ${s['price']:>6.2f}  "
            f"{_pct(s['mom_6m']):>8}  {_pct(s['mom_3m']):>8}  {_pct(s['mom_1m']):>8}  "
            f"{s['avg_vol']:>10,.0f}  {_dv_str(s['avg_dv']):>9}  "
            f"{_mc_str(s['mkt_cap']):>9}"
        )
    return "\n".join(lines)


def fmt_html(signals: list[dict], scan_date: date, n_universe: int) -> str:
    n = len(signals)
    if n == 0:
        body = "<p><em>No signals found today.</em></p>"
    else:
        rows = ""
        for rank, s in enumerate(signals, 1):
            bg   = "#f9f9f9" if rank % 2 == 0 else "#ffffff"
            ret6 = s["mom_6m"]
            color_6m = "#27ae60" if ret6 > 0 else "#e74c3c"
            rows += (
                f'<tr style="background:{bg}">'
                f"<td align='center'>{rank}</td>"
                f"<td><strong>{s['ticker']}</strong></td>"
                f"<td align='right'>${s['price']:.2f}</td>"
                f"<td align='right' style='color:{color_6m}'><strong>{_pct(s['mom_6m'])}</strong></td>"
                f"<td align='right'>{_pct(s['mom_3m'])}</td>"
                f"<td align='right'>{_pct(s['mom_1m'])}</td>"
                f"<td align='right'>{s['avg_vol']:,.0f}</td>"
                f"<td align='right'>{_dv_str(s['avg_dv'])}</td>"
                f"<td align='right'>{_mc_str(s['mkt_cap'])}</td>"
                f"</tr>\n"
            )
        body = f"""
        <table border="0" cellpadding="8" cellspacing="0"
               style="border-collapse:collapse;font-family:'Courier New',monospace;
                      font-size:13px;width:100%;max-width:1000px">
          <thead>
            <tr style="background:#1a3a2a;color:white;text-align:center">
              <th>Rank</th><th>Ticker</th><th>Price</th>
              <th>6M Return</th><th>3M Return</th><th>1M Return</th>
              <th>Avg Volume</th><th>Avg $ Vol</th><th>Mkt Cap</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        <p style="margin:6px 0;color:#555;font-size:12px">
          Universe: {n_universe:,} qualifying tickers &bull;
          Top {TOP_PCT:.0%} selected ({n}) &bull;
          Ranked by 126-day return (skip last 5 days)
        </p>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family:Arial,sans-serif;color:#222;max-width:960px;margin:0 auto;padding:20px">
  <h2 style="color:#1a3a2a;border-bottom:2px solid #1a3a2a;padding-bottom:8px">
    [MOMENTUM-SCAN] &mdash; {scan_date.strftime("%B %d, %Y")}
  </h2>
  <p style="margin:4px 0">
    <strong>{n} stocks</strong> in top {TOP_PCT:.0%} momentum tier &nbsp;&bull;&nbsp;
    <strong>Hold:</strong> 21 trading days (monthly rebalance)
  </p>
  <p style="margin:4px 0;color:#555;font-size:13px">
    Universe: avg daily $ vol &gt; $10M &bull; not ETF &bull;
    ranked by 126-day return (skipping last 5 days)
  </p>
  <p style="margin:4px 0;color:#27ae60;font-size:13px">
    <strong>Backtest edge:</strong>
    Sharpe 5.01 | WR 52.6% | Ann. +21.6% | MaxDD &minus;21.4%
    &nbsp;(in-sample 2022&ndash;2025, n=25,002 trades)
  </p>
  <br>{body}
  <br>
  <p style="color:#888;font-size:11px;border-top:1px solid #ddd;padding-top:8px">
    Generated by momentum_scanner.py &mdash;
    Enter at open on next trading day; equal-weight or size by conviction.
    Past backtest results do not guarantee future performance.
  </p>
</body>
</html>"""


def fmt_text(signals: list[dict], scan_date: date, n_universe: int) -> str:
    return (
        f"[MOMENTUM-SCAN] {scan_date}\n"
        f"Top {TOP_PCT:.0%} of {n_universe} qualifying tickers  ({len(signals)} stocks)\n"
        f"Ranked by 126-day return, skip last 5 days  |  Hold: 21 trading days\n"
        f"Edge: Sharpe 5.01 | WR 52.6% | Ann +21.6% (IS 2022-2025, n=25,002)\n\n"
        f"{fmt_console(signals)}\n"
    )


# ── Email ──────────────────────────────────────────────────────────────────────
def send_email(subject: str, html_body: str, text_body: str) -> bool:
    if not GMAIL_APP_PASSWORD:
        print("  [email] GMAIL_APP_PASSWORD not set — skipping email.", flush=True)
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_FROM
        msg["To"]      = RECIPIENT
        msg.attach(MIMEText(text_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html",  "utf-8"))
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo(); smtp.starttls()
            smtp.login(GMAIL_FROM, GMAIL_APP_PASSWORD)
            smtp.sendmail(GMAIL_FROM, [RECIPIENT], msg.as_string())
        print(f"  [email] Sent to {RECIPIENT}", flush=True)
        return True
    except Exception as exc:
        print(f"  [email] FAILED: {exc}", flush=True)
        return False


# ── CSV ────────────────────────────────────────────────────────────────────────
def save_csv(signals: list[dict], scan_date: date) -> Path:
    out = RESULTS_DIR / f"momentum_{scan_date}.csv"
    rows = []
    for rank, s in enumerate(signals, 1):
        rows.append(dict(
            rank          = rank,
            ticker        = s["ticker"],
            scan_date     = str(scan_date),
            price         = round(s["price"], 2),
            mom_6m_pct    = round(s["mom_6m"] * 100, 2) if s["mom_6m"] is not None else None,
            mom_3m_pct    = round(s["mom_3m"] * 100, 2) if s["mom_3m"] is not None else None,
            mom_1m_pct    = round(s["mom_1m"] * 100, 2) if s["mom_1m"] is not None else None,
            avg_volume    = int(s["avg_vol"]),
            avg_dollar_vol = int(s["avg_dv"]),
            market_cap    = int(s["mkt_cap"]) if s["mkt_cap"] else None,
        ))
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Monthly momentum scanner")
    parser.add_argument("--date", metavar="YYYY-MM-DD", default=None)
    args = parser.parse_args()
    requested = date.fromisoformat(args.date) if args.date else None

    SEP = "=" * 72
    print(SEP, flush=True)
    print("  [MOMENTUM-SCAN] Cross-Sectional 6-Month Momentum", flush=True)
    print("  Strategy #1 from autonomous strategy discovery", flush=True)
    print(SEP, flush=True)

    scan_date = find_scan_date(requested)
    next_day  = next_bday(scan_date)
    print(f"\n  Scan date  : {scan_date}", flush=True)
    print(f"  Entry day  : {next_day} (next open)", flush=True)
    if not is_rebalance_day(scan_date):
        print(f"  NOTE: {scan_date} is not the first business day of the month. "
              f"Run on monthly rebalance day for best results.", flush=True)
    print(flush=True)

    # ── Fetch today's bars ────────────────────────────────────────────────────
    print(f"  Fetching grouped bars for {scan_date} ...", flush=True)
    today_raw = fetch_grouped_day(scan_date)
    if today_raw.empty:
        print("  No data for scan date. Exiting.", flush=True)
        return

    # Broad pre-filter: active stocks with reasonable price and volume
    today_df = today_raw[
        (today_raw["close"] > 1.0) &
        (today_raw["volume"] > 10_000)
    ].copy()
    print(f"  {len(today_df):,} tickers after broad pre-filter.", flush=True)

    # ── ETF exclusion ─────────────────────────────────────────────────────────
    etf_set   = get_etf_set()
    today_df  = today_df[~today_df["ticker"].apply(lambda t: is_etf(t, etf_set))].copy()
    print(f"  {len(today_df):,} tickers after ETF exclusion.", flush=True)

    active = set(today_df["ticker"])

    # ── Load price history ────────────────────────────────────────────────────
    hist_days = _bdays_list(scan_date, HISTORY_CAL_DAYS)
    cached = sum(1 for d in hist_days if (CACHE_DIR / f"grouped_{d}.pkl").exists())
    print(f"  Loading {cached} cached history days "
          f"(need ~{LOOKBACK + SKIP + 10} bars) ...", flush=True)
    bars = load_history(active, hist_days)
    print(f"  History loaded for {len(bars):,} tickers.", flush=True)

    # ── Market caps ───────────────────────────────────────────────────────────
    print("  Loading market caps ...", flush=True)
    mkt_caps = load_market_caps()
    print(f"  Market caps available for {len(mkt_caps):,} tickers.", flush=True)
    print(flush=True)

    # ── Compute momentum signals ──────────────────────────────────────────────
    print("  Computing momentum scores and applying filters ...", flush=True)
    signals = compute_signals(today_df, bars, mkt_caps)
    n_universe = 0
    for _tkr, _hist in bars.items():
        if len(_hist) < MIN_BARS:
            continue
        _days  = sorted(_hist)
        _n     = min(DV_PERIOD, len(_days))
        _c     = np.array([_hist[d][0] for d in _days[-_n:]])
        _v     = np.array([_hist[d][1] for d in _days[-_n:]])
        if float(np.mean(_c * _v)) >= MIN_DV:
            n_universe += 1
    print(f"  {len(signals)} stocks in top {TOP_PCT:.0%} momentum tier "
          f"(universe {n_universe:,} qualifying).", flush=True)
    print(flush=True)

    # ── Console ───────────────────────────────────────────────────────────────
    print(SEP, flush=True)
    print(f"  MOMENTUM SIGNALS — {scan_date}  "
          f"(top {TOP_PCT:.0%}  |  {len(signals)} stocks)", flush=True)
    print(f"  Entry: open on {next_day}  |  Hold: 21 trading days", flush=True)
    print(SEP, flush=True)
    print(fmt_console(signals), flush=True)
    print(SEP, flush=True)
    print(flush=True)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = save_csv(signals, scan_date)
    print(f"  CSV saved: {csv_path}", flush=True)

    # ── Email ─────────────────────────────────────────────────────────────────
    n   = len(signals)
    subj = f"[MOMENTUM-SCAN] {scan_date} — {n} stock{'s' if n!=1 else ''} (entry {next_day})"
    send_email(subj,
               fmt_html(signals, scan_date, n_universe or len(bars)),
               fmt_text(signals, scan_date, n_universe or len(bars)))

    print(f"\n  Done. {n} stock(s) in momentum portfolio for {scan_date}.", flush=True)
    print(SEP, flush=True)


if __name__ == "__main__":
    main()

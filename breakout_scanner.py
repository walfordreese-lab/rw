#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
breakout_scanner.py
===================
Daily end-of-day volume breakout scanner.

Strategy: Volume-confirmed 20-day high breakout — buy when today's close
exceeds the prior 20-day high AND volume is >= 1.5× the 20-day average.
Enter at next open; stop 3% below entry.

Backtest edge (strategy_discovery.py, Jan 2022 – Jun 2026):
  In-sample  (2022-2025) : Sharpe 3.73 | WR 36.8% | PF 1.32 | Ann. +285%  | MaxDD -32%
  Out-of-sample (H1 2026): Sharpe 6.66  | WR 36.2% | PF 1.49 | Total +353% | MaxDD -13%
  Regime-agnostic: WR holds 36-38% in bear, neutral, and bull markets.

Position sizing:  $60,000 portfolio | 1% risk = $600/trade | 3% stop
  → shares = $600 / (entry × 0.03)  |  position $ = shares × entry ≈ $20,000/trade

Universe:
  - Avg daily dollar volume > $10 M (mid/large cap)
  - ETFs, leveraged, and inverse products excluded
  - Min 22 bars of history required

Output:
  - Console table sorted by volume ratio (strongest signal first)
  - scanner_results/breakout/breakout_YYYY-MM-DD.csv
  - HTML email to walfordreese@gmail.com  subject [BREAKOUT-SCAN]

Usage:
  python breakout_scanner.py               # scan most recent trading day
  python breakout_scanner.py --date 2026-06-02
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
RESULTS_DIR  = BASE_DIR / "scanner_results" / "breakout"
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
MIN_BARS        = 22             # minimum bars of history
HI_WINDOW       = 20             # breakout lookback (close-to-close)
VOL_MULT        = 1.5            # minimum volume multiplier vs 20-day avg
STOP_PCT        = 0.03           # 3% below entry
HISTORY_CAL_DAYS = 45            # calendar days of history to load

# ── Position sizing ────────────────────────────────────────────────────────────
PORTFOLIO_SIZE  = 60_000         # $
RISK_PER_TRADE  = 600            # $ (1% of portfolio)
MAX_POSITION    = 20_000         # practical cap: ~1/3 portfolio per trade


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


# ── Data loading ───────────────────────────────────────────────────────────────
def load_history(tickers: set, hist_days: list[date]) -> dict[str, dict]:
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
def compute_signals(today_df: pd.DataFrame, bars: dict) -> list[dict]:
    signals = []

    for row in today_df.itertuples(index=False):
        tkr         = row.ticker
        today_close = float(row.close)
        today_vol   = float(row.volume)

        hist      = bars.get(tkr, {})
        hist_days = sorted(hist.keys())
        n_hist    = len(hist_days)

        if n_hist < MIN_BARS:
            continue

        hist_closes = np.array([hist[d][0] for d in hist_days])
        hist_vols   = np.array([hist[d][1] for d in hist_days])

        # ── Dollar volume filter (last 20 history days) ───────────────────────
        dv_n   = min(20, n_hist)
        avg_dv = float(np.mean(hist_closes[-dv_n:] * hist_vols[-dv_n:]))
        if avg_dv < MIN_DV:
            continue

        # ── 20-day high (excluding today) ─────────────────────────────────────
        hi_n  = min(HI_WINDOW, n_hist)
        hi_20 = float(np.max(hist_closes[-hi_n:]))

        # ── 20-day avg volume (excluding today) ───────────────────────────────
        vol_n   = min(HI_WINDOW, n_hist)
        avg_vol = float(np.mean(hist_vols[-vol_n:]))

        # ── Breakout check ────────────────────────────────────────────────────
        if today_close <= hi_20:
            continue
        vol_ratio = today_vol / avg_vol if avg_vol > 0 else 0.0
        if vol_ratio < VOL_MULT:
            continue

        # ── % above 20-day high ───────────────────────────────────────────────
        pct_above = (today_close - hi_20) / hi_20

        # ── Position sizing ───────────────────────────────────────────────────
        # Entry price = today's close (proxy; actual entry = next-day open)
        entry_est   = today_close
        stop_est    = round(entry_est * (1 - STOP_PCT), 2)
        risk_ps     = round(entry_est * STOP_PCT, 4)
        shares_raw  = int(RISK_PER_TRADE / risk_ps) if risk_ps > 0 else 0
        pos_size    = round(shares_raw * entry_est, 2)

        # Cap at MAX_POSITION — note if truncated
        if pos_size > MAX_POSITION:
            shares_raw = int(MAX_POSITION / entry_est)
            pos_size   = round(shares_raw * entry_est, 2)
            capped     = True
        else:
            capped = False

        pct_port = pos_size / PORTFOLIO_SIZE

        signals.append(dict(
            ticker      = tkr,
            price       = today_close,
            hi_20d      = hi_20,
            pct_above   = pct_above,
            vol_today   = today_vol,
            vol_ratio   = vol_ratio,
            avg_vol     = avg_vol,
            avg_dv      = avg_dv,
            entry_est   = entry_est,
            stop        = stop_est,
            risk_ps     = risk_ps,
            shares      = shares_raw,
            pos_size    = pos_size,
            pct_port    = pct_port,
            capped      = capped,
        ))

    # Sort by volume ratio (strongest conviction first)
    signals.sort(key=lambda s: s["vol_ratio"], reverse=True)
    return signals


# ── Formatting ─────────────────────────────────────────────────────────────────
def _dv_str(v: float) -> str:
    if v >= 1e9:  return f"${v/1e9:.2f}B"
    if v >= 1e6:  return f"${v/1e6:.1f}M"
    return f"${v/1e3:.0f}K"


def fmt_console(signals: list[dict]) -> str:
    if not signals:
        return "  No breakout signals today.\n"
    hdr = (f"  {'Rank':<4}  {'Ticker':<8}  {'Price':>7}  {'20dHigh':>8}  "
           f"{'%Above':>7}  {'VolRatio':>8}  "
           f"{'Entry~':>7}  {'Stop':>7}  {'Risk/Sh':>7}  "
           f"{'Shares':>7}  {'Pos $':>8}  {'%Port':>6}  {'AvgDVol':>9}")
    sep = "  " + "-" * (len(hdr) - 2)
    lines = [hdr, sep]
    for rank, s in enumerate(signals, 1):
        cap_flag = "*" if s["capped"] else " "
        lines.append(
            f"  {rank:<4}  {s['ticker']:<8}  ${s['price']:>6.2f}  "
            f"${s['hi_20d']:>7.2f}  {s['pct_above']:>+6.1%}  "
            f"{s['vol_ratio']:>6.1f}x  "
            f"${s['entry_est']:>6.2f}  ${s['stop']:>6.2f}  ${s['risk_ps']:>6.2f}  "
            f"{s['shares']:>7,}  ${s['pos_size']:>7,.0f}{cap_flag}  {s['pct_port']:>5.1%}  "
            f"{_dv_str(s['avg_dv']):>9}"
        )
    lines.append("")
    lines.append("  * Position capped at $20,000 (natural sizing exceeded portfolio cap)")
    lines.append(f"  Entry shown is today's close — actual entry = next-day open price.")
    lines.append(f"  Stop = estimated entry × {1-STOP_PCT:.0%}  |  Risk = 1% of ${PORTFOLIO_SIZE:,} portfolio = ${RISK_PER_TRADE:,}/trade")
    return "\n".join(lines)


def fmt_html(signals: list[dict], scan_date: date, next_day: date) -> str:
    n = len(signals)
    if n == 0:
        body = "<p><em>No breakout signals today.</em></p>"
    else:
        rows = ""
        for rank, s in enumerate(signals, 1):
            bg      = "#f9f9f9" if rank % 2 == 0 else "#ffffff"
            cap_str = " *" if s["capped"] else ""
            rows += (
                f'<tr style="background:{bg}">'
                f"<td align='center'>{rank}</td>"
                f"<td><strong>{s['ticker']}</strong></td>"
                f"<td align='right'>${s['price']:.2f}</td>"
                f"<td align='right'>${s['hi_20d']:.2f}</td>"
                f"<td align='right' style='color:#27ae60'><strong>{s['pct_above']:+.1%}</strong></td>"
                f"<td align='right' style='color:#2980b9'><strong>{s['vol_ratio']:.1f}×</strong></td>"
                f"<td align='right'>{s['vol_today']:,.0f}</td>"
                f"<td align='right'>{_dv_str(s['avg_dv'])}</td>"
                f"<td align='right' style='background:#eafaf1'>${s['entry_est']:.2f}</td>"
                f"<td align='right' style='color:#c0392b'>${s['stop']:.2f}</td>"
                f"<td align='right'>${s['risk_ps']:.2f}</td>"
                f"<td align='right'><strong>{s['shares']:,}</strong></td>"
                f"<td align='right'>${s['pos_size']:,.0f}{cap_str}</td>"
                f"<td align='right'>{s['pct_port']:.1%}</td>"
                f"</tr>\n"
            )
        body = f"""
        <table border="0" cellpadding="8" cellspacing="0"
               style="border-collapse:collapse;font-family:'Courier New',monospace;
                      font-size:13px;width:100%;max-width:1100px">
          <thead>
            <tr style="background:#1a2a3a;color:white;text-align:center">
              <th>Rank</th><th>Ticker</th><th>Price</th><th>20d High</th>
              <th>% Above</th><th>Vol Ratio</th><th>Vol Today</th><th>Avg $ Vol</th>
              <th style="background:#1e6b40">Entry~</th>
              <th style="background:#2c3e50">Stop</th>
              <th style="background:#2c3e50">Risk/Sh</th>
              <th style="background:#2c3e50">Shares</th>
              <th style="background:#2c3e50">Pos $</th>
              <th style="background:#2c3e50">% Port</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        <p style="margin:6px 0;color:#555;font-size:12px">
          Entry~ = today's close (proxy) &bull; actual entry = {next_day} open &bull;
          * = capped at ${MAX_POSITION:,} &bull;
          $60K portfolio &bull; 1% risk ($600/trade) &bull; 3% stop
        </p>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family:Arial,sans-serif;color:#222;max-width:1100px;margin:0 auto;padding:20px">
  <h2 style="color:#1a2a3a;border-bottom:2px solid #1a2a3a;padding-bottom:8px">
    [BREAKOUT-SCAN] &mdash; {scan_date.strftime("%B %d, %Y")}
  </h2>
  <p style="margin:4px 0">
    <strong>{n} breakout signal{'s' if n!=1 else ''}</strong> found &nbsp;&bull;&nbsp;
    <strong>Entry:</strong> Open on {next_day.strftime("%A, %B %d")} &nbsp;&bull;&nbsp;
    <strong>Hold:</strong> 5 trading days &nbsp;&bull;&nbsp;
    <strong>Stop:</strong> 3% below entry
  </p>
  <p style="margin:4px 0;color:#555;font-size:13px">
    Signal: close &gt; 20-day high &amp; volume &ge; 1.5&times; 20-day avg &bull;
    Universe: avg $ vol &gt; $10M &bull; not ETF
  </p>
  <p style="margin:4px 0;color:#27ae60;font-size:13px">
    <strong>Backtest edge:</strong>
    Sharpe 3.73 | WR 36.8% | PF 1.32 | Ann. +285%
    &nbsp;(in-sample 2022&ndash;2025, n=75,015 trades, regime-agnostic)
  </p>
  <br>{body}
  <br>
  <p style="color:#888;font-size:11px;border-top:1px solid #ddd;padding-top:8px">
    Generated by breakout_scanner.py &mdash;
    Entry is at next-day open; actual stop should be placed at entry &times; 0.97.
    Positions exceeding $20K are marked * — consider reducing size.
    Past backtest results do not guarantee future performance.
  </p>
</body>
</html>"""


def fmt_text(signals: list[dict], scan_date: date, next_day: date) -> str:
    return (
        f"[BREAKOUT-SCAN] {scan_date}\n"
        f"{len(signals)} breakout signal(s) found\n"
        f"Entry: open on {next_day}  |  Hold: 5 days  |  Stop: 3% below entry\n"
        f"Signal: close > 20d high & volume >= 1.5x 20d avg  |  Universe: DV > $10M\n"
        f"Edge: Sharpe 3.73 | WR 36.8% | PF 1.32 (IS 2022-2025, n=75,015)\n\n"
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
    out = RESULTS_DIR / f"breakout_{scan_date}.csv"
    rows = []
    for rank, s in enumerate(signals, 1):
        rows.append(dict(
            rank            = rank,
            ticker          = s["ticker"],
            scan_date       = str(scan_date),
            price           = round(s["price"], 2),
            high_20d        = round(s["hi_20d"], 2),
            pct_above_high  = round(s["pct_above"] * 100, 2),
            volume_today    = int(s["vol_today"]),
            volume_ratio    = round(s["vol_ratio"], 2),
            avg_volume_20d  = int(s["avg_vol"]),
            avg_dollar_vol  = int(s["avg_dv"]),
            suggested_entry = round(s["entry_est"], 2),
            stop_loss       = s["stop"],
            risk_per_share  = round(s["risk_ps"], 4),
            shares          = s["shares"],
            position_size   = s["pos_size"],
            pct_portfolio   = round(s["pct_port"] * 100, 1),
            size_capped     = s["capped"],
        ))
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Daily volume breakout scanner")
    parser.add_argument("--date", metavar="YYYY-MM-DD", default=None)
    args = parser.parse_args()
    requested = date.fromisoformat(args.date) if args.date else None

    SEP = "=" * 72
    print(SEP, flush=True)
    print("  [BREAKOUT-SCAN] Volume-Confirmed 20-Day High Breakout", flush=True)
    print("  Strategy #3 from autonomous strategy discovery", flush=True)
    print(SEP, flush=True)

    scan_date = find_scan_date(requested)
    next_day  = next_bday(scan_date)
    print(f"\n  Scan date  : {scan_date}", flush=True)
    print(f"  Entry day  : {next_day} (next-day open)", flush=True)
    print(f"  Signal     : close > 20d high & volume >= {VOL_MULT}× 20d avg", flush=True)
    print(f"  Stop       : {STOP_PCT:.0%} below entry", flush=True)
    print(flush=True)

    # ── Today's bars ──────────────────────────────────────────────────────────
    print(f"  Fetching grouped bars for {scan_date} ...", flush=True)
    today_raw = fetch_grouped_day(scan_date)
    if today_raw.empty:
        print("  No data for scan date. Exiting.", flush=True)
        return

    today_df = today_raw[
        (today_raw["close"] > 1.0) &
        (today_raw["volume"] > 10_000)
    ].copy()
    print(f"  {len(today_df):,} tickers after broad pre-filter.", flush=True)

    # ── ETF exclusion ─────────────────────────────────────────────────────────
    etf_set  = get_etf_set()
    today_df = today_df[~today_df["ticker"].apply(lambda t: is_etf(t, etf_set))].copy()
    print(f"  {len(today_df):,} tickers after ETF exclusion.", flush=True)

    active = set(today_df["ticker"])

    # ── Load price history ────────────────────────────────────────────────────
    hist_days = _bdays_list(scan_date, HISTORY_CAL_DAYS)
    cached    = sum(1 for d in hist_days if (CACHE_DIR / f"grouped_{d}.pkl").exists())
    print(f"  Loading {cached} cached history days (need {MIN_BARS} bars) ...", flush=True)
    bars = load_history(active, hist_days)
    print(f"  History loaded for {len(bars):,} tickers.", flush=True)
    print(flush=True)

    # ── Compute breakout signals ───────────────────────────────────────────────
    print("  Scanning for breakout signals ...", flush=True)
    signals = compute_signals(today_df, bars)
    print(f"  {len(signals)} breakout signal(s) found.", flush=True)
    print(flush=True)

    # ── Console ───────────────────────────────────────────────────────────────
    print(SEP, flush=True)
    print(f"  BREAKOUT SIGNALS — {scan_date}  ({len(signals)} found)", flush=True)
    print(f"  Entry: open on {next_day}  |  Hold: 5 days  |  Stop: 3% below entry", flush=True)
    print(SEP, flush=True)
    print(fmt_console(signals), flush=True)
    print(SEP, flush=True)
    print(flush=True)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = save_csv(signals, scan_date)
    print(f"  CSV saved: {csv_path}", flush=True)

    # ── Email ─────────────────────────────────────────────────────────────────
    n    = len(signals)
    subj = (f"[BREAKOUT-SCAN] {scan_date} — "
            f"{n} signal{'s' if n!=1 else ''} (entry {next_day})")
    send_email(subj,
               fmt_html(signals, scan_date, next_day),
               fmt_text(signals, scan_date, next_day))

    print(f"\n  Done. {n} signal(s) on {scan_date}.", flush=True)
    print(SEP, flush=True)


if __name__ == "__main__":
    main()

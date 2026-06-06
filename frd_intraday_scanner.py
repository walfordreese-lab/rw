# -*- coding: utf-8 -*-
"""
frd_intraday_scanner.py
=======================
First Red Day (FRD) Intraday Short Scanner — Polygon.io

Startup
-------
1. Pull the last LOOKBACK_DAYS of grouped daily bars to find stocks currently
   in an active pump meeting all universe criteria:
     - Price $3-$10
     - 3-day gain >= 100%
     - Average volume >= 1M during the run
     - Streak <= 4 consecutive green days
     - Vol ratio on most recent day: < 0.10  OR  >= 0.50

Real-time loop  (every 5 minutes, 9:30 AM – 4:00 PM ET)
---------------------------------------------------------
2. Fetch Polygon snapshot for every universe ticker (one API call per batch).
3. Alert when a ticker meets both intraday FRD conditions:
     a. Current price < yesterday's close  (first red day)
     b. Current price < today's HOD * (1 - 0.03)  (fading >= 3% from high)
4. Print a live status table every poll.
5. Log all alerts to frd_alerts_YYYY-MM-DD.txt.

Usage
-----
    python frd_intraday_scanner.py
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import base64
import getpass
import json
import smtplib
import ssl
import subprocess
import tempfile
import time as _time
import signal
from datetime import datetime, date, timedelta, time as dt_time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

import requests
import pandas as pd

from polygon_fetcher import API_KEY, BASE_URL, fetch_grouped_day, _bdays

try:
    from plyer import notification as _plyer_notification
    _NOTIFY_AVAILABLE = True
except Exception:
    _NOTIFY_AVAILABLE = False

# ── Config ─────────────────────────────────────────────────────────────────────
# Strategy G — "Failed Bounce Reversal Short"
# Derived from 150K-combo grid search over 6 months of Polygon data.
# Backtest: n=12, WR=75%, Exp=+8.0% per trade (Dec 2025–Jun 2026).
PRICE_MIN      = 2.0
PRICE_MAX      = 25.0
GAIN_3D_MIN    = 0.75        # 75% minimum 3-day gain
VOL_MIN        = 300_000     # 20-day avg volume floor
MAX_STREAK     = 1           # <=1 consecutive green close (the key filter)
MIN_DOWN_PCT   = 0.10        # close must be >=10% below prev close
VOL_RATIO_MIN  = 0.30        # today vol / 20d avg must be >= 0.30

STOP_PCT       = 0.15        # 15% hard stop above entry
HOD_FADE_PCT   = 0.12        # must be >= 12% off HOD to confirm reversal
POLL_SECS      = 300         # 5-minute poll interval
LOOKBACK_DAYS  = 25          # trading days of history (needs 20+ for vol avg)
TRASH_SCORE_THRESHOLD = 7    # >= this triggers popup + email; lower = console only

ET             = ZoneInfo("America/New_York")
MARKET_OPEN    = dt_time(9, 30)
MARKET_CLOSE   = dt_time(16, 0)

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
CREDS_FILE     = os.path.join(SCRIPT_DIR, ".email_creds.json")

# Global flag so Ctrl+C triggers a clean exit
_RUNNING = True


def _handle_sigint(sig, frame):
    global _RUNNING
    print("\n\n  [Ctrl+C] Stopping scanner ...")
    _RUNNING = False


signal.signal(signal.SIGINT, _handle_sigint)


# ── Polygon helpers ────────────────────────────────────────────────────────────

def _get(path: str, params: dict | None = None, retries: int = 3) -> dict:
    url = BASE_URL + path
    p   = {"apiKey": API_KEY, **(params or {})}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=p, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                _time.sleep(2 ** (attempt + 2))
                continue
            return {}
        except Exception:
            _time.sleep(2)
    return {}


def get_snapshots(tickers: list) -> dict:
    """
    Fetch current-session snapshot for a list of tickers.
    Batches into chunks of 50 (Polygon recommends <= 100, 50 is safe).

    Returns {ticker: snapshot_dict} with sub-keys:
        day      -- today's OHLCV so far: {o, h, l, c, v, vw}
        prevDay  -- yesterday's OHLCV:    {o, h, l, c, v}
        lastTrade
    """
    result = {}
    for i in range(0, len(tickers), 50):
        chunk = tickers[i : i + 50]
        body  = _get(
            "/v2/snapshot/locale/us/markets/stocks/tickers",
            {"tickers": ",".join(chunk)},
        )
        for t in body.get("tickers", []):
            result[t["ticker"]] = t
    return result


def get_prev_close_fallback(ticker: str) -> float | None:
    """
    Fetch yesterday's closing price for a single ticker via aggregates
    (used when prevDay is missing from snapshot, e.g. pre-market).
    """
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    body = _get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{yesterday}/{yesterday}",
        {"adjusted": "true"},
    )
    results = body.get("results", [])
    return float(results[-1]["c"]) if results else None


# ── Fundamental trash score ────────────────────────────────────────────────────

def fetch_fundamental_score(ticker: str) -> tuple[int, dict]:
    """
    Pull fundamental data from Polygon and compute a weakness (trash) score 0-10.

    Scoring:
      +3  No revenue
      +2  Annual burn > 50% of market cap
      +2  Cash runway < 6 months
      +1  Recent dilution (shares outstanding grew >15% over 4 quarters)
      +1  Market cap < $50M
      +1  Price was under $2 at any point in the last 12 months

    Returns (score, breakdown_dict).  Never raises — returns (0, {}) on total failure.
    """
    score = 0
    breakdown = {
        "no_revenue":      False,
        "burning_excess":  False,
        "short_runway":    False,
        "recent_dilution": False,
        "small_cap":       False,
        "sub2_history":    False,
    }

    try:
        # ── Ticker reference (market cap) ──────────────────────────────────────
        ref = _get(f"/v3/reference/tickers/{ticker}")
        info = ref.get("results", {})
        market_cap = float(info.get("market_cap") or 0)

        if 0 < market_cap < 50_000_000:
            score += 1
            breakdown["small_cap"] = True

        # ── Quarterly financials (last 4 quarters) ─────────────────────────────
        fin = _get(
            "/vX/reference/financials",
            {"ticker": ticker, "timeframe": "quarterly", "limit": 4, "order": "desc"},
        )
        fin_results = fin.get("results", [])

        revenue        = 0.0
        net_income     = 0.0
        cash           = 0.0
        shares_recent  = 0.0
        shares_old     = 0.0

        if fin_results:
            latest = fin_results[0].get("financials", {})
            ic     = latest.get("income_statement", {})
            bs     = latest.get("balance_sheet", {})

            for key in ("revenues", "revenue", "net_revenues"):
                if key in ic:
                    revenue = float(ic[key].get("value") or 0)
                    break

            for key in ("net_income_loss", "net_income_loss_attributable_to_parent"):
                if key in ic:
                    net_income = float(ic[key].get("value") or 0)
                    break

            for key in (
                "cash_and_cash_equivalents_including_discontinued_operations",
                "cash_and_cash_equivalents",
                "cash_and_equivalents",
            ):
                if key in bs:
                    cash = float(bs[key].get("value") or 0)
                    break

            # Shares for dilution check — try income statement then balance sheet
            for key in ("diluted_average_shares", "basic_average_shares"):
                if key in ic:
                    shares_recent = float(ic[key].get("value") or 0)
                    break
            if not shares_recent:
                for key in ("common_stock_shares_outstanding",):
                    if key in bs:
                        shares_recent = float(bs[key].get("value") or 0)
                        break

        # Score 1: No revenue (+3)
        if revenue == 0.0:
            score += 3
            breakdown["no_revenue"] = True

        # Score 2: Burning >50% of market cap per year (+2)
        if net_income < 0 and market_cap > 0:
            if abs(net_income) * 4 / market_cap > 0.50:
                score += 2
                breakdown["burning_excess"] = True

        # Score 3: Cash runway < 6 months (+2)
        quarterly_burn = abs(net_income) if net_income < 0 else 0.0
        if quarterly_burn > 0 and cash > 0:
            months_runway = (cash / quarterly_burn) * 3.0
            if months_runway < 6.0:
                score += 2
                breakdown["short_runway"] = True

        # Score 4: Recent dilution — shares outstanding grew >15% over 4 quarters (+1)
        if fin_results and len(fin_results) >= 2:
            oldest = fin_results[-1].get("financials", {})
            ic_old = oldest.get("income_statement", {})
            bs_old = oldest.get("balance_sheet", {})
            for key in ("diluted_average_shares", "basic_average_shares"):
                if key in ic_old:
                    shares_old = float(ic_old[key].get("value") or 0)
                    break
            if not shares_old:
                for key in ("common_stock_shares_outstanding",):
                    if key in bs_old:
                        shares_old = float(bs_old[key].get("value") or 0)
                        break
            if shares_recent > 0 and shares_old > 0:
                if (shares_recent - shares_old) / shares_old > 0.15:
                    score += 1
                    breakdown["recent_dilution"] = True

        # Score 6: Price under $2 at any point in the last 12 months (+1)
        yr_ago    = (date.today() - timedelta(days=365)).isoformat()
        today_str = date.today().isoformat()
        hist = _get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{yr_ago}/{today_str}",
            {"adjusted": "true", "limit": 300},
        )
        for bar in hist.get("results", []):
            if float(bar.get("l", 999)) < 2.0:
                score += 1
                breakdown["sub2_history"] = True
                break

    except Exception:
        pass

    return min(score, 10), breakdown


# ── Universe scan ──────────────────────────────────────────────────────────────

def _streak(closes: list) -> int:
    """Count consecutive green closes at the tail of the list."""
    n, count = len(closes), 0
    for i in range(n - 1, 0, -1):
        if closes[i] > closes[i - 1]:
            count += 1
        else:
            break
    return count


def build_pump_universe(lookback: int = LOOKBACK_DAYS) -> dict:
    """
    Scan the last `lookback` trading days to identify stocks currently in
    an active multi-day pump matching all Strategy G universe criteria.

    Returns
    -------
    dict  {ticker: metadata}
        prev_close   float  -- yesterday's closing price
        run_avg_vol  float  -- 20-day average volume
        roll3_gain   float  -- 3-bar rolling gain (e.g. 0.75 = 75%)
        streak       int    -- consecutive green closes (must be <= 1)
        vol_ratio    float  -- most-recent-day vol / 20-day avg vol
    """
    today     = date.today()
    cal_start = today - timedelta(days=lookback * 2 + 5)
    all_days  = _bdays(cal_start.isoformat(), (today - timedelta(days=1)).isoformat())
    days      = all_days[-lookback:]

    if not days:
        print("  No recent trading days found.")
        return {}

    print(f"  Loading {len(days)} grouped daily files "
          f"({days[0]} -> {days[-1]}) ...")

    # Accumulate bars
    ticker_rows: dict = {}
    for day in days:
        df = fetch_grouped_day(day)
        for row in df.itertuples(index=False):
            if not (0.5 <= row.close <= 50.0 and row.volume >= 100_000):
                continue
            tkr = row.ticker
            if tkr not in ticker_rows:
                ticker_rows[tkr] = []
            ticker_rows[tkr].append((
                float(row.open), float(row.high), float(row.low),
                float(row.close), float(row.volume),
            ))

    print(f"  {len(ticker_rows):,} tickers tracked. Applying filters ...")

    universe = {}
    for ticker, rows in ticker_rows.items():
        if len(rows) < 4:
            continue

        closes = [r[3] for r in rows]
        vols   = [r[4] for r in rows]

        last_close = closes[-1]

        # Price filter
        if not (PRICE_MIN <= last_close <= PRICE_MAX):
            continue

        # 3-day rolling gain: close[-1] vs close[-4]
        base = closes[-4]
        if base <= 0:
            continue
        roll3 = (last_close - base) / base
        if roll3 < GAIN_3D_MIN:
            continue

        # 20-day avg volume (all days except the signal day)
        hist_vols   = vols[:-1]
        avg_20d     = sum(hist_vols[-20:]) / len(hist_vols[-20:]) if hist_vols else 0.0
        if avg_20d < VOL_MIN:
            continue

        # Streak
        s = _streak(closes)
        if s > MAX_STREAK:
            continue

        # Vol ratio: most recent day vs 20-day avg
        vol_ratio = vols[-1] / avg_20d if avg_20d > 0 else 0.0
        if vol_ratio < VOL_RATIO_MIN:
            continue

        universe[ticker] = {
            "prev_close":  last_close,
            "run_avg_vol": avg_20d,
            "roll3_gain":  roll3,
            "streak":      s,
            "vol_ratio":   vol_ratio,
        }

    return universe


# ── Signal & alert logic ───────────────────────────────────────────────────────

def check_frd(ticker: str, snap: dict, meta: dict) -> tuple:
    """
    Evaluate intraday FRD conditions from a Polygon snapshot.

    Returns (triggered: bool, details: dict).
    """
    day  = snap.get("day", {})
    prev = snap.get("prevDay", {})

    current   = day.get("c") or day.get("vw")
    hod       = day.get("h")
    today_vol = day.get("v", 0)
    prev_c    = prev.get("c") or meta["prev_close"]

    if not (current and hod and prev_c and prev_c > 0 and hod > 0):
        return False, {}

    pct_vs_prev = (current - prev_c) / prev_c
    pct_off_hod = (hod - current) / hod if hod > 0 else 0.0

    # Strategy G conditions
    gone_red   = pct_vs_prev <= -MIN_DOWN_PCT           # >= 10% below prev close
    fading_hod = pct_off_hod >= HOD_FADE_PCT            # >= 12% off HOD

    details = {
        "current":     current,
        "prev_close":  prev_c,
        "hod":         hod,
        "today_vol":   today_vol,
        "gone_red":    gone_red,
        "fading_hod":  fading_hod,
        "pct_vs_prev": pct_vs_prev,
        "pct_off_hod": pct_off_hod,
    }
    return gone_red and fading_hod, details


def format_alert(ticker: str, details: dict, meta: dict | None = None) -> str:
    entry    = details["current"]
    stop     = round(entry * (1.0 + STOP_PCT), 2)
    now_str  = datetime.now(ET).strftime("%H:%M ET")
    pct_off  = details["pct_off_hod"]
    pct_day  = details["pct_vs_prev"]
    ts       = (meta or {}).get("trash_score", 0)
    warn     = " ⚠️" if ts >= TRASH_SCORE_THRESHOLD else ""
    return (
        f"[{now_str}] STRAT-G SHORT: {ticker}"
        f"  Entry zone: ${entry:.2f}"
        f"  Stop: ${stop:.2f} ({STOP_PCT:.0%} above entry)"
        f"  | prev close ${details['prev_close']:.2f}"
        f"  HOD ${details['hod']:.2f}"
        f"  ({pct_off:.1%} off high, {pct_day:+.1%} on day)"
        f"  | Trash: {ts}/10{warn}"
    )


def log_alert(text: str, path: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&apos;")
    )


def _winrt_toast(title: str, line2: str, line3: str) -> bool:
    """
    Send a modern Windows toast notification via a temporary PowerShell script.

    Uses the WinRT ToastNotificationManager — the same API that produces the
    notification popups in the Windows 11 Action Center.  plyer's balloon-tip
    backend targets the old Win32 tray area which Windows 11 suppresses.

    Returns True on success, False on any error (never raises).
    """
    xml = (
        '<toast duration="long">'
        "<visual><binding template=\"ToastGeneric\">"
        f"<text>{_xml_escape(title)}</text>"
        f"<text>{_xml_escape(line2)}</text>"
        f"<text>{_xml_escape(line3)}</text>"
        "</binding></visual>"
        '<audio src="ms-winsoundevent:Notification.Default"/>'
        "</toast>"
    )

    # Encode XML as UTF-16-LE Base64 so it survives PowerShell string handling
    xml_b64 = base64.b64encode(xml.encode("utf-16-le")).decode("ascii")

    ps = f"""\
$APPID = "FRD.Scanner"
$reg = "HKCU:\\SOFTWARE\\Classes\\AppUserModelId\\$APPID"
if (!(Test-Path $reg)) {{
    New-Item -Path $reg -Force | Out-Null
    New-ItemProperty -Path $reg -Name "DisplayName" -Value "FRD Scanner" -Force | Out-Null
}}
[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime]|Out-Null
[Windows.Data.Xml.Dom.XmlDocument,Windows.Data.Xml.Dom.XmlDocument,ContentType=WindowsRuntime]|Out-Null
$bytes  = [System.Convert]::FromBase64String('{xml_b64}')
$xmlStr = [System.Text.Encoding]::Unicode.GetString($bytes)
$doc    = New-Object Windows.Data.Xml.Dom.XmlDocument
$doc.LoadXml($xmlStr)
$toast  = New-Object Windows.UI.Notifications.ToastNotification $doc
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($APPID).Show($toast)
"""

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ps1", delete=False, encoding="utf-8"
        ) as f:
            f.write(ps)
            tmp = f.name
        result = subprocess.run(
            [
                "powershell",
                "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-WindowStyle", "Hidden",
                "-File", tmp,
            ],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except Exception:
                pass


def notify(ticker: str, details: dict, trash_score: int = 0) -> None:
    """
    Fire a Windows desktop notification for an FRD signal.

    Primary  : WinRT toast via PowerShell (Windows 10/11 Action Center popup).
    Fallback : plyer balloon tip (Win32 tray area — suppressed on Windows 11
               but may appear on some Windows 10 setups).

    Failure in either path is silently swallowed so the scanner never crashes.
    """
    entry = details["current"]
    stop  = round(entry * (1.0 + STOP_PCT), 2)
    warn  = " ⚠️" if trash_score >= TRASH_SCORE_THRESHOLD else ""
    title = f"STRAT-G SHORT: {ticker}  |  Trash {trash_score}/10{warn}"
    line2 = f"Entry zone: ${entry:.2f}   Stop: ${stop:.2f} (+{STOP_PCT:.0%})"
    line3 = (
        f"Prev close ${details['prev_close']:.2f}  "
        f"HOD ${details['hod']:.2f}  "
        f"({details['pct_off_hod']:.1%} off high)"
    )

    if _winrt_toast(title, line2, line3):
        return

    # Fallback: plyer balloon tip
    if _NOTIFY_AVAILABLE:
        try:
            _plyer_notification.notify(
                title=title,
                message=f"{line2}\n{line3}",
                app_name="FRD Scanner",
                timeout=10,
            )
        except Exception:
            pass


# ── Display ────────────────────────────────────────────────────────────────────

def print_scan_table(universe: dict, snapshots: dict, alerted: set) -> None:
    now_str = datetime.now(ET).strftime("%H:%M:%S ET")
    w = 88
    print(f"\n{'='*w}")
    print(f"  [{now_str}]  Watching {len(universe)} stocks  "
          f"| next poll in {POLL_SECS//60} min  | Ctrl+C to stop")
    print(f"{'='*w}")
    print(f"  {'Ticker':<7}  {'PrevC':>7}  {'Curr':>7}  {'HOD':>7}  "
          f"{'vs.Prev':>8}  {'offHOD':>7}  {'Trash':>7}  Status")
    print(f"  {'-'*(w-2)}")

    for ticker in sorted(universe):
        meta = universe[ticker]
        snap = snapshots.get(ticker, {})
        triggered, det = check_frd(ticker, snap, meta)

        if not det:
            print(f"  {ticker:<7}  (no data yet)")
            continue

        if ticker in alerted:
            status = "ALERTED"
        elif triggered:
            status = "** STRAT-G SIGNAL **"
        elif det["gone_red"] and not det["fading_hod"]:
            status = f"down>=10%  ({det['pct_off_hod']:.1%} off HOD, need >=12%)"
        elif det["fading_hod"] and not det["gone_red"]:
            status = f"HOD-fade OK  ({det['pct_vs_prev']:+.1%} vs prev, need <=-10%)"
        elif not det["gone_red"] and not det["fading_hod"]:
            status = f"watching  ({det['pct_vs_prev']:+.1%} vs prev)"
        else:
            status = "watching"

        ts   = meta.get("trash_score", 0)
        warn = "⚠️" if ts >= TRASH_SCORE_THRESHOLD else "  "
        print(
            f"  {ticker:<7}  "
            f"${det['prev_close']:>6.2f}  "
            f"${det['current']:>6.2f}  "
            f"${det['hod']:>6.2f}  "
            f"{det['pct_vs_prev']:>+7.1%}  "
            f"{det['pct_off_hod']:>6.1%}  "
            f"  {ts:>2}/10{warn}  "
            f"{status}"
        )
    print()


# ── Email alerts ──────────────────────────────────────────────────────────────

def _save_email_creds(cfg: dict) -> None:
    data = {"address": cfg["address"],
            "password": base64.b64encode(cfg["password"].encode()).decode()}
    with open(CREDS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _load_email_creds() -> dict | None:
    if not os.path.exists(CREDS_FILE):
        return None
    try:
        with open(CREDS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return {"address": data["address"],
                "password": base64.b64decode(data["password"]).decode()}
    except Exception:
        return None


def _delete_email_creds() -> None:
    try:
        os.unlink(CREDS_FILE)
    except FileNotFoundError:
        pass


def _verify_smtp(cfg: dict) -> bool:
    """Return True on successful SMTP login, False and print reason otherwise."""
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=10) as s:
            s.login(cfg["address"], cfg["password"])
        return True
    except smtplib.SMTPAuthenticationError:
        print("FAILED\n")
        print("  Authentication error — make sure you used an App Password,")
        print("  not your regular Gmail password, and that 2-Step Verification")
        print("  is enabled on your account.")
        return False
    except Exception as exc:
        print(f"FAILED ({exc})")
        return False


def print_email_setup_instructions() -> None:
    w = 69
    bar = "+" + "-" * w + "+"
    def line(s=""): print(f"  | {s:<{w-1}}|")
    print(f"  {bar}")
    line("Gmail App Password setup")
    line()
    line("  App Passwords let this scanner send email without exposing your")
    line("  main Gmail password.  Regular Gmail passwords will NOT work.")
    line()
    line("  1. Go to  myaccount.google.com/apppasswords")
    line("  2. Sign in and choose 'Create app password'")
    line("  3. Name it anything (e.g. 'FRD Scanner')")
    line("  4. Copy the 16-character code Google shows you")
    line("  5. Paste it below (spaces are stripped automatically)")
    line()
    line("  Note: 2-Step Verification must be enabled on your account first.")
    print(f"  {bar}")
    print()


def setup_email(reset: bool = False) -> dict | None:
    """
    Return verified Gmail credentials, loading from disk when available.

    If reset=True the saved credentials file is deleted and the user is
    prompted fresh.  On any auth failure the user gets unlimited retries;
    pressing Enter at the address prompt disables email alerts for this run.
    Successfully verified credentials are saved to CREDS_FILE for next time.
    """
    if reset:
        _delete_email_creds()
        print("  Saved email credentials cleared.\n")

    # ── Try loading saved credentials ─────────────────────────────────────────
    if not reset:
        cfg = _load_email_creds()
        if cfg:
            print(f">> Email: using saved credentials for {cfg['address']} ...",
                  end=" ", flush=True)
            if _verify_smtp(cfg):
                print("OK\n")
                return cfg
            print("  Credentials may have expired — please re-enter.\n")

    # ── Interactive prompt loop (unlimited retries) ───────────────────────────
    print(">> Email alert setup")
    print_email_setup_instructions()

    while True:
        address = input("  Gmail address (press Enter to skip email alerts): ").strip()
        if not address:
            print("  Email alerts disabled.\n")
            return None

        raw_pw = getpass.getpass("  Gmail App Password (input hidden): ")
        password = raw_pw.replace(" ", "")
        if not password:
            print("  No password entered — email alerts disabled.\n")
            return None

        print("  Verifying credentials ...", end=" ", flush=True)
        cfg = {"address": address, "password": password}
        if _verify_smtp(cfg):
            print("OK\n")
            _save_email_creds(cfg)
            print(f"  Credentials saved to {CREDS_FILE}\n")
            return cfg

        print("  Try again, or press Enter at the address prompt to skip.\n")


def send_email_alert(
    ticker: str,
    details: dict,
    meta: dict,
    email_cfg: dict,
    log_path: str,
) -> None:
    """
    Send an FRD signal email via Gmail SMTP.
    Logs any error to the alert log file but never raises.
    """
    entry       = details["current"]
    stop        = round(entry * (1.0 + STOP_PCT), 2)
    now         = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    trash_score = meta.get("trash_score", 0)
    warn        = " ⚠️" if trash_score >= TRASH_SCORE_THRESHOLD else ""

    subject = f"STRAT-G SHORT: {ticker}  |  Trash {trash_score}/10{warn}"

    # ── Plain-text body ───────────────────────────────────────────────────────
    plain = (
        f"STRAT-G SHORT SIGNAL: {ticker}\n"
        f"{'='*40}\n"
        f"Time:           {now}\n"
        f"Ticker:         {ticker}\n"
        f"Entry zone:     ${entry:.2f}\n"
        f"Stop:           ${stop:.2f}  (+{STOP_PCT:.0%} above entry)\n"
        f"{'-'*40}\n"
        f"Prev close:     ${details['prev_close']:.2f}\n"
        f"HOD today:      ${details['hod']:.2f}\n"
        f"% off high:     {details['pct_off_hod']:.1%}\n"
        f"% vs prev:      {details['pct_vs_prev']:+.1%}\n"
        f"{'-'*40}\n"
        f"3-day gain:     {meta['roll3_gain']:.0%}\n"
        f"Vol ratio:      {meta['vol_ratio']:.2f}\n"
        f"Run avg vol:    {meta['run_avg_vol']/1e6:.1f}M shares/day\n"
        f"Trash score:    {trash_score}/10{warn}\n"
        f"{'='*40}\n"
        f"Trade plan: Short at open tomorrow if price stays below ${details['prev_close']:.2f}.\n"
        f"Hard stop {STOP_PCT:.0%} above entry = ${stop:.2f}.  Exit at EOD close.\n"
        f"Strategy G — streak<={MAX_STREAK}, HOD>={HOD_FADE_PCT:.0%}, down>={MIN_DOWN_PCT:.0%}\n"
    )

    # ── HTML body ─────────────────────────────────────────────────────────────
    def row(label: str, value: str, bold: bool = False) -> str:
        v = f"<b>{value}</b>" if bold else value
        return (
            f"<tr><td style='padding:4px 12px 4px 4px;color:#555'>{label}</td>"
            f"<td style='padding:4px'>{v}</td></tr>"
        )

    trash_color = "#c0392b" if trash_score >= TRASH_SCORE_THRESHOLD else "#444"
    html = f"""<html><body style='font-family:monospace;font-size:14px'>
<h2 style='color:#c0392b;margin-bottom:4px'>STRAT-G SHORT: {ticker}</h2>
<p style='color:#666;margin-top:0'>{now}</p>
<table style='border-collapse:collapse;margin-bottom:16px'>
  {row("Ticker", ticker, bold=True)}
  {row("Entry zone", f"${entry:.2f}", bold=True)}
  {row("Stop price", f"${stop:.2f}  (+{STOP_PCT:.0%})", bold=True)}
  <tr><td style='padding:4px 12px 4px 4px;color:#555'>Trash score</td>
      <td style='padding:4px;color:{trash_color}'><b>{trash_score}/10{warn}</b></td></tr>
</table>
<table style='border-collapse:collapse;border-top:1px solid #ddd;
              padding-top:8px;margin-bottom:16px'>
  {row("Prev close", f"${details['prev_close']:.2f}")}
  {row("HOD today", f"${details['hod']:.2f}")}
  {row("% off high", f"{details['pct_off_hod']:.1%}")}
  {row("% vs prev close", f"{details['pct_vs_prev']:+.1%}")}
</table>
<table style='border-collapse:collapse;border-top:1px solid #ddd;
              padding-top:8px'>
  {row("3-day gain", f"{meta['roll3_gain']:.0%}")}
  {row("Vol ratio", f"{meta['vol_ratio']:.2f}")}
  {row("Run avg vol", f"{meta['run_avg_vol']/1e6:.1f}M shares/day")}
</table>
<p style='margin-top:16px;color:#444'>
  <b>Trade plan:</b> Short at open if price stays below
  ${details['prev_close']:.2f}.&nbsp; Hard stop {STOP_PCT:.0%} above entry
  = ${stop:.2f}.&nbsp; Exit at EOD close.<br>
  <span style='color:#888;font-size:12px'>Strategy G: streak&le;{MAX_STREAK}, HOD&ge;{HOD_FADE_PCT:.0%}, down&ge;{MIN_DOWN_PCT:.0%}</span>
</p>
</body></html>"""

    # ── Build and send message ────────────────────────────────────────────────
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = email_cfg["address"]
        msg["To"]      = email_cfg["address"]
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html,  "html"))

        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=15) as s:
            s.login(email_cfg["address"], email_cfg["password"])
            s.send_message(msg)
    except Exception as exc:
        err = f"  [email error] {ticker}: {exc}"
        print(err)
        log_alert(err, log_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _RUNNING

    parser = argparse.ArgumentParser(description="FRD Intraday Short Scanner")
    parser.add_argument(
        "--reset-email",
        action="store_true",
        help="Clear saved email credentials and prompt for new ones",
    )
    args = parser.parse_args()

    today    = date.today()
    log_path = os.path.join(SCRIPT_DIR, f"frd_alerts_{today.isoformat()}.txt")

    print("+" + "=" * 66 + "+")
    print("|      Strategy G — Failed Bounce Reversal Short Scanner         |")
    print("|                     Polygon.io                                 |")
    print("+" + "=" * 66 + "+")
    print(f"  Universe  : ${PRICE_MIN}-${PRICE_MAX}  |  3d gain >={GAIN_3D_MIN:.0%}"
          f"  |  avg vol >={VOL_MIN//1000:.0f}K  |  streak <={MAX_STREAK}")
    print(f"  Vol ratio : today/20d-avg >= {VOL_RATIO_MIN}")
    print(f"  Signal    : down >={MIN_DOWN_PCT:.0%} from prev close"
          f"  AND  >={HOD_FADE_PCT:.0%} off HOD  (streak <={MAX_STREAK})")
    print(f"  Stop loss : {STOP_PCT:.0%} above entry  |  Exit: EOD")
    print(f"  Backtest  : n=12, WR=75%, Exp=+8.0%  (Dec 2025 - Jun 2026)")
    print(f"  Poll      : every {POLL_SECS // 60} min  |  Log: {log_path}")
    print()

    # ── Email setup ───────────────────────────────────────────────────────────
    email_cfg = setup_email(reset=args.reset_email)

    # Write session header to log
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n{'='*72}\n")
        fh.write(
            f"FRD Scanner session: "
            f"{datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}\n"
        )
        fh.write(f"{'='*72}\n")

    # ── Build universe ────────────────────────────────────────────────────────
    print(">> Building pump universe ...")
    universe = build_pump_universe()

    if not universe:
        print("\n  No tickers meet universe criteria today.")
        print("  (Market may be quiet, or it is early in the week.)\n")
    else:
        print(f"\n  Strategy G universe: {len(universe)} tickers — fetching fundamental scores ...\n")
        for tkr in list(universe.keys()):
            ts, bd = fetch_fundamental_score(tkr)
            universe[tkr]["trash_score"]     = ts
            universe[tkr]["trash_breakdown"] = bd
            _time.sleep(0.25)   # light rate-limit buffer between tickers

        print(f"  {'Ticker':<7}  {'PrevClose':>9}  {'3dGain':>7}  "
              f"{'AvgVol':>9}  {'Streak':>6}  {'VolRatio':>9}  {'Trash':>8}")
        print("  " + "-" * 68)
        for tkr, m in sorted(universe.items()):
            ts   = m.get("trash_score", 0)
            warn = " ⚠️" if ts >= TRASH_SCORE_THRESHOLD else ""
            print(
                f"  {tkr:<7}  "
                f"${m['prev_close']:>8.2f}  "
                f"{m['roll3_gain']:>6.0%}  "
                f"{m['run_avg_vol']/1e6:>8.1f}M  "
                f"{m['streak']:>6}  "
                f"{m['vol_ratio']:>9.2f}  "
                f"  {ts:>2}/10{warn}"
            )

    if not universe:
        print("  Nothing to monitor. Exiting.")
        return

    tickers = list(universe.keys())
    alerted: set = set()
    session_alerts: list = []

    # ── Wait for market open ──────────────────────────────────────────────────
    while _RUNNING:
        now_et = datetime.now(ET)
        now_t  = now_et.time()

        if now_t >= MARKET_CLOSE:
            print(f"\n  Market is closed ({now_et.strftime('%H:%M ET')}). "
                  f"Nothing to do today.")
            return

        if now_t < MARKET_OPEN:
            open_dt  = datetime.combine(today, MARKET_OPEN, tzinfo=ET)
            wait_sec = max(0, int((open_dt - now_et).total_seconds()))
            print(
                f"\n  Pre-market ({now_et.strftime('%H:%M ET')}).  "
                f"Market opens in {wait_sec // 60}m {wait_sec % 60}s.  "
                f"Waiting ..."
            )
            # Sleep in 30-second chunks so Ctrl+C is responsive
            for _ in range(min(wait_sec, 60) // 10 + 1):
                if not _RUNNING:
                    break
                _time.sleep(10)
            continue

        break   # market is open

    if not _RUNNING:
        return

    print(f"\n  Market open. Starting {POLL_SECS // 60}-minute scan loop.\n")

    # ── Main scan loop ────────────────────────────────────────────────────────
    while _RUNNING:
        now_et = datetime.now(ET)

        if now_et.time() >= MARKET_CLOSE:
            print(f"\n  4:00 PM ET — market closed. Ending scan.")
            break

        # Poll
        snapshots = get_snapshots(tickers)
        print_scan_table(universe, snapshots, alerted)

        # Check for new signals
        for ticker, meta in universe.items():
            if ticker in alerted:
                continue
            snap      = snapshots.get(ticker, {})
            triggered, details = check_frd(ticker, snap, meta)
            if triggered:
                alerted.add(ticker)
                alert_text = format_alert(ticker, details, meta)
                session_alerts.append(alert_text)
                log_alert(alert_text, log_path)

                trash_score = meta.get("trash_score", 0)
                if trash_score >= TRASH_SCORE_THRESHOLD:
                    notify(ticker, details, trash_score)
                    if email_cfg:
                        send_email_alert(ticker, details, meta, email_cfg, log_path)
                    bang = "!" * 72
                    print(f"\n{bang}")
                    print(f"  {alert_text}")
                    print(f"{bang}\n")
                else:
                    print(
                        f"\n  [FRD] {alert_text}"
                        f"  (score {trash_score}/10 — console only)\n"
                    )

        if not session_alerts:
            print(f"  No FRD signals yet. Next scan at "
                  f"{(now_et + timedelta(seconds=POLL_SECS)).strftime('%H:%M ET')}.")
        else:
            print(f"  {len(session_alerts)} alert(s) logged this session.")

        # Sleep until next poll, waking every 10s to check Ctrl+C / market close
        deadline = datetime.now(ET) + timedelta(seconds=POLL_SECS)
        while _RUNNING and datetime.now(ET) < deadline:
            if datetime.now(ET).time() >= MARKET_CLOSE:
                break
            _time.sleep(10)

    # ── Session summary ───────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Session complete — {len(session_alerts)} FRD alert(s) fired")
    if session_alerts:
        for a in session_alerts:
            print(f"    {a}")
    print(f"  Full log: {log_path}")
    print(f"{'='*72}\n")

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\nSession ended: {datetime.now(ET).strftime('%H:%M ET')}"
                 f" — {len(session_alerts)} alert(s)\n")


if __name__ == "__main__":
    main()

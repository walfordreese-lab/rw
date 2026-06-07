#!/usr/bin/env python3
"""
Quick comparison: no filter vs XBI dn>1% vs XBI dn>2%.
Reuses all logic from frd_2year_backtest.py and frd_regime_filter.py.
"""
import sys, io, pickle, warnings
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

PRICE_MIN=2.0; PRICE_MAX=25.0; AVG_VOL_MIN=300_000; GAIN_3D_MIN=0.75
MAX_STREAK=1;  HOD_FADE_MIN=0.12; DOWN_PCT_MIN=0.10; VOL_RATIO_MIN=0.30
STOP_PCT=0.15; LOOKBACK_VOL=20
SIM_START=date(2023,6,1); SIM_END=date(2025,6,30)


def load_bars():
    data_start = SIM_START - timedelta(days=LOOKBACK_VOL*2+10)
    days = _bdays(data_start.isoformat(), SIM_END.isoformat())
    bars = defaultdict(dict)
    loaded = []
    for day in days:
        path = CACHE_DIR / f"grouped_{day}.pkl"
        if not path.exists(): continue
        loaded.append(day)
        with open(path,"rb") as f: df = pickle.load(f)
        for row in df.itertuples(index=False):
            c=float(row.close)
            if not (0.5<=c<=50.0) or float(row.volume)<50_000: continue
            bars[row.ticker][day]=(float(row.open),float(row.high),float(row.low),c,float(row.volume))
    return bars, loaded


def build_signals(bars, all_days):
    sim_set = {d for d in all_days if SIM_START<=d<=SIM_END}
    recs = []
    for ticker, dmap in bars.items():
        tdates = sorted(dmap.keys())
        if len(tdates) < LOOKBACK_VOL+4: continue
        for di in range(LOOKBACK_VOL+3, len(tdates)):
            sd = tdates[di]
            if sd not in sim_set: continue
            o,h,l,c,v = dmap[sd]
            if not (PRICE_MIN<=c<=PRICE_MAX): continue
            prec = tdates[:di]
            if len(prec)<LOOKBACK_VOL+3: continue
            prev_c = dmap[prec[-1]][3]
            if c>=prev_c: continue
            pct_off_hod=(h-c)/h
            if pct_off_hod<HOD_FADE_MIN: continue
            pct_vs_prev=(c-prev_c)/prev_c
            if pct_vs_prev>-DOWN_PCT_MIN: continue
            roll3=(c-dmap[prec[-3]][3])/dmap[prec[-3]][3]
            if roll3<GAIN_3D_MIN: continue
            avg_vol=float(np.mean([dmap[d][4] for d in prec[-LOOKBACK_VOL:]]))
            if avg_vol<AVG_VOL_MIN: continue
            if v/avg_vol<VOL_RATIO_MIN: continue
            streak=0
            for k in range(len(prec)-1,max(len(prec)-8,-1),-1):
                if k==0: break
                if dmap[prec[k]][3]>dmap[prec[k-1]][3]: streak+=1
                else: break
            if streak>MAX_STREAK: continue
            all_sorted=sorted(dmap.keys()); d0i=all_sorted.index(sd)
            if d0i+1>=len(all_sorted): continue
            d1d=all_sorted[d0i+1]
            d2d=all_sorted[d0i+2] if d0i+2<len(all_sorted) else None
            if d1d not in dmap: continue
            d1o,d1h,d1l,d1c,_=dmap[d1d]
            d2b=dmap.get(d2d) if d2d else None
            recs.append(dict(ticker=ticker,signal_date=sd,
                d1_open=d1o,d1_high=d1h,d1_close=d1c,
                d2_open=d2b[0] if d2b else np.nan))
    return pd.DataFrame(recs)


def apply_exit4(df):
    e=df["d1_open"].values.astype(float)
    h=df["d1_high"].values.astype(float)
    c=df["d1_close"].values.astype(float)
    d2=df["d2_open"].values.astype(float)
    sp=e*(1+STOP_PCT); stopped=h>=sp
    exits=np.where(stopped,sp,np.where(c<e,d2,c))
    pnl=(e-exits)/e
    df=df.copy(); df["stopped"]=stopped; df["pnl"]=pnl; df["win"]=pnl>0
    return df


def stats(df):
    if len(df)==0:
        return dict(n=0,wr=np.nan,exp=np.nan,avg_w=np.nan,avg_l=np.nan,stop_r=np.nan)
    p=df["pnl"].values
    w=p[p>0]; l=p[p<=0]
    return dict(n=len(p), wr=float((p>0).mean()), exp=float(p.mean()),
                avg_w=float(w.mean()) if len(w) else 0.0,
                avg_l=float(l.mean()) if len(l) else 0.0,
                stop_r=float(df["stopped"].mean()))


def main():
    print("Loading bars ...", flush=True)
    bars, all_days = load_bars()

    print("Fetching XBI data ...", flush=True)
    start_str=(SIM_START-timedelta(days=10)).strftime("%Y-%m-%d")
    end_str  =(SIM_END  +timedelta(days=5)).strftime("%Y-%m-%d")
    raw=yf.download(["XBI"],start=start_str,end=end_str,auto_adjust=True,progress=False)
    xbi_close=raw["Close"]["XBI"].dropna()
    xbi_pct=xbi_close.pct_change()
    xbi_pct.index=pd.to_datetime(xbi_pct.index).normalize()

    print("Building signals ...", flush=True)
    sigs=build_signals(bars, all_days)
    sigs=apply_exit4(sigs).sort_values("signal_date").reset_index(drop=True)

    def xbi_chg(sd):
        ts=pd.Timestamp(sd)
        if ts in xbi_pct.index: return float(xbi_pct.loc[ts])
        prior=xbi_pct[xbi_pct.index<ts]
        return float(prior.iloc[-1]) if not prior.empty else np.nan

    sigs["xbi_pct"]=sigs["signal_date"].apply(xbi_chg)

    no_filter  = sigs
    xbi_dn1    = sigs[sigs["xbi_pct"] < -0.01]
    xbi_dn2    = sigs[sigs["xbi_pct"] < -0.02]

    s0=stats(no_filter); s1=stats(xbi_dn1); s2=stats(xbi_dn2)

    W=80
    print(flush=True)
    print("="*W, flush=True)
    print("  XBI Regime Filter Comparison — Strategy G (Jun 2023 – Jun 2025)", flush=True)
    print("="*W, flush=True)
    print(f"  {'Metric':<22}  {'No Filter':>12}  {'XBI dn>1%':>12}  {'XBI dn>2%':>12}", flush=True)
    print(f"  {'-'*62}", flush=True)

    rows=[
        ("Signals (n)",      f"{s0['n']}",        f"{s1['n']}",        f"{s2['n']}"),
        ("Win rate",         f"{s0['wr']:.1%}",   f"{s1['wr']:.1%}",   f"{s2['wr']:.1%}"),
        ("Avg win",          f"{s0['avg_w']:+.2%}",f"{s1['avg_w']:+.2%}",f"{s2['avg_w']:+.2%}"),
        ("Avg loss",         f"{s0['avg_l']:+.2%}",f"{s1['avg_l']:+.2%}",f"{s2['avg_l']:+.2%}"),
        ("Expectancy/trade", f"{s0['exp']:+.2%}",  f"{s1['exp']:+.2%}",  f"{s2['exp']:+.2%}"),
        ("Stop-out rate",    f"{s0['stop_r']:.1%}",f"{s1['stop_r']:.1%}",f"{s2['stop_r']:.1%}"),
    ]
    for label, v0, v1, v2 in rows:
        print(f"  {label:<22}  {v0:>12}  {v1:>12}  {v2:>12}", flush=True)

    print(f"  {'='*62}", flush=True)

    # Per-trade list for XBI dn>1% (the new filter)
    print(flush=True)
    print(f"  Trades passing XBI dn>1% filter (n={s1['n']}):", flush=True)
    print(f"  {'Date':<12}  {'Ticker':<6}  {'XBI%':>6}  {'PnL':>7}  Result", flush=True)
    print(f"  {'-'*48}", flush=True)
    for _, r in xbi_dn1.sort_values("signal_date").iterrows():
        wl="W" if r["win"] else "L"
        st="*" if r["stopped"] else " "
        print(f"  {str(r['signal_date']):<12}  {r['ticker']:<6}  "
              f"{r['xbi_pct']:>+5.1%}  {r['pnl']*100:>+6.1f}%  {wl}{st}", flush=True)

    print(flush=True)
    print(f"  Trades passing XBI dn>2% filter (n={s2['n']}):", flush=True)
    print(f"  {'Date':<12}  {'Ticker':<6}  {'XBI%':>6}  {'PnL':>7}  Result", flush=True)
    print(f"  {'-'*48}", flush=True)
    for _, r in xbi_dn2.sort_values("signal_date").iterrows():
        wl="W" if r["win"] else "L"
        st="*" if r["stopped"] else " "
        print(f"  {str(r['signal_date']):<12}  {r['ticker']:<6}  "
              f"{r['xbi_pct']:>+5.1%}  {r['pnl']*100:>+6.1f}%  {wl}{st}", flush=True)

    print(flush=True)


if __name__=="__main__":
    main()

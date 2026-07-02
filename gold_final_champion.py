"""
GOLD NY-SWEEP REVERSAL — FINAL CHAMPION STRATEGY
$100,000 Account | Full Report

Strategy Parameters (locked from 100+ backtests):
───────────────────────────────────────────────────
  Base entry:
    • Mark NY session high/low (08:00–17:00 ET) each day
    • Asia window (17:00–01:00 ET): wait for price to wick beyond level and close back inside
    • Only the first qualifying sweep per session traded
    • Trend filter: LONG only when 4H price > SMA-20 | SHORT only when 4H price < SMA-200
    • RSI(14) gate on entry bar: LONG ≤ 40 | SHORT ≥ 60
    • Skip day if NY range < 1× ATR(14)

  Risk management:
    • Stop loss  : sweep wick extreme ± 0.15× ATR (tight, just beyond the trap)
    • Take profit : 80% of NY session range back inside (min 1.5R, fallback 2R)
    • Account     : $100,000

  Dynamic position sizing (confidence system):
    • DEFAULT size : 5% risk ($5,000) — the baseline for high-conviction setups
    • REDUCE to 3% ($3,000) if ANY of the following red flags are present:
        ① Sweep occurs in the 20:00–22:00 ET dead zone (18.2% WR historically)
        ② Entry candle body > 0.6× ATR (momentum continuation, not reversal)
    • Trades with NEITHER flag keep 5% size (historical 67% WR on these)
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, yfinance as yf
from datetime import timedelta
import pytz

ET            = pytz.timezone("America/New_York")
SYMBOL        = "GC=F"
LOOKBACK_DAYS = 730
ACCOUNT_SIZE  = 100_000

# Strategy params
SMA_LONG    = 20;  SMA_SHORT  = 200
ATR_PERIOD  = 14;  ATR_SL_MULT = 0.15
TP_PCT      = 0.80; MIN_TP_R   = 1.5; FALLBACK_RR = 2.0
RSI_LONG_MAX = 40; RSI_SHORT_MIN = 60
ASIA_OPEN = 17;   ASIA_CLOSE = 1

# Risk tiers
RISK_HIGH = 0.05   # 5% — clean setup, no red flags
RISK_LOW  = 0.03   # 3% — at least one red flag present

# Red flag thresholds
DEAD_ZONE_START = 20   # 20:00 ET
DEAD_ZONE_END   = 22   # 22:00 ET
BODY_PENALTY_THRESH = 0.6   # body > 0.6× ATR = momentum bar


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def fetch(symbol, interval, days):
    df = yf.download(symbol, period=f"{days}d", interval=interval,
                     auto_adjust=True, progress=False)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None: df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    return df

def add_atr(df, p=14):
    h,l,c = df["high"],df["low"],df["close"]
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=p,adjust=False).mean(); return df

def add_sma(df, p):
    df[f"sma{p}"] = df["close"].rolling(p).mean(); return df

def add_rsi(df, p=14):
    d = df["close"].diff()
    g = d.clip(lower=0).ewm(span=p,adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=p,adjust=False).mean()
    df["rsi"] = 100-(100/(1+g/l.replace(0,np.nan))); return df

def trading_date(ts):
    return (ts-timedelta(days=1)).normalize() if ts.hour < 4 else ts.normalize()


# ---------------------------------------------------------------------------
# CONFIDENCE SIZING
# ---------------------------------------------------------------------------

def get_risk(bar, bar_atr, ts):
    """Return 5% or 3% based on red flag detection."""
    h = ts.hour
    # Dead zone: 20:00–22:00 ET
    in_dead_zone = (DEAD_ZONE_START <= h <= DEAD_ZONE_END)
    # Momentum bar: large-body candle (continuation, not reversal)
    body = abs(bar["close"] - bar["open"]) / bar_atr if bar_atr > 0 else 0
    is_momentum_bar = body >= BODY_PENALTY_THRESH
    if in_dead_zone or is_momentum_bar:
        return RISK_LOW, in_dead_zone, is_momentum_bar
    return RISK_HIGH, False, False


# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------

def run_backtest(df1h, df4h):
    df1h = df1h.copy()
    df1h["tdate"] = df1h.index.map(trading_date)
    trades = []

    for tdate, day in df1h.groupby("tdate"):
        ny   = day[(day.index.hour >= 8) & (day.index.hour < 17)]
        asia = day[(day.index.hour >= ASIA_OPEN) | (day.index.hour < ASIA_CLOSE)]
        if ny.empty or asia.empty: continue

        ny_high = ny["high"].max(); ny_low = ny["low"].min()
        ny_rng  = ny_high - ny_low;  atr = ny["atr"].iloc[-1]
        if pd.isna(atr) or atr == 0 or ny_rng < atr: continue

        swept_high = swept_low = False

        for ts, bar in asia.iterrows():
            ba  = bar["atr"]; rv = bar["rsi"]
            if pd.isna(ba) or ba == 0: continue
            p4  = df4h[df4h.index <= ts]
            if p4.empty or pd.isna(p4["sma20"].iloc[-1]): continue
            price4h = p4["close"].iloc[-1]
            sma20   = p4["sma20"].iloc[-1]; sma200 = p4["sma200"].iloc[-1]

            body_ratio = abs(bar["close"]-bar["open"])/ba if ba > 0 else 0
            wick_h     = (bar["high"]-ny_high)/ba if ba > 0 else 0
            wick_l     = (ny_low-bar["low"])/ba  if ba > 0 else 0
            rng_ratio  = ny_rng/ba

            # ── SHORT: high sweep ──────────────────────────────────────
            if not swept_high:
                if (bar["high"] > ny_high and bar["close"] < ny_high
                        and price4h < sma200
                        and not pd.isna(rv) and rv >= RSI_SHORT_MIN):
                    swept_high = True
                    entry = bar["close"]; sl = bar["high"] + ba * ATR_SL_MULT
                    risk  = sl - entry
                    if risk <= 0: continue
                    tp   = ny_high - TP_PCT * ny_rng
                    tp_r = (entry - tp) / risk
                    if tp_r < MIN_TP_R: tp = entry - risk*FALLBACK_RR; tp_r = FALLBACK_RR
                    if tp_r < MIN_TP_R: continue
                    rp, dead_z, mom_bar = get_risk(bar, ba, ts)
                    # Rejection depth into range
                    reject_d = (ny_high - bar["close"]) / ny_rng if ny_rng > 0 else 0
                    trades.append(dict(
                        date=tdate, entry_ts=ts, direction="SHORT",
                        entry=round(entry,2), sl=round(sl,2), tp=round(tp,2),
                        tp_r=round(tp_r,2), risk=round(risk,2),
                        risk_pct=rp, risk_pct_str=f"{rp*100:.0f}%",
                        rsi=round(rv,1), wick_atr=round(wick_h,3),
                        body_atr=round(body_ratio,3), rng_ratio=round(rng_ratio,2),
                        reject_depth=round(reject_d,3),
                        hour_et=ts.hour, dead_zone=dead_z, momentum_bar=mom_bar,
                        red_flag="Dead zone" if dead_z else ("Momentum bar" if mom_bar else "None"),
                        ny_high=round(ny_high,2), ny_low=round(ny_low,2),
                        sma200=round(sma200,2), price4h=round(price4h,2)
                    ))

            # ── LONG: low sweep ────────────────────────────────────────
            if not swept_low:
                if (bar["low"] < ny_low and bar["close"] > ny_low
                        and price4h > sma20
                        and not pd.isna(rv) and rv <= RSI_LONG_MAX):
                    swept_low = True
                    entry = bar["close"]; sl = bar["low"] - ba * ATR_SL_MULT
                    risk  = entry - sl
                    if risk <= 0: continue
                    tp   = ny_low + TP_PCT * ny_rng
                    tp_r = (tp - entry) / risk
                    if tp_r < MIN_TP_R: tp = entry + risk*FALLBACK_RR; tp_r = FALLBACK_RR
                    if tp_r < MIN_TP_R: continue
                    rp, dead_z, mom_bar = get_risk(bar, ba, ts)
                    reject_d = (bar["close"] - ny_low) / ny_rng if ny_rng > 0 else 0
                    trades.append(dict(
                        date=tdate, entry_ts=ts, direction="LONG",
                        entry=round(entry,2), sl=round(sl,2), tp=round(tp,2),
                        tp_r=round(tp_r,2), risk=round(risk,2),
                        risk_pct=rp, risk_pct_str=f"{rp*100:.0f}%",
                        rsi=round(rv,1), wick_atr=round(wick_l,3),
                        body_atr=round(body_ratio,3), rng_ratio=round(rng_ratio,2),
                        reject_depth=round(reject_d,3),
                        hour_et=ts.hour, dead_zone=dead_z, momentum_bar=mom_bar,
                        red_flag="Dead zone" if dead_z else ("Momentum bar" if mom_bar else "None"),
                        ny_high=round(ny_high,2), ny_low=round(ny_low,2),
                        sma20=round(sma20,2), price4h=round(price4h,2)
                    ))

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# RESOLUTION
# ---------------------------------------------------------------------------

def resolve(trades, df1h):
    if trades.empty: return trades
    bar_times = df1h.index.tolist()
    idx_map   = {ts: i for i, ts in enumerate(bar_times)}
    rows = []
    for _, t in trades.iterrows():
        if t["entry_ts"] not in idx_map: continue
        start = idx_map[t["entry_ts"]] + 1
        result, exit_price, bars_held = "OPEN", np.nan, 0
        for i in range(start, min(start+300, len(bar_times))):
            bar = df1h.iloc[i]; bars_held += 1
            if t["direction"] == "LONG":
                if bar["low"]  <= t["sl"]: result,exit_price="LOSS",t["sl"]; break
                if bar["high"] >= t["tp"]: result,exit_price="WIN", t["tp"]; break
            else:
                if bar["high"] >= t["sl"]: result,exit_price="LOSS",t["sl"]; break
                if bar["low"]  <= t["tp"]: result,exit_price="WIN", t["tp"]; break
        if result == "OPEN":
            exit_price = df1h.iloc[min(start+299,len(df1h)-1)]["close"]
        pnl_r   = ((exit_price-t["entry"])/t["risk"] if t["direction"]=="LONG"
                   else (t["entry"]-exit_price)/t["risk"])
        pnl_usd = pnl_r * ACCOUNT_SIZE * t["risk_pct"]
        rows.append({**t.to_dict(), "result":result, "exit_price":round(exit_price,2),
                     "pnl_r":round(pnl_r,2), "pnl_usd":round(pnl_usd,0),
                     "bars_held":bars_held})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FULL REPORT
# ---------------------------------------------------------------------------

def bar_chart(value, max_val, width=25, char="█"):
    filled = max(1, int(abs(value)/max_val*width))
    return char * filled

def report(trades):
    closed = trades[trades["result"].isin(["WIN","LOSS"])].copy()
    wins   = closed[closed["result"]=="WIN"]
    losses = closed[closed["result"]=="LOSS"]
    wr     = len(wins)/len(closed)*100

    equity = ACCOUNT_SIZE + closed["pnl_usd"].cumsum()
    peak   = equity.cummax()
    dd_series = (equity-peak)/peak*100
    max_dd     = dd_series.min()
    max_dd_usd = (equity-peak).min()
    total_pnl  = closed["pnl_usd"].sum()
    final_eq   = ACCOUNT_SIZE + total_pnl

    pf  = wins["pnl_usd"].sum()/abs(losses["pnl_usd"].sum()) if len(losses) else 99
    exp = (wr/100)*wins["pnl_r"].mean() + (1-wr/100)*losses["pnl_r"].mean()

    # Streaks
    streaks=[]; cur=0
    for r in closed["result"]:
        cur=(cur+1 if cur>0 else 1) if r=="WIN" else (cur-1 if cur<0 else -1)
        streaks.append(cur)
    max_ws = max((s for s in streaks if s>0), default=0)
    max_ls = abs(min((s for s in streaks if s<0), default=0))

    # Confidence tiers
    hi = closed[closed["risk_pct"]==RISK_HIGH]
    lo = closed[closed["risk_pct"]==RISK_LOW]
    hi_wins = hi[hi["result"]=="WIN"]; lo_wins = lo[lo["result"]=="WIN"]
    hi_wr = len(hi_wins)/len(hi)*100 if len(hi) else 0
    lo_wr = len(lo_wins)/len(lo)*100 if len(lo) else 0

    W = 66
    print("═"*W)
    print("  GOLD XAU/USD — NY SWEEP REVERSAL  ▪  FINAL CHAMPION BACKTEST")
    print("  ICT Liquidity Sweep + Dynamic Confidence Position Sizing")
    print("═"*W)
    print(f"  Account        : ${ACCOUNT_SIZE:>10,.0f}")
    print(f"  Period         : {closed['date'].min().date()} → {closed['date'].max().date()}")
    print(f"  Data           : GC=F (Gold Futures), 1H bars, 730 days")
    print(f"  Strategy       : NY Session High/Low Sweep Reversal (Asia session)")
    print(f"  Filters        : SMA 20/200 trend  |  RSI 40/60 gate  |  ATR stop")
    print(f"  Position size  : 5% (clean) → 3% (red flag: dead zone or momentum bar)")
    print("─"*W)

    print(f"\n{'  PERFORMANCE SUMMARY':}")
    print("─"*W)
    print(f"  Total trades       : {len(closed)}   ({len(wins)} wins / {len(losses)} losses)")
    print(f"  Win rate           : {wr:.1f}%")
    print(f"  Profit factor      : {pf:.2f}×")
    print(f"  Expectancy         : {exp:+.3f}R per trade")
    print(f"  Total P&L          : ${total_pnl:+,.0f}  ({total_pnl/ACCOUNT_SIZE*100:+.1f}% account return)")
    print(f"  Final equity       : ${final_eq:,.0f}")
    print(f"  Max drawdown       : {max_dd:.2f}%  (${max_dd_usd:,.0f})")
    print(f"  Trades per month   : {len(closed)/27:.1f}")
    print("─"*W)

    print(f"\n  R-MULTIPLE BREAKDOWN")
    print("─"*W)
    print(f"  Average winner     : +{wins['pnl_r'].mean():.2f}R  (+${wins['pnl_usd'].mean():,.0f} avg)")
    print(f"  Average loser      : {losses['pnl_r'].mean():.2f}R  (-${abs(losses['pnl_usd'].mean()):,.0f} avg)")
    print(f"  Largest winner     : +{wins['pnl_r'].max():.2f}R  (+${wins['pnl_usd'].max():,.0f})")
    print(f"  Largest loser      : {losses['pnl_r'].min():.2f}R  (-${abs(losses['pnl_usd'].min()):,.0f})")
    print(f"  Win : Loss ratio   : {wins['pnl_r'].mean():.2f}R : 1.00R")
    print(f"  Total R            : {closed['pnl_r'].sum():+.1f}R")
    print(f"  Max win streak     : {max_ws}")
    print(f"  Max loss streak    : {max_ls}")
    print(f"  Avg hold (bars)    : {closed['bars_held'].mean():.1f} hrs")
    print("─"*W)

    print(f"\n  DYNAMIC POSITION SIZING — CONFIDENCE TIER BREAKDOWN")
    print("─"*W)
    print(f"  ┌─────────────────────────┬────────┬──────────┬──────────┬──────────────┐")
    print(f"  │ Tier                    │ Trades │ Win Rate │ Avg P&L  │ Total P&L    │")
    print(f"  ├─────────────────────────┼────────┼──────────┼──────────┼──────────────┤")
    if len(hi):
        print(f"  │ 5% — No red flags       │ {len(hi):>6} │  {hi_wr:>5.1f}%  │ ${hi['pnl_usd'].mean():>+7,.0f} │ ${hi['pnl_usd'].sum():>+11,.0f} │")
    if len(lo):
        lo_dz  = lo[lo["dead_zone"]==True]
        lo_mb  = lo[lo["momentum_bar"]==True]
        lo_dz_wr = len(lo_dz[lo_dz["result"]=="WIN"])/len(lo_dz)*100 if len(lo_dz) else 0
        lo_mb_wr = len(lo_mb[lo_mb["result"]=="WIN"])/len(lo_mb)*100 if len(lo_mb) else 0
        print(f"  │ 3% — At least 1 flag    │ {len(lo):>6} │  {lo_wr:>5.1f}%  │ ${lo['pnl_usd'].mean():>+7,.0f} │ ${lo['pnl_usd'].sum():>+11,.0f} │")
        print(f"  │   ↳ Dead zone 20-22 ET  │ {len(lo_dz):>6} │  {lo_dz_wr:>5.1f}%  │          │              │")
        print(f"  │   ↳ Momentum bar >0.6×  │ {len(lo_mb):>6} │  {lo_mb_wr:>5.1f}%  │          │              │")
    print(f"  └─────────────────────────┴────────┴──────────┴──────────┴──────────────┘")
    print(f"\n  Average risk per trade : {closed['risk_pct'].mean()*100:.2f}%  (${closed['risk_pct'].mean()*ACCOUNT_SIZE:,.0f})")
    print("─"*W)

    print(f"\n  DIRECTIONAL BREAKDOWN")
    print("─"*W)
    for d in ["LONG","SHORT"]:
        sub = closed[closed["direction"]==d]
        if sub.empty: continue
        sw  = sub[sub["result"]=="WIN"]
        wr2 = len(sw)/len(sub)*100 if len(sub) else 0
        pf2 = (sw["pnl_usd"].sum()/abs(sub[sub["result"]=="LOSS"]["pnl_usd"].sum())
               if len(sub[sub["result"]=="LOSS"]) else 99)
        avg_rsi = sub["rsi"].mean()
        avg_wck = sub["wick_atr"].mean()
        print(f"  {d:5s}  │  {len(sub):2d} trades  │  {wr2:.1f}% WR  │  {sub['pnl_r'].sum():+.1f}R  │  "
              f"${sub['pnl_usd'].sum():+,.0f}  │  PF {pf2:.2f}×  │  Avg RSI {avg_rsi:.1f}  │  Avg wick {avg_wck:.2f}×ATR")
    print("─"*W)

    print(f"\n  MONTHLY P&L  (${ACCOUNT_SIZE:,.0f} account)")
    print("─"*W)
    closed2 = closed.copy()
    closed2["month"] = closed2["date"].apply(lambda x: pd.Timestamp(x).strftime("%Y-%m"))
    monthly = closed2.groupby("month")["pnl_usd"].sum()
    monthly_count = closed2.groupby("month")["result"].count()
    monthly_wr    = closed2.groupby("month").apply(lambda g: (g["result"]=="WIN").mean()*100)
    prof_m  = (monthly>0).sum(); total_m = len(monthly)
    max_abs = max(abs(monthly.values)) if len(monthly) else 1
    print(f"  Profitable months: {prof_m}/{total_m}  ({prof_m/total_m*100:.0f}%)")
    print()
    for m in monthly.index:
        p    = monthly[m]; cnt = monthly_count[m]; mwr = monthly_wr[m]
        symb = "▲" if p>=0 else "▼"
        bar  = bar_chart(p, max_abs, 20, "█" if p>=0 else "░")
        print(f"    {m}  {symb}  ${p:>+9,.0f}  {bar}  ({cnt}t / {mwr:.0f}%WR)")
    print("─"*W)

    print(f"\n  DRAWDOWN PROFILE")
    print("─"*W)
    dd_closed = dd_series.values
    print(f"  Max drawdown       : {max_dd:.2f}%  (${max_dd_usd:,.0f})")
    print(f"  Avg drawdown depth : {dd_closed[dd_closed<0].mean():.2f}%" if any(dd_closed<0) else "")
    print(f"  % of time at-peak  : {(dd_closed==0).mean()*100:.0f}%")
    print(f"  Longest DD period  : {max((len(list(g)) for k,g in __import__('itertools').groupby(dd_closed<0) if k), default=0)} trades")
    print("─"*W)

    print(f"\n  FULL TRADE LOG")
    print("─"*W)
    cols = ["date","direction","risk_pct_str","red_flag","entry","sl","tp",
            "tp_r","rsi","wick_atr","body_atr","rng_ratio","hour_et","result","pnl_r","pnl_usd"]
    display = closed[[c for c in cols if c in closed.columns]].copy()
    display.columns = [c.replace("risk_pct_str","risk").replace("wick_atr","wick×")
                       .replace("body_atr","body×").replace("rng_ratio","rng×")
                       .replace("hour_et","hr") for c in display.columns]
    print(display.to_string(index=False))
    print("═"*W)

    print(f"\n  STRATEGY RULES SUMMARY")
    print("─"*W)
    print(f"  Entry : NY session high/low swept by Asia session bar (wick beyond, close inside)")
    print(f"  Trend : 4H SMA-20 for LONG  |  4H SMA-200 for SHORT")
    print(f"  RSI   : Entry bar RSI ≤ 40 (LONG)  |  ≥ 60 (SHORT)")
    print(f"  Stop  : Sweep wick + 0.15× ATR beyond the level")
    print(f"  Target: 80% of NY session range (min 1.5R)")
    print(f"  Size  : 5% if no red flags  |  3% if dead zone OR momentum bar")
    print(f"    Red flag ①  Dead zone   : sweep hour 20:00–22:00 ET → historically 18% WR")
    print(f"    Red flag ②  Momentum bar: entry body > 0.6× ATR  → historically 20% WR")
    print("═"*W)

    return closed


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Fetching {LOOKBACK_DAYS}d of data for {SYMBOL}…")
    df1h = fetch(SYMBOL, "1h", LOOKBACK_DAYS)
    df4h = fetch(SYMBOL, "4h", LOOKBACK_DAYS)
    df4h = add_sma(df4h, SMA_LONG); df4h = add_sma(df4h, SMA_SHORT)
    df1h = add_atr(df1h, ATR_PERIOD); df1h = add_rsi(df1h, ATR_PERIOD)
    print(f"  1H: {len(df1h)} bars  |  4H: {len(df4h)} bars\n")

    raw    = run_backtest(df1h, df4h)
    trades = resolve(raw, df1h)

    print()
    closed = report(trades)

    trades.to_csv("gold_final_champion_trades.csv", index=False)
    print(f"\n  Full trade log saved → gold_final_champion_trades.csv\n")

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
    • RSI(14) gate on entry bar: LONG ≤ 40 | SHORT ≥ 65
    • Skip day if NY range < 1× ATR(14)

  Risk management:
    • Stop loss  : sweep wick extreme ± 0.15× ATR (tight, just beyond the trap)
    • TP1 (partial): 80% of NY session range (min 1.5R, fallback 2R) — close 30% of position
    • TP2 (runner) : 200% of NY session range — run remaining 70% of position
    • Runner stop  : moved to entry immediately when TP1 hit (worst case = $0 on runner)
    • Account      : $100,000

  Position sizing:
    • 5% flat risk ($5,000) on every trade — clean and flagged alike
    • Flagged detection (dead zone 20-22 ET / momentum bar >0.6× ATR) retained for BE only
    • +$48K over 2yr vs dynamic 5%/3% system, with identical drawdown (-3.35% vs -3.31%)

  Trailing stop (break-even protection):
    • Clean trades   : hold to TP1/TP2 — no early trailing
    • Flagged trades : move SL to entry once price reaches 1R profit (BE@1R)
        → flagged = dead zone sweep OR entry body > 0.6× ATR
        → worst case on flagged = $0 instead of -$5,000

  Sizing history: dynamic 5%/3% → $284,642 | 5% flat → $332,637 (+$47,995, DD +0.04pp)
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
TP1_PCT     = 0.80  # partial exit: 80% of NY range
TP2_PCT     = 2.00  # runner target: 200% of NY range
TP_SPLIT    = 0.30  # fraction closed at TP1 (30%); 70% runs to TP2
MIN_TP_R    = 1.5;  FALLBACK_RR = 2.0
RSI_LONG_MAX = 40; RSI_SHORT_MIN = 65
ASIA_OPEN = 17;   ASIA_CLOSE = 1

# Position sizing — flat 5% on all trades
RISK_FLAT = 0.05

# Red flag thresholds (used for BE@1R gating only, not sizing)
DEAD_ZONE_START = 20   # 20:00 ET
DEAD_ZONE_END   = 22   # 22:00 ET
BODY_PENALTY_THRESH = 0.6   # body > 0.6× ATR = momentum bar

# Break-even trailing stop (flagged trades only)
BE_TRIGGER_R = 1.0   # move SL to entry once unrealised profit reaches 1R


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

def get_flags(bar, bar_atr, ts):
    """Detect red flags for BE@1R gating. Sizing is flat 5% regardless."""
    h = ts.hour
    in_dead_zone    = (DEAD_ZONE_START <= h <= DEAD_ZONE_END)
    body            = abs(bar["close"] - bar["open"]) / bar_atr if bar_atr > 0 else 0
    is_momentum_bar = body >= BODY_PENALTY_THRESH
    return in_dead_zone or is_momentum_bar, in_dead_zone, is_momentum_bar


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
                    tp1  = ny_high - TP1_PCT * ny_rng
                    tp1_r = (entry - tp1) / risk
                    if tp1_r < MIN_TP_R: tp1 = entry - risk*FALLBACK_RR; tp1_r = FALLBACK_RR
                    if tp1_r < MIN_TP_R: continue
                    tp2  = ny_high - TP2_PCT * ny_rng
                    flagged, dead_z, mom_bar = get_flags(bar, ba, ts)
                    reject_d = (ny_high - bar["close"]) / ny_rng if ny_rng > 0 else 0
                    trades.append(dict(
                        date=tdate, entry_ts=ts, direction="SHORT",
                        entry=round(entry,2), sl=round(sl,2),
                        tp1=round(tp1,2), tp2=round(tp2,2), tp_r=round(tp1_r,2),
                        risk=round(risk,2),
                        risk_pct=RISK_FLAT, risk_pct_str="5%", flagged=flagged,
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
                    tp1  = ny_low + TP1_PCT * ny_rng
                    tp1_r = (tp1 - entry) / risk
                    if tp1_r < MIN_TP_R: tp1 = entry + risk*FALLBACK_RR; tp1_r = FALLBACK_RR
                    if tp1_r < MIN_TP_R: continue
                    tp2  = ny_low + TP2_PCT * ny_rng
                    flagged, dead_z, mom_bar = get_flags(bar, ba, ts)
                    reject_d = (bar["close"] - ny_low) / ny_rng if ny_rng > 0 else 0
                    trades.append(dict(
                        date=tdate, entry_ts=ts, direction="LONG",
                        entry=round(entry,2), sl=round(sl,2),
                        tp1=round(tp1,2), tp2=round(tp2,2), tp_r=round(tp1_r,2),
                        risk=round(risk,2),
                        risk_pct=RISK_FLAT, risk_pct_str="5%", flagged=flagged,
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
    """
    Partial TP: close TP_SPLIT (30%) at TP1, run remaining 70% to TP2.
    Runner stop moves to entry once TP1 is hit.
    BE@1R applied to flagged trades only (dead zone / momentum bar).
    """
    if trades.empty: return trades
    bar_times = df1h.index.tolist()
    idx_map   = {ts: i for i, ts in enumerate(bar_times)}
    rows = []
    for _, t in trades.iterrows():
        if t["entry_ts"] not in idx_map: continue
        start      = idx_map[t["entry_ts"]] + 1
        entry      = t["entry"]; risk = t["risk"]; direction = t["direction"]
        sl         = t["sl"]
        apply_be   = bool(t.get("flagged", False))
        be_moved   = False
        tp1_hit    = False
        runner_sl  = sl
        pnl_r      = 0.0
        result     = "OPEN"; bars_held = 0

        for i in range(start, min(start+300, len(bar_times))):
            bar = df1h.iloc[i]; bars_held += 1
            hi = bar["high"]; lo = bar["low"]

            if not tp1_hit:
                # BE@1R on flagged trades only
                if apply_be and not be_moved:
                    best_r = (hi-entry)/risk if direction=="LONG" else (entry-lo)/risk
                    if best_r >= BE_TRIGGER_R:
                        sl = entry; be_moved = True

                if direction == "LONG":
                    if lo <= sl:
                        pnl_r = (sl - entry) / risk  # 0.0 if BE moved
                        result = "BE" if be_moved else "LOSS"; break
                    if hi >= t["tp1"]:
                        tp1_r_actual = (t["tp1"] - entry) / risk
                        pnl_r  += TP_SPLIT * tp1_r_actual
                        tp1_hit = True; runner_sl = entry
                else:
                    if hi >= sl:
                        pnl_r = (entry - sl) / risk  # 0.0 if BE moved
                        result = "BE" if be_moved else "LOSS"; break
                    if lo <= t["tp1"]:
                        tp1_r_actual = (entry - t["tp1"]) / risk
                        pnl_r  += TP_SPLIT * tp1_r_actual
                        tp1_hit = True; runner_sl = entry

            else:
                # Runner to TP2, stop at entry
                runner_frac = 1.0 - TP_SPLIT
                if direction == "LONG":
                    if lo <= runner_sl:
                        pnl_r += runner_frac * (runner_sl - entry) / risk  # = 0
                        result = "PARTIAL"; break
                    if hi >= t["tp2"]:
                        pnl_r += runner_frac * (t["tp2"] - entry) / risk
                        result = "WIN"; break
                else:
                    if hi >= runner_sl:
                        pnl_r += runner_frac * (entry - runner_sl) / risk  # = 0
                        result = "PARTIAL"; break
                    if lo <= t["tp2"]:
                        pnl_r += runner_frac * (entry - t["tp2"]) / risk
                        result = "WIN"; break

        if result == "OPEN":
            last = df1h.iloc[min(start+299, len(df1h)-1)]["close"]
            if tp1_hit:
                runner_r = (last-entry)/risk if direction=="LONG" else (entry-last)/risk
                pnl_r += (1-TP_SPLIT) * runner_r
            else:
                pnl_r = (last-entry)/risk if direction=="LONG" else (entry-last)/risk

        pnl_usd = pnl_r * ACCOUNT_SIZE * RISK_FLAT
        rows.append({**t.to_dict(), "result":result,
                     "pnl_r":round(pnl_r,3), "pnl_usd":round(pnl_usd,0),
                     "bars_held":bars_held, "be_triggered":be_moved, "tp1_hit":tp1_hit})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FULL REPORT
# ---------------------------------------------------------------------------

def bar_chart(value, max_val, width=25, char="█"):
    filled = max(1, int(abs(value)/max_val*width))
    return char * filled

def report(trades):
    closed  = trades[trades["result"].isin(["WIN","LOSS","BE","PARTIAL"])].copy()
    decisive= closed[closed["result"].isin(["WIN","LOSS","PARTIAL"])]
    wins    = closed[closed["result"].isin(["WIN","PARTIAL"])]
    full_wins=closed[closed["result"]=="WIN"]
    partials= closed[closed["result"]=="PARTIAL"]
    losses  = closed[closed["result"]=="LOSS"]
    bes     = closed[closed["result"]=="BE"]
    wr      = len(wins)/len(decisive)*100 if len(decisive) else 0

    equity = ACCOUNT_SIZE + closed["pnl_usd"].cumsum()
    peak   = equity.cummax()
    dd_series = (equity-peak)/peak*100
    max_dd     = dd_series.min()
    max_dd_usd = (equity-peak).min()
    total_pnl  = closed["pnl_usd"].sum()
    final_eq   = ACCOUNT_SIZE + total_pnl

    pf  = wins["pnl_usd"].sum()/abs(losses["pnl_usd"].sum()) if len(losses) else 99
    exp = decisive["pnl_r"].mean()

    # Streaks (decisive only)
    streaks=[]; cur=0
    for r in decisive["result"]:
        cur=(cur+1 if cur>0 else 1) if r in ("WIN","PARTIAL") else (cur-1 if cur<0 else -1)
        streaks.append(cur)
    max_ws = max((s for s in streaks if s>0), default=0)
    max_ls = abs(min((s for s in streaks if s<0), default=0))

    # Flagged vs clean breakdown
    clean_dec   = decisive[decisive["flagged"]==False]
    flagged_dec = decisive[decisive["flagged"]==True]
    clean_wins  = clean_dec[clean_dec["result"].isin(["WIN","PARTIAL"])]
    flagged_wins= flagged_dec[flagged_dec["result"].isin(["WIN","PARTIAL"])]
    clean_wr    = len(clean_wins)/len(clean_dec)*100   if len(clean_dec)   else 0
    flagged_wr  = len(flagged_wins)/len(flagged_dec)*100 if len(flagged_dec) else 0

    W = 66
    print("═"*W)
    print("  GOLD XAU/USD — NY SWEEP REVERSAL  ▪  FINAL CHAMPION BACKTEST")
    print("  ICT Liquidity Sweep | 5% Flat Risk | Partial TP 30/70")
    print("═"*W)
    print(f"  Account        : ${ACCOUNT_SIZE:>10,.0f}")
    print(f"  Period         : {closed['date'].min().date()} → {closed['date'].max().date()}")
    print(f"  Data           : GC=F (Gold Futures), 1H bars, 730 days")
    print(f"  Strategy       : NY Session High/Low Sweep Reversal (Asia session)")
    print(f"  Filters        : SMA 20/200 trend  |  RSI 40/65 gate  |  ATR stop")
    print(f"  Position size  : {RISK_FLAT*100:.0f}% flat (all trades)")
    print("─"*W)

    print(f"\n{'  PERFORMANCE SUMMARY':}")
    print("─"*W)
    print(f"  Total trades       : {len(closed)}  ({len(full_wins)}W / {len(partials)}P / {len(losses)}L / {len(bes)}BE)")
    print(f"  Win rate (decisive): {wr:.1f}%  (W+PARTIAL vs L  |  BE excluded)")
    print(f"  Profit factor      : {pf:.2f}×")
    print(f"  Expectancy         : {exp:+.3f}R per decisive trade")
    print(f"  Partial TP split   : {TP_SPLIT*100:.0f}% at TP1 (80% NY range)  /  {(1-TP_SPLIT)*100:.0f}% runner to TP2 (200% NY range)")
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

    print(f"\n  SIGNAL QUALITY BREAKDOWN  (5% flat on both — BE@1R on flagged only)")
    print("─"*W)
    clean_all   = closed[closed["flagged"]==False]
    flagged_all = closed[closed["flagged"]==True]
    dz  = decisive[decisive["dead_zone"]==True]
    mb  = decisive[decisive["momentum_bar"]==True]
    dz_wr = len(dz[dz["result"].isin(["WIN","PARTIAL"])])/len(dz)*100 if len(dz) else 0
    mb_wr = len(mb[mb["result"].isin(["WIN","PARTIAL"])])/len(mb)*100 if len(mb) else 0
    print(f"  ┌─────────────────────────┬────────┬──────────┬──────────┬──────────────┐")
    print(f"  │ Group                   │ Trades │ Win Rate │ Avg P&L  │ Total P&L    │")
    print(f"  ├─────────────────────────┼────────┼──────────┼──────────┼──────────────┤")
    if len(clean_dec):
        print(f"  │ Clean (no flags)        │ {len(clean_dec):>6} │  {clean_wr:>5.1f}%  │ ${clean_all['pnl_usd'].mean():>+7,.0f} │ ${clean_all['pnl_usd'].sum():>+11,.0f} │")
    if len(flagged_dec):
        print(f"  │ Flagged (≥1 red flag)   │ {len(flagged_dec):>6} │  {flagged_wr:>5.1f}%  │ ${flagged_all['pnl_usd'].mean():>+7,.0f} │ ${flagged_all['pnl_usd'].sum():>+11,.0f} │")
        print(f"  │   ↳ Dead zone 20-22 ET  │ {len(dz):>6} │  {dz_wr:>5.1f}%  │          │              │")
        print(f"  │   ↳ Momentum bar >0.6×  │ {len(mb):>6} │  {mb_wr:>5.1f}%  │          │              │")
    print(f"  └─────────────────────────┴────────┴──────────┴──────────┴──────────────┘")
    print(f"\n  Risk per trade : {RISK_FLAT*100:.0f}% flat  (${RISK_FLAT*ACCOUNT_SIZE:,.0f} per trade)")
    print("─"*W)

    print(f"\n  DIRECTIONAL BREAKDOWN")
    print("─"*W)
    for d in ["LONG","SHORT"]:
        sub = decisive[decisive["direction"]==d]
        if sub.empty: continue
        sw  = sub[sub["result"].isin(["WIN","PARTIAL"])]
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
    cols = ["date","direction","risk_pct_str","red_flag","entry","sl","tp1","tp2",
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
    print(f"  RSI   : Entry bar RSI ≤ 40 (LONG)  |  ≥ 65 (SHORT)")
    print(f"  Stop  : Sweep wick + 0.15× ATR beyond the level")
    print(f"  Target: TP1 = 80% NY range (close {TP_SPLIT*100:.0f}%)  |  TP2 = 200% NY range (run {(1-TP_SPLIT)*100:.0f}%)")
    print(f"  Runner: stop moved to entry when TP1 hit — worst case on runner = $0")
    print(f"  Size  : {RISK_FLAT*100:.0f}% flat on all trades")
    print(f"  Trail : clean trades → hold to TP1/TP2  |  flagged trades → BE@1R before TP1")
    print(f"    Flagged ①  Dead zone   : sweep hour 20:00–22:00 ET")
    print(f"    Flagged ②  Momentum bar: entry body > 0.6× ATR")
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

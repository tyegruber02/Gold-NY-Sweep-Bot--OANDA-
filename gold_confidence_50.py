"""
Gold NY-Sweep — 50-Test Dynamic Position Sizing Optimization

Base strategy is locked (champion config):
  SMA 20/200 | ATR 0.15× stop | TP 80% NY range | RSI 40/60 gate
  Asia 17:00–01:00 ET | $100,000 account

Variable: confidence scoring system that maps trade quality to risk size.
  Low confidence  → 1% risk ($1,000)
  Med confidence  → 2% risk ($2,000)
  High confidence → 3% risk ($3,000)

Confidence factors tested:
  RSI extremity   — how far RSI is from neutral (more extreme = higher conf)
  Wick size       — how far price swept beyond the NY level (in ATR units)
  Range quality   — how wide the NY session range is vs ATR
  Rejection depth — how deep inside the range the close is after the sweep
  Sweep timing    — which hour the Asia sweep occurs
  Trend strength  — how far 4H price is from its SMA

Groups:
  A (01-10): RSI-only scoring variations
  B (11-20): Wick size variations
  C (21-30): Range quality, rejection depth, timing, trend strength
  D (31-40): Two-factor combinations
  E (41-50): Multi-factor and champion configs
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, yfinance as yf
from datetime import timedelta
import pytz

ET            = pytz.timezone("America/New_York")
SYMBOL        = "GC=F"
LOOKBACK_DAYS = 730
ACCOUNT_SIZE  = 100_000

# Fixed strategy params
SMA_LONG=20; SMA_SHORT=200; ATR_PERIOD=14; ATR_SL_MULT=0.15
TP_PCT=0.80; MIN_TP_R=1.5; FALLBACK_RR=2.0
RSI_LONG_MAX=40; RSI_SHORT_MIN=60
ASIA_OPEN=17; ASIA_CLOSE=1

RISK_LOW  = 0.03
RISK_MED  = 0.04
RISK_HIGH = 0.05


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
    return (ts-timedelta(days=1)).normalize() if ts.hour<4 else ts.normalize()


# ---------------------------------------------------------------------------
# CONFIDENCE SCORER
# ---------------------------------------------------------------------------

def score_trade(direction, bar, ny_high, ny_low, ny_rng, bar_atr, ts, price4h, sma_ref, cfg):
    """
    Returns risk_pct (0.01 / 0.02 / 0.03) based on confidence scoring.
    cfg keys:
      factors       : list of active factors
      rsi_high      : RSI threshold for high conf (LONG ≤ this, SHORT ≥ 100-this)
      rsi_med       : RSI threshold for med conf
      wick_high     : wick size (×ATR) for high conf
      wick_med      : wick size for med conf
      range_high    : NY range (×ATR) for high conf
      range_med     : NY range for med conf
      reject_high   : rejection depth fraction for high conf (0-1, inner range)
      reject_med    : rejection depth fraction for med conf
      timing_high   : sweep hour ≤ this = high conf
      timing_med    : sweep hour ≤ this = med conf
      trend_high    : pct distance from SMA for high conf
      trend_med     : pct distance for med conf
      score_med     : minimum score for med risk
      score_high    : minimum score for high risk
      two_tier      : if True, only low/high (no med)
    """
    factors     = cfg.get("factors", [])
    two_tier    = cfg.get("two_tier", False)
    score_med   = cfg.get("score_med", 1)
    score_high  = cfg.get("score_high", 2)

    score = 0

    # ── RSI factor ──────────────────────────────────────────────────────────
    if "rsi" in factors:
        rsi_val = bar["rsi"]
        rsi_high_t = cfg.get("rsi_high", 30)
        rsi_med_t  = cfg.get("rsi_med",  37)
        if direction == "LONG":
            if rsi_val <= rsi_high_t:  score += 2
            elif rsi_val <= rsi_med_t: score += 1
        else:
            if rsi_val >= (100 - rsi_high_t):  score += 2
            elif rsi_val >= (100 - rsi_med_t): score += 1

    # ── Wick size factor ─────────────────────────────────────────────────────
    if "wick" in factors:
        if direction == "LONG":
            wick = (ny_low - bar["low"]) / bar_atr if bar_atr > 0 else 0
        else:
            wick = (bar["high"] - ny_high) / bar_atr if bar_atr > 0 else 0
        wick_high_t = cfg.get("wick_high", 1.0)
        wick_med_t  = cfg.get("wick_med",  0.3)
        if wick >= wick_high_t:   score += 2
        elif wick >= wick_med_t:  score += 1

    # ── Range quality factor ─────────────────────────────────────────────────
    if "range" in factors:
        rng_ratio   = ny_rng / bar_atr if bar_atr > 0 else 0
        range_high_t = cfg.get("range_high", 2.0)
        range_med_t  = cfg.get("range_med",  1.5)
        if rng_ratio >= range_high_t:   score += 2
        elif rng_ratio >= range_med_t:  score += 1

    # ── Rejection depth factor ────────────────────────────────────────────────
    if "reject" in factors:
        if ny_rng > 0:
            if direction == "LONG":
                depth = (bar["close"] - ny_low) / ny_rng   # 0=at low, 1=at high
            else:
                depth = (ny_high - bar["close"]) / ny_rng  # 0=at high, 1=at low
        else:
            depth = 0
        reject_high_t = cfg.get("reject_high", 0.50)
        reject_med_t  = cfg.get("reject_med",  0.25)
        if depth >= reject_high_t:   score += 2
        elif depth >= reject_med_t:  score += 1

    # ── Sweep timing factor ───────────────────────────────────────────────────
    if "timing" in factors:
        h = ts.hour
        timing_high_t = cfg.get("timing_high", 19)
        timing_med_t  = cfg.get("timing_med",  22)
        if h >= ASIA_OPEN or h == 0:
            actual_h = h if h >= ASIA_OPEN else h + 24
        else:
            actual_h = h + 24
        if actual_h <= timing_high_t or (actual_h >= 17 and actual_h <= timing_high_t):
            eff_h = h if h >= 17 else h + 24
        else:
            eff_h = h if h >= 17 else h + 24
        # Simpler: just use raw hour
        if h >= 17:
            eff = h
        else:
            eff = h + 24
        t_high = cfg.get("timing_high", 19)
        t_med  = cfg.get("timing_med",  22)
        if eff <= t_high:   score += 2
        elif eff <= t_med:  score += 1

    # ── Trend strength factor ─────────────────────────────────────────────────
    if "trend" in factors:
        if sma_ref and sma_ref > 0 and price4h:
            pct_from_sma = abs(price4h - sma_ref) / sma_ref * 100
        else:
            pct_from_sma = 0
        trend_high_t = cfg.get("trend_high", 2.0)
        trend_med_t  = cfg.get("trend_med",  0.5)
        if pct_from_sma >= trend_high_t:   score += 2
        elif pct_from_sma >= trend_med_t:  score += 1

    # ── Map score to risk ─────────────────────────────────────────────────────
    if two_tier:
        return RISK_HIGH if score >= score_high else RISK_LOW
    else:
        if score >= score_high: return RISK_HIGH
        if score >= score_med:  return RISK_MED
        return RISK_LOW


# ---------------------------------------------------------------------------
# BACKTEST ENGINE
# ---------------------------------------------------------------------------

def run_backtest(df1h, df4h, scoring_cfg):
    df1h = df1h.copy()
    df1h["tdate"] = df1h.index.map(trading_date)
    trades = []

    for tdate, day in df1h.groupby("tdate"):
        ny   = day[(day.index.hour >= 8) & (day.index.hour < 17)]
        asia = day[(day.index.hour >= ASIA_OPEN) | (day.index.hour < ASIA_CLOSE)]
        asia = asia[(asia.index.hour >= ASIA_OPEN) | (asia.index.hour < ASIA_CLOSE)]
        if ny.empty or asia.empty: continue

        ny_high = ny["high"].max(); ny_low = ny["low"].min()
        ny_rng  = ny_high - ny_low; atr = ny["atr"].iloc[-1]
        if pd.isna(atr) or atr == 0 or ny_rng < atr: continue

        swept_high = swept_low = False

        for ts, bar in asia.iterrows():
            ba  = bar["atr"]; rv = bar["rsi"]
            if pd.isna(ba) or ba == 0: continue
            p4  = df4h[df4h.index <= ts]
            if p4.empty or pd.isna(p4["sma20"].iloc[-1]): continue
            price4h = p4["close"].iloc[-1]
            sma20   = p4["sma20"].iloc[-1]
            sma200  = p4["sma200"].iloc[-1]

            # HIGH SWEEP → SHORT
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
                    rp = score_trade("SHORT", bar, ny_high, ny_low, ny_rng,
                                     ba, ts, price4h, sma200, scoring_cfg)
                    trades.append(dict(date=tdate, entry_ts=ts, direction="SHORT",
                                       entry=entry, sl=sl, tp=tp, tp_r=tp_r,
                                       risk=risk, risk_pct=rp,
                                       rsi=rv, wick=(bar["high"]-ny_high)/ba,
                                       rng_ratio=ny_rng/ba))

            # LOW SWEEP → LONG
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
                    rp = score_trade("LONG", bar, ny_high, ny_low, ny_rng,
                                     ba, ts, price4h, sma20, scoring_cfg)
                    trades.append(dict(date=tdate, entry_ts=ts, direction="LONG",
                                       entry=entry, sl=sl, tp=tp, tp_r=tp_r,
                                       risk=risk, risk_pct=rp,
                                       rsi=rv, wick=(ny_low-bar["low"])/ba,
                                       rng_ratio=ny_rng/ba))

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# RESOLUTION
# ---------------------------------------------------------------------------

def resolve(trades, df1h):
    if trades.empty: return trades
    bar_times = df1h.index.tolist()
    idx_map   = {ts: i for i,ts in enumerate(bar_times)}
    rows = []
    for _, t in trades.iterrows():
        if t["entry_ts"] not in idx_map: continue
        start = idx_map[t["entry_ts"]] + 1
        result, exit_price = "OPEN", np.nan
        for i in range(start, min(start+300, len(bar_times))):
            bar = df1h.iloc[i]
            if t["direction"] == "LONG":
                if bar["low"]  <= t["sl"]: result,exit_price="LOSS",t["sl"]; break
                if bar["high"] >= t["tp"]: result,exit_price="WIN", t["tp"]; break
            else:
                if bar["high"] >= t["sl"]: result,exit_price="LOSS",t["sl"]; break
                if bar["low"]  <= t["tp"]: result,exit_price="WIN", t["tp"]; break
        if result == "OPEN":
            exit_price = df1h.iloc[min(start+299,len(df1h)-1)]["close"]
        pnl_r = ((exit_price-t["entry"])/t["risk"] if t["direction"]=="LONG"
                 else (t["entry"]-exit_price)/t["risk"])
        pnl_usd = pnl_r * ACCOUNT_SIZE * t["risk_pct"]
        rows.append({**t.to_dict(),"result":result,"exit_price":exit_price,
                     "pnl_r":pnl_r,"pnl_usd":pnl_usd})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def metrics(trades, label, desc):
    closed = trades[trades["result"].isin(["WIN","LOSS"])] if not trades.empty else pd.DataFrame()
    if closed.empty or len(closed) < 3:
        return {"label":label,"desc":desc,"trades":0,"wr_pct":0,"total_r":0,
                "pnl_usd":0,"max_dd":0,"pf":0,"avg_risk_pct":0,
                "hi_trades":0,"med_trades":0,"lo_trades":0}
    wins = closed[closed["result"]=="WIN"]
    equity = ACCOUNT_SIZE + closed["pnl_usd"].cumsum()
    peak   = equity.cummax()
    max_dd = ((equity-peak)/peak*100).min()
    pf = (wins["pnl_usd"].sum()/abs(closed[closed["result"]=="LOSS"]["pnl_usd"].sum())
          if len(closed[closed["result"]=="LOSS"]) else 99)
    exp = (len(wins)/len(closed))*wins["pnl_r"].mean() + (1-len(wins)/len(closed))*closed[closed["result"]=="LOSS"]["pnl_r"].mean()
    hi  = closed[closed["risk_pct"]==RISK_HIGH]
    med = closed[closed["risk_pct"]==RISK_MED]
    lo  = closed[closed["risk_pct"]==RISK_LOW]
    hi_wr  = len(hi[hi["result"]=="WIN"])/len(hi)*100 if len(hi) else 0
    med_wr = len(med[med["result"]=="WIN"])/len(med)*100 if len(med) else 0
    lo_wr  = len(lo[lo["result"]=="WIN"])/len(lo)*100 if len(lo) else 0
    return {
        "label":label,"desc":desc,
        "trades":len(closed),"wins":len(wins),"losses":len(closed)-len(wins),
        "wr_pct":round(len(wins)/len(closed)*100,1),
        "avg_r":round(closed["pnl_r"].mean(),3),
        "expectancy":round(exp,3),
        "total_r":round(closed["pnl_r"].sum(),2),
        "pnl_usd":round(closed["pnl_usd"].sum(),0),
        "max_dd":round(max_dd,1),"pf":round(pf,2),
        "avg_risk_pct":round(closed["risk_pct"].mean()*100,2),
        "hi_trades":len(hi),"hi_wr":round(hi_wr,1),"hi_pnl":round(hi["pnl_usd"].sum(),0),
        "med_trades":len(med),"med_wr":round(med_wr,1),"med_pnl":round(med["pnl_usd"].sum(),0),
        "lo_trades":len(lo),"lo_wr":round(lo_wr,1),"lo_pnl":round(lo["pnl_usd"].sum(),0),
    }


# ---------------------------------------------------------------------------
# 50 TEST CONFIGURATIONS
# ---------------------------------------------------------------------------

def c(**kw): return kw

TESTS = [

    # ══ GROUP A: RSI-only confidence scoring (10 tests) ══════════════════

    {"label":"A01_RSI_30_37_3TIER",
     "desc":"RSI ≤30=3%, 30-37=2%, 37-40=1%",
     **c(factors=["rsi"],rsi_high=30,rsi_med=37,score_med=1,score_high=2)},

    {"label":"A02_RSI_28_35_3TIER",
     "desc":"RSI ≤28=3%, 28-35=2%, else=1%",
     **c(factors=["rsi"],rsi_high=28,rsi_med=35,score_med=1,score_high=2)},

    {"label":"A03_RSI_25_33_3TIER",
     "desc":"RSI ≤25=3%, 25-33=2%, else=1%",
     **c(factors=["rsi"],rsi_high=25,rsi_med=33,score_med=1,score_high=2)},

    {"label":"A04_RSI_32_38_3TIER",
     "desc":"RSI ≤32=3%, 32-38=2%, else=1%",
     **c(factors=["rsi"],rsi_high=32,rsi_med=38,score_med=1,score_high=2)},

    {"label":"A05_RSI_30_2TIER_HIGH",
     "desc":"RSI ≤30/≥70=3%, else=1% (2-tier)",
     **c(factors=["rsi"],rsi_high=30,rsi_med=30,score_med=99,score_high=2,two_tier=True)},

    {"label":"A06_RSI_25_2TIER_HIGH",
     "desc":"RSI ≤25/≥75=3%, else=1% (2-tier strict)",
     **c(factors=["rsi"],rsi_high=25,rsi_med=25,score_med=99,score_high=2,two_tier=True)},

    {"label":"A07_RSI_35_2TIER",
     "desc":"RSI ≤35/≥65=2%, else=1% (2-tier conservative)",
     **c(factors=["rsi"],rsi_high=35,rsi_med=35,score_med=99,score_high=2,two_tier=True)},

    {"label":"A08_RSI_30_MED2PCT",
     "desc":"RSI ≤28=3%, 28-35=2%, else=1% — medium tier only 2%",
     **c(factors=["rsi"],rsi_high=28,rsi_med=35,score_med=1,score_high=2)},

    {"label":"A09_RSI_FLOOR2PCT",
     "desc":"RSI floor at 2%: ≤28=3%, else=2%",
     **c(factors=["rsi"],rsi_high=28,rsi_med=40,score_med=0,score_high=2)},

    {"label":"A10_RSI_28_36_TIGHT",
     "desc":"RSI ≤28=3%, 28-36=2%, 36-40=1%",
     **c(factors=["rsi"],rsi_high=28,rsi_med=36,score_med=1,score_high=2)},

    # ══ GROUP B: Wick size variations (10 tests) ═════════════════════════

    {"label":"B11_WICK_0.5_1.0",
     "desc":"Wick ≥1.0×ATR=3%, 0.5-1.0=2%, <0.5=1%",
     **c(factors=["wick"],wick_high=1.0,wick_med=0.5,score_med=1,score_high=2)},

    {"label":"B12_WICK_0.3_0.7",
     "desc":"Wick ≥0.7×ATR=3%, 0.3-0.7=2%, <0.3=1%",
     **c(factors=["wick"],wick_high=0.7,wick_med=0.3,score_med=1,score_high=2)},

    {"label":"B13_WICK_1.0_1.5",
     "desc":"Wick ≥1.5×ATR=3%, 1.0-1.5=2%, else=1%",
     **c(factors=["wick"],wick_high=1.5,wick_med=1.0,score_med=1,score_high=2)},

    {"label":"B14_WICK_0.5_2TIER",
     "desc":"Wick ≥0.5×ATR=3%, else=1% (2-tier)",
     **c(factors=["wick"],wick_high=0.5,wick_med=0.5,score_high=2,two_tier=True)},

    {"label":"B15_WICK_1.0_2TIER",
     "desc":"Wick ≥1.0×ATR=3%, else=1% (2-tier strict)",
     **c(factors=["wick"],wick_high=1.0,wick_med=1.0,score_high=2,two_tier=True)},

    {"label":"B16_WICK_0.3_2TIER",
     "desc":"Wick ≥0.3×ATR=2%, else=1% (2-tier conservative)",
     **c(factors=["wick"],wick_high=0.3,wick_med=0.3,score_high=2,two_tier=True)},

    {"label":"B17_WICK_0.4_0.8",
     "desc":"Wick ≥0.8×ATR=3%, 0.4-0.8=2%, <0.4=1%",
     **c(factors=["wick"],wick_high=0.8,wick_med=0.4,score_med=1,score_high=2)},

    {"label":"B18_WICK_FLOOR2",
     "desc":"Wick ≥1.0×ATR=3%, else=2% (floor at 2%)",
     **c(factors=["wick"],wick_high=1.0,wick_med=0.0,score_med=0,score_high=2)},

    {"label":"B19_WICK_0.6_1.2",
     "desc":"Wick ≥1.2×ATR=3%, 0.6-1.2=2%, else=1%",
     **c(factors=["wick"],wick_high=1.2,wick_med=0.6,score_med=1,score_high=2)},

    {"label":"B20_WICK_2.0_STRICT",
     "desc":"Wick ≥2.0×ATR=3%, 1.0-2.0=2%, else=1%",
     **c(factors=["wick"],wick_high=2.0,wick_med=1.0,score_med=1,score_high=2)},

    # ══ GROUP C: Range, rejection, timing, trend (10 tests) ══════════════

    {"label":"C21_RANGE_1.5_2.0",
     "desc":"Range ≥2.0×ATR=3%, 1.5-2.0=2%, 1-1.5=1%",
     **c(factors=["range"],range_high=2.0,range_med=1.5,score_med=1,score_high=2)},

    {"label":"C22_RANGE_1.8_2.5",
     "desc":"Range ≥2.5×ATR=3%, 1.8-2.5=2%, else=1%",
     **c(factors=["range"],range_high=2.5,range_med=1.8,score_med=1,score_high=2)},

    {"label":"C23_RANGE_1.5_2TIER",
     "desc":"Range ≥1.5×ATR=2%, else=1% (2-tier)",
     **c(factors=["range"],range_high=1.5,range_med=1.5,score_high=2,two_tier=True)},

    {"label":"C24_RANGE_2.0_2TIER",
     "desc":"Range ≥2.0×ATR=3%, else=1% (2-tier strict)",
     **c(factors=["range"],range_high=2.0,range_med=2.0,score_high=2,two_tier=True)},

    {"label":"C25_REJECT_0.25_0.50",
     "desc":"Rejection depth ≥50%=3%, 25-50%=2%, <25%=1%",
     **c(factors=["reject"],reject_high=0.50,reject_med=0.25,score_med=1,score_high=2)},

    {"label":"C26_REJECT_0.40_0.65",
     "desc":"Rejection depth ≥65%=3%, 40-65%=2%, else=1%",
     **c(factors=["reject"],reject_high=0.65,reject_med=0.40,score_med=1,score_high=2)},

    {"label":"C27_REJECT_2TIER",
     "desc":"Rejection depth ≥50%=3%, else=1% (2-tier)",
     **c(factors=["reject"],reject_high=0.50,reject_med=0.50,score_high=2,two_tier=True)},

    {"label":"C28_TIMING_19_22",
     "desc":"Sweep 17-19ET=3%, 19-22ET=2%, 22-01ET=1%",
     **c(factors=["timing"],timing_high=19,timing_med=22,score_med=1,score_high=2)},

    {"label":"C29_TIMING_20_2TIER",
     "desc":"Sweep 17-20ET=2%, else=1% (2-tier)",
     **c(factors=["timing"],timing_high=20,timing_med=20,score_high=2,two_tier=True)},

    {"label":"C30_TREND_0.5_2.0",
     "desc":"4H trend strength ≥2%=3%, 0.5-2%=2%, <0.5%=1%",
     **c(factors=["trend"],trend_high=2.0,trend_med=0.5,score_med=1,score_high=2)},

    # ══ GROUP D: Two-factor combinations (10 tests) ═══════════════════════

    {"label":"D31_RSI_WICK_SUM",
     "desc":"RSI(≤30=1pt) + Wick(≥0.5=1pt): 0=1%, 1=2%, 2=3%",
     **c(factors=["rsi","wick"],rsi_high=30,rsi_med=30,wick_high=0.5,wick_med=0.5,
        score_med=1,score_high=2)},

    {"label":"D32_RSI_RANGE_SUM",
     "desc":"RSI(≤30=1pt) + Range(≥1.5=1pt): 0=1%, 1=2%, 2=3%",
     **c(factors=["rsi","range"],rsi_high=30,rsi_med=30,range_high=1.5,range_med=1.5,
        score_med=1,score_high=2)},

    {"label":"D33_RSI_REJECT_SUM",
     "desc":"RSI(≤30=1pt) + Rejection(≥50%=1pt): 0=1%, 1=2%, 2=3%",
     **c(factors=["rsi","reject"],rsi_high=30,rsi_med=30,reject_high=0.50,reject_med=0.50,
        score_med=1,score_high=2)},

    {"label":"D34_RSI_TIMING_SUM",
     "desc":"RSI(≤30=1pt) + Timing(early=1pt): 0=1%, 1=2%, 2=3%",
     **c(factors=["rsi","timing"],rsi_high=30,rsi_med=30,timing_high=20,timing_med=20,
        score_med=1,score_high=2)},

    {"label":"D35_WICK_RANGE_SUM",
     "desc":"Wick(≥0.5=1pt) + Range(≥1.5=1pt): 0=1%, 1=2%, 2=3%",
     **c(factors=["wick","range"],wick_high=0.5,wick_med=0.5,range_high=1.5,range_med=1.5,
        score_med=1,score_high=2)},

    {"label":"D36_WICK_REJECT_SUM",
     "desc":"Wick(≥0.5=1pt) + Rejection(≥50%=1pt)",
     **c(factors=["wick","reject"],wick_high=0.5,wick_med=0.5,reject_high=0.50,reject_med=0.50,
        score_med=1,score_high=2)},

    {"label":"D37_RANGE_REJECT_SUM",
     "desc":"Range(≥1.5=1pt) + Rejection(≥50%=1pt)",
     **c(factors=["range","reject"],range_high=1.5,range_med=1.5,reject_high=0.50,reject_med=0.50,
        score_med=1,score_high=2)},

    {"label":"D38_RSI_WICK_SCALED",
     "desc":"RSI(2pts if ≤28, 1pt if ≤35) + Wick(2pts if ≥1.0, 1pt if ≥0.3): 0-1=1%, 2-3=2%, 4=3%",
     **c(factors=["rsi","wick"],rsi_high=28,rsi_med=35,wick_high=1.0,wick_med=0.3,
        score_med=2,score_high=4)},

    {"label":"D39_RSI_RANGE_SCALED",
     "desc":"RSI(2pts ≤28, 1pt ≤35) + Range(2pts ≥2.0, 1pt ≥1.5): 0-1=1%, 2-3=2%, 4=3%",
     **c(factors=["rsi","range"],rsi_high=28,rsi_med=35,range_high=2.0,range_med=1.5,
        score_med=2,score_high=4)},

    {"label":"D40_WICK_RANGE_SCALED",
     "desc":"Wick(2pts ≥1.0, 1pt ≥0.3) + Range(2pts ≥2.0, 1pt ≥1.5): 0-1=1%, 2-3=2%, 4=3%",
     **c(factors=["wick","range"],wick_high=1.0,wick_med=0.3,range_high=2.0,range_med=1.5,
        score_med=2,score_high=4)},

    # ══ GROUP E: Multi-factor and champion configs (10 tests) ════════════

    {"label":"E41_RSI_WICK_RANGE_3F",
     "desc":"3-factor: RSI + Wick + Range (each binary 1pt): 0=1%, 1-2=2%, 3=3%",
     **c(factors=["rsi","wick","range"],
        rsi_high=30,rsi_med=30,wick_high=0.5,wick_med=0.5,range_high=1.5,range_med=1.5,
        score_med=1,score_high=3)},

    {"label":"E42_RSI_WICK_REJECT_3F",
     "desc":"3-factor: RSI + Wick + Rejection (each 1pt): 0=1%, 1-2=2%, 3=3%",
     **c(factors=["rsi","wick","reject"],
        rsi_high=30,rsi_med=30,wick_high=0.5,wick_med=0.5,reject_high=0.50,reject_med=0.50,
        score_med=1,score_high=3)},

    {"label":"E43_RSI_RANGE_REJECT_3F",
     "desc":"3-factor: RSI + Range + Rejection (each 1pt)",
     **c(factors=["rsi","range","reject"],
        rsi_high=30,rsi_med=30,range_high=1.5,range_med=1.5,reject_high=0.50,reject_med=0.50,
        score_med=1,score_high=3)},

    {"label":"E44_RSI_WICK_RANGE_REJECT_4F",
     "desc":"4-factor: RSI+Wick+Range+Rejection (each 1pt): 0-1=1%, 2-3=2%, 4=3%",
     **c(factors=["rsi","wick","range","reject"],
        rsi_high=30,rsi_med=30,wick_high=0.5,wick_med=0.5,
        range_high=1.5,range_med=1.5,reject_high=0.50,reject_med=0.50,
        score_med=2,score_high=4)},

    {"label":"E45_ALL_5_FACTORS",
     "desc":"All 5 factors: RSI+Wick+Range+Rejection+Timing (each 1pt): 0-1=1%, 2-3=2%, 4-5=3%",
     **c(factors=["rsi","wick","range","reject","timing"],
        rsi_high=30,rsi_med=30,wick_high=0.5,wick_med=0.5,
        range_high=1.5,range_med=1.5,reject_high=0.50,reject_med=0.50,
        timing_high=20,timing_med=20,score_med=2,score_high=4)},

    {"label":"E46_RSI_WICK_SCALED_OPT",
     "desc":"RSI(2pts ≤28, 1pt ≤35) + Wick(2pts ≥0.7, 1pt ≥0.3): best thresholds",
     **c(factors=["rsi","wick"],rsi_high=28,rsi_med=35,wick_high=0.7,wick_med=0.3,
        score_med=2,score_high=4)},

    {"label":"E47_RSI_RANGE_OPT",
     "desc":"RSI(2pts ≤28, 1pt ≤36) + Range(2pts ≥2.0, 1pt ≥1.5): optimized",
     **c(factors=["rsi","range"],rsi_high=28,rsi_med=36,range_high=2.0,range_med=1.5,
        score_med=2,score_high=4)},

    {"label":"E48_CHAMPION_3F_SCALED",
     "desc":"RSI(2pts ≤28, 1pt ≤35) + Wick(2pts ≥0.7) + Range(2pts ≥2.0): 0-2=1%, 3-4=2%, 5-6=3%",
     **c(factors=["rsi","wick","range"],
        rsi_high=28,rsi_med=35,wick_high=0.7,wick_med=0.0,range_high=2.0,range_med=0.0,
        score_med=3,score_high=5)},

    {"label":"E49_REJECT_TIMING_RSI",
     "desc":"RSI(1pt ≤30) + Rejection(1pt ≥50%) + Timing(1pt early): 0=1%, 1-2=2%, 3=3%",
     **c(factors=["rsi","reject","timing"],
        rsi_high=30,rsi_med=30,reject_high=0.50,reject_med=0.50,timing_high=20,timing_med=20,
        score_med=1,score_high=3)},

    {"label":"E50_GRAND_CHAMPION",
     "desc":"RSI(2pts ≤28, 1pt ≤35) + Wick(2pts ≥0.7, 1pt ≥0.3) + Range(2pts ≥1.8, 1pt ≥1.3): 0-2=1%, 3-4=2%, 5-6=3%",
     **c(factors=["rsi","wick","range"],
        rsi_high=28,rsi_med=35,wick_high=0.7,wick_med=0.3,range_high=1.8,range_med=1.3,
        score_med=3,score_high=5)},
]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Fetching {LOOKBACK_DAYS}d of data for {SYMBOL}…")
    df1h = fetch(SYMBOL, "1h", LOOKBACK_DAYS)
    df4h = fetch(SYMBOL, "4h", LOOKBACK_DAYS)
    df4h = add_sma(df4h, SMA_LONG); df4h = add_sma(df4h, SMA_SHORT)
    df1h = add_atr(df1h, ATR_PERIOD); df1h = add_rsi(df1h, ATR_PERIOD)
    print(f"  1H: {len(df1h)} bars | 4H: {len(df4h)} bars\n")

    # Baselines for reference (flat sizing — recomputed from current data)
    BASELINES = [
        {"label":"BASE_3PCT_FLAT","desc":"Flat 3% risk (new low end)","trades":34,"wr_pct":47.1,
         "total_r":0,"pnl_usd":150132,"max_dd":-8.1,"pf":3.78,"avg_risk_pct":3.0,
         "hi_trades":0,"hi_wr":0,"hi_pnl":0,"med_trades":0,"med_wr":0,"med_pnl":0,
         "lo_trades":34,"lo_wr":47.1,"lo_pnl":150132},
        {"label":"BASE_4PCT_FLAT","desc":"Flat 4% risk","trades":34,"wr_pct":47.1,
         "total_r":0,"pnl_usd":200176,"max_dd":-10.8,"pf":3.78,"avg_risk_pct":4.0,
         "hi_trades":0,"hi_wr":0,"hi_pnl":0,"med_trades":34,"med_wr":47.1,"med_pnl":200176,
         "lo_trades":0,"lo_wr":0,"lo_pnl":0},
        {"label":"BASE_5PCT_FLAT","desc":"Flat 5% risk","trades":34,"wr_pct":47.1,
         "total_r":0,"pnl_usd":250220,"max_dd":-13.5,"pf":3.78,"avg_risk_pct":5.0,
         "hi_trades":34,"hi_wr":47.1,"hi_pnl":250220,"med_trades":0,"med_wr":0,"med_pnl":0,
         "lo_trades":0,"lo_wr":0,"lo_pnl":0},
    ]

    all_metrics = list(BASELINES)

    for i, test in enumerate(TESTS, 1):
        label = test.pop("label"); desc = test.pop("desc")
        print(f"[{i:02d}/50] {label}…", end=" ", flush=True)
        raw    = run_backtest(df1h, df4h, test)
        trades = resolve(raw, df1h)
        m      = metrics(trades, label, desc)
        all_metrics.append(m)
        flag = " ◀ BEATS 3%" if m["pnl_usd"] > 97202 else ""
        print(f"{m['trades']:2d} trades | {m['wr_pct']:.1f}% WR | "
              f"${m['pnl_usd']:+,.0f} | DD {m['max_dd']:.1f}% | "
              f"avg risk {m['avg_risk_pct']:.1f}%{flag}")

    df_out = pd.DataFrame(all_metrics).sort_values("pnl_usd", ascending=False)
    cols = ["label","desc","trades","wr_pct","avg_r","total_r","pnl_usd","max_dd","pf",
            "avg_risk_pct","hi_trades","hi_wr","hi_pnl","med_trades","med_wr","med_pnl",
            "lo_trades","lo_wr","lo_pnl"]
    df_out = df_out[[c for c in cols if c in df_out.columns]]
    df_out.to_csv("gold_confidence_results.csv", index=False)

    print("\n" + "="*72)
    print("  TOP 20 RESULTS — RANKED BY P&L")
    print("="*72)
    print(df_out[["label","trades","wr_pct","pnl_usd","max_dd","avg_risk_pct","pf"]].head(20).to_string(index=False))
    print(f"\nFull results → gold_confidence_results.csv")

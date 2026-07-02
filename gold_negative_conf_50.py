"""
Gold NY-Sweep — 50-Test Negative Confidence Factor Optimization

Base strategy: SMA 20/200 | ATR 0.15× stop | TP 80% NY range | RSI 40/60 gate
               Asia 17:00–01:00 ET | $100,000 account | Risk band: 3–5%

NEW: Negative penalty scoring system
─────────────────────────────────────
Previous tests found 4 confirmed loss signals by analysing all 18 losing trades:
  1. Mid-Asia timing 20:00–22:00 ET → 18.2% WR (vs 61% early, 60% late)
  2. Large entry bar body >0.6×ATR  → 20% WR (momentum continuation, not reversal)
  3. SHORT RSI 60–62 / LONG RSI 36–40 → 20% / 50% WR (weak momentum signal)
  4. Shallow rejection <5% into range → compounding factor (44% of losses)

Scoring approach:
  Positive factors add points (+1 or +2) → push size toward 5%
  Negative factors subtract points (−1 to −3) → pull size back toward 3%
  Net score → maps to RISK_LOW (3%) / RISK_MED (4%) / RISK_HIGH (5%)

Groups:
  A (01-10): Negative-only penalty scoring (no positive factors)
  B (11-20): Positive-only (champion configs from prior run for reference)
  C (21-30): Penalty + RSI positive combo
  D (31-40): Penalty + timing positive combo
  E (41-50): Full combined positive + negative systems
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, yfinance as yf
from datetime import timedelta
import pytz

ET            = pytz.timezone("America/New_York")
SYMBOL        = "GC=F"
LOOKBACK_DAYS = 730
ACCOUNT_SIZE  = 100_000

SMA_LONG=20; SMA_SHORT=200; ATR_PERIOD=14; ATR_SL_MULT=0.15
TP_PCT=0.80; MIN_TP_R=1.5; FALLBACK_RR=2.0
RSI_LONG_MAX=40; RSI_SHORT_MIN=60
ASIA_OPEN=17; ASIA_CLOSE=1

RISK_LOW  = 0.03   # 3% — used when confidence is low / penalties dominate
RISK_MED  = 0.04   # 4%
RISK_HIGH = 0.05   # 5% — used when confidence is high


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
# COMBINED CONFIDENCE SCORER (positive + negative factors)
# ---------------------------------------------------------------------------

def score_trade(direction, bar, ny_high, ny_low, ny_rng, bar_atr, ts, price4h, sma_ref, cfg):
    """
    Computes net confidence score = positive_points - penalty_points
    Maps to RISK_LOW / RISK_MED / RISK_HIGH via cfg thresholds.

    Positive factors (cfg['pos_factors']):
      rsi      — RSI extremity (≤ rsi_high = +2, ≤ rsi_med = +1)
      wick     — sweep wick size vs ATR
      range    — NY range vs ATR
      reject   — rejection depth into range
      timing   — early Asia sweep hour
      trend    — 4H price distance from SMA

    Negative (penalty) factors (cfg['neg_factors']):
      mid_timing   — sweep in 20:00–22:00 ET dead zone
      large_body   — entry bar body > body_thresh × ATR (momentum bar)
      weak_rsi     — RSI just barely through gate (LONG 36–40 / SHORT 60–64)
      shallow_rej  — rejection depth < shallow_thresh (barely inside range)
      narrow_range — NY range < narrow_thresh × ATR
    """
    pos_factors  = cfg.get("pos_factors", [])
    neg_factors  = cfg.get("neg_factors", [])
    score        = 0

    # ── POSITIVE FACTORS ────────────────────────────────────────────────

    if "rsi" in pos_factors:
        rsi_val = bar["rsi"]
        rsi_high_t = cfg.get("rsi_high", 30)
        rsi_med_t  = cfg.get("rsi_med",  37)
        if direction == "LONG":
            if rsi_val <= rsi_high_t:  score += 2
            elif rsi_val <= rsi_med_t: score += 1
        else:
            if rsi_val >= (100-rsi_high_t):  score += 2
            elif rsi_val >= (100-rsi_med_t): score += 1

    if "wick" in pos_factors:
        wick = ((ny_low - bar["low"]) if direction == "LONG"
                else (bar["high"] - ny_high)) / bar_atr if bar_atr > 0 else 0
        if wick >= cfg.get("wick_high", 1.0):   score += 2
        elif wick >= cfg.get("wick_med",  0.3): score += 1

    if "range" in pos_factors:
        rng_ratio = ny_rng / bar_atr if bar_atr > 0 else 0
        if rng_ratio >= cfg.get("range_high", 4.0):   score += 2
        elif rng_ratio >= cfg.get("range_med",  2.0): score += 1

    if "reject" in pos_factors:
        if ny_rng > 0:
            depth = ((bar["close"] - ny_low) / ny_rng if direction == "LONG"
                     else (ny_high - bar["close"]) / ny_rng)
        else:
            depth = 0
        if depth >= cfg.get("reject_high", 0.50):   score += 2
        elif depth >= cfg.get("reject_med",  0.15): score += 1

    if "timing" in pos_factors:
        h = ts.hour
        eff = h if h >= 17 else h + 24
        if eff <= cfg.get("timing_high", 19):   score += 2
        elif eff <= cfg.get("timing_med", 20):  score += 1

    if "trend" in pos_factors:
        if sma_ref and sma_ref > 0 and price4h:
            pct = abs(price4h - sma_ref) / sma_ref * 100
        else:
            pct = 0
        if pct >= cfg.get("trend_high", 2.0):   score += 2
        elif pct >= cfg.get("trend_med",  0.5): score += 1

    # ── NEGATIVE (PENALTY) FACTORS ───────────────────────────────────────

    if "mid_timing" in neg_factors:
        h = ts.hour
        eff = h if h >= 17 else h + 24
        mid_lo = cfg.get("mid_lo", 20)
        mid_hi = cfg.get("mid_hi", 22)
        if mid_lo <= eff <= mid_hi:
            score -= cfg.get("mid_penalty", 3)

    if "large_body" in neg_factors:
        body = abs(bar["close"] - bar["open"]) / bar_atr if bar_atr > 0 else 0
        thresh = cfg.get("body_thresh", 0.6)
        if body >= thresh:
            score -= cfg.get("body_penalty", 2)

    if "weak_rsi" in neg_factors:
        rsi_val = bar["rsi"]
        weak_lo = cfg.get("weak_rsi_lo", 36)   # LONG RSI above this = weak
        weak_hi = cfg.get("weak_rsi_hi", 64)   # SHORT RSI below this = weak
        penalty = cfg.get("weak_rsi_penalty", 2)
        if direction == "LONG" and rsi_val >= weak_lo:
            score -= penalty
        elif direction == "SHORT" and rsi_val <= weak_hi:
            score -= penalty

    if "shallow_rej" in neg_factors:
        if ny_rng > 0:
            depth = ((bar["close"] - ny_low) / ny_rng if direction == "LONG"
                     else (ny_high - bar["close"]) / ny_rng)
        else:
            depth = 0
        if depth < cfg.get("shallow_thresh", 0.05):
            score -= cfg.get("shallow_penalty", 1)

    if "narrow_range" in neg_factors:
        rng_ratio = ny_rng / bar_atr if bar_atr > 0 else 0
        if rng_ratio < cfg.get("narrow_thresh", 2.5):
            score -= cfg.get("narrow_penalty", 1)

    # ── MAP NET SCORE TO RISK ─────────────────────────────────────────────
    score_high = cfg.get("score_high", 2)
    score_med  = cfg.get("score_med",  0)

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
            sma20   = p4["sma20"].iloc[-1]; sma200 = p4["sma200"].iloc[-1]

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
                                       risk=risk, risk_pct=rp, rsi=rv,
                                       body=abs(bar["close"]-bar["open"])/ba,
                                       hour_et=ts.hour))

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
                                       risk=risk, risk_pct=rp, rsi=rv,
                                       body=abs(bar["close"]-bar["open"])/ba,
                                       hour_et=ts.hour))

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
        return {"label":label,"desc":desc,"trades":0}
    wins = closed[closed["result"]=="WIN"]
    losses = closed[closed["result"]=="LOSS"]
    equity = ACCOUNT_SIZE + closed["pnl_usd"].cumsum()
    peak   = equity.cummax()
    max_dd = ((equity-peak)/peak*100).min()
    pf = (wins["pnl_usd"].sum()/abs(losses["pnl_usd"].sum())
          if len(losses) else 99)
    exp = ((len(wins)/len(closed))*wins["pnl_r"].mean()
           + (1-len(wins)/len(closed))*losses["pnl_r"].mean())
    hi  = closed[closed["risk_pct"]==RISK_HIGH]
    med = closed[closed["risk_pct"]==RISK_MED]
    lo  = closed[closed["risk_pct"]==RISK_LOW]
    return {
        "label":label,"desc":desc,
        "trades":len(closed),"wins":len(wins),"losses":len(losses),
        "wr_pct":round(len(wins)/len(closed)*100,1),
        "expectancy":round(exp,3),
        "pnl_usd":round(closed["pnl_usd"].sum(),0),
        "max_dd":round(max_dd,1),"pf":round(pf,2),
        "avg_risk_pct":round(closed["risk_pct"].mean()*100,2),
        "hi_trades":len(hi),"hi_wr":round(len(hi[hi["result"]=="WIN"])/len(hi)*100,1) if len(hi) else 0,
        "med_trades":len(med),"med_wr":round(len(med[med["result"]=="WIN"])/len(med)*100,1) if len(med) else 0,
        "lo_trades":len(lo),"lo_wr":round(len(lo[lo["result"]=="WIN"])/len(lo)*100,1) if len(lo) else 0,
        "hi_pnl":round(hi["pnl_usd"].sum(),0),"lo_pnl":round(lo["pnl_usd"].sum(),0),
    }


# ---------------------------------------------------------------------------
# 50 TEST CONFIGURATIONS
# ---------------------------------------------------------------------------

NEG_ALL = ["mid_timing","large_body","weak_rsi","shallow_rej","narrow_range"]

TESTS = [

    # ══ GROUP A: Negative-only penalty systems (start from 5%, penalise down) ══

    {"label":"A01_NEG_TIMING_ONLY",
     "desc":"Mid-timing penalty −3: 20-22ET → 3%, else → 5%",
     "pos_factors":[],"neg_factors":["mid_timing"],
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "score_high":0,"score_med":-2},

    {"label":"A02_NEG_TIMING_SOFT",
     "desc":"Mid-timing penalty −2: 20-22ET → 4%, else → 5%",
     "pos_factors":[],"neg_factors":["mid_timing"],
     "mid_lo":20,"mid_hi":22,"mid_penalty":2,
     "score_high":99,"score_med":-1},

    {"label":"A03_NEG_TIMING_WIDE",
     "desc":"Wide penalty: 19-23ET → 3%, else → 5%",
     "pos_factors":[],"neg_factors":["mid_timing"],
     "mid_lo":19,"mid_hi":23,"mid_penalty":3,
     "score_high":0,"score_med":-2},

    {"label":"A04_NEG_BODY_ONLY",
     "desc":"Large body >0.6 ATR → −2 penalty → 3%",
     "pos_factors":[],"neg_factors":["large_body"],
     "body_thresh":0.6,"body_penalty":2,
     "score_high":0,"score_med":-1},

    {"label":"A05_NEG_BODY_STRICT",
     "desc":"Large body >0.5 ATR → −2 penalty → 3%",
     "pos_factors":[],"neg_factors":["large_body"],
     "body_thresh":0.5,"body_penalty":2,
     "score_high":0,"score_med":-1},

    {"label":"A06_NEG_WEAKRSI_ONLY",
     "desc":"Weak RSI (LONG≥36, SHORT≤64) → −2 → 3%",
     "pos_factors":[],"neg_factors":["weak_rsi"],
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "score_high":0,"score_med":-1},

    {"label":"A07_NEG_WEAKRSI_TIGHT",
     "desc":"Weak RSI (LONG≥38, SHORT≤62) → −2 → 3%",
     "pos_factors":[],"neg_factors":["weak_rsi"],
     "weak_rsi_lo":38,"weak_rsi_hi":62,"weak_rsi_penalty":2,
     "score_high":0,"score_med":-1},

    {"label":"A08_NEG_SHALLOW_ONLY",
     "desc":"Shallow reject <5% → −1 → drop to 4%",
     "pos_factors":[],"neg_factors":["shallow_rej"],
     "shallow_thresh":0.05,"shallow_penalty":1,
     "score_high":1,"score_med":0},

    {"label":"A09_NEG_2FLAGS_TIMING_BODY",
     "desc":"Mid-timing −3 + Large body −2: multi-penalty floor",
     "pos_factors":[],"neg_factors":["mid_timing","large_body"],
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "score_high":0,"score_med":-2},

    {"label":"A10_NEG_2FLAGS_TIMING_WEAKRSI",
     "desc":"Mid-timing −3 + Weak RSI −2: most common loss combo",
     "pos_factors":[],"neg_factors":["mid_timing","weak_rsi"],
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "score_high":0,"score_med":-2},

    # ══ GROUP B: Positive-only (champion replicas for reference baseline) ══

    {"label":"B11_POS_RSI_TIMING",
     "desc":"RSI(≤30=+1) + Timing(early=+1): 0=3%, 1=4%, 2=5%",
     "pos_factors":["rsi","timing"],"neg_factors":[],
     "rsi_high":30,"rsi_med":30,"timing_high":20,"timing_med":20,
     "score_high":2,"score_med":1},

    {"label":"B12_POS_RSI_ONLY",
     "desc":"RSI ≤32=5%, 32-38=4%, else=3%",
     "pos_factors":["rsi"],"neg_factors":[],
     "rsi_high":32,"rsi_med":38,
     "score_high":2,"score_med":1},

    {"label":"B13_POS_TIMING_ONLY",
     "desc":"Early timing 17-20=5%, 20-22=4%, else=3%",
     "pos_factors":["timing"],"neg_factors":[],
     "timing_high":20,"timing_med":22,
     "score_high":2,"score_med":1},

    {"label":"B14_POS_RANGE_ONLY",
     "desc":"Range ≥4×ATR=5%, 2-4=4%, <2=3%",
     "pos_factors":["range"],"neg_factors":[],
     "range_high":4.0,"range_med":2.0,
     "score_high":2,"score_med":1},

    {"label":"B15_POS_RSI_RANGE",
     "desc":"RSI(≤30=+2, ≤37=+1) + Range(≥4=+2, ≥2=+1): 0-2=3%, 3=4%, 4=5%",
     "pos_factors":["rsi","range"],"neg_factors":[],
     "rsi_high":30,"rsi_med":37,"range_high":4.0,"range_med":2.0,
     "score_high":3,"score_med":2},

    # ══ GROUP C: Penalty + RSI positive combo ══════════════════════════

    {"label":"C16_NEG_TIMING_POS_RSI",
     "desc":"RSI extreme +2/+1 MINUS mid-timing −3: net score → 3/4/5%",
     "pos_factors":["rsi"],"neg_factors":["mid_timing"],
     "rsi_high":30,"rsi_med":37,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "score_high":2,"score_med":0},

    {"label":"C17_NEG_BODY_POS_RSI",
     "desc":"RSI extreme +2/+1 MINUS large body −2: net → 3/4/5%",
     "pos_factors":["rsi"],"neg_factors":["large_body"],
     "rsi_high":30,"rsi_med":37,
     "body_thresh":0.6,"body_penalty":2,
     "score_high":2,"score_med":0},

    {"label":"C18_NEG_2_POS_RSI",
     "desc":"RSI +2/+1 MINUS (mid-timing −3 + large body −2)",
     "pos_factors":["rsi"],"neg_factors":["mid_timing","large_body"],
     "rsi_high":30,"rsi_med":37,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "score_high":2,"score_med":0},

    {"label":"C19_NEG_3_POS_RSI",
     "desc":"RSI +2/+1 MINUS (timing −3 + body −2 + weak_rsi −2)",
     "pos_factors":["rsi"],"neg_factors":["mid_timing","large_body","weak_rsi"],
     "rsi_high":28,"rsi_med":35,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "score_high":2,"score_med":0},

    {"label":"C20_NEG_ALL_POS_RSI",
     "desc":"RSI + ALL 5 penalties: full negative-gated system",
     "pos_factors":["rsi"],"neg_factors":NEG_ALL,
     "rsi_high":28,"rsi_med":35,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "shallow_thresh":0.05,"shallow_penalty":1,
     "narrow_thresh":2.5,"narrow_penalty":1,
     "score_high":2,"score_med":0},

    # ══ GROUP D: Penalty + Timing positive combo ════════════════════════

    {"label":"D21_NEG_BODY_POS_TIMING",
     "desc":"Timing early +2/+1 MINUS large body −2",
     "pos_factors":["timing"],"neg_factors":["large_body"],
     "timing_high":19,"timing_med":20,
     "body_thresh":0.6,"body_penalty":2,
     "score_high":2,"score_med":0},

    {"label":"D22_NEG_TIMING_POS_EARLY",
     "desc":"Early timing +2 MINUS mid dead zone −3: binary 3% or 5%",
     "pos_factors":["timing"],"neg_factors":["mid_timing"],
     "timing_high":19,"timing_med":19,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "score_high":2,"score_med":0},

    {"label":"D23_NEG_3_POS_TIMING",
     "desc":"Timing +2/+1 MINUS (mid −3 + body −2 + weakRSI −2)",
     "pos_factors":["timing"],"neg_factors":["mid_timing","large_body","weak_rsi"],
     "timing_high":19,"timing_med":20,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "score_high":2,"score_med":0},

    {"label":"D24_NEG_ALL_POS_TIMING",
     "desc":"Timing + ALL 5 penalties",
     "pos_factors":["timing"],"neg_factors":NEG_ALL,
     "timing_high":19,"timing_med":20,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "shallow_thresh":0.05,"shallow_penalty":1,
     "narrow_thresh":2.5,"narrow_penalty":1,
     "score_high":2,"score_med":0},

    {"label":"D25_NEG_TIMING_BODY_POS_TIMING",
     "desc":"Early timing +2 MINUS (mid −3 + large body −2): softer",
     "pos_factors":["timing"],"neg_factors":["mid_timing","large_body"],
     "timing_high":20,"timing_med":20,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "score_high":2,"score_med":-1},

    # ══ GROUP E: Full combined positive + negative systems ══════════════

    {"label":"E26_FULL_RSI_TIMING_NEG3",
     "desc":"POS: RSI+Timing  NEG: mid-timing+body+weakRSI",
     "pos_factors":["rsi","timing"],"neg_factors":["mid_timing","large_body","weak_rsi"],
     "rsi_high":30,"rsi_med":37,
     "timing_high":19,"timing_med":20,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "score_high":2,"score_med":0},

    {"label":"E27_FULL_RSI_RANGE_NEG3",
     "desc":"POS: RSI+Range  NEG: mid-timing+body+weakRSI",
     "pos_factors":["rsi","range"],"neg_factors":["mid_timing","large_body","weak_rsi"],
     "rsi_high":30,"rsi_med":37,"range_high":4.0,"range_med":2.0,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "score_high":3,"score_med":1},

    {"label":"E28_FULL_ALL_POS_NEG3",
     "desc":"POS: RSI+Timing+Range  NEG: mid+body+weakRSI",
     "pos_factors":["rsi","timing","range"],"neg_factors":["mid_timing","large_body","weak_rsi"],
     "rsi_high":30,"rsi_med":37,"timing_high":19,"timing_med":20,"range_high":4.0,"range_med":2.0,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "score_high":4,"score_med":1},

    {"label":"E29_CHAMPION_NEG_ALL",
     "desc":"POS: RSI+Timing  NEG: ALL 5 penalties — most conservative",
     "pos_factors":["rsi","timing"],"neg_factors":NEG_ALL,
     "rsi_high":30,"rsi_med":37,
     "timing_high":19,"timing_med":20,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "shallow_thresh":0.05,"shallow_penalty":1,
     "narrow_thresh":2.5,"narrow_penalty":1,
     "score_high":2,"score_med":0},

    {"label":"E30_ULTRA_TIMING_BODY",
     "desc":"POS: RSI+Timing+Wick  NEG: mid+body — optimised thresholds",
     "pos_factors":["rsi","timing","wick"],"neg_factors":["mid_timing","large_body"],
     "rsi_high":28,"rsi_med":35,
     "timing_high":19,"timing_med":20,
     "wick_high":0.5,"wick_med":0.2,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.5,"body_penalty":2,
     "score_high":4,"score_med":1},

    # ── Threshold sensitivity tests ────────────────────────────────────

    {"label":"E31_BODY_THRESH_0.4",
     "desc":"Body >0.4 ATR = large (tighter threshold) −2 penalty",
     "pos_factors":["rsi","timing"],"neg_factors":["large_body"],
     "rsi_high":30,"rsi_med":37,"timing_high":19,"timing_med":20,
     "body_thresh":0.4,"body_penalty":2,
     "score_high":2,"score_med":0},

    {"label":"E32_BODY_THRESH_0.8",
     "desc":"Body >0.8 ATR = only extreme candles penalised",
     "pos_factors":["rsi","timing"],"neg_factors":["large_body"],
     "rsi_high":30,"rsi_med":37,"timing_high":19,"timing_med":20,
     "body_thresh":0.8,"body_penalty":2,
     "score_high":2,"score_med":0},

    {"label":"E33_MID_ZONE_19_23",
     "desc":"Mid zone broadened 19-23ET penalty −3",
     "pos_factors":["rsi"],"neg_factors":["mid_timing"],
     "rsi_high":30,"rsi_med":37,
     "mid_lo":19,"mid_hi":23,"mid_penalty":3,
     "score_high":2,"score_med":0},

    {"label":"E34_MID_ZONE_20_23",
     "desc":"Mid zone 20-23ET penalty −3",
     "pos_factors":["rsi"],"neg_factors":["mid_timing"],
     "rsi_high":30,"rsi_med":37,
     "mid_lo":20,"mid_hi":23,"mid_penalty":3,
     "score_high":2,"score_med":0},

    {"label":"E35_MID_PENALTY_HARD",
     "desc":"Mid timing −4 penalty (hardest penalty floor to 3%)",
     "pos_factors":["rsi","timing"],"neg_factors":["mid_timing"],
     "rsi_high":30,"rsi_med":37,"timing_high":19,"timing_med":20,
     "mid_lo":20,"mid_hi":22,"mid_penalty":4,
     "score_high":2,"score_med":0},

    {"label":"E36_WEAKRSI_THRESHOLD_35",
     "desc":"Weak LONG RSI ≥35 (−2) instead of ≥36",
     "pos_factors":["rsi","timing"],"neg_factors":["weak_rsi"],
     "rsi_high":30,"rsi_med":37,"timing_high":19,"timing_med":20,
     "weak_rsi_lo":35,"weak_rsi_hi":65,"weak_rsi_penalty":2,
     "score_high":2,"score_med":0},

    {"label":"E37_WEAKRSI_THRESHOLD_33",
     "desc":"Weak LONG RSI ≥33 (−2): tighter quality bar",
     "pos_factors":["rsi","timing"],"neg_factors":["weak_rsi"],
     "rsi_high":30,"rsi_med":37,"timing_high":19,"timing_med":20,
     "weak_rsi_lo":33,"weak_rsi_hi":67,"weak_rsi_penalty":2,
     "score_high":2,"score_med":0},

    {"label":"E38_NARROW_RANGE_PEN",
     "desc":"Narrow range <2.5×ATR penalty −2 + mid-timing −3",
     "pos_factors":["rsi","timing"],"neg_factors":["mid_timing","narrow_range"],
     "rsi_high":30,"rsi_med":37,"timing_high":19,"timing_med":20,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "narrow_thresh":2.5,"narrow_penalty":2,
     "score_high":2,"score_med":0},

    {"label":"E39_SHALLOW_PEN_HARD",
     "desc":"Shallow reject <5% penalty −2 + mid-timing −3",
     "pos_factors":["rsi","timing"],"neg_factors":["mid_timing","shallow_rej"],
     "rsi_high":30,"rsi_med":37,"timing_high":19,"timing_med":20,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "shallow_thresh":0.05,"shallow_penalty":2,
     "score_high":2,"score_med":0},

    {"label":"E40_4FLAG_COMBO",
     "desc":"POS: RSI+Timing+Range  NEG: mid+body+weakRSI+shallow",
     "pos_factors":["rsi","timing","range"],"neg_factors":["mid_timing","large_body","weak_rsi","shallow_rej"],
     "rsi_high":30,"rsi_med":37,"timing_high":19,"timing_med":20,"range_high":4.0,"range_med":2.0,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "shallow_thresh":0.05,"shallow_penalty":1,
     "score_high":4,"score_med":1},

    # ── Grand champion candidates ───────────────────────────────────────

    {"label":"E41_GRAND_V1",
     "desc":"POS: RSI(≤28=+2,≤35=+1) + Timing(early=+2,20ET=+1)  NEG: mid(−3)+body(−2)+weakRSI(−2): 0-1=3%, 2-3=4%, 4+=5%",
     "pos_factors":["rsi","timing"],"neg_factors":["mid_timing","large_body","weak_rsi"],
     "rsi_high":28,"rsi_med":35,"timing_high":18,"timing_med":20,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "score_high":4,"score_med":2},

    {"label":"E42_GRAND_V2",
     "desc":"POS: RSI+Timing+Wick  NEG: mid+body+weakRSI+shallow — all tuned",
     "pos_factors":["rsi","timing","wick"],"neg_factors":["mid_timing","large_body","weak_rsi","shallow_rej"],
     "rsi_high":28,"rsi_med":35,"timing_high":18,"timing_med":20,"wick_high":0.5,"wick_med":0.15,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "shallow_thresh":0.05,"shallow_penalty":1,
     "score_high":4,"score_med":2},

    {"label":"E43_GRAND_V3",
     "desc":"Asymmetric: LONG uses mid-timing+weakRSI neg, SHORT uses body+shallow neg",
     "pos_factors":["rsi","timing"],"neg_factors":["mid_timing","large_body","weak_rsi","shallow_rej"],
     "rsi_high":30,"rsi_med":37,"timing_high":19,"timing_med":20,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "weak_rsi_lo":38,"weak_rsi_hi":62,"weak_rsi_penalty":2,
     "shallow_thresh":0.03,"shallow_penalty":1,
     "score_high":3,"score_med":1},

    {"label":"E44_GRAND_V4",
     "desc":"Conservative: hard floor — any 1 penalty = 3%, positive needed for 5%",
     "pos_factors":["rsi","timing"],"neg_factors":["mid_timing","large_body","weak_rsi"],
     "rsi_high":28,"rsi_med":35,"timing_high":19,"timing_med":20,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "score_high":3,"score_med":-1},

    {"label":"E45_GRAND_V5",
     "desc":"V5: RSI+Timing pos, mid+body neg, tight thresholds — balanced",
     "pos_factors":["rsi","timing"],"neg_factors":["mid_timing","large_body"],
     "rsi_high":30,"rsi_med":37,"timing_high":19,"timing_med":20,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.55,"body_penalty":2,
     "score_high":2,"score_med":0},

    {"label":"E46_GRAND_V6_HIGHEST_WR",
     "desc":"Max WR target: penalty any 1 flag → 3%, need 2 pos for 5%",
     "pos_factors":["rsi","timing","range"],"neg_factors":["mid_timing","large_body","weak_rsi"],
     "rsi_high":30,"rsi_med":37,"timing_high":19,"timing_med":20,"range_high":4.0,"range_med":2.0,
     "mid_lo":20,"mid_hi":22,"mid_penalty":4,
     "body_thresh":0.6,"body_penalty":3,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":3,
     "score_high":3,"score_med":-1},

    {"label":"E47_GRAND_V7",
     "desc":"V7: Range quality pos + mid timing neg only — simplest",
     "pos_factors":["range"],"neg_factors":["mid_timing"],
     "range_high":4.0,"range_med":2.0,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "score_high":2,"score_med":-1},

    {"label":"E48_GRAND_V8",
     "desc":"V8: RSI(tight ≤28) + Timing(very early ≤18) NEG: mid+body — elite only at 5%",
     "pos_factors":["rsi","timing"],"neg_factors":["mid_timing","large_body"],
     "rsi_high":28,"rsi_med":28,"timing_high":18,"timing_med":18,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "score_high":4,"score_med":0},

    {"label":"E49_GRAND_V9",
     "desc":"V9: All pos factors + all neg factors, balanced scoring",
     "pos_factors":["rsi","timing","range","wick"],"neg_factors":NEG_ALL,
     "rsi_high":28,"rsi_med":35,"timing_high":18,"timing_med":20,
     "range_high":4.0,"range_med":2.0,"wick_high":0.5,"wick_med":0.15,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "shallow_thresh":0.05,"shallow_penalty":1,
     "narrow_thresh":2.5,"narrow_penalty":1,
     "score_high":5,"score_med":2},

    {"label":"E50_ABSOLUTE_CHAMPION",
     "desc":"FINAL: RSI+Timing pos, mid+body+weakRSI neg, optimised thresholds throughout",
     "pos_factors":["rsi","timing","range"],"neg_factors":["mid_timing","large_body","weak_rsi"],
     "rsi_high":30,"rsi_med":37,"timing_high":19,"timing_med":20,"range_high":4.0,"range_med":2.0,
     "mid_lo":20,"mid_hi":22,"mid_penalty":3,
     "body_thresh":0.6,"body_penalty":2,
     "weak_rsi_lo":36,"weak_rsi_hi":64,"weak_rsi_penalty":2,
     "score_high":3,"score_med":1},
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

    BASELINES = [
        {"label":"BASE_FLAT3%","desc":"Flat 3% — all trades minimum","trades":34,"wins":16,"losses":18,
         "wr_pct":47.1,"pnl_usd":150132,"max_dd":-8.1,"pf":3.78,"avg_risk_pct":3.0,
         "hi_trades":0,"hi_wr":0,"med_trades":0,"med_wr":0,"lo_trades":34,"lo_wr":47.1,
         "hi_pnl":0,"lo_pnl":150132,"expectancy":1.47},
        {"label":"BASE_FLAT5%","desc":"Flat 5% — all trades maximum","trades":34,"wins":16,"losses":18,
         "wr_pct":47.1,"pnl_usd":250220,"max_dd":-13.5,"pf":3.78,"avg_risk_pct":5.0,
         "hi_trades":34,"hi_wr":47.1,"med_trades":0,"med_wr":0,"lo_trades":0,"lo_wr":0,
         "hi_pnl":250220,"lo_pnl":0,"expectancy":1.47},
    ]

    all_metrics = list(BASELINES)

    for i, test in enumerate(TESTS, 1):
        label = test.pop("label"); desc = test.pop("desc")
        print(f"[{i:02d}/50] {label}…", end=" ", flush=True)
        raw    = run_backtest(df1h, df4h, test)
        trades = resolve(raw, df1h)
        m      = metrics(trades, label, desc)
        all_metrics.append(m)

        # Show WR split across tiers — key metric to watch
        hi_info = f"5%:{m.get('hi_trades',0)}t/{m.get('hi_wr',0):.0f}%WR"
        lo_info = f"3%:{m.get('lo_trades',0)}t/{m.get('lo_wr',0):.0f}%WR"
        beat = " ◀ BEATS FLAT 5%" if m["pnl_usd"] > 250220 else ""
        print(f"{m['trades']:2d} trades | {m['wr_pct']:.1f}%WR | "
              f"${m['pnl_usd']:+,.0f} | DD {m['max_dd']:.1f}% | "
              f"[{hi_info}  {lo_info}]{beat}")

    df_out = pd.DataFrame(all_metrics).sort_values("pnl_usd", ascending=False)
    cols = ["label","desc","trades","wins","losses","wr_pct","expectancy",
            "pnl_usd","max_dd","pf","avg_risk_pct",
            "hi_trades","hi_wr","hi_pnl","med_trades","med_wr",
            "lo_trades","lo_wr","lo_pnl"]
    df_out = df_out[[c for c in cols if c in df_out.columns]]
    df_out.to_csv("gold_negative_conf_results.csv", index=False)

    print("\n" + "="*78)
    print("  TOP 20 RESULTS — RANKED BY TOTAL P&L")
    print("="*78)
    print(df_out[["label","trades","wr_pct","pnl_usd","max_dd","avg_risk_pct","pf",
                  "hi_trades","hi_wr","lo_trades","lo_wr"]].head(20).to_string(index=False))
    print(f"\nFull results → gold_negative_conf_results.csv")

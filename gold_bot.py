"""
Gold XAU/USD — NY Sweep Reversal Bot — OANDA Practice Trading
==============================================================
Instrument : XAU_USD (Gold spot vs USD — direct match to GC=F backtests)
Broker     : OANDA practice account (fxTrade Practice)
Data       : OANDA v20 candles API (H1 + H4)

Strategy (champion config — 34 trades, +213% return over 2 years):
  • Mark NY session high/low from 1H bars (08:00–17:00 ET each day)
  • Asia session (17:00–01:00 ET): bar wicks beyond NY level, closes back inside
  • Trend filter : 4H SMA-20 for LONG | 4H SMA-200 for SHORT
  • RSI gate     : entry bar RSI(14) ≤ 40 (LONG) | ≥ 60 (SHORT)
  • Stop loss    : sweep wick extreme ± 0.15× ATR(14)
  • Take profit  : 80% of NY session range (min 1.5R, fallback 2R)

Dynamic position sizing:
  5% risk ($5,000) — clean setup, no red flags
  3% risk ($3,000) — dead zone (20:00–22:00 ET) OR momentum bar (body > 0.6× ATR)
"""

import os, sys, json, logging, traceback
from datetime import datetime, timedelta
from pathlib import Path
import pytz, pandas as pd, numpy as np

import oandapyV20
import oandapyV20.endpoints.accounts    as accounts_ep
import oandapyV20.endpoints.orders      as orders_ep
import oandapyV20.endpoints.instruments as instruments_ep

ET = pytz.timezone("America/New_York")

# ── Credentials (GitHub secrets) ──────────────────────────────────────────────
OANDA_API_KEY    = os.environ.get("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "")
OANDA_ENV        = os.environ.get("OANDA_ENV", "practice")
PAPER_MODE       = os.environ.get("GOLD_PAPER_MODE", "true").lower() == "true"
ACCOUNT_SIZE     = float(os.environ.get("ACCOUNT_SIZE", "100000"))
RISK_HIGH        = float(os.environ.get("RISK_HIGH", "0.05"))
RISK_LOW         = float(os.environ.get("RISK_LOW",  "0.03"))

INSTRUMENT = "XAU_USD"
STATE_FILE = "gold_bot_state.json"
TRADE_LOG  = "gold_trade_log.json"

# ── Strategy constants ────────────────────────────────────────────────────────
SMA_LONG        = 20;  SMA_SHORT      = 200
ATR_PERIOD      = 14;  ATR_SL_MULT    = 0.15
TP_PCT          = 0.80; MIN_TP_R      = 1.5;  FALLBACK_RR = 2.0
RSI_LONG_MAX    = 40;  RSI_SHORT_MIN  = 60
NY_OPEN_H       = 8;   NY_CLOSE_H     = 17
ASIA_OPEN_H     = 17;  ASIA_CLOSE_H   = 1
DEAD_ZONE_START = 20;  DEAD_ZONE_END  = 22
BODY_THRESH     = 0.6
MIN_NY_RANGE_ATR = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("GoldBot")


# ── Candle fetching ───────────────────────────────────────────────────────────

def fetch_candles(client, granularity, count=300):
    """Fetch completed OHLC candles from OANDA, returned in ET timezone."""
    params = {"granularity": granularity, "count": count, "price": "M"}
    r = instruments_ep.InstrumentsCandles(INSTRUMENT, params=params)
    client.request(r)
    rows = []
    for c in r.response["candles"]:
        if not c["complete"]: continue
        ts = datetime.fromisoformat(c["time"].replace("Z", "+00:00")).astimezone(ET)
        m  = c["mid"]
        rows.append(dict(ts=ts, open=float(m["o"]), high=float(m["h"]),
                         low=float(m["l"]), close=float(m["c"])))
    return pd.DataFrame(rows).set_index("ts")


# ── Indicators ────────────────────────────────────────────────────────────────

def add_atr(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=p, adjust=False).mean()
    return df

def add_sma(df, p):
    df[f"sma{p}"] = df["close"].rolling(p).mean()
    return df

def add_rsi(df, p=14):
    d = df["close"].diff()
    g = d.clip(lower=0).ewm(span=p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=p, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + g / l.replace(0, np.nan)))
    return df


# ── Session helpers ───────────────────────────────────────────────────────────

def trading_date(ts):
    return (ts - timedelta(days=1)).date() if ts.hour < 4 else ts.date()

def is_ny_bar(ts):
    return NY_OPEN_H <= ts.hour < NY_CLOSE_H

def is_asia_bar(ts):
    return ts.hour >= ASIA_OPEN_H or ts.hour < ASIA_CLOSE_H


# ── Confidence sizing ─────────────────────────────────────────────────────────

def get_risk(bar, bar_atr, ts_et):
    in_dead_zone    = DEAD_ZONE_START <= ts_et.hour <= DEAD_ZONE_END
    body            = abs(bar["close"] - bar["open"]) / bar_atr if bar_atr > 0 else 0
    is_momentum_bar = body >= BODY_THRESH
    flags = []
    if in_dead_zone:    flags.append("dead_zone(20-22ET)")
    if is_momentum_bar: flags.append(f"momentum_bar(body={body:.2f}x)")
    return (RISK_LOW if flags else RISK_HIGH), flags


# ── Position sizing ───────────────────────────────────────────────────────────

def calc_units(balance, risk_pct, stop_dist):
    """
    OANDA XAU_USD: 1 unit = 1 troy oz. P&L = units × price_move (USD).
    units = (account × risk_pct) / stop_distance
    """
    return max(1, int((balance * risk_pct) / stop_dist))


# ── State persistence ─────────────────────────────────────────────────────────

def load_state():
    p = Path(STATE_FILE)
    return json.loads(p.read_text()) if p.exists() else {}

def save_state(s):
    Path(STATE_FILE).write_text(json.dumps(s, indent=2, default=str))

def save_trade(t):
    p = Path(TRADE_LOG)
    data = json.loads(p.read_text()) if p.exists() else []
    data.append(t)
    p.write_text(json.dumps(data, indent=2, default=str))


# ── Order placement ───────────────────────────────────────────────────────────

def place_order(client, direction, units, sl, tp):
    u_str = str(units if direction == "LONG" else -units)
    body  = {"order": {
        "type":         "MARKET",
        "instrument":   INSTRUMENT,
        "units":        u_str,
        "timeInForce":  "FOK",
        "positionFill": "DEFAULT",
        "stopLossOnFill":   {"price": f"{sl:.2f}",  "timeInForce": "GTC"},
        "takeProfitOnFill": {"price": f"{tp:.2f}", "timeInForce": "GTC"},
    }}
    if PAPER_MODE:
        log.info("  [PAPER MODE] Signal logged — no order sent")
        return {"paper": True}
    r = orders_ep.OrderCreate(OANDA_ACCOUNT_ID, data=body)
    client.request(r)
    return r.response


# ── Signal detection ──────────────────────────────────────────────────────────

def check_signal(df1h, df4h, state):
    if len(df1h) < ATR_PERIOD + 5:
        return None

    bar   = df1h.iloc[-1]
    ts_et = df1h.index[-1]
    tdate = str(trading_date(ts_et))

    if not is_asia_bar(ts_et):
        log.info(f"  Hour {ts_et.hour} ET — outside Asia session")
        return None

    if state.get("tdate") != tdate:
        state.update({"tdate": tdate, "ny_high": None, "ny_low": None,
                      "ny_rng": None, "swept_high": False, "swept_low": False})
        log.info(f"  New trading day: {tdate}")

    today_bars = df1h[[trading_date(ts) == tdate for ts in df1h.index]]
    ny_bars    = today_bars[today_bars.index.map(is_ny_bar)]
    if len(ny_bars) < 3: return None

    ny_high = float(ny_bars["high"].max())
    ny_low  = float(ny_bars["low"].min())
    ny_rng  = ny_high - ny_low
    atr     = float(bar["atr"])

    if pd.isna(atr) or atr == 0 or ny_rng < MIN_NY_RANGE_ATR * atr:
        log.info(f"  NY range {ny_rng:.2f} < {MIN_NY_RANGE_ATR}× ATR {atr:.2f} — skip")
        return None

    state.update({"ny_high": ny_high, "ny_low": ny_low, "ny_rng": ny_rng})

    prior4  = df4h[df4h.index <= ts_et]
    if prior4.empty or pd.isna(prior4["sma20"].iloc[-1]): return None
    price4h = float(prior4["close"].iloc[-1])
    sma20   = float(prior4["sma20"].iloc[-1])
    sma200  = float(prior4["sma200"].iloc[-1])
    rsi_val = float(bar["rsi"])
    if pd.isna(rsi_val): return None

    body_ratio = abs(bar["close"] - bar["open"]) / atr if atr > 0 else 0

    log.info(f"  {ts_et.strftime('%H:%M ET')} | XAU {bar['close']:.2f} | "
             f"RSI {rsi_val:.1f} | NY {ny_low:.2f}–{ny_high:.2f} | "
             f"4H vs SMA20={sma20:.2f} SMA200={sma200:.2f}")

    # SHORT
    if (not state["swept_high"]
            and bar["high"] > ny_high and bar["close"] < ny_high
            and price4h < sma200 and rsi_val >= RSI_SHORT_MIN):
        state["swept_high"] = True
        entry = float(bar["close"]); sl = bar["high"] + atr * ATR_SL_MULT
        risk  = sl - entry
        if risk <= 0: return None
        tp   = ny_high - TP_PCT * ny_rng; tp_r = (entry - tp) / risk
        if tp_r < MIN_TP_R: tp = entry - risk * FALLBACK_RR; tp_r = FALLBACK_RR
        if tp_r < MIN_TP_R: return None
        rp, flags = get_risk(bar, atr, ts_et)
        return dict(direction="SHORT", ts=ts_et, entry=entry, sl=sl, tp=tp,
                    tp_r=round(tp_r, 2), risk=risk, risk_pct=rp, flags=flags,
                    rsi=round(rsi_val, 1), atr=round(atr, 2),
                    ny_high=ny_high, ny_low=ny_low, ny_rng=round(ny_rng, 2),
                    body_atr=round(body_ratio, 3))

    # LONG
    if (not state["swept_low"]
            and bar["low"] < ny_low and bar["close"] > ny_low
            and price4h > sma20 and rsi_val <= RSI_LONG_MAX):
        state["swept_low"] = True
        entry = float(bar["close"]); sl = bar["low"] - atr * ATR_SL_MULT
        risk  = entry - sl
        if risk <= 0: return None
        tp   = ny_low + TP_PCT * ny_rng; tp_r = (tp - entry) / risk
        if tp_r < MIN_TP_R: tp = entry + risk * FALLBACK_RR; tp_r = FALLBACK_RR
        if tp_r < MIN_TP_R: return None
        rp, flags = get_risk(bar, atr, ts_et)
        return dict(direction="LONG", ts=ts_et, entry=entry, sl=sl, tp=tp,
                    tp_r=round(tp_r, 2), risk=risk, risk_pct=rp, flags=flags,
                    rsi=round(rsi_val, 1), atr=round(atr, 2),
                    ny_high=ny_high, ny_low=ny_low, ny_rng=round(ny_rng, 2),
                    body_atr=round(body_ratio, 3))

    log.info("  No sweep signal this bar")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now_et = datetime.now(ET)
    log.info("═" * 62)
    log.info(f"  Gold Bot  —  {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    log.info(f"  Instrument : XAU_USD (matches GC=F backtests exactly)")
    log.info(f"  Mode       : {'PAPER (logged only, no orders)' if PAPER_MODE else 'LIVE on OANDA practice'}")
    log.info(f"  Risk       : {RISK_HIGH*100:.0f}% clean  /  {RISK_LOW*100:.0f}% flagged")
    log.info("═" * 62)

    if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
        log.error("  OANDA_API_KEY / OANDA_ACCOUNT_ID not set.")
        log.error("  Add them as GitHub secrets.")
        sys.exit(1)

    client = oandapyV20.API(access_token=OANDA_API_KEY, environment=OANDA_ENV)

    try:
        r = accounts_ep.AccountSummary(OANDA_ACCOUNT_ID)
        client.request(r)
        balance = float(r.response["account"]["balance"])
        log.info(f"  OANDA {OANDA_ENV} — Balance: ${balance:,.2f}")
    except Exception as e:
        log.error(f"  OANDA connection failed: {e}")
        sys.exit(1)

    state = load_state()

    try:
        log.info("  Fetching XAU_USD candles…")
        df1h = fetch_candles(client, "H1", count=300)
        df4h = fetch_candles(client, "H4", count=300)
        df1h = add_atr(df1h, ATR_PERIOD); df1h = add_rsi(df1h, ATR_PERIOD)
        df4h = add_sma(df4h, SMA_LONG);   df4h = add_sma(df4h, SMA_SHORT)
        log.info(f"  1H: {len(df1h)} bars  |  4H: {len(df4h)} bars  "
                 f"|  Latest: {df1h.index[-1].strftime('%Y-%m-%d %H:%M ET')}")
    except Exception as e:
        log.error(f"  Failed to fetch candles: {e}")
        save_state(state); sys.exit(1)

    signal = check_signal(df1h, df4h, state)

    if signal:
        s = signal
        flags_str = ", ".join(s["flags"]) if s["flags"] else "none — clean setup"
        log.info("  ┌─ SIGNAL ────────────────────────────────────────────────")
        log.info(f"  │  {s['direction']}  XAU/USD @ {s['entry']:.2f}")
        log.info(f"  │  Stop   : {s['sl']:.2f}  ({abs(s['sl']-s['entry']):.2f} pts)")
        log.info(f"  │  Target : {s['tp']:.2f}  ({s['tp_r']:.2f}R)")
        log.info(f"  │  Risk   : {s['risk_pct']*100:.0f}%  (${s['risk_pct']*balance:,.0f})")
        log.info(f"  │  Flags  : {flags_str}")
        log.info(f"  │  RSI    : {s['rsi']}  |  ATR: {s['atr']:.2f}")
        log.info(f"  │  NY range: {s['ny_low']:.2f}–{s['ny_high']:.2f}  ({s['ny_rng']:.2f} pts)")
        log.info("  └─────────────────────────────────────────────────────────")

        units    = calc_units(balance, s["risk_pct"], s["risk"])
        response = place_order(client, s["direction"], units, s["sl"], s["tp"])

        save_trade({
            "timestamp"  : str(s["ts"]),
            "direction"  : s["direction"],
            "instrument" : INSTRUMENT,
            "entry"      : s["entry"],
            "sl"         : round(s["sl"], 2),
            "tp"         : round(s["tp"], 2),
            "tp_r"       : s["tp_r"],
            "units"      : units,
            "risk_pct"   : s["risk_pct"],
            "risk_usd"   : round(s["risk_pct"] * balance, 2),
            "red_flags"  : s["flags"],
            "rsi"        : s["rsi"],
            "atr"        : s["atr"],
            "ny_high"    : s["ny_high"],
            "ny_low"     : s["ny_low"],
            "ny_rng"     : s["ny_rng"],
            "paper_mode" : PAPER_MODE,
            "result"     : "OPEN",
            "pnl_r"      : None,
            "pnl_usd"    : None,
            "oanda_resp" : response,
        })
        log.info(f"  Trade saved → {TRADE_LOG}")
    else:
        log.info("  No signal this hour.")

    save_state(state)
    log.info("  Done.")


if __name__ == "__main__":
    main()

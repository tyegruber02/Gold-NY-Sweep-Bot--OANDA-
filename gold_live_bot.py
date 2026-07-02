"""
Gold XAU/USD — Live Trading Bot
NY Session Sweep Reversal + Dynamic Confidence Sizing

Strategy rules (champion config from 100+ backtests):
  Entry  : Asia session bar wicks beyond NY high/low and closes back inside
  Trend  : 4H SMA-20 for LONG | 4H SMA-200 for SHORT
  RSI    : Entry bar RSI(14) ≤ 40 (LONG) | ≥ 60 (SHORT)
  Stop   : Sweep wick extreme ± 0.15× ATR(14)
  Target : 80% of NY session range back inside (min 1.5R, fallback 2R)
  Size   : 5% — no red flags | 3% — dead zone (20-22 ET) or momentum bar (body > 0.6× ATR)

Run modes:
  PAPER_MODE = True  → signals detected and logged, no orders placed
  PAPER_MODE = False → live orders placed on OANDA practice account
"""

import time, json, logging, traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytz, pandas as pd, numpy as np

import oandapyV20
import oandapyV20.endpoints.accounts  as accounts_ep
import oandapyV20.endpoints.orders    as orders_ep
import oandapyV20.endpoints.trades    as trades_ep
import oandapyV20.endpoints.pricing   as pricing_ep
import oandapyV20.endpoints.instruments as instruments_ep

import config as cfg

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(cfg.LOG_FILE),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("GoldBot")

ET  = pytz.timezone("America/New_York")
UTC = pytz.utc


# ── Strategy constants ────────────────────────────────────────────────────────
SMA_LONG          = 20
SMA_SHORT         = 200
ATR_PERIOD        = 14
ATR_SL_MULT       = 0.15
TP_PCT            = 0.80
MIN_TP_R          = 1.5
FALLBACK_RR       = 2.0
RSI_LONG_MAX      = 40
RSI_SHORT_MIN     = 60
ASIA_OPEN_H       = 17    # 17:00 ET
ASIA_CLOSE_H      = 1     # 01:00 ET
NY_OPEN_H         = 8     # 08:00 ET
NY_CLOSE_H        = 17    # 17:00 ET
DEAD_ZONE_START   = 20    # 20:00 ET — red flag window start
DEAD_ZONE_END     = 22    # 22:00 ET — red flag window end
BODY_THRESH       = 0.6   # body > 0.6× ATR = momentum bar red flag
MIN_NY_RANGE_ATR  = 1.0   # skip day if NY range < 1× ATR


# ── OANDA client ──────────────────────────────────────────────────────────────

def make_client():
    return oandapyV20.API(
        access_token=cfg.OANDA_API_KEY,
        environment=cfg.OANDA_ENV
    )


# ── Candle fetching ───────────────────────────────────────────────────────────

def fetch_candles(client, granularity, count=300):
    """Fetch completed OHLC candles from OANDA. Returns DataFrame in ET timezone."""
    params = {"granularity": granularity, "count": count, "price": "M"}
    r = instruments_ep.InstrumentsCandles(cfg.INSTRUMENT, params=params)
    client.request(r)
    rows = []
    for c in r.response["candles"]:
        if not c["complete"]: continue
        ts = datetime.fromisoformat(c["time"].replace("Z","+00:00")).astimezone(ET)
        m  = c["mid"]
        rows.append(dict(
            ts=ts,
            open=float(m["o"]), high=float(m["h"]),
            low=float(m["l"]),  close=float(m["c"]),
            volume=int(c["volume"])
        ))
    df = pd.DataFrame(rows).set_index("ts")
    return df


# ── Indicators ────────────────────────────────────────────────────────────────

def add_atr(df, p=14):
    h,l,c = df["high"],df["low"],df["close"]
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=p,adjust=False).mean()
    return df

def add_sma(df, p):
    df[f"sma{p}"] = df["close"].rolling(p).mean()
    return df

def add_rsi(df, p=14):
    d = df["close"].diff()
    g = d.clip(lower=0).ewm(span=p,adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=p,adjust=False).mean()
    df["rsi"] = 100-(100/(1+g/l.replace(0,np.nan)))
    return df


# ── Session helpers ───────────────────────────────────────────────────────────

def is_ny_bar(ts):
    return NY_OPEN_H <= ts.hour < NY_CLOSE_H

def is_asia_bar(ts):
    return ts.hour >= ASIA_OPEN_H or ts.hour < ASIA_CLOSE_H

def trading_date(ts):
    """Bars before 04:00 ET belong to the prior calendar day's session."""
    if ts.hour < 4:
        return (ts - timedelta(days=1)).date()
    return ts.date()


# ── Confidence sizing ─────────────────────────────────────────────────────────

def get_risk(bar, bar_atr, ts_et):
    """Returns risk fraction (0.05 or 0.03) and flags for logging."""
    in_dead_zone   = DEAD_ZONE_START <= ts_et.hour <= DEAD_ZONE_END
    body           = abs(bar["close"] - bar["open"]) / bar_atr if bar_atr > 0 else 0
    is_momentum_bar = body >= BODY_THRESH
    flags = []
    if in_dead_zone:    flags.append("dead_zone")
    if is_momentum_bar: flags.append("momentum_bar")
    risk = cfg.RISK_LOW if flags else cfg.RISK_HIGH
    return risk, flags


# ── Position sizing ───────────────────────────────────────────────────────────

def calc_units(account_balance, risk_pct, stop_distance_usd):
    """
    OANDA XAU_USD: 1 unit = 1 troy oz. P&L in USD = units × price_change.
    stop_distance_usd = stop price distance in USD per oz.
    units = (account × risk_pct) / stop_distance
    """
    risk_usd = account_balance * risk_pct
    units    = risk_usd / stop_distance_usd
    return max(1, int(units))


# ── Order placement ───────────────────────────────────────────────────────────

def place_order(client, direction, entry, sl, tp, units, signal_info):
    """Place a bracket order on OANDA (market entry + attached SL/TP)."""
    side  = "buy" if direction == "LONG" else "sell"
    u_str = str(units if direction == "LONG" else -units)

    order_body = {
        "order": {
            "type":         "MARKET",
            "instrument":   cfg.INSTRUMENT,
            "units":        u_str,
            "timeInForce":  "FOK",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {
                "price": f"{sl:.2f}",
                "timeInForce": "GTC"
            },
            "takeProfitOnFill": {
                "price": f"{tp:.2f}",
                "timeInForce": "GTC"
            }
        }
    }

    if cfg.PAPER_MODE:
        log.info("  [PAPER] ORDER NOT SENT — details logged only")
        return {"paper": True, "order": order_body}

    r = orders_ep.OrderCreate(cfg.OANDA_ACCOUNT_ID, data=order_body)
    client.request(r)
    return r.response


# ── Trade log ─────────────────────────────────────────────────────────────────

def load_trade_log():
    p = Path(cfg.TRADE_LOG)
    if p.exists():
        return json.loads(p.read_text())
    return []

def save_trade(trade_dict):
    log_data = load_trade_log()
    log_data.append(trade_dict)
    Path(cfg.TRADE_LOG).write_text(json.dumps(log_data, indent=2, default=str))


# ── Session state ─────────────────────────────────────────────────────────────

def load_state():
    p = Path(cfg.STATE_FILE)
    if p.exists():
        return json.loads(p.read_text())
    return {}

def save_state(state):
    Path(cfg.STATE_FILE).write_text(json.dumps(state, indent=2, default=str))


# ── Signal detection ──────────────────────────────────────────────────────────

def check_for_signal(df1h, df4h, state):
    """
    Check the latest completed 1H bar for a valid sweep signal.
    Returns a signal dict if found, else None.
    Updates state in place.
    """
    if len(df1h) < ATR_PERIOD + 5: return None

    bar    = df1h.iloc[-1]
    ts_et  = df1h.index[-1]
    tdate  = trading_date(ts_et)
    tdate_str = str(tdate)

    # Only act during Asia session
    if not is_asia_bar(ts_et):
        return None

    # Initialise day state
    if state.get("tdate") != tdate_str:
        state["tdate"]        = tdate_str
        state["ny_high"]      = None
        state["ny_low"]       = None
        state["ny_range"]     = None
        state["swept_high"]   = False
        state["swept_low"]    = False
        state["trade_today"]  = False
        log.info(f"── New trading day: {tdate_str}")

    # Build NY session range from today's 1H bars (completed during NY hours)
    today_bars = df1h[[trading_date(ts) == tdate for ts in df1h.index]]
    ny_bars    = today_bars[today_bars.index.map(is_ny_bar)]

    if len(ny_bars) < 3:
        return None

    ny_high = ny_bars["high"].max()
    ny_low  = ny_bars["low"].min()
    ny_rng  = ny_high - ny_low
    atr     = bar["atr"]

    if pd.isna(atr) or atr == 0:
        return None
    if ny_rng < MIN_NY_RANGE_ATR * atr:
        log.debug(f"  Skip: NY range {ny_rng:.2f} < {MIN_NY_RANGE_ATR}× ATR {atr:.2f}")
        return None

    state["ny_high"]  = ny_high
    state["ny_low"]   = ny_low
    state["ny_range"] = ny_rng

    # 4H indicators for trend filter
    prior4  = df4h[df4h.index <= ts_et]
    if prior4.empty or pd.isna(prior4["sma20"].iloc[-1]):
        return None
    price4h = prior4["close"].iloc[-1]
    sma20   = prior4["sma20"].iloc[-1]
    sma200  = prior4["sma200"].iloc[-1]
    rsi_val = bar["rsi"]

    if pd.isna(rsi_val): return None

    signal = None

    # ── HIGH SWEEP → SHORT ─────────────────────────────────────────────
    if (not state["swept_high"]
            and bar["high"] > ny_high
            and bar["close"] < ny_high
            and price4h < sma200
            and rsi_val >= RSI_SHORT_MIN):

        state["swept_high"] = True
        entry = bar["close"]
        sl    = bar["high"] + atr * ATR_SL_MULT
        risk  = sl - entry
        if risk <= 0: return None

        tp   = ny_high - TP_PCT * ny_rng
        tp_r = (entry - tp) / risk
        if tp_r < MIN_TP_R:
            tp   = entry - risk * FALLBACK_RR
            tp_r = FALLBACK_RR
        if tp_r < MIN_TP_R: return None

        risk_pct, flags = get_risk(bar, atr, ts_et)
        units = calc_units(cfg.ACCOUNT_SIZE, risk_pct, risk)
        signal = dict(direction="SHORT", ts=ts_et, entry=entry, sl=sl, tp=tp,
                      tp_r=tp_r, units=units, risk_pct=risk_pct, flags=flags,
                      rsi=rsi_val, atr=atr, ny_high=ny_high, ny_low=ny_low,
                      ny_rng=ny_rng, price4h=price4h, sma200=sma200,
                      body_atr=abs(bar["close"]-bar["open"])/atr)

    # ── LOW SWEEP → LONG ───────────────────────────────────────────────
    elif (not state["swept_low"]
            and bar["low"] < ny_low
            and bar["close"] > ny_low
            and price4h > sma20
            and rsi_val <= RSI_LONG_MAX):

        state["swept_low"] = True
        entry = bar["close"]
        sl    = bar["low"] - atr * ATR_SL_MULT
        risk  = entry - sl
        if risk <= 0: return None

        tp   = ny_low + TP_PCT * ny_rng
        tp_r = (tp - entry) / risk
        if tp_r < MIN_TP_R:
            tp   = entry + risk * FALLBACK_RR
            tp_r = FALLBACK_RR
        if tp_r < MIN_TP_R: return None

        risk_pct, flags = get_risk(bar, atr, ts_et)
        units = calc_units(cfg.ACCOUNT_SIZE, risk_pct, risk)
        signal = dict(direction="LONG", ts=ts_et, entry=entry, sl=sl, tp=tp,
                      tp_r=tp_r, units=units, risk_pct=risk_pct, flags=flags,
                      rsi=rsi_val, atr=atr, ny_high=ny_high, ny_low=ny_low,
                      ny_rng=ny_rng, price4h=price4h, sma20=sma20,
                      body_atr=abs(bar["close"]-bar["open"])/atr)

    return signal


# ── Account balance ───────────────────────────────────────────────────────────

def get_account_balance(client):
    try:
        r = accounts_ep.AccountSummary(cfg.OANDA_ACCOUNT_ID)
        client.request(r)
        return float(r.response["account"]["balance"])
    except Exception:
        return cfg.ACCOUNT_SIZE


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    log.info("═" * 60)
    log.info("  GOLD NY-SWEEP BOT  —  Starting up")
    log.info(f"  Mode      : {'PAPER (simulation)' if cfg.PAPER_MODE else 'LIVE ORDERS'}")
    log.info(f"  Instrument: {cfg.INSTRUMENT}")
    log.info(f"  Risk      : {cfg.RISK_HIGH*100:.0f}% clean / {cfg.RISK_LOW*100:.0f}% flagged")
    log.info("═" * 60)

    client = make_client()
    state  = load_state()

    # Verify connection
    try:
        bal = get_account_balance(client)
        log.info(f"  Connected to OANDA {cfg.OANDA_ENV} — Balance: ${bal:,.2f}")
    except Exception as e:
        log.error(f"  OANDA connection failed: {e}")
        log.error("  Check OANDA_API_KEY and OANDA_ACCOUNT_ID in config.py")
        return

    poll_interval = 60   # seconds between candle checks (1 min is plenty for 1H bars)

    while True:
        try:
            now_et = datetime.now(ET)
            hour   = now_et.hour

            # Only run during or near Asia session (17:00 → 02:00 ET)
            # Outside this window just sleep to avoid unnecessary API calls
            if not (hour >= 17 or hour < 2):
                next_run = now_et.replace(hour=17, minute=0, second=0, microsecond=0)
                if now_et.hour >= 2:
                    next_run += timedelta(days=1)
                wait = (next_run - now_et).seconds
                log.info(f"  Outside Asia session — sleeping until 17:00 ET "
                         f"({wait//3600}h {(wait%3600)//60}m)")
                time.sleep(min(wait, 3600))
                continue

            # Fetch latest candles
            df1h = fetch_candles(client, "H1",  count=300)
            df4h = fetch_candles(client, "H4",  count=300)

            # Add indicators
            df1h = add_atr(df1h, ATR_PERIOD)
            df1h = add_rsi(df1h, ATR_PERIOD)
            df4h = add_sma(df4h, SMA_LONG)
            df4h = add_sma(df4h, SMA_SHORT)

            latest = df1h.index[-1]
            log.info(f"  Bar: {latest.strftime('%Y-%m-%d %H:%M ET')}  "
                     f"C={df1h['close'].iloc[-1]:.2f}  "
                     f"RSI={df1h['rsi'].iloc[-1]:.1f}  "
                     f"ATR={df1h['atr'].iloc[-1]:.2f}")

            # Check for signal
            signal = check_for_signal(df1h, df4h, state)

            if signal:
                s = signal
                flag_str = ", ".join(s["flags"]) if s["flags"] else "none"
                log.info("  ┌─ SIGNAL DETECTED " + "─"*40)
                log.info(f"  │  Direction  : {s['direction']}")
                log.info(f"  │  Entry      : {s['entry']:.2f}")
                log.info(f"  │  Stop loss  : {s['sl']:.2f}")
                log.info(f"  │  Take profit: {s['tp']:.2f}  ({s['tp_r']:.2f}R)")
                log.info(f"  │  Units      : {s['units']}")
                log.info(f"  │  Risk       : {s['risk_pct']*100:.0f}%  (${s['risk_pct']*cfg.ACCOUNT_SIZE:,.0f})")
                log.info(f"  │  Red flags  : {flag_str}")
                log.info(f"  │  RSI        : {s['rsi']:.1f}  |  ATR: {s['atr']:.2f}")
                log.info(f"  │  NY range   : {s['ny_low']:.2f} – {s['ny_high']:.2f}  ({s['ny_rng']:.2f})")
                log.info(f"  │  Body×ATR   : {s['body_atr']:.3f}")
                log.info("  └" + "─"*50)

                # Place order
                bal   = get_account_balance(client)
                units = calc_units(bal, s["risk_pct"], abs(s["entry"] - s["sl"]))
                response = place_order(client, s["direction"], s["entry"],
                                       s["sl"], s["tp"], units, s)

                # Log trade
                trade_record = {
                    "timestamp"  : str(s["ts"]),
                    "direction"  : s["direction"],
                    "entry"      : s["entry"],
                    "sl"         : s["sl"],
                    "tp"         : s["tp"],
                    "tp_r"       : s["tp_r"],
                    "units"      : units,
                    "risk_pct"   : s["risk_pct"],
                    "risk_usd"   : round(s["risk_pct"] * bal, 2),
                    "red_flags"  : s["flags"],
                    "rsi"        : s["rsi"],
                    "atr"        : s["atr"],
                    "ny_high"    : s["ny_high"],
                    "ny_low"     : s["ny_low"],
                    "ny_rng"     : s["ny_rng"],
                    "paper_mode" : cfg.PAPER_MODE,
                    "result"     : "OPEN",
                    "pnl_r"      : None,
                    "pnl_usd"    : None,
                    "oanda_resp" : response,
                }
                save_trade(trade_record)
                log.info(f"  Trade logged → {cfg.TRADE_LOG}")

            # Save state after every check
            save_state(state)

        except KeyboardInterrupt:
            log.info("  Shutdown requested — saving state and exiting.")
            save_state(state)
            break
        except Exception as e:
            log.error(f"  Error: {e}")
            log.error(traceback.format_exc())

        time.sleep(poll_interval)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()

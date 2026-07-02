"""
Gold NY-Sweep Reversal Bot — Alpaca Paper Trading
==================================================
Instrument : GLD (SPDR Gold Trust ETF) — tracks XAU/USD, ~1/10 oz per share
Broker     : Alpaca paper trading (extended-hours orders for Asia session)
Data       : yfinance 1H GLD bars (includes pre/after-market)

Strategy (champion config — 34 trades, +213% return over 2 years):
  • Mark NY session high/low from GLD 1H bars (09:30–16:00 ET each day)
  • Asia session window (17:00–01:00 ET): detect bar that wicks beyond
    NY high/low and closes back inside — liquidity sweep reversal
  • Trend filter : 4H SMA-20 for LONG | 4H SMA-200 for SHORT
  • RSI gate     : entry bar RSI(14) ≤ 40 (LONG) | ≥ 60 (SHORT)
  • Stop loss    : sweep wick extreme ± 0.15× ATR(14)
  • Take profit  : 80% of NY session range back inside (min 1.5R, else 2R)

Dynamic position sizing (confidence system):
  5% risk  — clean setup, no red flags
  3% risk  — dead zone (20:00–22:00 ET) OR momentum bar (body > 0.6× ATR)
"""

import os, sys, json, logging, traceback
from datetime import datetime, timedelta, date
from pathlib import Path
import pytz, pandas as pd, numpy as np, yfinance as yf

from alpaca.trading.client   import TradingClient
from alpaca.trading.requests import (MarketOrderRequest, LimitOrderRequest,
                                     TakeProfitRequest, StopLossRequest)
from alpaca.trading.enums    import OrderSide, TimeInForce, OrderClass

ET = pytz.timezone("America/New_York")

# ── Credentials (from environment / GitHub secrets) ───────────────────────────
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY_GOLD") or os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY_GOLD") or os.environ.get("ALPACA_SECRET_KEY", "")
PAPER_MODE        = os.environ.get("GOLD_PAPER_MODE", "true").lower() == "true"

# ── Bot settings ──────────────────────────────────────────────────────────────
SYMBOL       = "GLD"          # SPDR Gold Trust ETF
ACCOUNT_SIZE = 100_000        # USD — keep in sync with your Alpaca paper balance
RISK_HIGH    = 0.05           # 5% — clean setup
RISK_LOW     = 0.03           # 3% — red flag detected

STATE_FILE = "gold_bot_state.json"
TRADE_LOG  = "gold_trade_log.json"

# ── Strategy constants ────────────────────────────────────────────────────────
SMA_LONG        = 20;   SMA_SHORT      = 200
ATR_PERIOD      = 14;   ATR_SL_MULT    = 0.15
TP_PCT          = 0.80; MIN_TP_R       = 1.5;  FALLBACK_RR = 2.0
RSI_LONG_MAX    = 40;   RSI_SHORT_MIN  = 60
NY_OPEN_H       = 9;    NY_CLOSE_H     = 16    # GLD regular market hours
ASIA_OPEN_H     = 17;   ASIA_CLOSE_H   = 1
DEAD_ZONE_START = 20;   DEAD_ZONE_END  = 22
BODY_THRESH     = 0.6
MIN_NY_RANGE_ATR = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("GoldBot")


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


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_bars(symbol, interval, days=60, prepost=True):
    """Fetch OHLC bars via yfinance. prepost=True includes pre/after-market."""
    import warnings; warnings.filterwarnings("ignore")
    df = yf.download(symbol, period=f"{days}d", interval=interval,
                     auto_adjust=True, prepost=prepost, progress=False)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df[["open","high","low","close","volume"]].dropna()
    return df


# ── Session helpers ───────────────────────────────────────────────────────────

def trading_date(ts):
    """Bars before 04:00 ET belong to the previous calendar date's session."""
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

def calc_shares(account_bal, risk_pct, stop_dist_usd):
    """
    GLD: 1 share ≈ 0.1 oz gold, priced in USD.
    P&L per share = price change in USD.
    shares = (account × risk_pct) / stop_distance_per_share
    """
    risk_usd = account_bal * risk_pct
    shares   = risk_usd / stop_dist_usd
    return max(1, int(shares))


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

def place_order(client, direction, shares, entry, sl_price, tp_price):
    """
    Bracket order: market entry with attached stop-loss and take-profit.
    extended_hours=True allows execution outside 09:30–16:00 ET.
    """
    side = OrderSide.BUY if direction == "LONG" else OrderSide.SELL

    if PAPER_MODE:
        log.info(f"  [PAPER MODE] Would place: {direction} {shares} shares GLD")
        log.info(f"    Entry ~{entry:.2f}  |  SL {sl_price:.2f}  |  TP {tp_price:.2f}")
        return {"paper": True}

    req = MarketOrderRequest(
        symbol        = SYMBOL,
        qty           = shares,
        side          = side,
        time_in_force = TimeInForce.DAY,
        extended_hours= True,       # allows execution during Asia session hours
        order_class   = OrderClass.BRACKET,
        take_profit   = TakeProfitRequest(limit_price=round(tp_price, 2)),
        stop_loss     = StopLossRequest(stop_price=round(sl_price, 2)),
    )
    order = client.submit_order(req)
    return {"order_id": str(order.id), "status": str(order.status)}


# ── Signal detection ──────────────────────────────────────────────────────────

def check_signal(df1h, df4h, state):
    """Check the latest 1H bar for a valid sweep reversal signal."""
    if len(df1h) < ATR_PERIOD + 5:
        return None

    bar   = df1h.iloc[-1]
    ts_et = df1h.index[-1]
    tdate = str(trading_date(ts_et))

    if not is_asia_bar(ts_et):
        log.info(f"  Hour {ts_et.hour} ET — outside Asia session, skipping")
        return None

    # Reset daily state on new trading day
    if state.get("tdate") != tdate:
        state.update({
            "tdate": tdate, "ny_high": None, "ny_low": None,
            "ny_rng": None, "swept_high": False, "swept_low": False,
        })
        log.info(f"  New trading day: {tdate}")

    # Build NY session range from today's 1H bars (09:30–16:00)
    today_bars = df1h[[trading_date(ts) == tdate for ts in df1h.index]]
    ny_bars    = today_bars[today_bars.index.map(is_ny_bar)]

    if len(ny_bars) < 3:
        log.info(f"  Fewer than 3 NY bars available today — skipping")
        return None

    ny_high = float(ny_bars["high"].max())
    ny_low  = float(ny_bars["low"].min())
    ny_rng  = ny_high - ny_low
    atr     = float(bar["atr"])

    if pd.isna(atr) or atr == 0 or ny_rng < MIN_NY_RANGE_ATR * atr:
        log.info(f"  NY range {ny_rng:.2f} < min {MIN_NY_RANGE_ATR}× ATR {atr:.2f} — skip")
        return None

    state.update({"ny_high": ny_high, "ny_low": ny_low, "ny_rng": ny_rng})

    # 4H indicators
    prior4 = df4h[df4h.index <= ts_et]
    if prior4.empty or pd.isna(prior4["sma20"].iloc[-1]):
        return None
    price4h = float(prior4["close"].iloc[-1])
    sma20   = float(prior4["sma20"].iloc[-1])
    sma200  = float(prior4["sma200"].iloc[-1])
    rsi_val = float(bar["rsi"])
    if pd.isna(rsi_val): return None

    body_ratio = abs(bar["close"] - bar["open"]) / atr if atr > 0 else 0

    log.info(
        f"  Bar {ts_et.strftime('%H:%M ET')} | GLD {bar['close']:.2f} | "
        f"RSI {rsi_val:.1f} | NY {ny_low:.2f}–{ny_high:.2f} | "
        f"4H vs SMA20={sma20:.2f} SMA200={sma200:.2f}"
    )

    # ── SHORT: high sweep ─────────────────────────────────────────────────
    if (not state["swept_high"]
            and bar["high"] > ny_high
            and bar["close"] < ny_high
            and price4h < sma200
            and rsi_val >= RSI_SHORT_MIN):
        state["swept_high"] = True
        entry = float(bar["close"])
        sl    = bar["high"] + atr * ATR_SL_MULT
        risk  = sl - entry
        if risk <= 0: return None
        tp   = ny_high - TP_PCT * ny_rng
        tp_r = (entry - tp) / risk
        if tp_r < MIN_TP_R: tp = entry - risk * FALLBACK_RR; tp_r = FALLBACK_RR
        if tp_r < MIN_TP_R: return None
        rp, flags = get_risk(bar, atr, ts_et)
        return dict(direction="SHORT", ts=ts_et, entry=entry, sl=sl, tp=tp,
                    tp_r=round(tp_r,2), risk=risk, risk_pct=rp, flags=flags,
                    rsi=round(rsi_val,1), atr=round(atr,3),
                    ny_high=ny_high, ny_low=ny_low, ny_rng=round(ny_rng,3),
                    body_atr=round(body_ratio,3))

    # ── LONG: low sweep ───────────────────────────────────────────────────
    if (not state["swept_low"]
            and bar["low"] < ny_low
            and bar["close"] > ny_low
            and price4h > sma20
            and rsi_val <= RSI_LONG_MAX):
        state["swept_low"] = True
        entry = float(bar["close"])
        sl    = bar["low"] - atr * ATR_SL_MULT
        risk  = entry - sl
        if risk <= 0: return None
        tp   = ny_low + TP_PCT * ny_rng
        tp_r = (tp - entry) / risk
        if tp_r < MIN_TP_R: tp = entry + risk * FALLBACK_RR; tp_r = FALLBACK_RR
        if tp_r < MIN_TP_R: return None
        rp, flags = get_risk(bar, atr, ts_et)
        return dict(direction="LONG", ts=ts_et, entry=entry, sl=sl, tp=tp,
                    tp_r=round(tp_r,2), risk=risk, risk_pct=rp, flags=flags,
                    rsi=round(rsi_val,1), atr=round(atr,3),
                    ny_high=ny_high, ny_low=ny_low, ny_rng=round(ny_rng,3),
                    body_atr=round(body_ratio,3))

    log.info("  No sweep signal on this bar")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now_et = datetime.now(ET)
    log.info("═" * 62)
    log.info(f"  Gold Bot  —  {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    log.info(f"  Instrument : {SYMBOL} (SPDR Gold Trust ETF)")
    log.info(f"  Mode       : {'PAPER (signals logged, no orders)' if PAPER_MODE else 'LIVE PAPER ORDERS on Alpaca'}")
    log.info(f"  Risk       : {RISK_HIGH*100:.0f}% clean  /  {RISK_LOW*100:.0f}% flagged")
    log.info("═" * 62)

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        log.error("  ALPACA_API_KEY / ALPACA_SECRET_KEY not set.")
        log.error("  Add them as GitHub secrets: ALPACA_API_KEY_GOLD and ALPACA_SECRET_KEY_GOLD")
        sys.exit(1)

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

    try:
        acct    = client.get_account()
        balance = float(acct.cash)
        log.info(f"  Alpaca paper account — Cash: ${balance:,.2f}  |  Equity: ${float(acct.equity):,.2f}")
    except Exception as e:
        log.error(f"  Alpaca connection failed: {e}")
        sys.exit(1)

    state = load_state()

    # Fetch bars via yfinance (prepost=True gives us extended hours GLD data)
    try:
        log.info("  Fetching GLD bars…")
        df1h = fetch_bars(SYMBOL, "1h",  days=60, prepost=True)
        df4h = fetch_bars(SYMBOL, "1h",  days=90, prepost=True)   # resample to 4H below
        # Resample 1H → 4H
        df4h = df4h.resample("4h").agg({"open":"first","high":"max",
                                         "low":"min","close":"last","volume":"sum"}).dropna()
        df1h = add_atr(df1h, ATR_PERIOD); df1h = add_rsi(df1h, ATR_PERIOD)
        df4h = add_sma(df4h, SMA_LONG);   df4h = add_sma(df4h, SMA_SHORT)
        log.info(f"  GLD 1H: {len(df1h)} bars  |  4H: {len(df4h)} bars  "
                 f"|  Latest: {df1h.index[-1].strftime('%Y-%m-%d %H:%M ET')}")
    except Exception as e:
        log.error(f"  Failed to fetch GLD bars: {e}")
        save_state(state); sys.exit(1)

    signal = check_signal(df1h, df4h, state)

    if signal:
        s = signal
        flags_str = ", ".join(s["flags"]) if s["flags"] else "none — clean setup"
        log.info("  ┌─ SIGNAL DETECTED ───────────────────────────────────────")
        log.info(f"  │  {s['direction']}  GLD @ ~{s['entry']:.2f}")
        log.info(f"  │  Stop   : {s['sl']:.2f}  ({abs(s['sl']-s['entry']):.2f} pts)")
        log.info(f"  │  Target : {s['tp']:.2f}  ({s['tp_r']:.2f}R)")
        log.info(f"  │  Risk   : {s['risk_pct']*100:.0f}%  (${s['risk_pct']*balance:,.0f})")
        log.info(f"  │  Flags  : {flags_str}")
        log.info(f"  │  RSI    : {s['rsi']}  |  ATR: {s['atr']:.3f}")
        log.info(f"  │  NY range: {s['ny_low']:.2f} – {s['ny_high']:.2f}  ({s['ny_rng']:.2f} pts)")
        log.info("  └─────────────────────────────────────────────────────────")

        shares   = calc_shares(balance, s["risk_pct"], s["risk"])
        response = place_order(client, s["direction"], shares,
                               s["entry"], s["sl"], s["tp"])

        save_trade({
            "timestamp"  : str(s["ts"]),
            "direction"  : s["direction"],
            "symbol"     : SYMBOL,
            "entry"      : s["entry"],
            "sl"         : round(s["sl"], 2),
            "tp"         : round(s["tp"], 2),
            "tp_r"       : s["tp_r"],
            "shares"     : shares,
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
            "alpaca_resp": response,
        })
        log.info(f"  Trade saved → {TRADE_LOG}")

    save_state(state)
    log.info("  Done.")


if __name__ == "__main__":
    main()

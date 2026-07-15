"""
Gold XAU/USD — NY Sweep Reversal Bot — OANDA Practice Trading
==============================================================
Instrument : XAU_USD (Gold spot vs USD — direct match to GC=F backtests)
Broker     : OANDA practice account (fxTrade Practice)
Data       : OANDA v20 candles API (H1 + H4)

Strategy (champion config — 28 trades, +285% return over 2 years):
  • Mark NY session high/low from 1H bars (08:00–17:00 ET each day)
  • Asia session (17:00–01:00 ET): bar wicks beyond NY level, closes back inside
  • Trend filter : 4H SMA-20 for LONG | 4H SMA-200 for SHORT
  • RSI gate     : entry bar RSI(14) ≤ 40 (LONG) | ≥ 65 (SHORT)
  • Stop loss    : sweep wick extreme ± 0.15× ATR(14)
  • TP1 (30%)    : 80% of NY session range (min 1.5R, fallback 2R)
  • TP2 (70%)    : 200% of NY session range — runner continues after TP1 hit
  • Runner stop  : moved to entry immediately when TP1 hit ($0 worst case on runner)

Position sizing:
  5% flat risk ($5,000) on every trade — clean and flagged alike
  Flagged detection (dead zone 20-22 ET / momentum bar >0.6× ATR) used for BE@1R only:
    → flagged trades: move BOTH legs' SL to entry once price reaches 1R profit

Order structure: two OANDA orders placed on entry
  Order A: 30% of units, stop loss + take profit at TP1
  Order B: 70% of units, stop loss only (TP2 set after TP1 hit via check_tp1_hit)
"""

import os, sys, json, logging, traceback
from datetime import datetime, timedelta
from pathlib import Path
import pytz, pandas as pd, numpy as np

import oandapyV20
import oandapyV20.endpoints.accounts    as accounts_ep
import oandapyV20.endpoints.orders      as orders_ep
import oandapyV20.endpoints.trades      as trades_ep
import oandapyV20.endpoints.instruments as instruments_ep

ET = pytz.timezone("America/New_York")

# ── Credentials (GitHub secrets) ──────────────────────────────────────────────
OANDA_API_KEY    = os.environ.get("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "")
OANDA_ENV        = os.environ.get("OANDA_ENV", "practice")
PAPER_MODE       = os.environ.get("GOLD_PAPER_MODE", "true").lower() == "true"
ACCOUNT_SIZE     = float(os.environ.get("ACCOUNT_SIZE", "100000"))
RISK_FLAT        = float(os.environ.get("RISK_FLAT", "0.15"))

INSTRUMENT = "XAU_USD"
STATE_FILE = "gold_bot_state.json"
TRADE_LOG  = "gold_trade_log.json"

# ── Strategy constants ────────────────────────────────────────────────────────
SMA_LONG        = 20;  SMA_SHORT      = 200
ATR_PERIOD      = 14;  ATR_SL_MULT    = 0.15
TP1_PCT         = 0.80  # partial exit level: 80% of NY range
TP2_PCT         = 2.00  # runner target: 200% of NY range
TP_SPLIT        = 0.30  # 30% closes at TP1; 70% runs to TP2
MIN_TP_R        = 1.5;  FALLBACK_RR = 2.0
RSI_LONG_MAX    = 40;  RSI_SHORT_MIN  = 65
NY_OPEN_H       = 8;   NY_CLOSE_H     = 17
ASIA_OPEN_H     = 17;  ASIA_CLOSE_H   = 1
DEAD_ZONE_START = 20;  DEAD_ZONE_END  = 22
BODY_THRESH     = 0.6
MIN_NY_RANGE_ATR = 1.0
BE_TRIGGER_R    = 1.0   # move SL to entry once flagged trade reaches 1R profit

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

def get_flags(bar, bar_atr, ts_et):
    """Detect red flags for BE@1R gating. Sizing is flat 5% regardless."""
    in_dead_zone    = DEAD_ZONE_START <= ts_et.hour <= DEAD_ZONE_END
    body            = abs(bar["close"] - bar["open"]) / bar_atr if bar_atr > 0 else 0
    is_momentum_bar = body >= BODY_THRESH
    flags = []
    if in_dead_zone:    flags.append("dead_zone(20-22ET)")
    if is_momentum_bar: flags.append(f"momentum_bar(body={body:.2f}x)")
    flagged = bool(flags)
    return flagged, flags


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

def update_trade(index, updates):
    """Patch fields on an existing trade log entry by index."""
    p = Path(TRADE_LOG)
    data = json.loads(p.read_text()) if p.exists() else []
    if 0 <= index < len(data):
        data[index].update(updates)
        p.write_text(json.dumps(data, indent=2, default=str))


# ── Order placement ───────────────────────────────────────────────────────────

def _place_single_order(client, direction, units, sl, tp=None):
    """Place one market order. tp=None means no take profit (runner leg)."""
    u_str = str(units if direction == "LONG" else -units)
    order = {
        "type":         "MARKET",
        "instrument":   INSTRUMENT,
        "units":        u_str,
        "timeInForce":  "FOK",
        "positionFill": "DEFAULT",
        "stopLossOnFill": {"price": f"{sl:.2f}", "timeInForce": "GTC"},
    }
    if tp is not None:
        order["takeProfitOnFill"] = {"price": f"{tp:.2f}", "timeInForce": "GTC"}
    if PAPER_MODE:
        return {"paper": True}
    r = orders_ep.OrderCreate(OANDA_ACCOUNT_ID, data={"order": order})
    client.request(r)
    return r.response

def place_orders(client, direction, total_units, sl, tp1, tp2):
    """
    Place two orders: 30% at TP1, 70% runner (no TP until TP1 hit).
    Returns (tp1_trade_id, runner_trade_id).
    """
    units_tp1    = max(1, round(total_units * TP_SPLIT))
    units_runner = max(1, total_units - units_tp1)

    if PAPER_MODE:
        log.info(f"  [PAPER MODE] Would place Order A: {units_tp1} units → TP1 {tp1:.2f}")
        log.info(f"  [PAPER MODE] Would place Order B: {units_runner} units → runner to TP2 {tp2:.2f}")
        return None, None

    resp_tp1 = _place_single_order(client, direction, units_tp1, sl, tp1)
    tp1_id   = (resp_tp1.get("orderFillTransaction", {}).get("tradeOpened", {}).get("tradeID")
                or resp_tp1.get("orderFillTransaction", {}).get("tradeID"))

    if not tp1_id:
        log.error("  TP1 order did not fill — aborting runner placement")
        return None, None

    try:
        resp_run = _place_single_order(client, direction, units_runner, sl, tp=None)
        run_id   = (resp_run.get("orderFillTransaction", {}).get("tradeOpened", {}).get("tradeID")
                    or resp_run.get("orderFillTransaction", {}).get("tradeID"))
    except Exception as e:
        log.error(f"  Runner order failed: {e}. TP1 leg {tp1_id} left open — manual review needed")
        return tp1_id, None

    if not run_id:
        log.error(f"  Runner order did not fill. TP1 leg {tp1_id} left open — manual review needed")
        return tp1_id, None

    return tp1_id, run_id


# ── Break-even management ─────────────────────────────────────────────────────

def _move_sl_to_entry(client, trade_id, entry):
    """Move a single OANDA trade's stop loss to entry price."""
    body = {"stopLoss": {"price": f"{entry:.2f}", "timeInForce": "GTC"}}
    r = trades_ep.TradeCRCDO(OANDA_ACCOUNT_ID, trade_id, data=body)
    client.request(r)


def check_break_even(client, state):
    """
    For flagged trades: if price reaches BE_TRIGGER_R profit, move BOTH legs'
    SL to entry so worst case becomes $0. Uses runner leg's unrealised P&L to
    measure price movement (runner reflects live price; TP1 leg is same instrument).
    """
    open_trade = state.get("open_flagged_trade")
    if not open_trade or open_trade.get("be_moved"):
        return

    runner_id = open_trade.get("trade_id")   # runner leg (70%)
    entry     = open_trade.get("entry")
    risk      = open_trade.get("risk")
    direction = open_trade.get("direction")
    if not all([runner_id, entry, risk, direction]):
        return

    # Also need TP1 leg ID to move its SL
    partial_trade = state.get("open_partial_trade", {})
    tp1_id = partial_trade.get("tp1_trade_id")

    try:
        r = trades_ep.TradeDetails(OANDA_ACCOUNT_ID, runner_id)
        client.request(r)
        t = r.response["trade"]

        if t["state"] != "OPEN":
            log.info(f"  Runner {runner_id} no longer open — clearing flagged trade from state")
            state.pop("open_flagged_trade", None)
            return

        unrealised_pl = float(t.get("unrealizedPL", 0))
        units         = abs(int(t["currentUnits"]))
        if units == 0:
            return

        # price_r: how many R's has price moved in our favour on the runner leg
        price_r = unrealised_pl / (risk * units) if units > 0 else 0
        log.info(f"  Flagged trade runner {runner_id}: unrealised ${unrealised_pl:+.2f}  (~{price_r:+.2f}R)")

        if price_r >= BE_TRIGGER_R:
            log.info(f"  BE trigger reached ({price_r:.2f}R ≥ {BE_TRIGGER_R}R) — moving both legs' SL to entry {entry:.2f}")
            if not PAPER_MODE:
                # Move runner SL to entry
                _move_sl_to_entry(client, runner_id, entry)
                log.info(f"  Runner {runner_id}: SL → entry {entry:.2f}")
                # Move TP1 leg SL to entry (prevents full-leg loss if price reverses before TP1)
                if tp1_id:
                    try:
                        _move_sl_to_entry(client, tp1_id, entry)
                        log.info(f"  TP1 leg {tp1_id}: SL → entry {entry:.2f}")
                    except Exception as e:
                        log.warning(f"  Could not move TP1 leg SL (may already be closed): {e}")
            else:
                log.info(f"  [PAPER MODE] Would move both legs SL → entry {entry:.2f}")

            open_trade["be_moved"] = True
            state["open_flagged_trade"] = open_trade
    except Exception as e:
        log.warning(f"  BE check failed for runner {runner_id}: {e}")


# ── Closed trade scanner ──────────────────────────────────────────────────────

def scan_closed_trades(client):
    """
    Read the trade log, find any entries with result="OPEN", check OANDA for
    their current state, and write back the outcome (WIN/LOSS/PARTIAL/BE).
    Handles the two-leg structure: tp1_trade_id + runner_trade_id.
    """
    p = Path(TRADE_LOG)
    if not p.exists():
        return
    data = json.loads(p.read_text())
    changed = False

    for i, entry in enumerate(data):
        if entry.get("result") != "OPEN":
            continue
        if entry.get("paper_mode") and not entry.get("tp1_trade_id"):
            continue  # paper mode entry with no real IDs — can't check OANDA

        tp1_id  = entry.get("tp1_trade_id")
        run_id  = entry.get("runner_trade_id")
        entry_p = entry.get("entry", 0)
        risk    = entry.get("risk", 1)
        rp      = entry.get("risk_pct", 0.03)
        direction = entry.get("direction", "LONG")

        try:
            # Check TP1 leg
            tp1_closed = False; tp1_exit = None
            if tp1_id:
                r = trades_ep.TradeDetails(OANDA_ACCOUNT_ID, tp1_id)
                client.request(r)
                t1 = r.response["trade"]
                if t1["state"] == "CLOSED":
                    tp1_closed = True
                    tp1_exit   = float(t1.get("averageClosePrice", entry_p))

            # Check runner leg
            run_closed = False; run_exit = None; run_state = "OPEN"
            if run_id:
                r2 = trades_ep.TradeDetails(OANDA_ACCOUNT_ID, run_id)
                client.request(r2)
                t2 = r2.response["trade"]
                run_state  = t2["state"]
                if run_state == "CLOSED":
                    run_closed = True
                    run_exit   = float(t2.get("averageClosePrice", entry_p))

            if not tp1_closed and not run_closed:
                log.info(f"  Trade {tp1_id}/{run_id} still OPEN — no update")
                continue

            # Determine outcome
            tp1_pnl_r  = 0.0; run_pnl_r = 0.0
            tp1_closed_result = "OPEN"
            run_closed_result = "OPEN"

            if tp1_closed and tp1_exit:
                tp1_pnl_r = ((tp1_exit - entry_p) / risk if direction == "LONG"
                             else (entry_p - tp1_exit) / risk)
                tp1_closed_result = "WIN" if tp1_pnl_r > 0 else "LOSS"

            if run_closed and run_exit:
                run_pnl_r = ((run_exit - entry_p) / risk if direction == "LONG"
                             else (entry_p - run_exit) / risk)
                if abs(run_exit - entry_p) < 0.50:  # within $0.50 = BE
                    run_closed_result = "BE"
                elif run_pnl_r > 0:
                    run_closed_result = "WIN"
                else:
                    run_closed_result = "LOSS"

            # Combine legs into final trade outcome
            total_pnl_r = (TP_SPLIT * tp1_pnl_r) + ((1 - TP_SPLIT) * run_pnl_r)
            total_pnl_usd = total_pnl_r * ACCOUNT_SIZE * rp

            if tp1_closed and run_closed:
                if tp1_closed_result == "WIN" and run_closed_result in ("WIN","BE"):
                    final_result = "WIN" if run_pnl_r > 0.5 else "PARTIAL"
                elif tp1_closed_result == "WIN" and run_closed_result == "LOSS":
                    final_result = "PARTIAL"
                elif tp1_closed_result == "LOSS":
                    final_result = "BE" if abs(tp1_pnl_r) < 0.1 else "LOSS"
                else:
                    final_result = "PARTIAL"
            elif tp1_closed and not run_closed:
                final_result = "OPEN"  # TP1 closed but runner still running
            else:
                # Only runner closed (unusual — stop hit before TP1)
                final_result = run_closed_result

            updates = {
                "result"       : final_result,
                "tp1_result"   : tp1_closed_result,
                "tp1_exit"     : round(tp1_exit, 2) if tp1_exit else None,
                "runner_result": run_closed_result,
                "runner_exit"  : round(run_exit, 2) if run_exit else None,
                "pnl_r"        : round(total_pnl_r, 3),
                "pnl_usd"      : round(total_pnl_usd, 0),
                "closed_at"    : str(datetime.now(ET)),
            }
            update_trade(i, updates)
            changed = True
            log.info(f"  Trade {i} closed → {final_result}  P&L: {total_pnl_r:+.3f}R  ${total_pnl_usd:+,.0f}")

        except Exception as e:
            log.warning(f"  scan_closed_trades: trade {i} check failed: {e}")

    if changed:
        log.info("  Trade log updated with closed outcomes.")


# ── TP1 hit → activate runner ─────────────────────────────────────────────────

def check_tp1_hit(client, state):
    """
    Check if the TP1 leg of an open trade has been filled (closed by OANDA TP).
    If so: move runner stop to entry and set runner TP to tp2.
    Clears tp1_trade_id from state once confirmed closed.
    """
    open_trade = state.get("open_partial_trade")
    if not open_trade:
        return
    if open_trade.get("runner_activated"):
        return

    tp1_id  = open_trade.get("tp1_trade_id")
    run_id  = open_trade.get("runner_trade_id")
    entry   = open_trade.get("entry")
    tp2     = open_trade.get("tp2")

    if not tp1_id or not run_id:
        return

    try:
        r = trades_ep.TradeDetails(OANDA_ACCOUNT_ID, tp1_id)
        client.request(r)
        tp1_state = r.response["trade"]["state"]

        if tp1_state != "CLOSED":
            log.info(f"  TP1 leg {tp1_id} still OPEN — runner waiting")
            return

        log.info(f"  TP1 leg {tp1_id} confirmed closed — activating runner {run_id}")

        # Verify runner is still open before trying to modify it
        r_run = trades_ep.TradeDetails(OANDA_ACCOUNT_ID, run_id)
        client.request(r_run)
        run_state = r_run.response["trade"]["state"]
        if run_state != "OPEN":
            log.info(f"  Runner {run_id} already closed — clearing partial trade state")
            state.pop("open_partial_trade", None)
            return

        if not PAPER_MODE:
            body = {
                "stopLoss":   {"price": f"{entry:.2f}", "timeInForce": "GTC"},
                "takeProfit": {"price": f"{tp2:.2f}",   "timeInForce": "GTC"},
            }
            r2 = trades_ep.TradeCRCDO(OANDA_ACCOUNT_ID, run_id, data=body)
            client.request(r2)
            log.info(f"  Runner {run_id}: SL → entry {entry:.2f}, TP → {tp2:.2f}")
        else:
            log.info(f"  [PAPER MODE] Would set runner SL={entry:.2f}, TP={tp2:.2f}")

        open_trade["runner_activated"] = True
        state["open_partial_trade"] = open_trade
    except Exception as e:
        log.warning(f"  check_tp1_hit failed: {e}")


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

    today_bars = df1h[[str(trading_date(ts)) == tdate for ts in df1h.index]]
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

    # Block new signals if a position from a previous day is still open
    if state.get("open_partial_trade") and state["open_partial_trade"].get("runner_trade_id"):
        if not state["open_partial_trade"].get("runner_activated"):
            log.info("  Existing position still open — skipping signal detection")
            return None

    prior4  = df4h[df4h.index <= ts_et]
    if prior4.empty: return None
    sma20_val  = prior4["sma20"].iloc[-1]
    sma200_val = prior4["sma200"].iloc[-1]
    if pd.isna(sma20_val) or pd.isna(sma200_val): return None
    price4h = float(prior4["close"].iloc[-1])
    sma20   = float(sma20_val)
    sma200  = float(sma200_val)
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
        tp1  = ny_high - TP1_PCT * ny_rng; tp1_r = (entry - tp1) / risk
        if tp1_r < MIN_TP_R: tp1 = entry - risk * FALLBACK_RR; tp1_r = FALLBACK_RR
        if tp1_r < MIN_TP_R: return None
        tp2  = ny_high - TP2_PCT * ny_rng
        flagged, flags = get_flags(bar, atr, ts_et)
        return dict(direction="SHORT", ts=ts_et, entry=entry, sl=sl,
                    tp1=tp1, tp2=tp2, tp_r=round(tp1_r, 2),
                    risk=risk, risk_pct=RISK_FLAT, flagged=flagged, flags=flags,
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
        tp1  = ny_low + TP1_PCT * ny_rng; tp1_r = (tp1 - entry) / risk
        if tp1_r < MIN_TP_R: tp1 = entry + risk * FALLBACK_RR; tp1_r = FALLBACK_RR
        if tp1_r < MIN_TP_R: return None
        tp2  = ny_low + TP2_PCT * ny_rng
        flagged, flags = get_flags(bar, atr, ts_et)
        return dict(direction="LONG", ts=ts_et, entry=entry, sl=sl,
                    tp1=tp1, tp2=tp2, tp_r=round(tp1_r, 2),
                    risk=risk, risk_pct=RISK_FLAT, flagged=flagged, flags=flags,
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
    log.info(f"  Risk       : {RISK_FLAT*100:.0f}% flat  (BE@1R on flagged trades only)")
    log.info("═" * 62)

    if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
        log.error("  OANDA_API_KEY / OANDA_ACCOUNT_ID not set.")
        log.error("  Add them as GitHub secrets.")
        sys.exit(1)

    client = oandapyV20.API(access_token=OANDA_API_KEY, environment=OANDA_ENV)

    import time
    balance = None
    for attempt in range(1, 4):
        try:
            r = accounts_ep.AccountSummary(OANDA_ACCOUNT_ID)
            client.request(r)
            balance = float(r.response["account"]["balance"])
            log.info(f"  OANDA {OANDA_ENV} — Balance: ${balance:,.2f}")
            break
        except Exception as e:
            log.warning(f"  OANDA connection attempt {attempt}/3 failed: {e}")
            if attempt < 3:
                time.sleep(10)
    if balance is None:
        log.error("  OANDA connection failed after 3 attempts — exiting")
        sys.exit(1)

    state = load_state()

    df1h = df4h = None
    for attempt in range(1, 4):
        try:
            log.info(f"  Fetching XAU_USD candles… (attempt {attempt}/3)")
            df1h = fetch_candles(client, "H1", count=300)
            df4h = fetch_candles(client, "H4", count=300)
            df1h = add_atr(df1h, ATR_PERIOD); df1h = add_rsi(df1h, ATR_PERIOD)
            df4h = add_sma(df4h, SMA_LONG);   df4h = add_sma(df4h, SMA_SHORT)
            log.info(f"  1H: {len(df1h)} bars  |  4H: {len(df4h)} bars  "
                     f"|  Latest: {df1h.index[-1].strftime('%Y-%m-%d %H:%M ET')}")
            break
        except Exception as e:
            log.warning(f"  Candle fetch attempt {attempt}/3 failed: {e}")
            if attempt < 3:
                time.sleep(10)
    if df1h is None:
        log.error("  Failed to fetch candles after 3 attempts — exiting")
        save_state(state); sys.exit(1)

    # Check BE on any existing open flagged trade before looking for new signals
    check_break_even(client, state)
    # Check if TP1 has been hit and runner needs activating
    check_tp1_hit(client, state)
    # Scan all open log entries and record any that have now closed
    scan_closed_trades(client)

    signal = check_signal(df1h, df4h, state)

    if signal:
        s = signal
        flags_str = ", ".join(s["flags"]) if s["flags"] else "none — clean setup"
        be_note   = "BE@1R will apply" if s["flagged"] else "hold to TP1/TP2 (no trailing)"
        units_tp1    = max(1, round(calc_units(balance, s["risk_pct"], s["risk"]) * TP_SPLIT))
        units_runner = max(1, calc_units(balance, s["risk_pct"], s["risk"]) - units_tp1)
        log.info("  ┌─ SIGNAL ────────────────────────────────────────────────")
        log.info(f"  │  {s['direction']}  XAU/USD @ {s['entry']:.2f}")
        log.info(f"  │  Stop      : {s['sl']:.2f}  ({abs(s['sl']-s['entry']):.2f} pts)")
        log.info(f"  │  TP1 (30%) : {s['tp1']:.2f}  ({s['tp_r']:.2f}R)  — {units_tp1} units")
        log.info(f"  │  TP2 (70%) : {s['tp2']:.2f}  — {units_runner} units runner")
        log.info(f"  │  Risk      : {s['risk_pct']*100:.0f}%  (${s['risk_pct']*balance:,.0f})")
        log.info(f"  │  Flags     : {flags_str}")
        log.info(f"  │  Trail     : {be_note}")
        log.info(f"  │  RSI       : {s['rsi']}  |  ATR: {s['atr']:.2f}")
        log.info(f"  │  NY range  : {s['ny_low']:.2f}–{s['ny_high']:.2f}  ({s['ny_rng']:.2f} pts)")
        log.info("  └─────────────────────────────────────────────────────────")

        total_units = units_tp1 + units_runner
        tp1_trade_id, runner_trade_id = place_orders(
            client, s["direction"], total_units, s["sl"], s["tp1"], s["tp2"]
        )

        if not PAPER_MODE and not runner_trade_id:
            log.error("  Order placement incomplete — state not saved, manual review required")
            return

        # Track partial trade state so check_tp1_hit can activate runner
        state["open_partial_trade"] = {
            "tp1_trade_id"     : tp1_trade_id,
            "runner_trade_id"  : runner_trade_id,
            "entry"            : s["entry"],
            "risk"             : s["risk"],
            "tp2"              : s["tp2"],
            "direction"        : s["direction"],
            "runner_activated" : False,
        }

        # Track flagged trades separately for BE management
        if s["flagged"]:
            state["open_flagged_trade"] = {
                "trade_id" : runner_trade_id,
                "entry"    : s["entry"],
                "risk"     : s["risk"],
                "direction": s["direction"],
                "be_moved" : False,
            }

        save_trade({
            "timestamp"       : str(s["ts"]),
            "direction"       : s["direction"],
            "instrument"      : INSTRUMENT,
            "entry"           : s["entry"],
            "sl"              : round(s["sl"], 2),
            "tp1"             : round(s["tp1"], 2),
            "tp2"             : round(s["tp2"], 2),
            "tp1_r"           : s["tp_r"],
            "units_tp1"       : units_tp1,
            "units_runner"    : units_runner,
            "tp1_trade_id"    : tp1_trade_id,
            "runner_trade_id" : runner_trade_id,
            "risk_pct"        : s["risk_pct"],
            "risk_usd"        : round(s["risk_pct"] * balance, 2),
            "red_flags"       : s["flags"],
            "flagged"         : s["flagged"],
            "trailing"        : "BE@1R" if s["flagged"] else "none",
            "rsi"             : s["rsi"],
            "atr"             : s["atr"],
            "ny_high"         : s["ny_high"],
            "ny_low"          : s["ny_low"],
            "ny_rng"          : s["ny_rng"],
            "paper_mode"      : PAPER_MODE,
            "result"          : "OPEN",
        })
        log.info(f"  Trade saved → {TRADE_LOG}")
    else:
        log.info("  No signal this hour.")

    save_state(state)
    log.info("  Done.")


if __name__ == "__main__":
    main()

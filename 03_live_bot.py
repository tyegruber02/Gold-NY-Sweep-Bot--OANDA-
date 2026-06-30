"""
Congressional Trade Mirroring — Live Bot (Alpaca Paper Trading)

Mirrors the exact strategy from 01_backtest.py in real time:
  - Quarterly re-rank of congress members using Quiver Quantitative data
  - On each run, check for new disclosures from qualified members and place orders
  - Park idle cash in SPY when holdings fall below 80% of portfolio
  - All signals from public disclosure dates only

Run on a schedule (e.g. daily cron):
    QUIVER_API_KEY=<key> ALPACA_API_KEY=<key> ALPACA_SECRET_KEY=<key> python3 03_live_bot.py

Keys must be set as environment variables — never hardcoded.
Retrieve from macOS Keychain:
    QUIVER_API_KEY=$(security find-generic-password -s "QUIVER_API_KEY" -w) \
    ALPACA_API_KEY=$(security find-generic-password -s "ALPACA_API_KEY" -w) \
    ALPACA_SECRET_KEY=$(security find-generic-password -s "ALPACA_SECRET_KEY" -w) \
    python3 03_live_bot.py
"""

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, date

import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

load_dotenv()  # loads .env file from project directory if present

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ---------------------------------------------------------------------------
# CONFIG — mirrors 01_backtest.py
# ---------------------------------------------------------------------------

LOOKBACK_MONTHS = 12
FORWARD_RETURN_DAYS = 90
RERANK_INTERVAL_MONTHS = 3

MIN_PRICED_BUYS = 3
MIN_WIN_RATE = 0.65
MIN_AVG_RETURN = 0.0
MAX_TRADES_IN_WINDOW = 500
TOP_N_MEMBERS = 5
RECENCY_STALE_DAYS = 30
RECENCY_DECAY_DAYS = 45
RECENCY_DECAY_FACTOR = 0.5
MIN_AMOUNT = 5_000
MAX_LATE_FILING_DAYS = 365
CONVICTION_MIN_HOLD_DAYS = 30

POSITION_SIZE_PCT = 0.10
MAX_POSITION_SIZE_PCT = 0.25
CONVICTION_MULTIPLIER = 3.0

AMOUNT_MULTIPLIERS = [
    (15_000,       1.00),
    (50_000,       1.00),
    (100_000,      1.00),
    (250_000,      1.25),
    (1_000_000,    1.50),
    (float('inf'), 2.00),
]

MAX_BUYS_PER_MEMBER_PER_DAY = 3
SPY_FLOOR_PCT = 0.80

STATE_FILE = "bot_state.json"
TRADES_FILE = "quiver_trades.json"

QUIVER_ENDPOINT = "https://api.quiverquant.com/beta/bulk/congresstrading"

# ---------------------------------------------------------------------------
# ENV KEYS
# ---------------------------------------------------------------------------

QUIVER_API_KEY = os.environ.get("QUIVER_API_KEY")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

if not QUIVER_API_KEY:
    sys.exit("Error: QUIVER_API_KEY environment variable not set.")
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    sys.exit("Error: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.")

# ---------------------------------------------------------------------------
# NAME NORMALIZATION
# ---------------------------------------------------------------------------

_STRIP_TITLES = re.compile(
    r'\b(hon\.?|dr\.?|mr\.?|mrs\.?|ms\.?|jr\.?|sr\.?|ii|iii|iv|esq\.?)\b',
    re.IGNORECASE
)

def normalize_name(name: str) -> str:
    name = _STRIP_TITLES.sub('', name)
    name = re.sub(r'[^a-z\s]', '', name.lower())
    return ' '.join(name.split())

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "last_rerank_date": None,
            "active_members": {},
            "seen_disclosure_ids": [],
            "position_origins": {},     # ticker -> senator who originated the position
        }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# QUIVER DATA
# ---------------------------------------------------------------------------

def fetch_quiver_trades():
    print("Fetching latest congressional disclosures from Quiver...")
    resp = requests.get(QUIVER_ENDPOINT,
                        headers={"Authorization": f"Bearer {QUIVER_API_KEY}"})
    resp.raise_for_status()
    raw = resp.json()
    print(f"  Received {len(raw)} total records")

    TYPE_MAP = {
        "Purchase": "Buy",
        "Sale": "Sell",
        "Sale (Full)": "Sell",
        "Sale (Partial)": "Sell",
        "Exchange": None,
    }

    trades = []
    for r in raw:
        tx_type = TYPE_MAP.get(r.get("Transaction"))
        if tx_type is None:
            continue
        ticker = r.get("Ticker", "").strip()
        if not ticker or r.get("TickerType") not in ("ST", "Stock", None):
            continue
        try:
            tx_date = datetime.fromisoformat(r["Traded"]).date()
        except (KeyError, TypeError, ValueError):
            continue

        filed_raw = r.get("Filed")
        try:
            filed_date = datetime.fromisoformat(filed_raw).date() if filed_raw else None
        except ValueError:
            filed_date = None

        disclosure_date = filed_date or (tx_date + timedelta(days=45))

        # Skip extremely late filings
        if (datetime.combine(disclosure_date, datetime.min.time()) -
                datetime.combine(tx_date, datetime.min.time())).days > MAX_LATE_FILING_DAYS:
            continue

        trades.append({
            "senator": normalize_name(r.get("Name", "unknown")),
            "ticker": ticker,
            "transaction_date": tx_date,
            "disclosure_date": disclosure_date,
            "type": tx_type,
            "amount": r.get("Trade_Size_USD"),
            "filed_date": filed_date,
        })

    print(f"  Kept {len(trades)} trades after type/ticker filtering")
    return trades

# ---------------------------------------------------------------------------
# AMOUNT MULTIPLIER
# ---------------------------------------------------------------------------

def amount_multiplier(amount):
    if not amount:
        return 1.0
    try:
        amt = float(amount)
    except (TypeError, ValueError):
        return 1.0
    for threshold, mult in AMOUNT_MULTIPLIERS:
        if amt < threshold:
            return mult
    return AMOUNT_MULTIPLIERS[-1][1]

# ---------------------------------------------------------------------------
# MEMBER RANKING  (same logic as backtest)
# ---------------------------------------------------------------------------

def recency_weight(trade_date, window_start, window_end):
    span = (window_end - window_start).days or 1
    days_in = (trade_date - window_start).days
    return 0.5 + 0.5 * (days_in / span)


def fetch_price(ticker, as_of_date):
    """
    Fetch the most relevant closing price for as_of_date.
    Looks back up to 5 trading days (handles weekends/holidays) and
    forward up to 10 days (handles historical ranking lookups).
    Suppresses yfinance stderr to avoid dot-ticker noise (BRK.B etc).
    """
    try:
        import yfinance as yf
        import pandas as pd
        import warnings
        start = as_of_date - timedelta(days=7)
        end = as_of_date + timedelta(days=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hist = yf.download(ticker,
                               start=start.strftime("%Y-%m-%d"),
                               end=end.strftime("%Y-%m-%d"),
                               progress=False, auto_adjust=True)
        if hist.empty:
            return None
        close = hist["Close"] if isinstance(hist["Close"], pd.Series) else hist["Close"].squeeze()
        # Prefer the most recent close on or before as_of_date (handles weekends)
        past = close[close.index <= pd.Timestamp(as_of_date)]
        if not past.empty:
            return float(past.iloc[-1])
        # Fall back to next available trading day (for historical forward-return lookups)
        future = close[close.index > pd.Timestamp(as_of_date)]
        return float(future.iloc[0]) if not future.empty else None
    except Exception:
        return None


def rank_members(trades, as_of_date):
    """Re-rank all members as of as_of_date using the same logic as the backtest."""
    as_of_dt = datetime.combine(as_of_date, datetime.min.time())
    window_end_dt = as_of_dt - timedelta(days=FORWARD_RETURN_DAYS)
    window_start_dt = window_end_dt - relativedelta(months=LOOKBACK_MONTHS)

    latest_trade = defaultdict(lambda: datetime.min)
    for t in trades:
        disc_dt = datetime.combine(t["disclosure_date"], datetime.min.time())
        if disc_dt <= as_of_dt:
            latest_trade[t["senator"]] = max(latest_trade[t["senator"]], disc_dt)

    window_trades = [
        t for t in trades
        if window_start_dt <= datetime.combine(t["disclosure_date"], datetime.min.time()) <= window_end_dt
    ]

    by_member = defaultdict(list)
    for t in window_trades:
        by_member[t["senator"]].append(t)

    qualified = {m: ts for m, ts in by_member.items()
                 if len(ts) <= MAX_TRADES_IN_WINDOW}

    print(f"\n  Ranking {len(qualified)} members with trades in lookback window...")
    results = []
    for member, ts in qualified.items():
        weighted_returns = []
        raw_returns = []

        for t in ts:
            if t["type"] != "Buy":
                continue
            entry_date = t["disclosure_date"]
            exit_date = entry_date + timedelta(days=FORWARD_RETURN_DAYS)
            entry_price = fetch_price(t["ticker"], entry_date)
            exit_price = fetch_price(t["ticker"], exit_date)
            if entry_price and exit_price and entry_price > 0:
                ret = (exit_price - entry_price) / entry_price
                weight = recency_weight(
                    datetime.combine(t["transaction_date"], datetime.min.time()),
                    window_start_dt, window_end_dt
                )
                weighted_returns.append((ret, weight))
                raw_returns.append(ret)

        if len(raw_returns) < MIN_PRICED_BUYS:
            continue

        win_rate = sum(1 for r in raw_returns if r > 0) / len(raw_returns)
        if win_rate < MIN_WIN_RATE:
            continue

        total_weight = sum(w for _, w in weighted_returns)
        avg_return = sum(r * w for r, w in weighted_returns) / total_weight
        if avg_return < MIN_AVG_RETURN:
            continue

        days_since_last = (as_of_dt - latest_trade[member]).days
        if days_since_last > RECENCY_STALE_DAYS:
            continue
        if days_since_last > RECENCY_DECAY_DAYS:
            avg_return *= RECENCY_DECAY_FACTOR

        results.append({
            "senator": member,
            "num_priced_trades": len(raw_returns),
            "win_rate": win_rate,
            "avg_return_per_trade": avg_return,
            "days_since_last_trade": days_since_last,
        })

    results.sort(key=lambda r: -r["avg_return_per_trade"])
    top = results[:TOP_N_MEMBERS]

    if top:
        max_score = top[0]["avg_return_per_trade"]
        min_score = top[-1]["avg_return_per_trade"]
        score_range = max(max_score - min_score, 0.001)
        for r in top:
            normalized = (r["avg_return_per_trade"] - min_score) / score_range
            size = POSITION_SIZE_PCT * (1 + normalized * CONVICTION_MULTIPLIER)
            r["position_size"] = min(size, MAX_POSITION_SIZE_PCT)

    print(f"  {len(top)} members qualified:")
    for r in top:
        print(f"    {r['senator']:<38} avg {r['avg_return_per_trade']:>7.2%} "
              f"| wr {r['win_rate']:.0%} | {r['num_priced_trades']} trades "
              f"| pos {r['position_size']*100:.1f}%")

    return {r["senator"]: r["position_size"] for r in top}

# ---------------------------------------------------------------------------
# ALPACA HELPERS
# ---------------------------------------------------------------------------

def get_portfolio(client):
    account = client.get_account()
    equity = float(account.equity)
    cash = float(account.cash)
    positions = {p.symbol: p for p in client.get_all_positions()}
    holdings_value = sum(float(p.market_value) for p in positions.values())
    return equity, cash, holdings_value, positions


def place_buy(client, ticker, dollar_amount, dry_run=False):
    price = fetch_price(ticker, date.today())
    if not price or price <= 0:
        print(f"  [SKIP] Could not fetch price for {ticker}")
        return False
    shares = round(dollar_amount / price, 6)
    if shares <= 0:
        return False
    print(f"  {'[DRY RUN] ' if dry_run else ''}BUY  {ticker:6s} — ${dollar_amount:,.0f} "
          f"≈ {shares:.4f} shares @ ${price:.2f}")
    if not dry_run:
        try:
            req = MarketOrderRequest(
                symbol=ticker,
                notional=round(dollar_amount, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = client.submit_order(req)
            print(f"    Order submitted: {order.id} | status: {order.status}")
        except Exception as e:
            print(f"  [ERROR] Order failed for {ticker}: {e}")
            return False
    return True


def place_sell(client, ticker, dry_run=False):
    positions = {p.symbol: p for p in client.get_all_positions()}
    if ticker not in positions:
        print(f"  [SKIP] No position in {ticker} to sell")
        return False
    qty = float(positions[ticker].qty)
    market_val = float(positions[ticker].market_value)
    print(f"  {'[DRY RUN] ' if dry_run else ''}SELL {ticker:6s} — {qty:.4f} shares ≈ ${market_val:,.0f}")
    if not dry_run:
        try:
            req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = client.submit_order(req)
            print(f"    Order submitted: {order.id} | status: {order.status}")
        except Exception as e:
            print(f"  [ERROR] Order failed for {ticker}: {e}")
            return False
    return True


def park_in_spy(client, equity, cash, holdings_value, dry_run=False):
    """Park idle cash in SPY when holdings fall below the floor threshold."""
    shortfall = (equity * SPY_FLOOR_PCT) - holdings_value
    if shortfall <= 100 or cash <= 100:
        return
    invest = min(shortfall, cash * 0.95)
    print(f"\n  SPY floor: deploying ${invest:,.0f} idle cash into SPY")
    place_buy(client, "SPY", invest, dry_run=dry_run)


def liquidate_spy_for_buy(client, amount_needed, cash, dry_run=False):
    """Sell just enough SPY to cover a congress buy when free cash is short."""
    positions = {p.symbol: p for p in client.get_all_positions()}
    if "SPY" not in positions:
        return cash
    shortfall = amount_needed - cash
    if shortfall <= 0:
        return cash
    spy_value = float(positions["SPY"].market_value)
    liquidate = min(shortfall * 1.05, spy_value)  # 5% buffer for price movement
    print(f"  Liquidating ${liquidate:,.0f} of SPY to fund congress buy")
    if not dry_run:
        try:
            req = MarketOrderRequest(
                symbol="SPY",
                notional=round(liquidate, 2),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            client.submit_order(req)
        except Exception as e:
            print(f"  [ERROR] SPY liquidation failed: {e}")
            return cash
    return cash + liquidate



# ---------------------------------------------------------------------------
# DISCLOSURE ID
# ---------------------------------------------------------------------------

def disclosure_id(t):
    """Stable identifier for a disclosure so we don't act on it twice."""
    return f"{t['senator']}|{t['ticker']}|{t['type']}|{t['transaction_date']}"

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    dry_run = "--dry-run" in sys.argv
    today = date.today()

    print("=" * 70)
    print(f"CONGRESSIONAL TRADE MIRRORING — LIVE BOT  ({today}){'  [DRY RUN]' if dry_run else ''}")
    print("=" * 70)

    state = load_state()
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

    clock = client.get_clock()
    market_open = dry_run or clock.is_open
    if not market_open:
        print(f"\n  Market is closed (next open: {clock.next_open}). "
              f"Data/rankings will update but no orders will be placed.")

    # ── 1. Fetch latest trade data ────────────────────────────────────────
    trades = fetch_quiver_trades()

    # Cache fetched trades locally for use in ranking
    with open(TRADES_FILE, "w") as f:
        json.dump([{**t,
                    "transaction_date": t["transaction_date"].isoformat(),
                    "disclosure_date": t["disclosure_date"].isoformat(),
                    "filed_date": t["filed_date"].isoformat() if t["filed_date"] else None}
                   for t in trades], f)

    # ── 2. Re-rank if due ────────────────────────────────────────────────
    last_rerank = (datetime.fromisoformat(state["last_rerank_date"]).date()
                   if state["last_rerank_date"] else None)

    rerank_due = (last_rerank is None or
                  today >= last_rerank + relativedelta(months=RERANK_INTERVAL_MONTHS))

    if rerank_due:
        print(f"\nRe-ranking members (last: {last_rerank or 'never'})...")
        active_members = rank_members(trades, today)
        # Close positions originated by members who just dropped off
        prev_members = set(state.get("active_members", {}).keys())
        dropped = prev_members - set(active_members.keys())
        if dropped:
            print(f"\n  Dropped members: {', '.join(sorted(dropped))}")
            position_origins = state.get("position_origins", {})
            alpaca_positions = {p.symbol: p for p in client.get_all_positions()}
            for ticker, origin in list(position_origins.items()):
                if origin in dropped and ticker in alpaca_positions:
                    print(f"  Closing {ticker} (originated by dropped member {origin})")
                    place_sell(client, ticker, dry_run=dry_run)
                    del position_origins[ticker]
            state["position_origins"] = position_origins

        state["active_members"] = active_members
        state["last_rerank_date"] = today.isoformat()
        print(f"\n  Active member set updated: {len(active_members)} members")
    else:
        active_members = state["active_members"]
        next_rerank = last_rerank + relativedelta(months=RERANK_INTERVAL_MONTHS)
        print(f"\n  Using existing active member set ({len(active_members)} members). "
              f"Next re-rank: {next_rerank}")

    if not active_members:
        print("\n  No qualifying members — nothing to trade.")
        save_state(state)
        return

    print(f"\n  Active members: {', '.join(sorted(active_members))}")

    # ── 3. Build sell lookup for flip filter ────────────────────────────
    sells_by_member = defaultdict(list)
    for t in trades:
        if t["type"] == "Sell":
            sells_by_member[(t["senator"], t["ticker"])].append(t["transaction_date"])

    # ── 4. Find new actionable disclosures ───────────────────────────────
    seen = set(state.get("seen_disclosure_ids", []))
    daily_buys = defaultdict(int)

    equity, cash, holdings_value, positions = get_portfolio(client)
    print(f"\n  Portfolio: equity ${equity:,.0f} | cash ${cash:,.0f} | "
          f"holdings ${holdings_value:,.0f}")

    new_trades = [
        t for t in trades
        if t["senator"] in active_members
        and disclosure_id(t) not in seen
        and t["disclosure_date"] <= today
        and t["disclosure_date"] >= today - timedelta(days=7)  # only act on recent filings
    ]

    print(f"\n  New actionable disclosures from active members: {len(new_trades)}")

    if not market_open:
        print("  Market closed — skipping order placement.")
    for t in sorted(new_trades if market_open else [], key=lambda x: x["disclosure_date"]):
        did = disclosure_id(t)

        if t["type"] == "Buy":
            # Skip below min amount
            amt = t.get("amount")
            if amt is not None:
                try:
                    if float(amt) < MIN_AMOUNT:
                        print(f"  [SKIP] {t['ticker']} — amount below ${MIN_AMOUNT:,}")
                        seen.add(did)
                        continue
                except (TypeError, ValueError):
                    pass

            # Flip filter: skip if member sold this ticker recently
            key = (t["senator"], t["ticker"])
            if any(0 < (t["transaction_date"] - s).days <= CONVICTION_MIN_HOLD_DAYS
                   for s in sells_by_member.get(key, [])):
                print(f"  [SKIP] {t['ticker']} — flip trade (sold within {CONVICTION_MIN_HOLD_DAYS}d)")
                seen.add(did)
                continue

            # Daily cap
            day_key = (t["senator"], t["disclosure_date"])
            if daily_buys[day_key] >= MAX_BUYS_PER_MEMBER_PER_DAY:
                print(f"  [SKIP] {t['ticker']} — daily buy cap reached for {t['senator']}")
                seen.add(did)
                continue
            daily_buys[day_key] += 1

            member_size_pct = active_members[t["senator"]]
            amt_mult = amount_multiplier(t.get("amount"))
            final_size_pct = min(member_size_pct * amt_mult, MAX_POSITION_SIZE_PCT)
            dollar_size = equity * final_size_pct

            # Liquidate SPY if cash is short — never skip a qualified signal
            if dollar_size > cash:
                cash = liquidate_spy_for_buy(client, dollar_size, cash, dry_run=dry_run)

            if dollar_size > cash:
                print(f"  [SKIP] {t['ticker']} — insufficient cash even after SPY liquidation")
                seen.add(did)
                continue

            success = place_buy(client, t["ticker"], dollar_size, dry_run=dry_run)
            if success:
                cash -= dollar_size
                holdings_value += dollar_size
                state.setdefault("position_origins", {})[t["ticker"]] = t["senator"]

        elif t["type"] == "Sell":
            if t["ticker"] in {p for p in positions}:
                success = place_sell(client, t["ticker"], dry_run=dry_run)
                if success:
                    val = float(positions[t["ticker"]].market_value)
                    cash += val
                    holdings_value -= val
                    state.get("position_origins", {}).pop(t["ticker"], None)

        seen.add(did)

    # ── 5. Park remaining idle cash in SPY ──────────────────────────────
    equity, cash, holdings_value, _ = get_portfolio(client)
    park_in_spy(client, equity, cash, holdings_value, dry_run=dry_run)

    # ── 6. Save state ────────────────────────────────────────────────────
    state["seen_disclosure_ids"] = list(seen)
    save_state(state)

    print(f"\n{'=' * 70}")
    print("RUN COMPLETE")
    print(f"  Active members:  {len(active_members)}")
    print(f"  New signals:     {len(new_trades)}")
    equity, cash, holdings_value, positions = get_portfolio(client)
    print(f"  Portfolio now:   equity ${equity:,.0f} | cash ${cash:,.0f} | "
          f"positions {len(positions)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

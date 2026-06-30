"""
Congressional Trade Mirroring — Backtest

Strategy:
  - Every quarter, re-rank all members on trailing 12-month win rate and avg
    90-day forward return (only trades where the full return window has elapsed).
  - Mirror the top 20 qualified members. Position size is rank-weighted and
    scaled by the disclosed dollar amount as a conviction signal.
  - Park idle cash in SPY whenever holdings fall below 80% of portfolio.
  - All signals derived from publicly filed disclosure dates only — no lookahead.

Run: python3 01_backtest.py
"""

import json
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import defaultdict

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

TRADES_FILE = "quiver_trades.json"

RERANK_INTERVAL_MONTHS = 3
LOOKBACK_MONTHS = 12
FORWARD_RETURN_DAYS = 90

MIN_PRICED_BUYS = 3
MIN_WIN_RATE = 0.65
MIN_AVG_RETURN = 0.0
MAX_TRADES_IN_WINDOW = 500
TOP_N_MEMBERS = 5
RECENCY_STALE_DAYS = 30    # hard drop: remove member if no trade in 30 days
RECENCY_DECAY_DAYS = 45    # soft decay: halve score if no trade in 45 days
RECENCY_DECAY_FACTOR = 0.5
MIN_AMOUNT = 5_000         # skip buys with disclosed amount below this

MAX_LATE_FILING_DAYS = 365

POSITION_SIZE_PCT = 0.10      # base size for lowest-ranked active member
MAX_POSITION_SIZE_PCT = 0.25  # hard cap per position
CONVICTION_MULTIPLIER = 3.0   # top member gets base × (1 + CONVICTION_MULTIPLIER)

AMOUNT_MULTIPLIERS = [
    (15_000,        1.00),
    (50_000,        1.00),
    (100_000,       1.00),
    (250_000,       1.25),
    (1_000_000,     1.50),
    (float('inf'),  2.00),
]

MAX_BUYS_PER_MEMBER_PER_DAY = 3
SPY_FLOOR_PCT = 0.80
CONVICTION_MIN_HOLD_DAYS = 30

STARTING_CASH = 100_000
SIMULATION_START = datetime(2023, 1, 1)
SIMULATION_END = datetime(2025, 12, 31)

PRICE_CACHE_FILE = "price_cache.json"


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
# LOAD TRADES
# ---------------------------------------------------------------------------

def load_trades():
    with open(TRADES_FILE) as f:
        raw = json.load(f)

    for r in raw:
        r["senator"] = normalize_name(r.get("senator", "unknown"))

    sells = defaultdict(list)
    for r in raw:
        if r.get("type") == "Sell":
            try:
                sells[(r["senator"], r["ticker"])].append(
                    datetime.fromisoformat(r["transaction_date"])
                )
            except (KeyError, ValueError):
                pass

    kept = []
    skipped_flip = skipped_late = 0

    for r in raw:
        try:
            tx_date = datetime.fromisoformat(r["transaction_date"])
        except (KeyError, ValueError):
            continue

        filed = r.get("filed_date")
        if filed:
            try:
                disclosure_date = datetime.fromisoformat(filed)
            except ValueError:
                disclosure_date = tx_date + timedelta(days=45)
        else:
            disclosure_date = tx_date + timedelta(days=45)

        if (disclosure_date - tx_date).days > MAX_LATE_FILING_DAYS:
            skipped_late += 1
            continue

        if r.get("type") == "Buy":
            amt = r.get("amount")
            if amt is not None:
                try:
                    if float(amt) < MIN_AMOUNT:
                        continue
                except (TypeError, ValueError):
                    pass

        if r.get("type") == "Buy":
            key = (r["senator"], r["ticker"])
            if any(0 < (s - tx_date).days <= CONVICTION_MIN_HOLD_DAYS
                   for s in sells.get(key, [])):
                skipped_flip += 1
                continue

        r["transaction_date"] = tx_date
        r["disclosure_date"] = disclosure_date
        kept.append(r)

    print(f"  Skipped {skipped_flip} flip trades (buy+sell within {CONVICTION_MIN_HOLD_DAYS} days)")
    print(f"  Skipped {skipped_late} late-filed trades (filed > {MAX_LATE_FILING_DAYS} days after transaction)")
    return kept


# ---------------------------------------------------------------------------
# PRICE CACHE
# ---------------------------------------------------------------------------

class PriceCache:
    def __init__(self, path):
        self.path = path
        try:
            with open(path) as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            self.cache = {}

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.cache, f)

    def preload(self, ticker_dates: dict):
        tickers_to_fetch = [t for t in ticker_dates if not self._all_cached(t, ticker_dates[t])]
        if not tickers_to_fetch:
            print("  All prices already cached.")
            return

        print(f"  Batch-downloading price history for {len(tickers_to_fetch)} tickers...")
        all_dates = [d for dates in ticker_dates.values() for d in dates]
        range_start = min(all_dates) - timedelta(days=5)
        range_end = max(all_dates) + timedelta(days=10)

        try:
            hist = yf.download(
                tickers_to_fetch,
                start=range_start.strftime("%Y-%m-%d"),
                end=range_end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            print(f"  Batch download failed: {e}")
            return

        if hist.empty:
            return

        close = hist["Close"] if "Close" in hist.columns else hist.xs("Close", axis=1, level=0)
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers_to_fetch[0])

        for ticker in tickers_to_fetch:
            if ticker not in close.columns:
                for date in ticker_dates[ticker]:
                    self.cache[self._key(ticker, date)] = None
                continue
            series = close[ticker].dropna()
            for date in ticker_dates[ticker]:
                key = self._key(ticker, date)
                if key in self.cache:
                    continue
                future = series[series.index >= pd.Timestamp(date)]
                self.cache[key] = float(future.iloc[0]) if not future.empty else None

        print(f"  Done. Cache now has {len(self.cache)} entries.")

    def _all_cached(self, ticker, dates):
        return all(self._key(ticker, d) in self.cache for d in dates)

    def _key(self, ticker, date):
        return f"{ticker}_{date.date().isoformat()}"

    def get_price(self, ticker, date):
        key = self._key(ticker, date)
        if key in self.cache:
            return self.cache[key]

        start = date - timedelta(days=2)
        end = date + timedelta(days=7)
        try:
            hist = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                               end=end.strftime("%Y-%m-%d"), progress=False,
                               auto_adjust=True)
        except Exception:
            self.cache[key] = None
            return None

        if hist.empty:
            self.cache[key] = None
            return None

        close_col = hist["Close"] if isinstance(hist["Close"], pd.Series) \
            else hist["Close"].squeeze()
        future = close_col[close_col.index >= pd.Timestamp(date)]
        if future.empty:
            self.cache[key] = None
            return None

        price = float(future.iloc[0])
        self.cache[key] = price
        return price


# ---------------------------------------------------------------------------
# MEMBER RANKING
# ---------------------------------------------------------------------------

def recency_weight(trade_date, window_start, window_end):
    span = (window_end - window_start).days or 1
    days_in = (trade_date - window_start).days
    return 0.5 + 0.5 * (days_in / span)


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


def rank_members_at_date(trades, cache, as_of_date):
    """
    Rank members using the 12-month lookback window where the full
    FORWARD_RETURN_DAYS have already elapsed. Returns top N with position sizes.
    """
    window_end = as_of_date - timedelta(days=FORWARD_RETURN_DAYS)
    window_start = window_end - relativedelta(months=LOOKBACK_MONTHS)

    latest_trade = defaultdict(lambda: datetime.min)
    for t in trades:
        if t["disclosure_date"] <= as_of_date:
            latest_trade[t["senator"]] = max(latest_trade[t["senator"]], t["disclosure_date"])

    window_trades = [
        t for t in trades
        if window_start <= t["disclosure_date"] <= window_end
    ]

    by_member = defaultdict(list)
    for t in window_trades:
        by_member[t["senator"]].append(t)

    qualified = {
        m: ts for m, ts in by_member.items()
        if MAX_TRADES_IN_WINDOW is None or len(ts) <= MAX_TRADES_IN_WINDOW
    }

    results = []
    for member, ts in qualified.items():
        weighted_returns = []
        raw_returns = []

        for t in ts:
            if t["type"] != "Buy":
                continue

            entry_date = t["disclosure_date"]
            exit_date = entry_date + timedelta(days=FORWARD_RETURN_DAYS)
            entry_price = cache.get_price(t["ticker"], entry_date)
            exit_price = cache.get_price(t["ticker"], exit_date)

            if entry_price and exit_price and entry_price > 0:
                ret = (exit_price - entry_price) / entry_price
                weight = recency_weight(t["transaction_date"], window_start, window_end)
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

        days_since_last = (as_of_date - latest_trade[member]).days
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

    return top


# ---------------------------------------------------------------------------
# SIMULATION
# ---------------------------------------------------------------------------

def simulate(trades, cache):
    rerank_dates = []
    d = SIMULATION_START
    while d <= SIMULATION_END:
        rerank_dates.append(d)
        d += relativedelta(months=RERANK_INTERVAL_MONTHS)

    all_signals = [
        t for t in trades
        if SIMULATION_START <= t["disclosure_date"] <= SIMULATION_END
    ]
    all_signals.sort(key=lambda t: t["disclosure_date"])

    cash = STARTING_CASH
    positions = {}
    trade_log = []
    rerank_log = []

    active_members = {}
    next_rerank_idx = 0
    daily_buy_counts = defaultdict(int)
    position_origins = {}  # ticker -> senator who originated the position

    def liquidate_spy_for_buy(amount_needed, as_of_date):
        """Sell just enough SPY to cover amount_needed. Returns cash freed."""
        nonlocal cash
        if "SPY" not in positions or cash >= amount_needed:
            return
        shortfall = amount_needed - cash
        spy_price = cache.get_price("SPY", as_of_date)
        if not spy_price:
            return
        shares_to_sell = min(shortfall / spy_price, positions["SPY"]["shares"])
        proceeds = shares_to_sell * spy_price
        positions["SPY"]["shares"] -= shares_to_sell
        if positions["SPY"]["shares"] <= 0:
            del positions["SPY"]
        cash += proceeds
        trade_log.append({
            "date": as_of_date.date().isoformat(),
            "member": "SPY_LIQUIDATE",
            "action": "SELL",
            "ticker": "SPY",
            "price": round(spy_price, 2),
            "dollar_amount": round(proceeds, 2),
        })

    def park_idle_cash_in_spy(as_of_date):
        nonlocal cash
        holdings_value = sum(
            pos["shares"] * pos["entry_price"]
            for pos in positions.values()
        )
        total = cash + holdings_value
        shortfall = (total * SPY_FLOOR_PCT) - holdings_value
        if shortfall <= 100 or cash <= 0:
            return
        invest = min(shortfall, cash)
        spy_price = cache.get_price("SPY", as_of_date)
        if not spy_price:
            return
        shares = invest / spy_price
        cash -= invest
        if "SPY" in positions:
            old = positions["SPY"]
            positions["SPY"] = {"shares": old["shares"] + shares, "entry_price": spy_price}
        else:
            positions["SPY"] = {"shares": shares, "entry_price": spy_price}
        trade_log.append({
            "date": as_of_date.date().isoformat(),
            "member": "SPY_PARK",
            "action": "BUY",
            "ticker": "SPY",
            "price": round(spy_price, 2),
            "dollar_amount": round(invest, 2),
        })

    for sig in all_signals:
        sig_date = sig["disclosure_date"]

        while next_rerank_idx < len(rerank_dates) and sig_date >= rerank_dates[next_rerank_idx]:
            rerank_date = rerank_dates[next_rerank_idx]
            ranked = rank_members_at_date(trades, cache, rerank_date)

            new_active = {r["senator"]: r["position_size"] for r in ranked}
            added = set(new_active) - set(active_members)
            dropped = set(active_members) - set(new_active)
            active_members = new_active

            # Close positions originated by dropped members
            for ticker, origin in list(position_origins.items()):
                if origin in dropped and ticker in positions:
                    exit_price = cache.get_price(ticker, rerank_date)
                    if exit_price:
                        pos = positions.pop(ticker)
                        proceeds = pos["shares"] * exit_price
                        pnl = proceeds - (pos["shares"] * pos["entry_price"])
                        cash += proceeds
                        del position_origins[ticker]
                        trade_log.append({
                            "date": rerank_date.date().isoformat(),
                            "member": origin,
                            "action": "SELL",
                            "ticker": ticker,
                            "price": round(exit_price, 2),
                            "dollar_amount": round(proceeds, 2),
                            "pnl": round(pnl, 2),
                        })

            rerank_log.append({
                "date": rerank_date.date().isoformat(),
                "active": sorted(active_members),
                "added": sorted(added),
                "dropped": sorted(dropped),
                "rankings": ranked,
            })
            park_idle_cash_in_spy(rerank_date)
            next_rerank_idx += 1

        if sig["senator"] not in active_members:
            continue

        price = cache.get_price(sig["ticker"], sig_date)
        if not price:
            continue

        if sig["type"] == "Buy":
            day_key = (sig["senator"], sig_date.date())
            if daily_buy_counts[day_key] >= MAX_BUYS_PER_MEMBER_PER_DAY:
                continue
            daily_buy_counts[day_key] += 1

            member_size_pct = active_members[sig["senator"]]
            amt_mult = amount_multiplier(sig.get("amount"))
            final_size_pct = min(member_size_pct * amt_mult, MAX_POSITION_SIZE_PCT)

            position_dollar_size = cash * final_size_pct
            if position_dollar_size <= 0:
                continue
            # If cash is short, liquidate just enough SPY to cover
            if position_dollar_size > cash:
                liquidate_spy_for_buy(position_dollar_size, sig_date)
            position_dollar_size = min(position_dollar_size, cash)
            if position_dollar_size <= 0:
                continue

            shares = position_dollar_size / price
            cash -= position_dollar_size

            if sig["ticker"] in positions:
                old = positions[sig["ticker"]]
                positions[sig["ticker"]] = {
                    "shares": old["shares"] + shares,
                    "entry_price": price,
                }
            else:
                positions[sig["ticker"]] = {"shares": shares, "entry_price": price}
            position_origins[sig["ticker"]] = sig["senator"]

            trade_log.append({
                "date": sig_date.date().isoformat(),
                "member": sig["senator"],
                "action": "BUY",
                "ticker": sig["ticker"],
                "price": round(price, 2),
                "dollar_amount": round(position_dollar_size, 2),
                "position_size_pct": round(final_size_pct * 100, 1),
                "amt_mult": amt_mult,
            })

        elif sig["type"] == "Sell" and sig["ticker"] in positions:
            pos = positions.pop(sig["ticker"])
            proceeds = pos["shares"] * price
            cash += proceeds
            pnl = proceeds - (pos["shares"] * pos["entry_price"])
            trade_log.append({
                "date": sig_date.date().isoformat(),
                "member": sig["senator"],
                "action": "SELL",
                "ticker": sig["ticker"],
                "price": round(price, 2),
                "dollar_amount": round(proceeds, 2),
                "pnl": round(pnl, 2),
            })
            park_idle_cash_in_spy(sig_date)

    open_pos_summary = {}
    final_holdings = 0
    for ticker, pos in positions.items():
        final_price = cache.get_price(ticker, SIMULATION_END) or pos["entry_price"]
        market_value = pos["shares"] * final_price
        cost_basis = pos["shares"] * pos["entry_price"]
        final_holdings += market_value
        open_pos_summary[ticker] = {
            "market_value": round(market_value, 2),
            "unrealized_pnl": round(market_value - cost_basis, 2),
        }

    return {
        "trade_log": trade_log,
        "rerank_log": rerank_log,
        "final_cash": cash,
        "open_positions": open_pos_summary,
        "final_portfolio_value": cash + final_holdings,
    }


# ---------------------------------------------------------------------------
# BENCHMARK
# ---------------------------------------------------------------------------

def benchmark_spy():
    hist = yf.download("SPY",
                       start=SIMULATION_START.strftime("%Y-%m-%d"),
                       end=(SIMULATION_END + timedelta(days=5)).strftime("%Y-%m-%d"),
                       progress=False, auto_adjust=True)
    if hist.empty:
        return None
    close = hist["Close"] if isinstance(hist["Close"], pd.Series) else hist["Close"].squeeze()
    return (float(close.iloc[-1]) - float(close.iloc[0])) / float(close.iloc[0])


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CONGRESSIONAL TRADE MIRRORING — ROLLING RERANK BACKTEST")
    print("=" * 70)

    trades = load_trades()
    cache = PriceCache(PRICE_CACHE_FILE)

    print(f"\nLoaded {len(trades)} trades after filtering.")
    print("Pre-loading price data...")

    ticker_dates = defaultdict(list)
    for t in trades:
        disc = t["disclosure_date"]
        ticker_dates[t["ticker"]].append(disc)
        ticker_dates[t["ticker"]].append(disc + timedelta(days=FORWARD_RETURN_DAYS))

    spy_dates = [SIMULATION_START, SIMULATION_END]
    d = SIMULATION_START
    while d <= SIMULATION_END:
        spy_dates.append(d)
        d += relativedelta(months=RERANK_INTERVAL_MONTHS)
    for t in trades:
        if SIMULATION_START <= t["disclosure_date"] <= SIMULATION_END:
            spy_dates.append(t["disclosure_date"])
    ticker_dates["SPY"].extend(spy_dates)

    for date in ticker_dates["SPY"]:
        key = f"SPY_{date.date().isoformat()}"
        if key in cache.cache and cache.cache[key] is None:
            del cache.cache[key]

    cache.preload(dict(ticker_dates))
    cache.save()

    print("\nRunning simulation with rolling quarterly re-rank...")
    result = simulate(trades, cache)
    cache.save()

    print(f"\n{'=' * 70}")
    print("QUARTERLY RE-RANK LOG")
    print("=" * 70)
    for entry in result["rerank_log"]:
        print(f"\n  {entry['date']} — Active members ({len(entry['active'])}):")
        for r in entry["rankings"]:
            stale = f" ⚠ {r['days_since_last_trade']}d stale" if r['days_since_last_trade'] > RECENCY_DECAY_DAYS else ""
            print(f"    {r['senator']:<38} avg {r['avg_return_per_trade']:>7.2%} "
                  f"| wr {r['win_rate']:.0%} | {r['num_priced_trades']:>3} trades "
                  f"| pos {r['position_size']*100:.1f}%{stale}")
        if entry["added"]:
            print(f"    + Added:   {', '.join(entry['added'])}")
        if entry["dropped"]:
            print(f"    - Dropped: {', '.join(entry['dropped'])}")

    print(f"\n{'=' * 70}")
    print("TRADE LOG")
    print("=" * 70)
    for t in result["trade_log"]:
        pnl_str = f", P&L ${t['pnl']:,.0f}" if "pnl" in t else ""
        size_str = f" ({t['position_size_pct']}%)" if "position_size_pct" in t else ""
        amt_str = f" [amt×{t['amt_mult']}]" if t.get("amt_mult", 1.0) != 1.0 else ""
        print(f"  {t['date']} | {t['action']:4s} | {t['ticker']:6s} | "
              f"{t['member']:<38} | ${t['dollar_amount']:>10,.0f}{size_str}{amt_str}{pnl_str}")

    print(f"\n{'=' * 70}")
    print("OPEN POSITIONS (mark-to-market at simulation end)")
    print("=" * 70)
    for ticker, pos in sorted(result["open_positions"].items(),
                               key=lambda x: -x[1]["unrealized_pnl"]):
        sign = "+" if pos["unrealized_pnl"] >= 0 else ""
        print(f"  {ticker:6s} | ${pos['market_value']:>10,.0f} | "
              f"unrealized P&L: {sign}${pos['unrealized_pnl']:,.0f}")

    starting = STARTING_CASH
    ending = result["final_portfolio_value"]
    strategy_return = (ending - starting) / starting
    spy_return = benchmark_spy()

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print("=" * 70)
    print(f"  Starting value:           ${starting:>12,.2f}")
    print(f"  Ending value:             ${ending:>12,.2f}")
    print(f"    Cash:                   ${result['final_cash']:>12,.2f}")
    print(f"    Open positions (MTM):   ${ending - result['final_cash']:>12,.2f}")
    print(f"  Strategy return:          {strategy_return:>12.2%}")
    if spy_return is not None:
        print(f"  SPY buy-and-hold:         {spy_return:>12.2%}")
        print(f"  Strategy vs SPY:          {strategy_return - spy_return:>+12.2%}")
    print(f"  Trades executed:          {len(result['trade_log']):>12,}")
    print(f"  Open positions at end:    {len(result['open_positions']):>12,}")
    print(f"\n  Simulation: {SIMULATION_START.date()} → {SIMULATION_END.date()}")
    print(f"  Re-rank: every {RERANK_INTERVAL_MONTHS}mo | Lookback: {LOOKBACK_MONTHS}mo "
          f"| Forward return: {FORWARD_RETURN_DAYS}d")
    print(f"  Top N: {TOP_N_MEMBERS} | Min buys: {MIN_PRICED_BUYS} | Min win rate: {MIN_WIN_RATE:.0%} | "
          f"Min amount: ${MIN_AMOUNT:,.0f} | Max trades/window: {MAX_TRADES_IN_WINDOW} | SPY floor: {SPY_FLOOR_PCT:.0%}")


if __name__ == "__main__":
    main()

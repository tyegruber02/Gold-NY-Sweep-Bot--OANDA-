"""
Fetch congressional trading data from Quiver Quantitative API.

Usage:
    QUIVER_API_KEY=<key> python3 02_fetch_quiver.py
    (or retrieve key from macOS Keychain — see README)

Output:
    quiver_trades.json  — normalized trade records for 01_backtest.py
"""

import json
import os
import sys
from datetime import datetime

import requests

API_KEY = os.environ.get("QUIVER_API_KEY")
if not API_KEY:
    sys.exit("Error: QUIVER_API_KEY environment variable not set.")

ENDPOINT = "https://api.quiverquant.com/beta/bulk/congresstrading"

# Include both chambers, 2022–present
CHAMBER_FILTER = None   # None = both Senate and House
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime.now()

# Transaction type mapping from Quiver → our format
TYPE_MAP = {
    "Purchase": "Buy",
    "Sale": "Sell",
    "Sale (Full)": "Sell",
    "Sale (Partial)": "Sell",
    "Exchange": None,   # skip — not a directional trade
}


def fetch():
    print(f"Fetching congressional trades from Quiver Quantitative...")
    resp = requests.get(ENDPOINT, headers={"Authorization": f"Bearer {API_KEY}"})
    resp.raise_for_status()
    data = resp.json()
    print(f"  Received {len(data)} total records")
    return data


def normalize(data):
    out = []
    skipped_chamber = 0
    skipped_date = 0
    skipped_type = 0
    skipped_ticker = 0

    for r in data:
        # Filter by chamber
        if CHAMBER_FILTER and r.get("Chamber") != CHAMBER_FILTER:
            skipped_chamber += 1
            continue

        # Filter to date range
        try:
            traded = datetime.fromisoformat(r["Traded"])
        except (KeyError, TypeError, ValueError):
            skipped_date += 1
            continue
        if not (START_DATE <= traded <= END_DATE):
            skipped_date += 1
            continue

        # Map transaction type
        tx_type = TYPE_MAP.get(r.get("Transaction"))
        if tx_type is None:
            skipped_type += 1
            continue

        # Skip options, mutual funds, etc. — stock tickers only
        ticker = r.get("Ticker", "").strip()
        if not ticker or r.get("TickerType") not in ("ST", "Stock", None):
            skipped_ticker += 1
            continue

        out.append({
            "senator": r.get("Name", "Unknown"),
            "ticker": ticker,
            "transaction_date": traded.date().isoformat(),
            "filed_date": r.get("Filed"),
            "type": tx_type,
            "amount": r.get("Trade_Size_USD"),
            "party": r.get("Party"),
            "state": r.get("State"),
            "chamber": r.get("Chamber"),
        })

    print(f"  Kept:             {len(out)}")
    print(f"  Skipped (chamber): {skipped_chamber}")
    print(f"  Skipped (date):    {skipped_date}")
    print(f"  Skipped (type):    {skipped_type}")
    print(f"  Skipped (ticker):  {skipped_ticker}")
    return out


def main():
    data = fetch()
    trades = normalize(data)

    if not trades:
        sys.exit("No trades after filtering — check filters or API response.")

    dates = [t["transaction_date"] for t in trades]
    print(f"\nDate range in output: {min(dates)} to {max(dates)}")
    senators = set(t["senator"] for t in trades)
    print(f"Unique senators: {len(senators)}")

    out_path = "quiver_trades.json"
    with open(out_path, "w") as f:
        json.dump(trades, f, indent=2)
    print(f"\nSaved {len(trades)} trades to {out_path}")
    print("Run 01_backtest.py with TRADES_FILE = 'quiver_trades.json' to backtest.")


if __name__ == "__main__":
    main()

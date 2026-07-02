"""
Paper Trade Reviewer
Shows all logged signals, P&L (if manually updated), and running stats.
Usage: python3 review_trades.py
"""

import json, sys
from pathlib import Path
import config as cfg

def load():
    p = Path("gold_trade_log.json")
    if not p.exists():
        print("No trade log found yet. Bot hasn't fired a signal.")
        sys.exit(0)
    return json.loads(p.read_text())

def review():
    trades = load()
    if not trades:
        print("Trade log is empty.")
        return

    W = 68
    print("═"*W)
    print("  GOLD BOT — PAPER TRADE REVIEW")
    print("═"*W)
    print(f"  Total signals logged : {len(trades)}")
    open_t   = [t for t in trades if t.get("result") == "OPEN"]
    closed_t = [t for t in trades if t.get("result") in ("WIN","LOSS")]
    print(f"  Open trades          : {len(open_t)}")
    print(f"  Closed trades        : {len(closed_t)}")
    print("─"*W)

    print(f"\n  {'#':<3} {'Date/Time':20} {'Dir':5} {'Risk':5} {'Flags':25} {'Entry':>8} {'SL':>8} {'TP':>8} {'R':>5} {'RSI':>5} {'Result':>6}")
    print("  " + "─"*(W-2))
    for i, t in enumerate(trades, 1):
        flags = ", ".join(t.get("red_flags",[])) or "clean"
        res   = t.get("result","OPEN")
        pnl   = f"{t['pnl_r']:+.2f}R" if t.get("pnl_r") is not None else "—"
        ts    = str(t["timestamp"])[:16]
        print(f"  {i:<3} {ts:20} {t['direction']:5} "
              f"{t['risk_pct']*100:.0f}%   {flags:25} "
              f"{t['entry']:>8.2f} {t['sl']:>8.2f} {t['tp']:>8.2f} "
              f"{t['tp_r']:>5.2f} {t['rsi']:>5.1f} {res:>6}")

    if closed_t:
        wins = [t for t in closed_t if t["result"]=="WIN"]
        losses = [t for t in closed_t if t["result"]=="LOSS"]
        wr = len(wins)/len(closed_t)*100
        pnl_rs = [t["pnl_r"] for t in closed_t if t.get("pnl_r") is not None]
        pnl_usds = [t["pnl_usd"] for t in closed_t if t.get("pnl_usd") is not None]
        print("\n" + "─"*W)
        print(f"  Closed: {len(closed_t)}  |  WR: {wr:.1f}%  |  "
              f"Total R: {sum(pnl_rs):+.2f}R  |  Total P&L: ${sum(pnl_usds):+,.0f}")

    print("─"*W)
    print(f"\n  To manually close a paper trade, edit {cfg.TRADE_LOG}")
    print(f"  Set result to WIN or LOSS and fill pnl_r / pnl_usd.")
    print("═"*W)

if __name__ == "__main__":
    review()

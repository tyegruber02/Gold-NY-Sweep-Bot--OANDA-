"""
Gold Bot — Live Performance Dashboard
======================================
Reads gold_trade_log.json (written by gold_bot.py each GitHub Actions run).
Compares live results against the champion backtest baseline.

Usage:
    python3 gold_performance.py                   # basic dashboard
    python3 gold_performance.py --live            # also fetches OANDA for open trade P&L
    python3 gold_performance.py --watch           # refresh every 60s (pair with --live)

Champion baseline (locked config, 730-day backtest):
  28 trades | 66.7% WR | $+284,642 | DD -3.31% | PF 12.86×
  30% at TP1 (80% NY range) | 70% runner to TP2 (200% NY range)
"""

import os, sys, json, time, argparse
from datetime import datetime
from pathlib import Path
import pytz

ET = pytz.timezone("America/New_York")

TRADE_LOG    = "gold_trade_log.json"
ACCOUNT_SIZE = 100_000

# Champion baseline (locked from backtest)
CHAMP = dict(
    trades=28, wr=66.7, pnl=998_325, dd=-4.32, pf=12.09,
    avg_win_r=6.04, avg_loss_r=-1.00, period_days=730
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_log():
    p = Path(TRADE_LOG)
    if not p.exists() or p.stat().st_size <= 2:
        return []
    return json.loads(p.read_text())

def pct_bar(value, reference, width=20, pos_char="█", neg_char="░"):
    if reference == 0: return " " * width
    ratio = min(abs(value / reference), 1.5)
    filled = max(1, int(ratio * width))
    char = pos_char if value >= 0 else neg_char
    return char * filled

def status_icon(result):
    return {"WIN": "✓", "PARTIAL": "~", "LOSS": "✗", "BE": "○", "OPEN": "…"}.get(result, "?")

def since(ts_str):
    try:
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None: ts = ts.replace(tzinfo=ET)
        delta = datetime.now(ET) - ts.astimezone(ET)
        h = int(delta.total_seconds() // 3600)
        if h < 24: return f"{h}h ago"
        return f"{delta.days}d ago"
    except: return "—"


# ── OANDA live fetch ──────────────────────────────────────────────────────────

def fetch_open_pnl(entries):
    """Fetch unrealised P&L for any OPEN trades from OANDA. Returns dict of index → pnl_usd."""
    api_key    = os.environ.get("OANDA_API_KEY", "")
    account_id = os.environ.get("OANDA_ACCOUNT_ID", "")
    oanda_env  = os.environ.get("OANDA_ENV", "practice")
    if not api_key or not account_id:
        return {}

    try:
        import oandapyV20
        import oandapyV20.endpoints.trades as trades_ep
        client = oandapyV20.API(access_token=api_key, environment=oanda_env)
    except ImportError:
        print("  oandapyV20 not installed — skipping live fetch")
        return {}

    live_pnl = {}
    for i, entry in enumerate(entries):
        if entry.get("result") != "OPEN": continue
        tp1_id = entry.get("tp1_trade_id")
        run_id = entry.get("runner_trade_id")
        total_unreal = 0.0
        for tid in [tp1_id, run_id]:
            if not tid: continue
            try:
                r = trades_ep.TradeDetails(account_id, tid)
                client.request(r)
                total_unreal += float(r.response["trade"].get("unrealizedPL", 0))
            except: pass
        if total_unreal != 0.0:
            live_pnl[i] = total_unreal
    return live_pnl


# ── Stats computation ─────────────────────────────────────────────────────────

def compute_stats(entries, live_pnl=None):
    live_pnl = live_pnl or {}
    closed   = [e for e in entries if e.get("result") in ("WIN","LOSS","PARTIAL","BE")]
    open_tr  = [e for e in entries if e.get("result") == "OPEN"]
    decisive = [e for e in closed   if e.get("result") in ("WIN","LOSS","PARTIAL")]
    wins     = [e for e in decisive if e.get("result") in ("WIN","PARTIAL")]
    losses   = [e for e in decisive if e.get("result") == "LOSS"]
    bes      = [e for e in closed   if e.get("result") == "BE"]

    total_pnl  = sum(e.get("pnl_usd", 0) for e in closed)
    wr         = len(wins) / len(decisive) * 100 if decisive else 0.0
    win_pnl    = sum(e.get("pnl_usd", 0) for e in wins)
    loss_pnl   = abs(sum(e.get("pnl_usd", 0) for e in losses))
    pf         = win_pnl / loss_pnl if loss_pnl > 0 else (99.0 if win_pnl > 0 else 0.0)
    avg_win_r  = (sum(e.get("pnl_r", 0) for e in wins) / len(wins)) if wins else 0.0
    avg_loss_r = (sum(e.get("pnl_r", 0) for e in losses) / len(losses)) if losses else 0.0

    # Equity curve + drawdown
    eq = ACCOUNT_SIZE
    peak = ACCOUNT_SIZE; max_dd = 0.0
    equity_points = [ACCOUNT_SIZE]
    for e in closed:
        eq += e.get("pnl_usd", 0)
        equity_points.append(eq)
        if eq > peak: peak = eq
        dd = (eq - peak) / peak * 100
        if dd < max_dd: max_dd = dd

    unrealised = sum(live_pnl.values())

    return dict(
        n_total=len(entries), n_closed=len(closed), n_open=len(open_tr),
        n_decisive=len(decisive), n_wins=len(wins), n_losses=len(losses), n_bes=len(bes),
        wr=wr, pnl=total_pnl, pf=pf,
        avg_win_r=avg_win_r, avg_loss_r=avg_loss_r,
        max_dd=max_dd, final_equity=ACCOUNT_SIZE + total_pnl,
        equity_points=equity_points, unrealised=unrealised,
        decisive=decisive, closed=closed, open_tr=open_tr
    )


# ── Render dashboard ──────────────────────────────────────────────────────────

def render(entries, live_pnl=None):
    live_pnl = live_pnl or {}
    s = compute_stats(entries, live_pnl)
    now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    W = 70

    print("\033[2J\033[H", end="")  # clear screen

    print("═" * W)
    print("  GOLD XAU/USD — LIVE PERFORMANCE TRACKER")
    print(f"  {now}  |  Log: {TRADE_LOG}")
    print("═" * W)

    if not entries:
        print("\n  No trades logged yet.")
        print("  The bot commits gold_trade_log.json after each GitHub Actions run.")
        print("  First trade will appear once the bot detects a valid NY-sweep signal.\n")
        print("═" * W)
        print(f"\n  CHAMPION BASELINE (backtest, {CHAMP['period_days']}d):")
        print(f"  {CHAMP['trades']} trades  |  {CHAMP['wr']:.1f}% WR  |  "
              f"${CHAMP['pnl']:+,.0f}  |  DD {CHAMP['dd']:.2f}%  |  PF {CHAMP['pf']:.2f}×")
        print("═" * W)
        return

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n  LIVE SUMMARY  ({s['n_closed']} closed  /  {s['n_open']} open)")
    print("─" * W)

    def delta_str(live_val, champ_val, fmt=".1f", higher_better=True):
        if live_val == 0 and champ_val == 0: return "—"
        delta = live_val - champ_val
        arrow = ("▲" if delta > 0 else "▼") if higher_better else ("▼" if delta > 0 else "▲")
        return f"{live_val:{fmt}}  {arrow} {abs(delta):{fmt}} vs champ"

    rows = [
        ("Trades (closed)",  f"{s['n_wins']}W / {s['n_losses']}L / {s['n_bes']}BE  "
                             f"({s['n_decisive']} decisive)", ""),
        ("Win rate",         f"{s['wr']:.1f}%",
                             f"{'▲' if s['wr']>=CHAMP['wr'] else '▼'} {abs(s['wr']-CHAMP['wr']):.1f}pp vs {CHAMP['wr']:.1f}%"),
        ("Profit factor",    f"{s['pf']:.2f}×" if s['pf'] < 90 else "—",
                             f"{'▲' if s['pf']>=CHAMP['pf'] else '▼'} vs {CHAMP['pf']:.2f}×" if s['pf'] < 90 else ""),
        ("Total P&L",        f"${s['pnl']:+,.0f}",
                             f"{'▲' if s['pnl']>=0 else '▼'} {s['pnl']/ACCOUNT_SIZE*100:+.1f}% account return"),
        ("Max drawdown",     f"{s['max_dd']:.2f}%",
                             f"{'▲ better' if s['max_dd']>CHAMP['dd'] else '▼ worse'} vs {CHAMP['dd']:.2f}% champ"),
        ("Avg win",          f"{s['avg_win_r']:+.2f}R"  if s['n_wins']   else "—",
                             f"champ {CHAMP['avg_win_r']:+.2f}R"),
        ("Avg loss",         f"{s['avg_loss_r']:+.2f}R" if s['n_losses'] else "—",
                             f"champ {CHAMP['avg_loss_r']:+.2f}R"),
        ("Final equity",     f"${s['final_equity']:,.0f}", ""),
    ]
    if s["unrealised"] != 0:
        rows.append(("Unrealised P&L", f"${s['unrealised']:+,.2f}  (open trades)", ""))

    for label, val, note in rows:
        print(f"  {label:<20} {val:<22} {note}")
    print("─" * W)

    # ── vs Champion ────────────────────────────────────────────────────────────
    if s["n_decisive"] > 0:
        # Scale champion to same number of trades for pace comparison
        pace_factor = s["n_decisive"] / CHAMP["trades"]
        champ_pnl_scaled = CHAMP["pnl"] * pace_factor
        print(f"\n  PACE vs CHAMPION  ({s['n_decisive']} trades = {pace_factor*100:.0f}% of backtest sample)")
        print("─" * W)
        print(f"  {'Metric':<22} {'Live':>12} {'Expected pace':>14} {'Δ':>10}")
        print(f"  {'─'*58}")

        def pace_row(label, live_val, champ_full, fmt_str="+,.0f", is_pct=False):
            expected = champ_full * pace_factor if not is_pct else champ_full
            delta    = live_val - expected
            symbol   = "▲" if delta >= 0 else "▼"
            fmtd_live = f"${live_val:{fmt_str}}" if "," in fmt_str else f"{live_val:{fmt_str}}"
            fmtd_exp  = f"${expected:{fmt_str}}" if "," in fmt_str else f"{expected:{fmt_str}}"
            print(f"  {label:<22} {fmtd_live:>12} {fmtd_exp:>14} {symbol}{abs(delta):{fmt_str.lstrip('+')!s}:>9}")

        pace_row("P&L",       s["pnl"],    CHAMP["pnl"])
        pace_row("Win rate",  s["wr"],     CHAMP["wr"],  ".1f", is_pct=True)
        pace_row("Drawdown",  s["max_dd"], CHAMP["dd"],  ".2f", is_pct=True)
        print("─" * W)

    # ── Equity curve ───────────────────────────────────────────────────────────
    if len(s["equity_points"]) > 1:
        print(f"\n  EQUITY CURVE  (${ACCOUNT_SIZE:,.0f} start)")
        print("─" * W)
        pts   = s["equity_points"]
        hi    = max(pts); lo = min(pts)
        rng   = hi - lo if hi != lo else 1
        rows_h = 8; col_w = max(1, (W - 10) // len(pts))
        chart  = [[" "] * len(pts) for _ in range(rows_h)]
        for j, val in enumerate(pts):
            row = int((hi - val) / rng * (rows_h - 1))
            chart[row][j] = "•"
        for row in chart:
            print("  │  " + "".join(row))
        print(f"  └{'─'*(W-4)}")
        print(f"  ${lo:>10,.0f}  ←low  high→  ${hi:>10,.0f}   "
              f"current: ${pts[-1]:>10,.0f}")
        print("─" * W)

    # ── Monthly breakdown ──────────────────────────────────────────────────────
    if s["closed"]:
        print(f"\n  MONTHLY BREAKDOWN")
        print("─" * W)
        monthly: dict = {}
        for e in s["closed"]:
            ts_raw = e.get("timestamp", e.get("closed_at", ""))
            try:
                ts = datetime.fromisoformat(ts_raw)
                month = ts.strftime("%Y-%m")
            except: month = "unknown"
            if month not in monthly: monthly[month] = {"pnl": 0.0, "n": 0, "wins": 0}
            monthly[month]["pnl"]  += e.get("pnl_usd", 0)
            monthly[month]["n"]    += 1
            monthly[month]["wins"] += 1 if e.get("result") in ("WIN","PARTIAL") else 0

        max_abs = max((abs(v["pnl"]) for v in monthly.values()), default=1)
        for m in sorted(monthly):
            v   = monthly[m]
            bar = pct_bar(v["pnl"], max_abs, width=20)
            wr  = v["wins"] / v["n"] * 100 if v["n"] else 0
            sym = "▲" if v["pnl"] >= 0 else "▼"
            print(f"  {m}  {sym}  ${v['pnl']:>+9,.0f}  {bar:<20}  ({v['n']}t / {wr:.0f}%WR)")
        print("─" * W)

    # ── Open trades ────────────────────────────────────────────────────────────
    if s["open_tr"]:
        print(f"\n  OPEN POSITIONS  ({len(s['open_tr'])})")
        print("─" * W)
        for i, e in enumerate(entries):
            if e.get("result") != "OPEN": continue
            unreal = live_pnl.get(i, None)
            unreal_str = f"  unrealised: ${unreal:+,.2f}" if unreal is not None else "  (no OANDA fetch)"
            tp1_str    = f"{e.get('tp1', '?'):.2f}" if e.get("tp1") else "?"
            tp2_str    = f"{e.get('tp2', '?'):.2f}" if e.get("tp2") else "?"
            print(f"  {e.get('direction','?'):<6} @ {e.get('entry',0):.2f}  "
                  f"SL {e.get('sl',0):.2f}  TP1 {tp1_str}  TP2 {tp2_str}  "
                  f"{e.get('risk_pct',0)*100:.0f}%{unreal_str}  [{since(e.get('timestamp',''))}]")
        print("─" * W)

    # ── Full trade log ─────────────────────────────────────────────────────────
    if s["closed"]:
        print(f"\n  CLOSED TRADE LOG")
        print("─" * W)
        print(f"  {'#':<4} {'Date':<12} {'Dir':<6} {'Risk':<5} {'Entry':>8} "
              f"{'SL':>8} {'TP1':>8} {'Result':<9} {'P&L R':>7} {'P&L $':>10}")
        print(f"  {'─'*72}")
        n = 0
        for e in entries:
            if e.get("result") not in ("WIN","LOSS","PARTIAL","BE"): continue
            n += 1
            ts_raw = e.get("timestamp", "")
            try:    date_s = datetime.fromisoformat(ts_raw).strftime("%Y-%m-%d")
            except: date_s = "—"
            result = e.get("result", "?")
            icon   = status_icon(result)
            pnl_r  = e.get("pnl_r", 0) or 0
            pnl_u  = e.get("pnl_usd", 0) or 0
            print(f"  {n:<4} {date_s:<12} {e.get('direction','?'):<6} "
                  f"{e.get('risk_pct',0)*100:.0f}%   "
                  f"{e.get('entry',0):>8.2f} {e.get('sl',0):>8.2f} "
                  f"{e.get('tp1', e.get('tp',0)):>8.2f} "
                  f"{icon} {result:<7} {pnl_r:>+7.3f}R {pnl_u:>+10,.0f}")
        print("─" * W)

    # ── Champion reminder ──────────────────────────────────────────────────────
    print(f"\n  CHAMPION BASELINE ({CHAMP['period_days']}d backtest):")
    print(f"  {CHAMP['trades']}t  |  {CHAMP['wr']:.1f}% WR  |  ${CHAMP['pnl']:+,.0f}  "
          f"|  DD {CHAMP['dd']:.2f}%  |  PF {CHAMP['pf']:.2f}×  "
          f"|  avg win {CHAMP['avg_win_r']:+.2f}R")
    print("═" * W)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live",  action="store_true", help="Fetch open trade P&L from OANDA")
    parser.add_argument("--watch", action="store_true", help="Refresh every 60s")
    args = parser.parse_args()

    while True:
        entries  = load_log()
        live_pnl = fetch_open_pnl(entries) if args.live else {}
        render(entries, live_pnl)

        if not args.watch:
            break
        print(f"\n  Refreshing in 60s… (Ctrl+C to quit)")
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            print("\n  Stopped.")
            break


if __name__ == "__main__":
    main()

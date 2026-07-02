"""
Bot configuration — fill in your OANDA practice credentials.

To get credentials:
  1. Go to https://www.oanda.com/demo-account/
  2. Create a free practice account
  3. Settings → Manage API Access → Generate token
  4. Copy your account ID from the dashboard URL or account summary
"""

# ── OANDA Credentials ────────────────────────────────────────────────────────
OANDA_API_KEY    = "YOUR_API_KEY_HERE"
OANDA_ACCOUNT_ID = "YOUR_ACCOUNT_ID_HERE"
OANDA_ENV        = "practice"   # "practice" or "live" — NEVER change to live until ready

# ── Instrument ────────────────────────────────────────────────────────────────
INSTRUMENT = "XAU_USD"   # Gold spot vs USD on OANDA

# ── Account ───────────────────────────────────────────────────────────────────
ACCOUNT_SIZE   = 100_000   # USD — must match your OANDA practice account balance
RISK_HIGH      = 0.05      # 5% per trade — clean setups (no red flags)
RISK_LOW       = 0.03      # 3% per trade — red flag detected

# ── Paper mode ────────────────────────────────────────────────────────────────
# When True: signals are detected and logged but NO real orders are placed.
# Orders are printed and saved to trade_log.json for review.
# Set to False only when you are ready to place live paper orders on OANDA practice.
PAPER_MODE = True

# ── Notifications (optional) ──────────────────────────────────────────────────
# Set to your email to receive trade alerts. Leave empty to disable.
ALERT_EMAIL = ""

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE   = "bot.log"
TRADE_LOG  = "trade_log.json"
STATE_FILE = "bot_state.json"

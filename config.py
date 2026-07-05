"""
config.py — central configuration for the Deriv TOUCH/NOTOUCH self-learning bot.

All tunables live here. Secrets come from environment variables (.env via python-dotenv),
never hardcoded.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Deriv connection
# ---------------------------------------------------------------------------
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")  # 1089 = Deriv's public demo app id
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "")  # token for the account you want to trade on

# Deriv's current auth flow no longer connects a static app_id WS url with an
# "authorize" message. Instead, deriv_client.py resolves the account_id for
# ACCOUNT_TYPE via REST, then exchanges it for a one-time, account-bound WS
# url. If you already know your numeric account_id you can set it here to skip
# the resolution call; otherwise leave it unset and it'll be looked up.
DERIV_ACCOUNT_ID = os.getenv("DERIV_ACCOUNT_ID") or None

# ACCOUNT_TYPE is a safety check, not a switch — Deriv tokens are already tied to a specific
# account (demo or real) when you generate them. Set this to what you EXPECT the token above
# to be; on startup the bot verifies the authorized account actually matches and refuses to
# run otherwise, so a copy-paste mistake with tokens can't silently trade real money.
ACCOUNT_TYPE = os.getenv("ACCOUNT_TYPE", "demo").lower()   # "demo" or "real"
if ACCOUNT_TYPE not in ("demo", "real"):
    raise ValueError(f"ACCOUNT_TYPE must be 'demo' or 'real', got {ACCOUNT_TYPE!r}")

# Extra deliberate gate for real money: must be explicitly set to "yes" in the Railway
# variables for the bot to place real trades, even if a real-account token is provided.
CONFIRM_LIVE_TRADING = os.getenv("CONFIRM_LIVE_TRADING", "no").lower() == "yes"

# ---------------------------------------------------------------------------
# Symbol universe — scanned by the calibrator every cycle
# ---------------------------------------------------------------------------
SYMBOL_UNIVERSE = [
    # Volatility Indices (2s tick)
    "R_10", "R_25", "R_50", "R_75", "R_100",
    # 1-second Volatility Indices
    "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V",
    # Bull/Bear market indices
    "RDBULL", "RDBEAR",
]

TOP_N_SYMBOLS = 3               # how many symbols the bot actively trades at once
CONSECUTIVE_LOSS_RECAL_TRIGGER = 2   # recalibrate a symbol slot after this many losses in a row

# ---------------------------------------------------------------------------
# Calibration grid — barrier distances (in units of current EWMA sigma) and durations
# ---------------------------------------------------------------------------
BARRIER_DISTANCE_GRID_SIGMA = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]   # multiples of sigma*sqrt(T)
DURATION_GRID = [
    # (duration_value, duration_unit) — Deriv units: t (ticks), s, m, h, d
    (15, "m"), (30, "m"), (1, "h"), (2, "h"), (4, "h"), (1, "d"),
]

HISTORY_TICKS_FOR_CALIBRATION = 5000     # ticks pulled per symbol for regime/hurst/vol estimation
BOOTSTRAP_N_PATHS = 2000                 # Monte Carlo / block-bootstrap paths per candidate
BOOTSTRAP_BLOCK_SIZE = 50                # stationary block bootstrap block length (ticks)

# ---------------------------------------------------------------------------
# Regime detection (HMM)
# ---------------------------------------------------------------------------
HMM_N_REGIMES = 3           # low-vol/mean-reverting, trending, high-vol/crisis
HMM_REFIT_EVERY_N_CYCLES = 20

# ---------------------------------------------------------------------------
# Risk / staking (moderate posture: live, small fixed-ish stakes, EV floor gated)
# ---------------------------------------------------------------------------
BASE_STAKE = float(os.getenv("BASE_STAKE", "1.0"))       # account currency units
MAX_STAKE = float(os.getenv("MAX_STAKE", "5.0"))
EV_FLOOR = float(os.getenv("EV_FLOOR", "0.02"))          # candidate must clear +2% EV/stake to be tradable
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))  # fractional Kelly (safety multiplier)
MIN_WIN_PROB_FLOOR = float(os.getenv("MIN_WIN_PROB_FLOOR", "0.55"))  # Bayesian posterior floor to trade a cell

# ---------------------------------------------------------------------------
# Bayesian priors (Beta-Bernoulli), weak uninformative prior per new cell
# ---------------------------------------------------------------------------
BETA_PRIOR_ALPHA = 2.0
BETA_PRIOR_BETA = 2.0

# ---------------------------------------------------------------------------
# Bucketing for the Bayesian posterior table
# ---------------------------------------------------------------------------
BARRIER_BUCKET_EDGES = [0.0, 0.75, 1.25, 1.75, 2.25, 2.75, 10.0]  # in sigma units
DURATION_BUCKETS = ["<=30m", "30m-2h", "2h-6h", ">6h"]

# ---------------------------------------------------------------------------
# Persistence (Supabase) — reuse existing project, same as other bots
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ---------------------------------------------------------------------------
# Loop timing
# ---------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = 5
FULL_RECAL_EVERY_SECONDS = 6 * 3600   # safety-net full recalibration cadence regardless of losses

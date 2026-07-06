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
# Calibration grid — barrier distances (in units of current EWMA sigma), and the
# DURATION SEARCH SPACE the Intelligence Stack auto-selects from (see
# ensemble.select_durations_for_barrier). Durations run from 5 ticks up to
# ~30 minutes -- short-horizon NOTOUCH only, no fixed-duration grid: for each
# barrier distance, the ensemble sweeps this whole space and picks whichever
# duration(s) sit just above the win-probability floor (best payout-per-risk),
# rather than trading a duration a human picked in advance.
# ---------------------------------------------------------------------------
BARRIER_DISTANCE_GRID_SIGMA = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]   # multiples of sigma*T^exponent
DURATION_SEARCH_SPACE = [
    # (duration_value, duration_unit) — Deriv units: t (ticks), m (minutes)
    (5, "t"), (10, "t"), (15, "t"), (20, "t"), (30, "t"), (50, "t"), (75, "t"), (100, "t"),
    (1, "m"), (2, "m"), (3, "m"), (5, "m"), (7, "m"), (10, "m"), (15, "m"), (20, "m"), (30, "m"),
]
DURATION_AUTOSELECT_TOP_K = 2   # how many auto-selected durations per barrier proceed to quoting

# Approximate seconds-per-tick per symbol family, used only to convert minute-based
# search-space entries into a simulation step count consistent with the tick-based
# entries (Deriv's own tick cadence is what actually matters at execution time --
# this is just so "3 minutes" and "90 ticks" get treated as roughly the same
# simulation horizon on a 2s-tick symbol).
SYMBOL_TICK_INTERVAL_SECONDS = {
    "R_10": 2.0, "R_25": 2.0, "R_50": 2.0, "R_75": 2.0, "R_100": 2.0,
    "1HZ10V": 1.0, "1HZ25V": 1.0, "1HZ50V": 1.0, "1HZ75V": 1.0, "1HZ100V": 1.0,
    "RDBULL": 2.0, "RDBEAR": 2.0,
}
DEFAULT_TICK_INTERVAL_SECONDS = 2.0

HISTORY_TICKS_FOR_CALIBRATION = 5000     # ticks pulled per symbol for regime/hurst/vol estimation

# Monte Carlo / block-bootstrap paths. Sweeping the full (symbol x duration-search-space
# x barrier) grid at the full path count would be far too slow, so the search/screening
# pass uses a cheap path count, and only the single best candidate per symbol gets a
# full-precision confirmation pass (config.MC_SIMULATIONS = 75,000, matching the MC path
# count used consistently across the other bots) right before it's quoted/traded.
MC_SCREENING_PATHS = int(os.getenv("MC_SCREENING_PATHS", "2000"))
MC_SIMULATIONS = int(os.getenv("MC_SIMULATIONS", "75000"))
BOOTSTRAP_BLOCK_SIZE = 50                # stationary block bootstrap block length (ticks)

# ---------------------------------------------------------------------------
# Intelligence Stack toggles — each can be disabled independently (e.g. while
# debugging) without touching ensemble.py. All default on.
# ---------------------------------------------------------------------------
GARCH_ENABLED = os.getenv("GARCH_ENABLED", "1").strip().lower() in ("1", "true", "yes")
KALMAN_ENABLED = os.getenv("KALMAN_ENABLED", "1").strip().lower() in ("1", "true", "yes")
HAWKES_ENABLED = os.getenv("HAWKES_ENABLED", "1").strip().lower() in ("1", "true", "yes")
ARFIMA_ENABLED = os.getenv("ARFIMA_ENABLED", "1").strip().lower() in ("1", "true", "yes")
MARKOV_ENABLED = os.getenv("MARKOV_ENABLED", "1").strip().lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Self-Improvement Engine — daily walk-forward validation + rollback protection
# ---------------------------------------------------------------------------
# What it actually validates: the Bayesian posterior layer is the only piece of
# PERSISTED state that accumulates online across trades (the HMM/GARCH/Kalman/
# ARFIMA models are refit fresh from live tick history every calibration cycle,
# so they can't go stale/need a rollback the way persisted state can). Once a
# day, it checks whether the tracker's calibration quality (Brier score of
# logged p_no_touch_est vs realized outcome) over the last
# SELF_IMPROVEMENT_VALIDATION_DAYS has gotten WORSE than the last snapshot's
# score by more than the tolerance below -- if so it rolls the tracker back to
# that last known-good snapshot rather than continuing to serve a degraded one.
SELF_IMPROVEMENT_ENABLED = os.getenv("SELF_IMPROVEMENT_ENABLED", "1").strip().lower() in ("1", "true", "yes")
SELF_IMPROVEMENT_INTERVAL_SECONDS = int(os.getenv("SELF_IMPROVEMENT_INTERVAL_SECONDS", str(24 * 3600)))
SELF_IMPROVEMENT_VALIDATION_DAYS = float(os.getenv("SELF_IMPROVEMENT_VALIDATION_DAYS", "3"))
SELF_IMPROVEMENT_MIN_TRADES_FOR_VALIDATION = int(os.getenv("SELF_IMPROVEMENT_MIN_TRADES_FOR_VALIDATION", "20"))
# Brier score is in [0,1] (lower = better calibrated); this is an absolute-points
# tolerance, not a percentage -- a jump of more than this over the last snapshot
# triggers a rollback.
SELF_IMPROVEMENT_ROLLBACK_BRIER_TOLERANCE = float(os.getenv("SELF_IMPROVEMENT_ROLLBACK_BRIER_TOLERANCE", "0.05"))

# ---------------------------------------------------------------------------
# Regime detection (HMM)
# ---------------------------------------------------------------------------
HMM_N_REGIMES = 3           # low-vol/mean-reverting, trending, high-vol/crisis
HMM_REFIT_EVERY_N_CYCLES = 20

# ---------------------------------------------------------------------------
# Risk / staking (tiny bankroll posture: starting balance is $0.35, so BASE_STAKE
# and MAX_STAKE default to Deriv's own practical minimum stake for these contracts
# rather than an arbitrary larger number -- there's no room for Kelly sizing to
# size UP from here until the balance actually grows past the minimum many times
# over. MIN_STAKE is a hard floor: below it, Deriv will reject the contract
# outright, so staking.py refuses to size a trade under it rather than attempting
# one that will just error out.)
# ---------------------------------------------------------------------------
MIN_STAKE = float(os.getenv("MIN_STAKE", "0.35"))        # Deriv's practical floor for these contracts
BASE_STAKE = float(os.getenv("BASE_STAKE", "0.35"))       # account currency units
MAX_STAKE = float(os.getenv("MAX_STAKE", "0.35"))
EV_FLOOR = float(os.getenv("EV_FLOOR", "0.02"))          # candidate must clear +2% EV/stake to be tradable
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))  # fractional Kelly (safety multiplier)
MIN_WIN_PROB_FLOOR = float(os.getenv("MIN_WIN_PROB_FLOOR", "0.55"))  # Bayesian posterior floor to trade a cell

# Explicit minimum PAYOUT RETURN gate, checked against Deriv's actual quoted proposal --
# distinct from EV_FLOOR. EV_FLOOR asks "is probability x payout worth it", which is the
# mathematically correct gate, but is opaque as a single number. MIN_RETURN_PCT is the plain
# version: "don't bother unless this specific barrier/duration actually pays out N% or more"
# (0.40 = candidate must offer at least a 40% return, i.e. payout >= 1.4x stake).
#
# IMPORTANT: these two gates are NOT redundant, and passing both does not guarantee the other.
# A candidate can clear MIN_WIN_PROB_FLOOR and MIN_RETURN_PCT individually while still being
# -EV: e.g. p=0.55, payout=1.40x stake -> EV = 0.55*1.40 - 1 = -0.23 (a bad trade that would
# pass both individual floors). Both gates are enforced together in calibrator.py precisely
# to close that gap -- MIN_RETURN_PCT narrows down to attractively-priced candidates first,
# EV_FLOOR is still the final word on whether it's actually worth taking.
MIN_RETURN_PCT = float(os.getenv("MIN_RETURN_PCT", "0.01"))

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

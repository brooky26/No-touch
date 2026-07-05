# Deriv TOUCH/NOTOUCH Self-Learning Bot

A self-calibrating NOTOUCH trading bot for Deriv synthetic indices. It scans a symbol
universe, models each symbol's price process with the appropriate stochastic tool for its
regime, sweeps a (barrier x duration) grid for expected value, trades the best candidates,
and recalibrates itself — either on a timer or after a losing streak.

## Architecture

| File | Responsibility |
|---|---|
| `config.py` | All tunables: symbol universe, grids, risk floors, env-var wiring |
| `stochastic.py` | GBM first-passage, OU calibration/simulation, Merton jump-diffusion, fBM (Davies-Harte), Hurst (R/S), EWMA vol, stationary block-bootstrap touch-probability estimator |
| `regime.py` | Gaussian HMM regime detector (3 states: mean-reverting/low-vol, trending, high-vol/crisis), multi-restart fitting with a degenerate-state safety clamp |
| `bayesian.py` | Beta-Bernoulli posterior per (symbol, regime, barrier-bucket, duration-bucket) cell; blends the model estimate with realized outcomes as evidence accumulates |
| `staking.py` | EV-floor gate + fractional-Kelly stake sizing, sized off the *lower* credible-interval bound so uncertain cells get smaller stakes automatically |
| `deriv_client.py` | Async Deriv WebSocket v3 client: history, proposals, buy, settlement polling |
| `calibrator.py` | Per-symbol calibration pipeline + universe-wide ranking by EV |
| `persistence.py` | Supabase store for Bayesian state, trade log, calibration log, active-symbol slots |
| `bot.py` | Main loop tying it all together |
| `supabase_schema.sql` | Table DDL — run once in your Supabase SQL editor |

## How a calibration cycle works

For each symbol in the universe (`R_10..R_100`, `1HZ10V..1HZ100V`, `RDBULL`, `RDBEAR`):

1. Pull recent tick history, compute EWMA vol and the Hurst exponent **on returns** (not
   raw price levels — R/S on price levels trivially reads H≈1 since price is an integrated
   process).
2. Classify the current HMM regime and its (μ, σ) at the same time.
3. Pick an analytical prior based on regime character:
   - Hurst < 0.45 or ADF says stationary → Ornstein-Uhlenbeck (mean-reverting) simulation
   - Otherwise → GBM analytical first-passage, corrected for Hurst deviation from 0.5
4. Always refine with a **stationary block-bootstrap** over historical return blocks (this
   is the primary, most robust estimator — it preserves autocorrelation and volatility
   clustering that a Gaussian Monte Carlo would erase). The final probability blends
   40% analytical prior / 60% bootstrap.
5. Blend the model probability with the Bayesian posterior for that specific
   (symbol, regime, barrier-bucket, duration-bucket) cell — the blend weight shifts toward
   the empirical posterior as more real trades resolve in that cell.
6. Pull live Deriv proposals for the top model candidates to get the *actual* quoted payout,
   compute EV per unit stake, and keep only candidates that clear `EV_FLOOR` **and** whose
   conservative (CI lower-bound) win probability clears `MIN_WIN_PROB_FLOOR`.
7. Rank all symbols by their best surviving candidate's EV; take the top 3
   (`TOP_N_SYMBOLS`).

## Self-learning loop

- Every trade's outcome updates the Beta-Bernoulli posterior for its cell — this is what
  actually lets the bot learn which (symbol, regime, barrier, duration) combinations work,
  independent of whether the original model estimate was right.
- **Recalibration triggers:**
  - A traded symbol slot hits `CONSECUTIVE_LOSS_RECAL_TRIGGER` (2) losses in a row → full
    universe recalibration, which can swap that symbol out for a better-ranked one.
  - Safety-net timer (`FULL_RECAL_EVERY_SECONDS`, default 6h) recalibrates regardless, since
    regimes can drift even without a loss streak.
- Kelly-fraction staking uses the *lower* bound of each cell's credible interval, so a cell
  with few observations (wide CI) automatically gets a smaller stake even if its point
  estimate looks good — uncertainty is priced in, not just the win-rate.

## Running this on Railway

The `.py` files aren't meant to be run individually — `bot.py` is the one entrypoint that
imports everything else. Railway just needs to know to run `python bot.py` as a
long-lived background worker (not a web server — it never listens on a port), plus your
secrets as environment variables. Concretely:

1. **Push this folder to a GitHub repo** (Railway deploys from a repo, or via `railway up`
   from the CLI pointed at this folder directly if you don't want GitHub involved).
2. **Railway dashboard → New Project → Deploy from GitHub repo**, pick the repo.
3. Railway auto-detects Python from `requirements.txt` and builds it. It also reads the
   included `Procfile` (`worker: python bot.py`) to know the start command. If it doesn't
   pick that up automatically: go to **Settings → Deploy → Custom Start Command** and set it
   to `python bot.py` manually.
4. **Settings → Variables** — paste in every variable from `.env.example` with your real
   values (see table below). This is the "I need variables" part — Railway's Variables tab
   *is* your `.env` file in production; nothing is read from a `.env` file on Railway itself.
5. Deploy. Check the **Deployments → logs** tab — on startup you should see a line like:
   `[bot] connected to Deriv API — account=VRTC1234 type=demo currency=USD balance=10000`
   If instead it exits immediately with an `ACCOUNT_TYPE mismatch` error, your token doesn't
   match what you set `ACCOUNT_TYPE` to — fix one or the other.

### Required variables

| Variable | What it is | Where to get it |
|---|---|---|
| `DERIV_APP_ID` | Your Deriv app ID | [api.deriv.com](https://api.deriv.com) → register an app (or use the public demo app id `1089` for testing) |
| `DERIV_API_TOKEN` | API token for the account you want to trade on | Deriv → Settings → **API Token** (`app.deriv.com/account/api-token`). Generate a token scoped to *Trade* permissions. A demo-account token and a real-account token are different tokens tied to different accounts. |
| `ACCOUNT_TYPE` | `demo` or `real` | You set this — it's a safety check, not a switch. The bot authorizes with your token, reads Deriv's own answer for whether that account is virtual, and refuses to start if it doesn't match this. |
| `CONFIRM_LIVE_TRADING` | `yes` or `no` | Extra gate — even with a real token + `ACCOUNT_TYPE=real`, trades are blocked unless this is `yes`. Keep it `no` until you've actually reviewed the config. |
| `SUPABASE_URL` / `SUPABASE_KEY` | Your existing Supabase project | Supabase dashboard → Project Settings → API |
| `BASE_STAKE` | Stake per trade, account currency (e.g. `1.0`) | Your call |
| `MAX_STAKE` | Hard per-trade ceiling regardless of Kelly sizing | Your call |
| `EV_FLOOR` | Minimum required edge per unit stake (e.g. `0.02` = 2%) | Your call |
| `KELLY_FRACTION` | Fractional Kelly multiplier (e.g. `0.25`) | Your call |
| `MIN_WIN_PROB_FLOOR` | Conservative win-probability floor to trade a cell (e.g. `0.55`) | Your call |

Run the SQL in `supabase_schema.sql` once, in your Supabase project's SQL editor, before
first deploy — it just adds four new tables, all prefixed `touch_bot_`.

## Local run (for testing before you deploy)

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in real values
python bot.py
```


## Before going live with real money

This ships configured for your stated posture — live trading, small fixed stakes
(`BASE_STAKE=1.0`, `MAX_STAKE=5.0`), gated by the EV floor. A few things worth doing before
you actually flip it on with the real token, since this is real money and this is
speculative/high-variance short-duration derivatives trading:

- **Demo-run it first.** Point `DERIV_APP_ID`/`DERIV_API_TOKEN` at a Deriv demo account for
  at least a few hundred trades across a few days so the Bayesian posteriors have real
  observations before any live capital is at risk — the model priors alone (block-bootstrap
  on a few thousand ticks) are a reasonable starting point, not a validated edge.
  Deriv NOTOUCH pricing already bakes in the house edge, so the EV floor and posterior gating
  are there to keep the bot honest about that, not to promise a positive edge exists.
- **Watch the first few live recalibration cycles manually** — confirm the symbols it picks
  and the barriers/durations it's choosing make sense before leaving it fully unattended.
- **`MAX_STAKE` and `EV_FLOOR` are conservative defaults, not calibrated to your bankroll or
  risk tolerance** — size them to what you're actually willing to lose while the Bayesian
  layer accumulates evidence.
- I'm not a financial advisor and this isn't financial advice — the code implements the
  statistical framework you asked for; whether and how much to risk on it is your call.

## Known simplifications worth knowing about

- The OU/bootstrap simulations operate in "tick step" units using the historical return
  series' native tick spacing; for durations spanning far more ticks than your calibration
  window covers, consider pulling deeper history or resampling to the target duration's tick
  density for that symbol.
- The HMM uses multi-restart fitting with a degenerate-covariance safety clamp, but with only
  a few thousand ticks it can still occasionally under-fit ambiguous regimes — the
  block-bootstrap layer is intentionally weighted higher (60%) precisely because it doesn't
  depend on the HMM being right.

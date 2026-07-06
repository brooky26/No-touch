"""
staking.py — EV floor gating + Kelly-fraction stake sizing, scaled down by posterior uncertainty.

Same idea as the Kelly-scaled EXPIRYRANGE payout work: wider credible interval on the win
probability -> smaller stake, even if the point estimate clears the EV floor.
"""
from __future__ import annotations
import config


def expected_value_per_stake(p_win: float, payout: float, stake: float) -> float:
    """
    EV per unit stake for a binary contract: win pays `payout` (total return, not profit),
    lose returns 0. EV = p*payout + (1-p)*0 - stake, expressed per unit stake.
    """
    if stake <= 0:
        return -1.0
    ev = p_win * payout - stake
    return ev / stake


def kelly_fraction(p_win: float, payout: float, stake: float) -> float:
    """
    Kelly fraction for a binary bet with net odds b = (payout - stake) / stake.
    f* = (p*(b+1) - 1) / b = (p*payout/stake - 1) / (payout/stake - 1)
    Clipped to [0, 1].
    """
    if stake <= 0 or payout <= stake:
        return 0.0
    b = (payout - stake) / stake
    f = (p_win * (b + 1) - 1) / b
    return max(0.0, min(1.0, f))


def bankroll_can_trade(bankroll: float, min_stake: float = config.MIN_STAKE) -> bool:
    """False once the bankroll has dropped below what Deriv will even accept as a stake --
    the bot should stand down cleanly rather than attempt trades that will just error out."""
    return bankroll >= min_stake


def size_stake(p_win_mean: float, p_win_low_ci: float, payout_per_unit_stake: float,
               bankroll: float, base_stake: float = config.BASE_STAKE,
               max_stake: float = config.MAX_STAKE,
               kelly_frac_multiplier: float = config.KELLY_FRACTION) -> float:
    """
    Compute the stake for a candidate trade.

    - Uses the *lower* credible-interval bound of p_win (conservative) to size Kelly, so
      uncertain cells naturally get smaller stakes even at the same point estimate.
    - Applies fractional Kelly (config.KELLY_FRACTION) for safety.
    - Hard floor at config.MIN_STAKE (Deriv rejects anything below it) and hard ceiling at
      both max_stake AND the actual bankroll -- sizing must never risk more than the
      account actually has, regardless of what Kelly's formula alone would suggest.
    """
    if not bankroll_can_trade(bankroll):
        return 0.0

    f_conservative = kelly_fraction(p_win_low_ci, payout_per_unit_stake, stake=1.0)
    f_scaled = f_conservative * kelly_frac_multiplier
    kelly_stake = f_scaled * bankroll

    hard_ceiling = min(max_stake, bankroll)
    stake = max(0.0, min(kelly_stake, hard_ceiling))
    # never trade below Deriv's minimum granularity, and never below zero conviction
    if stake < config.MIN_STAKE:
        stake = 0.0 if f_conservative <= 0 else min(base_stake, hard_ceiling)
    if stake < config.MIN_STAKE:
        # even base_stake doesn't fit under the bankroll ceiling -- can't trade this cycle
        return 0.0
    return round(min(stake, hard_ceiling), 2)


def passes_ev_floor(p_win_mean: float, payout_per_unit_stake: float,
                     ev_floor: float = config.EV_FLOOR) -> bool:
    ev = expected_value_per_stake(p_win_mean, payout_per_unit_stake, stake=1.0)
    return ev >= ev_floor

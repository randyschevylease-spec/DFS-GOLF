"""Shared Lineup Sampling — Single implementation for all sampling needs.

Used by both field_generator (opponent lineups) and candidate_generator
(your lineups). One function, one set of salary-filling logic, zero drift.
"""
import numpy as np
from config import ROSTER_SIZE, SALARY_CAP


def sample_lineups(players, n_lineups, probs, alpha_scale, min_sal,
                   min_salary, rng, sal_bias_power=4.0):
    """Fast Dirichlet-multinomial sampling for stochastic lineup generation.

    Enforces salary floor and cap. Last slots bias toward expensive players
    to push lineups toward full salary usage (realistic DK behavior).

    Args:
        players: list of player dicts
        n_lineups: target number of lineups
        probs: (n_players,) probability distribution for Dirichlet alpha
        alpha_scale: concentration parameter scale
        min_sal: minimum individual player salary
        min_salary: minimum TOTAL lineup salary (salary floor)
        rng: numpy random generator
        sal_bias_power: salary-filling aggressiveness (higher = use more cap)

    Returns:
        list of lineups (each a list of player indices)
    """
    n = len(players)
    sal_arr = np.array([p["salary"] for p in players], dtype=np.float64)
    alpha = np.maximum(probs * alpha_scale * n, 0.01)
    lineups = []
    attempts = 0

    while len(lineups) < n_lineups and attempts < n_lineups * 25:
        attempts += 1
        try:
            draw = rng.dirichlet(alpha)
        except Exception:
            draw = probs / probs.sum() if probs.sum() > 0 else np.ones(n) / n

        selected = []
        budget = SALARY_CAP
        avail = np.ones(n, dtype=bool)
        ok = True

        for slot in range(ROSTER_SIZE):
            remaining_slots = ROSTER_SIZE - slot - 1
            min_remaining = remaining_slots * min_sal
            max_affordable = budget - min_remaining

            afford = (sal_arr <= max_affordable) & avail
            if not afford.any():
                ok = False
                break

            vp = draw * afford
            sal_weight = (sal_arr / min_sal) ** (sal_bias_power + slot * 1.5)
            vp = vp * sal_weight * afford

            vp_sum = vp.sum()
            if vp_sum <= 0:
                ok = False
                break
            vp /= vp_sum

            try:
                c = rng.choice(n, p=vp)
            except Exception:
                ok = False
                break

            selected.append(c)
            budget -= sal_arr[c]
            avail[c] = False

        if ok and len(selected) == ROSTER_SIZE:
            total_sal = sal_arr[selected].sum()
            if min_salary <= total_sal <= SALARY_CAP:
                lineups.append(selected)

    return lineups

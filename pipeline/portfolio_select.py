"""
portfolio_select.py — Bergman E[max] portfolio selection.

From a pool of candidate lineups, selects the optimal portfolio of N lineups
that maximizes expected maximum payout (E[max]) across the portfolio.

The key insight: in a GPP, you only need ONE lineup to hit. So the optimal
portfolio maximizes the probability that at least one lineup finishes in
the money, weighted by payout size.

Algorithm:
  1. Score all candidate lineups across sim iterations
  2. Greedy forward selection:
     a. Start with empty portfolio
     b. For each candidate, compute marginal E[max] if added
     c. Add the candidate with highest marginal gain
     d. Repeat until portfolio is full

  This is a greedy approximation to the NP-hard E[max] optimization,
  which works well because the objective is submodular.

Inputs:
  - Candidate lineups from build_candidates.py
  - Player sim results from player_sim.simulate_tournament()
  - Payout structure from payout.py

Output:
  - data/cache/portfolio_selected.csv
"""

import csv
import math
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "engine"))

from payout import get_payout, MILLY_MAKER_PAYOUTS, MILLY_MAKER_ENTRY, MILLY_MAKER_FIELD

CANDIDATES_PATH = os.path.join(PROJECT_ROOT, "data", "cache", "candidates.csv")
OUTPUT = os.path.join(PROJECT_ROOT, "data", "cache", "portfolio_selected.csv")


def load_candidates(path=None):
    """Load candidate lineups with ownership data. Returns list of candidate dicts."""
    if path is None:
        path = CANDIDATES_PATH
    candidates = []
    with open(path) as f:
        for row in csv.DictReader(f):
            dg_ids = []
            names = []
            for i in range(1, 7):
                dg_ids.append(int(row[f"p{i}_id"]))
                names.append(row[f"p{i}_name"])
            candidates.append({
                "lineup_id": int(row["lineup_id"]),
                "dg_ids": tuple(dg_ids),
                "names": names,
                "total_salary": int(row["total_salary"]),
                "strategy": row["strategy"],
                "ownership_product": float(row.get("ownership_product", 0)),
                "expected_dupes": float(row.get("expected_dupes", 0)),
            })
    return candidates


def score_lineups(candidates, player_results, n_iterations=None):
    """
    Score all candidate lineups across sim iterations.

    Args:
        candidates: list of candidate dicts with 'dg_ids'
        player_results: dict of dg_id -> list of DK pts per iteration
        n_iterations: limit iterations (default: all)

    Returns:
        list of score arrays, one per candidate.
        Each array has n_iterations entries of total lineup DK pts.
    """
    sample_key = next(iter(player_results))
    max_iters = len(player_results[sample_key])
    if n_iterations is None:
        n_iterations = max_iters
    n_iterations = min(n_iterations, max_iters)

    scored = []
    for cand in candidates:
        lineup_scores = []
        for i in range(n_iterations):
            pts = sum(player_results.get(did, [0] * max_iters)[i] for did in cand["dg_ids"])
            lineup_scores.append(pts)
        scored.append(lineup_scores)

    return scored, n_iterations


def estimate_rank(lineup_score, opponent_mean, opponent_std, n_opponents, rng):
    """Estimate contest rank using normal approximation."""
    # What percentile is this score in the opponent distribution?
    if opponent_std <= 0:
        return 1
    z = (lineup_score - opponent_mean) / opponent_std
    # Approximate percentile using standard normal CDF
    # P(X < score) ≈ using error function
    p_better = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    # Rank = (1 - p_better) * n_opponents + 1
    rank = max(1, int((1 - p_better) * n_opponents) + 1)
    return rank


def greedy_emax_select(scored, candidates, n_portfolio,
                       opponent_mean=330, opponent_std=45,
                       n_opponents=MILLY_MAKER_FIELD,
                       payout_table=None, entry_fee=None,
                       dupe_adjust=True, seed=42):
    """
    Greedy forward selection maximizing E[max payout] across portfolio.

    For each iteration, the portfolio payout = max(payout of each lineup).
    We greedily add the lineup that maximizes the expected portfolio payout.

    When dupe_adjust=True, raw payouts are divided by (dupe_count + 1) where
    dupe_count is drawn from Poisson(expected_dupes) per iteration. This
    penalizes high-ownership lineups that are likely to be duplicated.

    Args:
        scored: list of score arrays per candidate
        candidates: candidate metadata (with ownership_product, expected_dupes)
        n_portfolio: number of lineups to select
        opponent_mean/std: opponent lineup score distribution
        n_opponents: contest field size
        payout_table: payout structure
        entry_fee: cost per entry
        dupe_adjust: apply ownership-based dupe penalty (default True)
        seed: random seed for Poisson draws

    Returns:
        list of selected candidate indices
    """
    if payout_table is None:
        payout_table = MILLY_MAKER_PAYOUTS
    if entry_fee is None:
        entry_fee = MILLY_MAKER_ENTRY

    rng = random.Random(seed)
    n_iters = len(scored[0])
    n_candidates = len(scored)

    # Precompute payouts per candidate per iteration (dupe-adjusted)
    print("  Precomputing payouts...")
    payouts = []
    for c_idx in range(n_candidates):
        exp_dupes = candidates[c_idx].get("expected_dupes", 0) if dupe_adjust else 0
        c_payouts = []
        for i in range(n_iters):
            rank = estimate_rank(scored[c_idx][i], opponent_mean, opponent_std, n_opponents, None)
            raw_payout = get_payout(rank, payout_table)
            if exp_dupes > 0:
                dupe_count = _poisson_sample(exp_dupes, rng)
                c_payouts.append(raw_payout / (dupe_count + 1))
            else:
                c_payouts.append(raw_payout)
        payouts.append(c_payouts)

    if dupe_adjust:
        n_with_dupes = sum(1 for c in candidates if c.get("expected_dupes", 0) > 0.01)
        print(f"  Dupe-adjusted payouts for {n_with_dupes} lineups with >0.01 expected dupes")

    # Greedy selection
    selected = []
    # portfolio_max[i] = current max payout in portfolio for iteration i
    portfolio_max = [0.0] * n_iters

    for step in range(n_portfolio):
        best_idx = -1
        best_marginal = -float("inf")

        for c_idx in range(n_candidates):
            if c_idx in selected:
                continue

            # Marginal gain: how much does adding this lineup improve E[max]?
            marginal = 0.0
            for i in range(n_iters):
                new_max = max(portfolio_max[i], payouts[c_idx][i])
                marginal += (new_max - portfolio_max[i])
            marginal /= n_iters

            if marginal > best_marginal:
                best_marginal = marginal
                best_idx = c_idx

        if best_idx < 0:
            break

        selected.append(best_idx)
        # Update portfolio max
        for i in range(n_iters):
            portfolio_max[i] = max(portfolio_max[i], payouts[best_idx][i])

        avg_portfolio_ev = sum(portfolio_max) / n_iters - entry_fee * (step + 1)
        cand = candidates[best_idx]
        dupes_str = f" dupes={cand.get('expected_dupes', 0):.3f}" if dupe_adjust else ""
        print(f"  Step {step+1}: +lineup {cand['lineup_id']} "
              f"(marginal=${best_marginal:.2f}{dupes_str}) "
              f"portfolio_ev=${avg_portfolio_ev:.2f}")

    return selected


def _poisson_sample(lam, rng):
    """Draw from Poisson(lam) using inverse transform / normal approx."""
    if lam <= 0:
        return 0
    if lam < 30:
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= rng.random()
            if p <= L:
                return k - 1
    else:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))


def select_portfolio(player_results, n_portfolio=20,
                     opponent_mean=330, opponent_std=45,
                     n_iterations=None, seed=42):
    """
    Full portfolio selection pipeline.

    Args:
        player_results: dict of dg_id -> list of DK pts per iteration
        n_portfolio: number of lineups to select
        opponent_mean/std: calibrated opponent distribution
        n_iterations: sim iterations to use
        seed: random seed

    Returns:
        list of selected candidate dicts
    """
    print("Loading candidates...")
    candidates = load_candidates()
    print(f"  {len(candidates)} candidates loaded")

    print("Scoring lineups across sim iterations...")
    scored, n_iters = score_lineups(candidates, player_results, n_iterations)
    print(f"  Scored {len(scored)} lineups x {n_iters} iterations")

    print(f"\nRunning greedy E[max] selection (target: {n_portfolio} lineups)...")
    selected_indices = greedy_emax_select(
        scored, candidates, n_portfolio,
        opponent_mean=opponent_mean,
        opponent_std=opponent_std,
    )

    # Build output
    selected = [candidates[i] for i in selected_indices]

    # Save
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["portfolio_rank", "lineup_id", "strategy", "total_salary",
                         "p1", "p2", "p3", "p4", "p5", "p6"])
        for rank, cand in enumerate(selected, 1):
            writer.writerow([
                rank, cand["lineup_id"], cand["strategy"], cand["total_salary"],
                *cand["names"],
            ])

    print(f"\nSaved {len(selected)} lineups to {OUTPUT}")
    return selected


if __name__ == "__main__":
    print("portfolio_select.py requires player_results from player_sim.")
    print("Run via the main pipeline orchestrator.")

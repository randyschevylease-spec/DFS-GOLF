"""Contest Simulation Engine — Score lineups against shared player sims.

Uses the PlayerSimulation score matrix to:
  1. Score all candidate lineups
  2. Score all opponent lineups
  3. Rank candidates against opponents (vectorized)
  4. Map finish positions to DK payout table
  5. Return payout matrix and ROI array

Key change from original: NO score generation here.
All player scores come from player_sim.PlayerSimulation.
"""
import numpy as np
from player_sim import build_lineup_matrix


def simulate_contest(player_sim, candidates, opponents, players,
                     payout_table, entry_fee):
    """Monte Carlo contest simulation using shared player score matrix.

    For each simulation (row in player_sim.scores):
      1. Score candidate and opponent lineups from shared player draws
      2. Rank each candidate against opponents only
      3. Assign payouts from DK payout table by finish position

    Args:
        player_sim: PlayerSimulation object with (n_sims, n_players) scores
        candidates: list of candidate lineups (each a list of player indices)
        opponents: list of opponent lineups (each a list of player indices)
        players: player dicts (for n_players count)
        payout_table: list of (min_pos, max_pos, prize) from DK API
        entry_fee: contest entry fee in dollars

    Returns:
        payouts: (n_candidates, n_sims) array of dollar payouts per sim
        roi: (n_candidates,) array of mean ROI %
    """
    n_players = len(players)
    n_cands = len(candidates)
    n_opps = len(opponents)
    n_sims = player_sim.n_sims

    # Build lineup matrices
    cand_matrix = build_lineup_matrix(candidates, n_players)
    opp_matrix = build_lineup_matrix(opponents, n_players)

    # Score all lineups against shared player sims
    # player_sim.scores: (n_sims, n_players)
    # cand_matrix: (n_cands, n_players)
    # result: (n_sims, n_cands)
    print(f"  Scoring {n_cands:,} candidates vs {n_opps:,} opponents "
          f"across {n_sims:,} sims...")

    # Build payout lookup sized to n_opps + 1 (candidate + opponents)
    sim_field = n_opps + 1
    payout_by_pos = np.zeros(sim_field + 1, dtype=np.float64)
    for min_pos, max_pos, prize in sorted(payout_table, key=lambda x: x[0]):
        for pos in range(min_pos, min(max_pos, sim_field) + 1):
            payout_by_pos[pos] = prize

    # Process in batches for memory efficiency
    batch_size = min(2000, n_sims)
    payouts = np.zeros((n_cands, n_sims), dtype=np.float64)

    for batch_start in range(0, n_sims, batch_size):
        bs = min(batch_size, n_sims - batch_start)
        batch_scores = player_sim.scores[batch_start:batch_start + bs]

        # Score lineups: (bs, n_players) @ (n_players, n_cands) = (bs, n_cands)
        cand_scores = (batch_scores @ cand_matrix.T)   # (bs, n_cands)
        opp_scores = (batch_scores @ opp_matrix.T)     # (bs, n_opps)

        # Vectorized ranking
        positions = _rank_candidates_vectorized(
            opp_scores, cand_scores, n_opps, sim_field
        )
        payouts[:, batch_start:batch_start + bs] = payout_by_pos[positions].T

    mean_payouts = payouts.mean(axis=1)
    roi = (mean_payouts - entry_fee) / entry_fee * 100

    return payouts, roi


def _rank_candidates_vectorized(opp_scores, cand_scores, n_opps, max_pos):
    """Rank all candidates against opponents for a batch of sims.

    Vectorized: sorts opponents once per batch, then uses searchsorted
    to rank ALL candidates simultaneously.

    Args:
        opp_scores: (bs, n_opps) float32 — opponent lineup scores
        cand_scores: (bs, n_cands) float32 — candidate lineup scores
        n_opps: number of opponents
        max_pos: maximum position value (n_opps + 1)

    Returns:
        positions: (bs, n_cands) int32 — 1-indexed finish positions
    """
    bs, n_cands = cand_scores.shape
    opp_sorted = np.sort(opp_scores, axis=1)  # (bs, n_opps) ascending

    positions = np.empty((bs, n_cands), dtype=np.int32)
    for s in range(bs):
        insert_idx = np.searchsorted(opp_sorted[s], cand_scores[s], side='left')
        positions[s] = n_opps - insert_idx + 1

    np.clip(positions, 1, max_pos, out=positions)
    return positions

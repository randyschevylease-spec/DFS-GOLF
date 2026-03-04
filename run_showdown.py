#!/usr/bin/env python3
"""AceEdge Classic Showdown (Single-Round) DFS Pipeline.

Architecture:
  1. Parse CSV projections (DataGolf upload — NO API calls)
  2. Validate players, derive std devs and ownership
  3. Generate opponent field (archetype-based)
  4. Generate candidate lineups (randomized MIP)
  5. ** PLAYER SIMULATION ** — generate (n_sims, n_players) shared score matrix
     sim count = field_size × 10 (tied to contest size, not arbitrary)
  6. Score ALL lineups (candidates + opponents) against shared player sims
  7. Pre-filter candidates by ROI
  8. Portfolio optimization (greedy/genetic/hybrid via dispatcher)
  9. Export results

Usage:
    python3 -u run_showdown.py --csv projections.csv --contest 188424004
    python3 -u run_showdown.py --csv projections.csv --contest 188424004 --sim-mult 15
    python3 -u run_showdown.py --csv projections.csv --contest 188424004 --method hybrid
    python3 -u run_showdown.py --csv projections.csv --contest 188424004 --entries 150 --pool-size 30000
"""
import sys
import os
import time
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dfs-core'))

from config import (ROSTER_SIZE, SALARY_CAP, SHOWDOWN_SALARY_FLOORS,
                    PLAYER_SIM_MULTIPLIER)
from dk_contests import fetch_contest
from players import parse_projections, validate_players, derive_std_devs, compute_ownership
from player_sim import generate_player_sims, build_lineup_matrix
from candidate_generator import generate_candidates
from field_generator import generate_field, field_to_index_lists
from showdown_engine import simulate_contest
from portfolio_optimizer import optimize_portfolio
from export import export_all
from log_utility import compute_w_star, print_w_star_summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AceEdge Classic Showdown DFS Pipeline")
    parser.add_argument("--csv", required=True, help="DataGolf projections CSV path")
    parser.add_argument("--contest", required=True, type=int, help="DK contest ID")
    parser.add_argument("--entries", type=int, default=150, help="Max entries (default: 150)")
    parser.add_argument("--pool-size", type=int, default=20000, help="MIP solves for candidates")
    parser.add_argument("--diversity", type=float, default=0.4, help="Portfolio diversity weight")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--method", type=str, default="greedy",
                        choices=["greedy", "genetic", "hybrid"],
                        help="Portfolio optimization method (default: greedy)")
    parser.add_argument("--sim-mult", type=int, default=None,
                        help=f"Player sim multiplier (default: {PLAYER_SIM_MULTIPLIER}x field)")
    parser.add_argument("--max-sims", type=int, default=1_500_000,
                        help="Hard cap on simulations (default: 1,500,000)")
    args = parser.parse_args()

    t_start = time.time()

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Load and validate projections
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 1: Load and validate projections")
    print(f"{'='*70}")

    players = parse_projections(args.csv)
    is_valid, issues = validate_players(players)
    for issue in issues:
        print(f"  {issue}")
    if not is_valid:
        print("  FATAL: Player validation failed. Fix CSV and retry.")
        sys.exit(1)

    print(f"  Players: {len(players)}")
    print(f"  Projection range: {min(p['projected_points'] for p in players):.1f} – "
          f"{max(p['projected_points'] for p in players):.1f}")
    print(f"  Salary range: ${min(p['salary'] for p in players):,} – "
          f"${max(p['salary'] for p in players):,}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Fetch contest and derive player attributes
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 2: Contest data + player attributes")
    print(f"{'='*70}")

    contest = fetch_contest(args.contest)
    entry_fee = contest["entry_fee"]
    field_size = contest["entries"]
    max_entries = min(args.entries, contest.get("max_entries_per_user", args.entries))
    payout_table = contest["payouts"]

    print(f"  Contest: {contest['name']}")
    print(f"  Entry fee: ${entry_fee} | Field: {field_size:,} | Max entries: {max_entries}")
    print(f"  Prize pool: ${contest['prize_pool']:,} | 1st: ${contest['first_place_prize']:,}")

    derive_std_devs(players)
    compute_ownership(players)

    waves = [p["wave"] for p in players]

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Generate opponent field
    # ══════════════════════════════════════════════════════════════════
    n_opponents = field_size - max_entries
    print(f"\n{'='*70}")
    print(f"  STEP 3: Generate opponent field ({n_opponents:,} lineups)")
    print(f"{'='*70}")

    t_field = time.time()
    field = generate_field(
        players, n_opponents,
        seed=args.seed,
        min_salary_map=SHOWDOWN_SALARY_FLOORS,
    )
    opponents = field_to_index_lists(field)
    print(f"  Field generated in {time.time()-t_field:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Generate candidate lineups
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 4: Generate candidates ({args.pool_size:,} MIP solves)")
    print(f"{'='*70}")

    t_cand = time.time()
    candidates = generate_candidates(
        players,
        pool_size=args.pool_size,
        candidate_exposure_cap=1.0,
        ceiling_weight=0.0,
        min_proj_pct=0.0,
        seed=args.seed,
    )
    print(f"  Candidates: {len(candidates):,} unique in {time.time()-t_cand:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: PLAYER SIMULATION — shared score matrix
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 5: Player simulation (individual score generation)")
    print(f"{'='*70}")

    player_sim = generate_player_sims(
        players, field_size,
        waves=waves,
        sim_multiplier=args.sim_mult,
        max_sims=args.max_sims,
        seed=args.seed,
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: Contest simulation (score lineups against shared sims)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 6: Simulate contest ({player_sim.n_sims:,} sims)")
    print(f"{'='*70}")

    t_sim = time.time()
    payouts, roi = simulate_contest(
        player_sim, candidates, opponents, players,
        payout_table, entry_fee,
    )
    sim_elapsed = time.time() - t_sim
    print(f"  Simulation complete in {sim_elapsed:.1f}s")

    n_positive = (roi > 0).sum()
    print(f"  Positive ROI candidates: {n_positive:,}/{len(candidates):,}")
    print(f"  Top candidate ROI: {roi.max():.1f}%")
    print(f"  Median candidate ROI: {np.median(roi):.1f}%")

    # w* log-utility scoring
    w_star, p_cash, kelly_frac = compute_w_star(payouts, entry_fee)
    print_w_star_summary(w_star, p_cash, kelly_frac, roi)

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: Pre-filter candidates (by w* growth rate)
    # ══════════════════════════════════════════════════════════════════
    TOP_CANDIDATES = max(max_entries * 10, 3000)
    top_idx = np.argsort(-w_star)[:TOP_CANDIDATES]
    payouts_filtered = payouts[top_idx]
    candidates_filtered = [candidates[i] for i in top_idx]
    roi_filtered = roi[top_idx]
    w_star_filtered = w_star[top_idx]

    print(f"\n  Pre-filtered to top {TOP_CANDIDATES:,} candidates (by w*)")
    finite_w = w_star_filtered[w_star_filtered > -np.inf]
    if len(finite_w) > 0:
        print(f"  w* range: {finite_w.min():.6f} – {finite_w.max():.6f}")
    print(f"  ROI range: {roi_filtered.min():.1f}% – {roi_filtered.max():.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # STEP 8: Portfolio optimization (via dispatcher)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 8: Portfolio optimization ({max_entries} lineups, method={args.method})")
    print(f"{'='*70}")

    t_port = time.time()
    result = optimize_portfolio(
        payouts_filtered, entry_fee, max_entries,
        candidates_filtered, len(players),
        method=args.method,
        diversity_weight=args.diversity,
        waves=waves,
    )
    port_elapsed = time.time() - t_port
    print(f"  Portfolio optimization: {port_elapsed:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════
    sel = result.selected_indices
    sel_payouts = payouts_filtered[sel]
    portfolio_max = sel_payouts.max(axis=0)
    portfolio_cost = entry_fee * len(sel)
    portfolio_roi = (portfolio_max.mean() - portfolio_cost) / portfolio_cost * 100

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Lineups: {len(sel)}")
    print(f"  Portfolio ROI: {portfolio_roi:+.1f}%")
    print(f"  Mean best payout: ${portfolio_max.mean():.2f}")
    print(f"  Investment: ${portfolio_cost:.0f}")
    print(f"  Cash rate: {(portfolio_max > 0).mean()*100:.1f}%")

    # Exposure
    player_counts = Counter()
    for si in sel:
        for idx in candidates_filtered[si]:
            player_counts[idx] += 1

    print(f"\n  Top exposures:")
    for idx, count in player_counts.most_common(15):
        pct = count / len(sel) * 100
        print(f"    {players[idx]['name']:<25} {pct:5.1f}% ({count}/{len(sel)})")
    print(f"  Unique players: {len(player_counts)}/{len(players)}")

    total_elapsed = time.time() - t_start
    print(f"\n  Total pipeline: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"  Player sims: {player_sim.n_sims:,} ({player_sim.generation_time:.1f}s)")

    # ══════════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════════
    export_all(
        sel, candidates_filtered, players, payouts_filtered,
        entry_fee, args.contest, portfolio_roi, sim_elapsed,
    )


if __name__ == "__main__":
    main()

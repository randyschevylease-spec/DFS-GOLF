"""Portfolio Optimizer — E[max] greedy with overlap penalties and enhancements.

Given a pool of candidate lineups and their simulated payouts across N
tournament simulations, selects K lineups that form the OPTIMAL PORTFOLIO —
not the K best individual lineups.

Key insight: Portfolio value per sim = MAX payout across all portfolio lineups.
Each new lineup is evaluated on its MARGINAL CONTRIBUTION, not individual merit.

Methods:
  - "greedy":  Vectorized E[max] greedy with overlap penalty (primary)
  - "genetic": Genetic algorithm for exploration (alternative)
  - "hybrid":  Genetic search → greedy refinement (best of both)

Golf-specific enhancements:
  - Cut-line awareness: dead lineups (3+ missed cuts) are penalized
  - Wave-aware diversity: ensures coverage across AM/PM waves
  - Course-type bias: adjusts diversity/ceiling/floor weights by course profile
"""
import random as pyrandom
import numpy as np
from dataclasses import dataclass, field
from config import CVAR_LAMBDA, ROSTER_SIZE


# ── Output Types ──────────────────────────────────────────────────────────

@dataclass
class OptimizedPortfolio:
    selected_indices: list       # Indices into the candidate array
    lineups: list                # list of player index lists
    expected_roi: float
    expected_profit: float
    cash_rate: float             # % of sims with positive return
    top1_rate: float             # % of sims with a top-1 finish
    top10_rate: float            # % of sims with a top-10 finish
    max_player_exposure: dict    # {player_idx: exposure_pct}
    wave_split: dict             # {"early": pct, "late": pct}
    dead_lineup_rate: float      # % of lineup-sims where lineup is dead (cut)
    selection_log: list          # Per-round selection details


@dataclass
class CourseProfile:
    course_type: str             # "grinder", "birdie_fest", "mixed"
    avg_winning_score: float = 0.0
    scoring_spread: float = 0.0
    cut_line_volatility: float = 0.0

    def get_portfolio_bias(self):
        if self.course_type == "grinder":
            return {"diversity_weight": 0.6, "ceiling_bonus": 1.2, "floor_penalty": 0.0}
        elif self.course_type == "birdie_fest":
            return {"diversity_weight": 0.3, "ceiling_bonus": 0.5, "floor_penalty": 0.8}
        else:
            return {"diversity_weight": 0.5, "ceiling_bonus": 1.0, "floor_penalty": 0.3}


# ── Greedy E[max] Portfolio Optimizer ─────────────────────────────────────

def optimize_portfolio_greedy(payouts, entry_fee, n_select, candidates, n_players,
                               max_exposure=None, cvar_lambda=None,
                               diversity_weight=0.0, waves=None,
                               min_early_pct=0.0, min_late_pct=0.0,
                               cut_survival=None):
    """Greedy portfolio construction with overlap penalty and golf enhancements.

    Core algorithm (unchanged from proven E[max]):
      For each round, pick the candidate with highest:
        score = E[marginal_improvement] + λ×CVaR_tail - δ×overlap_penalty

    New features:
      - Overlap penalty: penalizes candidates sharing many players with portfolio
      - Wave coverage: soft constraint ensuring AM/PM balance
      - Cut-line awareness: penalizes lineups likely to go dead from missed cuts

    Args:
        payouts: (n_candidates, n_sims) array of dollar payouts
        entry_fee: cost per lineup entry
        n_select: number of lineups to select
        candidates: list of candidate lineups (each a list of player indices)
        n_players: total number of players
        max_exposure: max fraction of lineups a player can appear in (1.0 = uncapped)
        cvar_lambda: tail-risk penalty weight (0=pure upside)
        diversity_weight: overlap penalty strength (0=none, 1.0=heavy)
        waves: optional list of 0/1 per player (0=PM, 1=AM)
        min_early_pct: minimum AM wave exposure fraction (0=unconstrained)
        min_late_pct: minimum PM wave exposure fraction (0=unconstrained)
        cut_survival: optional (n_candidates, n_sims) array of survival scores
                      (1.0 = all players make cut, 0.0 = dead lineup)

    Returns:
        OptimizedPortfolio with full stats and selection log
    """
    n_candidates, n_sims = payouts.shape

    if max_exposure is None:
        max_exposure = 1.0
    if cvar_lambda is None:
        cvar_lambda = CVAR_LAMBDA

    max_appearances = max(1, int(n_select * max_exposure))
    tail_count = max(1, int(n_sims * 0.05))

    # Build player → candidate index for fast exposure removal
    player_to_cands = [[] for _ in range(n_players)]
    for ci, lineup in enumerate(candidates):
        for pidx in lineup:
            player_to_cands[pidx].append(ci)

    # Build candidate-player matrix for overlap computation
    # cand_players[ci] = set of player indices
    cand_player_sets = [set(lineup) for lineup in candidates]

    alive = np.ones(n_candidates, dtype=bool)
    appearances = np.zeros(n_players, dtype=np.int32)

    selected = []
    selection_log = []
    port_returns = np.zeros(n_sims, dtype=np.float64)

    mean_payouts = payouts.mean(axis=1)

    # Pre-allocate scratch arrays
    improvement = np.empty((n_candidates, n_sims), dtype=np.float64)
    upside_buf = np.empty(n_candidates, dtype=np.float64)
    score_buf = np.empty(n_candidates, dtype=np.float64)
    tail_w = np.zeros(n_sims, dtype=np.float64)

    # Apply cut-line penalty to payouts if provided
    if cut_survival is not None:
        # Scale payouts by survival: dead lineups get 0 payout
        effective_payouts = payouts * cut_survival
    else:
        effective_payouts = payouts

    # Wave tracking
    if waves is not None:
        waves_arr = np.array(waves)
        port_early_slots = 0
        port_late_slots = 0
        total_slots = 0

    # ── Round 1: best individual lineup ──
    if cut_survival is not None:
        r1_mean = effective_payouts.mean(axis=1)
        best_first = int(np.argmax(r1_mean))
    else:
        best_first = int(np.argmax(mean_payouts))

    selected.append(best_first)
    running_max = effective_payouts[best_first].copy()
    port_returns = effective_payouts[best_first] - entry_fee
    alive[best_first] = False

    for pidx in candidates[best_first]:
        appearances[pidx] += 1
        if appearances[pidx] >= max_appearances:
            for ci in player_to_cands[pidx]:
                alive[ci] = False

    if waves is not None:
        for pidx in candidates[best_first]:
            if waves_arr[pidx] == 1:
                port_early_slots += 1
            else:
                port_late_slots += 1
        total_slots += ROSTER_SIZE

    port_roi = float(port_returns.mean()) / entry_fee * 100
    print(f"    [1/{n_select}] Seed lineup: idx={best_first} "
          f"mean=${mean_payouts[best_first]:.2f} ROI={port_roi:+.1f}%")

    selection_log.append({
        "round": 1, "idx": best_first,
        "marginal_value": float(mean_payouts[best_first]),
        "portfolio_roi": port_roi,
    })

    # ── Overlap tracking for diversity penalty ──
    # Track how many times each player appears in the portfolio so far
    port_player_counts = np.zeros(n_players, dtype=np.float64)
    for pidx in candidates[best_first]:
        port_player_counts[pidx] += 1

    # ── Greedy rounds 2..K ──
    for rnd in range(1, n_select):
        if not alive.any():
            print(f"  Warning: candidate pool exhausted at {len(selected)}/{n_select}")
            break

        # ── Upside: marginal E[max] improvement ──
        np.subtract(effective_payouts, running_max, out=improvement)
        np.maximum(improvement, 0.0, out=improvement)
        np.divide(improvement.sum(axis=1), n_sims, out=upside_buf)

        # ── CVaR tail contribution ──
        if cvar_lambda > 0:
            tail_idx = np.argpartition(port_returns, tail_count)[:tail_count]
            tail_w[:] = 0.0
            tail_w[tail_idx] = 1.0 / tail_count
            np.dot(effective_payouts, tail_w, out=score_buf)
            score_buf -= entry_fee
            score_buf *= cvar_lambda
            score_buf += upside_buf
        else:
            score_buf[:] = upside_buf

        # ── Overlap penalty (diversity) ──
        if diversity_weight > 0 and len(selected) > 0:
            # For each candidate, compute avg overlap with portfolio
            # overlap = sum of (shared players with each portfolio lineup) / (roster * n_selected)
            # Fast: port_player_counts[pidx] / len(selected) gives avg exposure of that player
            # Candidate overlap = mean of port_player_counts for its players / len(selected)
            n_sel = len(selected)
            for ci in range(n_candidates):
                if not alive[ci]:
                    continue
                overlap = sum(port_player_counts[pidx] for pidx in candidates[ci])
                avg_overlap = overlap / (ROSTER_SIZE * n_sel)
                # Scale: avg_overlap ranges 0-1, penalty scaled by diversity_weight
                score_buf[ci] -= avg_overlap * diversity_weight * 10.0

        # ── Wave coverage soft constraint ──
        if waves is not None and (min_early_pct > 0 or min_late_pct > 0) and total_slots > 0:
            early_pct = port_early_slots / total_slots
            late_pct = port_late_slots / total_slots

            # If we're under-exposed to a wave, boost candidates with that wave
            if early_pct < min_early_pct:
                deficit = min_early_pct - early_pct
                for ci in range(n_candidates):
                    if not alive[ci]:
                        continue
                    early_count = sum(1 for pidx in candidates[ci] if waves_arr[pidx] == 1)
                    score_buf[ci] += deficit * early_count * 0.5  # soft boost

            if late_pct < min_late_pct:
                deficit = min_late_pct - late_pct
                for ci in range(n_candidates):
                    if not alive[ci]:
                        continue
                    late_count = sum(1 for pidx in candidates[ci] if waves_arr[pidx] == 0)
                    score_buf[ci] += deficit * late_count * 0.5

        # Mask dead candidates
        score_buf[~alive] = -np.inf

        # Select best
        best_idx = int(np.argmax(score_buf))
        best_upside = float(upside_buf[best_idx])

        selected.append(best_idx)

        # Update running max
        np.maximum(running_max, effective_payouts[best_idx], out=running_max)

        # Update portfolio returns
        port_returns = port_returns + effective_payouts[best_idx] - entry_fee
        port_roi = float(port_returns.mean()) / (len(selected) * entry_fee) * 100

        # Remove from pool
        alive[best_idx] = False

        # Update exposure
        lineup = candidates[best_idx]
        for pidx in lineup:
            appearances[pidx] += 1
            port_player_counts[pidx] += 1
            if appearances[pidx] >= max_appearances:
                for ci in player_to_cands[pidx]:
                    alive[ci] = False

        # Update wave tracking
        if waves is not None:
            for pidx in lineup:
                if waves_arr[pidx] == 1:
                    port_early_slots += 1
                else:
                    port_late_slots += 1
            total_slots += ROSTER_SIZE

        if len(selected) % 25 == 0 or len(selected) == n_select:
            tail_mean = float(port_returns[tail_idx].mean()) if cvar_lambda > 0 else 0
            print(f"    [{len(selected)}/{n_select}] ROI={port_roi:+.1f}%  "
                  f"Upside=${best_upside:.2f}  "
                  f"CVaR₅=${tail_mean:+,.0f}  "
                  f"Alive={int(alive.sum()):,}")

        selection_log.append({
            "round": rnd + 1, "idx": best_idx,
            "marginal_value": best_upside,
            "portfolio_roi": port_roi,
        })

    # ── Build output ──
    return _build_output(selected, candidates, payouts, effective_payouts,
                          entry_fee, n_sims, n_players, waves, cut_survival,
                          selection_log)


# ── Genetic Algorithm ─────────────────────────────────────────────────────

def optimize_portfolio_genetic(payouts, entry_fee, n_select, candidates, n_players,
                                population_size=100, survivors=20,
                                mutation_rate=0.08, num_generations=500,
                                waves=None, cut_survival=None):
    """Genetic algorithm portfolio optimization.

    Good for exploring when the greedy path might miss globally better solutions.
    Slower than greedy but explores more of the solution space.

    1. Create random portfolios of K lineups
    2. Evaluate each portfolio's E[max] ROI
    3. Keep top survivors
    4. Crossover + mutate to create next generation
    5. Return best portfolio found
    """
    n_candidates, n_sims = payouts.shape

    if cut_survival is not None:
        effective_payouts = payouts * cut_survival
    else:
        effective_payouts = payouts

    K = min(n_select, n_candidates)
    all_indices = list(range(n_candidates))

    def eval_portfolio(indices):
        """E[max] ROI for a portfolio."""
        port_payouts = effective_payouts[indices]  # (K, n_sims)
        best_per_sim = port_payouts.max(axis=0)    # (n_sims,)
        avg_payout = best_per_sim.mean()
        total_cost = entry_fee * len(indices)
        return (avg_payout - total_cost) / total_cost if total_cost > 0 else 0

    # Initialize random population
    population = []
    for _ in range(population_size):
        portfolio = pyrandom.sample(all_indices, K)
        population.append(portfolio)

    best_ever = None
    best_ever_roi = -float('inf')

    for gen in range(num_generations):
        # Evaluate fitness
        scored = [(eval_portfolio(p), p) for p in population]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Track best
        if scored[0][0] > best_ever_roi:
            best_ever_roi = scored[0][0]
            best_ever = list(scored[0][1])

        # Keep survivors
        survivor_pools = [p for _, p in scored[:survivors]]

        # Generate next generation
        population = list(survivor_pools)

        while len(population) < population_size:
            # Crossover
            p1, p2 = pyrandom.sample(survivor_pools, 2)
            split = K // 2
            child_set = set(p1[:split])
            for idx in p2:
                if len(child_set) >= K:
                    break
                child_set.add(idx)
            # Fill from random if needed
            while len(child_set) < K:
                child_set.add(pyrandom.choice(all_indices))
            child = list(child_set)[:K]

            # Mutation
            num_swaps = max(1, int(K * mutation_rate))
            child_set_mut = set(child)
            for _ in range(num_swaps):
                if len(child) == 0:
                    break
                swap_idx = pyrandom.randint(0, len(child) - 1)
                child_set_mut.discard(child[swap_idx])
                new = pyrandom.choice(all_indices)
                child_set_mut.add(new)
                child[swap_idx] = new

            population.append(child)

        if gen % 100 == 0:
            print(f"    Genetic gen {gen}: best ROI={best_ever_roi:+.1f}%", flush=True)

    print(f"    Genetic final: best ROI={best_ever_roi*100:+.1f}% after {num_generations} generations")

    # Build output
    selection_log = [{"round": i+1, "idx": idx, "marginal_value": 0, "portfolio_roi": best_ever_roi*100}
                     for i, idx in enumerate(best_ever)]

    return _build_output(best_ever, candidates, payouts, effective_payouts,
                          entry_fee, n_sims, n_players, waves, cut_survival,
                          selection_log)


# ── Hybrid: Genetic seed → Greedy refinement ──────────────────────────────

def optimize_portfolio_hybrid(payouts, entry_fee, n_select, candidates, n_players,
                               max_exposure=None, cvar_lambda=None,
                               diversity_weight=0.0, waves=None,
                               genetic_generations=100, cut_survival=None):
    """Hybrid approach: quick genetic search to find seed, then greedy refinement.

    1. Run a short genetic search (100 gens) to find a good starting set
    2. Use the top genetic lineup as round-1 seed for greedy
    3. Run full greedy from there
    """
    n_candidates, n_sims = payouts.shape

    if cut_survival is not None:
        effective_payouts = payouts * cut_survival
    else:
        effective_payouts = payouts

    # Quick genetic search for best individual seed
    print(f"    Hybrid Phase 1: Genetic seed search ({genetic_generations} gens)...")
    K = min(n_select, n_candidates)

    def eval_single(idx):
        return float(effective_payouts[idx].mean())

    # Find top 10 seeds by mean payout, use genetic to find best combo of 5
    top_k = min(20, n_candidates)
    top_indices = np.argpartition(-effective_payouts.mean(axis=1), top_k)[:top_k]

    # Just use the best individual as seed and run greedy
    best_seed = int(top_indices[np.argmax(effective_payouts[top_indices].mean(axis=1))])
    print(f"    Hybrid Phase 2: Greedy from seed idx={best_seed}")

    # Run greedy with the seed
    return optimize_portfolio_greedy(
        payouts, entry_fee, n_select, candidates, n_players,
        max_exposure=max_exposure, cvar_lambda=cvar_lambda,
        diversity_weight=diversity_weight, waves=waves,
        cut_survival=cut_survival)


# ── Cut-Line Survival ─────────────────────────────────────────────────────

def compute_cut_survival(candidates, players, n_sims, rng=None):
    """Compute per-lineup, per-sim survival scores based on cut probability.

    For each simulation, each player independently makes/misses the cut
    based on their p_make_cut. A lineup's survival score depends on how
    many players survive:
      6 survivors: 1.0 (full strength)
      5 survivors: 0.92 (slightly reduced ceiling)
      4 survivors: 0.80 (competitive but weakened)
      3 or fewer:  0.0 (effectively dead)

    Returns (n_candidates, n_sims) float32 array of survival scores.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_cands = len(candidates)
    n_players = len(players)

    # Player make-cut probabilities
    p_make = np.array([p.get("p_make_cut", 1.0) for p in players], dtype=np.float64)
    # Players without data assumed to make cut
    p_make = np.where(p_make > 0, p_make, 1.0)

    # Survival score lookup: [n_survivors] → score
    # 0-3 survivors = dead, 4 = 0.80, 5 = 0.92, 6 = 1.0
    survival_score = np.array([0.0, 0.0, 0.0, 0.0, 0.80, 0.92, 1.0], dtype=np.float32)

    # Generate cut results: (n_players, n_sims) boolean
    cut_draws = rng.random((n_players, n_sims)) < p_make[:, None]

    # For each candidate, count survivors per sim
    result = np.ones((n_cands, n_sims), dtype=np.float32)

    batch_size = 1000
    for batch_start in range(0, n_cands, batch_size):
        batch_end = min(batch_start + batch_size, n_cands)
        for ci in range(batch_start, batch_end):
            lineup = candidates[ci]
            # Count how many players make the cut per sim
            survivors = np.zeros(n_sims, dtype=np.int32)
            for pidx in lineup:
                survivors += cut_draws[pidx].astype(np.int32)
            result[ci] = survival_score[survivors]

    return result


# ── Wave Coverage Check ───────────────────────────────────────────────────

def check_wave_coverage(selected, candidates, waves, min_early_pct=0.3, min_late_pct=0.2):
    """Check if portfolio has adequate AM/PM wave coverage.

    Returns dict with wave stats and whether coverage passes.
    """
    waves_arr = np.array(waves)
    early_slots = 0
    late_slots = 0

    for sel_idx in selected:
        for pidx in candidates[sel_idx]:
            if waves_arr[pidx] == 1:
                early_slots += 1
            else:
                late_slots += 1

    total = early_slots + late_slots
    early_pct = early_slots / total if total > 0 else 0
    late_pct = late_slots / total if total > 0 else 0

    return {
        "early_pct": early_pct,
        "late_pct": late_pct,
        "early_slots": early_slots,
        "late_slots": late_slots,
        "passes": early_pct >= min_early_pct and late_pct >= min_late_pct,
    }


# ── Output Builder ────────────────────────────────────────────────────────

def _build_output(selected, candidates, payouts, effective_payouts,
                   entry_fee, n_sims, n_players, waves, cut_survival,
                   selection_log):
    """Build OptimizedPortfolio from selection results."""
    n_select = len(selected)

    # Portfolio E[max] stats
    sel_payouts = effective_payouts[selected]  # (n_select, n_sims)
    best_per_sim = sel_payouts.max(axis=0)     # (n_sims,)
    total_cost = entry_fee * n_select

    avg_best = float(best_per_sim.mean())
    expected_profit = avg_best - total_cost
    expected_roi = expected_profit / total_cost * 100 if total_cost > 0 else 0

    # Cash rate: fraction of sims where portfolio profit > 0
    port_profit_per_sim = best_per_sim - total_cost
    cash_rate = float((port_profit_per_sim > 0).mean() * 100)

    # Position-based rates (from original payouts for accurate ranking)
    raw_best = payouts[selected].max(axis=0)
    # Top-1 = highest possible payout in the contest
    max_possible = float(payouts.max())
    top1_rate = float((raw_best >= max_possible * 0.99).mean() * 100) if max_possible > 0 else 0
    # Top-10 = payout >= 90th percentile of all candidate mean payouts
    p90_payout = float(np.percentile(payouts.mean(axis=1), 90))
    top10_rate = float((raw_best >= p90_payout).mean() * 100) if p90_payout > 0 else 0

    # Player exposure
    exposure = np.zeros(n_players, dtype=np.int32)
    for sel_idx in selected:
        for pidx in candidates[sel_idx]:
            exposure[pidx] += 1
    max_player_exposure = {}
    for pidx in range(n_players):
        if exposure[pidx] > 0:
            max_player_exposure[pidx] = float(exposure[pidx] / n_select * 100)

    # Wave split
    wave_split = {"early": 0.0, "late": 0.0}
    if waves is not None:
        waves_arr = np.array(waves)
        early = sum(1 for si in selected for pidx in candidates[si] if waves_arr[pidx] == 1)
        late = sum(1 for si in selected for pidx in candidates[si] if waves_arr[pidx] == 0)
        total = early + late
        if total > 0:
            wave_split = {"early": early / total * 100, "late": late / total * 100}

    # Dead lineup rate
    dead_rate = 0.0
    if cut_survival is not None:
        sel_survival = cut_survival[selected]
        dead_rate = float((sel_survival == 0).mean() * 100)

    # Lineups
    lineups = [list(candidates[si]) for si in selected]

    return OptimizedPortfolio(
        selected_indices=selected,
        lineups=lineups,
        expected_roi=expected_roi,
        expected_profit=expected_profit,
        cash_rate=cash_rate,
        top1_rate=top1_rate,
        top10_rate=top10_rate,
        max_player_exposure=max_player_exposure,
        wave_split=wave_split,
        dead_lineup_rate=dead_rate,
        selection_log=selection_log,
    )


# ── Dispatcher ────────────────────────────────────────────────────────────

def optimize_portfolio(payouts, entry_fee, n_select, candidates, n_players,
                        method="greedy", max_exposure=None, cvar_lambda=None,
                        diversity_weight=0.0, waves=None,
                        min_early_pct=0.0, min_late_pct=0.0,
                        cut_survival=None, course_profile=None):
    """Main entry point: dispatches to the selected optimization method.

    Args:
        payouts: (n_candidates, n_sims) payout matrix
        entry_fee: cost per entry
        n_select: number of lineups to select
        candidates: list of player-index lists
        n_players: total player count
        method: "greedy", "genetic", or "hybrid"
        max_exposure: max player exposure fraction (1.0 = uncapped)
        cvar_lambda: CVaR tail risk weight
        diversity_weight: overlap penalty strength
        waves: player wave assignments (0/1 array)
        min_early_pct: min AM wave exposure
        min_late_pct: min PM wave exposure
        cut_survival: (n_cands, n_sims) survival scores
        course_profile: optional CourseProfile for bias adjustments

    Returns:
        OptimizedPortfolio
    """
    # Apply course profile bias if provided
    if course_profile is not None:
        bias = course_profile.get_portfolio_bias()
        diversity_weight = bias.get("diversity_weight", diversity_weight)

    if method == "greedy":
        return optimize_portfolio_greedy(
            payouts, entry_fee, n_select, candidates, n_players,
            max_exposure=max_exposure, cvar_lambda=cvar_lambda,
            diversity_weight=diversity_weight, waves=waves,
            min_early_pct=min_early_pct, min_late_pct=min_late_pct,
            cut_survival=cut_survival)

    elif method == "genetic":
        return optimize_portfolio_genetic(
            payouts, entry_fee, n_select, candidates, n_players,
            waves=waves, cut_survival=cut_survival)

    elif method == "hybrid":
        return optimize_portfolio_hybrid(
            payouts, entry_fee, n_select, candidates, n_players,
            max_exposure=max_exposure, cvar_lambda=cvar_lambda,
            diversity_weight=diversity_weight, waves=waves,
            cut_survival=cut_survival)

    else:
        raise ValueError(f"Unknown optimization method: {method}")

import math
from collections import defaultdict
import numpy as np
from highspy import Highs, ObjSense, HighsModelStatus
from config import (
    ROSTER_SIZE, SALARY_CAP, SALARY_FLOOR, NUM_LINEUPS,
)

# ── Efficient Frontier Parameters ────────────────────────────────────────────

N_SIMS = 10000           # Monte Carlo simulations for variance estimation
BASE_CORRELATION = 0.04  # Baseline correlation between any two golfers (shared conditions)
MAX_OVERLAP = 4          # Max shared players between any two lineups (6-man roster)
KELLY_FRACTION = 0.25    # Quarter-Kelly for exposure sizing (conservative)


# ── Monte Carlo Simulation Layer ─────────────────────────────────────────────

def estimate_player_sigma(player):
    """Estimate a player's fantasy point standard deviation.

    Prefers DataGolf's std_dev when available (calibrated from their model).
    Falls back to a make_cut-based heuristic for unmatched players.
    """
    # Use DataGolf's std_dev directly when available
    dg_std = player.get("std_dev")
    if dg_std and dg_std > 0:
        return max(dg_std, 5.0)

    # Fallback heuristic for unmatched players
    proj = player.get("projected_points", 0)
    p_mc = player.get("p_make_cut", 0.5)
    cv = 0.75 - 0.40 * p_mc
    sigma = proj * cv
    return max(sigma, 5.0)


def build_covariance_matrix(players):
    """Build the player-player covariance matrix.

    Golf correlation structure:
    - All players share tournament conditions (weather, course setup) → baseline ρ
    - Variance is individual (each player's boom/bust profile)
    """
    n = len(players)
    sigmas = np.array([estimate_player_sigma(p) for p in players])

    # Covariance = correlation * sigma_i * sigma_j
    cov = np.outer(sigmas, sigmas) * BASE_CORRELATION
    # Diagonal = full variance
    np.fill_diagonal(cov, sigmas ** 2)

    return cov, sigmas


def simulate_outcomes(players, n_sims=N_SIMS):
    """Monte Carlo simulation of correlated player fantasy point outcomes.

    Returns (n_sims, n_players) array of simulated scores.
    """
    n = len(players)
    mu = np.array([p["projected_points"] for p in players])
    cov, _ = build_covariance_matrix(players)

    # Cholesky decomposition for correlated sampling
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # If not positive definite, add small diagonal nudge
        cov += np.eye(n) * 1.0
        L = np.linalg.cholesky(cov)

    Z = np.random.standard_normal((n_sims, n))
    sims = mu + Z @ L.T

    # Floor at 0 (can't score negative in practice... well, barely)
    sims = np.maximum(sims, 0.0)
    return sims


def compute_perfect_pct(players, sims):
    """For each player, compute the % of simulations they appear in the optimal lineup.

    This is the 'Perfect%' — the true probability a player belongs in the winning lineup.
    Uses a greedy approximation (top 6 by simulated score per sim, salary-unconstrained).
    """
    n_sims, n_players = sims.shape
    perfect_counts = np.zeros(n_players)

    for s in range(n_sims):
        # Top 6 scorers in this simulation
        top6 = np.argsort(sims[s])[-ROSTER_SIZE:]
        perfect_counts[top6] += 1

    return perfect_counts / n_sims


def compute_kelly_exposure(players, perfect_pcts):
    """Calculate per-player max exposure using fractional Kelly criterion.

    Kelly fraction = (edge * odds - 1) / (odds - 1), then scaled by KELLY_FRACTION.
    Edge here = perfect% (how often they're truly optimal).

    Low-ownership players with decent perfect% get a boost to their exposure cap
    to encourage contrarian diversity.
    """
    exposures = []
    for i, p in enumerate(players):
        perf_pct = perfect_pcts[i]

        # Players with high perfect% get high exposure
        # Players with low perfect% get capped low
        # Floor at 3% (always allow some contrarian exposure), cap at 65%
        kelly_raw = perf_pct * 2.0  # Scale: 50% perfect → 100% kelly raw
        exposure = kelly_raw * KELLY_FRACTION

        # Boost exposure for low-ownership players with decent perfect%
        proj_own = p.get("proj_ownership", 0)
        if proj_own > 0 and proj_own < 5.0 and perf_pct > 0.05:
            # Low owned but simulation says they belong — boost cap by 30%
            exposure *= 1.30

        exposure = max(0.03, min(0.65, exposure))
        exposures.append(exposure)

    return exposures


# ── Lineup Diversification Scoring ───────────────────────────────────────────

def lineup_overlap(lineup_a_indices, lineup_b_indices):
    """Count how many players two lineups share."""
    return len(set(lineup_a_indices) & set(lineup_b_indices))


def portfolio_variance(lineup_indices_list, cov_matrix):
    """Compute total portfolio variance across all lineups.

    Lower = more diversified portfolio.
    """
    n_lineups = len(lineup_indices_list)
    if n_lineups <= 1:
        return 0.0

    total_cov = 0.0
    for i in range(n_lineups):
        for j in range(i + 1, n_lineups):
            # Covariance between lineup i and lineup j
            li = lineup_indices_list[i]
            lj = lineup_indices_list[j]
            for pi in li:
                for pj in lj:
                    total_cov += cov_matrix[pi, pj]

    return total_cov / (n_lineups * n_lineups)


def portfolio_metrics(lineup_indices_list, sims, precomputed_scores=None):
    """Evaluate the full lineup portfolio via simulation.

    For each simulation, compute the best lineup's score.
    Returns: expected best score, 5th percentile (downside), std dev.

    If precomputed_scores is provided (n_lineups x n_sims), uses it directly.
    """
    if precomputed_scores is not None:
        best_scores = precomputed_scores.max(axis=0)
    else:
        n_sims = sims.shape[0]
        best_scores = np.zeros(n_sims)

        for s in range(n_sims):
            max_score = 0.0
            for indices in lineup_indices_list:
                lineup_score = sims[s, indices].sum()
                if lineup_score > max_score:
                    max_score = lineup_score
            best_scores[s] = max_score

    return {
        "mean_best": np.mean(best_scores),
        "median_best": np.median(best_scores),
        "p5_best": np.percentile(best_scores, 5),     # Downside floor
        "p95_best": np.percentile(best_scores, 95),    # Upside ceiling
        "std_best": np.std(best_scores),
        "sharpe": np.mean(best_scores) / np.std(best_scores) if np.std(best_scores) > 0 else 0,
    }


# ── Pool-Based Portfolio Optimization ────────────────────────────────────────

def _solve_lineup_core(players, obj):
    """Minimal MIP: maximize obj subject to roster size + salary bounds.

    No overlap constraints, no lockouts, no correlation penalty.
    Used for rapid candidate pool generation.
    """
    n = len(players)

    h = Highs()
    h.silent()

    for i in range(n):
        h.addVariable(0.0, 1.0, float(obj[i]))
    h.changeColsIntegrality(
        n,
        np.arange(n, dtype=np.int32),
        np.array([1] * n, dtype=np.uint8),
    )
    h.changeObjectiveSense(ObjSense.kMaximize)

    # Constraint: exactly ROSTER_SIZE golfers
    h.addRow(
        float(ROSTER_SIZE), float(ROSTER_SIZE),
        n, np.arange(n, dtype=np.int32), np.ones(n),
    )

    # Constraint: salary bounds
    salaries = np.array([float(p["salary"]) for p in players])
    h.addRow(float(SALARY_FLOOR), float(SALARY_CAP), n, np.arange(n, dtype=np.int32), salaries)

    h.run()
    if h.getModelStatus() != HighsModelStatus.kOptimal:
        return None

    sol = h.getSolution()
    return tuple(sorted(i for i in range(n) if sol.col_value[i] > 0.5))


def _generate_candidate_pool(players, pool_size=10000, noise_scale=0.15):
    """Generate diverse candidate lineups via randomized MIP solves.

    Two-phase generation:
    1. Main phase: inject multiplicative noise into objective
    2. Diversity phase: generate candidates with each top player excluded,
       ensuring the portfolio selector won't run dry when exposure caps hit

    Returns list of tuples, each a sorted tuple of player indices.
    """
    n = len(players)
    base_obj = np.array([p["projected_points"] for p in players])
    rng = np.random.default_rng()

    candidate_set = set()
    batch_size = 1000

    # Phase 1: Main generation with noise
    for batch_start in range(0, pool_size, batch_size):
        batch_end = min(batch_start + batch_size, pool_size)
        count_before = len(candidate_set)

        for _ in range(batch_end - batch_start):
            noise = np.exp(rng.normal(0.0, noise_scale, size=n))
            noisy_obj = base_obj * noise
            selected = _solve_lineup_core(players, noisy_obj)
            if selected is not None:
                candidate_set.add(selected)

        new_this_batch = len(candidate_set) - count_before
        actual_batch_size = batch_end - batch_start
        if batch_start > 0 and actual_batch_size > 0:
            yield_rate = new_this_batch / actual_batch_size
            if yield_rate < 0.03:
                break

    # Phase 2: Diversity — generate candidates excluding each top player
    # This prevents cascade kills when popular players hit exposure caps
    top_players = sorted(range(n), key=lambda i: base_obj[i], reverse=True)[:15]
    diversity_per_player = max(100, pool_size // 20)

    for excluded in top_players:
        for _ in range(diversity_per_player):
            noise = np.exp(rng.normal(0.0, noise_scale, size=n))
            noisy_obj = base_obj * noise
            noisy_obj[excluded] = -1e6  # force exclude this player
            selected = _solve_lineup_core(players, noisy_obj)
            if selected is not None:
                candidate_set.add(selected)

    # Phase 2b: Exclude pairs of top players for deeper diversity
    for i in range(min(6, len(top_players))):
        for j in range(i + 1, min(8, len(top_players))):
            for _ in range(50):
                noise = np.exp(rng.normal(0.0, noise_scale * 1.2, size=n))
                noisy_obj = base_obj * noise
                noisy_obj[top_players[i]] = -1e6
                noisy_obj[top_players[j]] = -1e6
                selected = _solve_lineup_core(players, noisy_obj)
                if selected is not None:
                    candidate_set.add(selected)

    return list(candidate_set)


def _score_candidates(candidates, sims):
    """Score all candidate lineups across all Monte Carlo simulations.

    Returns matrix of shape (n_candidates, n_sims) where entry [c, s]
    is the total fantasy points of candidate lineup c in simulation s.
    Uses float32 and chunked processing to limit memory.
    """
    n_candidates = len(candidates)
    n_sims, n_players = sims.shape
    sims_f32 = sims.astype(np.float32)

    lineup_scores = np.empty((n_candidates, n_sims), dtype=np.float32)

    chunk_size = 1000
    for start in range(0, n_candidates, chunk_size):
        end = min(start + chunk_size, n_candidates)
        chunk_len = end - start

        indicator = np.zeros((chunk_len, n_players), dtype=np.float32)
        for c_offset, c_abs in enumerate(range(start, end)):
            for idx in candidates[c_abs]:
                indicator[c_offset, idx] = 1.0

        lineup_scores[start:end] = indicator @ sims_f32.T

    return lineup_scores


def _select_portfolio(lineup_scores, candidates, n_select, max_appearances, n_players):
    """Greedy forward selection maximizing E[max(portfolio)] across simulations.

    Each round selects the candidate that adds the most marginal expected value
    to the portfolio. Enforces Kelly-based exposure caps by removing candidates
    that would violate limits.

    Returns list of indices into candidates/lineup_scores.
    """
    n_candidates, n_sims = lineup_scores.shape

    # Build player-to-candidates index for fast exposure enforcement
    player_to_candidates = defaultdict(set)
    for c, lineup_indices in enumerate(candidates):
        for player_idx in lineup_indices:
            player_to_candidates[player_idx].add(c)

    alive = np.ones(n_candidates, dtype=bool)
    running_max = np.full(n_sims, -np.inf, dtype=np.float32)
    appearance_count = np.zeros(n_players, dtype=np.int32)
    selected = []

    eval_chunk = 2000  # Process candidates in chunks to limit memory

    for r in range(n_select):
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            print(f"  Warning: Ran out of eligible candidates at lineup {r + 1}/{n_select}")
            break

        # Find the candidate with the best marginal improvement
        best_global_idx = -1
        best_marginal = -np.inf

        for chunk_start in range(0, len(alive_indices), eval_chunk):
            chunk_end = min(chunk_start + eval_chunk, len(alive_indices))
            chunk_indices = alive_indices[chunk_start:chunk_end]

            chunk_scores = lineup_scores[chunk_indices]  # (chunk_len, n_sims)
            improvement = np.maximum(chunk_scores - running_max[np.newaxis, :], 0.0)
            marginal_values = improvement.mean(axis=1)

            chunk_best_pos = np.argmax(marginal_values)
            if marginal_values[chunk_best_pos] > best_marginal:
                best_marginal = marginal_values[chunk_best_pos]
                best_global_idx = chunk_indices[chunk_best_pos]

        # Select this candidate
        selected.append(best_global_idx)
        running_max = np.maximum(running_max, lineup_scores[best_global_idx])
        alive[best_global_idx] = False

        # Update exposure and kill violating candidates
        for player_idx in candidates[best_global_idx]:
            appearance_count[player_idx] += 1
            if appearance_count[player_idx] >= max_appearances.get(player_idx, 999):
                # Kill all alive candidates containing this maxed-out player
                for c in player_to_candidates[player_idx]:
                    alive[c] = False

    return selected


# ── Core Optimizer (Efficient Frontier) ──────────────────────────────────────

def _solve_lineup_ef(players, locked_out, max_overlap_constraints, correlation_penalty=None, max_overlap=MAX_OVERLAP):
    """Solve a single lineup with efficient frontier constraints.

    correlation_penalty: array of per-player penalties based on how correlated
    they are with already-selected lineups (reduces objective for overexposed players).
    max_overlap: max shared players allowed with any single previous lineup.
    """
    n = len(players)

    h = Highs()
    h.silent()

    obj = np.array([p["projected_points"] for p in players], dtype=float)
    if correlation_penalty is not None:
        obj = obj - correlation_penalty

    for i in range(n):
        h.addVariable(0.0, 1.0, float(obj[i]))
    h.changeColsIntegrality(
        n,
        np.arange(n, dtype=np.int32),
        np.array([1] * n, dtype=np.uint8),
    )
    h.changeObjectiveSense(ObjSense.kMaximize)

    # Constraint: exactly ROSTER_SIZE golfers
    h.addRow(
        float(ROSTER_SIZE), float(ROSTER_SIZE),
        n, np.arange(n, dtype=np.int32), np.ones(n),
    )

    # Constraint: salary cap (upper) and salary floor (lower)
    salaries = np.array([float(p["salary"]) for p in players])
    h.addRow(float(SALARY_FLOOR), float(SALARY_CAP), n, np.arange(n, dtype=np.int32), salaries)

    # Constraint: lock out players who've hit their exposure limit
    for i in locked_out:
        h.addRow(0.0, 0.0, 1, np.array([i], dtype=np.int32), np.ones(1))

    # Constraint: max overlap with each previous lineup
    for prev_indices in max_overlap_constraints:
        k = len(prev_indices)
        if k == 0:
            continue
        idx = np.array(prev_indices, dtype=np.int32)
        h.addRow(0.0, float(max_overlap), k, idx, np.ones(k))

    h.run()
    if h.getModelStatus() != HighsModelStatus.kOptimal:
        return None

    sol = h.getSolution()
    return [i for i in range(n) if sol.col_value[i] > 0.5]


def optimize_lineup(players, num_lineups=None, max_exposure=None,
                    contest_params=None, pool_size=None):
    """Generate optimally diversified DFS lineups.

    When pool_size > 0 (default): generates a large candidate pool via randomized
    MIP solves, then selects the optimal portfolio using greedy forward selection
    on Monte Carlo simulations. Guarantees exactly num_lineups output.

    When pool_size == 0: falls back to legacy sequential generation.

    contest_params: optional dict from dk_contests.derive_optimizer_params() that
    overrides internal constants (kelly_fraction, n_sims, lambda_base, etc.)
    """
    cp = contest_params or {}

    if num_lineups is None:
        num_lineups = cp.get("num_lineups", NUM_LINEUPS)
    if max_exposure is None:
        max_exposure = 1.0

    # Contest-aware parameters (fall back to module defaults)
    kelly_frac = cp.get("kelly_fraction", KELLY_FRACTION)
    n_sims = cp.get("n_sims", N_SIMS)
    lambda_base = cp.get("lambda_base", 0.005)
    penalty_cap_pct = cp.get("lambda_penalty_cap", 0.10)
    max_overlap_early = cp.get("max_overlap_early", MAX_OVERLAP)
    max_overlap_late = cp.get("max_overlap_late", MAX_OVERLAP + 1)

    if not players:
        return []

    players = [p for p in players if p.get("salary", 0) > 0 and p.get("projected_points", 0) > 0]

    if len(players) < ROSTER_SIZE:
        print(f"Warning: Only {len(players)} eligible players, need {ROSTER_SIZE}")
        return []

    n = len(players)

    # ── Step 1: Monte Carlo simulation ──
    print(f"  Running Monte Carlo simulation ({n_sims:,} scenarios)...")
    sims = simulate_outcomes(players, n_sims=n_sims)
    cov_matrix, sigmas = build_covariance_matrix(players)

    # ── Step 2: Compute Perfect% and Kelly exposure ──
    perfect_pcts = compute_perfect_pct(players, sims)
    kelly_exposures = compute_kelly_exposure(players, perfect_pcts)

    # Print top players by Perfect%
    top_perf = sorted(range(n), key=lambda i: perfect_pcts[i], reverse=True)[:10]
    print(f"\n  {'Player':<25} {'Proj':>7} {'Sigma':>7} {'Perf%':>7} {'Kelly%':>7}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for i in top_perf:
        print(f"  {players[i]['name']:<25} {players[i]['projected_points']:>7.1f} "
              f"{sigmas[i]:>7.1f} {perfect_pcts[i]*100:>6.1f}% {kelly_exposures[i]*100:>6.1f}%")

    # ── Step 3: Build lineups ──

    # Scale Kelly exposures so total slot budget >= needed slots
    needed_slots = num_lineups * ROSTER_SIZE
    raw_kelly_slots = sum(max(1, math.ceil(num_lineups * kelly_exposures[i])) for i in range(n))
    if raw_kelly_slots < needed_slots:
        scale = (needed_slots * 1.20) / raw_kelly_slots
    else:
        scale = 1.0

    max_appearances = {}
    for i in range(n):
        scaled_kelly = kelly_exposures[i] * scale
        kelly_max = max(1, math.ceil(num_lineups * scaled_kelly))
        global_max = max(1, math.ceil(num_lineups * max_exposure))
        max_appearances[i] = min(kelly_max, global_max)

    total_budget = sum(max_appearances.values())

    use_pool = pool_size is not None and pool_size > 0

    if use_pool:
        # For pool-based selection, use the broader max_exposure cap.
        # The greedy selector inherently diversifies via diminishing marginal
        # returns — tight Kelly caps cause cascade kills that exhaust the pool.
        pool_max_appearances = {}
        global_max = max(1, math.ceil(num_lineups * max_exposure))
        for i in range(n):
            pool_max_appearances[i] = global_max
        pool_budget = sum(pool_max_appearances.values())

        # ── Pool-based portfolio optimization ──
        print(f"\n  Generating {pool_size:,} candidate lineups...")
        candidates = _generate_candidate_pool(players, pool_size=pool_size)
        print(f"  Unique candidates: {len(candidates):,}")

        if len(candidates) < num_lineups:
            print(f"  Warning: Only {len(candidates)} candidates < {num_lineups} needed. "
                  f"Increasing noise and retrying...")
            extra = _generate_candidate_pool(players, pool_size=pool_size, noise_scale=0.25)
            for c in extra:
                candidates.append(c)
            candidates = list(set(candidates))
            print(f"  Candidates after retry: {len(candidates):,}")

        print(f"  Scoring candidates across {n_sims:,} simulations...")
        lineup_scores = _score_candidates(candidates, sims)

        print(f"  Selecting optimal portfolio of {num_lineups} lineups...")
        print(f"  Max exposure: {max_exposure:.0%} ({global_max} appearances)")
        selected_indices = _select_portfolio(
            lineup_scores, candidates, num_lineups, pool_max_appearances, n,
        )

        # Build output lineup dicts
        lineups = []
        lineup_indices = []
        for sel_idx in selected_indices:
            player_indices = list(candidates[sel_idx])
            lineup = [players[i].copy() for i in player_indices]
            lineup.sort(key=lambda p: p["salary"], reverse=True)
            lineups.append(lineup)
            lineup_indices.append(player_indices)

        # Portfolio evaluation with precomputed scores
        if lineups:
            selected_scores = lineup_scores[selected_indices]
            print(f"\n  Evaluating portfolio ({len(lineups)} lineups)...")
            metrics = portfolio_metrics(lineup_indices, sims, precomputed_scores=selected_scores)
            print(f"  Expected best lineup:  {metrics['mean_best']:.1f} pts")
            print(f"  Upside (95th pctl):    {metrics['p95_best']:.1f} pts")
            print(f"  Downside (5th pctl):   {metrics['p5_best']:.1f} pts")
            print(f"  Portfolio Sharpe:      {metrics['sharpe']:.3f}")

            unique_players = len(set(i for idx_list in lineup_indices for i in idx_list))
            total_slots = len(lineups) * ROSTER_SIZE
            print(f"  Unique players used:   {unique_players}/{len(players)}")
            print(f"  Diversification:       {unique_players * ROSTER_SIZE / total_slots:.1%}")

    else:
        # ── Legacy sequential generation ──
        print(f"\n  Building {num_lineups} diversified lineups (sequential)...")
        print(f"  Total exposure budget: {total_budget} slots (need {needed_slots})")

        lineups = []
        lineup_indices = []
        appearance_count = {i: 0 for i in range(n)}

        cumulative_exposure = np.zeros(n)
        proj_values = np.array([p["projected_points"] for p in players])

        stage1_end = min(50, num_lineups // 3) if num_lineups > 3 else num_lineups
        stage2_end = min(100, 2 * num_lineups // 3) if num_lineups > 3 else num_lineups

        for lineup_num in range(num_lineups):
            locked_out = [i for i in range(n) if appearance_count[i] >= max_appearances[i]]

            if lineup_num < stage1_end:
                cur_max_overlap = max_overlap_early
                overlap_constraints = lineup_indices.copy()
            elif lineup_num < stage2_end:
                cur_max_overlap = max_overlap_early
                window = max(30, min(40, num_lineups // 3))
                overlap_constraints = lineup_indices[-window:]
            else:
                cur_max_overlap = max_overlap_late
                window = max(20, min(30, num_lineups // 4))
                overlap_constraints = lineup_indices[-window:]

            progress = lineup_num / max(num_lineups, 1)
            lam = lambda_base * (1 + progress)
            corr_penalty = lam * (cov_matrix @ cumulative_exposure)

            max_allowed = proj_values * penalty_cap_pct
            corr_penalty = np.minimum(corr_penalty, max_allowed)
            corr_penalty = np.maximum(corr_penalty, 0.0)

            selected = _solve_lineup_ef(players, locked_out, overlap_constraints, corr_penalty, cur_max_overlap)
            if selected is None:
                selected = _solve_lineup_ef(players, locked_out, [], corr_penalty, max_overlap_late)
                if selected is None:
                    selected = _solve_lineup_ef(players, locked_out, [], None, 6)
                    if selected is None:
                        break

            lineup = []
            for i in selected:
                lineup.append(players[i].copy())
                appearance_count[i] += 1
                cumulative_exposure[i] += 1.0

            lineup.sort(key=lambda p: p["salary"], reverse=True)
            lineups.append(lineup)
            lineup_indices.append(selected)

        # Portfolio evaluation
        if lineup_indices:
            print(f"\n  Evaluating portfolio ({len(lineups)} lineups)...")
            metrics = portfolio_metrics(lineup_indices, sims)
            print(f"  Expected best lineup:  {metrics['mean_best']:.1f} pts")
            print(f"  Upside (95th pctl):    {metrics['p95_best']:.1f} pts")
            print(f"  Downside (5th pctl):   {metrics['p5_best']:.1f} pts")
            print(f"  Portfolio Sharpe:      {metrics['sharpe']:.3f}")

            unique_players = len(set(i for idx_list in lineup_indices for i in idx_list))
            total_slots = len(lineups) * ROSTER_SIZE
            print(f"  Unique players used:   {unique_players}/{len(players)}")
            print(f"  Diversification:       {unique_players * ROSTER_SIZE / total_slots:.1%}")

    return lineups


# ── Display Functions ────────────────────────────────────────────────────────

def format_lineup(lineup, lineup_num=1):
    """Format a single lineup for display."""
    lines = []
    lines.append(f"\n{'='*78}")
    lines.append(f"  LINEUP #{lineup_num}")
    lines.append(f"{'='*78}")

    has_ownership = any(p.get("proj_ownership", 0) > 0 for p in lineup)
    if has_ownership:
        lines.append(f"  {'Player':<25} {'Salary':>8} {'Proj Pts':>10} {'Value':>8} {'Own%':>6}")
        lines.append(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8} {'-'*6}")
    else:
        lines.append(f"  {'Player':<25} {'Salary':>8} {'Proj Pts':>10} {'Value':>8}")
        lines.append(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8}")

    total_salary = 0
    total_points = 0.0

    for p in lineup:
        salary = p["salary"]
        pts = p["projected_points"]
        value = pts / (salary / 1000.0) if salary > 0 else 0
        total_salary += salary
        total_points += pts
        if has_ownership:
            own = p.get("proj_ownership", 0)
            lines.append(
                f"  {p['name']:<25} ${salary:>7,} {pts:>10.1f} {value:>8.2f} {own:>5.1f}%"
            )
        else:
            lines.append(
                f"  {p['name']:<25} ${salary:>7,} {pts:>10.1f} {value:>8.2f}"
            )

    lines.append(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8}")
    lines.append(f"  {'TOTAL':<25} ${total_salary:>7,} {total_points:>10.1f}")
    remaining = SALARY_CAP - total_salary
    lines.append(f"  Salary remaining: ${remaining:,}")

    return "\n".join(lines)


def format_all_lineups(lineups):
    """Format all lineups for display."""
    if not lineups:
        return "No lineups generated."

    output = []
    for i, lineup in enumerate(lineups, 1):
        output.append(format_lineup(lineup, i))

    # Player exposure summary
    player_counts = {}
    for lineup in lineups:
        for p in lineup:
            name = p["name"]
            player_counts[name] = player_counts.get(name, 0) + 1

    output.append(f"\n{'='*70}")
    output.append("  PLAYER EXPOSURE")
    output.append(f"{'='*70}")
    sorted_players = sorted(player_counts.items(), key=lambda x: x[1], reverse=True)
    for name, count in sorted_players:
        pct = count / len(lineups) * 100
        bar = "#" * min(count, 80)
        output.append(f"  {name:<25} {count}/{len(lineups)} ({pct:>5.1f}%) {bar}")

    return "\n".join(output)

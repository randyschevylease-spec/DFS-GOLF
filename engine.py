"""DFS Golf Engine — Three-step contest simulation.

Step 1: Generate a contest field from DataGolf projected ownership.
Step 2: Score every candidate lineup against that field via Monte Carlo,
        assign payouts from the real DK payout table, compute ROI.
Step 3: Select the best portfolio of N lineups maximizing E[max(portfolio)]
        via greedy marginal-contribution selection with exposure caps.
"""
import math
import numpy as np
from highspy import Highs, ObjSense, HighsModelStatus
from config import ROSTER_SIZE, SALARY_CAP, SALARY_FLOOR, MAX_EXPOSURE, CVAR_LAMBDA


# ── Step 1: Generate Contest Field ──────────────────────────────────────────

def generate_field(players, field_size, calibration_rounds=8, pilot_size=5000, seed=None):
    """Generate opponent lineups calibrated to DataGolf projected ownership.

    Uses iterative Dirichlet-multinomial sampling with pilot-based calibration
    so the resulting field's ownership distribution converges on DG targets.

    Args:
        players: list of dicts with 'salary', 'proj_ownership' keys
        field_size: total opponent lineups to generate
        calibration_rounds: iterations of pilot → measure → adjust
        pilot_size: lineups per calibration pilot batch
        seed: random seed for reproducibility

    Returns:
        list of lineups, each a list of player indices
    """
    rng = np.random.default_rng(seed)
    n = len(players)
    min_sal = min(p["salary"] for p in players)

    # Target ownership as lineup inclusion % (e.g. 42.6 means 42.6% of lineups)
    target_pct = np.array([max(p.get("proj_ownership", 1.0), 0.1) for p in players])

    # Initial sampling probs proportional to target ownership
    adjusted_probs = target_pct.copy()
    adjusted_probs /= adjusted_probs.sum()
    alpha_scale = 12.0

    # Iterative calibration: measure actual ownership, adjust to converge on targets
    for rnd in range(calibration_rounds):
        pilot = _sample_lineups(players, min(pilot_size, field_size), adjusted_probs, alpha_scale, min_sal, rng=rng)
        if not pilot:
            break

        # Measure resulting ownership as lineup inclusion %
        counts = np.zeros(n)
        for lu in pilot:
            for idx in lu:
                counts[idx] += 1
        pilot_pct = counts / len(pilot) * 100  # % of lineups containing each player
        pilot_pct = np.maximum(pilot_pct, 0.01)

        # Adjust: if target=30% but measured=10%, boost by (30/10)^damping
        ratio = (target_pct / pilot_pct) ** 0.7
        adjusted_probs = adjusted_probs * ratio
        adjusted_probs /= adjusted_probs.sum()

    # Generate full field with calibrated probabilities
    lineups = _sample_lineups(players, field_size, adjusted_probs, alpha_scale, min_sal, rng=rng)
    return lineups


def _sample_lineups(players, n_lineups, probs, alpha_scale, min_sal, rng=None):
    """Sample salary-valid lineups using Dirichlet-multinomial."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(players)
    alpha = np.maximum(probs * alpha_scale * n, 0.01)
    salaries = [p["salary"] for p in players]
    lineups = []

    attempts = 0
    while len(lineups) < n_lineups and attempts < n_lineups * 15:
        attempts += 1
        try:
            draw = rng.dirichlet(alpha)
        except Exception:
            draw = probs

        selected = []
        budget = SALARY_CAP
        avail = np.ones(n, dtype=bool)

        ok = True
        for slot in range(ROSTER_SIZE):
            remaining_slots = ROSTER_SIZE - slot - 1
            min_remaining = remaining_slots * min_sal
            max_affordable = budget - min_remaining

            afford = np.array([salaries[i] <= max_affordable for i in range(n)])
            valid = avail & afford
            if not valid.any():
                ok = False
                break

            vp = draw * valid
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
            budget -= salaries[c]
            avail[c] = False

        if ok and len(selected) == ROSTER_SIZE:
            total_sal = sum(salaries[i] for i in selected)
            if total_sal <= SALARY_CAP:
                lineups.append(selected)

    return lineups


# ── Step 2: Simulate & Calculate ROI ───────────────────────────────────────

def simulate_contest(candidates, opponents, players, payout_table, entry_fee, n_sims=10000):
    """Monte Carlo contest simulation with real DK payout table.

    For each simulation:
      1. Sample correlated player scores
      2. Score all lineups (candidates + opponents) via matmul
      3. Rank all lineups
      4. Assign payouts from the actual DK payout table by finish position

    Args:
        candidates: list of candidate lineups (list of player index lists)
        opponents: list of opponent lineups (list of player index lists)
        players: player dicts with 'projected_points', 'std_dev'
        payout_table: list of (min_pos, max_pos, prize) from DK API
        entry_fee: contest entry fee in dollars
        n_sims: number of Monte Carlo simulations

    Returns:
        payouts: (n_candidates, n_sims) array of dollar payouts per sim
        roi: (n_candidates,) array of mean ROI %
    """
    n_players = len(players)
    n_cands = len(candidates)
    n_opps = len(opponents)
    n_total = n_cands + n_opps

    # Build binary lineup matrix: (n_total, n_players)
    all_lineups = candidates + opponents
    matrix = np.zeros((n_total, n_players), dtype=np.float32)
    for i, lu in enumerate(all_lineups):
        for idx in lu:
            matrix[i, idx] = 1.0

    # Player score parameters
    means = np.array([p["projected_points"] for p in players], dtype=np.float64)
    sigmas = np.array([_get_sigma(p) for p in players], dtype=np.float64)

    # Build covariance matrix (baseline correlation = 0.07 for shared conditions)
    base_corr = 0.07
    cov = np.outer(sigmas, sigmas) * base_corr
    np.fill_diagonal(cov, sigmas ** 2)

    # Ensure positive definite
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += np.eye(n_players) * 1.0
        L = np.linalg.cholesky(cov)

    # Build position → payout lookup (sorted by position)
    # payout_table: [(min_pos, max_pos, prize), ...]
    pos_payouts = []
    for min_pos, max_pos, prize in sorted(payout_table, key=lambda x: x[0]):
        for pos in range(min_pos, max_pos + 1):
            pos_payouts.append((pos, prize))
    max_payout_pos = max(pos for pos, _ in pos_payouts) if pos_payouts else 0

    # Pre-build payout array indexed by position (1-indexed)
    payout_by_pos = np.zeros(n_total + 1, dtype=np.float64)
    for pos, prize in pos_payouts:
        if pos <= n_total:
            payout_by_pos[pos] = prize

    # Monte Carlo
    payouts = np.zeros((n_cands, n_sims), dtype=np.float64)

    print(f"  Simulating {n_sims:,} contests: {n_cands} candidates vs {n_opps:,} opponents...")

    rng = np.random.default_rng()

    for sim in range(n_sims):
        # Sample correlated player scores
        Z = rng.standard_normal(n_players)
        scores = means + L @ Z
        scores = np.maximum(scores, 0.0).astype(np.float32)

        # Score all lineups
        lu_scores = matrix @ scores

        # Rank: highest score = position 1
        # argsort descending, then map to positions
        order = np.argsort(-lu_scores)
        positions = np.empty(n_total, dtype=np.int32)
        positions[order] = np.arange(1, n_total + 1)

        # Assign payouts to our candidates (first n_cands entries)
        for c in range(n_cands):
            pos = positions[c]
            if pos <= max_payout_pos:
                payouts[c, sim] = payout_by_pos[pos]

    mean_payouts = payouts.mean(axis=1)
    roi = (mean_payouts - entry_fee) / entry_fee * 100

    return payouts, roi


def _get_sigma(player):
    """Get player score standard deviation."""
    sd = player.get("std_dev", 0)
    if sd and sd > 0:
        return max(sd, 5.0)
    # Heuristic fallback
    proj = player.get("projected_points", 50)
    return max(proj * 0.35, 5.0)


# ── Step 3: Select Portfolio ───────────────────────────────────────────────

def select_portfolio(payouts, entry_fee, n_select, candidates, n_players,
                     max_exposure=None, cvar_lambda=None):
    """Greedy marginal-contribution portfolio selection with CVaR tail penalty.

    Objective per round:
        score = E[max improvement] + λ × E[payout in worst 5% of sims]

    The first term (upside) picks lineups that win in sims where the
    existing portfolio loses. The second term (CVaR) rewards lineups
    that cash in the portfolio's worst outcomes — insurance against
    total wipeout.

    Args:
        payouts: (n_candidates, n_sims) array of dollar payouts
        entry_fee: cost per lineup entry
        n_select: number of lineups to select
        candidates: list of candidate lineups (each a list of player indices)
        n_players: total number of players (for exposure tracking)
        max_exposure: max fraction of lineups a player can appear in
        cvar_lambda: tail-risk penalty weight (0=pure upside, 0.5=balanced)

    Returns:
        list of selected candidate indices
    """
    n_candidates, n_sims = payouts.shape

    if max_exposure is None:
        max_exposure = MAX_EXPOSURE
    if cvar_lambda is None:
        cvar_lambda = CVAR_LAMBDA

    max_appearances = max(1, int(n_select * max_exposure))
    tail_count = max(1, int(n_sims * 0.05))  # bottom 5% of sims for CVaR

    # Build player → candidate index for fast exposure removal
    player_to_cands = [[] for _ in range(n_players)]
    for ci, lineup in enumerate(candidates):
        for pidx in lineup:
            player_to_cands[pidx].append(ci)

    alive = np.ones(n_candidates, dtype=bool)
    running_max = np.full(n_sims, -np.inf, dtype=np.float64)
    appearances = np.zeros(n_players, dtype=np.int32)

    selected = []
    port_returns = np.zeros(n_sims, dtype=np.float64)

    mean_payouts = payouts.mean(axis=1)

    # Pre-allocate scratch arrays for the hot loop
    improvement = np.empty((n_candidates, n_sims), dtype=np.float64)
    upside_buf = np.empty(n_candidates, dtype=np.float64)
    score_buf = np.empty(n_candidates, dtype=np.float64)

    # Tail weight vector for fast CVaR via matmul: payouts @ tail_w
    tail_w = np.zeros(n_sims, dtype=np.float64)

    for rnd in range(n_select):
        if not alive.any():
            print(f"  Warning: candidate pool exhausted at {len(selected)}/{n_select}")
            break

        # ── Upside: marginal E[max] improvement ──
        np.subtract(payouts, running_max, out=improvement)
        np.maximum(improvement, 0.0, out=improvement)
        np.divide(improvement.sum(axis=1), n_sims, out=upside_buf)

        # ── Downside: CVaR tail contribution ──
        if cvar_lambda > 0:
            # Identify the worst 5% of sims by current portfolio P&L
            tail_idx = np.argpartition(port_returns, tail_count)[:tail_count]

            # Build weight vector: 1/tail_count at tail positions, 0 elsewhere
            tail_w[:] = 0.0
            tail_w[tail_idx] = 1.0 / tail_count

            # Each candidate's mean payout in tail sims (via single BLAS call)
            # score_buf = payouts @ tail_w  →  (n_candidates,)
            np.dot(payouts, tail_w, out=score_buf)
            score_buf -= entry_fee  # net contribution: payout - cost

            # Combined: upside + λ × tail_contribution
            score_buf *= cvar_lambda
            score_buf += upside_buf
        else:
            score_buf[:] = upside_buf

        # Mask dead candidates
        score_buf[~alive] = -np.inf

        # Select the candidate with highest combined score
        best_idx = int(np.argmax(score_buf))
        best_score = float(score_buf[best_idx])
        best_upside = float(upside_buf[best_idx])

        selected.append(best_idx)

        # Update running max
        np.maximum(running_max, payouts[best_idx], out=running_max)

        # Update portfolio returns
        port_returns = port_returns + payouts[best_idx] - entry_fee
        port_roi = float(port_returns.mean()) / (len(selected) * entry_fee) * 100

        # Remove this candidate from the pool
        alive[best_idx] = False

        # Update exposure counts and remove over-exposed players' candidates
        lineup = candidates[best_idx]
        for pidx in lineup:
            appearances[pidx] += 1
            if appearances[pidx] >= max_appearances:
                for ci in player_to_cands[pidx]:
                    alive[ci] = False

        if len(selected) % 25 == 0 or len(selected) == n_select:
            tail_mean = float(port_returns[tail_idx].mean()) if cvar_lambda > 0 else 0
            print(f"    [{len(selected)}/{n_select}] ROI={port_roi:+.1f}%  "
                  f"Upside=${best_upside:.2f}  "
                  f"CVaR₅=${tail_mean:+,.0f}  "
                  f"Alive={int(alive.sum()):,}")

    return selected


# ── Candidate Generation ──────────────────────────────────────────────────

def generate_candidates(players, pool_size=5000, noise_scale=0.15, seed=None,
                        min_proj_pct=0.85, candidate_exposure_cap=0.40):
    """Generate diverse, high-quality candidate lineups via randomized MIP solves.

    Uses HiGHS for fast (~1ms/solve) binary integer programming with
    multiplicative noise to explore the solution space. A projection floor
    constraint ensures every candidate lineup is high-quality, and a post-
    generation diversity filter caps per-player exposure in the candidate pool.

    Args:
        players: list of player dicts
        pool_size: target number of raw MIP solves
        noise_scale: log-normal noise std dev for objective perturbation
        seed: random seed
        min_proj_pct: minimum lineup projection as fraction of optimal (0.85 = 85%)
        candidate_exposure_cap: max fraction of final candidates any player can appear in

    Returns list of unique lineups (each a list of sorted player indices).
    """
    n = len(players)
    # Use leverage-adjusted mip_value if available, otherwise projected_points
    base_obj = np.array([p.get("mip_value", p.get("projected_points", 0)) for p in players])
    proj_pts = np.array([p.get("projected_points", 0) for p in players], dtype=np.float64)
    rng = np.random.default_rng(seed)
    candidate_set = set()

    # Solve optimal lineup to establish projection quality floor
    optimal = _solve_mip(players, base_obj)
    proj_floor = None
    if optimal is not None and min_proj_pct > 0:
        max_proj = sum(proj_pts[i] for i in optimal)
        proj_floor = max_proj * min_proj_pct
        print(f"    Projection floor: {proj_floor:.1f} pts ({min_proj_pct:.0%} of optimal {max_proj:.1f})")

    # Phase 1: Projection-based with noise (increased for more exploration)
    phase1_noise = noise_scale * 1.3  # more exploration; floor constraint keeps quality
    batch = 1000
    for batch_start in range(0, pool_size, batch):
        before = len(candidate_set)
        for _ in range(min(batch, pool_size - batch_start)):
            noise = np.exp(rng.normal(0.0, phase1_noise, size=n))
            sel = _solve_mip(players, base_obj * noise,
                             proj_pts=proj_pts, proj_floor=proj_floor)
            if sel is not None:
                candidate_set.add(sel)

        # Early stop if yield drops
        new = len(candidate_set) - before
        if batch_start > 0 and new / batch < 0.03:
            break

    # Phase 2: Exclude each top player for diversity
    top = sorted(range(n), key=lambda i: base_obj[i], reverse=True)[:12]
    for excluded in top:
        for _ in range(100):
            noise = np.exp(rng.normal(0.0, phase1_noise, size=n))
            obj = base_obj * noise
            obj[excluded] = -1e6
            sel = _solve_mip(players, obj,
                             proj_pts=proj_pts, proj_floor=proj_floor)
            if sel is not None:
                candidate_set.add(sel)

    # Phase 3: Exclude pairs of top players
    for i in range(min(6, len(top))):
        for j in range(i + 1, min(8, len(top))):
            for _ in range(30):
                noise = np.exp(rng.normal(0.0, noise_scale * 1.5, size=n))
                obj = base_obj * noise
                obj[top[i]] = -1e6
                obj[top[j]] = -1e6
                sel = _solve_mip(players, obj,
                                 proj_pts=proj_pts, proj_floor=proj_floor)
                if sel is not None:
                    candidate_set.add(sel)

    # Phase 4: Salary-tier diversification
    salaries = np.array([float(p["salary"]) for p in players])
    tier_count = max(200, pool_size // 5)

    # Stars & scrubs: 2 players >= $9K, 4 players < $7.5K
    high_idx = np.where(salaries >= 9000)[0]
    low_idx = np.where(salaries < 7500)[0]
    if len(high_idx) >= 2 and len(low_idx) >= 4:
        for _ in range(tier_count):
            obj = base_obj * np.exp(rng.normal(0.0, noise_scale * 1.5, size=n))
            # Heavily penalize mid-range players
            mid_mask = (salaries >= 7500) & (salaries < 9000)
            obj[mid_mask] *= 0.3
            sel = _solve_mip(players, obj,
                             proj_pts=proj_pts, proj_floor=proj_floor)
            if sel is not None:
                candidate_set.add(sel)

    # Balanced: all players $7K-$9K
    mid_idx = np.where((salaries >= 7000) & (salaries <= 9000))[0]
    if len(mid_idx) >= ROSTER_SIZE:
        for _ in range(tier_count):
            obj = base_obj * np.exp(rng.normal(0.0, noise_scale * 1.5, size=n))
            # Penalize extremes
            extreme_mask = (salaries < 7000) | (salaries > 9000)
            obj[extreme_mask] *= 0.3
            sel = _solve_mip(players, obj,
                             proj_pts=proj_pts, proj_floor=proj_floor)
            if sel is not None:
                candidate_set.add(sel)

    raw_count = len(candidate_set)

    # Diversity filter: cap per-player exposure in candidate pool
    all_candidates = list(candidate_set)
    if candidate_exposure_cap < 1.0 and len(all_candidates) > 100:
        all_candidates = _diversity_filter(all_candidates, proj_pts, n,
                                           candidate_exposure_cap)
        print(f"    Candidates: {raw_count} raw → {len(all_candidates)} after diversity filter "
              f"(exposure cap {candidate_exposure_cap:.0%})")
    else:
        print(f"    Candidates: {len(all_candidates)}")

    return [list(c) for c in all_candidates]


def _diversity_filter(candidates, proj_pts, n_players, exposure_cap):
    """Greedy diversity filter: keep highest-projection candidates first,
    skipping any that would push a player over the exposure cap.

    This ensures the candidate pool has both high quality AND low concentration
    on any single player.
    """
    target_size = len(candidates)
    max_appearances = max(10, int(target_size * exposure_cap))

    # Sort by total projected points descending (keep best first)
    scored = [(sum(proj_pts[i] for i in c), c) for c in candidates]
    scored.sort(reverse=True)

    appearances = np.zeros(n_players, dtype=np.int32)
    filtered = []

    for proj, cand in scored:
        if all(appearances[i] < max_appearances for i in cand):
            filtered.append(cand)
            for i in cand:
                appearances[i] += 1

    return filtered


def _solve_mip(players, obj, proj_pts=None, proj_floor=None):
    """Solve a single lineup MIP using HiGHS.

    Args:
        players: list of player dicts
        obj: objective coefficients (one per player)
        proj_pts: array of projected points per player (for floor constraint)
        proj_floor: minimum total projected points for the lineup
    """
    n = len(players)
    h = Highs()
    h.silent()

    for i in range(n):
        h.addVariable(0.0, 1.0, float(obj[i]))
    h.changeColsIntegrality(n, np.arange(n, dtype=np.int32), np.array([1]*n, dtype=np.uint8))
    h.changeObjectiveSense(ObjSense.kMaximize)

    # Exactly ROSTER_SIZE players
    h.addRow(float(ROSTER_SIZE), float(ROSTER_SIZE), n, np.arange(n, dtype=np.int32), np.ones(n))

    # Salary bounds
    salaries = np.array([float(p["salary"]) for p in players])
    h.addRow(float(SALARY_FLOOR), float(SALARY_CAP), n, np.arange(n, dtype=np.int32), salaries)

    # Minimum projection floor (ensures lineup quality)
    if proj_pts is not None and proj_floor is not None:
        h.addRow(float(proj_floor), float(1e9), n, np.arange(n, dtype=np.int32),
                 proj_pts.astype(np.float64))

    h.run()
    if h.getModelStatus() != HighsModelStatus.kOptimal:
        return None

    sol = h.getSolution()
    return tuple(sorted(i for i in range(n) if sol.col_value[i] > 0.5))

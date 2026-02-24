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

def generate_field(players, field_size, calibration_rounds=8, pilot_size=5000):
    """Generate opponent lineups calibrated to DataGolf projected ownership.

    Uses iterative Dirichlet-multinomial sampling with pilot-based calibration
    so the resulting field's ownership distribution converges on DG targets.

    Args:
        players: list of dicts with 'salary', 'proj_ownership' keys
        field_size: total opponent lineups to generate
        calibration_rounds: iterations of pilot → measure → adjust
        pilot_size: lineups per calibration pilot batch

    Returns:
        list of lineups, each a list of player indices
    """
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
        pilot = _sample_lineups(players, min(pilot_size, field_size), adjusted_probs, alpha_scale, min_sal)
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
    lineups = _sample_lineups(players, field_size, adjusted_probs, alpha_scale, min_sal)
    return lineups


def _sample_lineups(players, n_lineups, probs, alpha_scale, min_sal):
    """Sample salary-valid lineups using Dirichlet-multinomial."""
    n = len(players)
    alpha = np.maximum(probs * alpha_scale * n, 0.01)
    salaries = [p["salary"] for p in players]
    lineups = []

    attempts = 0
    while len(lineups) < n_lineups and attempts < n_lineups * 15:
        attempts += 1
        try:
            draw = np.random.dirichlet(alpha)
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
                c = np.random.choice(n, p=vp)
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

    # Build covariance matrix (baseline correlation = 0.04 for shared conditions)
    base_corr = 0.04
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

def generate_candidates(players, pool_size=5000, noise_scale=0.15):
    """Generate diverse candidate lineups via randomized MIP solves.

    Uses HiGHS for fast (~1ms/solve) binary integer programming with
    multiplicative noise to explore the solution space.

    Returns list of unique lineups (each a tuple of sorted player indices).
    """
    n = len(players)
    base_obj = np.array([p.get("projected_points", 0) for p in players])
    rng = np.random.default_rng()
    candidate_set = set()

    # Phase 1: Projection-based with noise
    batch = 1000
    for batch_start in range(0, pool_size, batch):
        before = len(candidate_set)
        for _ in range(min(batch, pool_size - batch_start)):
            noise = np.exp(rng.normal(0.0, noise_scale, size=n))
            sel = _solve_mip(players, base_obj * noise)
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
            noise = np.exp(rng.normal(0.0, noise_scale, size=n))
            obj = base_obj * noise
            obj[excluded] = -1e6
            sel = _solve_mip(players, obj)
            if sel is not None:
                candidate_set.add(sel)

    # Phase 3: Exclude pairs of top players
    for i in range(min(6, len(top))):
        for j in range(i + 1, min(8, len(top))):
            for _ in range(30):
                noise = np.exp(rng.normal(0.0, noise_scale * 1.2, size=n))
                obj = base_obj * noise
                obj[top[i]] = -1e6
                obj[top[j]] = -1e6
                sel = _solve_mip(players, obj)
                if sel is not None:
                    candidate_set.add(sel)

    return [list(c) for c in candidate_set]


def _solve_mip(players, obj):
    """Solve a single lineup MIP using HiGHS."""
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

    h.run()
    if h.getModelStatus() != HighsModelStatus.kOptimal:
        return None

    sol = h.getSolution()
    return tuple(sorted(i for i in range(n) if sol.col_value[i] > 0.5))

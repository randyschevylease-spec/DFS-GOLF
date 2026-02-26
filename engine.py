"""DFS Golf Engine — Three-step contest simulation.

Step 1: Generate a contest field from DataGolf projected ownership.
Step 2: Score every candidate lineup against that field via Monte Carlo,
        assign payouts from the real DK payout table, compute ROI.
Step 3: Select the best portfolio of N lineups maximizing E[max(portfolio)]
        via greedy marginal-contribution selection with exposure caps.
"""
import math
import numpy as np
from scipy.stats import norm as sp_norm
from highspy import Highs, ObjSense, HighsModelStatus
from config import (ROSTER_SIZE, SALARY_CAP, SALARY_FLOOR, CVAR_LAMBDA,
                     BASE_CORRELATION, SAME_WAVE_CORRELATION, DIFF_WAVE_CORRELATION,
                     PORTFOLIO_METHOD, MPT_FRONTIER_TOLERANCE,
                     MPT_MIN_FRONTIER_SIZE, MPT_SIGMA_BINS, MPT_FRONTIER_MAX)


# ── Step 1: Generate Contest Field ──────────────────────────────────────────

def generate_field(players, field_size, calibration_rounds=8, pilot_size=5000, seed=None,
                   shark_ratio=0.15, shark_noise=0.25):
    """Generate opponent lineups calibrated to DataGolf projected ownership.

    Uses iterative Dirichlet-multinomial sampling with pilot-based calibration
    so the resulting field's ownership distribution converges on DG targets.

    A fraction of the field (shark_ratio) is generated via MIP optimization with
    noise, simulating sophisticated optimizer users in the contest. These "shark"
    lineups are high-quality opponents that prevent ROI inflation from facing a
    purely random field.

    Args:
        players: list of dicts with 'salary', 'proj_ownership' keys
        field_size: total opponent lineups to generate
        calibration_rounds: iterations of pilot → measure → adjust
        pilot_size: lineups per calibration pilot batch
        seed: random seed for reproducibility
        shark_ratio: fraction of field that should be optimizer-quality (0.0-1.0)
        shark_noise: noise scale for shark lineup generation (higher = more diverse)

    Returns:
        list of lineups, each a list of player indices
    """
    rng = np.random.default_rng(seed)
    n = len(players)
    min_sal = min(p["salary"] for p in players)

    # Split field into recreational and shark portions
    n_sharks = int(field_size * shark_ratio)
    n_recreational = field_size - n_sharks

    # Target ownership as lineup inclusion % (e.g. 42.6 means 42.6% of lineups)
    target_pct = np.array([max(p.get("proj_ownership", 1.0), 0.1) for p in players])

    # Initial sampling probs proportional to target ownership
    adjusted_probs = target_pct.copy()
    adjusted_probs /= adjusted_probs.sum()
    alpha_scale = 12.0

    # Iterative calibration: measure actual ownership, adjust to converge on targets
    for rnd in range(calibration_rounds):
        pilot = _sample_lineups(players, min(pilot_size, n_recreational), adjusted_probs, alpha_scale, min_sal, rng=rng)
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

    # Generate recreational lineups with calibrated probabilities
    lineups = _sample_lineups(players, n_recreational, adjusted_probs, alpha_scale, min_sal, rng=rng)

    # Generate shark lineups via MIP optimization with noise
    if n_sharks > 0:
        shark_lineups = _generate_shark_lineups(players, n_sharks, noise_scale=shark_noise, seed=seed)
        lineups = lineups + shark_lineups
        # Shuffle so sharks are evenly distributed
        rng.shuffle(lineups)

    return lineups


def _generate_shark_lineups(players, n_lineups, noise_scale=0.25, seed=None):
    """Generate optimizer-quality opponent lineups via noisy MIP solves.

    Simulates sophisticated DK contestants who use lineup optimizers.
    Uses higher noise and lower projection floor than candidate generation
    to represent the diversity of optimizer strategies in the field.
    """
    n = len(players)
    base_obj = np.array([p["projected_points"] for p in players])
    proj_pts = base_obj.copy()
    rng = np.random.default_rng(seed)
    lineup_set = set()

    # Looser quality floor than candidates (80% of optimal vs 85-88%)
    optimal = _solve_mip(players, base_obj)
    proj_floor = None
    if optimal is not None:
        max_proj = sum(proj_pts[i] for i in optimal)
        proj_floor = max_proj * 0.80

    # Generate with more noise for diversity
    target = n_lineups * 3  # overshoot to compensate for dedup
    for _ in range(target):
        noise = np.exp(rng.normal(0.0, noise_scale, size=n))
        sel = _solve_mip(players, base_obj * noise,
                         proj_pts=proj_pts, proj_floor=proj_floor)
        if sel is not None:
            lineup_set.add(sel)
        if len(lineup_set) >= n_lineups:
            break

    lineups = [list(s) for s in lineup_set]

    # If we didn't get enough unique lineups, duplicate some
    if len(lineups) < n_lineups and lineups:
        extra = n_lineups - len(lineups)
        indices = rng.integers(0, len(lineups), size=extra)
        lineups.extend([lineups[i] for i in indices])

    return lineups[:n_lineups]


def _sample_lineups(players, n_lineups, probs, alpha_scale, min_sal, rng=None):
    """Sample salary-valid lineups using Dirichlet-multinomial.

    Enforces SALARY_FLOOR to match real DK optimizer behavior (players use
    nearly all of their cap). Last 2 slots bias toward higher-salary players
    when budget remains, pushing lineups toward $49.5K-$50K usage.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(players)
    alpha = np.maximum(probs * alpha_scale * n, 0.01)
    sal_arr = np.array([p["salary"] for p in players], dtype=np.float64)
    lineups = []

    attempts = 0
    while len(lineups) < n_lineups and attempts < n_lineups * 25:
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

            # Floor-aware: need enough salary to hit SALARY_FLOOR
            spent = SALARY_CAP - budget
            floor_remaining = SALARY_FLOOR - spent
            min_this_slot = max(min_sal, (floor_remaining - remaining_slots * max(sal_arr)) if remaining_slots > 0
                                else floor_remaining)

            afford = (sal_arr <= max_affordable) & (sal_arr >= min_sal)
            valid = avail & afford
            if not valid.any():
                ok = False
                break

            vp = draw * valid

            # Salary-filling bias: real DK users optimize to use their full cap.
            # Weight = (salary / min_salary)^power — monotonically prefers
            # expensive players. Power increases in later slots to aggressively
            # spend remaining budget rather than waste it.
            sal_weight = (sal_arr / min_sal) ** (5.0 + slot * 1.5)
            vp = vp * sal_weight * valid

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
            if SALARY_FLOOR <= total_sal <= SALARY_CAP:
                lineups.append(selected)

    return lineups


# ── Step 2: Simulate & Calculate ROI ───────────────────────────────────────

def simulate_contest(candidates, opponents, players, payout_table, entry_fee,
                     n_sims=10000, waves=None, mixture_params=None):
    """Monte Carlo contest simulation with real DK payout table.

    For each simulation:
      1. Sample correlated player scores (optionally via Gaussian copula mixture)
      2. Score candidate and opponent lineups separately
      3. Rank each candidate against opponents only (not against other candidates)
      4. Assign payouts from the actual DK payout table by finish position

    Args:
        candidates: list of candidate lineups (list of player index lists)
        opponents: list of opponent lineups (list of player index lists)
        players: player dicts with 'projected_points', 'std_dev'
        payout_table: list of (min_pos, max_pos, prize) from DK API
        entry_fee: contest entry fee in dollars
        n_sims: number of Monte Carlo simulations
        waves: optional list of 0/1 per player (0=PM, 1=AM) for wave-aware correlation
        mixture_params: optional tuple from compute_mixture_params() for bimodal scores

    Returns:
        payouts: (n_candidates, n_sims) array of dollar payouts per sim
        roi: (n_candidates,) array of mean ROI %
    """
    n_players = len(players)
    n_cands = len(candidates)
    n_opps = len(opponents)

    # Build SEPARATE candidate and opponent lineup matrices
    cand_matrix = np.zeros((n_cands, n_players), dtype=np.float32)
    for i, lu in enumerate(candidates):
        for idx in lu:
            cand_matrix[i, idx] = 1.0
    opp_matrix = np.zeros((n_opps, n_players), dtype=np.float32)
    for i, lu in enumerate(opponents):
        for idx in lu:
            opp_matrix[i, idx] = 1.0

    # Player score parameters
    means = np.array([p["projected_points"] for p in players], dtype=np.float64)
    sigmas = np.array([_get_sigma(p) for p in players], dtype=np.float64)

    # Build covariance matrix with wave-aware correlation
    if waves is not None:
        waves_arr = np.array(waves)
        same_wave = (waves_arr[:, None] == waves_arr[None, :])
        corr_matrix = np.where(same_wave, SAME_WAVE_CORRELATION, DIFF_WAVE_CORRELATION)
        np.fill_diagonal(corr_matrix, 1.0)
        cov = np.outer(sigmas, sigmas) * corr_matrix
    else:
        base_corr = BASE_CORRELATION
        cov = np.outer(sigmas, sigmas) * base_corr
        np.fill_diagonal(cov, sigmas ** 2)

    # Ensure positive definite
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += np.eye(n_players) * 1.0
        L = np.linalg.cholesky(cov)

    # Build payout lookup sized to n_opps + 1 (candidate + opponents)
    sim_field = n_opps + 1
    payout_by_pos = np.zeros(sim_field + 1, dtype=np.float64)
    for min_pos, max_pos, prize in sorted(payout_table, key=lambda x: x[0]):
        for pos in range(min_pos, min(max_pos, sim_field) + 1):
            payout_by_pos[pos] = prize

    # Monte Carlo (batched for memory efficiency)
    payouts = np.zeros((n_cands, n_sims), dtype=np.float64)

    print(f"  Simulating {n_sims:,} contests: {n_cands} candidates vs {n_opps:,} opponents...")
    print(f"  Ranking: opponent-only (candidates ranked independently against field)")

    # Unpack mixture params if provided
    use_mix = (mixture_params is not None and mixture_params[5].any())
    if use_mix:
        mix_p_miss, mix_mu_miss, mix_sigma_miss, mix_mu_make, mix_sigma_make, mix_flag = mixture_params
        n_mix = int(mix_flag.sum())
        print(f"  Mixture distribution: {n_mix}/{n_players} players with bimodal scores")

    rng = np.random.default_rng()
    batch_size = 500
    cand_matrix_f32 = cand_matrix.astype(np.float32)
    opp_matrix_f32 = opp_matrix.astype(np.float32)

    for batch_start in range(0, n_sims, batch_size):
        bs = min(batch_size, n_sims - batch_start)
        Z = rng.standard_normal((bs, n_players))
        X = Z @ L.T                                                 # (bs, n_players)
        if use_mix:
            scores = transform_mixture_scores(
                X, sigmas, mix_p_miss, mix_mu_miss, mix_sigma_miss,
                mix_mu_make, mix_sigma_make, mix_flag)
        else:
            scores = means[None, :] + X                              # (bs, n_players)
            np.maximum(scores, 0.0, out=scores)

        # Score candidates and opponents separately
        cand_scores = scores.astype(np.float32) @ cand_matrix_f32.T   # (bs, n_cands)
        opp_scores = scores.astype(np.float32) @ opp_matrix_f32.T     # (bs, n_opps)

        # Sort opponent scores ascending for searchsorted
        opp_sorted = np.sort(opp_scores, axis=1)

        # Rank each candidate against opponents only
        # Position = 1 + (# opponents scoring higher)
        for s in range(bs):
            insert_idx = np.searchsorted(opp_sorted[s], cand_scores[s], side='left')
            pos = np.minimum((n_opps - insert_idx + 1).astype(np.int32), sim_field)
            payouts[:, batch_start + s] = payout_by_pos[pos]

    mean_payouts = payouts.mean(axis=1)
    roi = (mean_payouts - entry_fee) / entry_fee * 100

    return payouts, roi


def empirical_sigma_from_projection(proj):
    """Quadratic fit from 4,746 calibration records: non-monotonic σ(proj).

    Low variance for longshots (mostly miss cut), peak variance for mid-field
    (boom-or-bust), compressed variance for favorites.
    Peak σ ≈ 29 at proj ≈ 57.
    """
    sigma = -0.01455 * proj**2 + 1.6622 * proj - 18.86
    return max(sigma, 5.0)


def _get_sigma(player):
    """Get player score standard deviation."""
    sd = player.get("std_dev", 0)
    if sd and sd > 0:
        return max(sd, 5.0)
    # Empirical fallback from calibration data
    proj = player.get("projected_points", 50)
    return empirical_sigma_from_projection(proj)


# ── Mixture Distribution (Gaussian Copula) ─────────────────────────────────

def compute_mixture_params(players):
    """Derive per-player mixture distribution parameters.

    Returns arrays: p_miss, mu_miss, sigma_miss, mu_make, sigma_make, use_mixture
    """
    n = len(players)
    p_miss = np.zeros(n)
    mu_miss = np.zeros(n)
    sigma_miss = np.full(n, 6.0)
    mu_make = np.zeros(n)
    sigma_make = np.zeros(n)
    use_mixture = np.zeros(n, dtype=bool)

    for i, p in enumerate(players):
        mc = p.get("p_make_cut", 0)
        floor_val = p.get("floor", 0)
        ceil_val = p.get("ceiling", 0)
        mean_val = p.get("projected_points", 0)

        if mc <= 0 or mc >= 1.0 or floor_val <= 0 or ceil_val <= 0:
            # Fallback: pure normal
            mu_make[i] = mean_val
            sigma_make[i] = _get_sigma(p)
            continue

        p_miss_i = 1.0 - mc
        p_make_i = mc
        mu_miss_i = floor_val

        # Law of total expectation: mu_make = (mean - p_miss * mu_miss) / p_make
        mu_make_i = (mean_val - p_miss_i * mu_miss_i) / p_make_i

        # CEILING ~ mu_make + 2*sigma
        sigma_make_i = max((ceil_val - mu_make_i) / 2.0, 5.0)

        # Guard: if mu_make < mu_miss, model is degenerate → pure normal
        if mu_make_i < mu_miss_i:
            mu_make[i] = mean_val
            sigma_make[i] = _get_sigma(p)
            continue

        p_miss[i] = p_miss_i
        mu_miss[i] = mu_miss_i
        mu_make[i] = mu_make_i
        sigma_make[i] = sigma_make_i
        use_mixture[i] = True

    return p_miss, mu_miss, sigma_miss, mu_make, sigma_make, use_mixture


def transform_mixture_scores(X, sigmas, p_miss, mu_miss, sigma_miss,
                              mu_make, sigma_make, use_mixture):
    """Transform correlated normals to mixture-marginal scores via Gaussian copula.

    Args:
        X: (batch, n_players) correlated normals from Cholesky (X ~ N(0, Σ))
        sigmas: (n_players,) player std devs (for normalization)
        p_miss, mu_miss, sigma_miss: miss-cut distribution params
        mu_make, sigma_make: make-cut distribution params
        use_mixture: (n_players,) bool array

    Returns:
        scores: (batch, n_players) transformed scores
    """
    bs, n = X.shape
    scores = np.empty((bs, n), dtype=np.float64)

    # Non-mixture players: standard normal → mean + sigma * Z
    non_mix = ~use_mixture
    if non_mix.any():
        X_norm = X[:, non_mix] / sigmas[None, non_mix]
        scores[:, non_mix] = mu_make[non_mix] + sigma_make[non_mix] * X_norm

    # Mixture players: Gaussian copula transform
    mix = use_mixture
    if mix.any():
        X_mix = X[:, mix]
        sig_mix = sigmas[mix]

        # Normalize to standard normal marginals for copula
        X_norm = X_mix / sig_mix[None, :]
        U = sp_norm.cdf(X_norm)  # uniform marginals

        p_miss_mix = p_miss[mix]
        p_make_mix = 1.0 - p_miss_mix
        mu_miss_mix = mu_miss[mix]
        sigma_miss_mix = sigma_miss[mix]
        mu_make_mix = mu_make[mix]
        sigma_make_mix = sigma_make[mix]

        # Split at P(miss)
        miss_mask = U < p_miss_mix[None, :]

        # Miss-cut scores
        U_miss = np.clip(U / p_miss_mix[None, :], 1e-6, 1 - 1e-6)
        miss_scores = mu_miss_mix + sigma_miss_mix * sp_norm.ppf(U_miss)

        # Make-cut scores
        U_make = np.clip((U - p_miss_mix[None, :]) / p_make_mix[None, :], 1e-6, 1 - 1e-6)
        make_scores = mu_make_mix + sigma_make_mix * sp_norm.ppf(U_make)

        scores[:, mix] = np.where(miss_mask, miss_scores, make_scores)

    np.maximum(scores, 0.0, out=scores)
    return scores


# ── Step 3: Select Portfolio ───────────────────────────────────────────────

def select_portfolio_emax(payouts, entry_fee, n_select, candidates, n_players,
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
        max_exposure = 1.0
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

    # Round 1: pick the lineup with highest mean payout (best individual lineup)
    best_first = int(np.argmax(mean_payouts))
    selected.append(best_first)
    running_max = payouts[best_first].copy()
    port_returns = payouts[best_first] - entry_fee
    alive[best_first] = False
    for pidx in candidates[best_first]:
        appearances[pidx] += 1
        if appearances[pidx] >= max_appearances:
            for ci in player_to_cands[pidx]:
                alive[ci] = False

    port_roi = float(port_returns.mean()) / entry_fee * 100
    print(f"    [1/{n_select}] Seed lineup: idx={best_first} "
          f"mean=${mean_payouts[best_first]:.2f} ROI={port_roi:+.1f}%")

    for rnd in range(1, n_select):
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


def select_portfolio_mpt(payouts, entry_fee, n_select, candidates, n_players,
                         max_exposure=None, cvar_lambda=None):
    """MPT efficient-frontier portfolio selection maximizing Sharpe ratio.

    1. Compute per-lineup mean profit and std dev from payouts matrix.
    2. Filter to efficient frontier via binned upper-envelope Sharpe sweep.
    3. Greedy selection: each round picks the lineup that maximizes portfolio
       Sharpe ratio using incremental variance with covariance tracking.

    Same signature as select_portfolio_emax — drop-in replacement.
    """
    n_candidates, n_sims = payouts.shape

    if max_exposure is None:
        max_exposure = 1.0

    max_appearances = max(1, int(n_select * max_exposure))

    # ── Per-lineup stats ──
    profits = payouts - entry_fee                       # (n_candidates, n_sims)
    mu = profits.mean(axis=1)                           # mean profit
    sigma = profits.std(axis=1, ddof=0)                 # std dev
    sigma = np.maximum(sigma, 1e-9)                     # avoid div-by-zero
    sharpe = mu / sigma                                 # per-lineup Sharpe

    # ── Efficient frontier filter ──
    # Bin by sigma, keep only lineups near the upper Sharpe envelope
    sigma_min, sigma_max = sigma.min(), sigma.max()
    bin_edges = np.linspace(sigma_min, sigma_max, MPT_SIGMA_BINS + 1)
    bin_idx = np.digitize(sigma, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, MPT_SIGMA_BINS - 1)

    # Upper envelope: best Sharpe in each bin
    best_sharpe = np.full(MPT_SIGMA_BINS, -np.inf)
    for b in range(MPT_SIGMA_BINS):
        mask = bin_idx == b
        if mask.any():
            best_sharpe[b] = sharpe[mask].max()

    # Fill gaps: propagate best from neighboring bins
    for b in range(1, MPT_SIGMA_BINS):
        best_sharpe[b] = max(best_sharpe[b], best_sharpe[b - 1])

    # Keep lineups within tolerance of envelope
    envelope_sharpe = best_sharpe[bin_idx]
    frontier_mask = sharpe >= (envelope_sharpe - MPT_FRONTIER_TOLERANCE)

    # Ensure minimum frontier size by relaxing tolerance if needed
    n_frontier = int(frontier_mask.sum())
    if n_frontier < MPT_MIN_FRONTIER_SIZE:
        # Fall back: take top candidates by Sharpe
        n_take = min(MPT_MIN_FRONTIER_SIZE, n_candidates)
        top_idx = np.argpartition(-sharpe, n_take)[:n_take]
        frontier_mask[:] = False
        frontier_mask[top_idx] = True
        n_frontier = n_take

    # Hard cap on frontier size
    if n_frontier > MPT_FRONTIER_MAX:
        frontier_indices = np.where(frontier_mask)[0]
        top_idx = frontier_indices[np.argpartition(-sharpe[frontier_indices], MPT_FRONTIER_MAX)[:MPT_FRONTIER_MAX]]
        frontier_mask[:] = False
        frontier_mask[top_idx] = True
        n_frontier = MPT_FRONTIER_MAX

    frontier_idx = np.where(frontier_mask)[0]
    print(f"  MPT frontier: {n_frontier} lineups from {n_candidates} candidates "
          f"(Sharpe range {sharpe[frontier_idx].min():.3f} – {sharpe[frontier_idx].max():.3f})")

    # ── Covariance matrix on frontier profits ──
    frontier_profits = profits[frontier_idx]             # (n_frontier, n_sims)
    cov_matrix = np.cov(frontier_profits, ddof=0)        # (n_frontier, n_frontier)
    if cov_matrix.ndim == 0:
        cov_matrix = cov_matrix.reshape(1, 1)

    frontier_mu = mu[frontier_idx]
    frontier_var = np.diag(cov_matrix).copy()

    # Build player → frontier index for fast exposure removal
    player_to_frontier = [[] for _ in range(n_players)]
    for fi, ci in enumerate(frontier_idx):
        for pidx in candidates[ci]:
            player_to_frontier[pidx].append(fi)

    alive = np.ones(n_frontier, dtype=bool)
    appearances = np.zeros(n_players, dtype=np.int32)
    selected = []

    # Incremental tracking
    port_mean = 0.0
    port_var = 0.0
    cov_sum = np.zeros(n_frontier, dtype=np.float64)     # sum of cov with selected

    for rnd in range(n_select):
        if not alive.any():
            print(f"  Warning: frontier pool exhausted at {len(selected)}/{n_select}")
            break

        # New portfolio variance and Sharpe if we add each candidate
        new_var = port_var + frontier_var + 2.0 * cov_sum
        new_mean = port_mean + frontier_mu
        new_std = np.sqrt(np.maximum(new_var, 1e-18))
        new_sharpe = new_mean / new_std

        # Mask dead candidates
        new_sharpe[~alive] = -np.inf

        best_fi = int(np.argmax(new_sharpe))
        selected.append(int(frontier_idx[best_fi]))

        # Update incremental tracking
        port_mean += frontier_mu[best_fi]
        port_var = float(new_var[best_fi])
        cov_sum += cov_matrix[best_fi]

        # Mark as used
        alive[best_fi] = False

        # Update exposure counts and remove over-exposed players' candidates
        lineup = candidates[frontier_idx[best_fi]]
        for pidx in lineup:
            appearances[pidx] += 1
            if appearances[pidx] >= max_appearances:
                for fi in player_to_frontier[pidx]:
                    alive[fi] = False

        if len(selected) % 25 == 0 or len(selected) == n_select:
            port_std = math.sqrt(max(port_var, 1e-18))
            port_sharpe = port_mean / port_std
            port_roi = port_mean / (len(selected) * entry_fee) * 100
            print(f"    [{len(selected)}/{n_select}] ROI={port_roi:+.1f}%  "
                  f"Sharpe={port_sharpe:.3f}  "
                  f"Alive={int(alive.sum()):,}")

    return selected


def select_portfolio(payouts, entry_fee, n_select, candidates, n_players,
                     max_exposure=None, cvar_lambda=None):
    """Dispatcher: routes to MPT or E[max] based on PORTFOLIO_METHOD config."""
    if PORTFOLIO_METHOD == "mpt":
        return select_portfolio_mpt(payouts, entry_fee, n_select, candidates,
                                    n_players, max_exposure=max_exposure)
    else:
        return select_portfolio_emax(payouts, entry_fee, n_select, candidates,
                                     n_players, max_exposure=max_exposure,
                                     cvar_lambda=cvar_lambda)


# ── Candidate Generation ──────────────────────────────────────────────────

def generate_candidates(players, pool_size=5000, noise_scale=0.15, seed=None,
                        min_proj_pct=0.88, candidate_exposure_cap=0.40,
                        ceiling_pts=None, ceiling_weight=0.0,
                        salary_floor_override=None, proj_floor_override=None):
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
        ceiling_pts: optional array of ceiling points per player for blended objective
        ceiling_weight: weight for ceiling in objective blend (0=pure mean, 1=pure ceiling)
        salary_floor_override: optional override for minimum salary usage
        proj_floor_override: optional explicit projection floor (overrides min_proj_pct)

    Returns list of unique lineups (each a list of sorted player indices).
    """
    n = len(players)
    base_obj = np.array([p["projected_points"] for p in players])

    # Blend with ceiling if provided
    if ceiling_pts is not None and ceiling_weight > 0:
        ceiling_arr = np.array(ceiling_pts, dtype=np.float64)
        base_obj = (1 - ceiling_weight) * base_obj + ceiling_weight * ceiling_arr
        print(f"    Ceiling weight: {ceiling_weight:.0%} (blended objective)")

    proj_pts = np.array([p.get("projected_points", 0) for p in players], dtype=np.float64)
    rng = np.random.default_rng(seed)
    candidate_set = set()

    sal_floor = salary_floor_override if salary_floor_override is not None else None

    # Solve optimal lineup to establish projection quality floor
    optimal = _solve_mip(players, base_obj, salary_floor_override=sal_floor)
    proj_floor = None
    if proj_floor_override is not None:
        proj_floor = proj_floor_override
        print(f"    Projection floor: {proj_floor:.1f} pts (explicit override)")
    elif optimal is not None and min_proj_pct > 0:
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
                             proj_pts=proj_pts, proj_floor=proj_floor,
                             salary_floor_override=sal_floor)
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
                             proj_pts=proj_pts, proj_floor=proj_floor,
                             salary_floor_override=sal_floor)
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
                                 proj_pts=proj_pts, proj_floor=proj_floor,
                                 salary_floor_override=sal_floor)
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
                             proj_pts=proj_pts, proj_floor=proj_floor,
                             salary_floor_override=sal_floor)
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
                             proj_pts=proj_pts, proj_floor=proj_floor,
                             salary_floor_override=sal_floor)
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


def _solve_mip(players, obj, proj_pts=None, proj_floor=None, salary_floor_override=None):
    """Solve a single lineup MIP using HiGHS.

    Args:
        players: list of player dicts
        obj: objective coefficients (one per player)
        proj_pts: array of projected points per player (for floor constraint)
        proj_floor: minimum total projected points for the lineup
        salary_floor_override: optional override for minimum salary usage
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
    sal_floor = salary_floor_override if salary_floor_override is not None else SALARY_FLOOR
    salaries = np.array([float(p["salary"]) for p in players])
    h.addRow(float(sal_floor), float(SALARY_CAP), n, np.arange(n, dtype=np.int32), salaries)

    # Minimum projection floor (ensures lineup quality)
    if proj_pts is not None and proj_floor is not None:
        h.addRow(float(proj_floor), float(1e9), n, np.arange(n, dtype=np.int32),
                 proj_pts.astype(np.float64))

    h.run()
    if h.getModelStatus() != HighsModelStatus.kOptimal:
        return None

    sol = h.getSolution()
    return tuple(sorted(i for i in range(n) if sol.col_value[i] > 0.5))

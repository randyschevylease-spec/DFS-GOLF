"""Player Simulation Engine — Individual Player Score Generation.

THIS IS THE MISSING ARCHITECTURAL PIECE.

Generates a reusable (n_sims, n_players) score matrix that represents
individual player performance outcomes. The simulation count is tied
to the contest field size (default: 10x entries), not an arbitrary number.

Architecture:
  1. Build covariance matrix (wave-aware correlation)
  2. Cholesky decompose for correlated sampling
  3. Generate n_sims correlated player score draws
  4. Store as a reusable PlayerSimulation object
  5. Both candidate and opponent lineups score against the SAME draws
     (they experience the same "weather" each sim)

Why this matters (poker analogy):
  Think of each sim as a different "board runout" in poker.
  Every player at the table sees the same board — your hand
  (candidate lineup) and villain's hand (opponent lineup) are
  evaluated against identical community cards. Without this shared
  score matrix, candidates and opponents play on different boards,
  which is like calculating EV with different runouts for hero vs villain.

Sim count scaling:
  - 500-entry SE contest → 5,000 sims
  - 10K-entry GPP → 100,000 sims
  - 150K Milly Maker → 1,500,000 sims (capped to avoid OOM)
  This ensures tail payouts stabilize proportionally to field size.
"""
import time
import numpy as np
from scipy.stats import norm as sp_norm
from dataclasses import dataclass
from config import (SAME_WAVE_CORRELATION, DIFF_WAVE_CORRELATION,
                    BASE_CORRELATION, PLAYER_SIM_MULTIPLIER)


@dataclass
class PlayerSimulation:
    """Reusable player score matrix from Monte Carlo simulation.

    Attributes:
        scores: (n_sims, n_players) float32 — individual player DK point outcomes
        n_sims: number of simulations run
        n_players: number of players in the pool
        field_size: contest field size this was calibrated to
        cholesky_L: (n_players, n_players) — stored for potential re-use
        generation_time: seconds to generate
    """
    scores: np.ndarray          # (n_sims, n_players) float32
    n_sims: int
    n_players: int
    field_size: int
    cholesky_L: np.ndarray
    generation_time: float

    def score_lineups(self, lineup_matrix):
        """Score lineups against the shared player sim.

        Args:
            lineup_matrix: (n_lineups, n_players) binary float32 matrix

        Returns:
            (n_sims, n_lineups) float32 — total DK points per lineup per sim
        """
        # scores: (n_sims, n_players) @ lineup_matrix.T: (n_players, n_lineups)
        # result: (n_sims, n_lineups)
        return self.scores @ lineup_matrix.T


def build_covariance_matrix(players, waves=None):
    """Build wave-aware covariance matrix for correlated player scores.

    Args:
        players: list of player dicts with 'std_dev' key
        waves: optional list of 0/1 per player (0=PM, 1=AM)

    Returns:
        (n_players, n_players) covariance matrix
    """
    n = len(players)
    sigmas = np.array([_get_sigma(p) for p in players], dtype=np.float64)

    if waves is not None:
        waves_arr = np.array(waves)
        same_wave = (waves_arr[:, None] == waves_arr[None, :])
        corr_matrix = np.where(same_wave, SAME_WAVE_CORRELATION, DIFF_WAVE_CORRELATION)
        np.fill_diagonal(corr_matrix, 1.0)
        cov = np.outer(sigmas, sigmas) * corr_matrix
    else:
        cov = np.outer(sigmas, sigmas) * BASE_CORRELATION
        np.fill_diagonal(cov, sigmas ** 2)

    # Ensure positive definite
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += np.eye(n) * 1.0
        L = np.linalg.cholesky(cov)

    return cov, L, sigmas


def generate_player_sims(players, field_size, waves=None,
                         sim_multiplier=None, max_sims=1_500_000,
                         min_sims=5_000, seed=None,
                         mixture_params=None):
    """Generate the master player score simulation matrix.

    This is the core function — creates all individual player outcomes
    that every lineup (candidate and opponent) will be scored against.

    Args:
        players: list of player dicts
        field_size: total contest entries (drives sim count)
        waves: optional wave assignments per player
        sim_multiplier: override for PLAYER_SIM_MULTIPLIER config
        max_sims: hard cap to prevent OOM on massive contests
        min_sims: minimum sims regardless of field size
        seed: random seed for reproducibility
        mixture_params: optional bimodal distribution params

    Returns:
        PlayerSimulation object with reusable score matrix
    """
    t_start = time.time()
    n_players = len(players)

    # Calculate sim count from field size
    multiplier = sim_multiplier or PLAYER_SIM_MULTIPLIER
    n_sims = max(min_sims, min(field_size * multiplier, max_sims))

    print(f"  Player sim: {n_sims:,} sims ({multiplier}x field of {field_size:,})")
    if n_sims == max_sims:
        print(f"  ⚠ Capped at {max_sims:,} (field × {multiplier} = {field_size * multiplier:,})")

    # Build covariance and Cholesky
    means = np.array([p["projected_points"] for p in players], dtype=np.float64)
    cov, L, sigmas = build_covariance_matrix(players, waves)

    # Check mixture model availability
    use_mix = (mixture_params is not None and mixture_params[5].any())
    if use_mix:
        n_mix = int(mixture_params[5].sum())
        print(f"  Mixture distribution: {n_mix}/{n_players} players with bimodal scores")

    # Generate in batches for memory efficiency
    # Batch size scaled to available memory — Mac Mini Pro w/ 512GB can handle 2000+
    batch_size = min(2000, n_sims)
    scores = np.empty((n_sims, n_players), dtype=np.float32)

    rng = np.random.default_rng(seed)
    sims_done = 0

    while sims_done < n_sims:
        bs = min(batch_size, n_sims - sims_done)
        Z = rng.standard_normal((bs, n_players))
        X = Z @ L.T  # (bs, n_players) — correlated normals

        if use_mix:
            mix_p_miss, mix_mu_miss, mix_sigma_miss, mix_mu_make, mix_sigma_make, mix_flag = mixture_params
            batch_scores = _transform_mixture_scores(
                X, sigmas, mix_p_miss, mix_mu_miss, mix_sigma_miss,
                mix_mu_make, mix_sigma_make, mix_flag
            )
        else:
            batch_scores = means[None, :] + X  # (bs, n_players)

        np.maximum(batch_scores, 0.0, out=batch_scores)
        scores[sims_done:sims_done + bs] = batch_scores.astype(np.float32)
        sims_done += bs

    elapsed = time.time() - t_start
    mem_mb = scores.nbytes / 1024 / 1024
    print(f"  Generated {n_sims:,} × {n_players} score matrix "
          f"({mem_mb:.0f}MB) in {elapsed:.1f}s")

    return PlayerSimulation(
        scores=scores,
        n_sims=n_sims,
        n_players=n_players,
        field_size=field_size,
        cholesky_L=L,
        generation_time=elapsed,
    )


def build_lineup_matrix(lineups, n_players):
    """Convert list of lineup index-lists to binary matrix.

    Args:
        lineups: list of lists of player indices
        n_players: total player count

    Returns:
        (n_lineups, n_players) float32 binary matrix
    """
    n = len(lineups)
    matrix = np.zeros((n, n_players), dtype=np.float32)
    for i, lu in enumerate(lineups):
        for idx in lu:
            matrix[i, idx] = 1.0
    return matrix


# ── Scoring Helpers ──────────────────────────────────────────────────────────

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
    proj = player.get("projected_points", 50)
    return empirical_sigma_from_projection(proj)


# ── Mixture Distribution (Gaussian Copula) ─────────────────────────────────

def compute_mixture_params(players):
    """Derive per-player mixture distribution parameters.

    Miss-cut expected score is derived as proj - 2*std_dev (floored at 15 DK pts).
    The CSV FLOOR column is only proj - 1*std_dev, which is too high to represent
    a real missed cut (2 rounds of bad golf, no weekend, no finish bonus).

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
        ceil_val = p.get("ceiling", 0)
        mean_val = p.get("projected_points", 0)
        sd = p.get("std_dev", 0)

        if mc <= 0 or mc >= 1.0 or ceil_val <= 0 or sd <= 0:
            mu_make[i] = mean_val
            sigma_make[i] = _get_sigma(p)
            continue

        p_miss_i = 1.0 - mc
        p_make_i = mc

        mu_miss_i = max(mean_val - 2.0 * sd, 15.0)
        mu_make_i = (mean_val - p_miss_i * mu_miss_i) / p_make_i
        sigma_make_i = max((ceil_val - mu_make_i) / 2.0, 5.0)

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


def _transform_mixture_scores(X, sigmas, p_miss, mu_miss, sigma_miss,
                              mu_make, sigma_make, use_mixture):
    """Transform correlated normals to mixture-marginal scores via Gaussian copula."""
    bs, n = X.shape
    scores = np.empty((bs, n), dtype=np.float64)

    non_mix = ~use_mixture
    if non_mix.any():
        X_norm = X[:, non_mix] / sigmas[None, non_mix]
        scores[:, non_mix] = mu_make[non_mix] + sigma_make[non_mix] * X_norm

    mix = use_mixture
    if mix.any():
        X_mix = X[:, mix]
        sig_mix = sigmas[mix]
        X_norm = X_mix / sig_mix[None, :]
        U = sp_norm.cdf(X_norm)

        p_miss_mix = p_miss[mix]
        p_make_mix = 1.0 - p_miss_mix
        mu_miss_mix = mu_miss[mix]
        sigma_miss_mix = sigma_miss[mix]
        mu_make_mix = mu_make[mix]
        sigma_make_mix = sigma_make[mix]

        miss_mask = U < p_miss_mix[None, :]
        U_miss = np.clip(U / p_miss_mix[None, :], 1e-6, 1 - 1e-6)
        miss_scores = mu_miss_mix + sigma_miss_mix * sp_norm.ppf(U_miss)

        U_make = np.clip((U - p_miss_mix[None, :]) / p_make_mix[None, :], 1e-6, 1 - 1e-6)
        make_scores = mu_make_mix + sigma_make_mix * sp_norm.ppf(U_make)

        scores[:, mix] = np.where(miss_mask, miss_scores, make_scores)

    np.maximum(scores, 0.0, out=scores)
    return scores

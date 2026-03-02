"""Candidate Lineup Generator — Your lineups for the optimizer.

Generates diverse, high-quality candidate lineups via:
  Phase 1: Randomized MIP solves with noise (bulk exploration)
  Phase 2: Top-player exclusion (single + pair) for diversity
  Phase 3: Salary-tier diversification (stars & scrubs + balanced)
  Post: Diversity filter caps per-player exposure in candidate pool
"""
import numpy as np
from config import ROSTER_SIZE, SALARY_FLOOR
from mip_solver import solve_mip


def generate_candidates(players, pool_size=5000, noise_scale=0.15, seed=None,
                        min_proj_pct=0.88, candidate_exposure_cap=1.0,
                        ceiling_pts=None, ceiling_weight=0.0,
                        salary_floor_override=None, proj_floor_override=None):
    """Generate diverse, high-quality candidate lineups via randomized MIP solves.

    Args:
        players: list of player dicts
        pool_size: target number of raw MIP solves
        noise_scale: log-normal noise std dev for objective perturbation
        seed: random seed
        min_proj_pct: minimum lineup projection as fraction of optimal
        candidate_exposure_cap: max fraction of final candidates any player can appear in
        ceiling_pts: optional ceiling points per player for blended objective
        ceiling_weight: weight for ceiling in objective blend
        salary_floor_override: optional override for minimum salary usage
        proj_floor_override: optional explicit projection floor

    Returns:
        list of unique lineups (each a list of sorted player indices)
    """
    n = len(players)
    base_obj = np.array([p["projected_points"] for p in players])

    if ceiling_pts is not None and ceiling_weight > 0:
        ceiling_arr = np.array(ceiling_pts, dtype=np.float64)
        base_obj = (1 - ceiling_weight) * base_obj + ceiling_weight * ceiling_arr
        print(f"  Ceiling weight: {ceiling_weight:.0%} (blended objective)")

    proj_pts = np.array([p.get("projected_points", 0) for p in players], dtype=np.float64)
    rng = np.random.default_rng(seed)
    candidate_set = set()

    sal_floor = salary_floor_override if salary_floor_override is not None else None

    # Solve optimal lineup to establish projection quality floor
    optimal = solve_mip(players, base_obj, salary_floor_override=sal_floor)
    proj_floor = None
    if proj_floor_override is not None:
        proj_floor = proj_floor_override
        print(f"  Projection floor: {proj_floor:.1f} pts (explicit override)")
    elif optimal is not None and min_proj_pct > 0:
        max_proj = sum(proj_pts[i] for i in optimal)
        proj_floor = max_proj * min_proj_pct
        print(f"  Projection floor: {proj_floor:.1f} pts "
              f"({min_proj_pct:.0%} of optimal {max_proj:.1f})")

    # Phase 1: Projection-based with noise
    phase1_noise = noise_scale * 1.3
    batch = 1000
    for batch_start in range(0, pool_size, batch):
        before = len(candidate_set)
        for _ in range(min(batch, pool_size - batch_start)):
            noise = np.exp(rng.normal(0.0, phase1_noise, size=n))
            sel = solve_mip(players, base_obj * noise,
                            proj_pts=proj_pts, proj_floor=proj_floor,
                            salary_floor_override=sal_floor)
            if sel is not None:
                candidate_set.add(sel)

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
            sel = solve_mip(players, obj,
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
                sel = solve_mip(players, obj,
                                proj_pts=proj_pts, proj_floor=proj_floor,
                                salary_floor_override=sal_floor)
                if sel is not None:
                    candidate_set.add(sel)

    # Phase 4: Salary-tier diversification
    salaries = np.array([float(p["salary"]) for p in players])
    tier_count = max(200, pool_size // 5)

    # Stars & scrubs
    high_idx = np.where(salaries >= 9000)[0]
    low_idx = np.where(salaries < 7500)[0]
    if len(high_idx) >= 2 and len(low_idx) >= 4:
        for _ in range(tier_count):
            obj = base_obj * np.exp(rng.normal(0.0, noise_scale * 1.5, size=n))
            mid_mask = (salaries >= 7500) & (salaries < 9000)
            obj[mid_mask] *= 0.3
            sel = solve_mip(players, obj,
                            proj_pts=proj_pts, proj_floor=proj_floor,
                            salary_floor_override=sal_floor)
            if sel is not None:
                candidate_set.add(sel)

    # Balanced
    mid_idx = np.where((salaries >= 7000) & (salaries <= 9000))[0]
    if len(mid_idx) >= ROSTER_SIZE:
        for _ in range(tier_count):
            obj = base_obj * np.exp(rng.normal(0.0, noise_scale * 1.5, size=n))
            extreme_mask = (salaries < 7000) | (salaries > 9000)
            obj[extreme_mask] *= 0.3
            sel = solve_mip(players, obj,
                            proj_pts=proj_pts, proj_floor=proj_floor,
                            salary_floor_override=sal_floor)
            if sel is not None:
                candidate_set.add(sel)

    raw_count = len(candidate_set)

    # Diversity filter
    all_candidates = list(candidate_set)
    if candidate_exposure_cap < 1.0 and len(all_candidates) > 100:
        all_candidates = _diversity_filter(all_candidates, proj_pts, n,
                                           candidate_exposure_cap)
        print(f"  Candidates: {raw_count} raw → {len(all_candidates)} after "
              f"diversity filter (exposure cap {candidate_exposure_cap:.0%})")
    else:
        print(f"  Candidates: {len(all_candidates)}")

    return [list(c) for c in all_candidates]


def _diversity_filter(candidates, proj_pts, n_players, exposure_cap):
    """Greedy diversity filter: keep highest-projection candidates first,
    skipping any that would push a player over the exposure cap."""
    target_size = len(candidates)
    max_appearances = max(10, int(target_size * exposure_cap))

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

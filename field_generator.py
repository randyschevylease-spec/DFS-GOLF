"""Archetype-Based DFS Field Generator.

Generates N simulated opponent lineups that realistically model a DFS
tournament field. The field is NOT homogeneous — it contains different
types of players with different construction strategies:

  - Chalk:     Heavy on top projected players, boosted ownership
  - Content:   Follows content-site value plays, tight salary usage
  - Optimizer: MIP-solved with ±4% noise (simulates real optimizer tools)
  - Sharp:     MIP-solved with ownership penalty (fades chalk)
  - Random:    Casual/recreational, salary-biased, loose constraints

Two generation modes for speed at scale:
  - Stochastic (chalk, content, random): Dirichlet-multinomial sampling
    with archetype-specific alpha profiles. Generates 1000s/sec.
  - MIP-based (optimizer, sharp): Batch MIP solves with noise/penalty.
    ~1ms per solve, deduplicates results.

Iterative ownership calibration adjusts selection weights so the
resulting field's ownership distribution converges on projected targets.
"""
import random
import numpy as np
from dataclasses import dataclass
from engine import _solve_mip
from config import ROSTER_SIZE, SALARY_CAP, SALARY_FLOOR


# ── Output Types ──────────────────────────────────────────────────────────

@dataclass
class FieldLineup:
    players: list           # List of player indices (into the players array)
    salary: int             # Total salary used
    projected_points: float # Sum of player projections
    geometric_ownership: float  # Geometric mean of individual ownership %s
    archetype: str          # Which archetype generated this lineup


@dataclass
class GeneratedField:
    lineups: list           # list[FieldLineup]
    ownership_validation: dict  # {player_name: {target, actual, delta}}
    archetype_distribution: dict  # {archetype_name: count}


# ── Default Archetype Weights ─────────────────────────────────────────────

DEFAULT_ARCHETYPE_WEIGHTS = {
    "chalk":     0.25,
    "content":   0.20,
    "optimizer": 0.25,
    "sharp":     0.10,
    "random":    0.20,
}


# ── Dirichlet-Multinomial Sampling (chalk, content, random) ──────────────

def _sample_lineups(players, n_lineups, probs, alpha_scale, min_sal,
                    min_salary, rng, sal_bias_power=4.0):
    """Fast Dirichlet-multinomial sampling for stochastic archetypes.

    Same proven algorithm from engine.py that generated 95K+ lineups.
    """
    n = len(players)
    sal_arr = np.array([p["salary"] for p in players], dtype=np.float64)
    alpha = np.maximum(probs * alpha_scale * n, 0.01)
    lineups = []
    attempts = 0

    while len(lineups) < n_lineups and attempts < n_lineups * 25:
        attempts += 1
        try:
            draw = rng.dirichlet(alpha)
        except Exception:
            draw = probs / probs.sum() if probs.sum() > 0 else np.ones(n) / n

        selected = []
        budget = SALARY_CAP
        avail = np.ones(n, dtype=bool)
        ok = True

        for slot in range(ROSTER_SIZE):
            remaining_slots = ROSTER_SIZE - slot - 1
            min_remaining = remaining_slots * min_sal
            max_affordable = budget - min_remaining

            afford = (sal_arr <= max_affordable) & avail
            if not afford.any():
                ok = False
                break

            vp = draw * afford
            sal_weight = (sal_arr / min_sal) ** (sal_bias_power + slot * 1.5)
            vp = vp * sal_weight * afford

            vp_sum = vp.sum()
            if vp_sum <= 0:
                ok = False
                break
            vp /= vp_sum

            c = rng.choice(n, p=vp)
            selected.append(c)
            budget -= sal_arr[c]
            avail[c] = False

        if ok and len(selected) == ROSTER_SIZE:
            total_sal = sal_arr[selected].sum()
            if min_salary <= total_sal <= SALARY_CAP:
                lineups.append(selected)

    return lineups


# ── MIP Batch Generation (optimizer, sharp) ───────────────────────────────

def _generate_mip_batch(players, n_lineups, mode, weight_adj, rng):
    """Batch-generate MIP-solved lineups with noise for diversity.

    mode="optimizer": ±4% noise on projections
    mode="sharp":     ownership penalty + ±3% noise
    """
    n = len(players)
    base_proj = np.array([p["projected_points"] for p in players])
    owns = np.array([p.get("proj_ownership", 5.0) for p in players])

    lineup_set = set()
    target = n_lineups * 3

    for _ in range(target):
        if len(lineup_set) >= n_lineups:
            break

        if mode == "optimizer":
            # ±15% noise for diverse optimizer lineups (real field has many different optimizer setups)
            noise = np.exp(rng.normal(0.0, 0.15, size=n))
            obj = base_proj * noise
        else:  # sharp
            penalty = rng.uniform(0.3, 0.8)
            # ±12% noise + ownership penalty
            noise = np.exp(rng.normal(0.0, 0.12, size=n))
            obj = (base_proj - penalty * owns) * noise

        result = _solve_mip(players, obj)
        if result is not None:
            lineup_set.add(result)

    lineups = [list(s) for s in lineup_set]

    # Duplicate to fill if needed
    if len(lineups) < n_lineups and lineups:
        extra = n_lineups - len(lineups)
        indices = rng.integers(0, len(lineups), size=extra)
        lineups.extend([lineups[i] for i in indices])

    return lineups[:n_lineups]


# ── Ownership Calibration ─────────────────────────────────────────────────

def _measure_ownership(index_lists, n_players):
    """Measure ownership % from a list of index-list lineups."""
    counts = np.zeros(n_players)
    for lu in index_lists:
        for idx in lu:
            counts[idx] += 1
    if len(index_lists) == 0:
        return counts
    return counts / len(index_lists) * 100


# ── Main Field Generator ─────────────────────────────────────────────────

def generate_field(players, field_size, archetype_weights=None,
                   ownership_tolerance=0.03, max_iterations=10, seed=None):
    """Generate a realistic opponent field using archetype-based construction.

    Two-phase approach:
      Phase 1: Pilot calibration — generate small batches to iteratively
               tune per-player selection probabilities until ownership
               converges on projected targets.
      Phase 2: Full generation with calibrated probabilities.

    Stochastic archetypes (chalk, content, random) use fast Dirichlet-
    multinomial sampling. MIP archetypes (optimizer, sharp) use batch
    MIP solves with noise for diversity.
    """
    rng = np.random.default_rng(seed)
    if seed is not None:
        random.seed(seed)

    if archetype_weights is None:
        archetype_weights = DEFAULT_ARCHETYPE_WEIGHTS.copy()

    total_weight = sum(archetype_weights.values())
    if abs(total_weight - 1.0) > 0.01:
        raise ValueError(f"Archetype weights must sum to 1.0, got {total_weight}")

    n = len(players)
    sal_arr = np.array([p["salary"] for p in players], dtype=np.float64)
    min_sal = float(sal_arr.min())
    target_own = np.array([max(p.get("proj_ownership", 1.0), 0.1) for p in players])
    tolerance_pct = ownership_tolerance * 100

    # Compute archetype counts
    archetype_counts = {}
    remaining = field_size
    archetypes_list = list(archetype_weights.keys())
    for i, name in enumerate(archetypes_list):
        if i == len(archetypes_list) - 1:
            archetype_counts[name] = remaining
        else:
            count = int(round(field_size * archetype_weights[name]))
            archetype_counts[name] = count
            remaining -= count

    print(f"  Archetype mix ({field_size:,} lineups):", flush=True)
    for name, count in archetype_counts.items():
        pct = count / field_size * 100
        print(f"    {name:<12} {count:>7,} ({pct:.0f}%)", flush=True)

    # Base selection probabilities (from projected ownership)
    base_probs = target_own.copy()
    base_probs /= base_probs.sum()

    # ── Phase 1: Pilot calibration (stochastic archetypes only) ──
    # MIP archetypes respond weakly to calibration, so we calibrate
    # the stochastic ones and accept MIP as-is.
    stochastic_weight = sum(archetype_weights.get(a, 0) for a in ["chalk", "content", "random"])
    adjusted_probs = base_probs.copy()

    pilot_size = min(3000, field_size)

    for iteration in range(max_iterations):
        # Generate pilot with current adjusted probs
        pilot = []
        for arch_name in ["chalk", "content", "random"]:
            arch_frac = archetype_weights.get(arch_name, 0) / stochastic_weight if stochastic_weight > 0 else 0
            arch_count = int(pilot_size * arch_frac)
            if arch_count == 0:
                continue

            alpha_probs = _archetype_alpha(arch_name, players, adjusted_probs, rng)
            min_salary = {"chalk": SALARY_CAP - 1000,
                          "content": SALARY_CAP - 800,
                          "random": SALARY_CAP - 3000}.get(arch_name, SALARY_FLOOR)

            lus = _sample_lineups(players, arch_count, alpha_probs, 12.0,
                                  min_sal, min_salary, rng)
            pilot.extend(lus)

        if not pilot:
            break

        actual_own = _measure_ownership(pilot, n)
        actual_safe = np.maximum(actual_own, 0.01)
        ratio = target_own / actual_safe
        adjustment = np.power(ratio, 0.5)  # conservative damping
        adjustment = np.clip(adjustment, 0.5, 2.0)  # tight bounds

        delta = np.abs(actual_own - target_own)
        max_delta = float(delta.max())
        n_bad = int((delta > tolerance_pct).sum())

        status = "CONVERGED" if n_bad == 0 else f"{n_bad} players off"
        print(f"    Pilot [{iteration+1}/{max_iterations}]: "
              f"max_delta={max_delta:.1f}% | {status}", flush=True)

        if n_bad == 0:
            break

        # Update probs with bounded adjustment
        adjusted_probs *= adjustment
        adjusted_probs = np.maximum(adjusted_probs, 0.001)
        adjusted_probs /= adjusted_probs.sum()

    # ── Phase 2: Full generation ──
    print(f"  Generating full field ({field_size:,} lineups)...", flush=True)

    all_lineups = []  # list of (index_list, archetype_name) tuples

    for arch_name, arch_count in archetype_counts.items():
        if arch_count <= 0:
            continue

        t0 = __import__('time').time()

        if arch_name in ("optimizer", "sharp"):
            # MIP batch — high noise for diversity (calibration is stochastic-only)
            lus = _generate_mip_batch(players, arch_count, arch_name,
                                       None, rng)
            for lu in lus:
                all_lineups.append((lu, arch_name))

        else:
            # Stochastic sampling
            alpha_probs = _archetype_alpha(arch_name, players, adjusted_probs, rng)
            min_salary = {"chalk": SALARY_CAP - 1000,
                          "content": SALARY_CAP - 800,
                          "random": SALARY_CAP - 3000}.get(arch_name, SALARY_FLOOR)
            sal_bias = {"chalk": 5.0, "content": 5.0, "random": 3.0}.get(arch_name, 4.0)

            lus = _sample_lineups(players, arch_count, alpha_probs, 12.0,
                                  min_sal, min_salary, rng, sal_bias_power=sal_bias)
            for lu in lus:
                all_lineups.append((lu, arch_name))

        elapsed = __import__('time').time() - t0
        actual = sum(1 for _, a in all_lineups if a == arch_name)
        print(f"    {arch_name:<12} {actual:>7,}/{arch_count:>7,} in {elapsed:.1f}s", flush=True)

    # Shuffle
    rng.shuffle(all_lineups)

    # Wrap as FieldLineup objects
    lineups = []
    for lu_idx, arch_name in all_lineups:
        total_sal = int(sal_arr[lu_idx].sum())
        total_proj = sum(players[j]["projected_points"] for j in lu_idx)
        owns = [max(players[j].get("proj_ownership", 0.1), 0.01) for j in lu_idx]
        geo_own = float(np.exp(np.mean(np.log(owns))))

        lineups.append(FieldLineup(
            players=list(lu_idx),
            salary=total_sal,
            projected_points=total_proj,
            geometric_ownership=geo_own,
            archetype=arch_name,
        ))

    # Ownership validation
    index_lists = [lu_idx for lu_idx, _ in all_lineups]
    actual_own = _measure_ownership(index_lists, n)
    ownership_validation = {}
    for i in range(n):
        ownership_validation[players[i]["name"]] = {
            "target": float(target_own[i]),
            "actual": float(actual_own[i]),
            "delta": float(actual_own[i] - target_own[i]),
        }

    deltas = [abs(v["delta"]) for v in ownership_validation.values()]
    print(f"  Final ownership: mean_delta={np.mean(deltas):.1f}% | "
          f"max_delta={max(deltas):.1f}%", flush=True)

    archetype_dist = {}
    for lu in lineups:
        archetype_dist[lu.archetype] = archetype_dist.get(lu.archetype, 0) + 1

    return GeneratedField(
        lineups=lineups,
        ownership_validation=ownership_validation,
        archetype_distribution=archetype_dist,
    )


def _archetype_alpha(arch_name, players, base_probs, rng):
    """Build archetype-specific alpha profile from base probabilities."""
    n = len(players)

    if arch_name == "chalk":
        # Boost top 15 projected, suppress rest
        sorted_idx = np.argsort([-p["projected_points"] for p in players])
        top_set = set(sorted_idx[:15].tolist())
        alpha = np.zeros(n)
        for i in range(n):
            if i in top_set:
                alpha[i] = base_probs[i] * rng.uniform(1.3, 1.8)
            else:
                alpha[i] = base_probs[i] * rng.uniform(0.4, 0.7)
        return np.maximum(alpha, 0.001)

    elif arch_name == "content":
        # Ownership + value bias
        alpha = np.zeros(n)
        for i in range(n):
            val = players[i].get("value", 0)
            if val > 0.5:
                alpha[i] = base_probs[i] * 1.2
            else:
                alpha[i] = base_probs[i]
        return np.maximum(alpha, 0.001)

    elif arch_name == "random":
        # Salary-biased with noise
        alpha = np.zeros(n)
        for i in range(n):
            alpha[i] = (players[i]["salary"] / 1000) + rng.uniform(0, 3)
        return np.maximum(alpha, 0.001)

    else:
        return np.maximum(base_probs, 0.001)


def field_to_index_lists(generated_field):
    """Convert GeneratedField to list of index lists (for simulator compatibility)."""
    return [lu.players for lu in generated_field.lineups]

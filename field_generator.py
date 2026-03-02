"""Archetype-Based DFS Field Generator.

Generates N simulated opponent lineups modeling a realistic DFS field.
Uses shared sampling.py and mip_solver.py — no duplicate code.

Archetypes:
  - Chalk: Heavy on top projected players, boosted ownership
  - Content: Follows content-site value plays, tight salary usage
  - Optimizer: MIP-solved with ±15% noise (simulates optimizer users)
  - Sharp: MIP-solved with ownership penalty (fades chalk)
  - Random: Casual/recreational, salary-biased, loose constraints
"""
import time
import numpy as np
from dataclasses import dataclass
from config import ROSTER_SIZE, SALARY_CAP, SALARY_FLOOR
from sampling import sample_lineups
from mip_solver import solve_mip


# ── Output Types ──────────────────────────────────────────────────────────

@dataclass
class FieldLineup:
    players: list
    salary: int
    projected_points: float
    geometric_ownership: float
    archetype: str


@dataclass
class GeneratedField:
    lineups: list
    ownership_validation: dict
    archetype_distribution: dict


# ── Default Archetype Weights ─────────────────────────────────────────────

DEFAULT_ARCHETYPE_WEIGHTS = {
    "chalk": 0.25,
    "content": 0.20,
    "optimizer": 0.25,
    "sharp": 0.10,
    "random": 0.20,
}


# ── MIP Batch Generation (optimizer, sharp) ───────────────────────────────

def _generate_mip_batch(players, n_lineups, mode, rng,
                        salary_floor_override=None):
    """Batch-generate MIP-solved lineups with noise for diversity."""
    n = len(players)
    base_proj = np.array([p["projected_points"] for p in players])
    owns = np.array([p.get("proj_ownership", 5.0) for p in players])

    lineup_set = set()
    target = int(n_lineups * 1.8)  # Reduced from 3x — 1.8x is plenty with good noise

    for _ in range(target):
        if len(lineup_set) >= n_lineups:
            break

        if mode == "optimizer":
            noise = np.exp(rng.normal(0.0, 0.15, size=n))
            obj = base_proj * noise
        else:  # sharp
            penalty = rng.uniform(0.3, 0.8)
            noise = np.exp(rng.normal(0.0, 0.12, size=n))
            obj = (base_proj - penalty * owns) * noise

        result = solve_mip(players, obj, salary_floor_override=salary_floor_override)
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


# ── Archetype Alpha Profiles ─────────────────────────────────────────────

def _archetype_alpha(arch_name, players, base_probs, rng):
    """Build archetype-specific alpha profile from base probabilities."""
    n = len(players)

    if arch_name == "chalk":
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
        alpha = np.zeros(n)
        for i in range(n):
            val = players[i].get("value", 0)
            if val > 0.5:
                alpha[i] = base_probs[i] * 1.2
            else:
                alpha[i] = base_probs[i]
        return np.maximum(alpha, 0.001)

    elif arch_name == "random":
        alpha = np.zeros(n)
        for i in range(n):
            alpha[i] = (players[i]["salary"] / 1000) + rng.uniform(0, 3)
        return np.maximum(alpha, 0.001)

    else:
        return np.maximum(base_probs, 0.001)


# ── Main Field Generator ─────────────────────────────────────────────────

def generate_field(players, field_size, archetype_weights=None,
                   ownership_tolerance=0.03, max_iterations=10, seed=None,
                   min_salary_map=None):
    """Generate a realistic opponent field using archetype-based construction.

    Two-phase approach:
      Phase 1: Pilot calibration for stochastic archetypes
      Phase 2: Full generation with calibrated probabilities
    """
    rng = np.random.default_rng(seed)

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

    # Base selection probabilities
    base_probs = target_own.copy()
    base_probs /= base_probs.sum()

    # Phase 1: Pilot calibration (stochastic archetypes only)
    stochastic_weight = sum(archetype_weights.get(a, 0) for a in ["chalk", "content", "random"])
    adjusted_probs = base_probs.copy()
    pilot_size = min(3000, field_size)

    for iteration in range(max_iterations):
        pilot = []
        for arch_name in ["chalk", "content", "random"]:
            arch_frac = archetype_weights.get(arch_name, 0) / stochastic_weight if stochastic_weight > 0 else 0
            arch_count = int(pilot_size * arch_frac)
            if arch_count == 0:
                continue

            alpha_probs = _archetype_alpha(arch_name, players, adjusted_probs, rng)
            default_mins = {"chalk": SALARY_CAP - 1000,
                            "content": SALARY_CAP - 800,
                            "random": SALARY_CAP - 3000}
            if min_salary_map and arch_name in min_salary_map:
                min_salary = min_salary_map[arch_name]
            else:
                min_salary = default_mins.get(arch_name, SALARY_FLOOR)

            lus = sample_lineups(players, arch_count, alpha_probs, 12.0,
                                 min_sal, min_salary, rng)
            pilot.extend(lus)

        if not pilot:
            break

        actual_own = _measure_ownership(pilot, n)
        actual_safe = np.maximum(actual_own, 0.01)
        ratio = target_own / actual_safe
        adjustment = np.power(ratio, 0.5)
        adjustment = np.clip(adjustment, 0.5, 2.0)

        delta = np.abs(actual_own - target_own)
        max_delta = float(delta.max())
        n_bad = int((delta > tolerance_pct).sum())

        status = "CONVERGED" if n_bad == 0 else f"{n_bad} players off"
        print(f"  Pilot [{iteration+1}/{max_iterations}]: "
              f"max_delta={max_delta:.1f}% | {status}", flush=True)

        if n_bad == 0:
            break

        adjusted_probs *= adjustment
        adjusted_probs = np.maximum(adjusted_probs, 0.001)
        adjusted_probs /= adjusted_probs.sum()

    # Phase 2: Full generation
    print(f"  Generating full field ({field_size:,} lineups)...", flush=True)

    all_lineups = []

    for arch_name, arch_count in archetype_counts.items():
        if arch_count <= 0:
            continue

        t0 = time.time()

        if arch_name in ("optimizer", "sharp"):
            mip_floor = min_salary_map.get(arch_name) if min_salary_map else None
            lus = _generate_mip_batch(players, arch_count, arch_name,
                                      rng, salary_floor_override=mip_floor)
            for lu in lus:
                all_lineups.append((lu, arch_name))
        else:
            alpha_probs = _archetype_alpha(arch_name, players, adjusted_probs, rng)
            default_mins = {"chalk": SALARY_CAP - 1000,
                            "content": SALARY_CAP - 800,
                            "random": SALARY_CAP - 3000}
            if min_salary_map and arch_name in min_salary_map:
                min_salary = min_salary_map[arch_name]
            else:
                min_salary = default_mins.get(arch_name, SALARY_FLOOR)
            sal_bias = {"chalk": 5.0, "content": 5.0, "random": 3.0}.get(arch_name, 4.0)

            lus = sample_lineups(players, arch_count, alpha_probs, 12.0,
                                 min_sal, min_salary, rng, sal_bias_power=sal_bias)
            for lu in lus:
                all_lineups.append((lu, arch_name))

        elapsed = time.time() - t0
        actual = sum(1 for _, a in all_lineups if a == arch_name)
        print(f"    {arch_name:<12} {actual:>7,}/{arch_count:>7,} in {elapsed:.1f}s", flush=True)

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


def field_to_index_lists(generated_field):
    """Convert GeneratedField to list of index lists (for simulator compatibility)."""
    return [lu.players for lu in generated_field.lineups]

"""
generate_field.py — Synthetic field generator for Milly Maker GPP simulation.

Generates 3 independent synthetic fields of 105,800 lineups each,
modeling how the real DFS field builds lineups using tiered ownership weighting.

Field composition per field:
  - 40% chalk-casual: weight by projected_ownership^1.5
  - 40% semi-sharp:   weight by projected_ownership * sim_mean
  - 20% contrarian:   weight by sim_p90 / (projected_ownership + 0.01)

Each field gets independent ownership perturbation before lineup construction.

Outputs:
  - data/cache/field_1.npy, field_2.npy, field_3.npy
  - data/cache/field_ownership_summary.csv
"""

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "engine"))

SIM_PROFILES = os.path.join(PROJECT_ROOT, "data", "cache", "sim_profiles_current.csv")
DK_PROJECTIONS = os.path.join(PROJECT_ROOT, "data", "raw", "dk_projections_players.csv")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")

SALARY_CAP = 50_000
LINEUP_SIZE = 6
FIELD_SIZE = 105_800


def load_player_pool():
    """
    Load and merge player pool from sim profiles + DK projections.

    Returns:
        dict with numpy arrays: dk_ids, salaries, ownership, sim_mean, sim_p90, names
    """
    # Load DK projections keyed by datagolf name
    dk_by_name = {}
    with open(DK_PROJECTIONS) as f:
        for row in csv.DictReader(f):
            name = row["datagolf_name"]
            salary = int(row["dk_salary"])
            if salary <= 0:
                continue
            dk_by_name[name] = {
                "dk_id": int(row["dk_id"]),
                "dk_salary": salary,
                "projected_ownership": float(row.get("projected_ownership") or 5),
                "total_points": float(row.get("total_points") or 0),
            }

    # Load sim profiles and merge
    names = []
    dk_ids = []
    salaries = []
    ownership = []
    sim_mean = []
    sim_p90 = []

    with open(SIM_PROFILES) as f:
        for row in csv.DictReader(f):
            name = row["player_name"]
            dk = dk_by_name.get(name)
            if dk is None:
                continue

            avg_made = float(row.get("avg_pts_made") or 0)
            avg_missed = float(row.get("avg_pts_missed") or 28)
            cut_rate = float(row.get("made_cut_rate") or 0.5)
            std_made = float(row.get("std_pts_made") or 19)

            mean = avg_made * cut_rate + avg_missed * (1 - cut_rate)
            p90 = avg_made + 1.28 * std_made

            names.append(name)
            dk_ids.append(dk["dk_id"])
            salaries.append(dk["dk_salary"])
            ownership.append(dk["projected_ownership"])
            sim_mean.append(mean)
            sim_p90.append(p90)

    return {
        "names": names,
        "dk_ids": np.array(dk_ids, dtype=np.int64),
        "salaries": np.array(salaries, dtype=np.int32),
        "ownership": np.array(ownership, dtype=np.float64),
        "sim_mean": np.array(sim_mean, dtype=np.float64),
        "sim_p90": np.array(sim_p90, dtype=np.float64),
    }


def perturb_ownership(ownership, rng):
    """
    Apply variance bands to projected ownership.

    - >= 15%: ±15% relative noise
    - 8-15%: ±25% relative noise
    - < 8%:  ±45% relative noise

    Returns perturbed ownership array clipped to [0.01, 0.85].
    """
    perturbed = ownership.copy()
    n = len(ownership)

    noise_scale = np.where(
        ownership >= 15, 0.15,
        np.where(ownership >= 8, 0.25, 0.45)
    )
    noise = rng.normal(0, noise_scale, n)
    perturbed = ownership * (1 + noise)
    return np.clip(perturbed, 0.01, 0.85)


def build_tier_lineups(n_lineups, weights, salaries, n_players, salary_cap,
                       lineup_size, rng):
    """
    Build lineups for one tier using vectorized numpy operations.

    Strategy: batch-sample many lineups at once, then fix salary violations.
    """
    # Normalize weights to probabilities
    probs = weights / weights.sum()

    # Pre-allocate output
    lineups = np.zeros((n_lineups, lineup_size), dtype=np.int32)
    valid_count = 0
    batch_size = min(n_lineups * 2, 500_000)

    while valid_count < n_lineups:
        remaining = n_lineups - valid_count
        batch = min(batch_size, remaining * 3)

        # Vectorized weighted sampling: sample lineup_size players per lineup
        # numpy choice doesn't support per-row without-replacement in batch,
        # so we sample with a Gumbel-max trick for speed
        # log(-log(U)) + log(p) — top-k gives weighted sample without replacement
        u = rng.random((batch, n_players))
        u = np.clip(u, 1e-30, 1.0)
        keys = np.log(-np.log(u)) + np.log(probs)[np.newaxis, :]
        # Smallest keys = highest priority (Gumbel-max trick for top-k)
        # We want top-k by weight, which corresponds to SMALLEST key values
        # Actually: argpartition for top-k largest log(p) - log(-log(u))
        # Flip sign: largest -keys = most likely to be sampled
        indices = np.argpartition(-keys, lineup_size, axis=1)[:, :lineup_size]

        # Compute salaries
        lineup_salaries = salaries[indices].sum(axis=1)

        # Filter valid salary lineups
        valid_mask = lineup_salaries <= salary_cap
        valid_indices = indices[valid_mask]

        # Check for duplicate players within lineup (shouldn't happen with
        # Gumbel trick, but safety check)
        if valid_indices.shape[0] > 0:
            sorted_li = np.sort(valid_indices, axis=1)
            no_dupes = np.all(np.diff(sorted_li, axis=1) > 0, axis=1)
            valid_indices = valid_indices[no_dupes]

        take = min(len(valid_indices), n_lineups - valid_count)
        if take > 0:
            lineups[valid_count:valid_count + take] = valid_indices[:take]
            valid_count += take

    return lineups


def attempt_salary_fix(lineups, salaries, salary_cap, n_players, rng):
    """
    For lineups exceeding salary cap, swap highest-salary player with a
    random cheaper player that brings lineup under cap.
    """
    lineup_costs = salaries[lineups].sum(axis=1)
    over_mask = lineup_costs > salary_cap

    if not over_mask.any():
        return lineups

    over_idx = np.where(over_mask)[0]
    for idx in over_idx:
        lu = lineups[idx]
        lu_salaries = salaries[lu]
        # Find highest salary player
        worst = np.argmax(lu_salaries)
        current_cost = lu_salaries.sum()
        max_replacement_salary = salary_cap - (current_cost - lu_salaries[worst])

        # Find valid replacements
        used = set(lu)
        candidates = np.where(salaries <= max_replacement_salary)[0]
        candidates = candidates[~np.isin(candidates, lu)]

        if len(candidates) > 0:
            replacement = rng.choice(candidates)
            lineups[idx, worst] = replacement

    return lineups


def generate_single_field(args):
    """Generate one complete field of FIELD_SIZE lineups. Called in parallel."""
    field_idx, pool_data = args
    t0 = time.perf_counter()
    seed = 42 + field_idx * 1000
    rng = np.random.default_rng(seed)

    dk_ids = pool_data["dk_ids"]
    salaries = pool_data["salaries"]
    ownership = pool_data["ownership"]
    sim_mean = pool_data["sim_mean"]
    sim_p90 = pool_data["sim_p90"]
    n_players = len(dk_ids)

    # Perturb ownership for this field
    perturbed_own = perturb_ownership(ownership, rng)

    # Tier sizes
    n_chalk = 42_320
    n_semi = 42_320
    n_contrarian = FIELD_SIZE - n_chalk - n_semi  # 21,160

    # Compute tier weights
    chalk_weights = perturbed_own ** 1.5
    semi_weights = perturbed_own * sim_mean
    contrarian_weights = sim_p90 / (perturbed_own + 0.01)

    # Ensure no negative/zero weights
    chalk_weights = np.maximum(chalk_weights, 1e-10)
    semi_weights = np.maximum(semi_weights, 1e-10)
    contrarian_weights = np.maximum(contrarian_weights, 1e-10)

    print(f"  Field {field_idx + 1}: building chalk tier ({n_chalk:,} lineups)...")
    chalk = build_tier_lineups(n_chalk, chalk_weights, salaries, n_players,
                               SALARY_CAP, LINEUP_SIZE, rng)

    print(f"  Field {field_idx + 1}: building semi-sharp tier ({n_semi:,} lineups)...")
    semi = build_tier_lineups(n_semi, semi_weights, salaries, n_players,
                              SALARY_CAP, LINEUP_SIZE, rng)

    print(f"  Field {field_idx + 1}: building contrarian tier ({n_contrarian:,} lineups)...")
    contrarian = build_tier_lineups(n_contrarian, contrarian_weights, salaries, n_players,
                                    SALARY_CAP, LINEUP_SIZE, rng)

    # Combine all tiers — indices into player pool
    all_lineups_idx = np.vstack([chalk, semi, contrarian])

    # Convert player indices to dk_ids
    field_dk_ids = dk_ids[all_lineups_idx]

    elapsed = time.perf_counter() - t0
    print(f"  Field {field_idx + 1}: done in {elapsed:.1f}s")

    return field_idx, field_dk_ids, all_lineups_idx, perturbed_own


def compute_field_ownership(lineup_indices, n_players, field_size):
    """Compute realized ownership % per player from lineup indices."""
    counts = np.bincount(lineup_indices.ravel(), minlength=n_players)
    return counts / field_size * 100


def validate_field(field_dk_ids, lineup_indices, salaries, field_idx):
    """Run all validation assertions on a generated field."""
    n = field_dk_ids.shape[0]
    assert n == FIELD_SIZE, f"Field {field_idx + 1}: expected {FIELD_SIZE} rows, got {n}"

    # Check 6 unique players per lineup
    sorted_ids = np.sort(field_dk_ids, axis=1)
    diffs = np.diff(sorted_ids, axis=1)
    assert np.all(diffs > 0), f"Field {field_idx + 1}: found lineups with duplicate players"

    # Check salary cap
    lineup_costs = salaries[lineup_indices].sum(axis=1)
    max_cost = lineup_costs.max()
    assert max_cost <= SALARY_CAP, \
        f"Field {field_idx + 1}: max salary {max_cost} exceeds cap {SALARY_CAP}"

    print(f"  Field {field_idx + 1} validation PASSED: "
          f"{n:,} lineups, 6 unique players each, "
          f"max salary ${max_cost:,}")


def generate_fields():
    """Main entry point: generate 3 fields in parallel with validation."""
    t_total = time.perf_counter()
    print("Loading player pool...")
    pool = load_player_pool()
    n_players = len(pool["dk_ids"])
    print(f"  {n_players} players loaded")

    print(f"\nGenerating {3} fields of {FIELD_SIZE:,} lineups each...")

    # Generate fields — use ProcessPoolExecutor for true parallelism
    # But numpy + large arrays can have pickle overhead, so we also
    # support sequential fallback
    results = []
    try:
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(3):
                futures.append(executor.submit(generate_single_field, (i, pool)))
            for future in futures:
                results.append(future.result())
    except Exception as e:
        print(f"  Parallel generation failed ({e}), falling back to sequential...")
        for i in range(3):
            results.append(generate_single_field((i, pool)))

    # Sort by field index
    results.sort(key=lambda x: x[0])

    # Save and validate
    os.makedirs(CACHE_DIR, exist_ok=True)
    field_ownerships = []

    print(f"\nValidation:")
    for field_idx, field_dk_ids, lineup_indices, perturbed_own in results:
        # Save .npy
        out_path = os.path.join(CACHE_DIR, f"field_{field_idx + 1}.npy")
        np.save(out_path, field_dk_ids)

        # Validate
        validate_field(field_dk_ids, lineup_indices, pool["salaries"], field_idx)

        # Compute realized ownership
        own_pct = compute_field_ownership(lineup_indices, n_players, FIELD_SIZE)
        field_ownerships.append(own_pct)

    # Save ownership summary
    summary_path = os.path.join(CACHE_DIR, "field_ownership_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "player_name", "dk_id",
            "field_1_ownership", "field_2_ownership", "field_3_ownership",
            "projected_ownership", "efficient_ownership",
        ])
        for i in range(n_players):
            writer.writerow([
                pool["names"][i],
                pool["dk_ids"][i],
                round(field_ownerships[0][i], 4),
                round(field_ownerships[1][i], 4),
                round(field_ownerships[2][i], 4),
                round(pool["ownership"][i], 4),
                "",  # efficient_ownership — filled post-sim
            ])

    # Print correlation and top-owned analysis
    print(f"\nOwnership correlation (field vs projected):")
    proj = pool["ownership"]
    for fi in range(3):
        r = np.corrcoef(proj, field_ownerships[fi])[0, 1]
        print(f"  Field {fi + 1}: Pearson r = {r:.4f}")

    print(f"\nTop 10 owned players per field:")
    for fi in range(3):
        own = field_ownerships[fi]
        top_idx = np.argsort(-own)[:10]
        print(f"\n  Field {fi + 1}:")
        print(f"  {'Player':<30s} {'Field Own':>10s} {'Proj Own':>10s}")
        print(f"  {'-'*30} {'-'*10} {'-'*10}")
        for idx in top_idx:
            print(f"  {pool['names'][idx]:<30s} {own[idx]:>9.2f}% {proj[idx]:>9.2f}%")

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Saved to {CACHE_DIR}/field_*.npy and field_ownership_summary.csv")

    return results


if __name__ == "__main__":
    generate_fields()

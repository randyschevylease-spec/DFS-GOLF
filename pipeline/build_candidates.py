"""
build_candidates.py — Bot army generating candidate lineups via 9 archetypes.

Generates 8,000+ valid DK golf lineups using distinct bot strategies:
  1. chalk_heavy     — weight by ownership^1.5 (800 lineups)
  2. ceiling_hunter  — weight by sim_p90 (1200 lineups)
  3. stars_scrubs    — 2 elite ($10k+) + 4 value (bottom 50%) (1000 lineups)
  4. balanced_mid    — all 6 from middle 60% by salary (800 lineups)
  5. contrarian      — weight by sim_p90 / (ownership + 0.01) (1000 lineups)
  6. cut_rate_focus  — weight by made_cut_rate * sim_mean (800 lineups)
  7. value_hunter    — weight by sim_mean / salary (1000 lineups)
  8. one_star_five   — 1 elite ($11k+) + 5 value (bottom 60%) (800 lineups)
  9. correlation_wave— same tee_time wave cluster (600 lineups)

Inputs:
  - sim_profiles_current.csv + dk_projections CSV (merged)

Output:
  - data/cache/candidates.csv
"""

import csv
import math
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "engine"))

SIM_PROFILES = os.path.join(PROJECT_ROOT, "data", "cache", "sim_profiles_current.csv")
DK_PROJECTIONS = os.path.join(PROJECT_ROOT, "data", "raw", "dk_projections_players.csv")
OUTPUT = os.path.join(PROJECT_ROOT, "data", "cache", "candidates.csv")

SALARY_CAP = 50_000
LINEUP_SIZE = 6


# ---------------------------------------------------------------------------
# Player pool loading
# ---------------------------------------------------------------------------

def load_player_pool():
    """
    Merge sim_profiles_current.csv + dk_projections to build player pool.

    Returns list of player dicts with: dk_id, dg_id, player_name, dk_name,
    salary, ownership, sim_mean, sim_p90, ceiling, made_cut_rate, tier,
    std_pts_made, early_late_wave, tee_time.
    """
    # Load sim profiles keyed by name
    sim_by_name = {}
    if os.path.exists(SIM_PROFILES):
        with open(SIM_PROFILES) as f:
            for row in csv.DictReader(f):
                sim_by_name[row["player_name"]] = row

    # Load DK projections keyed by name
    dk_by_name = {}
    with open(DK_PROJECTIONS) as f:
        for row in csv.DictReader(f):
            dk_by_name[row["datagolf_name"]] = row

    players = []
    for name, dk in dk_by_name.items():
        salary = int(dk["dk_salary"])
        if salary <= 0:
            continue

        sim = sim_by_name.get(name, {})

        total_pts = float(dk.get("total_points") or 0)
        std_dev = float(dk.get("std_dev") or 19)
        raw_own = float(dk.get("projected_ownership") or 5)

        avg_made = float(sim.get("avg_pts_made") or total_pts)
        avg_missed = float(sim.get("avg_pts_missed") or 28)
        cut_rate = float(sim.get("made_cut_rate") or 0.5)
        std_made = float(sim.get("std_pts_made") or std_dev)
        tier = int(sim.get("tier", 2))

        sim_mean = avg_made * cut_rate + avg_missed * (1 - cut_rate)
        sim_p90 = avg_made + 1.28 * std_made

        players.append({
            "dg_id": int(sim.get("dg_id", 0)),
            "dk_id": int(dk.get("dk_id", 0)),
            "player_name": name,
            "dk_name": dk["dk_name"],
            "salary": salary,
            "sim_mean": sim_mean,
            "sim_p90": sim_p90,
            "ceiling": total_pts + 1.5 * std_dev,
            "made_cut_rate": cut_rate,
            "std_pts_made": std_made,
            "tier": tier,
            "raw_ownership": raw_own,
            "ownership": 0.0,  # set by synthetic or real
            "early_late_wave": int(dk.get("early_late_wave", 0)),
            "tee_time": dk.get("tee_time", ""),
        })

    players.sort(key=lambda p: -p["salary"])

    # Check if ownership is uniform — if so, generate synthetic
    owns = [p["raw_ownership"] for p in players]
    if len(set(round(o, 2) for o in owns)) <= 2:
        print("  Ownership is uniform — generating synthetic ownership...")
        _apply_synthetic_ownership(players)
    else:
        for p in players:
            p["ownership"] = p["raw_ownership"] / 100.0

    return players


def _apply_synthetic_ownership(players):
    """
    Generate synthetic ownership when projected_ownership is uniform/missing.

    Uses weighted combination of sim_mean, salary, and tier to estimate
    what the DFS public would gravitate toward. Scaled so total ownership
    sums to ~600% (6 players per lineup × 100%).
    """
    sim_means = [p["sim_mean"] for p in players]
    salaries = [p["salary"] for p in players]

    mn_min, mn_max = min(sim_means), max(sim_means)
    sal_min, sal_max = min(salaries), max(salaries)

    mn_range = mn_max - mn_min if mn_max > mn_min else 1
    sal_range = sal_max - sal_min if sal_max > sal_min else 1

    raw_scores = []
    for p in players:
        norm_mean = (p["sim_mean"] - mn_min) / mn_range
        norm_sal = (p["salary"] - sal_min) / sal_range
        tier_bonus = 0.15 if p["tier"] == 1 else 0.0
        raw = 0.50 * norm_mean + 0.30 * norm_sal + 0.20 * tier_bonus
        raw_scores.append(max(raw, 0.01))

    # Scale so total sums to 600% (6 slots × 100%)
    total = sum(raw_scores)
    target_total = 6.0  # 600% as fraction
    for i, p in enumerate(players):
        p["ownership"] = raw_scores[i] / total * target_total

    # Print top/bottom for verification
    by_own = sorted(players, key=lambda p: -p["ownership"])
    print(f"  Synthetic ownership range: "
          f"{by_own[-1]['ownership']:.1%} - {by_own[0]['ownership']:.1%}")
    print(f"  Total ownership: {sum(p['ownership'] for p in players):.0%}")


# ---------------------------------------------------------------------------
# Core lineup generation
# ---------------------------------------------------------------------------

def weighted_sample(players, rng, weights, seen):
    """
    Generate one valid lineup via weighted sampling without replacement.

    Returns lineup (list of player dicts) or None if failed after 1 attempt.
    """
    total_w = sum(weights)
    if total_w <= 0:
        return None
    probs = [w / total_w for w in weights]

    lineup = []
    remaining_idx = list(range(len(players)))
    remaining_probs = list(probs)

    for _ in range(LINEUP_SIZE):
        if not remaining_idx:
            return None
        tw = sum(remaining_probs)
        if tw <= 0:
            return None
        norm = [w / tw for w in remaining_probs]

        r = rng.random()
        cumulative = 0
        chosen = 0
        for j, w in enumerate(norm):
            cumulative += w
            if r < cumulative:
                chosen = j
                break

        lineup.append(players[remaining_idx[chosen]])
        remaining_idx.pop(chosen)
        remaining_probs.pop(chosen)

    if len(lineup) != LINEUP_SIZE:
        return None
    if sum(p["salary"] for p in lineup) > SALARY_CAP:
        return None

    key = tuple(sorted(p["dk_id"] for p in lineup))
    if key in seen:
        return None

    return lineup


def generate_weighted(players, n_target, rng, weights, seen, label):
    """Generate lineups with given weights. Returns (lineups, strategy_label)."""
    lineups = []
    attempts = 0
    max_attempts = n_target * 25

    while len(lineups) < n_target and attempts < max_attempts:
        attempts += 1
        lu = weighted_sample(players, rng, weights, seen)
        if lu is not None:
            key = tuple(sorted(p["dk_id"] for p in lu))
            seen.add(key)
            lineups.append(lu)

    return lineups


def generate_pool_sample(pool, n_target, rng, seen, label):
    """Generate lineups by uniform sampling from a filtered pool."""
    lineups = []
    attempts = 0
    max_attempts = n_target * 30

    while len(lineups) < n_target and attempts < max_attempts:
        attempts += 1
        if len(pool) < LINEUP_SIZE:
            break
        lu = rng.sample(pool, LINEUP_SIZE)
        if sum(p["salary"] for p in lu) > SALARY_CAP:
            continue
        key = tuple(sorted(p["dk_id"] for p in lu))
        if key in seen:
            continue
        seen.add(key)
        lineups.append(lu)

    return lineups


def generate_stars_plus_value(players, n_target, rng, seen,
                              n_stars, star_min_salary, value_top_pct):
    """
    Generate lineups with n_stars elite + (6 - n_stars) value plays.

    Stars: players with salary >= star_min_salary
    Value: bottom value_top_pct% by salary
    """
    sorted_sal = sorted(players, key=lambda p: -p["salary"])
    stars_pool = [p for p in sorted_sal if p["salary"] >= star_min_salary]
    cutoff = int(len(sorted_sal) * (1 - value_top_pct))
    value_pool = sorted_sal[cutoff:]

    if len(stars_pool) < n_stars or len(value_pool) < LINEUP_SIZE - n_stars:
        return []

    lineups = []
    attempts = 0
    max_attempts = n_target * 30

    while len(lineups) < n_target and attempts < max_attempts:
        attempts += 1
        stars = rng.sample(stars_pool, n_stars)
        star_ids = {p["dk_id"] for p in stars}
        remaining = [p for p in value_pool if p["dk_id"] not in star_ids]
        if len(remaining) < LINEUP_SIZE - n_stars:
            continue
        value = rng.sample(remaining, LINEUP_SIZE - n_stars)
        lu = stars + value

        if sum(p["salary"] for p in lu) > SALARY_CAP:
            continue
        key = tuple(sorted(p["dk_id"] for p in lu))
        if key in seen:
            continue
        seen.add(key)
        lineups.append(lu)

    return lineups


def generate_wave_correlated(players, n_target, rng, seen):
    """
    Generate lineups where all 6 players share the same tee-time wave.

    Exploits weather correlation — if early wave gets calm conditions,
    all 6 benefit simultaneously.
    """
    wave_pools = {}
    for p in players:
        w = p["early_late_wave"]
        if w not in wave_pools:
            wave_pools[w] = []
        wave_pools[w].append(p)

    # Remove waves with fewer than 6 players
    wave_pools = {w: pl for w, pl in wave_pools.items() if len(pl) >= LINEUP_SIZE}

    if not wave_pools:
        return []

    lineups = []
    attempts = 0
    max_attempts = n_target * 30
    wave_keys = list(wave_pools.keys())

    while len(lineups) < n_target and attempts < max_attempts:
        attempts += 1
        wave = rng.choice(wave_keys)
        pool = wave_pools[wave]
        lu = rng.sample(pool, LINEUP_SIZE)

        if sum(p["salary"] for p in lu) > SALARY_CAP:
            continue
        key = tuple(sorted(p["dk_id"] for p in lu))
        if key in seen:
            continue
        seen.add(key)
        lineups.append(lu)

    return lineups


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

BOT_CONFIG = [
    ("chalk_heavy",      800_000),
    ("ceiling_hunter",  1_200_000),
    ("stars_scrubs",    1_000_000),
    ("balanced_mid",      800_000),
    ("contrarian",      1_000_000),
    ("cut_rate_focus",    800_000),
    ("value_hunter",    1_000_000),
    ("one_star_five",     800_000),
    ("wave_correlated",   600_000),
]


def build_candidates(seed=42):
    """
    Build full candidate pool using 9 bot archetypes.

    Returns list of (lineup, strategy) tuples.
    """
    players = load_player_pool()
    rng = random.Random(seed)
    n_players = len(players)

    print(f"\nPlayer pool: {n_players} players")
    print(f"  Salary range: ${min(p['salary'] for p in players):,} "
          f"- ${max(p['salary'] for p in players):,}")
    print(f"  Ownership range: {min(p['ownership'] for p in players):.1%} "
          f"- {max(p['ownership'] for p in players):.1%}")

    seen = set()
    all_results = []  # list of (lineup, strategy_name)

    for bot_name, n_target in BOT_CONFIG:
        lineups = _run_bot(bot_name, players, n_target, rng, seen)
        for lu in lineups:
            all_results.append((lu, bot_name))
        print(f"  {bot_name:<20s}: {len(lineups):>5,} / {n_target:>5,} target")

    print(f"\nTotal unique candidates: {len(all_results):,}")

    # Save
    _save_candidates(all_results, players)

    # Dupe summary
    from payout import MILLY_MAKER_FIELD
    _print_dupe_summary(all_results, MILLY_MAKER_FIELD)

    # Strategy breakdown
    _print_strategy_summary(all_results)

    return all_results


def _run_bot(name, players, n_target, rng, seen):
    """Dispatch to the appropriate bot archetype."""

    if name == "chalk_heavy":
        weights = [p["ownership"] ** 1.5 for p in players]
        return generate_weighted(players, n_target, rng, weights, seen, name)

    elif name == "ceiling_hunter":
        weights = [p["sim_p90"] for p in players]
        return generate_weighted(players, n_target, rng, weights, seen, name)

    elif name == "stars_scrubs":
        return generate_stars_plus_value(
            players, n_target, rng, seen,
            n_stars=2, star_min_salary=10_000, value_top_pct=0.50)

    elif name == "balanced_mid":
        sorted_sal = sorted(players, key=lambda p: -p["salary"])
        n = len(sorted_sal)
        mid_pool = sorted_sal[n // 5: 4 * n // 5]
        return generate_pool_sample(mid_pool, n_target, rng, seen, name)

    elif name == "contrarian":
        weights = [p["sim_p90"] / (p["ownership"] + 0.01) for p in players]
        return generate_weighted(players, n_target, rng, weights, seen, name)

    elif name == "cut_rate_focus":
        weights = [p["made_cut_rate"] * p["sim_mean"] for p in players]
        return generate_weighted(players, n_target, rng, weights, seen, name)

    elif name == "value_hunter":
        weights = [p["sim_mean"] / p["salary"] * 10000 for p in players]
        return generate_weighted(players, n_target, rng, weights, seen, name)

    elif name == "one_star_five":
        return generate_stars_plus_value(
            players, n_target, rng, seen,
            n_stars=1, star_min_salary=11_000, value_top_pct=0.60)

    elif name == "wave_correlated":
        return generate_wave_correlated(players, n_target, rng, seen)

    else:
        raise ValueError(f"Unknown bot: {name}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _save_candidates(all_results, players):
    """Save candidates to CSV."""
    from payout import MILLY_MAKER_FIELD
    field_size = MILLY_MAKER_FIELD

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lineup_id", "strategy",
            "p1_id", "p1_name", "p1_salary", "p1_own",
            "p2_id", "p2_name", "p2_salary", "p2_own",
            "p3_id", "p3_name", "p3_salary", "p3_own",
            "p4_id", "p4_name", "p4_salary", "p4_own",
            "p5_id", "p5_name", "p5_salary", "p5_own",
            "p6_id", "p6_name", "p6_salary", "p6_own",
            "total_salary", "avg_ceiling", "avg_sim_mean",
            "ownership_sum", "ownership_product", "expected_dupes",
        ])

        for i, (lineup, strategy) in enumerate(all_results):
            row = [i, strategy]
            lineup_sorted = sorted(lineup, key=lambda p: -p["salary"])
            own_product = 1.0
            own_sum = 0.0
            total_ceiling = 0.0
            total_mean = 0.0

            for p in lineup_sorted:
                own = p["ownership"]
                row.extend([p["dk_id"], p["player_name"], p["salary"],
                            round(own, 6)])
                own_product *= own
                own_sum += own
                total_ceiling += p["ceiling"]
                total_mean += p["sim_mean"]

            total_salary = sum(p["salary"] for p in lineup)
            expected_dupes = own_product * field_size

            row.append(total_salary)
            row.append(round(total_ceiling / LINEUP_SIZE, 1))
            row.append(round(total_mean / LINEUP_SIZE, 1))
            row.append(round(own_sum, 4))
            row.append(f"{own_product:.12e}")
            row.append(round(expected_dupes, 6))
            writer.writerow(row)

    print(f"\nSaved to {OUTPUT}")


def _print_dupe_summary(all_results, field_size):
    """Print dupe estimate distribution."""
    dupes = []
    for lineup, _ in all_results:
        prod = 1.0
        for p in lineup:
            prod *= p["ownership"]
        dupes.append(prod * field_size)

    dupes.sort(reverse=True)
    print(f"\nDupe estimate summary (field={field_size:,}):")
    print(f"  Max expected dupes:    {dupes[0]:.4f}")
    print(f"  Median expected dupes: {dupes[len(dupes)//2]:.6f}")
    print(f"  Min expected dupes:    {dupes[-1]:.8f}")
    print(f"  Lineups >0.1 dupes:   {sum(1 for d in dupes if d > 0.1)}")
    print(f"  Lineups >1.0 dupes:   {sum(1 for d in dupes if d > 1.0)}")


def _print_strategy_summary(all_results):
    """Print per-strategy salary and projection stats."""
    from collections import defaultdict
    stats = defaultdict(lambda: {"count": 0, "salary": [], "mean": [], "p90": [], "own": []})

    for lineup, strategy in all_results:
        s = stats[strategy]
        s["count"] += 1
        s["salary"].append(sum(p["salary"] for p in lineup))
        s["mean"].append(sum(p["sim_mean"] for p in lineup) / LINEUP_SIZE)
        s["own"].append(sum(p["ownership"] for p in lineup))

    print(f"\nStrategy summary:")
    print(f"  {'Bot':<20s} {'Count':>6s} {'Avg Sal':>9s} {'Avg Mean':>9s} {'Avg Own%':>9s}")
    print(f"  {'-'*20} {'-'*6} {'-'*9} {'-'*9} {'-'*9}")

    for bot_name, _ in BOT_CONFIG:
        s = stats[bot_name]
        if s["count"] == 0:
            continue
        avg_sal = sum(s["salary"]) / s["count"]
        avg_mean = sum(s["mean"]) / s["count"]
        avg_own = sum(s["own"]) / s["count"]
        print(f"  {bot_name:<20s} {s['count']:>6,} "
              f"${avg_sal:>7,.0f} {avg_mean:>9.1f} {avg_own:>8.1%}")


if __name__ == "__main__":
    build_candidates()

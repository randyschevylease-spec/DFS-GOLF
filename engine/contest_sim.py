"""
contest_sim.py -- Score 100k candidates against 3 synthetic fields.

Pipeline:
  1. Load candidates (100k lineups of dk_ids)
  2. Load player pool, build dk_id <-> dg_id mapping
  3. Run player_sim.simulate_tournament() for all unique players
  4. Score all candidates across iterations (vectorized)
  5. Score each field against candidates per iteration (chunked, low memory)
  6. Compute per-candidate metrics (EV, cash_rate, rank distributions)
  7. Save results to contest_sim_results.csv
"""

import csv
import math
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "engine"))
sys.stdout.reconfigure(encoding='utf-8')

from payout import get_payout, MILLY_MAKER_PAYOUTS, MILLY_MAKER_ENTRY, MILLY_MAKER_FIELD
from player_sim import simulate_tournament

CANDIDATES_PATH = os.path.join(PROJECT_ROOT, "data", "cache", "candidates_filtered.csv")
FIELD_DIR = os.path.join(PROJECT_ROOT, "data", "cache")
DK_PROJECTIONS = os.path.join(PROJECT_ROOT, "data", "raw", "dk_projections_players.csv")
SIM_PROFILES = os.path.join(PROJECT_ROOT, "data", "cache", "sim_profiles_current.csv")
OUTPUT = os.path.join(PROJECT_ROOT, "data", "cache", "contest_sim_results.csv")

N_ITERATIONS = 2000
FIELD_CHUNK = 2000
SALARY_CAP = 50_000
ENTRY_FEE = MILLY_MAKER_ENTRY
N_FIELDS = 3


# ---------------------------------------------------------------------------
# Step 1: Load candidates
# ---------------------------------------------------------------------------

def load_candidates():
    """Load candidates_filtered.csv. Returns (rows, dk_id_matrix, all_unique_dk_ids)."""
    rows = []
    dk_id_lists = []
    with open(CANDIDATES_PATH) as f:
        for row in csv.DictReader(f):
            ids = [int(row[f"p{i}_id"]) for i in range(1, 7)]
            dk_id_lists.append(ids)
            rows.append(row)

    dk_id_matrix = np.array(dk_id_lists, dtype=np.int64)  # (n_cand, 6)
    all_dk_ids = sorted(set(dk_id_matrix.ravel()))
    return rows, dk_id_matrix, all_dk_ids


# ---------------------------------------------------------------------------
# Step 2: Load player pool and build mappings
# ---------------------------------------------------------------------------

def load_player_mappings():
    """Build dk_id <-> dg_id and dk_id -> name mappings."""
    dk_by_name = {}
    with open(DK_PROJECTIONS) as f:
        for row in csv.DictReader(f):
            dk_by_name[row["datagolf_name"]] = {
                "dk_id": int(row["dk_id"]),
            }

    sim_by_name = {}
    with open(SIM_PROFILES) as f:
        for row in csv.DictReader(f):
            sim_by_name[row["player_name"]] = {
                "dg_id": int(row["dg_id"]),
                "dg_make_cut": float(row.get("dg_make_cut") or 0.5),
            }

    dk_to_dg = {}
    dg_to_dk = {}
    dk_to_name = {}
    dg_cut_probs = {}

    for name in dk_by_name:
        dk = dk_by_name[name]
        sim = sim_by_name.get(name)
        if sim is None:
            continue
        dk_id = dk["dk_id"]
        dg_id = sim["dg_id"]
        dk_to_dg[dk_id] = dg_id
        dg_to_dk[dg_id] = dk_id
        dk_to_name[dk_id] = name
        dg_cut_probs[dg_id] = sim["dg_make_cut"]

    return dk_to_dg, dg_to_dk, dk_to_name, dg_cut_probs


# ---------------------------------------------------------------------------
# Step 3: Run player sim
# ---------------------------------------------------------------------------

def run_player_sim(unique_dk_ids, dk_to_dg, dg_cut_probs):
    """Run simulate_tournament for all unique players. Returns dk_id -> score array."""
    dg_ids = []
    dk_for_dg = {}
    for dk_id in unique_dk_ids:
        dg_id = dk_to_dg.get(dk_id)
        if dg_id is not None:
            dg_ids.append(dg_id)
            dk_for_dg[dg_id] = dk_id

    print(f"  Running sim for {len(dg_ids)} players x {N_ITERATIONS} iterations...")
    t = time.perf_counter()
    results = simulate_tournament(
        dg_ids,
        n_iterations=N_ITERATIONS,
        cut_size=65,
        seed=42,
        dg_cut_probs=dg_cut_probs,
        course_avg_score=72.38,
    )
    print(f"  Sim complete in {time.perf_counter() - t:.1f}s")

    # Map dg_id results -> dk_id arrays
    dk_scores = {}
    for dg_id, pts_list in results.items():
        dk_id = dk_for_dg.get(dg_id)
        if dk_id is not None:
            dk_scores[dk_id] = np.array(pts_list, dtype=np.float32)

    return dk_scores


# ---------------------------------------------------------------------------
# Step 4: Score all candidates (vectorized)
# ---------------------------------------------------------------------------

def _build_dk_id_lookup(unique_dk_ids):
    """Build a numpy-based dk_id -> index lookup using searchsorted."""
    sorted_ids = np.array(sorted(unique_dk_ids), dtype=np.int64)
    return sorted_ids


def _map_ids_to_indices(id_array, sorted_ids):
    """Map an array of dk_ids to indices into sorted_ids. Fully vectorized."""
    flat = id_array.ravel()
    indices = np.searchsorted(sorted_ids, flat)
    # Clamp out-of-range (shouldn't happen with valid data)
    indices = np.clip(indices, 0, len(sorted_ids) - 1)
    return indices.reshape(id_array.shape).astype(np.int32)


def score_candidates(dk_id_matrix, dk_scores, unique_dk_ids):
    """
    Build candidate_score_matrix (n_cand, N_ITERATIONS) float32.
    Uses vectorized numpy index mapping throughout.
    """
    # Sorted dk_ids for searchsorted-based lookup
    sorted_ids = _build_dk_id_lookup(unique_dk_ids)
    dk_id_to_idx = {int(dk_id): i for i, dk_id in enumerate(sorted_ids)}
    n_players = len(sorted_ids)

    # Build player score matrix (n_players, N_ITERATIONS)
    player_matrix = np.zeros((n_players, N_ITERATIONS), dtype=np.float32)
    for dk_id in unique_dk_ids:
        idx = dk_id_to_idx[dk_id]
        if dk_id in dk_scores:
            arr = dk_scores[dk_id]
            player_matrix[idx, :len(arr)] = arr[:N_ITERATIONS]

    # Map candidate dk_ids to player indices (vectorized)
    cand_player_idx = _map_ids_to_indices(dk_id_matrix, sorted_ids)

    # Vectorized sum: candidate_scores[c, iter] = sum of 6 player scores
    candidate_scores = np.zeros((dk_id_matrix.shape[0], N_ITERATIONS), dtype=np.float32)
    for j in range(6):
        candidate_scores += player_matrix[cand_player_idx[:, j]]

    return candidate_scores, player_matrix, sorted_ids, dk_id_to_idx


# ---------------------------------------------------------------------------
# Step 5: Rank candidates against field (chunked, per iteration)
# ---------------------------------------------------------------------------

def build_payout_table():
    """Build vectorized payout lookup: rank -> payout amount."""
    # Max rank we care about is MILLY_MAKER_FIELD + 1
    max_rank = MILLY_MAKER_FIELD + 2
    table = np.zeros(max_rank, dtype=np.float32)
    for start, end, payout in MILLY_MAKER_PAYOUTS:
        table[start:end + 1] = payout
    return table


def rank_candidates_against_field(candidate_scores, field_path,
                                  player_matrix, sorted_ids,
                                  payout_table_arr, field_label):
    """
    For each iteration, score the full field and rank each candidate.

    Scores all 105k field lineups per iteration in one vectorized op,
    then uses a single searchsorted to rank all 100k candidates.
    Memory: ~420KB per iteration (105k float32) — well under budget.

    Returns:
        ranks: (n_cand, N_ITERATIONS) int32
        payouts: (n_cand, N_ITERATIONS) float32
    """
    field = np.load(field_path)  # (105800, 6) int64
    n_field = field.shape[0]
    n_cand = candidate_scores.shape[0]
    n_iters = candidate_scores.shape[1]

    # Map field dk_ids to player indices (vectorized, one-time)
    print(f"    Mapping {n_field:,} x 6 field dk_ids to player indices...")
    field_player_idx = _map_ids_to_indices(field, sorted_ids)
    del field

    ranks = np.zeros((n_cand, n_iters), dtype=np.int32)
    payouts = np.zeros((n_cand, n_iters), dtype=np.float32)

    report_interval = max(1, n_iters // 10)

    for it in range(n_iters):
        # Score all field lineups for this iteration (vectorized)
        field_scores = np.zeros(n_field, dtype=np.float32)
        for j in range(6):
            field_scores += player_matrix[field_player_idx[:, j], it]

        # Sort field scores ascending for searchsorted
        field_sorted = np.sort(field_scores)

        # Rank each candidate: count field lineups scoring >= candidate
        cand_scores_it = candidate_scores[:, it]
        # searchsorted(left) gives index of first element >= candidate
        insert_pos = np.searchsorted(field_sorted, cand_scores_it, side='left')
        # Number beating candidate = n_field - insert_pos
        iter_ranks = (n_field - insert_pos + 1).astype(np.int32)
        iter_ranks = np.clip(iter_ranks, 1, n_field + 1)
        ranks[:, it] = iter_ranks

        # Lookup payouts
        clipped_ranks = np.clip(iter_ranks, 0, len(payout_table_arr) - 1)
        payouts[:, it] = payout_table_arr[clipped_ranks]

        if (it + 1) % report_interval == 0 or it == 0:
            pct = (it + 1) / n_iters * 100
            med_rank = int(np.median(iter_ranks))
            cashing = int(np.sum(payouts[:, it] > 0))
            print(f"    {field_label} iter {it+1}/{n_iters} ({pct:.0f}%) "
                  f"-- median rank {med_rank:,}, {cashing:,} cashing")

    return ranks, payouts


# ---------------------------------------------------------------------------
# Step 6: Compute metrics
# ---------------------------------------------------------------------------

def compute_metrics(candidate_scores, all_ranks, all_payouts):
    """
    Compute per-candidate metrics across all iterations x 3 fields.

    Memory-efficient: computes per-field stats without full concatenation.

    all_ranks: list of 3 arrays, each (n_cand, N_ITERATIONS)
    all_payouts: list of 3 arrays, each (n_cand, N_ITERATIONS)
    """
    n_cand = candidate_scores.shape[0]
    n_fields = len(all_payouts)
    n_iters = all_payouts[0].shape[1]
    n_total = n_iters * n_fields

    # Accumulate stats per field to avoid concatenation
    min_cash = 40.0
    payout_sum = np.zeros(n_cand, dtype=np.float64)
    cash_count = np.zeros(n_cand, dtype=np.int32)
    top1000_count = np.zeros(n_cand, dtype=np.int32)
    top100_count = np.zeros(n_cand, dtype=np.int32)

    field_evs = []
    for fi in range(n_fields):
        p = all_payouts[fi]
        r = all_ranks[fi]
        field_mean_payout = p.mean(axis=1)
        field_evs.append(field_mean_payout - ENTRY_FEE)

        payout_sum += p.sum(axis=1).astype(np.float64)
        cash_count += (p >= min_cash).sum(axis=1).astype(np.int32)
        top1000_count += (r <= 1000).sum(axis=1).astype(np.int32)
        top100_count += (r <= 100).sum(axis=1).astype(np.int32)

    mean_payout = (payout_sum / n_total).astype(np.float32)
    mean_ev = mean_payout - ENTRY_FEE

    sim_p90 = np.percentile(candidate_scores, 90, axis=1)

    # Stack just the 3 per-field EV scalars per candidate (tiny: n_cand x 3)
    cross_field_stability = np.std(
        np.column_stack(field_evs), axis=1).astype(np.float32)

    return {
        "mean_ev": mean_ev,
        "mean_payout": mean_payout,
        "cash_rate": (cash_count / n_total).astype(np.float32),
        "top1000_rate": (top1000_count / n_total).astype(np.float32),
        "top100_rate": (top100_count / n_total).astype(np.float32),
        "sim_p90_score": sim_p90.astype(np.float32),
        "field_1_mean_ev": field_evs[0].astype(np.float32),
        "field_2_mean_ev": field_evs[1].astype(np.float32),
        "field_3_mean_ev": field_evs[2].astype(np.float32),
        "cross_field_stability": cross_field_stability,
    }


# ---------------------------------------------------------------------------
# Step 7: Output
# ---------------------------------------------------------------------------

def save_results(rows, metrics):
    """Merge metrics into candidate rows and save."""
    metric_cols = [
        "mean_ev", "mean_payout", "cash_rate", "top1000_rate", "top100_rate",
        "sim_p90_score", "field_1_mean_ev", "field_2_mean_ev", "field_3_mean_ev",
        "cross_field_stability",
    ]

    # Read original header
    with open(CANDIDATES_PATH) as f:
        reader = csv.DictReader(f)
        orig_fields = list(reader.fieldnames)

    all_fields = orig_fields + metric_cols

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()

        for i, row in enumerate(rows):
            for col in metric_cols:
                row[col] = round(float(metrics[col][i]), 6)
            writer.writerow({k: row.get(k, "") for k in all_fields})

    print(f"  Saved {len(rows):,} rows to {OUTPUT}")


# ---------------------------------------------------------------------------
# Validation / reporting
# ---------------------------------------------------------------------------

def print_validation(rows, metrics, candidate_scores):
    n = len(rows)

    # Mean EV distribution
    ev = metrics["mean_ev"]
    ev_sorted = np.sort(ev)
    print(f"\n  mean_ev distribution:")
    print(f"    min={ev_sorted[0]:.2f}  p25={ev_sorted[n//4]:.2f}  "
          f"median={ev_sorted[n//2]:.2f}  p75={ev_sorted[3*n//4]:.2f}  "
          f"max={ev_sorted[-1]:.2f}")

    # Cash rate distribution
    cr = metrics["cash_rate"]
    cr_sorted = np.sort(cr)
    print(f"\n  cash_rate distribution:")
    print(f"    min={cr_sorted[0]:.4f}  p25={cr_sorted[n//4]:.4f}  "
          f"median={cr_sorted[n//2]:.4f}  p75={cr_sorted[3*n//4]:.4f}  "
          f"max={cr_sorted[-1]:.4f}")

    # Candidate score stats
    print(f"\n  Candidate scores across all iterations:")
    print(f"    mean={candidate_scores.mean():.1f}  "
          f"std={candidate_scores.std():.1f}  "
          f"p90={np.percentile(candidate_scores, 90):.1f}")

    # Top 20 by mean_ev
    top_idx = np.argsort(-ev)[:20]
    print(f"\n  Top 20 lineups by mean_ev:")
    print(f"  {'Rank':>4s}  {'LineupID':>8s}  {'Bot':<20s}  "
          f"{'mean_ev':>8s}  {'cash%':>6s}  {'stability':>9s}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*20}  {'-'*8}  {'-'*6}  {'-'*9}")
    for rank, idx in enumerate(top_idx, 1):
        row = rows[idx]
        print(f"  {rank:>4d}  {row['lineup_id']:>8s}  {row['strategy']:<20s}  "
              f"${ev[idx]:>7.2f}  {cr[idx]:>5.1%}  "
              f"{metrics['cross_field_stability'][idx]:>9.3f}")

    # Field-sensitive outliers
    stab = metrics["cross_field_stability"]
    med_stab = np.median(stab)
    outliers = np.sum(stab > 2 * med_stab)
    print(f"\n  Cross-field stability: median={med_stab:.3f}")
    print(f"  Field-sensitive outliers (>2x median): {outliers:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_contest_sim():
    timers = {}
    t_total = time.perf_counter()

    # Step 1
    print("Step 1: Loading candidates...")
    t = time.perf_counter()
    rows, dk_id_matrix, all_dk_ids = load_candidates()
    n_cand = len(rows)
    timers["load_candidates"] = time.perf_counter() - t
    print(f"  {n_cand:,} candidates, {len(all_dk_ids)} unique players")

    # Step 2
    print("\nStep 2: Loading player mappings...")
    t = time.perf_counter()
    dk_to_dg, dg_to_dk, dk_to_name, dg_cut_probs = load_player_mappings()
    timers["load_mappings"] = time.perf_counter() - t
    print(f"  {len(dk_to_dg)} dk->dg mappings")

    # Step 3
    print(f"\nStep 3: Running player sim ({N_ITERATIONS} iterations)...")
    t = time.perf_counter()
    dk_scores = run_player_sim(all_dk_ids, dk_to_dg, dg_cut_probs)
    timers["player_sim"] = time.perf_counter() - t
    print(f"  {len(dk_scores)} players simulated")

    # Step 4
    print("\nStep 4: Scoring candidates (vectorized)...")
    t = time.perf_counter()
    candidate_scores, player_matrix, sorted_ids, dk_id_to_idx = score_candidates(
        dk_id_matrix, dk_scores, all_dk_ids)
    timers["score_candidates"] = time.perf_counter() - t
    print(f"  candidate_scores shape: {candidate_scores.shape}")
    print(f"  Mean score: {candidate_scores.mean():.1f}, "
          f"Std: {candidate_scores.std():.1f}, "
          f"P90: {np.percentile(candidate_scores, 90):.1f}")

    # Step 5
    print(f"\nStep 5: Ranking against {N_FIELDS} fields...")
    payout_table_arr = build_payout_table()
    all_ranks = []
    all_payouts = []

    for fi in range(1, N_FIELDS + 1):
        field_path = os.path.join(FIELD_DIR, f"field_{fi}.npy")
        print(f"\n  --- Field {fi} ---")
        t = time.perf_counter()
        ranks, payouts = rank_candidates_against_field(
            candidate_scores, field_path,
            player_matrix, sorted_ids,
            payout_table_arr, f"Field {fi}")
        elapsed = time.perf_counter() - t
        timers[f"field_{fi}"] = elapsed
        all_ranks.append(ranks)
        all_payouts.append(payouts)
        print(f"    Field {fi} done in {elapsed:.1f}s")

    # Step 6
    print("\nStep 6: Computing metrics...")
    t = time.perf_counter()
    metrics = compute_metrics(candidate_scores, all_ranks, all_payouts)
    timers["metrics"] = time.perf_counter() - t

    # Step 7
    print("\nStep 7: Saving results...")
    t = time.perf_counter()
    save_results(rows, metrics)
    timers["save"] = time.perf_counter() - t

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    print_validation(rows, metrics, candidate_scores)

    # Timing summary
    total = time.perf_counter() - t_total
    print(f"\n{'=' * 60}")
    print(f"TIMING SUMMARY")
    print(f"{'=' * 60}")
    for step, elapsed in timers.items():
        print(f"  {step:<20s}: {elapsed:>8.1f}s")
    print(f"  {'TOTAL':<20s}: {total:>8.1f}s")


if __name__ == "__main__":
    run_contest_sim()

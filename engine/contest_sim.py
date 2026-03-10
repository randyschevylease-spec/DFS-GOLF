"""
contest_sim.py -- Score 100k candidates against 3 synthetic fields.

Pipeline:
  1. Load candidates (100k lineups of dk_ids)
  2. Load player pool, build dk_id <-> dg_id mapping
  3. Run player_sim.simulate_tournament() for all unique players
  4. Score all candidates across iterations (vectorized)
  5. Score each field against candidates per iteration (chunked, low memory)
  6. Compute per-candidate metrics (EV, cash_rate, rank distributions)
  6b. Apply dupe adjustment, near-dupe penalty, wave correlation bonus
  7. Save results to contest_sim_results.csv
"""

import csv
import math
import os
import sys
import time
from collections import Counter

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "engine"))
sys.stdout.reconfigure(encoding='utf-8')

from payout import get_payout, MILLY_MAKER_PAYOUTS, MILLY_MAKER_ENTRY, MILLY_MAKER_FIELD
from player_sim import simulate_tournament

CANDIDATES_PATH = os.path.join(PROJECT_ROOT, "data", "cache", "candidates_filtered.csv")
FIELD_DIR = os.path.join(PROJECT_ROOT, "data", "cache")
DK_PROJECTIONS = os.path.join(PROJECT_ROOT, "data", "raw", "draftkings_main_projections__3_.csv")
SIM_PROFILES = os.path.join(PROJECT_ROOT, "data", "cache", "sim_profiles_current.csv")
OUTPUT = os.path.join(PROJECT_ROOT, "data", "cache", "contest_sim_results.csv")

N_ITERATIONS = 10000
FIELD_CHUNK = 2000
SALARY_CAP = 50_000
ENTRY_FEE = MILLY_MAKER_ENTRY
N_FIELDS = 3
PAYOUT_CAP = 2500.0
TOP_K_PAYOUTS = 20000
PAYOUT_ARRAYS_PATH = os.path.join(PROJECT_ROOT, "data", "cache", "payout_arrays_top20k.npz")


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


def load_wave_data():
    """Load early_late_wave per dk_id from dk_projections."""
    dk_wave = {}
    with open(DK_PROJECTIONS) as f:
        for row in csv.DictReader(f):
            dk_wave[int(row["dk_id"])] = int(row["early_late_wave"])
    return dk_wave


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
    indices = np.clip(indices, 0, len(sorted_ids) - 1)
    return indices.reshape(id_array.shape).astype(np.int32)


def score_candidates(dk_id_matrix, dk_scores, unique_dk_ids):
    """
    Build candidate_score_matrix (n_cand, N_ITERATIONS) float32.
    Uses vectorized numpy index mapping throughout.
    """
    sorted_ids = _build_dk_id_lookup(unique_dk_ids)
    dk_id_to_idx = {int(dk_id): i for i, dk_id in enumerate(sorted_ids)}
    n_players = len(sorted_ids)

    player_matrix = np.zeros((n_players, N_ITERATIONS), dtype=np.float32)
    for dk_id in unique_dk_ids:
        idx = dk_id_to_idx[dk_id]
        if dk_id in dk_scores:
            arr = dk_scores[dk_id]
            player_matrix[idx, :len(arr)] = arr[:N_ITERATIONS]

    cand_player_idx = _map_ids_to_indices(dk_id_matrix, sorted_ids)

    candidate_scores = np.zeros((dk_id_matrix.shape[0], N_ITERATIONS), dtype=np.float32)
    for j in range(6):
        candidate_scores += player_matrix[cand_player_idx[:, j]]

    return candidate_scores, player_matrix, sorted_ids, dk_id_to_idx


# ---------------------------------------------------------------------------
# Step 5: Rank candidates against field (chunked, per iteration)
# ---------------------------------------------------------------------------

def build_payout_table():
    """Build vectorized payout lookup: rank -> payout amount."""
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

    Memory-efficient: accumulates stats per iteration instead of storing
    full (n_cand, n_iters) rank/payout matrices.

    Returns:
        dict of accumulated per-field stats (n_cand,) arrays
    """
    field = np.load(field_path)  # (105800, 6) int64
    n_field = field.shape[0]
    n_cand = candidate_scores.shape[0]
    n_iters = candidate_scores.shape[1]

    print(f"    Mapping {n_field:,} x 6 field dk_ids to player indices...")
    field_player_idx = _map_ids_to_indices(field, sorted_ids)
    del field

    min_cash = 40.0
    payout_sum = np.zeros(n_cand, dtype=np.float64)
    cash_count = np.zeros(n_cand, dtype=np.int32)
    top1000_count = np.zeros(n_cand, dtype=np.int32)
    top100_count = np.zeros(n_cand, dtype=np.int32)

    report_interval = max(1, n_iters // 10)

    for it in range(n_iters):
        field_scores = np.zeros(n_field, dtype=np.float32)
        for j in range(6):
            field_scores += player_matrix[field_player_idx[:, j], it]

        field_sorted = np.sort(field_scores)

        cand_scores_it = candidate_scores[:, it]
        insert_pos = np.searchsorted(field_sorted, cand_scores_it, side='left')
        iter_ranks = (n_field - insert_pos + 1).astype(np.int32)
        iter_ranks = np.clip(iter_ranks, 1, n_field + 1)

        clipped_ranks = np.clip(iter_ranks, 0, len(payout_table_arr) - 1)
        iter_payouts = payout_table_arr[clipped_ranks]
        iter_payouts = np.minimum(iter_payouts, PAYOUT_CAP)

        payout_sum += iter_payouts.astype(np.float64)
        cash_count += (iter_payouts >= min_cash).astype(np.int32)
        top1000_count += (iter_ranks <= 1000).astype(np.int32)
        top100_count += (iter_ranks <= 100).astype(np.int32)

        if (it + 1) % report_interval == 0 or it == 0:
            pct = (it + 1) / n_iters * 100
            med_rank = int(np.median(iter_ranks))
            cashing = int(np.sum(iter_payouts > 0))
            print(f"    {field_label} iter {it+1}/{n_iters} ({pct:.0f}%) "
                  f"-- median rank {med_rank:,}, {cashing:,} cashing")

    field_mean_ev = (payout_sum / n_iters).astype(np.float32) - ENTRY_FEE

    return {
        "payout_sum": payout_sum,
        "cash_count": cash_count,
        "top1000_count": top1000_count,
        "top100_count": top100_count,
        "field_mean_ev": field_mean_ev,
    }


# ---------------------------------------------------------------------------
# Step 5b: Near-dupe penalty (per field, chunked)
# ---------------------------------------------------------------------------

def build_candidate_player_bool(dk_id_matrix, sorted_ids):
    """Build bool matrix (n_cand, n_players) for overlap computation."""
    n_cand = dk_id_matrix.shape[0]
    n_players = len(sorted_ids)
    cand_bool = np.zeros((n_cand, n_players), dtype=np.bool_)
    cand_idx = _map_ids_to_indices(dk_id_matrix, sorted_ids)
    for j in range(6):
        cand_bool[np.arange(n_cand), cand_idx[:, j]] = True
    return cand_bool


def build_field_player_bool(field_path, sorted_ids):
    """Build bool matrix (field_size, n_players) for a single field."""
    field = np.load(field_path)
    n_field = field.shape[0]
    n_players = len(sorted_ids)
    field_bool = np.zeros((n_field, n_players), dtype=np.bool_)
    field_idx = _map_ids_to_indices(field, sorted_ids)
    for j in range(6):
        field_bool[np.arange(n_field), field_idx[:, j]] = True
    return field_bool


def build_player_field_index(field_bool):
    """
    Returns list of n_players arrays.
    player_to_lineups[p] = sorted int32 array of field lineup
    indices containing player p.
    """
    n_players = field_bool.shape[1]
    player_to_lineups = []
    for p in range(n_players):
        lineup_indices = np.where(field_bool[:, p])[0].astype(np.int32)
        player_to_lineups.append(lineup_indices)
    return player_to_lineups


def compute_near_dupes_sparse(cand_bool, field_bool, field_label):
    """
    Sparse near-dupe computation using inverted index +
    overlap counter with selective reset.

    Returns:
        near_dupe_4: (n_cand,) int32
        near_dupe_5: (n_cand,) int32
    """
    n_cand = cand_bool.shape[0]
    n_field = field_bool.shape[0]
    near_dupe_4 = np.zeros(n_cand, dtype=np.int32)
    near_dupe_5 = np.zeros(n_cand, dtype=np.int32)

    print(f"    {field_label}: building inverted index...")
    t = time.perf_counter()
    player_to_lineups = build_player_field_index(field_bool)
    print(f"    Index built in {time.perf_counter()-t:.1f}s")

    # Pre-extract 6 player indices per candidate
    cand_player_indices = []
    for i in range(n_cand):
        players = np.where(cand_bool[i])[0]
        cand_player_indices.append(players)

    # Reusable overlap counter — reset only touched indices
    overlap_counter = np.zeros(n_field, dtype=np.int8)

    report_interval = max(1, n_cand // 10)
    t_start = time.perf_counter()

    for i in range(n_cand):
        players = cand_player_indices[i]

        # Accumulate overlaps
        for p in players:
            overlap_counter[player_to_lineups[p]] += 1

        # Count thresholds via concatenated index scan.
        # A lineup at overlap=k appears exactly k times in the concat,
        # so sum(counter[concat] == k) / k gives exact unique count.
        all_touched = np.concatenate([player_to_lineups[p] for p in players])
        vals = overlap_counter[all_touched]
        n_at_6 = int((vals == 6).sum()) // 6
        n_at_5 = int((vals == 5).sum()) // 5
        n_at_4 = int((vals == 4).sum()) // 4
        near_dupe_5[i] = n_at_5 + n_at_6
        near_dupe_4[i] = n_at_4

        # Reset only touched indices
        for p in players:
            overlap_counter[player_to_lineups[p]] = 0

        if (i + 1) % report_interval == 0 or i == n_cand - 1:
            elapsed = time.perf_counter() - t_start
            pct = (i + 1) / n_cand * 100
            rate = (i + 1) / elapsed
            eta = (n_cand - i - 1) / rate
            print(f"    {field_label} sparse: "
                  f"{i+1:,}/{n_cand:,} ({pct:.0f}%) "
                  f"rate={rate:.0f}/s eta={eta:.0f}s")

    return near_dupe_4, near_dupe_5


# ---------------------------------------------------------------------------
# Step 6a: Extract payout arrays for top 20K (second pass on field 1)
# ---------------------------------------------------------------------------

def extract_top20k_payouts(top20k_idx, candidate_scores, field_path,
                           player_matrix, sorted_ids, payout_table_arr,
                           metrics):
    """
    Re-run field_1 ranking for top 20K candidates only, storing full payout arrays.

    Returns adjusted payout matrix (TOP_K_PAYOUTS, N_ITERATIONS) float32.
    """
    # Subset candidate scores to top 20K
    cand_scores_sub = candidate_scores[top20k_idx]  # (20000, 10000)
    n_sub = cand_scores_sub.shape[0]
    n_iters = cand_scores_sub.shape[1]

    field = np.load(field_path)
    n_field = field.shape[0]
    field_player_idx = _map_ids_to_indices(field, sorted_ids)
    del field

    payout_matrix = np.zeros((n_sub, n_iters), dtype=np.float32)

    report_interval = max(1, n_iters // 10)
    for it in range(n_iters):
        field_scores = np.zeros(n_field, dtype=np.float32)
        for j in range(6):
            field_scores += player_matrix[field_player_idx[:, j], it]
        field_sorted = np.sort(field_scores)

        cand_scores_it = cand_scores_sub[:, it]
        insert_pos = np.searchsorted(field_sorted, cand_scores_it, side='left')
        iter_ranks = (n_field - insert_pos + 1).astype(np.int32)
        iter_ranks = np.clip(iter_ranks, 1, n_field + 1)

        clipped_ranks = np.clip(iter_ranks, 0, len(payout_table_arr) - 1)
        payout_matrix[:, it] = payout_table_arr[clipped_ranks]

        if (it + 1) % report_interval == 0 or it == 0:
            pct = (it + 1) / n_iters * 100
            print(f"    Payout extraction iter {it+1}/{n_iters} ({pct:.0f}%)")

    # Winsorize payout arrays before adjustments
    payout_matrix = np.minimum(payout_matrix, PAYOUT_CAP)

    # Apply dupe adjustments
    nd_penalty = metrics["near_dupe_penalty_factor"][top20k_idx]
    wave_mult = 1.0 + metrics["wave_correlation_bonus"][top20k_idx]
    dupe_factor = 1.0 / (metrics["expected_dupes"][top20k_idx] + 1.0)
    adj_factor = (nd_penalty * wave_mult * dupe_factor).astype(np.float32)

    payout_matrix *= adj_factor[:, np.newaxis]

    return payout_matrix


# ---------------------------------------------------------------------------
# Step 6: Compute metrics
# ---------------------------------------------------------------------------

def compute_metrics(candidate_scores, field_stats_list, rows,
                    near_dupe_4_avg, near_dupe_5_avg,
                    wave_bonus, expected_dupes_arr):
    """
    Compute per-candidate metrics from accumulated field stats.

    Applies dupe adjustment, near-dupe penalty, and wave correlation bonus.
    """
    n_cand = candidate_scores.shape[0]
    n_fields = len(field_stats_list)
    n_iters = candidate_scores.shape[1]
    n_total = n_iters * n_fields

    payout_sum = np.zeros(n_cand, dtype=np.float64)
    cash_count = np.zeros(n_cand, dtype=np.int32)
    top1000_count = np.zeros(n_cand, dtype=np.int32)
    top100_count = np.zeros(n_cand, dtype=np.int32)
    field_evs = []

    for fs in field_stats_list:
        payout_sum += fs["payout_sum"]
        cash_count += fs["cash_count"]
        top1000_count += fs["top1000_count"]
        top100_count += fs["top100_count"]
        field_evs.append(fs["field_mean_ev"])

    # Raw metrics (before dupe/near-dupe/wave adjustments)
    mean_payout_raw = (payout_sum / n_total).astype(np.float32)
    mean_ev_raw = mean_payout_raw - ENTRY_FEE

    sim_p90 = np.percentile(candidate_scores, 90, axis=1)

    cross_field_stability = np.std(
        np.column_stack(field_evs), axis=1).astype(np.float32)

    # --- Addition 1: Exact dupe adjustment ---
    dupe_factor = 1.0 / (expected_dupes_arr + 1.0)

    # --- Addition 2: Near-dupe penalty ---
    near_dupe_penalty = 1.0 / (
        1.0
        + 0.40 * near_dupe_5_avg
        + 0.15 * near_dupe_4_avg
    )

    # --- Addition 3: Wave correlation bonus ---
    wave_multiplier = 1.0 + wave_bonus

    # Adjusted payout and EV
    mean_payout_adj = mean_payout_raw * dupe_factor * near_dupe_penalty * wave_multiplier
    mean_ev = mean_payout_adj.astype(np.float32) - ENTRY_FEE

    return {
        "mean_ev_raw": mean_ev_raw,
        "mean_ev": mean_ev,
        "mean_payout": mean_payout_adj.astype(np.float32),
        "cash_rate": (cash_count / n_total).astype(np.float32),
        "top1000_rate": (top1000_count / n_total).astype(np.float32),
        "top100_rate": (top100_count / n_total).astype(np.float32),
        "sim_p90_score": sim_p90.astype(np.float32),
        "field_1_mean_ev": field_evs[0].astype(np.float32),
        "field_2_mean_ev": field_evs[1].astype(np.float32),
        "field_3_mean_ev": field_evs[2].astype(np.float32),
        "cross_field_stability": cross_field_stability,
        "expected_dupes": expected_dupes_arr.astype(np.float32),
        "near_dupe_4_count": near_dupe_4_avg.astype(np.float32),
        "near_dupe_5_count": near_dupe_5_avg.astype(np.float32),
        "near_dupe_penalty_factor": near_dupe_penalty.astype(np.float32),
        "wave_correlation_bonus": wave_bonus.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Step 7: Output
# ---------------------------------------------------------------------------

def save_results(rows, metrics):
    """Merge metrics into candidate rows and save."""
    metric_cols = [
        "mean_ev_raw", "mean_ev", "mean_payout", "cash_rate",
        "top1000_rate", "top100_rate",
        "sim_p90_score", "field_1_mean_ev", "field_2_mean_ev", "field_3_mean_ev",
        "cross_field_stability",
        "expected_dupes", "near_dupe_4_count", "near_dupe_5_count",
        "near_dupe_penalty_factor", "wave_correlation_bonus",
    ]

    with open(CANDIDATES_PATH) as f:
        reader = csv.DictReader(f)
        orig_fields = list(reader.fieldnames)

    # Remove expected_dupes from orig if present (we replace with computed version)
    write_fields = [f for f in orig_fields if f != "expected_dupes"] + metric_cols

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=write_fields)
        writer.writeheader()

        for i, row in enumerate(rows):
            out = {k: row.get(k, "") for k in orig_fields if k != "expected_dupes"}
            for col in metric_cols:
                out[col] = round(float(metrics[col][i]), 6)
            writer.writerow(out)

    print(f"  Saved {len(rows):,} rows to {OUTPUT}")


# ---------------------------------------------------------------------------
# Validation / reporting
# ---------------------------------------------------------------------------

def print_validation(rows, metrics, candidate_scores):
    n = len(rows)

    # Raw vs adjusted EV
    ev_raw = metrics["mean_ev_raw"]
    ev = metrics["mean_ev"]
    ev_raw_sorted = np.sort(ev_raw)
    ev_sorted = np.sort(ev)
    print(f"\n  mean_ev_raw distribution (before dupe/near-dupe/wave):")
    print(f"    min={ev_raw_sorted[0]:.2f}  p25={ev_raw_sorted[n//4]:.2f}  "
          f"median={ev_raw_sorted[n//2]:.2f}  p75={ev_raw_sorted[3*n//4]:.2f}  "
          f"max={ev_raw_sorted[-1]:.2f}")

    print(f"\n  mean_ev distribution (adjusted):")
    print(f"    min={ev_sorted[0]:.2f}  p25={ev_sorted[n//4]:.2f}  "
          f"median={ev_sorted[n//2]:.2f}  p75={ev_sorted[3*n//4]:.2f}  "
          f"max={ev_sorted[-1]:.2f}")

    delta = ev - ev_raw
    print(f"\n  EV shift from adjustments:")
    print(f"    mean shift: {delta.mean():.2f}  median shift: {np.median(delta):.2f}  "
          f"max penalty: {delta.min():.2f}  max boost: {delta.max():.2f}")

    # Near-dupe penalty distribution
    ndf = metrics["near_dupe_penalty_factor"]
    print(f"\n  Near-dupe penalty factor distribution:")
    print(f"    min={ndf.min():.4f}  p25={np.percentile(ndf,25):.4f}  "
          f"median={np.median(ndf):.4f}  p75={np.percentile(ndf,75):.4f}  "
          f"max={ndf.max():.4f}")

    nd4 = metrics["near_dupe_4_count"]
    nd5 = metrics["near_dupe_5_count"]
    print(f"    near_dupe_4 median={np.median(nd4):.0f}  mean={nd4.mean():.1f}  max={nd4.max():.0f}")
    print(f"    near_dupe_5 median={np.median(nd5):.0f}  mean={nd5.mean():.1f}  max={nd5.max():.0f}")

    # Wave bonus
    wb = metrics["wave_correlation_bonus"]
    n_wave4 = int((wb >= 0.08).sum())
    n_wave3 = int(((wb >= 0.04) & (wb < 0.08)).sum())
    n_wave0 = int((wb < 0.04).sum())
    print(f"\n  Wave correlation bonus:")
    print(f"    4+ same wave (8% boost): {n_wave4:,} lineups")
    print(f"    3 same wave (4% boost):  {n_wave3:,} lineups")
    print(f"    No bonus:                {n_wave0:,} lineups")

    # Payout winsorization
    print(f"\n  Payout winsorization:")
    print(f"    Cap applied: ${PAYOUT_CAP:,.0f}")
    print(f"    Mean EV after cap: ${ev.mean():.2f}")
    print(f"    Max EV after cap: ${ev.max():.2f}")

    # Cash rate
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

    # Top 20 by adjusted mean_ev
    top_idx = np.argsort(-ev)[:20]
    print(f"\n  Top 20 lineups by adjusted mean_ev:")
    print(f"  {'Rank':>4s}  {'LineupID':>8s}  {'Bot':<20s}  "
          f"{'raw_ev':>8s}  {'adj_ev':>8s}  {'nd_pen':>7s}  {'wave':>5s}  "
          f"{'cash%':>6s}  {'stab':>6s}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*20}  "
          f"{'-'*8}  {'-'*8}  {'-'*7}  {'-'*5}  "
          f"{'-'*6}  {'-'*6}")
    for rank, idx in enumerate(top_idx, 1):
        row = rows[idx]
        print(f"  {rank:>4d}  {row['lineup_id']:>8s}  {row['strategy']:<20s}  "
              f"${ev_raw[idx]:>7.2f}  ${ev[idx]:>7.2f}  "
              f"{ndf[idx]:>7.4f}  {wb[idx]:>5.2f}  "
              f"{cr[idx]:>5.1%}  "
              f"{metrics['cross_field_stability'][idx]:>6.1f}")

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
    dk_wave = load_wave_data()
    timers["load_mappings"] = time.perf_counter() - t
    print(f"  {len(dk_to_dg)} dk->dg mappings")
    print(f"  {len(dk_wave)} players with wave data")

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
    field_stats_list = []

    for fi in range(1, N_FIELDS + 1):
        field_path = os.path.join(FIELD_DIR, f"field_{fi}.npy")
        print(f"\n  --- Field {fi} ---")
        t = time.perf_counter()
        field_stats = rank_candidates_against_field(
            candidate_scores, field_path,
            player_matrix, sorted_ids,
            payout_table_arr, f"Field {fi}")
        elapsed = time.perf_counter() - t
        timers[f"field_{fi}"] = elapsed
        field_stats_list.append(field_stats)
        print(f"    Field {fi} done in {elapsed:.1f}s")

    # Step 5b: Near-dupe computation
    print(f"\nStep 5b: Computing near-dupe penalties...")
    t = time.perf_counter()
    cand_bool = build_candidate_player_bool(dk_id_matrix, sorted_ids)
    print(f"  Candidate bool matrix: {cand_bool.shape} ({cand_bool.nbytes / 1e6:.1f} MB)")

    all_nd4 = []
    all_nd5 = []
    for fi in range(1, N_FIELDS + 1):
        field_path = os.path.join(FIELD_DIR, f"field_{fi}.npy")
        print(f"\n  --- Near-dupes vs Field {fi} ---")
        field_bool = build_field_player_bool(field_path, sorted_ids)
        nd4, nd5 = compute_near_dupes_sparse(cand_bool, field_bool, f"Field {fi}")
        all_nd4.append(nd4.astype(np.float64))
        all_nd5.append(nd5.astype(np.float64))
        del field_bool

    # Average across 3 fields
    near_dupe_4_avg = (all_nd4[0] + all_nd4[1] + all_nd4[2]) / 3.0
    near_dupe_5_avg = (all_nd5[0] + all_nd5[1] + all_nd5[2]) / 3.0
    timers["near_dupes"] = time.perf_counter() - t
    print(f"\n  Near-dupe computation done in {timers['near_dupes']:.1f}s")
    del all_nd4, all_nd5, cand_bool

    # Step 5c: Wave correlation bonus
    print(f"\nStep 5c: Computing wave correlation bonus...")
    t = time.perf_counter()
    wave_bonus = np.zeros(n_cand, dtype=np.float32)
    for i in range(n_cand):
        wave_counts = Counter()
        for j in range(6):
            dk_id = int(dk_id_matrix[i, j])
            w = dk_wave.get(dk_id, -1)
            if w >= 0:
                wave_counts[w] += 1
        max_wave = max(wave_counts.values()) if wave_counts else 0
        if max_wave >= 4:
            wave_bonus[i] = 0.08
        elif max_wave >= 3:
            wave_bonus[i] = 0.04
    timers["wave_bonus"] = time.perf_counter() - t
    print(f"  Wave bonus computed in {timers['wave_bonus']:.1f}s")

    # Extract expected_dupes from rows
    expected_dupes_arr = np.array(
        [float(row.get("expected_dupes", 0)) for row in rows],
        dtype=np.float64)

    # Step 6
    print("\nStep 6: Computing metrics...")
    t = time.perf_counter()
    metrics = compute_metrics(candidate_scores, field_stats_list, rows,
                              near_dupe_4_avg, near_dupe_5_avg,
                              wave_bonus, expected_dupes_arr)
    timers["metrics"] = time.perf_counter() - t

    # Step 6b: Extract payout arrays for top 20K
    print(f"\nStep 6b: Extracting payout arrays for top {TOP_K_PAYOUTS:,}...")
    t = time.perf_counter()
    top20k_idx = np.argsort(-metrics["mean_ev"])[:TOP_K_PAYOUTS].astype(np.int32)
    field1_path = os.path.join(FIELD_DIR, "field_1.npy")
    top20k_payouts = extract_top20k_payouts(
        top20k_idx, candidate_scores, field1_path,
        player_matrix, sorted_ids, payout_table_arr, metrics)

    np.savez_compressed(PAYOUT_ARRAYS_PATH,
                        payouts=top20k_payouts,
                        candidate_indices=top20k_idx,
                        mean_ev=metrics["mean_ev"][top20k_idx])
    timers["payout_extract"] = time.perf_counter() - t

    ev_top = metrics["mean_ev"][top20k_idx]
    print(f"  Saved payout arrays: shape {top20k_payouts.shape}, "
          f"size {top20k_payouts.nbytes / 1e9:.2f}GB")
    print(f"  Mean adjusted EV top{TOP_K_PAYOUTS//1000}k: "
          f"min=${ev_top.min():.2f}  median=${np.median(ev_top):.2f}  max=${ev_top.max():.2f}")
    print(f"  Extraction done in {timers['payout_extract']:.1f}s")
    del top20k_payouts

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

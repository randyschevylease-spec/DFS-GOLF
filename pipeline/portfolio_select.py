"""
portfolio_select.py -- Bergman E[max] portfolio selection from contest sim results.

Selects 150 lineups from 100k contest_sim_results.csv using greedy forward
selection that maximizes E[max payout] with a diversity bonus to penalize
player overlap.

Pipeline:
  1. Light pre-filter: remove dead lineups (mean_ev < -15, cash_rate < 15%)
  2. Parse player IDs and build player membership matrix
  3. Vectorized greedy forward selection: score = mean_ev + diversity_bonus
  4. Portfolio metrics and save

Input:  data/cache/contest_sim_results.csv (100,000 rows)
Output: data/cache/portfolio_selected.csv (150 rows)
"""

import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(PROJECT_ROOT, "data", "cache", "contest_sim_results.csv")
OUTPUT = os.path.join(PROJECT_ROOT, "data", "cache", "portfolio_selected.csv")

N_PORTFOLIO = 150
ENTRY_FEE = 25
DIVERSITY_WEIGHT = 0.35


def load_results(path):
    """Load contest sim results. Returns list of dicts with parsed fields."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            player_ids = []
            player_names = []
            for i in range(1, 7):
                player_ids.append(row[f"p{i}_id"])
                player_names.append(row[f"p{i}_name"])
            rows.append({
                "lineup_id": row["lineup_id"],
                "strategy": row["strategy"],
                "player_ids": player_ids,
                "player_names": player_names,
                "total_salary": int(row["total_salary"]),
                "mean_ev": float(row["mean_ev"]),
                "mean_payout": float(row["mean_payout"]),
                "cash_rate": float(row["cash_rate"]),
                "top1000_rate": float(row["top1000_rate"]),
                "top100_rate": float(row["top100_rate"]),
                "cross_field_stability": float(row["cross_field_stability"]),
                "sim_p90_score": float(row["sim_p90_score"]),
                "raw": row,
            })
    return rows


def pre_filter(rows):
    """Light filter: remove only genuinely dead lineups."""
    n_total = len(rows)
    filtered = [
        r for r in rows
        if r["mean_ev"] >= -15
        and r["cash_rate"] >= 0.15
    ]
    ev_pass = sum(1 for r in rows if r["mean_ev"] >= -15)
    cash_pass = sum(1 for r in rows if r["cash_rate"] >= 0.15)
    print(f"  Pre-filter ({n_total:,} input):")
    print(f"    mean_ev >= -15:    {ev_pass:,} pass")
    print(f"    cash_rate >= 0.15: {cash_pass:,} pass")
    print(f"    Both:              {len(filtered):,} pass")
    return filtered


def build_player_matrix(pool):
    """
    Build player membership matrix for vectorized overlap calculation.

    Returns:
        player_matrix: bool array (n_lineups, n_players)
        player_id_list: list of unique player IDs (index corresponds to matrix column)
        mean_ev_array: float array (n_lineups,)
    """
    # Collect all unique player IDs
    all_ids = set()
    for r in pool:
        all_ids.update(r["player_ids"])
    player_id_list = sorted(all_ids)
    id_to_idx = {pid: i for i, pid in enumerate(player_id_list)}

    n = len(pool)
    n_players = len(player_id_list)
    player_matrix = np.zeros((n, n_players), dtype=np.bool_)
    mean_ev_array = np.zeros(n, dtype=np.float64)

    for i, r in enumerate(pool):
        for pid in r["player_ids"]:
            player_matrix[i, id_to_idx[pid]] = True
        mean_ev_array[i] = r["mean_ev"]

    return player_matrix, player_id_list, mean_ev_array


def greedy_select(pool, player_matrix, mean_ev_array, n_select):
    """
    Vectorized greedy forward selection.

    score = mean_ev + DIVERSITY_WEIGHT * (1 - overlap_fraction) * |median_ev|
    overlap_fraction = (# players in lineup that appear in portfolio) / 6
    """
    n = len(pool)
    median_ev = float(np.median(mean_ev_array))
    print(f"  Median EV of pool: ${median_ev:.2f}")
    abs_median = abs(median_ev)

    selected_indices = []
    selected_mask = np.zeros(n, dtype=np.bool_)

    # Track which players are in portfolio (count of appearances)
    portfolio_player_present = np.zeros(player_matrix.shape[1], dtype=np.bool_)

    for step in range(n_select):
        # Vectorized overlap: how many of each lineup's 6 players are in portfolio
        if step == 0:
            overlaps = np.zeros(n, dtype=np.float64)
        else:
            # player_matrix (n, 121) dot portfolio_player_present (121,) -> (n,)
            overlap_counts = player_matrix.dot(portfolio_player_present.astype(np.float64))
            overlaps = overlap_counts / 6.0

        diversity_bonus = DIVERSITY_WEIGHT * (1.0 - overlaps) * abs_median
        scores = mean_ev_array + diversity_bonus
        scores[selected_mask] = -1e9

        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]

        selected_indices.append(best_idx)
        selected_mask[best_idx] = True

        # Update portfolio player presence
        portfolio_player_present |= player_matrix[best_idx]

        cand = pool[best_idx]
        if (step + 1) <= 10 or (step + 1) % 25 == 0 or (step + 1) == n_select:
            print(f"  Step {step+1:>3d}: +{cand['lineup_id']:>8s} "
                  f"({cand['strategy']:<20s}) "
                  f"ev=${cand['mean_ev']:>7.2f}  "
                  f"overlap={overlaps[best_idx]:.3f}  "
                  f"score={best_score:.2f}")

    return selected_indices


def compute_portfolio_overlap(pool, selected_indices):
    """Compute per-lineup average overlap with rest of portfolio."""
    selected = [pool[i] for i in selected_indices]
    n = len(selected)
    id_sets = [set(c["player_ids"]) for c in selected]

    overlaps = []
    for i in range(n):
        if n == 1:
            overlaps.append(0.0)
            continue
        total = sum(len(id_sets[i] & id_sets[j]) for j in range(n) if j != i)
        overlaps.append(total / (6 * (n - 1)))
    return overlaps


def print_portfolio_metrics(pool, selected_indices, overlaps):
    """Print comprehensive portfolio-level metrics."""
    selected = [pool[i] for i in selected_indices]
    n = len(selected)

    bot_counts = defaultdict(int)
    for c in selected:
        bot_counts[c["strategy"]] += 1

    player_exposure = defaultdict(int)
    player_name_map = {}
    for c in selected:
        for pid, name in zip(c["player_ids"], c["player_names"]):
            player_exposure[pid] += 1
            player_name_map[pid] = name

    evs = [c["mean_ev"] for c in selected]
    cash_rates = [c["cash_rate"] for c in selected]

    print("\n" + "=" * 70)
    print("PORTFOLIO SUMMARY")
    print("=" * 70)

    print(f"\n  Lineups selected:    {n}")
    print(f"  Total invested:      ${n * ENTRY_FEE:,}")
    print(f"  Mean EV per lineup:  ${sum(evs)/n:.2f}")
    print(f"  Median EV:           ${sorted(evs)[n//2]:.2f}")
    print(f"  Min EV in portfolio: ${min(evs):.2f}")
    print(f"  Sum of mean_ev:      ${sum(evs):,.2f}")
    print(f"  Expected gross:      ${sum(c['mean_payout'] for c in selected):,.2f}")
    print(f"  Expected net profit: ${sum(evs):,.2f}")
    print(f"  Mean cash rate:      {sum(cash_rates)/n:.1%}")
    print(f"  Mean overlap:        {sum(overlaps)/n:.3f}  (target <= 0.500)")

    print(f"\n  Bot archetype distribution:")
    print(f"  {'Bot':<22s} {'Count':>6s} {'%':>7s}")
    print(f"  {'-'*22} {'-'*6} {'-'*7}")
    for strat in sorted(bot_counts, key=lambda s: -bot_counts[s]):
        c = bot_counts[strat]
        print(f"  {strat:<22s} {c:>6d} {c/n:>7.1%}")

    sorted_players = sorted(player_exposure.items(), key=lambda x: -x[1])
    print(f"\n  Top 10 most-used players:")
    print(f"  {'Player':<28s} {'Lineups':>8s} {'Exposure':>9s}")
    print(f"  {'-'*28} {'-'*8} {'-'*9}")
    for pid, count in sorted_players[:10]:
        print(f"  {player_name_map[pid]:<28s} {count:>8d} {count/n:>9.1%}")

    print(f"\n  Top 10 least-used players (contrarian):")
    print(f"  {'Player':<28s} {'Lineups':>8s} {'Exposure':>9s}")
    print(f"  {'-'*28} {'-'*8} {'-'*9}")
    for pid, count in sorted_players[-10:]:
        print(f"  {player_name_map[pid]:<28s} {count:>8d} {count/n:>9.1%}")

    # Per-lineup detail
    print(f"\n  Per-lineup detail:")
    print(f"  {'Rank':>4s}  {'LineupID':>8s}  {'Bot':<20s}  "
          f"{'mean_ev':>9s}  {'cash%':>6s}  {'stab':>6s}  {'overlap':>7s}")
    print(f"  {'----':>4s}  {'--------':>8s}  {'-'*20}  "
          f"{'-'*9}  {'-'*6}  {'-'*6}  {'-'*7}")
    for rank, (idx, ol) in enumerate(zip(selected_indices, overlaps), 1):
        cand = pool[idx]
        print(f"  {rank:>4d}  {cand['lineup_id']:>8s}  {cand['strategy']:<20s}  "
              f"${cand['mean_ev']:>8.2f}  {cand['cash_rate']:>5.1%}  "
              f"{cand['cross_field_stability']:>6.1f}  {ol:>7.3f}")


def save_portfolio(pool, selected_indices, overlaps, path):
    """Save selected portfolio with original columns plus selection metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    sample = pool[selected_indices[0]]["raw"]
    original_cols = list(sample.keys())
    extra_cols = ["selection_rank", "avg_portfolio_overlap"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(extra_cols + original_cols)
        for rank, (idx, ol) in enumerate(zip(selected_indices, overlaps), 1):
            raw = pool[idx]["raw"]
            writer.writerow([rank, f"{ol:.4f}"] + [raw[c] for c in original_cols])

    print(f"\n  Saved {len(selected_indices)} lineups to {path}")


def run():
    t0 = time.perf_counter()

    # Load
    print("Step 1: Loading contest sim results...")
    rows = load_results(INPUT)
    print(f"  {len(rows):,} lineups loaded")

    # Pre-filter
    print("\nStep 2: Pre-filtering (light)...")
    pool = pre_filter(rows)
    rows = None

    # Build player matrix
    print(f"\nStep 3: Building player membership matrix...")
    t_mat = time.perf_counter()
    player_matrix, player_id_list, mean_ev_array = build_player_matrix(pool)
    print(f"  Matrix shape: {player_matrix.shape} ({player_matrix.nbytes / 1e6:.1f} MB)")
    print(f"  Built in {time.perf_counter() - t_mat:.1f}s")

    # Greedy selection
    print(f"\nStep 4: Greedy E[max] selection ({N_PORTFOLIO} lineups from {len(pool):,} candidates)...")
    t_sel = time.perf_counter()
    selected_indices = greedy_select(pool, player_matrix, mean_ev_array, N_PORTFOLIO)
    print(f"  Selection done in {time.perf_counter() - t_sel:.1f}s")

    # Portfolio overlap
    print("\nStep 5: Computing portfolio metrics...")
    overlaps = compute_portfolio_overlap(pool, selected_indices)
    print_portfolio_metrics(pool, selected_indices, overlaps)

    # Save
    print("\nStep 6: Saving...")
    save_portfolio(pool, selected_indices, overlaps, OUTPUT)

    elapsed = time.perf_counter() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    run()

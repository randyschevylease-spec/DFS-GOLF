"""
portfolio_select.py -- Bergman E[max] portfolio selection with simulated annealing.

Phase 1: Greedy forward selection (existing) — fast, good starting point
Phase 2: Simulated annealing refinement — true E[max] optimization using
         per-iteration payout arrays from contest_sim

Input:  data/cache/contest_sim_results.csv (100,000 rows)
        data/cache/payout_arrays_top20k.npz (20,000 x 10,000 payout matrix)
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
PAYOUT_ARRAYS_PATH = os.path.join(PROJECT_ROOT, "data", "cache", "payout_arrays_top20k.npz")

N_PORTFOLIO = 150
ENTRY_FEE = 25
DIVERSITY_WEIGHT = 0.35

# Annealing parameters
N_ANNEALING_ITERATIONS = 100000
INITIAL_TEMPERATURE = 100.0
COOLING_RATE = 0.99992
MIN_TEMPERATURE = 0.01


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
    """Build player membership matrix for vectorized overlap calculation."""
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
    """Vectorized greedy forward selection."""
    n = len(pool)
    median_ev = float(np.median(mean_ev_array))
    print(f"  Median EV of pool: ${median_ev:.2f}")
    abs_median = abs(median_ev)

    selected_indices = []
    selected_mask = np.zeros(n, dtype=np.bool_)
    portfolio_player_present = np.zeros(player_matrix.shape[1], dtype=np.bool_)

    for step in range(n_select):
        if step == 0:
            overlaps = np.zeros(n, dtype=np.float64)
        else:
            overlap_counts = player_matrix.dot(portfolio_player_present.astype(np.float64))
            overlaps = overlap_counts / 6.0

        diversity_bonus = DIVERSITY_WEIGHT * (1.0 - overlaps) * abs_median
        scores = mean_ev_array + diversity_bonus
        scores[selected_mask] = -1e9

        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]

        selected_indices.append(best_idx)
        selected_mask[best_idx] = True
        portfolio_player_present |= player_matrix[best_idx]

        cand = pool[best_idx]
        if (step + 1) <= 10 or (step + 1) % 25 == 0 or (step + 1) == n_select:
            print(f"  Step {step+1:>3d}: +{cand['lineup_id']:>8s} "
                  f"({cand['strategy']:<20s}) "
                  f"ev=${cand['mean_ev']:>7.2f}  "
                  f"overlap={overlaps[best_idx]:.3f}  "
                  f"score={best_score:.2f}")

    return selected_indices


# ---------------------------------------------------------------------------
# Phase 2: Simulated Annealing
# ---------------------------------------------------------------------------

def anneal_portfolio(pool, greedy_indices, pool_row_to_sim_row):
    """
    Simulated annealing refinement of greedy portfolio using payout arrays.

    Args:
        pool: filtered pool list
        greedy_indices: list of pool indices from greedy selection
        pool_row_to_sim_row: dict mapping pool index -> contest_sim_results row index

    Returns:
        best_portfolio_pool_indices: list of pool indices for best portfolio found
        greedy_emax: E[max] of greedy starting point
        best_emax: E[max] of best annealed portfolio
    """
    if not os.path.exists(PAYOUT_ARRAYS_PATH):
        print("  WARNING: payout_arrays_top20k.npz not found, skipping annealing")
        return greedy_indices, 0.0, 0.0

    # Step 1: Load payout arrays
    print("  Loading payout arrays...")
    data = np.load(PAYOUT_ARRAYS_PATH)
    payouts = data["payouts"]  # (20000, 10000) float32
    candidate_indices = data["candidate_indices"]  # maps top20k pos -> contest_sim row
    print(f"  Payout arrays: {payouts.shape}, {payouts.nbytes / 1e9:.2f}GB")

    # Build reverse mapping: contest_sim_row -> top20k_index
    row_to_top20k = {}
    for i in range(len(candidate_indices)):
        row_to_top20k[int(candidate_indices[i])] = i

    # Step 2: Map greedy portfolio to top20k indices
    portfolio_top20k = []
    portfolio_pool_idx = []  # parallel list of pool indices
    for pool_idx in greedy_indices:
        sim_row = pool_row_to_sim_row.get(pool_idx)
        if sim_row is not None:
            top20k_i = row_to_top20k.get(sim_row)
            if top20k_i is not None:
                portfolio_top20k.append(top20k_i)
                portfolio_pool_idx.append(pool_idx)

    n_mapped = len(portfolio_top20k)
    if n_mapped < N_PORTFOLIO:
        # Fill missing slots from top of top20k not already selected
        used = set(portfolio_top20k)
        for i in range(len(payouts)):
            if len(portfolio_top20k) >= N_PORTFOLIO:
                break
            if i not in used:
                portfolio_top20k.append(i)
                portfolio_pool_idx.append(-1)  # placeholder
                used.add(i)
        print(f"  Mapped {n_mapped}/{N_PORTFOLIO} greedy picks to top20k, "
              f"filled {N_PORTFOLIO - n_mapped} from pool")

    portfolio_top20k = portfolio_top20k[:N_PORTFOLIO]

    # Step 3: Compute initial E[max]
    portfolio_payouts = payouts[portfolio_top20k].copy()  # (150, 10000)
    current_emax = float(np.mean(np.max(portfolio_payouts, axis=0)))
    greedy_emax = current_emax

    # Build available pool
    portfolio_set = set(portfolio_top20k)
    available = [i for i in range(len(payouts)) if i not in portfolio_set]

    print(f"  Greedy E[max]: ${greedy_emax:.2f}")
    print(f"  Portfolio size: {len(portfolio_top20k)}, Available pool: {len(available)}")

    # Step 4: Annealing loop
    rng = np.random.RandomState(42)
    temperature = INITIAL_TEMPERATURE
    best_emax = current_emax
    best_portfolio = list(portfolio_top20k)
    accepted = 0
    rejected = 0
    report_interval = N_ANNEALING_ITERATIONS // 10

    for iteration in range(N_ANNEALING_ITERATIONS):
        # Pick random lineup to remove
        remove_pos = rng.randint(0, N_PORTFOLIO)
        remove_idx = portfolio_top20k[remove_pos]

        # Pick random candidate to add
        add_pos_in_avail = rng.randint(0, len(available))
        add_idx = available[add_pos_in_avail]

        # In-place swap with rollback
        saved_row = portfolio_payouts[remove_pos].copy()
        portfolio_payouts[remove_pos] = payouts[add_idx]
        new_emax = float(np.mean(np.max(portfolio_payouts, axis=0)))

        delta = new_emax - current_emax

        if delta > 0 or rng.random() < np.exp(delta / max(temperature, 1e-10)):
            # Accept
            available[add_pos_in_avail] = remove_idx
            portfolio_top20k[remove_pos] = add_idx
            portfolio_set.discard(remove_idx)
            portfolio_set.add(add_idx)
            current_emax = new_emax
            accepted += 1

            if new_emax > best_emax:
                best_emax = new_emax
                best_portfolio = list(portfolio_top20k)
        else:
            # Reject — rollback
            portfolio_payouts[remove_pos] = saved_row
            rejected += 1

        temperature = max(temperature * COOLING_RATE, MIN_TEMPERATURE)

        if (iteration + 1) % report_interval == 0:
            accept_rate = accepted / (accepted + rejected)
            print(f"  Annealing {iteration+1}/{N_ANNEALING_ITERATIONS} "
                  f"temp={temperature:.3f} "
                  f"current_emax=${current_emax:.2f} "
                  f"best_emax=${best_emax:.2f} "
                  f"accept_rate={accept_rate:.2%}")

    # Step 5: Map best portfolio back to pool indices
    # Build top20k -> sim_row -> pool_idx mapping
    top20k_to_sim = {int(candidate_indices[i]): i for i in range(len(candidate_indices))}
    sim_to_pool = {v: k for k, v in pool_row_to_sim_row.items()}

    best_pool_indices = []
    for t20k_idx in best_portfolio:
        sim_row = int(candidate_indices[t20k_idx])
        pool_idx = sim_to_pool.get(sim_row)
        if pool_idx is not None:
            best_pool_indices.append(pool_idx)

    # Fill if any couldn't map back (shouldn't happen)
    if len(best_pool_indices) < N_PORTFOLIO:
        used_pool = set(best_pool_indices)
        # Fallback: use greedy picks not already in
        for gi in greedy_indices:
            if len(best_pool_indices) >= N_PORTFOLIO:
                break
            if gi not in used_pool:
                best_pool_indices.append(gi)
                used_pool.add(gi)

    return best_pool_indices[:N_PORTFOLIO], greedy_emax, best_emax


# ---------------------------------------------------------------------------
# Portfolio metrics and save (unchanged)
# ---------------------------------------------------------------------------

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


def print_portfolio_metrics(pool, selected_indices, overlaps, label=""):
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

    print(f"\n{'=' * 70}")
    print(f"PORTFOLIO SUMMARY{' — ' + label if label else ''}")
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

    # Per-lineup detail (first 20 + last 5)
    print(f"\n  Per-lineup detail (top 20):")
    print(f"  {'Rank':>4s}  {'LineupID':>8s}  {'Bot':<20s}  "
          f"{'mean_ev':>9s}  {'cash%':>6s}  {'stab':>6s}  {'overlap':>7s}")
    print(f"  {'----':>4s}  {'--------':>8s}  {'-'*20}  "
          f"{'-'*9}  {'-'*6}  {'-'*6}  {'-'*7}")
    for rank, (idx, ol) in enumerate(zip(selected_indices[:20], overlaps[:20]), 1):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    t0 = time.perf_counter()

    # Load
    print("Step 1: Loading contest sim results...")
    rows = load_results(INPUT)
    n_total = len(rows)
    print(f"  {n_total:,} lineups loaded")

    # Pre-filter
    print("\nStep 2: Pre-filtering (light)...")
    pool = pre_filter(rows)

    # Build mapping: pool index -> original contest_sim_results row index
    # We need this to map between pool indices and the payout arrays
    # The payout arrays reference contest_sim_results row indices
    pool_row_to_sim_row = {}
    pool_lineup_ids = {r["lineup_id"]: i for i, r in enumerate(pool)}
    for sim_idx, r in enumerate(rows):
        pool_idx = pool_lineup_ids.get(r["lineup_id"])
        if pool_idx is not None:
            pool_row_to_sim_row[pool_idx] = sim_idx
    rows = None

    # Build player matrix
    print(f"\nStep 3: Building player membership matrix...")
    t_mat = time.perf_counter()
    player_matrix, player_id_list, mean_ev_array = build_player_matrix(pool)
    print(f"  Matrix shape: {player_matrix.shape} ({player_matrix.nbytes / 1e6:.1f} MB)")
    print(f"  Built in {time.perf_counter() - t_mat:.1f}s")

    # Phase 1: Greedy selection
    print(f"\nPhase 1: Greedy E[max] selection ({N_PORTFOLIO} lineups from {len(pool):,} candidates)...")
    t_sel = time.perf_counter()
    greedy_indices = greedy_select(pool, player_matrix, mean_ev_array, N_PORTFOLIO)
    print(f"  Greedy done in {time.perf_counter() - t_sel:.1f}s")

    # Phase 2: Simulated annealing
    print(f"\nPhase 2: Simulated annealing ({N_ANNEALING_ITERATIONS:,} iterations)...")
    t_anneal = time.perf_counter()
    final_indices, greedy_emax, best_emax = anneal_portfolio(
        pool, greedy_indices, pool_row_to_sim_row)
    anneal_time = time.perf_counter() - t_anneal

    if best_emax > 0:
        improvement = best_emax - greedy_emax
        pct = improvement / greedy_emax * 100 if greedy_emax > 0 else 0
        print(f"\n  Phase 1 (greedy) E[max]:    ${greedy_emax:.2f}")
        print(f"  Phase 2 (annealing) E[max]: ${best_emax:.2f}")
        print(f"  Improvement: ${improvement:.2f} ({pct:.1f}%)")
        print(f"  Annealing done in {anneal_time:.1f}s")

    # Use annealed portfolio if available, else greedy
    selected_indices = final_indices if best_emax > 0 else greedy_indices

    # Portfolio metrics
    print("\nStep 5: Computing portfolio metrics...")
    overlaps = compute_portfolio_overlap(pool, selected_indices)
    label = "Annealed" if best_emax > 0 else "Greedy"
    print_portfolio_metrics(pool, selected_indices, overlaps, label=label)

    # Save
    print("\nStep 6: Saving...")
    save_portfolio(pool, selected_indices, overlaps, OUTPUT)

    elapsed = time.perf_counter() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    run()

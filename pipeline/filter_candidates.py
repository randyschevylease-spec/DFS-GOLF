"""
filter_candidates.py -- Filter 7.8M candidates down to 100,000 for sim.

Filter pipeline:
  1. Floor filter:     remove lineups with avg_sim_mean below 10th percentile
  2. Salary filter:    remove lineups using less than 80% of salary cap
  3. Ceiling filter:   remove lineups with avg_ceiling below 15th percentile
  4. Diversity score:  rank by composite (0.4*ceiling + 0.3*mean + 0.3*salary)
  5. Bot diversity:    guarantee at least 5,000 lineups per bot archetype
  6. Top-k selection:  fill remaining slots from global ranking

Input:  data/cache/candidates.csv (7.8M rows)
Output: data/cache/candidates_filtered.csv (100,000 rows)

Memory-efficient: two-pass approach. First pass reads only numeric columns
for filtering/scoring. Second pass copies only selected rows.
"""

import csv
import os
import sys
import time
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(PROJECT_ROOT, "data", "cache", "candidates.csv")
OUTPUT = os.path.join(PROJECT_ROOT, "data", "cache", "candidates_filtered.csv")

SALARY_CAP = 50_000
TARGET = 100_000


def filter_candidates():
    t0 = time.perf_counter()

    # ---------------------------------------------------------------
    # Pass 1: Read only the 4 numeric columns + strategy + line number
    # ---------------------------------------------------------------
    print("Pass 1: Reading numeric columns for filtering...")
    t1 = time.perf_counter()

    # Each entry: (line_number, strategy, total_salary, avg_sim_mean, avg_ceiling)
    entries = []
    with open(INPUT) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            entries.append((
                i,                                   # line index (0-based row)
                row["strategy"],
                int(row["total_salary"]),
                float(row["avg_sim_mean"]),
                float(row["avg_ceiling"]),
            ))

    n_total = len(entries)
    print(f"  Loaded {n_total:,} rows in {time.perf_counter() - t1:.1f}s")

    # ---------------------------------------------------------------
    # Step 1: Floor filter (remove bottom 10% by avg_sim_mean)
    # ---------------------------------------------------------------
    means = sorted(e[3] for e in entries)
    floor_cutoff = means[int(len(means) * 0.10)]
    before = len(entries)
    entries = [e for e in entries if e[3] >= floor_cutoff]
    removed_1 = before - len(entries)
    print(f"\n  Step 1 -- Floor filter (avg_sim_mean >= {floor_cutoff:.1f}):")
    print(f"    {before:,} -> {len(entries):,}  (removed {removed_1:,})")

    # ---------------------------------------------------------------
    # Step 2: Salary filter (must use >= 80% of cap)
    # ---------------------------------------------------------------
    salary_floor = int(SALARY_CAP * 0.80)
    before = len(entries)
    entries = [e for e in entries if e[2] >= salary_floor]
    removed_2 = before - len(entries)
    print(f"\n  Step 2 -- Salary filter (total_salary >= ${salary_floor:,}):")
    print(f"    {before:,} -> {len(entries):,}  (removed {removed_2:,})")

    # ---------------------------------------------------------------
    # Step 3: Ceiling filter (remove bottom 15% by avg_ceiling)
    # ---------------------------------------------------------------
    ceils = sorted(e[4] for e in entries)
    ceiling_cutoff = ceils[int(len(ceils) * 0.15)]
    before = len(entries)
    entries = [e for e in entries if e[4] >= ceiling_cutoff]
    removed_3 = before - len(entries)
    print(f"\n  Step 3 -- Ceiling filter (avg_ceiling >= {ceiling_cutoff:.1f}):")
    print(f"    {before:,} -> {len(entries):,}  (removed {removed_3:,})")

    # ---------------------------------------------------------------
    # Step 4: Score all remaining candidates
    # ---------------------------------------------------------------
    print(f"\n  Step 4 -- Scoring {len(entries):,} candidates...")

    sal_vals = [e[2] for e in entries]
    mean_vals = [e[3] for e in entries]
    ceil_vals = [e[4] for e in entries]

    sal_min, sal_max = min(sal_vals), max(sal_vals)
    mean_min, mean_max = min(mean_vals), max(mean_vals)
    ceil_min, ceil_max = min(ceil_vals), max(ceil_vals)

    sal_range = max(sal_max - sal_min, 1)
    mean_range = max(mean_max - mean_min, 1.0)
    ceil_range = max(ceil_max - ceil_min, 1.0)

    # Augment entries with score: (line_idx, strategy, salary, mean, ceiling, score)
    scored = []
    for e in entries:
        ns = (e[2] - sal_min) / sal_range
        nm = (e[3] - mean_min) / mean_range
        nc = (e[4] - ceil_min) / ceil_range
        score = 0.4 * nc + 0.3 * nm + 0.3 * ns
        scored.append((*e, score))

    entries = None  # free memory

    # ---------------------------------------------------------------
    # Step 5: Bot diversity selection (equal caps per bot)
    # ---------------------------------------------------------------
    # Group by strategy, sort each by score desc
    by_strategy = defaultdict(list)
    for s in scored:
        by_strategy[s[1]].append(s)

    n_bots = len(by_strategy)
    BOT_MAX = TARGET // n_bots  # ~11,111 per bot with 9 bots

    print(f"\n  Step 5 -- Selecting {TARGET:,} with equal caps "
          f"(BOT_MAX={BOT_MAX:,}/bot, {n_bots} bots)...")

    for strat in by_strategy:
        by_strategy[strat].sort(key=lambda x: -x[5])

    # First pass: each bot gets exactly min(available, BOT_MAX)
    selected_set = set()
    selected_list = []
    bot_counts = {}

    for strat in sorted(by_strategy.keys()):
        pool = by_strategy[strat]
        take = min(BOT_MAX, len(pool))
        count = 0
        for s in pool[:take]:
            if s[0] not in selected_set:
                selected_set.add(s[0])
                selected_list.append((s[0], s[1], s[5]))
                count += 1
        bot_counts[strat] = count

    print(f"    After bot caps: {len(selected_list):,} lineups")

    # Second pass: fill remaining slots proportionally from bots under cap
    remaining = TARGET - len(selected_list)
    if remaining > 0:
        # Build leftover pools (entries past BOT_MAX that weren't selected)
        leftover = []
        for strat in sorted(by_strategy.keys()):
            pool = by_strategy[strat]
            for s in pool[bot_counts[strat]:]:
                if s[0] not in selected_set:
                    leftover.append(s)

        # Sort leftovers by score desc, fill proportionally (round-robin by bot)
        # to avoid any single bot dominating the remainder
        leftover.sort(key=lambda x: -x[5])
        bot_extra = defaultdict(int)
        extra_max = remaining // n_bots + 1  # max extra per bot

        for s in leftover:
            if remaining <= 0:
                break
            strat = s[1]
            if bot_extra[strat] >= extra_max:
                continue
            if s[0] not in selected_set:
                selected_set.add(s[0])
                selected_list.append((s[0], s[1], s[5]))
                bot_extra[strat] += 1
                bot_counts[strat] = bot_counts.get(strat, 0) + bot_extra[strat]
                remaining -= 1

    selected_list = selected_list[:TARGET]
    scored = None  # free memory

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    strat_counts = defaultdict(int)
    for _, strat, _ in selected_list:
        strat_counts[strat] += 1

    print(f"\n  Final: {len(selected_list):,} lineups")
    print(f"\n  Per-bot representation:")
    print(f"  {'Bot':<20s} {'Count':>8s} {'%':>7s}")
    print(f"  {'-'*20} {'-'*8} {'-'*7}")
    for strat in sorted(strat_counts, key=lambda s: -strat_counts[s]):
        c = strat_counts[strat]
        print(f"  {strat:<20s} {c:>8,} {c/len(selected_list):>7.1%}")

    # ---------------------------------------------------------------
    # Pass 2: Copy selected rows from input to output
    # ---------------------------------------------------------------
    print(f"\n  Pass 2: Writing {len(selected_list):,} rows to output...")
    t2 = time.perf_counter()

    selected_indices = set(idx for idx, _, _ in selected_list)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(INPUT) as fin, open(OUTPUT, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader)
        writer.writerow(header)

        written = 0
        for i, row in enumerate(reader):
            if i in selected_indices:
                writer.writerow(row)
                written += 1

    print(f"    Wrote {written:,} rows in {time.perf_counter() - t2:.1f}s")

    elapsed = time.perf_counter() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"  Output: {OUTPUT}")

    return selected_list


if __name__ == "__main__":
    filter_candidates()

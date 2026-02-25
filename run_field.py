#!/usr/bin/env python3
"""Field CSV Portfolio Selector — pick 150 diverse, high-quality lineups.

Usage:
    python3 run_field.py                              # use default CSV
    python3 run_field.py cognizant_field.csv           # specify input CSV
"""

import csv
import sys
import random
from pathlib import Path

_default_csv = "cognizant_field.csv"
CSV_PATH = Path(__file__).parent / (sys.argv[1] if len(sys.argv) > 1 else _default_csv)
OUT_CSV = CSV_PATH.with_name(CSV_PATH.stem + "_portfolio.csv")
TARGET = 150
MAX_OVERLAP = 3          # max shared players with any already-selected lineup
MAX_EXPOSURE = 0.35      # max % of portfolio any single player can appear in
QUALITY_POOL_SIZE = 10000 # top N by combined rank


def parse_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            if not line or not line[0].strip() or not line[0].strip().isdigit():
                continue
            players = tuple(line[1:7])
            rows.append({
                "num": int(line[0]),
                "players": players,
                "salary": int(line[7]),
                "pts": float(line[8]),
                "roi": float(line[9]),
                "cash": float(line[10]),
                "gmown": float(line[11]),
                "dupes": float(line[12]),
            })
    return rows


def build_quality_pool(rows: list[dict], size: int) -> list[dict]:
    # Rank by Proj ROI % (desc) and Proj Pts (desc), combine ranks
    by_roi = sorted(rows, key=lambda r: r["roi"], reverse=True)
    by_pts = sorted(rows, key=lambda r: r["pts"], reverse=True)

    roi_rank = {id(r): i for i, r in enumerate(by_roi)}
    pts_rank = {id(r): i for i, r in enumerate(by_pts)}

    for r in rows:
        r["combined_rank"] = roi_rank[id(r)] + pts_rank[id(r)]

    pool = sorted(rows, key=lambda r: r["combined_rank"])[:size]
    return pool


def overlap(a: tuple, b: tuple) -> int:
    return len(set(a) & set(b))


def greedy_select(pool: list[dict], target: int, max_overlap: int,
                   max_exposure: float) -> list[dict]:
    from collections import Counter
    random.shuffle(pool)
    selected = []
    player_counts = Counter()
    exposure_cap = int(target * max_exposure)

    for cand in pool:
        # Check pairwise overlap
        if not all(overlap(cand["players"], s["players"]) <= max_overlap for s in selected):
            continue
        # Check per-player exposure cap
        if any(player_counts[p] >= exposure_cap for p in cand["players"]):
            continue
        selected.append(cand)
        for p in cand["players"]:
            player_counts[p] += 1
        if len(selected) == target:
            break
    return selected


def main():
    # Step 1: Parse
    rows = parse_csv(CSV_PATH)
    print(f"Parsed {len(rows):,} lineups")

    # Step 2: Filter dupes
    rows = [r for r in rows if r["dupes"] == 0.0]
    print(f"After dupe filter: {len(rows):,} lineups")

    # Step 3: Quality pool
    pool = build_quality_pool(rows, QUALITY_POOL_SIZE)
    print(f"Quality pool: {len(pool):,} lineups (top by combined ROI+Pts rank)")

    # Step 4: Greedy diversity sampling
    selected = greedy_select(pool, TARGET, MAX_OVERLAP, MAX_EXPOSURE)
    print(f"Selected: {len(selected)} lineups (max overlap ≤{MAX_OVERLAP}, max exposure {MAX_EXPOSURE:.0%})\n")

    # Step 5: Output
    print(f"{'#':>6}  {'Players':<80} {'Sal':>5} {'Pts':>6} {'ROI%':>6}")
    print("-" * 108)
    for lu in selected:
        names = ", ".join(lu["players"])
        print(f"{lu['num']:>6}  {names:<80} {lu['salary']:>5} {lu['pts']:>6.1f} {lu['roi']:>6.1f}")

    # Compute golfer exposure
    from collections import Counter
    golfer_counts = Counter()
    for lu in selected:
        for p in lu["players"]:
            golfer_counts[p] += 1
    exposure = sorted(golfer_counts.items(), key=lambda x: x[1], reverse=True)

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Lineup #", "G", "G", "G", "G", "G", "G",
                         "Salary", "Proj Pts", "Proj ROI %", "Geomean Own %"])
        for lu in selected:
            writer.writerow([lu["num"], *lu["players"],
                             lu["salary"], lu["pts"], lu["roi"], lu["gmown"]])
        # Blank row separator
        writer.writerow([])
        # Golfer exposure summary
        writer.writerow(["GOLFER EXPOSURE"])
        writer.writerow(["Golfer", "Lineups", f"Exposure % (of {len(selected)})"])
        for name, count in exposure:
            writer.writerow([name, count, f"{count / len(selected) * 100:.1f}"])
    print(f"\nWrote {len(selected)} lineups to {OUT_CSV.name}")

    # Summary stats
    avg_pts = sum(r["pts"] for r in selected) / len(selected)
    avg_roi = sum(r["roi"] for r in selected) / len(selected)

    total_overlap, pairs = 0, 0
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            total_overlap += overlap(selected[i]["players"], selected[j]["players"])
            pairs += 1
    avg_overlap = total_overlap / pairs if pairs else 0

    print(f"\n--- Summary ---")
    print(f"Avg Proj Pts:        {avg_pts:.1f}")
    print(f"Avg Proj ROI%:       {avg_roi:.1f}")
    print(f"Avg pairwise overlap: {avg_overlap:.2f} players")


if __name__ == "__main__":
    main()

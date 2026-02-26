#!/usr/bin/env python3
"""Create realistic opponent field + scored candidate pool for run_field.py.

Generates a field CSV with realistic duplication patterns:
  - Optimizer archetypes (MIP solver) get heavy duplication (100s of copies)
  - Near-optimal variations get medium duplication (10s)
  - Recreational/random lineups are mostly unique

Usage:
    python3 create_field.py
    python3 run_field.py          # then select 150 from the field
"""

import csv
import sys
import os
import time
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ROSTER_SIZE, SALARY_CAP, SALARY_FLOOR
from engine import generate_candidates, generate_field

# ── Configuration ──────────────────────────────────────────────────────────

PROJ_CSV = "/Users/rhbot/Downloads/draftkings_main_projections (1).csv"
MIN_PROJ_PTS = 55.0          # Scrub filter: remove players below this projection

# Defaults (overridden by CLI args)
DEFAULT_FIELD_SIZE = 71_000
DEFAULT_CANDIDATE_POOL = 15_000
DEFAULT_OUT_NAME = "cognizant_field.csv"
DEFAULT_LABEL = "Cognizant Classic"


def parse_projections(path):
    """Parse DG projections CSV into player list compatible with engine.py."""
    players = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pts = float(row["total_points"])
            sal = int(row["dk_salary"])
            if pts <= 0 or sal <= 0:
                continue
            players.append({
                "name": row["dk_name"],
                "name_id": f"{row['dk_name']} ({row['dk_id']})",
                "salary": sal,
                "projected_points": pts,
                "std_dev": float(row["std_dev"]),
                "proj_ownership": float(row["projected_ownership"]),
            })
    return players


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create field CSV for run_field.py")
    parser.add_argument("--field-size", type=int, default=DEFAULT_FIELD_SIZE,
                        help=f"Opponent field size (default: {DEFAULT_FIELD_SIZE:,})")
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATE_POOL,
                        help=f"Candidate pool MIP solves (default: {DEFAULT_CANDIDATE_POOL:,})")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_NAME,
                        help=f"Output CSV filename (default: {DEFAULT_OUT_NAME})")
    parser.add_argument("--label", type=str, default=DEFAULT_LABEL,
                        help=f"Contest label for display (default: {DEFAULT_LABEL})")
    args = parser.parse_args()

    FIELD_SIZE = args.field_size
    CANDIDATE_POOL = args.candidates
    OUT_CSV = Path(__file__).parent / args.out

    start = time.time()

    print("=" * 70)
    print(f"  FIELD CREATION — {args.label}")
    print("=" * 70)

    # ── Step 1: Parse & filter projections ──
    print(f"\n  Reading projections...")
    players = parse_projections(PROJ_CSV)
    all_projs = sorted([p["projected_points"] for p in players])
    print(f"  {len(players)} players | Range: {all_projs[0]:.1f} – {all_projs[-1]:.1f} pts")

    removed = [p for p in players if p["projected_points"] < MIN_PROJ_PTS]
    players = [p for p in players if p["projected_points"] >= MIN_PROJ_PTS]
    players.sort(key=lambda p: p["projected_points"], reverse=True)

    print(f"  Scrub filter (floor={MIN_PROJ_PTS:.0f} pts): removed {len(removed)} players")
    if removed:
        removed.sort(key=lambda p: p["projected_points"])
        names = ", ".join(p["name"] for p in removed)
        print(f"  Cut: {names}")
    print(f"  Active pool: {len(players)} players")

    print(f"\n  {'Player':<30} {'Sal':>7} {'Proj':>6} {'Own%':>6}")
    print(f"  {'-'*30} {'-'*7} {'-'*6} {'-'*6}")
    for p in players[:15]:
        print(f"  {p['name']:<30} ${p['salary']:>6,} {p['projected_points']:>6.1f} {p['proj_ownership']:>5.1f}%")
    print(f"  ... {len(players)} total")

    # ── Step 2: Generate opponent field ──
    print(f"\n{'='*70}")
    print(f"  OPPONENT FIELD ({FIELD_SIZE:,} entries)")
    print(f"{'='*70}")

    field = generate_field(players, FIELD_SIZE)

    # Duplication analysis
    opp_counter = Counter(tuple(sorted(lu)) for lu in field)
    n_unique = len(opp_counter)
    dupe_counts = list(opp_counter.values())
    top10 = opp_counter.most_common(10)

    print(f"\n  Duplication summary:")
    print(f"    {len(field):,} total entries → {n_unique:,} unique lineups")
    print(f"    1x (unique):   {sum(1 for c in dupe_counts if c == 1):>6,}")
    print(f"    2–10x:         {sum(1 for c in dupe_counts if 2 <= c <= 10):>6,}")
    print(f"    11–50x:        {sum(1 for c in dupe_counts if 11 <= c <= 50):>6,}")
    print(f"    51–100x:       {sum(1 for c in dupe_counts if 51 <= c <= 100):>6,}")
    print(f"    100–500x:      {sum(1 for c in dupe_counts if 100 < c <= 500):>6,}")
    print(f"    500+x:         {sum(1 for c in dupe_counts if c > 500):>6,}")
    print(f"\n  Top-10 most duplicated lineups:")
    for rank, (key, count) in enumerate(top10, 1):
        names = ", ".join(players[i]["name"] for i in sorted(key, key=lambda j: -players[j]["salary"]))
        print(f"    #{rank} ({count:,}x): {names}")

    # ── Step 3: Generate OUR candidate pool ──
    print(f"\n{'='*70}")
    print(f"  OUR CANDIDATE POOL")
    print(f"{'='*70}")

    candidates = generate_candidates(
        players, pool_size=CANDIDATE_POOL,
        noise_scale=0.15, min_proj_pct=0.85
    )
    print(f"  {len(candidates):,} unique candidates")

    # ── Step 4: Score candidates ──
    print(f"\n  Scoring {len(candidates):,} candidates...")

    own_arr = np.array([p["proj_ownership"] for p in players])
    pts_arr = np.array([p["projected_points"] for p in players])
    sal_arr = np.array([p["salary"] for p in players])

    scored = []
    for i, cand in enumerate(candidates):
        key = tuple(sorted(cand))
        pts = float(pts_arr[list(cand)].sum())
        sal = int(sal_arr[list(cand)].sum())

        owns = np.maximum(own_arr[list(cand)], 0.01)
        gmown = float(np.exp(np.log(owns).mean()))

        dupes = opp_counter.get(key, 0)

        # Names sorted by salary descending
        idxs_by_sal = sorted(cand, key=lambda j: -sal_arr[j])
        names = [players[j]["name"] for j in idxs_by_sal]

        scored.append({
            "num": i + 1,
            "names": names,
            "salary": sal,
            "pts": round(pts, 1),
            "gmown": round(gmown, 2),
            "dupes": dupes,
        })

    # Convert pts → ROI proxy (centered at 0)
    pts_vals = np.array([s["pts"] for s in scored])
    mean_pts = float(pts_vals.mean())
    sorted_pts = np.sort(pts_vals)

    for s in scored:
        s["roi"] = round((s["pts"] / mean_pts - 1) * 100, 1)
        s["cash"] = round(float(np.searchsorted(sorted_pts, s["pts"])) / len(scored) * 100, 1)

    # ── Step 5: Output CSV ──
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Lineup #", "G", "G", "G", "G", "G", "G",
                         "Salary", "Proj Pts", "Proj ROI %", "Proj Cash %",
                         "Geomean Own %", "Proj Dupes"])
        for s in scored:
            writer.writerow([s["num"], *s["names"], s["salary"], s["pts"],
                            s["roi"], s["cash"], s["gmown"], s["dupes"]])

    zero_dupes = sum(1 for s in scored if s["dupes"] == 0)
    avg_pts = float(np.mean([s["pts"] for s in scored]))
    avg_roi = float(np.mean([s["roi"] for s in scored]))

    print(f"\n{'='*70}")
    print(f"  OUTPUT: {OUT_CSV.name}")
    print(f"{'='*70}")
    print(f"  {len(scored):,} candidates scored")
    print(f"  {zero_dupes:,} with 0 projected dupes ({zero_dupes/len(scored)*100:.1f}%)")
    print(f"  Avg Proj Pts:  {avg_pts:.1f}")
    print(f"  Avg Proxy ROI: {avg_roi:+.1f}%")
    print(f"  Elapsed: {time.time()-start:.0f}s")
    print(f"{'='*70}")
    print(f"\n  Next: python3 run_field.py")


if __name__ == "__main__":
    main()

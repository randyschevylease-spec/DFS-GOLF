#!/usr/bin/env python3
"""Generate opponent field for a single FC tournament and export to CSV.

Outputs every lineup in the simulated opponent field with:
- 6 players, salary, projected pts, projected ROI, projected cash rate,
  geomean ownership, and projected dupes in the full contest field.
"""
import sys, os, csv
import numpy as np
from scipy.stats import gmean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ROSTER_SIZE, SALARY_CAP, MAX_EXPOSURE, BASE_CORRELATION
from engine import (generate_field, _get_sigma, empirical_sigma_from_projection)
from run_all import build_payout_lookup
from backtest_all import build_players_from_fc, load_fc_payout_table

# Masters 2024
EVENT_ID = 14
YEAR = 2024
EVENT_NAME = "Masters Tournament 2024"

OPP_FIELD = 50000
N_SIMS = 3000  # enough for stable per-lineup stats
SEED = 42

print(f"{'='*70}")
print(f"  {EVENT_NAME} — Opponent Field Analysis")
print(f"{'='*70}")

# Step 1: Build players from FC data
print(f"\n  Loading FC data...")
players = build_players_from_fc(EVENT_ID, YEAR)
if not players:
    print("  ERROR: No FC data found")
    sys.exit(1)
print(f"  Players: {len(players)}")

# Step 2: Generate opponent field
print(f"\n  Generating {OPP_FIELD:,} opponent lineups...")
opponents = generate_field(players, OPP_FIELD, seed=SEED)
n_lineups = len(opponents)
print(f"  Generated: {n_lineups:,}")

# Step 3: Load real payout structure
fc_payouts = load_fc_payout_table(EVENT_ID, YEAR)
if fc_payouts:
    payout_table, entry_fee, prize_pool, contest_field_size = fc_payouts
    print(f"  Real FC payouts: fee=${entry_fee:.0f} pool=${prize_pool:,.0f} field={contest_field_size:,}")
else:
    entry_fee = 10
    contest_field_size = 470000
    prize_pool = entry_fee * contest_field_size * 0.85
    from backtest_all import build_synthetic_payout_table
    payout_table, entry_fee, prize_pool = build_synthetic_payout_table(contest_field_size, entry_fee)
    print(f"  Synthetic payouts: fee=${entry_fee} field={contest_field_size:,}")

payout_by_pos = build_payout_lookup(payout_table, contest_field_size)
scale = contest_field_size / n_lineups

# Step 4: Compute static per-lineup metrics (no simulation needed)
print(f"\n  Computing lineup metrics...")

n_players = len(players)
own_fractions = np.array([p["proj_ownership"] / 100.0 for p in players])
proj_pts_arr = np.array([p["projected_points"] for p in players])
salary_arr = np.array([p["salary"] for p in players])

# Dupe counting
print(f"  Counting duplicates...")
dupe_map = {}
for lu in opponents:
    key = tuple(sorted(lu))
    dupe_map[key] = dupe_map.get(key, 0) + 1

# Step 5: Monte Carlo — score ALL opponent lineups against each other
print(f"\n  Monte Carlo: scoring {n_lineups:,} lineups × {N_SIMS:,} sims...")

# Build binary lineup matrix
matrix = np.zeros((n_lineups, n_players), dtype=np.float32)
for i, lu in enumerate(opponents):
    for idx in lu:
        matrix[i, idx] = 1.0

means = np.array([p["projected_points"] for p in players], dtype=np.float64)
sigmas = np.array([_get_sigma(p) for p in players], dtype=np.float64)

# Covariance
cov = np.outer(sigmas, sigmas) * BASE_CORRELATION
np.fill_diagonal(cov, sigmas ** 2)
try:
    L = np.linalg.cholesky(cov)
except np.linalg.LinAlgError:
    cov += np.eye(n_players) * 1.0
    L = np.linalg.cholesky(cov)

# Streaming accumulators (no need to store full payout matrix)
total_payouts = np.zeros(n_lineups, dtype=np.float64)
cash_counts = np.zeros(n_lineups, dtype=np.float64)

rng = np.random.default_rng(SEED)

for sim in range(N_SIMS):
    if (sim + 1) % 500 == 0:
        print(f"    Sim {sim+1}/{N_SIMS}...")

    Z = rng.standard_normal(n_players)
    scores = means + L @ Z
    scores = np.maximum(scores, 0.0).astype(np.float32)

    lu_scores = matrix @ scores

    # Rank
    order = np.argsort(-lu_scores)
    positions = np.empty(n_lineups, dtype=np.int32)
    positions[order] = np.arange(1, n_lineups + 1)

    # Scale to contest field size
    scaled = np.rint(positions * scale).astype(np.int32)
    np.clip(scaled, 1, contest_field_size, out=scaled)

    # Assign payouts
    sim_payouts = payout_by_pos[scaled]

    total_payouts += sim_payouts
    cash_counts += (sim_payouts > 0)

mean_payouts = total_payouts / N_SIMS
cash_rate = cash_counts / N_SIMS * 100
roi = (mean_payouts - entry_fee) / entry_fee * 100

print(f"  Simulation complete.")
print(f"  Field avg ROI: {roi.mean():+.1f}%  |  Avg cash rate: {cash_rate.mean():.1f}%")

# Step 6: Build per-lineup rows
print(f"\n  Building CSV rows...")

rows = []
for i, lu in enumerate(opponents):
    lu_sorted = sorted(lu, key=lambda idx: players[idx]["salary"], reverse=True)
    names = [players[idx]["name"] for idx in lu_sorted]

    total_sal = sum(salary_arr[idx] for idx in lu)
    total_proj = sum(proj_pts_arr[idx] for idx in lu)

    ownerships = [max(own_fractions[idx], 0.001) for idx in lu]
    geomean_own = gmean(ownerships) * 100

    lu_key = tuple(sorted(lu))
    sample_dupes = dupe_map.get(lu_key, 1) - 1  # subtract self
    proj_dupes = round(sample_dupes * (contest_field_size / n_lineups), 1)

    rows.append((
        i + 1,
        *names,
        total_sal,
        round(total_proj, 1),
        round(roi[i], 1),
        round(cash_rate[i], 1),
        round(geomean_own, 2),
        proj_dupes,
    ))

# Step 7: Write CSV
base = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base, "masters_2024_field.csv")

print(f"  Writing {len(rows):,} lineups to CSV...")

with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "Lineup #",
        "G", "G", "G", "G", "G", "G",
        "Salary",
        "Proj Pts",
        "Proj ROI %",
        "Proj Cash %",
        "Geomean Own %",
        "Proj Dupes",
    ])
    for row in rows:
        w.writerow(row)

    # Summary stats at bottom
    w.writerow([])
    w.writerow(["FIELD SUMMARY"])
    w.writerow(["Metric", "Value"])
    w.writerow(["Total Lineups", n_lineups])
    w.writerow(["Contest Field Size", contest_field_size])
    w.writerow(["Entry Fee", f"${entry_fee}"])
    w.writerow(["Prize Pool", f"${prize_pool:,.0f}"])

    all_proj = [r[8] for r in rows]
    all_sal = [r[7] for r in rows]
    all_roi = [r[9] for r in rows]
    all_cash = [r[10] for r in rows]
    all_geomean = [r[11] for r in rows]
    all_dupes = [r[12] for r in rows]

    w.writerow([])
    w.writerow(["Avg Salary", round(np.mean(all_sal))])
    w.writerow(["Avg Proj Pts", round(np.mean(all_proj), 1)])
    w.writerow(["Avg ROI", f"{np.mean(all_roi):+.1f}%"])
    w.writerow(["Avg Cash Rate", f"{np.mean(all_cash):.1f}%"])
    w.writerow(["Avg Geomean Own", f"{np.mean(all_geomean):.2f}%"])
    w.writerow(["Lineups with Dupes", sum(1 for d in all_dupes if d > 0)])
    w.writerow(["Max Dupes", max(all_dupes)])

    w.writerow([])
    w.writerow(["Salary Distribution"])
    for threshold in [50000, 49500, 49000, 48500, 48000]:
        count = sum(1 for s in all_sal if s >= threshold)
        w.writerow([f">= ${threshold:,}", f"{count:,} ({count/n_lineups*100:.1f}%)"])
    for threshold in [48000, 47000, 46000, 45000]:
        count = sum(1 for s in all_sal if s < threshold)
        w.writerow([f"< ${threshold:,}", f"{count:,} ({count/n_lineups*100:.1f}%)"])

    w.writerow([])
    w.writerow(["Proj Pts Distribution"])
    pts_sorted = sorted(all_proj, reverse=True)
    for pct in [1, 5, 10, 25, 50, 75, 90]:
        idx = int(len(pts_sorted) * pct / 100)
        w.writerow([f"Top {pct}%", round(pts_sorted[min(idx, len(pts_sorted)-1)], 1)])

    # Player ownership in generated field
    w.writerow([])
    w.writerow(["FIELD OWNERSHIP (generated)"])
    w.writerow(["Player", "Salary", "Proj Pts", "Std Dev", "Target Own %", "Actual Own %", "Delta"])
    player_counts = np.zeros(n_players)
    for lu in opponents:
        for idx in lu:
            player_counts[idx] += 1
    actual_own = player_counts / n_lineups * 100
    for idx in np.argsort(-actual_own):
        p = players[idx]
        target = p["proj_ownership"]
        actual = actual_own[idx]
        if actual < 0.01 and target < 0.1:
            continue
        w.writerow([
            p["name"], p["salary"], p["projected_points"],
            round(_get_sigma(p), 1),
            round(target, 1), round(actual, 1),
            round(actual - target, 1),
        ])

print(f"\n  CSV exported: {csv_path}")
print(f"  {len(rows):,} opponent lineups with full metrics")
print(f"{'='*70}")

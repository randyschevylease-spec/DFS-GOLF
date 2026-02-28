#!/usr/bin/env python3
"""Classic Showdown (Single-Round) DFS Pipeline.

Builds lineups for DraftKings Classic Showdown events (single-round scoring).
Uses wave-aware correlation, archetype-based opponent field generation with
showdown salary floors, and Cholesky-based Monte Carlo simulation.

Key differences from multi-round classic pipeline:
  - No cut line / bimodal mixture distribution
  - Lower salary floors for opponent field (showdown fields are looser)
  - Std devs derived from DG skill decomposition (single-round scaling)
  - Ownership computed independently when not in projections CSV

Usage:
    python3 -u run_showdown.py --csv projections.csv --contest 188424004
    python3 -u run_showdown.py --csv projections.csv --contest 188424004 --sims 50000
    python3 -u run_showdown.py --csv projections.csv --contest 188424004 --entries 150
"""
import sys
import os
import csv
import time
import argparse
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ROSTER_SIZE, SALARY_CAP
from dk_contests import fetch_contest
from engine import generate_candidates, simulate_contest, _get_sigma
from field_generator import (generate_field as generate_field_archetypes,
                             field_to_index_lists)
from portfolio_optimizer import optimize_portfolio_greedy


# ── Showdown-specific defaults ──────────────────────────────────────────────

SHOWDOWN_SALARY_FLOORS = {
    "chalk": 45000,
    "content": 44000,
    "optimizer": 44000,
    "sharp": 42000,
    "random": 40000,
}

# DK points per strokes-gained (empirically calibrated)
DK_PTS_PER_SG = 11.3
# Single-round std_dev = full-tournament SG std * conversion / sqrt(rounds)
SINGLE_ROUND_DIVISOR = 2.0


def parse_projections(csv_path):
    """Parse showdown projections CSV into player dicts."""
    players = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            proj = float(row.get("total_points", 0))
            if proj <= 0:
                continue
            players.append({
                "name": row.get("dk_name", row.get("datagolf_name", "")),
                "datagolf_name": row.get("datagolf_name", ""),
                "salary": int(row.get("dk_salary", 0)),
                "projected_points": proj,
                "dk_id": str(row.get("dk_id", "")),
                "value": float(row.get("value", 0)),
                "wave": 1 if row.get("morning_wave", "").upper() in ("TRUE", "1", "YES") else 0,
                "position": row.get("position", ""),
            })
    return players


def derive_std_devs(players):
    """Derive single-round std devs from DG skill decomposition.

    Falls back to projection-based estimate if DG data unavailable.
    """
    try:
        from datagolf_client import get_predictions
        preds = get_predictions()
        if preds and "baseline" in preds:
            # Build name lookup (Last, First -> First Last)
            dg_std = {}
            for entry in preds["baseline"]:
                name = entry.get("player_name", "")
                if "," in name:
                    parts = name.split(",", 1)
                    name = f"{parts[1].strip()} {parts[0].strip()}"
                sg_std = entry.get("std_dev")
                if sg_std:
                    dk_std = float(sg_std) * DK_PTS_PER_SG / SINGLE_ROUND_DIVISOR
                    dg_std[name] = dk_std

            matched = 0
            for p in players:
                for lookup_name in [p["name"], p["datagolf_name"]]:
                    if lookup_name in dg_std:
                        p["std_dev"] = dg_std[lookup_name]
                        matched += 1
                        break
                else:
                    p["std_dev"] = p["projected_points"] * 0.40
            print(f"  Std devs: {matched}/{len(players)} from DG decomposition, "
                  f"{len(players)-matched} fallback")
            return
    except Exception as e:
        print(f"  DG API unavailable ({e}), using projection-based std devs")

    for p in players:
        p["std_dev"] = p["projected_points"] * 0.40


def compute_ownership(players):
    """Compute projected ownership from salary/projection/position blend.

    Used when projected ownership isn't available in the CSV.
    """
    n = len(players)
    sals = np.array([p["salary"] for p in players], dtype=np.float64)
    projs = np.array([p["projected_points"] for p in players], dtype=np.float64)

    # Normalize to [0, 1]
    sal_norm = (sals - sals.min()) / max(sals.max() - sals.min(), 1)
    proj_norm = (projs - projs.min()) / max(projs.max() - projs.min(), 1)

    # Position bonus (leaders get more ownership)
    pos_score = np.zeros(n)
    for i, p in enumerate(players):
        pos = p.get("position", "")
        try:
            pos_num = int(pos.replace("T", ""))
            pos_score[i] = max(0, 1.0 - pos_num / 70.0)
        except (ValueError, AttributeError):
            pos_score[i] = 0.3

    # Blended score
    blend = 0.50 * proj_norm + 0.25 * sal_norm + 0.25 * pos_score
    temperature = 0.8
    exp_blend = np.exp(blend / temperature)
    ownership = exp_blend / exp_blend.sum()

    # Scale to ~600% total (6-player rosters, ~100% per slot)
    ownership *= ROSTER_SIZE * 100

    for i, p in enumerate(players):
        p["proj_ownership"] = float(ownership[i])

    print(f"  Ownership computed: {ownership.min():.1f}% – {ownership.max():.1f}% "
          f"(total {ownership.sum():.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Classic Showdown DFS Pipeline")
    parser.add_argument("--csv", required=True, help="Projections CSV path")
    parser.add_argument("--contest", required=True, type=int, help="DK contest ID")
    parser.add_argument("--entries", type=int, default=150, help="Max entries (default: 150)")
    parser.add_argument("--sims", type=int, default=50000, help="Monte Carlo sims (default: 50000)")
    parser.add_argument("--pool-size", type=int, default=20000, help="MIP solves for candidates")
    parser.add_argument("--diversity", type=float, default=0.4, help="Portfolio diversity weight")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    t_start = time.time()

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Load projections and contest data
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 1: Load projections and contest data")
    print(f"{'='*70}")

    players = parse_projections(args.csv)
    print(f"  Players: {len(players)}")
    print(f"  Projection range: {min(p['projected_points'] for p in players):.1f} – "
          f"{max(p['projected_points'] for p in players):.1f}")
    print(f"  Salary range: ${min(p['salary'] for p in players):,} – "
          f"${max(p['salary'] for p in players):,}")

    contest = fetch_contest(args.contest)
    entry_fee = contest["entry_fee"]
    field_size = contest["entries"]
    max_entries = min(args.entries, contest.get("max_entries_per_user", args.entries))
    payout_table = contest["payouts"]

    print(f"  Contest: {contest['name']}")
    print(f"  Entry fee: ${entry_fee} | Field: {field_size:,} | Max entries: {max_entries}")
    print(f"  Prize pool: ${contest['prize_pool']:,} | 1st: ${contest['first_place_prize']:,}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Derive std devs and ownership
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 2: Derive std devs and ownership")
    print(f"{'='*70}")

    derive_std_devs(players)
    compute_ownership(players)

    waves = [p["wave"] for p in players]

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Generate opponent field
    # ══════════════════════════════════════════════════════════════════
    n_opponents = field_size - max_entries
    print(f"\n{'='*70}")
    print(f"  STEP 3: Generate opponent field ({n_opponents:,} lineups)")
    print(f"{'='*70}")

    t_field = time.time()
    field = generate_field_archetypes(
        players, n_opponents,
        seed=args.seed,
        min_salary_map=SHOWDOWN_SALARY_FLOORS,
    )
    opponents = field_to_index_lists(field)
    print(f"  Field generated in {time.time()-t_field:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Generate candidate lineups
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 4: Generate candidates ({args.pool_size:,} MIP solves)")
    print(f"{'='*70}")

    t_cand = time.time()
    candidates = generate_candidates(
        players,
        pool_size=args.pool_size,
        candidate_exposure_cap=1.0,
        ceiling_weight=0.0,
        min_proj_pct=0.0,
        seed=args.seed,
    )
    print(f"  Candidates: {len(candidates):,} unique in {time.time()-t_cand:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Monte Carlo simulation (Cholesky-based)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 5: Simulate contest ({args.sims:,} sims)")
    print(f"{'='*70}")

    t_sim = time.time()
    payouts, roi = simulate_contest(
        candidates, opponents, players, payout_table, entry_fee,
        n_sims=args.sims, waves=waves,
    )
    sim_elapsed = time.time() - t_sim
    print(f"  Simulation complete in {sim_elapsed:.1f}s")

    n_positive = (roi > 0).sum()
    print(f"  Positive ROI candidates: {n_positive:,}/{len(candidates):,}")
    print(f"  Top candidate ROI: {roi.max():.1f}%")
    print(f"  Median candidate ROI: {np.median(roi):.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: Pre-filter candidates
    # ══════════════════════════════════════════════════════════════════
    TOP_CANDIDATES = max(max_entries * 10, 3000)
    top_idx = np.argsort(-roi)[:TOP_CANDIDATES]
    payouts_filtered = payouts[top_idx]
    candidates_filtered = [candidates[i] for i in top_idx]
    roi_filtered = roi[top_idx]

    print(f"\n  Pre-filtered to top {TOP_CANDIDATES:,} candidates")
    print(f"  ROI range: {roi_filtered[-1]:.1f}% – {roi_filtered[0]:.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: Portfolio optimization
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 7: Portfolio optimization ({max_entries} lineups)")
    print(f"{'='*70}")

    t_port = time.time()
    result = optimize_portfolio_greedy(
        payouts_filtered, entry_fee, max_entries,
        candidates_filtered, len(players),
        diversity_weight=args.diversity,
        waves=waves,
    )
    port_elapsed = time.time() - t_port
    print(f"  Portfolio optimization: {port_elapsed:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════
    sel = result.selected_indices
    sel_payouts = payouts_filtered[sel]
    portfolio_max = sel_payouts.max(axis=0)
    portfolio_cost = entry_fee * len(sel)
    portfolio_roi = (portfolio_max.mean() - portfolio_cost) / portfolio_cost * 100

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Lineups: {len(sel)}")
    print(f"  Portfolio ROI: {portfolio_roi:+.1f}%")
    print(f"  Mean best payout: ${portfolio_max.mean():.2f}")
    print(f"  Investment: ${portfolio_cost:.0f}")
    print(f"  Cash rate: {(portfolio_max > 0).mean()*100:.1f}%")

    # Exposure
    player_counts = Counter()
    for si in sel:
        for idx in candidates_filtered[si]:
            player_counts[idx] += 1

    print(f"\n  Top exposures:")
    for idx, count in player_counts.most_common(15):
        pct = count / len(sel) * 100
        print(f"    {players[idx]['name']:<25} {pct:5.1f}% ({count}/{len(sel)})")
    print(f"  Unique players: {len(player_counts)}/{len(players)}")

    total_elapsed = time.time() - t_start
    print(f"\n  Total pipeline: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    # ══════════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════════
    contest_name = contest["name"].replace(" ", "_").replace("/", "-")[:30]

    # DK upload CSV
    lineup_file = f"lineups_showdown_{args.contest}.csv"
    with open(lineup_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["G"] * ROSTER_SIZE)
        for si in sel:
            w.writerow([players[idx]["dk_id"] for idx in candidates_filtered[si]])
    print(f"\n  Exported {lineup_file}")

    # Detail CSV
    mean_pay = payouts_filtered.mean(axis=1)
    detail_roi = (mean_pay - entry_fee) / entry_fee * 100
    detail_file = f"showdown_{args.contest}_detail.csv"
    with open(detail_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lineup_num", "players", "salary", "projection",
                     "mean_payout", "roi_pct"])
        for rank, si in enumerate(sel, 1):
            lu = candidates_filtered[si]
            w.writerow([
                rank,
                " | ".join(players[i]["name"] for i in lu),
                sum(players[i]["salary"] for i in lu),
                f"{sum(players[i]['projected_points'] for i in lu):.1f}",
                f"{mean_pay[si]:.2f}",
                f"{detail_roi[si]:.1f}",
            ])
    print(f"  Exported {detail_file}")

    # Upload CSV with exposure summary
    upload_file = f"showdown_{args.contest}_upload.csv"
    with open(upload_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["G"] * ROSTER_SIZE)
        for si in sel:
            w.writerow([players[idx]["dk_id"] for idx in candidates_filtered[si]])
        w.writerow([])
        w.writerow(["EXPOSURE SUMMARY", "", "", "", "", ""])
        w.writerow(["Player", "Salary", "Projection", "Count", "Exposure %", "DK ID"])
        exposure_sorted = sorted(player_counts.items(), key=lambda x: -x[1])
        for idx, count in exposure_sorted:
            w.writerow([players[idx]["name"], f"${players[idx]['salary']:,}",
                        f"{players[idx]['projected_points']:.1f}",
                        count, f"{count/len(sel)*100:.1f}%", players[idx]["dk_id"]])
        for i in range(len(players)):
            if i not in player_counts:
                w.writerow([players[i]["name"], f"${players[i]['salary']:,}",
                            f"{players[i]['projected_points']:.1f}",
                            0, "0.0%", players[i]["dk_id"]])
        w.writerow([])
        w.writerow(["PORTFOLIO STATS", "", "", "", "", ""])
        w.writerow(["Lineups", len(sel), "", "", "", ""])
        w.writerow(["Investment", f"${portfolio_cost:.0f}", "", "", "", ""])
        w.writerow(["Portfolio ROI", f"{portfolio_roi:+.1f}%", "", "", "", ""])
        w.writerow(["Mean Best Payout", f"${portfolio_max.mean():.2f}", "", "", "", ""])
        w.writerow(["Unique Players", f"{len(player_counts)}/{len(players)}", "", "", "", ""])
        w.writerow(["Sim Time", f"{sim_elapsed:.1f}s", "", "", "", ""])
    print(f"  Exported {upload_file}")


if __name__ == "__main__":
    main()

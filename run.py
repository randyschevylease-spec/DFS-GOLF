#!/usr/bin/env python3
"""DFS Golf — Clean 3-Step Contest Simulation.

Step 1: Generate contest field from DataGolf projected ownership
Step 2: Calculate ROI for every candidate lineup against that field
Step 3: Select the best portfolio of N lineups via marginal E[max] selection

Usage:
    python run.py --contest 188100564               # Full run with real DK payout table
    python run.py --contest 188100564 --lineups 20  # Select 20 lineups
    python run.py --list-contests                   # Browse DK golf contests
    python run.py --contest 188100564 --sheets      # Export to Google Sheets
"""
import sys
import os
import csv
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ROSTER_SIZE, SALARY_CAP, NUM_LINEUPS, CVAR_LAMBDA

MAX_EXPOSURE = 1.0  # every player eligible for every lineup slot
from datagolf_client import get_fantasy_projections, get_predictions, find_current_event
from dk_salaries import find_latest_csv, parse_dk_csv, match_players_exact  # kept for optional CSV override
from dk_contests import fetch_contest, fetch_golf_contests, format_contest_list
from engine import generate_field, generate_candidates, simulate_contest, select_portfolio
from google_sheets import export_to_sheets


def main():
    parser = argparse.ArgumentParser(description="DFS Golf — 3-Step Contest Simulator")
    parser.add_argument("--contest", type=str, required=False,
                        help="DraftKings contest ID (required for real payout table)")
    parser.add_argument("--list-contests", action="store_true",
                        help="List current DraftKings golf contests")
    parser.add_argument("--lineups", type=int, default=None,
                        help="Number of lineups to select (default: contest max entries)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to DraftKings salary CSV")
    parser.add_argument("--candidates", type=int, default=5000,
                        help="Candidate pool size (default: 5000)")
    parser.add_argument("--sims", type=int, default=10000,
                        help="Monte Carlo simulations (default: 10000)")
    parser.add_argument("--field-size", type=int, default=None,
                        help="Opponent field size (default: from contest)")
    parser.add_argument("--sheets", action="store_true",
                        help="Export to Google Sheets")
    parser.add_argument("--cvar-lambda", type=float, default=None,
                        help=f"CVaR tail-risk penalty (0=pure upside, 0.5=balanced, default: {CVAR_LAMBDA})")
    args = parser.parse_args()

    # ── List contests ──
    if args.list_contests:
        print("=" * 70)
        print("  CURRENT DRAFTKINGS GOLF CONTESTS")
        print("=" * 70)
        try:
            contests = fetch_golf_contests()
            print(f"\n{format_contest_list(contests)}")
            print(f"\n  {len(contests)} contest(s) found")
            print(f"  Use: python run.py --contest <ID>")
        except Exception as e:
            print(f"\n  ERROR: {e}")
        print(f"{'='*70}\n")
        return

    if not args.contest:
        parser.print_help()
        print("\n  ERROR: --contest is required. Use --list-contests to find contest IDs.")
        return

    start_time = time.time()

    print("=" * 70)
    print("  DFS GOLF — 3-STEP CONTEST SIMULATION")
    print("=" * 70)

    # ── Fetch contest details (real payout table) ──
    print(f"\n  Fetching contest {args.contest}...")
    profile = fetch_contest(args.contest)
    entry_fee = profile["entry_fee"]
    field_size = args.field_size or profile["max_entries"]
    n_lineups = args.lineups or profile["max_entries_per_user"]
    payout_table = profile["payouts"]  # [(min_pos, max_pos, prize), ...]

    print(f"  Contest: {profile['name']}")
    print(f"  Entry Fee: ${entry_fee} | Field: {field_size:,} | Max Entries: {profile['max_entries_per_user']}")
    print(f"  Prize Pool: ${profile['prize_pool']:,.0f} | 1st: ${profile['first_place_prize']:,.0f}")
    print(f"  Payout Spots: {profile['payout_spots']:,} ({profile['payout_spots']/field_size*100:.1f}% of field)")
    print(f"  Selecting: {n_lineups} lineups")

    # ── Fetch DataGolf projections ──
    print(f"\n  Fetching DataGolf projections...")
    fantasy_data = get_fantasy_projections()
    dg_projections = fantasy_data.get("projections", []) if isinstance(fantasy_data, dict) else fantasy_data
    event_name = fantasy_data.get("event_name", "Unknown") if isinstance(fantasy_data, dict) else "Unknown"
    print(f"  Event: {event_name}")
    print(f"  Players: {len(dg_projections)}")

    # Event info
    try:
        predictions = get_predictions()
        event_info = find_current_event(predictions)
    except Exception:
        event_info = {"event_name": event_name, "event_id": None, "course": "Unknown"}

    # ── Build player list directly from DataGolf projections ──
    # DG fantasy-projection-defaults already includes DK salary, ownership,
    # site_name_id (DK upload format), and projections — no CSV needed.
    players = []
    for dg in dg_projections:
        salary = dg.get("salary", 0)
        proj_pts = dg.get("proj_points_total", 0)
        if not salary or salary <= 0 or proj_pts <= 0:
            continue

        # Flip "Last, First" → "First Last"
        raw_name = dg.get("player_name", "")
        if ", " in raw_name:
            parts = raw_name.split(", ", 1)
            name = f"{parts[1]} {parts[0]}"
        else:
            name = raw_name

        players.append({
            "name": name,
            "name_id": dg.get("site_name_id", name),
            "salary": salary,
            "projected_points": round(proj_pts, 2),
            "std_dev": dg.get("std_dev") or 0,
            "proj_ownership": dg.get("proj_ownership") or 0,
            "value": round(proj_pts / (salary / 1000), 2),
        })

    players.sort(key=lambda p: p["projected_points"], reverse=True)
    print(f"  Usable players: {len(players)}")

    # Synthesize ownership from projections if DG hasn't published it yet
    has_ownership = any(p["proj_ownership"] > 0 for p in players)
    if not has_ownership:
        print(f"  ⚠ DG ownership not yet available — synthesizing from projections")
        import numpy as _np
        projs = _np.array([p["projected_points"] for p in players])
        # Softmax-style: higher projection = higher ownership
        exp_projs = _np.exp((projs - projs.mean()) / max(projs.std(), 1))
        synth_own = exp_projs / exp_projs.sum() * 100 * ROSTER_SIZE
        for i, p in enumerate(players):
            p["proj_ownership"] = round(float(synth_own[i]), 2)

    # Print player board
    print(f"\n  {'Player':<25} {'Salary':>8} {'Proj':>7} {'Own%':>6} {'StdDv':>6} {'Value':>7}")
    print(f"  {'-'*25} {'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*7}")
    for p in players[:20]:
        print(f"  {p['name']:<25} ${p['salary']:>7,} {p['projected_points']:>7.1f} "
              f"{p['proj_ownership']:>5.1f}% {p['std_dev']:>6.1f} {p['value']:>7.2f}")
    print(f"  ... {len(players)} total players")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Generate Contest Field
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 1: Generating {field_size:,} opponent lineups from DG ownership")
    print(f"{'='*70}")

    opponents = generate_field(players, field_size)
    print(f"  Generated {len(opponents):,} opponent lineups")

    # Verify ownership calibration
    n = len(players)
    counts = [0] * n
    for lu in opponents:
        for idx in lu:
            counts[idx] += 1
    print(f"\n  Ownership calibration (top 10):")
    print(f"  {'Player':<25} {'DG Own%':>8} {'Sim Own%':>9}")
    print(f"  {'-'*25} {'-'*8} {'-'*9}")
    top_own = sorted(range(n), key=lambda i: players[i]["proj_ownership"], reverse=True)[:10]
    for i in top_own:
        sim_own = counts[i] / len(opponents) * 100
        print(f"  {players[i]['name']:<25} {players[i]['proj_ownership']:>7.1f}% {sim_own:>8.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Generate Candidates + Simulate ROI
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 2: Generate candidates + simulate ROI")
    print(f"{'='*70}")

    print(f"  Generating {args.candidates:,} candidate lineups...")
    candidates = generate_candidates(players, pool_size=args.candidates)
    print(f"  Unique candidates: {len(candidates):,}")

    payouts, roi = simulate_contest(
        candidates, opponents, players, payout_table, entry_fee, n_sims=args.sims,
    )

    print(f"\n  Candidate ROI distribution:")
    print(f"    Mean:   {roi.mean():+.1f}%")
    print(f"    Median: {float(sorted(roi)[len(roi)//2]):+.1f}%")
    print(f"    Best:   {roi.max():+.1f}%")
    print(f"    Worst:  {roi.min():+.1f}%")
    print(f"    +EV candidates: {(roi > 0).sum()}/{len(roi)}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Select Portfolio
    # ══════════════════════════════════════════════════════════════════
    cvar_lam = args.cvar_lambda if args.cvar_lambda is not None else CVAR_LAMBDA
    print(f"\n{'='*70}")
    print(f"  STEP 3: Select best {n_lineups} lineups (marginal E[max] + CVaR penalty)")
    print(f"  Max exposure: {int(MAX_EXPOSURE * 100)}% ({int(n_lineups * MAX_EXPOSURE)} appearances)  "
          f"CVaR λ={cvar_lam:.2f}")
    print(f"{'='*70}")

    selected = select_portfolio(payouts, entry_fee, n_lineups, candidates,
                                n_players=len(players), cvar_lambda=cvar_lam)

    # Build output lineups
    lineups = []
    for sel_idx in selected:
        player_indices = candidates[sel_idx]
        lineup = [players[i].copy() for i in player_indices]
        lineup.sort(key=lambda p: p["salary"], reverse=True)
        lineups.append(lineup)

    # ── Portfolio Summary ──
    sel_payouts = payouts[selected]
    total_cost = entry_fee * len(selected)
    port_returns = sel_payouts.sum(axis=0) - total_cost
    port_roi = port_returns / total_cost * 100

    print(f"\n  PORTFOLIO SUMMARY ({len(lineups)} lineups)")
    print(f"  {'='*50}")
    print(f"  Total Cost:       ${total_cost:,.0f}")
    print(f"  Expected Return:  ${port_returns.mean():+,.0f}")
    print(f"  Expected ROI:     {port_roi.mean():+.1f}%")
    print(f"  ROI Std Dev:      {port_roi.std():.1f}%")
    print(f"  5th Percentile:   {float(sorted(port_roi)[int(len(port_roi)*0.05)]):+.1f}%")
    print(f"  95th Percentile:  {float(sorted(port_roi)[int(len(port_roi)*0.95)]):+.1f}%")
    print(f"  Cash Rate:        {(sel_payouts.sum(axis=0) > total_cost).mean()*100:.1f}%")

    # Player exposure
    exposure = {}
    for lu in lineups:
        for p in lu:
            exposure[p["name"]] = exposure.get(p["name"], 0) + 1

    print(f"\n  PLAYER EXPOSURE (top 15)")
    print(f"  {'Player':<25} {'Count':>6} {'Pct':>7}")
    print(f"  {'-'*25} {'-'*6} {'-'*7}")
    for name, cnt in sorted(exposure.items(), key=lambda x: -x[1])[:15]:
        print(f"  {name:<25} {cnt:>6} {cnt/len(lineups)*100:>6.1f}%")

    # Top 5 lineups
    print(f"\n  TOP 5 LINEUPS")
    for i, (sel_idx, lineup) in enumerate(zip(selected[:5], lineups[:5])):
        names = ", ".join(p["name"] for p in lineup)
        total_sal = sum(p["salary"] for p in lineup)
        total_proj = sum(p["projected_points"] for p in lineup)
        avg_own = sum(p["proj_ownership"] for p in lineup) / ROSTER_SIZE
        print(f"  #{i+1} ROI={roi[sel_idx]:+.1f}% Sal=${total_sal:,} Proj={total_proj:.1f} Own={avg_own:.1f}%")
        print(f"     {names}")

    # ── Export CSV ──
    event_slug = event_name.replace(" ", "_").replace("'", "")
    csv_filename = f"lineups_{event_slug}_{len(lineups)}.csv"
    csv_path_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_filename)

    with open(csv_path_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["G"] * ROSTER_SIZE)
        for lineup in lineups:
            row = [p.get("name_id", p["name"]) for p in lineup]
            writer.writerow(row)
    print(f"\n  CSV: {csv_path_out}")

    # ── Export to Google Sheets ──
    if args.sheets and lineups:
        print(f"  Exporting to Google Sheets...")
        try:
            sheet_url = export_to_sheets(
                event_name=event_name,
                lineups=lineups,
                projected_players=players,
                contest_profile=profile,
            )
            print(f"  Sheet: {sheet_url}")
        except Exception as e:
            print(f"  Sheets ERROR: {e}")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Done in {elapsed:.0f}s — {len(lineups)} lineups for {event_name}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

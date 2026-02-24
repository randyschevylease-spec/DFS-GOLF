#!/usr/bin/env python3
"""Backtest any historical PGA event against a real DK contest payout table.

Uses DG prediction archive + DK salary CSV (or synthesized salaries for gaps).
Runs the full 3-step pipeline, scores against actual results, exports to Sheets.

Usage:
    python backtest_event.py --event-id 14 --year 2025 --contest 175290059 --sheets
"""
import os
import re
import csv
import time
import argparse
import requests
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATAGOLF_API_KEY, DATAGOLF_BASE_URL, ROSTER_SIZE, SALARY_CAP, SALARY_FLOOR
from dk_contests import fetch_contest
from engine import generate_field, generate_candidates, simulate_contest, select_portfolio
from google_sheets import _get_client, _get_or_create_worksheet, SHEET_ID, HEADER_FMT


# ── DraftKings Classic PGA Scoring ──────────────────────────────────────────

def calc_dk_points_round(rd):
    if rd is None:
        return 0.0
    pts = 0.0
    pts += (rd.get("eagles_or_better", 0) or 0) * 8
    pts += (rd.get("birdies", 0) or 0) * 3
    pts += (rd.get("pars", 0) or 0) * 0.5
    pts += (rd.get("bogies", 0) or 0) * (-0.5)
    pts += (rd.get("doubles_or_worse", 0) or 0) * (-1)
    if (rd.get("bogies", 0) or 0) == 0 and (rd.get("doubles_or_worse", 0) or 0) == 0:
        pts += 3
    if (rd.get("score") or 999) < 70:
        pts += 3
    return pts


def finish_bonus(fin_text):
    if not fin_text or fin_text == "CUT":
        return 0
    num = int(re.sub(r"[^0-9]", "", fin_text) or "999")
    if num == 1: return 30
    elif num == 2: return 20
    elif num == 3: return 18
    elif num == 4: return 16
    elif num == 5: return 14
    elif 6 <= num <= 10: return 12
    elif 11 <= num <= 15: return 10
    elif 16 <= num <= 20: return 8
    elif 21 <= num <= 25: return 6
    elif 26 <= num <= 30: return 4
    elif 31 <= num <= 40: return 2
    return 0


def calc_dk_total(player_data):
    total = 0.0
    for rd_key in ["round_1", "round_2", "round_3", "round_4"]:
        rd = player_data.get(rd_key)
        if rd:
            total += calc_dk_points_round(rd)
    total += finish_bonus(player_data.get("fin_text", ""))
    return round(total, 1)


def normalize_name(dg_name):
    if ", " in dg_name:
        parts = dg_name.split(", ", 1)
        return f"{parts[1]} {parts[0]}"
    return dg_name


# ── Name mappings for DG → DK mismatches ───────────────────────────────────

DG_TO_DK_NAME = {
    "Cam Davis": "Cameron Davis",
    "Nico Echavarria": "Nicolas Echavarria",
}

# Players NOT on DK slate (amateurs, long-retired past champions)
DK_EXCLUDED = {
    "Jose Maria Olazabal", "Bernhard Langer", "Fred Couples", "Mike Weir",
    "Noah Kent", "Hiroshi Tai", "Jose Luis Ballester", "Rafael Campos",
    "Justin Hastings",
}

# Estimated DK salaries for players not in the CSV
# Based on DG ranking position and typical DK pricing patterns
SALARY_ESTIMATES = {
    "Wyndham Clark": 7800,
    "Tony Finau": 7400,
    "Matt Fitzpatrick": 7700,
    "Justin Rose": 7100,
    "Max Homa": 7200,
    "Denny McCarthy": 6800,
    "Daniel Berger": 7000,
    "Taylor Pendrith": 6600,
    "Billy Horschel": 6900,
    "Chris Kirk": 6700,
    "Nick Taylor": 6500,
    "Maverick McNealy": 6600,
    "Dustin Johnson": 7500,
    "Christiaan Bezuidenhout": 6700,
    "Rasmus Hojgaard": 6800,
    "Max Greyserman": 6400,
    "Austin Eckroat": 6500,
    "Matt McCarty": 6300,
    "Kevin Yu": 6400,
    "Joe Highsmith": 6200,
    "Laurie Canter": 6300,
    "Nicolas Echavarria": 6400,
    "Davis Riley": 6500,
    "Adam Schenk": 6300,
    "Brian Campbell": 6200,
    "Jhonattan Vegas": 6200,
    "Charl Schwartzel": 6300,
    "Phil Mickelson": 6800,
    "Bubba Watson": 6200,
    "Danny Willett": 6100,
    "Zach Johnson": 6100,
    "Patton Kizzire": 6100,
    "Thriston Lawrence": 6200,
    "Evan Beck": 6000,
}


def load_dk_salaries(csv_path):
    """Load DK salaries from CSV, augmented with estimated salaries for gaps."""
    salaries = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)  # DKSalaries
            next(reader)  # header
            for row in reader:
                if len(row) >= 6 and row[2].strip():
                    salaries[row[2].strip()] = int(row[5])

    # Add estimated salaries for missing players
    for name, sal in SALARY_ESTIMATES.items():
        if name not in salaries:
            salaries[name] = sal

    return salaries


def build_players(predictions, dk_salaries, actual_scores):
    """Build player list from DG predictions + DK salaries + actual results.

    Derives projected DK points from DG probabilities and assigns real DK salaries.
    """
    # Average DK points by finish range (empirical)
    finish_pts = {
        "win": 135, "top_3": 120, "top_5": 110, "top_10": 95,
        "top_20": 80, "top_30": 68, "made_cut": 55, "missed_cut": 28,
    }

    # Actual results lookup
    actual_lookup = {}
    for s in actual_scores:
        name = normalize_name(s["player_name"])
        name = DG_TO_DK_NAME.get(name, name)
        actual_lookup[name] = {
            "dk_pts": calc_dk_total(s),
            "fin_text": s.get("fin_text", "?"),
        }

    players = []
    for pred in predictions:
        raw_name = normalize_name(pred["player_name"])
        name = DG_TO_DK_NAME.get(raw_name, raw_name)

        # Skip players not on DK
        if name in DK_EXCLUDED or raw_name in DK_EXCLUDED:
            continue

        # Get DK salary
        salary = dk_salaries.get(name, 0)
        if not salary:
            continue  # No salary data, can't include

        # Extract probabilities
        win = pred.get("win", 0) or 0
        top_3 = pred.get("top_3", 0) or 0
        top_5 = pred.get("top_5", 0) or 0
        top_10 = pred.get("top_10", 0) or 0
        top_20 = pred.get("top_20", 0) or 0
        top_30 = pred.get("top_30", 0) or 0
        make_cut = pred.get("make_cut", 0) or 0

        # Normalize to fractions if percentages
        if make_cut > 1:
            win /= 100; top_3 /= 100; top_5 /= 100
            top_10 /= 100; top_20 /= 100; top_30 /= 100
            make_cut /= 100

        # Expected DK points from probability-weighted finish ranges
        proj_pts = (
            win * finish_pts["win"]
            + max(0, top_3 - win) * finish_pts["top_3"]
            + max(0, top_5 - top_3) * finish_pts["top_5"]
            + max(0, top_10 - top_5) * finish_pts["top_10"]
            + max(0, top_20 - top_10) * finish_pts["top_20"]
            + max(0, top_30 - top_20) * finish_pts["top_30"]
            + max(0, make_cut - top_30) * finish_pts["made_cut"]
            + max(0, 1.0 - make_cut) * finish_pts["missed_cut"]
        )

        std_dev = max(25 - 10 * make_cut, 12)

        players.append({
            "name": name,
            "name_id": f"{name} ({pred.get('dg_id', 0)})",
            "salary": salary,
            "projected_points": round(proj_pts, 2),
            "std_dev": round(std_dev, 1),
            "win_prob": win,
            "make_cut_prob": make_cut,
            "top_5_prob": top_5,
            "top_10_prob": top_10,
            "actual_dk_pts": actual_lookup.get(name, {}).get("dk_pts", 0),
            "actual_finish": actual_lookup.get(name, {}).get("fin_text", "?"),
        })

    players.sort(key=lambda p: p["projected_points"], reverse=True)

    # Synthesize ownership from projections (softmax)
    projs = np.array([p["projected_points"] for p in players])
    exp_projs = np.exp((projs - projs.mean()) / max(projs.std(), 1))
    synth_own = exp_projs / exp_projs.sum() * 100 * ROSTER_SIZE
    for i, p in enumerate(players):
        p["proj_ownership"] = round(float(synth_own[i]), 2)
        p["value"] = round(p["projected_points"] / (p["salary"] / 1000), 2)

    return players, actual_lookup


def main():
    parser = argparse.ArgumentParser(description="Backtest historical PGA event")
    parser.add_argument("--event-id", type=int, required=True, help="DataGolf event ID")
    parser.add_argument("--year", type=int, required=True, help="Year")
    parser.add_argument("--contest", type=str, required=True, help="DK contest ID for payout table")
    parser.add_argument("--lineups", type=int, default=150, help="Lineups to generate")
    parser.add_argument("--candidates", type=int, default=5000, help="Candidate pool size")
    parser.add_argument("--sims", type=int, default=10000, help="Monte Carlo simulations")
    parser.add_argument("--field-size", type=int, default=None, help="Sim field size")
    parser.add_argument("--sheets", action="store_true", help="Export to Google Sheets")
    args = parser.parse_args()

    start_time = time.time()
    base = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("  DFS GOLF — HISTORICAL BACKTEST")
    print("=" * 70)

    # ── Contest details ──
    print(f"\n  Fetching contest {args.contest}...")
    profile = fetch_contest(args.contest)
    entry_fee = profile["entry_fee"]
    actual_field = profile["max_entries"]
    # Scale sim field: use proportional sampling (cap at 50K for performance)
    field_size = args.field_size or min(actual_field, 50000)
    n_lineups = args.lineups
    payout_table = profile["payouts"]

    # Scale payout positions to sim field size
    # If real field is 470K and sim is 50K, position 1000 in real ≈ position 106 in sim
    scale_factor = field_size / actual_field
    scaled_payout_table = []
    for min_pos, max_pos, prize in payout_table:
        scaled_min = max(1, int(round(min_pos * scale_factor)))
        scaled_max = max(scaled_min, int(round(max_pos * scale_factor)))
        if scaled_min <= field_size:
            scaled_payout_table.append((scaled_min, min(scaled_max, field_size), prize))

    # Deduplicate overlapping scaled positions (keep highest prize)
    deduped = {}
    for min_p, max_p, prize in scaled_payout_table:
        for pos in range(min_p, max_p + 1):
            if pos not in deduped or prize > deduped[pos]:
                deduped[pos] = prize
    # Rebuild payout table from deduped
    if deduped:
        sorted_pos = sorted(deduped.keys())
        final_payout_table = []
        i = 0
        while i < len(sorted_pos):
            start = sorted_pos[i]
            prize = deduped[start]
            end = start
            while i + 1 < len(sorted_pos) and sorted_pos[i + 1] == end + 1 and deduped[sorted_pos[i + 1]] == prize:
                i += 1
                end = sorted_pos[i]
            final_payout_table.append((start, end, prize))
            i += 1
        scaled_payout_table = final_payout_table

    payout_spots_scaled = sum(max_p - min_p + 1 for min_p, max_p, _ in scaled_payout_table)

    print(f"  Contest: {profile['name']}")
    print(f"  Entry Fee: ${entry_fee} | Real Field: {actual_field:,} | Sim Field: {field_size:,}")
    print(f"  Prize Pool: ${profile['prize_pool']:,.0f} | 1st: ${profile['first_place_prize']:,.0f}")
    print(f"  Payout Spots: {profile['payout_spots']:,} ({profile['payout_spots']/actual_field*100:.1f}%)")
    print(f"  Scaled Payout Spots: {payout_spots_scaled:,} ({payout_spots_scaled/field_size*100:.1f}%)")
    print(f"  Generating: {n_lineups} lineups")

    # ── DG predictions ──
    print(f"\n  Fetching DG predictions for event {args.event_id}, year {args.year}...")
    resp = requests.get(f"{DATAGOLF_BASE_URL}/preds/pre-tournament-archive", params={
        "key": DATAGOLF_API_KEY, "tour": "pga", "event_id": args.event_id,
        "year": args.year, "odds_format": "percent",
    })
    resp.raise_for_status()
    pred_data = resp.json()
    predictions = pred_data.get("baseline", [])
    event_name = pred_data.get("event_name", f"Event {args.event_id}")
    print(f"  Event: {event_name}")
    print(f"  Players in predictions: {len(predictions)}")

    # ── Actual results ──
    print(f"  Fetching actual results...")
    resp = requests.get(f"{DATAGOLF_BASE_URL}/historical-raw-data/rounds", params={
        "key": DATAGOLF_API_KEY, "tour": "pga", "event_id": args.event_id,
        "year": args.year, "file_format": "json",
    })
    resp.raise_for_status()
    actual_scores = resp.json()["scores"]
    print(f"  Players with actual results: {len(actual_scores)}")

    def fin_sort(x):
        ft = x.get("fin_text", "") or ""
        digits = re.sub(r"[^0-9]", "", ft)
        return int(digits) if digits else 999
    sorted_actual = sorted(actual_scores, key=fin_sort)
    winner = sorted_actual[0] if sorted_actual else None
    if winner:
        print(f"  Winner: {normalize_name(winner['player_name'])} (Fin: {winner['fin_text']})")

    # ── Build players ──
    print(f"\n  Loading DK salaries...")
    salary_csv = os.path.join(base, "salaries", "DKSalaries_2025_Masters.csv")
    dk_salaries = load_dk_salaries(salary_csv)
    print(f"  DK salary entries: {len(dk_salaries)}")

    players, actual_lookup = build_players(predictions, dk_salaries, actual_scores)
    print(f"  Usable players: {len(players)}")

    # Count real vs estimated salaries
    real_sal = sum(1 for p in players if p["name"] in dk_salaries and p["name"] not in SALARY_ESTIMATES)
    est_sal = len(players) - real_sal
    print(f"  Real DK salaries: {real_sal} | Estimated: {est_sal}")

    # Print player board
    print(f"\n  {'Player':<25} {'Salary':>8} {'Proj':>7} {'Own%':>6} {'Value':>7} {'Win%':>6} {'Actual':>7} {'Fin':>5}")
    print(f"  {'-'*25} {'-'*8} {'-'*7} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*5}")
    for p in players[:25]:
        print(f"  {p['name']:<25} ${p['salary']:>7,} {p['projected_points']:>7.1f} "
              f"{p['proj_ownership']:>5.1f}% {p['value']:>7.2f} {p['win_prob']*100:>5.1f}% "
              f"{p['actual_dk_pts']:>7.1f} {p['actual_finish']:>5}")
    print(f"  ... {len(players)} total players")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Generate Contest Field
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 1: Generating {field_size:,} opponent lineups")
    print(f"{'='*70}")

    opponents = generate_field(players, field_size)
    print(f"  Generated {len(opponents):,} opponent lineups")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Generate Candidates + Simulate ROI
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 2: Generate candidates + simulate ROI")
    print(f"{'='*70}")

    print(f"  Generating {args.candidates:,} candidate lineups...")
    candidates = generate_candidates(players, pool_size=args.candidates)
    print(f"  Unique candidates: {len(candidates):,}")

    payouts_matrix, roi = simulate_contest(
        candidates, opponents, players, scaled_payout_table, entry_fee, n_sims=args.sims,
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
    print(f"\n{'='*70}")
    print(f"  STEP 3: Select best {n_lineups} lineups (marginal E[max] + exposure caps)")
    print(f"{'='*70}")

    selected = select_portfolio(payouts_matrix, entry_fee, n_lineups, candidates,
                                n_players=len(players))

    lineups = []
    for sel_idx in selected:
        player_indices = candidates[sel_idx]
        lineup = [players[i].copy() for i in player_indices]
        lineup.sort(key=lambda p: p["salary"], reverse=True)
        lineups.append(lineup)

    # ══════════════════════════════════════════════════════════════════
    # SCORE AGAINST ACTUAL RESULTS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  SCORING AGAINST ACTUAL RESULTS")
    print(f"{'='*70}")

    lineup_results = []
    for i, lineup in enumerate(lineups):
        total_proj = sum(p["projected_points"] for p in lineup)
        total_actual = sum(actual_lookup.get(p["name"], {}).get("dk_pts", 0) for p in lineup)
        total_salary = sum(p["salary"] for p in lineup)
        lineup_results.append({
            "lineup_num": i + 1,
            "players": lineup,
            "total_salary": total_salary,
            "total_proj": round(total_proj, 1),
            "total_actual": round(total_actual, 1),
        })

    lineup_results_sorted = sorted(lineup_results, key=lambda x: x["total_actual"], reverse=True)

    total_cost = entry_fee * len(lineups)
    avg_proj = np.mean([lr["total_proj"] for lr in lineup_results])
    avg_actual = np.mean([lr["total_actual"] for lr in lineup_results])
    max_actual = max(lr["total_actual"] for lr in lineup_results)
    min_actual = min(lr["total_actual"] for lr in lineup_results)

    # Player exposure
    exposure = {}
    for lu in lineups:
        for p in lu:
            exposure[p["name"]] = exposure.get(p["name"], 0) + 1

    print(f"\n  BACKTEST SUMMARY")
    print(f"  {'='*50}")
    print(f"  Event: {event_name} ({args.year})")
    print(f"  Winner: {normalize_name(winner['player_name']) if winner else '?'}")
    print(f"  Lineups: {len(lineups)} | Entry: ${entry_fee} | Total Cost: ${total_cost:,.0f}")
    print(f"  Avg Projected: {avg_proj:.1f} pts | Avg Actual: {avg_actual:.1f} pts")
    print(f"  Best Lineup: {max_actual:.1f} pts | Worst: {min_actual:.1f} pts")
    print(f"  Unique Players: {len(exposure)}")

    print(f"\n  PLAYER EXPOSURE (top 15)")
    print(f"  {'Player':<25} {'Count':>6} {'Pct':>7} {'Actual':>8} {'Finish':>7}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*8} {'-'*7}")
    for name, cnt in sorted(exposure.items(), key=lambda x: -x[1])[:15]:
        act = actual_lookup.get(name, {})
        print(f"  {name:<25} {cnt:>6} {cnt/len(lineups)*100:>6.1f}% "
              f"{act.get('dk_pts', 0):>7.1f} {act.get('fin_text', '?'):>7}")

    print(f"\n  TOP 5 LINEUPS BY ACTUAL SCORE")
    for lr in lineup_results_sorted[:5]:
        names = ", ".join(p["name"] for p in lr["players"])
        print(f"  #{lr['lineup_num']} Sal=${lr['total_salary']:,} Proj={lr['total_proj']:.1f} Actual={lr['total_actual']:.1f}")
        print(f"     {names}")

    # ── Export CSV ──
    event_slug = event_name.replace(" ", "_").replace("'", "")
    csv_filename = f"backtest_{event_slug}_{args.year}_{len(lineups)}.csv"
    csv_path_out = os.path.join(base, csv_filename)
    with open(csv_path_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["G"] * ROSTER_SIZE)
        for lineup in lineups:
            writer.writerow([p["name_id"] for p in lineup])
    print(f"\n  CSV: {csv_path_out}")

    # ── Google Sheets ──
    if args.sheets:
        print(f"\n  Exporting to Google Sheets...")
        client = _get_client()
        spreadsheet = client.open_by_key(SHEET_ID)

        tab_prefix = "BT Masters 2025"

        # Tab 1: Summary
        summary_ws = _get_or_create_worksheet(spreadsheet, f"{tab_prefix} | Summary", 30, 4)
        summary_rows = [
            [f"BACKTEST: {event_name} {args.year}", "", "", ""],
            ["", "", "", ""],
            ["Metric", "Value", "", ""],
            ["Event", event_name, "", ""],
            ["Year", args.year, "", ""],
            ["Winner", normalize_name(winner["player_name"]) if winner else "?", "", ""],
            ["", "", "", ""],
            ["Contest", profile["name"], "", ""],
            ["Entry Fee", f"${entry_fee}", "", ""],
            ["Real Field Size", f"{actual_field:,}", "", ""],
            ["Sim Field Size", f"{field_size:,}", "", ""],
            ["Prize Pool", f"${profile['prize_pool']:,.0f}", "", ""],
            ["1st Place Prize", f"${profile['first_place_prize']:,.0f}", "", ""],
            ["Payout %", f"{profile['payout_spots']/actual_field*100:.1f}%", "", ""],
            ["", "", "", ""],
            ["Lineups Generated", len(lineups), "", ""],
            ["Total Cost", f"${total_cost:,.0f}", "", ""],
            ["Avg Projected Pts", round(avg_proj, 1), "", ""],
            ["Avg Actual Pts", round(avg_actual, 1), "", ""],
            ["Best Lineup (Actual)", round(max_actual, 1), "", ""],
            ["Worst Lineup (Actual)", round(min_actual, 1), "", ""],
            ["Unique Players", len(exposure), "", ""],
            ["", "", "", ""],
            ["Candidates Generated", len(candidates), "", ""],
            ["Monte Carlo Sims", args.sims, "", ""],
            ["Real DK Salaries", real_sal, "", ""],
            ["Estimated Salaries", est_sal, "", ""],
        ]
        summary_ws.update(summary_rows, value_input_option="RAW")
        summary_ws.format("A1:D1", HEADER_FMT)
        summary_ws.format("A3:D3", HEADER_FMT)

        # Tab 2: Lineups
        detail_ws = _get_or_create_worksheet(
            spreadsheet, f"{tab_prefix} | Lineups", len(lineups) * 8 + 5, 10
        )
        detail_header = ["Rank", "LU #", "Player", "Salary", "Proj Pts", "Actual Pts",
                         "Own%", "Finish", "LU Proj", "LU Actual"]
        detail_rows = [detail_header]
        for rank, lr in enumerate(lineup_results_sorted, 1):
            for j, p in enumerate(lr["players"]):
                act = actual_lookup.get(p["name"], {})
                detail_rows.append([
                    rank if j == 0 else "",
                    lr["lineup_num"] if j == 0 else "",
                    p["name"],
                    p["salary"],
                    round(p["projected_points"], 1),
                    round(act.get("dk_pts", 0), 1),
                    round(p["proj_ownership"], 1),
                    act.get("fin_text", "?"),
                    round(lr["total_proj"], 1) if j == 0 else "",
                    round(lr["total_actual"], 1) if j == 0 else "",
                ])
            detail_rows.append([""] * 10)
        detail_ws.update(detail_rows, value_input_option="RAW")
        detail_ws.format("A1:J1", HEADER_FMT)

        # Tab 3: Players
        sorted_exp = sorted(exposure.items(), key=lambda x: -x[1])
        player_ws = _get_or_create_worksheet(
            spreadsheet, f"{tab_prefix} | Players", len(sorted_exp) + 5, 9
        )
        player_header = ["Player", "Lineups", "Exposure %", "Salary", "Proj Pts",
                         "Actual Pts", "Diff", "Finish", "Win%"]
        player_rows = [player_header]
        for name, count in sorted_exp:
            p_data = next((p for p in players if p["name"] == name), {})
            act = actual_lookup.get(name, {})
            proj_pts = p_data.get("projected_points", 0)
            actual_pts = act.get("dk_pts", 0)
            player_rows.append([
                name, count, f"{count / len(lineups) * 100:.1f}%",
                p_data.get("salary", 0),
                round(proj_pts, 1), round(actual_pts, 1),
                round(actual_pts - proj_pts, 1),
                act.get("fin_text", "?"),
                f"{p_data.get('win_prob', 0) * 100:.1f}%",
            ])
        player_ws.update(player_rows, value_input_option="RAW")
        player_ws.format("A1:I1", HEADER_FMT)

        print(f"  Google Sheet: {spreadsheet.url}")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Done in {elapsed:.0f}s — {len(lineups)} lineups backtested for {event_name} {args.year}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

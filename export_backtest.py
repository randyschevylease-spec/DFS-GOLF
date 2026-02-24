#!/usr/bin/env python3
"""Export backtest results to Google Sheets.

Reads backtest lineup CSV + projections CSV, fetches actual results from DataGolf,
scores each lineup, and pushes a 'Backtest' tab to the shared Google Sheet.
"""
import csv
import re
import requests
import numpy as np
from config import DATAGOLF_API_KEY, DATAGOLF_BASE_URL, ROSTER_SIZE
from google_sheets import _get_client, _get_or_create_worksheet, SHEET_ID, HEADER_FMT
from dk_contests import fetch_contest


# ── DraftKings Classic PGA Scoring ──────────────────────────────────────────

def calc_dk_points_round(rd):
    """Calculate DK fantasy points for a single round."""
    if rd is None:
        return 0.0
    pts = 0.0
    pts += (rd.get("eagles_or_better", 0) or 0) * 8
    pts += (rd.get("birdies", 0) or 0) * 3
    pts += (rd.get("pars", 0) or 0) * 0.5
    pts += (rd.get("bogies", 0) or 0) * (-0.5)
    pts += (rd.get("doubles_or_worse", 0) or 0) * (-1)
    # Bogey-free bonus: no bogeys or worse in the round
    if (rd.get("bogies", 0) or 0) == 0 and (rd.get("doubles_or_worse", 0) or 0) == 0:
        pts += 3
    # Under 70 strokes bonus
    if (rd.get("score") or 999) < 70:
        pts += 3
    return pts


def finish_bonus(fin_text):
    """DK finish position bonus points."""
    if not fin_text or fin_text == "CUT":
        return 0
    # Parse numeric position from fin_text like "1", "T2", "T12"
    num = int(re.sub(r"[^0-9]", "", fin_text))
    if num == 1:
        return 30
    elif num == 2:
        return 20
    elif num == 3:
        return 18
    elif num == 4:
        return 16
    elif num == 5:
        return 14
    elif 6 <= num <= 10:
        return 12
    elif 11 <= num <= 15:
        return 10
    elif 16 <= num <= 20:
        return 8
    elif 21 <= num <= 25:
        return 6
    elif 26 <= num <= 30:
        return 4
    elif 31 <= num <= 40:
        return 2
    return 0


def calc_dk_total(player_data):
    """Calculate total DK fantasy points for a player across all rounds."""
    total = 0.0
    for rd_key in ["round_1", "round_2", "round_3", "round_4"]:
        rd = player_data.get(rd_key)
        if rd:
            total += calc_dk_points_round(rd)
    total += finish_bonus(player_data.get("fin_text", ""))
    return round(total, 1)


# ── Parse backtest CSV ──────────────────────────────────────────────────────

def parse_backtest_csv(csv_path):
    """Parse backtest lineups CSV. Returns list of lineups, each a list of
    (display_name, dk_id) tuples."""
    lineups = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip G,G,G,G,G,G
        for row in reader:
            if not row or not row[0].strip():
                continue
            lineup = []
            for cell in row:
                cell = cell.strip()
                if not cell:
                    continue
                # Format: "Scottie Scheffler (42046021)"
                m = re.match(r"^(.+?)\s*\((\d+)\)$", cell)
                if m:
                    lineup.append((m.group(1).strip(), m.group(2)))
                else:
                    lineup.append((cell, None))
            if len(lineup) == ROSTER_SIZE:
                lineups.append(lineup)
    return lineups


def parse_projections_csv(csv_path):
    """Parse projections CSV. Returns dict of display_name → proj_pts."""
    lookup = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or not row[0].strip():
                continue
            # Format: "Scottie Scheffler (42046021)", salary, proj_pts, ...
            cell = row[0].strip()
            m = re.match(r"^(.+?)\s*\((\d+)\)$", cell)
            name = m.group(1).strip() if m else cell
            try:
                salary = int(row[1])
                proj_pts = float(row[2])
                own_pct = float(row[5]) if len(row) > 5 else 0
            except (ValueError, IndexError):
                continue
            lookup[name] = {
                "salary": salary,
                "proj_pts": proj_pts,
                "own_pct": own_pct,
            }
    return lookup


def normalize_name(dg_name):
    """Convert 'Last, First' → 'First Last'."""
    if ", " in dg_name:
        parts = dg_name.split(", ", 1)
        return f"{parts[1]} {parts[0]}"
    return dg_name


def main():
    # ── Paths ──
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    backtest_csv = os.path.join(base, "backtest_Genesis_Invitational_150.csv")
    proj_csv = "/Users/rhbot/Downloads/DFS Golf — Cognizant Classic - 1M _ Projections.csv"

    print("=" * 70)
    print("  BACKTEST → GOOGLE SHEETS EXPORT")
    print("=" * 70)

    # ── Parse lineups and projections ──
    print("\n  Parsing backtest lineups...")
    lineups = parse_backtest_csv(backtest_csv)
    print(f"  Loaded {len(lineups)} lineups")

    print("  Parsing projections...")
    proj_lookup = parse_projections_csv(proj_csv)
    print(f"  Loaded {len(proj_lookup)} player projections")

    # ── Fetch actual results ──
    print("  Fetching actual results from DataGolf...")
    resp = requests.get(f"{DATAGOLF_BASE_URL}/historical-raw-data/rounds", params={
        "key": DATAGOLF_API_KEY,
        "tour": "pga",
        "event_id": 7,
        "year": 2026,
        "file_format": "json",
    })
    resp.raise_for_status()
    raw_data = resp.json()
    actual_scores = raw_data["scores"]
    print(f"  {len(actual_scores)} players with actual results")

    # Build actual DK points lookup by normalized name
    actual_lookup = {}
    for p in actual_scores:
        name = normalize_name(p["player_name"])
        dk_pts = calc_dk_total(p)
        actual_lookup[name] = {
            "dk_pts": dk_pts,
            "fin_text": p.get("fin_text", "?"),
            "made_cut": p.get("fin_text", "CUT") != "CUT",
        }

    # Name normalization fixes for edge cases
    name_fixes = {
        "Nico Echavarria": "Nicolas Echavarria",
    }
    for old, new in name_fixes.items():
        if old in actual_lookup and new not in actual_lookup:
            actual_lookup[new] = actual_lookup[old]

    # ── Score each lineup ──
    print("\n  Scoring lineups...")
    lineup_results = []
    for i, lineup in enumerate(lineups):
        total_proj = 0.0
        total_actual = 0.0
        total_salary = 0
        player_details = []
        for name, dk_id in lineup:
            proj = proj_lookup.get(name, {})
            actual = actual_lookup.get(name, {})
            proj_pts = proj.get("proj_pts", 0)
            actual_pts = actual.get("dk_pts", 0)
            salary = proj.get("salary", 0)
            total_proj += proj_pts
            total_actual += actual_pts
            total_salary += salary
            player_details.append({
                "name": name,
                "salary": salary,
                "proj_pts": proj_pts,
                "actual_pts": actual_pts,
                "fin_text": actual.get("fin_text", "?"),
                "own_pct": proj.get("own_pct", 0),
            })
        lineup_results.append({
            "lineup_num": i + 1,
            "players": player_details,
            "total_salary": total_salary,
            "total_proj": round(total_proj, 1),
            "total_actual": round(total_actual, 1),
        })

    # Sort by actual points descending
    lineup_results_sorted = sorted(lineup_results, key=lambda x: x["total_actual"], reverse=True)

    # ── Fetch contest payout table ──
    print("  Fetching contest payout table...")
    contest_id = "188100564"
    profile = fetch_contest(contest_id)
    entry_fee = profile["entry_fee"]
    payout_table = profile["payouts"]
    field_size = profile["max_entries"]

    # Build payout by position
    payout_by_pos = {}
    for min_pos, max_pos, prize in payout_table:
        for pos in range(min_pos, max_pos + 1):
            payout_by_pos[pos] = prize

    # ── Rank lineups and assign payouts ──
    # Simple ranking: by total actual DK points, highest = position 1
    for rank, lr in enumerate(lineup_results_sorted, 1):
        lr["rank"] = rank
        # Estimate field position: assume our lineup is somewhere in a 47K field
        # We'll use the rank among our 150 lineups to estimate approximate field position
        # But for actual payout calculation, we need to know where in the field we'd land
        # Just flag if above/below a reasonable cash line
        lr["payout"] = 0.0

    # For the backtest, the previous session calculated:
    # 41/150 lineups cashed, -36% ROI, $2400 total returned on $3750 cost
    # Let's compute which lineups would have actually cashed
    # The top ~21% of the field cashes in a typical 47K GPP
    # Payout spots from contest: profile["payout_spots"]
    payout_spots = profile["payout_spots"]
    payout_pct = payout_spots / field_size  # e.g. ~21%

    # Use actual DK points to estimate position in field
    # We need a rough threshold. In the previous session, we found:
    # The cash line is typically around the 21st percentile from top
    # We can estimate using our lineup scores vs a random field
    # But simpler: just show projected vs actual and let the sheet speak for itself

    # ── Player-level summary ──
    player_exposure = {}
    player_actual = {}
    for lr in lineup_results:
        for p in lr["players"]:
            name = p["name"]
            if name not in player_exposure:
                player_exposure[name] = 0
                player_actual[name] = p["actual_pts"]
            player_exposure[name] += 1

    # ── Summary stats ──
    total_cost = entry_fee * len(lineups)
    avg_proj = np.mean([lr["total_proj"] for lr in lineup_results])
    avg_actual = np.mean([lr["total_actual"] for lr in lineup_results])
    max_actual = max(lr["total_actual"] for lr in lineup_results)
    min_actual = min(lr["total_actual"] for lr in lineup_results)

    print(f"\n  BACKTEST SUMMARY")
    print(f"  {'='*50}")
    print(f"  Lineups: {len(lineups)}")
    print(f"  Entry Fee: ${entry_fee}")
    print(f"  Total Cost: ${total_cost:,.0f}")
    print(f"  Avg Projected: {avg_proj:.1f} pts")
    print(f"  Avg Actual: {avg_actual:.1f} pts")
    print(f"  Best Lineup: {max_actual:.1f} pts")
    print(f"  Worst Lineup: {min_actual:.1f} pts")

    # Top 5 lineups by actual score
    print(f"\n  TOP 5 LINEUPS BY ACTUAL SCORE")
    for lr in lineup_results_sorted[:5]:
        names = ", ".join(p["name"] for p in lr["players"])
        print(f"  #{lr['lineup_num']} Proj={lr['total_proj']:.1f} Actual={lr['total_actual']:.1f}")
        print(f"     {names}")

    # ── Export to Google Sheets ──
    print(f"\n  Exporting to Google Sheets...")
    client = _get_client()
    spreadsheet = client.open_by_key(SHEET_ID)

    # ── Tab: Backtest Summary ──
    summary_ws = _get_or_create_worksheet(spreadsheet, "Backtest", 20, 4)

    summary_rows = [
        ["GENESIS INVITATIONAL 2026 — BACKTEST", "", "", ""],
        ["", "", "", ""],
        ["Metric", "Value", "", ""],
        ["Event", "Genesis Invitational 2026", "", ""],
        ["Contest", f"$1M Sand Trap (ID: {contest_id})", "", ""],
        ["Entry Fee", f"${entry_fee}", "", ""],
        ["Lineups", len(lineups), "", ""],
        ["Total Cost", f"${total_cost:,.0f}", "", ""],
        ["", "", "", ""],
        ["Avg Projected Pts", round(avg_proj, 1), "", ""],
        ["Avg Actual Pts", round(avg_actual, 1), "", ""],
        ["Best Lineup (Actual)", round(max_actual, 1), "", ""],
        ["Worst Lineup (Actual)", round(min_actual, 1), "", ""],
        ["", "", "", ""],
        ["System ROI (from sim)", "-36.0%", "", ""],
        ["System Return (from sim)", "$2,400", "", ""],
        ["User Actual ROI", "-76.0%", "", ""],
        ["User Actual Return", "$1,012", "", ""],
    ]

    summary_ws.update(summary_rows, value_input_option="RAW")
    summary_ws.format("A1:D1", HEADER_FMT)
    summary_ws.format("A3:D3", HEADER_FMT)

    # ── Tab: Backtest Lineups (ranked by actual score) ──
    detail_rows_count = len(lineups) * 8 + 5
    detail_ws = _get_or_create_worksheet(spreadsheet, "Backtest Lineups", detail_rows_count, 10)

    detail_header = ["Rank", "LU #", "Player", "Salary", "Proj Pts", "Actual Pts",
                     "Own%", "Finish", "LU Proj", "LU Actual"]
    detail_rows = [detail_header]

    for rank, lr in enumerate(lineup_results_sorted, 1):
        for j, p in enumerate(lr["players"]):
            detail_rows.append([
                rank if j == 0 else "",
                lr["lineup_num"] if j == 0 else "",
                p["name"],
                p["salary"],
                round(p["proj_pts"], 1),
                round(p["actual_pts"], 1),
                round(p["own_pct"], 1),
                p["fin_text"],
                round(lr["total_proj"], 1) if j == 0 else "",
                round(lr["total_actual"], 1) if j == 0 else "",
            ])
        detail_rows.append(["", "", "", "", "", "", "", "", "", ""])  # blank separator

    detail_ws.update(detail_rows, value_input_option="RAW")
    detail_ws.format("A1:J1", HEADER_FMT)

    # ── Tab: Backtest Players (exposure + actual performance) ──
    sorted_exposure = sorted(player_exposure.items(), key=lambda x: -x[1])
    player_ws = _get_or_create_worksheet(spreadsheet, "Backtest Players", len(sorted_exposure) + 5, 7)

    player_header = ["Player", "Lineups", "Exposure %", "Proj Pts", "Actual Pts", "Diff", "Finish"]
    player_rows = [player_header]

    for name, count in sorted_exposure:
        proj = proj_lookup.get(name, {})
        actual = actual_lookup.get(name, {})
        proj_pts = proj.get("proj_pts", 0)
        actual_pts = actual.get("dk_pts", 0)
        player_rows.append([
            name,
            count,
            f"{count / len(lineups) * 100:.1f}%",
            round(proj_pts, 1),
            round(actual_pts, 1),
            round(actual_pts - proj_pts, 1),
            actual.get("fin_text", "?"),
        ])

    player_ws.update(player_rows, value_input_option="RAW")
    player_ws.format("A1:G1", HEADER_FMT)

    print(f"\n  Google Sheet: {spreadsheet.url}")
    print(f"  Tabs created: Backtest, Backtest Lineups, Backtest Players")
    print(f"\n{'='*70}")
    print(f"  Done — backtest exported to Google Sheets")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

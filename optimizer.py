#!/usr/bin/env python3
"""DFS Golf Lineup Optimizer — Main Entry Point

Trust DataGolf for projections. Optimize the portfolio.

Uses DataGolf fantasy projections as the baseline, applies situational
edges (weather/wave, line movement), then runs efficient frontier
portfolio optimization with contest-aware parameters.

Usage:
    python optimizer.py                          # Generate 150 lineups (default)
    python optimizer.py --contest 188100564      # Auto-tune from DK contest
    python optimizer.py --list-contests          # Browse current DK golf contests
    python optimizer.py --lineups 10             # Generate 10 lineups
    python optimizer.py --leverage 0.50          # Aggressive ownership leverage
    python optimizer.py --snapshot               # Save prediction snapshot (Mon/Tue)
    python optimizer.py --compare                # Compare to snapshot + wire movement
    python optimizer.py --sheets                 # Export lineups to Google Sheets
    python optimizer.py --pool-size 0            # Use legacy sequential generation
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NUM_LINEUPS, ROSTER_SIZE, LEVERAGE_POWER
from datagolf_client import (
    get_predictions,
    get_fantasy_projections,
    get_skill_ratings,
    find_current_event,
)
from dk_salaries import find_latest_csv, parse_dk_csv, match_players_exact
from projections import build_projection, build_fallback_projection
from lineup_optimizer import optimize_lineup, format_all_lineups
from weather import get_wave_adjustment
from line_movement import (
    save_prediction_snapshot,
    load_latest_snapshot,
    compare_snapshots,
    get_movement_adjustments,
)
from dk_contests import (
    fetch_contest,
    fetch_golf_contests,
    classify_contest,
    derive_optimizer_params,
    format_contest_summary,
    format_contest_list,
)
from google_sheets import export_to_sheets


def main():
    parser = argparse.ArgumentParser(description="DFS Golf Lineup Optimizer")
    parser.add_argument("--contest", type=str, default=None,
                        help="DraftKings contest ID — auto-derives optimal parameters")
    parser.add_argument("--list-contests", action="store_true",
                        help="List all current DraftKings golf contests")
    parser.add_argument("--lineups", type=int, default=None,
                        help=f"Number of lineups to generate (default: {NUM_LINEUPS})")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to DraftKings salary CSV (default: latest in salaries/)")
    parser.add_argument("--exposure", type=float, default=None,
                        help="Max player exposure across lineups (0.0-1.0)")
    parser.add_argument("--leverage", type=float, default=None,
                        help=f"Ownership leverage power (0=none, 0.35=default, 0.5=aggressive)")
    parser.add_argument("--snapshot", action="store_true",
                        help="Save current predictions as a snapshot (run Mon/Tue)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare current predictions to last snapshot (run Thu)")
    parser.add_argument("--sheets", action="store_true",
                        help="Export lineups and projections to Google Sheets")
    parser.add_argument("--pool-size", type=int, default=10000,
                        help="Candidate pool size for portfolio optimization (default: 10000, 0=legacy)")
    parser.add_argument("--pipeline", action="store_true",
                        help="Use lineups from agent pipeline (shared/optimized_lineups.json) instead of generating new ones")
    args = parser.parse_args()

    # ── List contests mode ──
    if args.list_contests:
        print("=" * 70)
        print("  CURRENT DRAFTKINGS GOLF CONTESTS")
        print("=" * 70)
        try:
            contests = fetch_golf_contests()
            print(f"\n{format_contest_list(contests)}")
            print(f"\n  {len(contests)} contest(s) found")
            print(f"  Use --contest <ID> to auto-tune optimizer for a specific contest")
        except Exception as e:
            print(f"\n  ERROR fetching contests: {e}")
        print(f"{'='*70}\n")
        return

    # ── Pipeline mode: use agent-generated lineups ──
    if args.pipeline:
        project_root = os.path.dirname(os.path.abspath(__file__))
        shared_dir = os.path.join(project_root, "shared")
        lineups_path = os.path.join(shared_dir, "optimized_lineups.json")
        projections_path = os.path.join(shared_dir, "projections.json")

        if not os.path.exists(lineups_path):
            print(f"  ERROR: Pipeline output not found at {lineups_path}")
            print("  Run the pipeline first: python agents/run_pipeline.py")
            sys.exit(1)

        print("=" * 70)
        print("  PIPELINE MODE — Using agent-generated lineups")
        print("=" * 70)

        with open(lineups_path) as f:
            pipeline_data = json.load(f)
        with open(projections_path) as f:
            proj_data = json.load(f)

        meta = pipeline_data.get("metadata", {})
        print(f"  Generated: {meta.get('generated_at', 'unknown')}")
        print(f"  Field: {meta.get('contest_field_size', '?'):,} | Sims: {meta.get('n_sims', '?'):,}")
        print(f"  Methodology: {meta.get('methodology', 'unknown')}")

        # Load contest config for tab prefix
        contest_config_path = os.path.join(shared_dir, "contest_config.json")
        pipeline_contest_profile = None
        if os.path.exists(contest_config_path):
            with open(contest_config_path) as f:
                cc = json.load(f)
            prize = cc.get("prize_pool", 0)
            if prize >= 950_000:
                prize_label = f"${round(prize/1_000_000)}M"
            elif prize >= 1_000:
                prize_label = f"${round(prize/1_000)}K"
            else:
                prize_label = f"${prize:.0f}"
            pipeline_contest_profile = {
                "name": f"PGA TOUR {prize_label} Contest",
                "max_entries_per_user": cc.get("max_entries_per_user", 150),
                "entry_fee": cc.get("entry_fee", 0),
                "prize_pool": prize,
                "field_size": cc.get("field_size", 0),
            }
            print(f"  Contest prefix: {prize_label}")

        # Build name_id lookup from projections
        name_id_lookup = {}
        for p in proj_data.get("players", []):
            name_id_lookup[p["dg_id"]] = p.get("site_name_id", "")

        # Convert pipeline lineups to the format google_sheets expects
        raw_lineups = pipeline_data.get("large_gpp", {}).get("lineups", [])
        lineups = []
        for entry in raw_lineups:
            lineup = []
            for p in entry["players"]:
                # Convert "Last, First" to "First Last" for display
                parts = p["name"].split(", ", 1)
                display_name = f"{parts[1]} {parts[0]}" if len(parts) == 2 else p["name"]
                lineup.append({
                    "name": display_name,
                    "name_id": name_id_lookup.get(p["dg_id"], display_name),
                    "salary": p["salary"],
                    "projected_points": p["proj_pts"],
                    "proj_ownership": p.get("ownership_pct", 0),
                    "value": round(p["proj_pts"] / (p["salary"] / 1000), 2) if p["salary"] > 0 else 0,
                    "std_dev": 0,
                    "wave": "?",
                    "wave_adj": 0,
                    "movement_adj": 0,
                    "proj_points_scoring": 0,
                    "proj_points_finish": 0,
                    "p_make_cut": 0,
                })
            lineups.append(lineup)

        # Convert pipeline projections to the format google_sheets expects
        projected_players = []
        for p in proj_data.get("players", []):
            enh = p.get("enhanced_projection", {})
            dg = p.get("dg_projection", {})
            tt = p.get("tee_time_info", {})
            mc = p.get("dg_model_probs", {})
            projected_players.append({
                "name": name_id_lookup.get(p["dg_id"], p["name"]),
                "salary": p["dk_salary"],
                "projected_points": enh.get("final_proj_dk_pts", dg.get("proj_dk_pts", 0)),
                "value": enh.get("value_per_1k", 0),
                "p_make_cut": mc.get("make_cut", 0),
                "proj_ownership": dg.get("proj_ownership_pct", 0),
                "std_dev": dg.get("std_dev", 0),
                "wave": tt.get("wave", "?"),
                "wave_adj": 0,
                "movement_adj": 0,
                "proj_points_scoring": 0,
                "proj_points_finish": 0,
            })

        # Extract per-lineup simulation results
        sim_results = []
        for entry in raw_lineups:
            parts_list = []
            for p in entry["players"]:
                parts = p["name"].split(", ", 1)
                parts_list.append(f"{parts[1]} {parts[0]}" if len(parts) == 2 else p["name"])
            sim_results.append({
                "lineup_id": entry["lineup_id"],
                "players": parts_list,
                "total_salary": entry.get("total_salary", 0),
                "total_proj": entry.get("total_proj", 0),
                "mean_roi_pct": entry.get("mean_roi_pct", 0),
                "roi_std": entry.get("roi_std", 0),
                "cash_rate_pct": entry.get("cash_rate_pct", 0),
                "mean_payout": entry.get("mean_payout", 0),
                "avg_ownership": entry.get("avg_ownership", 0),
            })

        sim_summary = pipeline_data.get("large_gpp", {}).get("simulation_summary", {})
        portfolio_analytics = pipeline_data.get("large_gpp", {}).get("portfolio_analytics", None)

        print(f"  Lineups: {len(lineups)} | Players: {len(projected_players)}")

        # Determine event name
        event_name = "PGA Tournament"
        try:
            predictions = get_predictions()
            event_info = find_current_event(predictions)
            event_name = event_info["event_name"]
        except Exception:
            pass

        # Export CSV
        if lineups:
            from config import ROSTER_SIZE
            event_slug = event_name.replace(" ", "_").replace("'", "")
            csv_filename = f"lineups_{event_slug}_{len(lineups)}.csv"
            csv_path_out = os.path.join(project_root, csv_filename)

            import csv
            with open(csv_path_out, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["G"] * ROSTER_SIZE)
                for lineup in lineups:
                    row = [p.get("name_id", p["name"]) for p in lineup]
                    writer.writerow(row)
            print(f"  CSV saved: {csv_path_out}")

        # Export to Google Sheets
        if args.sheets and lineups:
            print(f"\n  Exporting to Google Sheets...")
            try:
                sheet_url = export_to_sheets(
                    event_name=event_name,
                    lineups=lineups,
                    projected_players=projected_players,
                    contest_profile=pipeline_contest_profile,
                    sim_results=sim_results,
                    sim_summary=sim_summary,
                    portfolio_analytics=portfolio_analytics,
                )
                print(f"  Google Sheet: {sheet_url}")
            except Exception as e:
                print(f"  ERROR exporting to Sheets: {e}")

        print(f"\n{'='*70}")
        print(f"  Exported {len(lineups)} pipeline lineup(s) for {event_name}")
        print(f"{'='*70}\n")
        return

    salaries_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salaries")

    print("=" * 70)
    print("  DFS GOLF LINEUP OPTIMIZER")
    print("=" * 70)

    # ── Step 0: Contest-aware parameter derivation ──
    contest_params = None
    if args.contest:
        print(f"\n[0/7] Fetching DraftKings contest details...")
        try:
            profile = fetch_contest(args.contest)
            metrics = classify_contest(profile)
            contest_params = derive_optimizer_params(metrics, profile)
            print(format_contest_summary(profile, metrics, contest_params))
        except Exception as e:
            print(f"  ERROR fetching contest {args.contest}: {e}")
            print("  Falling back to default parameters.")
            contest_params = None

    # CLI overrides take precedence over contest-derived params
    effective_lineups = args.lineups
    effective_leverage = args.leverage
    effective_exposure = args.exposure

    if effective_lineups is None:
        effective_lineups = (contest_params or {}).get("num_lineups", NUM_LINEUPS)
    if effective_leverage is None:
        effective_leverage = (contest_params or {}).get("leverage_power", LEVERAGE_POWER)
    if effective_exposure is None and contest_params:
        effective_exposure = contest_params.get("max_exposure")

    # ── Step 1: Fetch DataGolf fantasy projections (PRIMARY data source) ──
    print("\n[1/7] Fetching DataGolf fantasy projections...")
    try:
        fantasy_data = get_fantasy_projections()
    except Exception as e:
        print(f"  ERROR fetching fantasy projections: {e}")
        sys.exit(1)

    dg_projections = fantasy_data if isinstance(fantasy_data, list) else fantasy_data.get("projections", [])
    event_name = fantasy_data.get("event_name", "Unknown Event") if isinstance(fantasy_data, dict) else "Unknown Event"
    last_updated = fantasy_data.get("last_updated", "") if isinstance(fantasy_data, dict) else ""
    print(f"  Event: {event_name}")
    print(f"  Last updated: {last_updated}")
    print(f"  Players: {len(dg_projections)}")

    # Also fetch predictions for event info and snapshot/compare
    predictions = None
    try:
        predictions = get_predictions()
        event_info = find_current_event(predictions)
        print(f"  Course: {event_info['course']}")
    except Exception:
        event_info = {"event_name": event_name, "event_id": None, "course": "Unknown"}

    # ── Handle snapshot/compare modes ──
    if args.snapshot:
        if predictions:
            save_prediction_snapshot(predictions)
        else:
            print("  Warning: Could not save snapshot (predictions unavailable)")
        if not args.compare:
            return

    # ── Step 2: Load DraftKings salary CSV ──
    print("\n[2/7] Loading DraftKings salary data...")
    csv_path = args.csv or find_latest_csv(salaries_dir)
    if not csv_path:
        print(f"  ERROR: No CSV files found in {salaries_dir}/")
        print("  Download the DK salary CSV and place it in the salaries/ folder.")
        sys.exit(1)

    print(f"  Using: {csv_path}")
    dk_players = parse_dk_csv(csv_path)
    print(f"  Loaded {len(dk_players)} players from DraftKings")

    # ── Step 3: Match players (exact on site_name_id ↔ name_id) ──
    print("\n[3/7] Matching players...")
    matches, unmatched = match_players_exact(dk_players, dg_projections)
    print(f"  Matched: {len(matches)}/{len(dk_players)} players (exact on name_id)")
    if unmatched:
        print(f"  Unmatched ({len(unmatched)}): {', '.join(unmatched[:10])}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more")

    # ── Step 4: Weather & wave adjustment ──
    print("\n[4/7] Fetching weather forecast & wave data...")
    am_advantage, wave_details, wave_conditions = get_wave_adjustment(event_info["event_name"])

    waves_in_data = set()
    for dg_proj in dg_projections:
        w = dg_proj.get("early_late_wave")
        if w is not None:
            waves_in_data.add(w)
    has_wave_split = len(waves_in_data) >= 2

    if not has_wave_split:
        am_advantage = 0.0
        print("  Wave assignments not yet available — skipping wave adjustments")

    # ── Step 5: Line movement ──
    movement_adjustments = {}
    if args.compare and predictions:
        print("\n[5/7] Checking line movement...")
        early = load_latest_snapshot(event_info["event_name"])
        if early:
            movement = compare_snapshots(early, predictions)
            if movement:
                movement_adjustments = get_movement_adjustments(movement)
                print(f"  LINE MOVEMENT REPORT ({len(movement)} movers):")
                print(f"  {'Player':<25} {'Early MC%':>10} {'Now MC%':>10} {'Delta':>8} {'Adj':>6}")
                print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
                for m in movement[:15]:
                    adj = movement_adjustments.get(m["dg_id"], 0)
                    print(f"  {m['player_name']:<25} {m['early_mc']:>9.1f}% {m['current_mc']:>9.1f}% {m['delta_mc']:>+7.1f}% {adj:>+5.1f}")
            else:
                print("  No significant line movement detected.")
        else:
            print("  No earlier snapshot found — run with --snapshot first.")
    else:
        print("\n[5/7] Line movement — skipped (use --compare to enable)")

    # ── Step 6: Build projections ──
    print("\n[6/7] Building projections...")

    # Optional: fetch skill ratings for display
    skill_lookup = {}
    try:
        skill_data = get_skill_ratings()
        skill_list = skill_data if isinstance(skill_data, list) else skill_data.get("players", [])
        for p in skill_list:
            dg_id = p.get("dg_id")
            if dg_id is not None:
                skill_lookup[dg_id] = p
    except Exception:
        pass

    # Optional: fetch predictions for make_cut% display
    mc_lookup = {}
    if predictions:
        pred_field = predictions.get("baseline", [])
        for p in pred_field:
            dg_id = p.get("dg_id")
            mc = p.get("make_cut", 0) or 0
            if dg_id is not None:
                mc_lookup[dg_id] = mc / 100.0 if mc > 1 else mc

    projected_players = []
    for dk_idx, row in dk_players.iterrows():
        dg_match = matches.get(dk_idx)

        if dg_match:
            dg_id = dg_match.get("dg_id")

            wave_adj = 0.0
            if abs(am_advantage) >= 0.5:
                player_wave = dg_match.get("early_late_wave")
                if player_wave is not None:
                    wave_adj = (am_advantage / 2.0) if player_wave == 1 else (-am_advantage / 2.0)

            move_adj = movement_adjustments.get(dg_id, 0.0)

            player = build_projection(dg_match, wave_adj=wave_adj, movement_adj=move_adj)
            player["dg_matched"] = True

            player["p_make_cut"] = mc_lookup.get(dg_id, 0)
            sg = skill_lookup.get(dg_id, {})
            player["sg_ott"] = sg.get("sg_ott", 0) or 0
            player["sg_app"] = sg.get("sg_app", 0) or 0
            player["sg_arg"] = sg.get("sg_arg", 0) or 0
            player["sg_putt"] = sg.get("sg_putt", 0) or 0

        else:
            player = build_fallback_projection(
                row["name"], row["name_id"], row["salary"], row["avg_points"]
            )
            player["p_make_cut"] = 0
            player["sg_ott"] = 0
            player["sg_app"] = 0
            player["sg_arg"] = 0
            player["sg_putt"] = 0

        projected_players.append(player)

    projected_players.sort(key=lambda p: p["projected_points"], reverse=True)

    # Print top projections
    wave_active = abs(am_advantage) >= 0.5
    has_sg = any(p["sg_ott"] != 0 or p["sg_app"] != 0 for p in projected_players[:25])
    has_own = any(p["proj_ownership"] > 0 for p in projected_players[:25])
    has_mc = any(p["p_make_cut"] > 0 for p in projected_players[:25])
    has_move = any(p.get("movement_adj", 0) != 0 for p in projected_players[:25])

    hdr = f"  {'Player':<25} {'Salary':>8} {'Proj':>7} {'Value':>7}"
    sep = f"  {'-'*25} {'-'*8} {'-'*7} {'-'*7}"
    if has_mc:
        hdr += f" {'MC%':>6}"
        sep += f" {'-'*6}"
    if has_own:
        hdr += f" {'Own%':>6}"
        sep += f" {'-'*6}"
    hdr += f" {'StdDv':>6} {'Wave':>5}"
    sep += f" {'-'*6} {'-'*5}"
    if wave_active:
        hdr += f" {'WAdj':>6}"
        sep += f" {'-'*6}"
    if has_move:
        hdr += f" {'MAdj':>6}"
        sep += f" {'-'*6}"

    print(f"\n{hdr}")
    print(sep)
    for p in projected_players[:25]:
        line = f"  {p['name']:<25} ${p['salary']:>7,} {p['projected_points']:>7.1f} {p['value']:>7.2f}"
        if has_mc:
            mc_pct = f"{p['p_make_cut']*100:.0f}%" if p['p_make_cut'] > 0 else "  —"
            line += f" {mc_pct:>6}"
        if has_own:
            line += f" {p['proj_ownership']:>5.1f}%"
        line += f" {p.get('std_dev', 0):>6.1f} {p['wave']:>5}"
        if wave_active:
            line += f" {p['wave_adj']:>+5.1f}"
        if has_move:
            line += f" {p.get('movement_adj', 0):>+5.1f}"
        print(line)

    if has_sg:
        print(f"\n  STROKES GAINED BREAKDOWN (Top 10)")
        print(f"  {'Player':<25} {'OTT':>6} {'APP':>6} {'ARG':>6} {'PUTT':>6}")
        print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
        for p in projected_players[:10]:
            print(f"  {p['name']:<25} {p['sg_ott']:>+5.2f} {p['sg_app']:>+5.2f} {p['sg_arg']:>+5.2f} {p['sg_putt']:>+5.2f}")

    # ── Step 7: Optimize lineups + export ──
    print(f"\n[7/7] Optimizing lineups (leverage={effective_leverage})...")
    kwargs = {
        "num_lineups": effective_lineups,
        "leverage_power": effective_leverage,
        "contest_params": contest_params,
        "pool_size": args.pool_size,
    }
    if effective_exposure is not None:
        kwargs["max_exposure"] = effective_exposure

    lineups = optimize_lineup(projected_players, **kwargs)

    print(format_all_lineups(lineups))

    # Export CSV
    if lineups:
        event_slug = event_info["event_name"].replace(" ", "_").replace("'", "")
        csv_filename = f"lineups_{event_slug}_{len(lineups)}.csv"
        csv_path_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_filename)

        import csv
        with open(csv_path_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["G"] * ROSTER_SIZE)
            for lineup in lineups:
                row = [p.get("name_id", p["name"]) for p in lineup]
                writer.writerow(row)

        print(f"\n  CSV saved: {csv_path_out}")

    # Export to Google Sheets
    if args.sheets and lineups:
        print(f"\n  Exporting to Google Sheets...")
        try:
            contest_profile = profile if args.contest and contest_params else None
            contest_metrics = metrics if args.contest and contest_params else None

            sheet_url = export_to_sheets(
                event_name=event_info["event_name"],
                lineups=lineups,
                projected_players=projected_players,
                contest_profile=contest_profile,
                contest_metrics=contest_metrics,
                contest_params=contest_params,
            )
            print(f"  Google Sheet: {sheet_url}")
        except Exception as e:
            print(f"  ERROR exporting to Sheets: {e}")

    print(f"\n{'='*70}")
    print(f"  Generated {len(lineups)} lineup(s) for {event_info['event_name']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

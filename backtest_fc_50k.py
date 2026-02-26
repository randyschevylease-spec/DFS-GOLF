#!/usr/bin/env python3
"""Backtest optimizer against real 50K+ field DK GPP contests from FantasyCruncher data.

Filters FC history to large-field (50K+ entrants) contests, uses real payout
structures, real pre-tournament ownership, and scores against actual DK points.
"""
import sys, os, csv, json, time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ROSTER_SIZE, SALARY_CAP, SALARY_FLOOR

MAX_EXPOSURE = 1.0  # every player eligible for every lineup slot
from engine import generate_field, generate_candidates, select_portfolio, _get_sigma
from run_all import simulate_positions, build_payout_lookup, assign_payouts
from backtest_all import (
    build_players_from_fc, load_fc_payout_table,
    compute_actual_roi, build_synthetic_payout_table, FC_DIR
)

HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")


def get_fc_contests_50k():
    """Find all FC contests with 50K+ lineups, deduplicated by (event_id, year)."""
    map_path = os.path.join(HISTORY_DIR, "fc_event_map.json")
    with open(map_path) as f:
        fc_map = json.load(f)

    # For each PE file, find the matching AE file and get field size
    contests = []
    for pe_fname, info in fc_map.items():
        ae_fname = pe_fname.replace("Player Exposures", "All Entrants")
        ae_path = os.path.join(FC_DIR, ae_fname)
        if not os.path.exists(ae_path):
            # Try finding variant
            base = pe_fname.split("Player Exposures")[0].strip()
            for f in os.listdir(FC_DIR):
                if f.startswith(base) and "All Entrants" in f and f.endswith(".csv"):
                    ae_path = os.path.join(FC_DIR, f)
                    break
            else:
                continue

        # Count lineups
        try:
            with open(ae_path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            total_lineups = sum(int(r.get("Lineups", "1")) for r in rows)

            # Entry fee from single-lineup losers
            entry_fees = []
            for r in rows:
                if r.get("Lineups") == "1" and r.get("# Cash") == "0":
                    entry_fees.append(abs(float(r["Profit"].replace("$", "").replace(",", ""))))
            import statistics
            entry_fee = statistics.median(entry_fees) if entry_fees else 0
        except Exception as e:
            continue

        if total_lineups < 50000:
            continue

        contests.append({
            "pe_fname": pe_fname,
            "ae_fname": os.path.basename(ae_path),
            "event_id": info["event_id"],
            "year": info["year"],
            "event_name": info["event_name"],
            "match_score": info["match_score"],
            "field_size": total_lineups,
            "entry_fee": entry_fee,
            "n_users": len(rows),
        })

    # Deduplicate by (event_id, year) — keep highest match_score
    best = {}
    for c in contests:
        key = (c["event_id"], c["year"])
        if key not in best or c["match_score"] > best[key]["match_score"]:
            best[key] = c

    result = sorted(best.values(), key=lambda x: x["field_size"], reverse=True)
    return result


def backtest_single_fc(contest, params):
    """Run full optimizer pipeline against a single FC contest with real payouts."""
    event_id = contest["event_id"]
    year = contest["year"]
    name = contest["event_name"]

    # Step 1: Build players directly from FC data (no DG projections)
    players = build_players_from_fc(event_id, year)
    if not players or len(players) < 30:
        print(f"    Skip: insufficient FC data ({len(players) if players else 0} players)")
        return None

    print(f"    Players: {len(players)} | "
          f"Top proj: {players[0]['name']} ({players[0]['projected_points']:.1f}pts)")

    # Step 3: Generate opponent field (calibrated to ownership)
    opponents = generate_field(players, params["field_size"])
    print(f"    Opponents: {len(opponents):,}")

    # Step 4: Generate candidate lineups (88% projection floor, no seed)
    candidates = generate_candidates(players, pool_size=params["n_candidates"])
    n_lineups = min(params["n_lineups"], len(candidates))
    if n_lineups < 5:
        print(f"    Skip: only {len(candidates)} candidates")
        return None
    print(f"    Candidates: {len(candidates):,}")

    # Step 5: MC simulation
    positions_matrix, n_total = simulate_positions(
        candidates, opponents, players, n_sims=params["n_sims"]
    )

    # Step 6: Real FC payout table
    fc_payouts = load_fc_payout_table(event_id, year)
    if fc_payouts:
        payout_table, entry_fee, prize_pool, fc_field_size = fc_payouts
        contest_field_size = fc_field_size
        print(f"    FC payouts: fee=${entry_fee:.0f} pool=${prize_pool:,.0f} "
              f"field={contest_field_size:,}")
    else:
        # Fallback: use contest metadata for synthetic
        entry_fee = contest["entry_fee"]
        contest_field_size = contest["field_size"]
        payout_table, entry_fee, prize_pool = build_synthetic_payout_table(
            contest_field_size, entry_fee
        )
        print(f"    Synthetic payouts (FC payout parse failed): "
              f"fee=${entry_fee:.0f} field={contest_field_size:,}")

    # Step 7: Assign simulated payouts
    payouts = assign_payouts(
        positions_matrix, payout_table, n_total, params["n_sims"],
        contest_field_size=contest_field_size
    )

    # Step 8: Select portfolio
    selected = select_portfolio(
        payouts, entry_fee, n_lineups, candidates,
        n_players=len(players), max_exposure=params["max_exposure"]
    )
    selected_lineups = [candidates[i] for i in selected]

    # Step 9: Score against actual results
    result = compute_actual_roi(
        selected_lineups, opponents, players,
        payout_table, entry_fee, contest_field_size
    )

    # Metadata
    result["event_name"] = name
    result["event_id"] = event_id
    result["year"] = year
    result["n_players"] = len(players)
    result["n_candidates"] = len(candidates)
    result["contest_field_size"] = contest_field_size
    result["entry_fee_actual"] = entry_fee
    result["prize_pool"] = prize_pool
    result["fc_payouts"] = fc_payouts is not None

    # Projection accuracy
    all_proj = [players[idx]["projected_points"] for lu in selected_lineups for idx in lu]
    all_actual = [players[idx]["actual_dk_pts"] for lu in selected_lineups for idx in lu]
    result["avg_proj_pts"] = float(np.mean(all_proj))
    result["avg_actual_pts"] = float(np.mean(all_actual))
    if len(all_proj) > 2:
        result["projection_corr"] = float(np.corrcoef(all_proj, all_actual)[0, 1])
    else:
        result["projection_corr"] = 0.0

    # Candidate quality metrics
    proj_pts = np.array([p["projected_points"] for p in players])
    cand_projs = [sum(proj_pts[i] for i in c) for c in candidates]
    result["cand_proj_mean"] = float(np.mean(cand_projs))
    result["cand_proj_min"] = float(np.min(cand_projs))

    # Selected portfolio quality
    sel_projs = [sum(proj_pts[i] for i in lu) for lu in selected_lineups]
    sel_actuals = [sum(players[i]["actual_dk_pts"] for i in lu) for lu in selected_lineups]
    result["sel_proj_mean"] = float(np.mean(sel_projs))
    result["sel_actual_mean"] = float(np.mean(sel_actuals))

    # Opponent actual scores for comparison
    opp_actuals = [sum(players[i]["actual_dk_pts"] for i in lu) for lu in opponents]
    result["opp_actual_mean"] = float(np.mean(opp_actuals))
    result["sel_vs_opp_delta"] = result["sel_actual_mean"] - result["opp_actual_mean"]

    return result


def print_results(results, params):
    """Print comprehensive results summary."""
    if not results:
        print("\n  No events completed.")
        return

    rois = [r["roi_pct"] for r in results]
    cash_rates = [r["cash_rate"] for r in results]

    mean_roi = np.mean(rois)
    median_roi = np.median(rois)
    std_roi = np.std(rois)
    sharpe = mean_roi / std_roi if std_roi > 0 else 0

    total_cost = sum(r["total_cost"] for r in results)
    total_payout = sum(r["total_payout"] for r in results)
    overall_roi = (total_payout - total_cost) / total_cost * 100 if total_cost > 0 else 0

    print(f"\n{'='*80}")
    print(f"  BACKTEST: 50K+ FIELD FC CONTESTS — {len(results)} events")
    print(f"{'='*80}")

    print(f"\n  Parameters:")
    print(f"    Lineups: {params['n_lineups']} | Sims: {params['n_sims']:,} | "
          f"Opp Field: {params['field_size']:,}")
    print(f"    Max Exposure: {params['max_exposure']:.0%} | "
          f"Proj Floor: 88% | No seed (random)")

    print(f"\n  {'─'*76}")
    print(f"  Per-Event ROI:")
    print(f"    Mean ROI:      {mean_roi:+.1f}%")
    print(f"    Median ROI:    {median_roi:+.1f}%")
    print(f"    Std Dev:       {std_roi:.1f}%")
    print(f"    Sharpe:        {sharpe:.3f}")
    print(f"    Win Rate:      {sum(1 for r in rois if r > 0)}/{len(rois)} "
          f"({sum(1 for r in rois if r > 0)/len(rois)*100:.0f}%)")
    print(f"    Best:          {max(rois):+.1f}%")
    print(f"    Worst:         {min(rois):+.1f}%")

    print(f"\n  {'─'*76}")
    print(f"  Dollar Summary:")
    print(f"    Total Invested:  ${total_cost:,.0f}")
    print(f"    Total Payout:    ${total_payout:,.0f}")
    print(f"    Net P&L:         ${total_payout - total_cost:+,.0f}")
    print(f"    Overall ROI:     {overall_roi:+.1f}%")
    print(f"    Mean Cash Rate:  {np.mean(cash_rates):.1f}%")

    print(f"\n  {'─'*76}")
    print(f"  Projection & Scoring Quality:")
    avg_corr = np.mean([r["projection_corr"] for r in results])
    avg_delta = np.mean([r["sel_vs_opp_delta"] for r in results])
    print(f"    Proj-Actual Correlation:  {avg_corr:.4f}")
    print(f"    Our Score vs Opponents:   {avg_delta:+.1f} pts/lineup")
    print(f"    Avg Cand Proj:            {np.mean([r['cand_proj_mean'] for r in results]):.1f}")

    # Per-event table
    print(f"\n  {'─'*76}")
    print(f"  {'Event':<32} {'Year':>4} {'Field':>8} {'Fee':>5} {'ROI':>8} {'Cash%':>6} "
          f"{'Us':>6} {'Opp':>6} {'Δ':>6}")
    print(f"  {'-'*32} {'-'*4} {'-'*8} {'-'*5} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for r in sorted(results, key=lambda x: x["roi_pct"], reverse=True):
        print(f"  {r['event_name'][:32]:<32} {r['year']:>4} "
              f"{r['contest_field_size']:>8,} ${r['entry_fee_actual']:>4.0f} "
              f"{r['roi_pct']:>+7.1f}% {r['cash_rate']:>5.1f}% "
              f"{r['sel_actual_mean']:>6.1f} {r['opp_actual_mean']:>6.1f} "
              f"{r['sel_vs_opp_delta']:>+5.1f}")

    # Analysis: what separates winners from losers?
    winners = [r for r in results if r["roi_pct"] > 0]
    losers = [r for r in results if r["roi_pct"] <= 0]

    if winners and losers:
        print(f"\n  {'─'*76}")
        print(f"  Winners vs Losers Analysis:")
        print(f"    {'Metric':<35} {'Winners({})'.format(len(winners)):>12} "
              f"{'Losers({})'.format(len(losers)):>12}")
        print(f"    {'-'*35} {'-'*12} {'-'*12}")
        print(f"    {'Avg field size':<35} "
              f"{np.mean([r['contest_field_size'] for r in winners]):>11,.0f} "
              f"{np.mean([r['contest_field_size'] for r in losers]):>11,.0f}")
        print(f"    {'Avg n_players':<35} "
              f"{np.mean([r['n_players'] for r in winners]):>12.0f} "
              f"{np.mean([r['n_players'] for r in losers]):>12.0f}")
        print(f"    {'Proj-Actual correlation':<35} "
              f"{np.mean([r['projection_corr'] for r in winners]):>12.4f} "
              f"{np.mean([r['projection_corr'] for r in losers]):>12.4f}")
        print(f"    {'Score vs Opponents (pts)':<35} "
              f"{np.mean([r['sel_vs_opp_delta'] for r in winners]):>+11.1f} "
              f"{np.mean([r['sel_vs_opp_delta'] for r in losers]):>+11.1f}")
        print(f"    {'Cash rate':<35} "
              f"{np.mean([r['cash_rate'] for r in winners]):>11.1f}% "
              f"{np.mean([r['cash_rate'] for r in losers]):>11.1f}%")
        print(f"    {'Entry fee':<35} "
              f"${np.mean([r['entry_fee_actual'] for r in winners]):>11.0f} "
              f"${np.mean([r['entry_fee_actual'] for r in losers]):>11.0f}")

    return {
        "mean_roi": mean_roi,
        "median_roi": median_roi,
        "overall_roi": overall_roi,
        "sharpe": sharpe,
        "total_cost": total_cost,
        "total_payout": total_payout,
    }


def main():
    start = time.time()

    print("=" * 80)
    print("  DFS GOLF — FC 50K+ FIELD BACKTEST")
    print("=" * 80)

    # Find qualifying contests
    print("\n  Scanning FC data for 50K+ field contests...")
    contests = get_fc_contests_50k()
    print(f"  Found {len(contests)} unique events with 50K+ lineups:\n")

    print(f"  {'Event':<35} {'Year':>4} {'Field':>9} {'Fee':>5} {'Users':>7}")
    print(f"  {'-'*35} {'-'*4} {'-'*9} {'-'*5} {'-'*7}")
    for c in contests:
        print(f"  {c['event_name'][:35]:<35} {c['year']:>4} "
              f"{c['field_size']:>9,} ${c['entry_fee']:>4.0f} {c['n_users']:>7,}")

    params = {
        "n_candidates": 5000,
        "n_sims": 5000,
        "field_size": 50000,
        "n_lineups": 150,
        "max_exposure": MAX_EXPOSURE,
    }

    print(f"\n  Parameters: lineups={params['n_lineups']} sims={params['n_sims']:,} "
          f"opp_field={params['field_size']:,} candidates={params['n_candidates']:,}")

    # Run backtests
    results = []
    for i, contest in enumerate(contests):
        print(f"\n{'='*80}")
        print(f"  [{i+1}/{len(contests)}] {contest['event_name']} {contest['year']} "
              f"({contest['field_size']:,} lineups, ${contest['entry_fee']:.0f})")
        print(f"{'='*80}")

        t0 = time.time()
        result = backtest_single_fc(contest, params)
        elapsed = time.time() - t0

        if result is None:
            continue

        results.append(result)
        print(f"    ROI: {result['roi_pct']:+.1f}% | Cash: {result['cash_rate']:.1f}% | "
              f"Us: {result['sel_actual_mean']:.1f} vs Opp: {result['opp_actual_mean']:.1f} | "
              f"{elapsed:.0f}s")

    # Summary
    summary = print_results(results, params)

    # Save results
    results_path = os.path.join(HISTORY_DIR, "backtest_fc_50k_results.json")
    with open(results_path, "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    elapsed = time.time() - start
    print(f"\n{'='*80}")
    print(f"  Done in {elapsed:.0f}s — {len(results)} events backtested")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

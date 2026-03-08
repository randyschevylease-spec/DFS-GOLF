#!/usr/bin/env python3
"""Backtest: score optimized portfolio against real contest results.

Parses DK contest standings to get actual player scores, then scores
our optimized lineups and ranks them against the full field.

With --csv, runs a POST-LOCK simulation using real ownership from the
standings to rebuild the opponent field, isolating ownership model error
from pure variance.

Three ROI numbers:
  1. Pre-lock sim ROI  — from the original optimizer run (upload CSV)
  2. Post-lock sim ROI — re-simulated with real ownership (--csv required)
  3. Actual ROI        — scored against real contest results

  Delta (pre → post-lock) = ownership model error
  Delta (post-lock → actual) = pure variance

Usage:
    python3 backtest_roi.py \
        --standings ~/Downloads/contest-standings-188617156.csv \
        --upload showdown_188617156_upload.csv \
        --contest 188617156

    python3 backtest_roi.py \
        --standings ~/Downloads/contest-standings-188617156.csv \
        --upload showdown_188617156_upload.csv \
        --contest 188617156 \
        --csv ~/Downloads/draftkings_r3_showdown_projections.csv \
        --detail showdown_188617156_detail.csv
"""
import sys
import os
import csv
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dfs-core'))

from dk_contests import fetch_contest


# ═══════════════════════════════════════════════════════════════════════
# PARSERS
# ═══════════════════════════════════════════════════════════════════════

def parse_upload_csv(path):
    """Parse the upload/lineups CSV.

    Returns:
        lineups: list of lists of DK ID strings (6 per lineup)
        id_to_name: dict mapping DK ID string → player name
        sim_roi: simulated portfolio ROI (float, e.g. 38.7) or None
    """
    lineups = []
    id_to_name = {}
    sim_roi = None
    section = "lineups"

    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)  # G,G,G,G,G,G

        for row in reader:
            if not row or not row[0].strip():
                continue

            if row[0] == "EXPOSURE SUMMARY":
                section = "exposure"
                continue
            if row[0] == "PORTFOLIO STATS":
                section = "stats"
                continue

            if section == "lineups":
                ids = [c.strip() for c in row[:6] if c.strip()]
                if len(ids) == 6:
                    lineups.append(ids)

            elif section == "exposure":
                if row[0] == "Player":  # header row
                    continue
                name = row[0].strip()
                dk_id = row[5].strip() if len(row) > 5 else ""
                if dk_id:
                    id_to_name[dk_id] = name

            elif section == "stats":
                key = row[0].strip()
                val = row[1].strip() if len(row) > 1 else ""
                if key == "Portfolio ROI" and val:
                    sim_roi = float(val.replace("%", "").replace("+", ""))

    return lineups, id_to_name, sim_roi


def parse_detail_csv(path):
    """Parse the detail CSV for per-lineup simulated metrics.

    Returns:
        dict mapping lineup_num (int) → {players, projection, mean_payout, roi_pct}
    """
    details = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lu_num = int(row["lineup_num"])
            details[lu_num] = {
                "players": row["players"],
                "projection": float(row["projection"]),
                "mean_payout": float(row["mean_payout"]),
                "roi_pct": float(row["roi_pct"]),
            }
    return details


def parse_standings(path):
    """Parse DK contest standings CSV.

    The CSV is a wide+tall hybrid: left columns have entry data (one row per entry),
    right columns have player data (one row per player, only populated for first ~N rows).

    Returns:
        field_scores: sorted list of floats (all entry point totals, descending)
        player_actuals: dict of player_name → actual FPTS (float)
        player_ownership: dict of player_name → ownership % (float)
    """
    field_scores = []
    player_actuals = {}
    player_ownership = {}

    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if not row or not row[0].strip():
                continue

            # Left side: entry score
            try:
                points = float(row[4])
                field_scores.append(points)
            except (IndexError, ValueError):
                continue

            # Right side: player actuals (only populated for first ~72 rows)
            if len(row) > 7 and row[7].strip():
                player_name = row[7].strip()
                # FPTS column
                if len(row) > 10 and row[10].strip():
                    try:
                        player_actuals[player_name] = float(row[10])
                    except ValueError:
                        pass
                # %Drafted column
                if len(row) > 9 and row[9].strip():
                    own_str = row[9].strip()
                    if own_str.endswith("%"):
                        try:
                            player_ownership[player_name] = float(own_str.rstrip("%"))
                        except ValueError:
                            pass

    field_scores.sort(reverse=True)
    return field_scores, player_actuals, player_ownership


# ═══════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════

def score_lineups(lineups, id_to_name, player_actuals):
    """Score each lineup by summing actual FPTS.

    Returns:
        list of dicts: {names: [str], total: float, missing: [str]}
    """
    results = []
    for lu in lineups:
        names = []
        total = 0.0
        missing = []
        for dk_id in lu:
            name = id_to_name.get(dk_id)
            if name is None:
                missing.append(dk_id)
                names.append(f"?{dk_id}")
                continue
            names.append(name)
            fpts = player_actuals.get(name)
            if fpts is None:
                missing.append(name)
                total += 0.0
            else:
                total += fpts
        results.append({"names": names, "total": total, "missing": missing})
    return results


def rank_in_field(score, field_scores):
    """Rank a single score against the sorted field. Rank 1 = best."""
    # field_scores is sorted descending
    # rank = number of entries strictly above + 1
    rank = 1
    for fs in field_scores:
        if fs > score:
            rank += 1
        else:
            break
    return rank


def lookup_payout(rank, payout_table):
    """Look up payout for a given rank."""
    for min_pos, max_pos, prize in payout_table:
        if min_pos <= rank <= max_pos:
            return prize
    return 0.0


# ═══════════════════════════════════════════════════════════════════════
# POST-LOCK SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def run_postlock_sim(csv_path, lineups, id_to_name, player_ownership,
                     field_size, payout_table, entry_fee,
                     max_sims=50_000, seed=42):
    """Re-run contest simulation with real post-lock ownership.

    Uses the original projections/std_devs (same player skill model) but
    replaces synthetic ownership with real %Drafted from standings. This
    rebuilds the opponent field with correct ownership, isolating ownership
    model error from pure variance.

    Returns:
        dict with portfolio_roi, per_lu_roi, cash_rate, n_sims, etc.
    """
    from players import parse_projections, derive_std_devs
    from config import ROSTER_SIZE, SHOWDOWN_SALARY_FLOORS
    from field_generator import generate_field, field_to_index_lists
    from player_sim import generate_player_sims
    from showdown_engine import resimulate_filtered

    t0 = time.time()

    # 1. Parse projections (same as pre-lock pipeline)
    players = parse_projections(csv_path)
    derive_std_devs(players)

    # 2. Override ownership with real post-lock %Drafted
    matched = 0
    for p in players:
        real_own = player_ownership.get(p["name"])
        if real_own is not None:
            p["proj_ownership"] = real_own
            matched += 1
        else:
            p["proj_ownership"] = 0.5  # minimal for unmatched

    # Normalize total ownership to ROSTER_SIZE × 100
    total_own = sum(p["proj_ownership"] for p in players)
    target = ROSTER_SIZE * 100
    if total_own > 0:
        for p in players:
            p["proj_ownership"] *= target / total_own

    print(f"  Ownership matched: {matched}/{len(players)} players")

    # 3. Generate opponent field with real ownership
    n_opponents = max(field_size - len(lineups), 100)
    field = generate_field(
        players, n_opponents,
        min_salary_map=SHOWDOWN_SALARY_FLOORS,
        seed=seed,
    )
    opponents = field_to_index_lists(field)
    print(f"  Opponents: {len(opponents):,}")

    # 4. Generate player sims
    waves = [p["wave"] for p in players]
    player_sim = generate_player_sims(
        players, field_size, waves=waves,
        max_sims=max_sims, seed=seed,
    )
    print(f"  Player sims: {player_sim.n_sims:,} "
          f"({player_sim.generation_time:.1f}s)")

    # 5. Map our lineups from DK IDs to player indices
    dk_id_to_idx = {p["dk_id"]: i for i, p in enumerate(players)}
    candidates = []
    skipped = 0
    for lu_ids in lineups:
        try:
            idx_list = [dk_id_to_idx[dk_id] for dk_id in lu_ids]
            candidates.append(idx_list)
        except KeyError:
            skipped += 1
    if skipped:
        print(f"  WARNING: Skipped {skipped} lineups with unmappable DK IDs")

    # 6. Simulate (materialize full payout matrix — 150 × 50K × 4B = 30 MB)
    payouts_matrix = resimulate_filtered(
        player_sim, candidates, opponents, players,
        payout_table, entry_fee,
    )

    # 7. Portfolio ROI: E[max payout across lineups per sim]
    portfolio_max = payouts_matrix.max(axis=0)  # (n_sims,)
    portfolio_cost = entry_fee * len(candidates)
    portfolio_roi = (portfolio_max.mean() - portfolio_cost) / portfolio_cost * 100

    # Per-lineup sim metrics
    per_lu_mean = payouts_matrix.mean(axis=1)  # (n_candidates,)
    per_lu_roi = (per_lu_mean - entry_fee) / entry_fee * 100

    # Portfolio cash rate (% of sims where at least one lineup cashes)
    cash_rate = (portfolio_max > 0).mean() * 100

    elapsed = time.time() - t0
    print(f"  Post-lock sim complete: {elapsed:.1f}s")

    return {
        "portfolio_roi": portfolio_roi,
        "per_lu_roi": per_lu_roi,
        "cash_rate": cash_rate,
        "n_sims": player_sim.n_sims,
        "n_opponents": len(opponents),
        "n_candidates": len(candidates),
        "elapsed": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Backtest optimized portfolio against real results")
    parser.add_argument("--standings", required=True, help="DK contest standings CSV")
    parser.add_argument("--upload", required=True, help="Upload/lineups CSV (with exposure summary)")
    parser.add_argument("--contest", required=True, type=int, help="DK contest ID (for payout table)")
    parser.add_argument("--csv", default=None,
                        help="Original projections CSV (enables post-lock simulation)")
    parser.add_argument("--detail", default=None, help="Detail CSV for per-lineup sim comparison")
    parser.add_argument("--entry-fee", type=float, default=None, help="Override entry fee")
    parser.add_argument("--max-sims", type=int, default=50_000,
                        help="Max sims for post-lock simulation (default: 50,000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for post-lock sim")
    args = parser.parse_args()

    # ── Parse inputs ──────────────────────────────────────────────────
    lineups, id_to_name, sim_roi = parse_upload_csv(args.upload)
    print(f"  Loaded {len(lineups)} lineups, {len(id_to_name)} player mappings")
    if sim_roi is not None:
        print(f"  Simulated portfolio ROI: {sim_roi:+.1f}%")

    field_scores, player_actuals, player_ownership = parse_standings(args.standings)
    print(f"  Field: {len(field_scores):,} entries | {len(player_actuals)} players with actuals")

    details = None
    if args.detail:
        details = parse_detail_csv(args.detail)
        print(f"  Detail CSV: {len(details)} lineup sim records")

    # ── Fetch contest payout table ────────────────────────────────────
    contest = fetch_contest(args.contest)
    entry_fee = args.entry_fee or contest["entry_fee"]
    payout_table = contest["payouts"]

    print(f"\n  Contest: {contest['name']}")
    print(f"  Entry fee: ${entry_fee} | Field: {len(field_scores):,}")
    print(f"  Prize pool: ${contest['prize_pool']:,} | 1st: ${contest['first_place_prize']:,}")

    # ── Score lineups against actuals ─────────────────────────────────
    scored = score_lineups(lineups, id_to_name, player_actuals)

    # Check for missing players
    all_missing = set()
    for s in scored:
        all_missing.update(s["missing"])
    if all_missing:
        print(f"\n  WARNING: {len(all_missing)} unmapped/missing players: {all_missing}")

    # ── Rank and compute payouts ──────────────────────────────────────
    ranks = [rank_in_field(s["total"], field_scores) for s in scored]
    payouts = [lookup_payout(r, payout_table) for r in ranks]

    # ── Portfolio metrics ─────────────────────────────────────────────
    n = len(lineups)
    total_cost = entry_fee * n
    total_payout = sum(payouts)
    actual_roi = (total_payout - total_cost) / total_cost * 100
    cash_count = sum(1 for p in payouts if p > 0)
    cash_rate = cash_count / n * 100
    mean_payout = total_payout / n
    best_finish = min(ranks)
    worst_finish = max(ranks)
    best_score = max(s["total"] for s in scored)
    mean_score = sum(s["total"] for s in scored) / n

    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS — Contest {args.contest}")
    print(f"{'='*70}")
    print(f"  Lineups: {n}")
    print(f"  Investment: ${total_cost:.0f}")
    print(f"  Total payout: ${total_payout:.2f}")
    print(f"  Actual ROI: {actual_roi:+.1f}%")
    print(f"  Mean payout per lineup: ${mean_payout:.2f}")
    print(f"  Cash rate: {cash_rate:.1f}% ({cash_count}/{n})")
    print(f"  Best finish: {best_finish:,} / {len(field_scores):,} ({best_score:.2f} pts)")
    print(f"  Mean lineup score: {mean_score:.2f} pts")

    # ── Post-lock simulation ─────────────────────────────────────────
    postlock = None
    if args.csv:
        print(f"\n{'='*70}")
        print(f"  POST-LOCK SIMULATION (real ownership → opponent field)")
        print(f"{'='*70}")

        postlock = run_postlock_sim(
            args.csv, lineups, id_to_name, player_ownership,
            field_size=len(field_scores),
            payout_table=payout_table,
            entry_fee=entry_fee,
            max_sims=args.max_sims,
            seed=args.seed,
        )

    # ── Three-way ROI comparison ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ROI DECOMPOSITION")
    print(f"{'='*70}")

    prelock_roi = sim_roi  # from upload CSV PORTFOLIO STATS
    postlock_roi = postlock["portfolio_roi"] if postlock else None

    if prelock_roi is not None and postlock_roi is not None:
        ownership_error = postlock_roi - prelock_roi
        pure_variance = actual_roi - postlock_roi
        total_delta = actual_roi - prelock_roi

        print(f"  Pre-lock sim ROI:    {prelock_roi:+.1f}%")
        print(f"  Post-lock sim ROI:   {postlock_roi:+.1f}%")
        print(f"  Actual ROI:          {actual_roi:+.1f}%")
        print()
        print(f"  Ownership model error (pre→post):   {ownership_error:+.1f}%")
        print(f"  Pure variance (post→actual):        {pure_variance:+.1f}%")
        print(f"  Total delta (pre→actual):           {total_delta:+.1f}%")
        print()
        print(f"  Post-lock sim cash rate: {postlock['cash_rate']:.1f}%")
        print(f"  Post-lock sims: {postlock['n_sims']:,} | "
              f"Opponents: {postlock['n_opponents']:,} | "
              f"Time: {postlock['elapsed']:.1f}s")
    elif prelock_roi is not None:
        print(f"  Pre-lock sim ROI:    {prelock_roi:+.1f}%")
        print(f"  Actual ROI:          {actual_roi:+.1f}%")
        print(f"  Total delta:         {actual_roi - prelock_roi:+.1f}%")
        print(f"\n  (Add --csv to enable post-lock sim for ownership/variance decomposition)")
    else:
        print(f"  Actual ROI:          {actual_roi:+.1f}%")
        print(f"\n  (No pre-lock sim ROI found in upload CSV)")

    # ── Per-lineup detail (sorted by rank) ────────────────────────────
    lu_data = []
    for i in range(n):
        row = {
            "lu_num": i + 1,
            "rank": ranks[i],
            "score": scored[i]["total"],
            "payout": payouts[i],
            "names": scored[i]["names"],
        }
        if details and (i + 1) in details:
            d = details[i + 1]
            row["sim_proj"] = d["projection"]
            row["sim_roi"] = d["roi_pct"]
        lu_data.append(row)

    lu_data.sort(key=lambda x: x["rank"])

    has_detail = details is not None

    print(f"\n{'='*70}")
    print(f"  LINEUP DETAIL (sorted by actual finish)")
    print(f"{'='*70}")

    if has_detail:
        print(f"  {'LU':>3}  {'Rank':>7}  {'Actual':>8}  {'Proj':>7}  {'Payout':>9}  {'SimROI':>7}  Players")
        print(f"  {'—'*3}  {'—'*7}  {'—'*8}  {'—'*7}  {'—'*9}  {'—'*7}  {'—'*40}")
    else:
        print(f"  {'LU':>3}  {'Rank':>7}  {'Actual':>8}  {'Payout':>9}  Players")
        print(f"  {'—'*3}  {'—'*7}  {'—'*8}  {'—'*9}  {'—'*40}")

    show_count = min(n, 30)
    for row in lu_data[:show_count]:
        payout_str = f"${row['payout']:.0f}" if row["payout"] > 0 else "—"
        players_short = " | ".join(row["names"][:3]) + f" +{len(row['names'])-3}"
        if has_detail and "sim_proj" in row:
            print(f"  {row['lu_num']:>3}  {row['rank']:>7,}  {row['score']:>8.2f}  "
                  f"{row['sim_proj']:>7.1f}  {payout_str:>9}  {row['sim_roi']:>+6.0f}%  {players_short}")
        else:
            print(f"  {row['lu_num']:>3}  {row['rank']:>7,}  {row['score']:>8.2f}  "
                  f"{payout_str:>9}  {players_short}")

    if n > show_count:
        print(f"  ... ({n - show_count} more lineups)")

    # ── Player performance ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PLAYER ACTUALS (our roster)")
    print(f"{'='*70}")

    # Count exposure from our lineups
    from collections import Counter
    player_counts = Counter()
    for lu in lineups:
        for dk_id in lu:
            name = id_to_name.get(dk_id, f"?{dk_id}")
            player_counts[name] += 1

    player_rows = []
    for name, count in player_counts.most_common():
        fpts = player_actuals.get(name, 0.0)
        own = player_ownership.get(name)
        exposure = count / n * 100
        player_rows.append((name, fpts, exposure, count, own))

    player_rows.sort(key=lambda x: -x[1])  # sort by actual FPTS desc

    print(f"  {'Player':<25} {'FPTS':>6}  {'Exp%':>5}  {'Count':>5}  {'Own%':>5}")
    print(f"  {'—'*25} {'—'*6}  {'—'*5}  {'—'*5}  {'—'*5}")
    for name, fpts, exposure, count, own in player_rows:
        own_str = f"{own:.1f}" if own is not None else "—"
        print(f"  {name:<25} {fpts:>6.1f}  {exposure:>5.1f}  {count:>5}  {own_str:>5}")

    # ── Payout distribution ───────────────────────────────────────────
    if cash_count > 0:
        print(f"\n{'='*70}")
        print(f"  PAYOUT BREAKDOWN")
        print(f"{'='*70}")
        payout_counts = Counter()
        for p in payouts:
            if p > 0:
                payout_counts[p] += 1
        for prize, count in sorted(payout_counts.items(), reverse=True):
            print(f"  ${prize:>10,.2f}  ×{count}")
        print(f"  {'—'*20}")
        print(f"  Total: ${total_payout:,.2f}")

    print()


if __name__ == "__main__":
    main()

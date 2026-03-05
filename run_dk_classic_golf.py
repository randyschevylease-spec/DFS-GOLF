#!/usr/bin/env python3
"""DraftKings Classic Golf — Ceiling-Maximizing Portfolio Builder.

Builds lineups that maximize upside using projections CSV with CEILING, FLOOR,
mean projection, ownership, wave, and tee-time data. Uses three-layer correlation
(base + wave + course-fit) and ceiling-weighted objectives.

Usage:
    python3 -u run_dk_classic_golf.py --csv projections.csv --entries DKEntries.csv
    python3 -u run_dk_classic_golf.py --sheets          # Also push to Google Sheets
    python3 -u run_dk_classic_golf.py --sims 5000        # Fewer sims for speed
"""
import sys
import os
import csv
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dfs-core'))

from config import (ROSTER_SIZE, SALARY_CAP,
                     BASE_CORRELATION, WAVE_CORR_BOOST, FIT_CORR_BOOST,
                     SAME_WAVE_CORRELATION, DIFF_WAVE_CORRELATION,
                     PLAYER_SIM_MULTIPLIER)
from datagolf_client import get_predictions
from dk_contests import fetch_contest
from engine import (generate_candidates, simulate_contest,
                    _get_sigma, compute_mixture_params,
                    transform_mixture_scores, _rank_candidates_vectorized)
from field_generator import (generate_field as generate_field_archetypes,
                             field_to_index_lists, DEFAULT_ARCHETYPE_WEIGHTS)
from portfolio_optimizer import (optimize_portfolio, compute_cut_survival)
from run_all import (simulate_positions, build_payout_lookup, assign_payouts,
                     export_all_to_sheets)
from log_utility import compute_w_star
from adaptive_floors import get_candidate_floors, get_opponent_floors, log_floor_config

# ── Contest definitions (fees/fields pulled live from DK API) ──
CONTEST_IDS = [
    "188375740",   # $300K Drive the Green ($5, 150 entries)
    "188508560",   # $40K mini-MAX ($0.50, 150 entries)
    "188508574",   # $70K Full Round Special ($10, 18 entries)
    "188508562",   # $70K Birdie ($3, 20 entries)
    "188508570",   # $60K Dogleg SE ($33, 1 entry)
]

CSV_PATH = "/Users/rhbot/Downloads/draftkings_main_projections (7).csv"
DK_ENTRIES_PATH = "/Users/rhbot/Downloads/DKEntries (6).csv"
CEILING_FILTER = 0
CEILING_WEIGHT = 0.0
CANDIDATE_POOL = 20000
W_STAR_CAP = 2000       # w* pre-filter: keep top N candidates after Phase 1
HARVEST_COUNT = 500      # opponent lineups harvested into candidate pool
FIELD_SIZE = 50000       # total opponent field size per draw
N_FIELD_DRAWS = 3        # independent field draws for w* averaging

# Contest-tier archetype weights: >40K entrants get sharper field model
LARGE_FIELD_ARCHETYPE_WEIGHTS = {
    "sharp": 0.35,
    "chalk": 0.30,
    "content": 0.25,
    "random": 0.10,
}
LARGE_FIELD_THRESHOLD = 40000  # contests with field > this use LARGE_FIELD weights


def parse_players(csv_path):
    """Parse the projections CSV and return player list with all fields.

    Supports two CSV formats:
      - Full format: dk_name, DK ID + NAME, FLOOR, CEILING, projected_ownership, ...
      - DataGolf export: dk_name, datagolf_name, make_cut, scoring_points, ...
    Missing columns are derived from available data.
    """
    players = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        for row in reader:
            try:
                name = row["dk_name"].strip()
                dk_id = int(row["dk_id"])
                salary = int(row["dk_salary"])
                proj = float(row["total_points"])
                sd = float(row["std_dev"])
                value = float(row.get("value", 0) or 0)

                # Wave / tee time
                wave = int(row.get("early_late_wave", 0) or 0)
                tee_time = row.get("tee_time", "N/A").strip()

                # Ceiling / floor: use CSV if available, else derive from proj + std_dev
                if "CEILING" in cols and row.get("CEILING"):
                    ceiling = float(row["CEILING"])
                else:
                    ceiling = proj + 2.0 * sd

                if "FLOOR" in cols and row.get("FLOOR"):
                    floor_val = float(row["FLOOR"])
                else:
                    floor_val = max(proj - 1.0 * sd, 0)

                # Ownership
                if "projected_ownership" in cols and row.get("projected_ownership"):
                    own = float(row["projected_ownership"])
                else:
                    own = 0.0  # will be synthesized downstream

                # Name ID — DK upload format: "Player Name (DK_ID)"
                name_id = row.get("DK ID + NAME", f"{name} ({dk_id})").strip()

                # Make cut probability (from CSV if available)
                mc = 0.0
                if "MAKE CUT" in cols and row.get("MAKE CUT", "").strip():
                    mc_raw = float(row["MAKE CUT"])
                    mc = mc_raw / 100.0 if mc_raw > 1.0 else mc_raw
                elif "make_cut" in cols and row.get("make_cut", "").strip():
                    mc_raw = float(row["make_cut"])
                    mc = mc_raw / 100.0 if mc_raw > 1.0 else mc_raw

                # Decomp edge from CSV (driving_dist_adj / driving_acc_adj columns)
                drive_dist_adj = 0.0
                drive_acc_adj = 0.0
                for col in ("driving_dist_adj", "DRIVE DIST"):
                    if col in cols and row.get(col, "").strip():
                        drive_dist_adj = float(row[col])
                        break
                for col in ("driving_acc_adj", "DRIVE ACC"):
                    if col in cols and row.get(col, "").strip():
                        drive_acc_adj = float(row[col])
                        break

                # Scoring/finish breakdown for bimodal
                scoring_pts = 0.0
                if "scoring_points" in cols and row.get("scoring_points", "").strip():
                    scoring_pts = float(row["scoring_points"])

                # Determine primary edge from decomp adjustments
                if drive_dist_adj > 0 and drive_dist_adj >= drive_acc_adj:
                    primary_edge = "driving_dist"
                elif drive_acc_adj > 0:
                    primary_edge = "driving_acc"
                else:
                    primary_edge = "baseline"

                player = {
                    "name": name,
                    "name_id": name_id,
                    "dk_id": dk_id,
                    "salary": salary,
                    "projected_points": proj,
                    "std_dev": sd,
                    "tee_time": tee_time,
                    "wave": wave,
                    "floor": floor_val,
                    "ceiling": ceiling,
                    "proj_ownership": own,
                    "value": value,
                    "scoring_points": scoring_pts,
                    "primary_edge": primary_edge,
                }
                if mc > 0:
                    player["p_make_cut"] = mc

                players.append(player)
            except (ValueError, KeyError) as e:
                print(f"  Warning: skipping row {row.get('dk_name', '?')}: {e}")
    return players


def main():
    parser = argparse.ArgumentParser(description="DK Classic Golf — Ceiling Portfolio Builder")
    parser.add_argument("--sims", type=int, default=10000, help="Monte Carlo sims for harvesting (default: 10000)")
    parser.add_argument("--presim-factor", type=int, default=5,
                        help="Portfolio sim multiplier vs --sims (default: 5x)")
    parser.add_argument("--candidates", type=int, default=CANDIDATE_POOL, help="Candidate pool size")
    parser.add_argument("--sheets", action="store_true", help="Export to Google Sheets")
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Path to projections CSV")
    parser.add_argument("--entries", type=str, default=DK_ENTRIES_PATH, help="Path to DKEntries CSV")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("  DK CLASSIC GOLF — CEILING-MAXIMIZING PORTFOLIO BUILDER")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Parse & filter players
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 1: Parse & filter players from CSV")
    print(f"{'='*70}")

    all_players = parse_players(args.csv)
    print(f"  Parsed {len(all_players)} players from CSV")

    # Quality filter: remove players where CEILING < threshold
    # Skip filter if ceiling was derived (not from CSV) — threshold was calibrated for CSV ceilings
    has_csv_ceiling = any("CEILING" in open(args.csv).readline() for _ in [1])
    if has_csv_ceiling and CEILING_FILTER > 0:
        players = [p for p in all_players if p["ceiling"] >= CEILING_FILTER]
        removed = len(all_players) - len(players)
        print(f"  CEILING filter (>= {CEILING_FILTER}): removed {removed} players")
    else:
        players = all_players
        print(f"  CEILING filter: skipped (ceiling derived from proj + 2*std_dev)")
    print(f"  Active pool: {len(players)} players")

    # Sort by projected points
    players.sort(key=lambda p: p["projected_points"], reverse=True)

    # Synthesize ownership from projections if not in CSV
    # Realistic DFS golf ownership: chalk ~30%, field tapers to ~2-4%
    # Uses linear projection weighting with cap at 40% per player
    has_ownership = any(p["proj_ownership"] > 0 for p in players)
    if not has_ownership:
        projs = np.array([p["projected_points"] for p in players])
        proj_min = projs.min()
        spread = projs.max() - proj_min
        if spread < 1:
            spread = 1.0
        # Linear weights: top player gets highest, bottom gets near-zero
        linear_w = (projs - proj_min) / spread
        linear_w = np.maximum(linear_w, 0.05)  # floor so everyone has some ownership
        raw_own = linear_w / linear_w.sum() * 600  # ~600% total (6-man lineups)
        # Cap individual ownership at 40% (realistic DFS max)
        np.minimum(raw_own, 40.0, out=raw_own)
        # Re-normalize after cap to maintain ~600% total
        synth_own = raw_own / raw_own.sum() * 600
        for p, own in zip(players, synth_own):
            p["proj_ownership"] = round(float(own), 1)
        print(f"  Ownership synthesized from projections (total: {synth_own.sum():.0f}%)")
        top5 = sorted(players, key=lambda x: x["proj_ownership"], reverse=True)[:5]
        print(f"  Top ownership: {', '.join(f'{p['name']} {p['proj_ownership']:.1f}%' for p in top5)}")

    # Wave split
    am_count = sum(1 for p in players if p["wave"] == 1)
    pm_count = sum(1 for p in players if p["wave"] == 0)
    print(f"  Wave split: {am_count} AM / {pm_count} PM")

    # Print player board
    print(f"\n  {'Player':<28} {'Sal':>7} {'Proj':>6} {'Ceil':>6} {'Floor':>6} "
          f"{'Own%':>6} {'StdDv':>6} {'Val':>5} {'Wave':>4}")
    print(f"  {'-'*28} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*4}")
    for p in players[:25]:
        wave_str = "AM" if p["wave"] == 1 else "PM"
        print(f"  {p['name']:<28} ${p['salary']:>6,} {p['projected_points']:>6.1f} "
              f"{p['ceiling']:>6.1f} {p['floor']:>6.1f} {p['proj_ownership']:>5.1f}% "
              f"{p['std_dev']:>6.1f} {p['value']:>5.2f} {wave_str:>4}")
    if len(players) > 25:
        print(f"  ... {len(players)} total players")

    # Extract wave array for correlation
    waves = [p["wave"] for p in players]
    ceiling_pts = [p["ceiling"] for p in players]

    # ── Fuzzy name matching helper ──
    import re as _re_mc
    def _normalize_name(name):
        """Normalize name for fuzzy matching: strip suffixes, hyphens, middle initials, periods."""
        n = name.strip()
        n = _re_mc.sub(r'\s+(III|II|IV|Jr\.?|Sr\.?)\s*$', '', n)  # suffixes
        n = _re_mc.sub(r'\s+[A-Z]\.\s+', ' ', n)                 # middle initials like "L."
        n = _re_mc.sub(r'\s+[A-Z]\s+', ' ', n)                   # middle initials without period
        n = n.replace('-', '').replace('.', '').replace("'", '')   # hyphens, periods, apostrophes
        n = n.lower().strip()
        n = _re_mc.sub(r'\s+', ' ', n)
        return n

    # ── Make-cut probabilities ──
    # If all players have make_cut from CSV, skip DataGolf API entirely
    mc_from_csv = sum(1 for p in players if p.get("p_make_cut", 0) > 0)
    print(f"\n  make_cut from CSV: {mc_from_csv}/{len(players)} players")
    if mc_from_csv < len(players):
        # Fall back to DataGolf API for missing make_cut values
        print(f"  Fetching DataGolf make_cut for {len(players) - mc_from_csv} unmatched players...")
        try:
            predictions = get_predictions()
            mc_lookup = {}
            for entry in predictions.get("baseline", []):
                raw_name = entry.get("player_name", "")
                mc_val = entry.get("make_cut", 0)
                if mc_val and mc_val > 0:
                    if ", " in raw_name:
                        parts = raw_name.split(", ", 1)
                        clean_name = f"{parts[1]} {parts[0]}"
                    else:
                        clean_name = raw_name
                    mc_lookup[clean_name] = mc_val / 100.0 if mc_val > 1 else mc_val

            norm_lookup = {}
            for dg_name, mc_val in mc_lookup.items():
                norm_lookup[_normalize_name(dg_name)] = mc_val

            for p in players:
                if p.get("p_make_cut", 0) > 0:
                    continue
                mc = mc_lookup.get(p["name"], 0)
                if mc <= 0:
                    mc = norm_lookup.get(_normalize_name(p["name"]), 0)
                if mc > 0:
                    p["p_make_cut"] = mc
        except Exception as e:
            print(f"  Warning: Could not fetch make_cut data: {e}")
    else:
        print(f"  All players have make_cut from CSV — skipping DataGolf API")

    # ── Projection-based fallback for unmatched players ──
    # Logistic curve: median projection → 50% make_cut, higher proj → higher p
    # k calibrated so +10 pts above median ≈ 80% make_cut
    unmatched = [p for p in players if p.get("p_make_cut", 0) <= 0]
    if unmatched:
        projs = np.array([p["projected_points"] for p in players])
        median_proj = np.median(projs)
        k_logistic = 0.139  # ln(4)/10 — +10 pts → ~80%, -10 pts → ~20%
        for p in unmatched:
            z = k_logistic * (p["projected_points"] - median_proj)
            p["p_make_cut"] = 1.0 / (1.0 + np.exp(-z))
        print(f"  Fallback make_cut for {len(unmatched)}/{len(players)} players "
              f"(logistic, median_proj={median_proj:.1f})")

    # ── Edge-source diversity from CSV decomp columns ──
    EDGE_CATEGORIES = ["baseline", "driving_dist", "driving_acc"]
    player_edges = [p.get("primary_edge", "baseline") for p in players]
    edge_sources = {"primary": player_edges, "categories": EDGE_CATEGORIES}
    from collections import Counter
    edge_dist = Counter(player_edges)
    dist_str = " | ".join(f"{k} {v}" for k, v in edge_dist.most_common())
    print(f"  Edge sources (from CSV): {dist_str}")

    # Compute mixture params for bimodal scoring
    mixture_params = compute_mixture_params(players)
    n_mixture = int(mixture_params[5].sum())
    print(f"  Mixture distribution: {n_mixture}/{len(players)} players with bimodal scores")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Fetch all contest details from DK API
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 2: Fetching contest details from DraftKings API")
    print(f"{'='*70}")

    contests = []
    for cid in CONTEST_IDS:
        try:
            profile = fetch_contest(cid)
            c = {
                "cid": cid,
                "name": profile["name"],
                "field": profile["max_entries"],
                "max_entries": profile["max_entries_per_user"],
                "fee": profile["entry_fee"],
                "profile": profile,
            }
            contests.append(c)
            print(f"  {c['name'][:55]:<55} fee=${c['fee']:<6} field={c['field']:>7,} max={c['max_entries']}")
        except Exception as e:
            print(f"  ERROR fetching {cid}: {e}")

    if not contests:
        print("  No contests fetched. Exiting.")
        return

    # Sort contests by total investment (max_entries × fee) descending —
    # highest-value contests get first pick from the candidate pool
    contests.sort(key=lambda c: c["max_entries"] * c["fee"], reverse=True)
    print(f"\n  Contest priority order (by investment):")
    for i, c in enumerate(contests):
        invest = c["max_entries"] * c["fee"]
        print(f"    {i+1}. ${invest:>6,} — {c['name'][:50]}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2.5: Compute adaptive floors from primary contest
    # ══════════════════════════════════════════════════════════════════
    from mip_solver import solve_mip as _solve_mip_floor
    _base_obj = np.array([p["projected_points"] for p in players])
    _optimal = _solve_mip_floor(players, _base_obj)
    optimal_proj = sum(players[i]["projected_points"] for i in _optimal) if _optimal else 362.0

    primary = contests[0]
    payout_pct = primary["profile"]["payout_spots"] / primary["field"]

    cand_floors = get_candidate_floors(optimal_proj, primary["field"], primary["fee"], payout_pct)
    opp_floors = get_opponent_floors(optimal_proj, primary["field"], primary["fee"], payout_pct)
    log_floor_config(cand_floors, optimal_proj, label="candidates")
    log_floor_config(opp_floors, optimal_proj, label="opponents")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2.5b: Generate ceiling-weighted candidates
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 2.5b: Generate {args.candidates:,} ceiling-weighted candidates")
    sal_label = f"${cand_floors.salary_floor:,}" if cand_floors.salary_floor else "default"
    proj_label = f"{cand_floors.proj_floor:.1f}" if cand_floors.proj_floor else "none"
    print(f"  Ceiling weight: {CEILING_WEIGHT} | Salary floor: {sal_label} | "
          f"Proj floor: {proj_label}")
    print(f"{'='*70}")

    candidates = generate_candidates(
        players,
        pool_size=args.candidates,
        min_proj_pct=0.0,
        ceiling_pts=ceiling_pts,
        ceiling_weight=CEILING_WEIGHT,
        salary_floor_override=cand_floors.salary_floor,
        proj_floor_override=cand_floors.proj_floor,
    )
    print(f"  Unique candidates: {len(candidates):,}")

    # Candidate stats
    cand_projs = [sum(players[i]["projected_points"] for i in c) for c in candidates]
    cand_ceils = [sum(players[i]["ceiling"] for i in c) for c in candidates]
    cand_sals = [sum(players[i]["salary"] for i in c) for c in candidates]
    print(f"  Avg projection: {np.mean(cand_projs):.1f} | "
          f"Avg ceiling: {np.mean(cand_ceils):.1f} | "
          f"Avg salary: ${np.mean(cand_sals):,.0f}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Generate opponent fields (N_FIELD_DRAWS independent draws)
    # ══════════════════════════════════════════════════════════════════
    max_field = max(c["field"] for c in contests)
    # Pick archetype weights: large-field weights for >40K entrant contests
    arch_weights = LARGE_FIELD_ARCHETYPE_WEIGHTS if max_field > LARGE_FIELD_THRESHOLD else DEFAULT_ARCHETYPE_WEIGHTS
    print(f"\n{'='*70}")
    print(f"  STEP 3: Generate {N_FIELD_DRAWS}x {FIELD_SIZE:,} opponent fields "
          f"({N_FIELD_DRAWS * FIELD_SIZE:,} total)")
    arch_str = " | ".join(f"{k} {v*100:.0f}%" for k, v in arch_weights.items())
    print(f"  Archetype weights: {arch_str}")
    print(f"{'='*70}")

    field_draws = []  # list of (opponents, generated_field) per draw
    for draw_i in range(N_FIELD_DRAWS):
        draw_seed = 1000 + draw_i * 7  # deterministic but different per draw
        print(f"\n  ── Field draw {draw_i + 1}/{N_FIELD_DRAWS} (seed={draw_seed}) ──")
        gf = generate_field_archetypes(
            players, FIELD_SIZE,
            archetype_weights=arch_weights,
            ownership_tolerance=0.03,
            max_iterations=10,
            seed=draw_seed,
        )
        opps = field_to_index_lists(gf)
        field_draws.append((opps, gf))
        print(f"  Draw {draw_i + 1}: {len(opps):,} lineups")
        for arch, cnt in sorted(gf.archetype_distribution.items(), key=lambda x: -x[1]):
            print(f"    {arch:<12} {cnt:>7,} ({cnt/len(opps)*100:.0f}%)")

    # Ownership calibration (average across draws)
    n = len(players)
    avg_counts = np.zeros(n, dtype=np.float64)
    for opps, _ in field_draws:
        for lu in opps:
            for idx in lu:
                avg_counts[idx] += 1
    avg_counts /= N_FIELD_DRAWS
    total_opps_per_draw = len(field_draws[0][0])
    print(f"\n  Ownership calibration (averaged across {N_FIELD_DRAWS} draws, top 10):")
    print(f"  {'Player':<28} {'Target':>8} {'Sim':>8}")
    print(f"  {'-'*28} {'-'*8} {'-'*8}")
    top_own = sorted(range(n), key=lambda i: players[i]["proj_ownership"], reverse=True)[:10]
    for i in top_own:
        sim_own = avg_counts[i] / total_opps_per_draw * 100
        print(f"  {players[i]['name']:<28} {players[i]['proj_ownership']:>7.1f}% {sim_own:>7.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Multi-draw Phase 1 — score candidates vs N independent fields,
    #         average w* across draws, filter to top W_STAR_CAP per contest
    # ══════════════════════════════════════════════════════════════════
    MAX_MEMORY_GB = 40
    BYTES_PER_ELEMENT = 28

    n_players = len(players)
    n_cands = len(candidates)
    n_opps_per_draw = FIELD_SIZE

    # Sims per draw: split total budget across N_FIELD_DRAWS
    max_field = max(c["field"] for c in contests)
    field_driven_sims = max_field * PLAYER_SIM_MULTIPLIER
    total_portfolio_sims = max(args.presim_factor * args.sims, field_driven_sims)

    # Memory cap for Phase 2 (post-filter)
    n_cands_est = len(contests) * (W_STAR_CAP + HARVEST_COUNT)
    max_elements = int(MAX_MEMORY_GB * 1024**3 / BYTES_PER_ELEMENT)
    if n_cands_est * total_portfolio_sims > max_elements:
        total_portfolio_sims = max_elements // n_cands_est
        print(f"  ⚠ Memory cap: sims capped to {total_portfolio_sims:,}")

    # Phase 1 sims per draw — split evenly
    sims_per_draw = max(args.sims, total_portfolio_sims // (N_FIELD_DRAWS * 2))
    n_presim = sims_per_draw * N_FIELD_DRAWS + total_portfolio_sims

    print(f"\n{'='*70}")
    print(f"  STEP 4: Multi-draw Phase 1 ({N_FIELD_DRAWS} draws × {sims_per_draw:,} sims)")
    print(f"  Candidates: {n_cands:,} | Opponents per draw: {n_opps_per_draw:,}")
    print(f"  Phase 2 sims: {total_portfolio_sims:,}")
    print(f"  Correlation: base={BASE_CORRELATION} + wave={WAVE_CORR_BOOST} + fit={FIT_CORR_BOOST}")
    print(f"{'='*70}")

    # Build candidate matrix
    cand_matrix = np.zeros((n_cands, n_players), dtype=np.float32)
    for i, lu in enumerate(candidates):
        for idx in lu:
            cand_matrix[i, idx] = 1.0

    means = np.array([p["projected_points"] for p in players], dtype=np.float64)
    sigmas = np.array([_get_sigma(p) for p in players], dtype=np.float64)

    # Three-layer correlation: base + wave + course-fit
    waves_arr = np.array(waves)
    same_wave = (waves_arr[:, None] == waves_arr[None, :]).astype(np.float64)

    if edge_sources is not None:
        fit_labels = edge_sources["primary"]
        fit_ids = np.array([hash(l) for l in fit_labels])
        same_fit = (fit_ids[:, None] == fit_ids[None, :]).astype(np.float64)
        n_fit_groups = len(set(fit_labels))
        fit_dist = {l: fit_labels.count(l) for l in set(fit_labels)}
        fit_str = " | ".join(f"{k} {v}" for k, v in sorted(fit_dist.items(), key=lambda x: -x[1]))
        print(f"  Course-fit groups: {n_fit_groups} ({fit_str})")
    else:
        same_fit = np.zeros_like(same_wave)
        print(f"  Course-fit groups: none (no decomp data)")

    corr_matrix = BASE_CORRELATION + WAVE_CORR_BOOST * same_wave + FIT_CORR_BOOST * same_fit
    np.fill_diagonal(corr_matrix, 1.0)
    cov = np.outer(sigmas, sigmas) * corr_matrix

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += np.eye(n_players) * 1.0
        L = np.linalg.cholesky(cov)

    # ── Generate all player score scenarios upfront ──
    use_mix = n_mixture > 0
    mix_p_miss, mix_mu_miss, mix_sigma_miss, mix_mu_make, mix_sigma_make, mix_flag = mixture_params

    presim_mem_mb = n_presim * n_players * 4 / 1024 / 1024
    print(f"  Generating {n_presim:,} correlated score scenarios ({presim_mem_mb:.0f} MB)...")
    presim_scores = np.empty((n_presim, n_players), dtype=np.float32)
    rng = np.random.default_rng()
    presim_batch = 5000
    presim_start = time.time()
    for batch_start in range(0, n_presim, presim_batch):
        bs = min(presim_batch, n_presim - batch_start)
        Z = rng.standard_normal((bs, n_players))
        X = Z @ L.T
        if use_mix:
            s = transform_mixture_scores(
                X, sigmas, mix_p_miss, mix_mu_miss, mix_sigma_miss,
                mix_mu_make, mix_sigma_make, mix_flag)
        else:
            s = means[None, :] + X
            np.maximum(s, 0.0, out=s)
        presim_scores[batch_start:batch_start + bs] = s.astype(np.float32)
    presim_elapsed = time.time() - presim_start
    print(f"  Pre-sim complete in {presim_elapsed:.1f}s ({n_presim/presim_elapsed:,.0f} scenarios/sec)")

    # ── Phase 1: Run each field draw sequentially (memory-safe) ──
    # Accumulate per-contest w* arrays across draws for averaging
    w_star_accum = {}  # ci -> list of w* arrays (one per draw)
    for ci in range(len(contests)):
        w_star_accum[ci] = []

    # Also accumulate opponent scores across all draws for harvesting
    all_draw_opponents = []  # list of (opponents, opp_mean_scores) per draw
    cand_matrix_f32 = cand_matrix.astype(np.float32)
    batch_size = 500
    filter_batch = 1000
    all_original_candidates = candidates

    for draw_i in range(N_FIELD_DRAWS):
        opponents, _ = field_draws[draw_i]
        n_opps = len(opponents)
        sim_offset = draw_i * sims_per_draw  # each draw uses different score scenarios

        print(f"\n  ── Phase 1 draw {draw_i + 1}/{N_FIELD_DRAWS}: "
              f"{n_cands:,} cands vs {n_opps:,} opps × {sims_per_draw:,} sims ──")

        # Build opponent matrix for this draw
        opp_matrix = np.zeros((n_opps, n_players), dtype=np.float32)
        for i, lu in enumerate(opponents):
            for idx in lu:
                opp_matrix[i, idx] = 1.0

        # Simulate and rank
        positions_matrix = np.empty((n_cands, sims_per_draw), dtype=np.int32)
        opp_score_sum = np.zeros(n_opps, dtype=np.float64)
        sim_start = time.time()

        for b_start in range(0, sims_per_draw, batch_size):
            bs = min(batch_size, sims_per_draw - b_start)
            scores = presim_scores[sim_offset + b_start:sim_offset + b_start + bs]
            cand_scores = scores @ cand_matrix_f32.T
            opp_scores = scores @ opp_matrix.T
            opp_score_sum += opp_scores.sum(axis=0).astype(np.float64)
            positions = _rank_candidates_vectorized(opp_scores, cand_scores, n_opps, n_opps + 1)
            positions_matrix[:, b_start:b_start + bs] = positions.T

        sim_elapsed = time.time() - sim_start
        print(f"    Sim complete in {sim_elapsed:.1f}s")

        # Compute per-contest w* for this draw
        sim_field_p1 = n_opps + 1
        for ci, contest in enumerate(contests):
            c_field = contest["field"]
            c_fee = contest["fee"]
            c_payout_by_pos = build_payout_lookup(contest["profile"]["payouts"], c_field)
            pos_scale = c_field / sim_field_p1

            w_star_all = np.full(n_cands, -np.inf, dtype=np.float64)
            for fb_start in range(0, n_cands, filter_batch):
                fb_end = min(fb_start + filter_batch, n_cands)
                batch_pos = positions_matrix[fb_start:fb_end]
                scaled = np.rint(batch_pos * pos_scale).astype(np.int32)
                np.clip(scaled, 1, c_field, out=scaled)
                batch_payouts = c_payout_by_pos[scaled]
                w_batch, _, _ = compute_w_star(batch_payouts, c_fee)
                w_star_all[fb_start:fb_end] = w_batch

            w_star_accum[ci].append(w_star_all)

        # Save opponent mean scores for harvesting
        opp_mean_scores = opp_score_sum / sims_per_draw
        all_draw_opponents.append((opponents, opp_mean_scores))

        del positions_matrix, opp_matrix  # free per-draw memory

    # ── Average w* across draws and filter ──
    print(f"\n  ── Averaging w* across {N_FIELD_DRAWS} draws ──")
    per_contest_candidates = {}
    union_indices = set()

    print(f"  {'Contest':<42} {'Field':>7} {'Fee':>5} {'+w*':>6} {'Kept':>6}")
    print(f"  {'-'*42} {'-'*7} {'-'*5} {'-'*6} {'-'*6}")

    for ci, contest in enumerate(contests):
        # Stack w* arrays and average (treat -inf as -inf in mean)
        w_arrays = np.stack(w_star_accum[ci])  # (N_FIELD_DRAWS, n_cands)
        # For averaging: replace -inf with NaN, use nanmean, then back to -inf
        w_finite = np.where(w_arrays > -1e10, w_arrays, np.nan)
        avg_w_star = np.nanmean(w_finite, axis=0)
        avg_w_star = np.where(np.isnan(avg_w_star), -np.inf, avg_w_star)

        n_positive_w = int((avg_w_star > 0).sum())
        actual_cap = min(W_STAR_CAP, n_cands)
        top_idx = np.argsort(-avg_w_star)[:actual_cap]
        per_contest_candidates[ci] = [all_original_candidates[i] for i in top_idx]
        union_indices.update(top_idx.tolist())

        cname_short = contest["name"][:42]
        print(f"  {cname_short:<42} {contest['field']:>7,} ${contest['fee']:>4} "
              f"{n_positive_w:>6,} {actual_cap:>6,}")

    total_kept = sum(len(v) for v in per_contest_candidates.values())
    print(f"\n  Union of all per-contest candidates: {len(union_indices):,} unique "
          f"(of {n_cands:,} original)")
    if len(union_indices) > 0:
        print(f"  Total across contests: {total_kept:,} "
              f"(avg overlap: {total_kept / len(union_indices):.1f}x)")

    del w_star_accum  # free w* accumulator

    # ══════════════════════════════════════════════════════════════════
    # STEP 4b: Harvest top opponent lineups from ALL draws
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 4b: Harvest top field lineups from {N_FIELD_DRAWS} draws")
    print(f"{'='*70}")

    # Dedup against candidate sets
    cand_set = set()
    for ci in per_contest_candidates:
        for lu in per_contest_candidates[ci]:
            cand_set.add(tuple(sorted(lu)))

    # Merge all draw opponents, rank by mean score, harvest best unique ones
    all_opp_entries = []  # (mean_score, lineup, draw_idx)
    for draw_i, (opps, opp_means) in enumerate(all_draw_opponents):
        for oi, lu in enumerate(opps):
            all_opp_entries.append((float(opp_means[oi]), lu, draw_i))
    all_opp_entries.sort(key=lambda x: -x[0])

    harvested = []
    harvested_details = []
    for score, opp_lu, draw_i in all_opp_entries:
        opp_key = tuple(sorted(opp_lu))
        if opp_key in cand_set:
            continue
        opp_sal = sum(players[idx]["salary"] for idx in opp_lu)
        opp_proj = sum(players[idx]["projected_points"] for idx in opp_lu)
        if (opp_floors.salary_floor and opp_sal < opp_floors.salary_floor) or \
           (opp_floors.proj_floor and opp_proj < opp_floors.proj_floor):
            continue
        harvested.append(list(opp_lu))
        cand_set.add(opp_key)
        harvested_details.append({
            "score": score, "salary": opp_sal, "proj": opp_proj,
            "names": [players[idx]["name"] for idx in opp_lu],
            "draw": draw_i,
        })
        if len(harvested) >= HARVEST_COUNT:
            break

    total_opps = sum(len(opps) for opps, _ in all_draw_opponents)
    print(f"  Harvested {len(harvested)} unique field lineups (from {total_opps:,} "
          f"opponents across {N_FIELD_DRAWS} draws)")
    if harvested_details:
        print(f"  Top harvested: score={harvested_details[0]['score']:.1f} "
              f"proj={harvested_details[0]['proj']:.1f} "
              f"sal=${harvested_details[0]['salary']:,}")
        for h in harvested_details[:5]:
            names = ", ".join(h["names"][:6])
            print(f"    {h['score']:.1f} pts | ${h['salary']:,} | {h['proj']:.1f} proj | {names}")

    # Append harvested lineups to every per-contest candidate pool
    for ci in per_contest_candidates:
        n_orig = len(per_contest_candidates[ci])
        per_contest_candidates[ci] = per_contest_candidates[ci] + harvested
        print(f"  Contest {ci}: {n_orig:,} w*-filtered + {len(harvested)} harvested = "
              f"{len(per_contest_candidates[ci]):,} total")

    del all_draw_opponents  # free opponent data

    # ══════════════════════════════════════════════════════════════════
    # STEP 4c: Per-contest position simulation (Phase 2)
    #   Merge all field draws into combined opponent pool.
    #   Subsample opponents per contest so positions are direct — no scaling.
    # ══════════════════════════════════════════════════════════════════
    n_portfolio_sims = total_portfolio_sims
    phase2_offset = sims_per_draw * N_FIELD_DRAWS  # use different scenarios than Phase 1

    # Merge all field draw opponents into one combined pool
    combined_opponents = []
    for opps, _ in field_draws:
        combined_opponents.extend(opps)
    n_opps = len(combined_opponents)
    print(f"\n{'='*70}")
    print(f"  STEP 4c: Per-contest position simulation ({n_portfolio_sims:,} sims)")
    print(f"  Combined opponent pool: {n_opps:,} (from {N_FIELD_DRAWS} draws)")
    print(f"{'='*70}")

    # Build combined opponent matrix
    opp_matrix = np.zeros((n_opps, n_players), dtype=np.float32)
    for i, lu in enumerate(combined_opponents):
        for idx in lu:
            opp_matrix[i, idx] = 1.0
    opp_matrix_f32 = opp_matrix

    # Build per-contest candidate matrices
    per_contest_cand_matrix = {}
    for ci in range(len(contests)):
        ci_cands = per_contest_candidates[ci]
        n_ci = len(ci_cands)
        mat = np.zeros((n_ci, n_players), dtype=np.float32)
        for i, lu in enumerate(ci_cands):
            for idx in lu:
                mat[i, idx] = 1.0
        per_contest_cand_matrix[ci] = mat
        print(f"  Contest {ci}: {n_ci:,} candidates → matrix {mat.shape}")

    # Pre-generate fixed opponent subsets per contest
    sub_rng = np.random.default_rng(seed=42)
    opp_subsets = {}
    for ci, contest in enumerate(contests):
        if contest["field"] - 1 >= n_opps:
            opp_subsets[ci] = None  # use all opponents
        else:
            opp_subsets[ci] = sub_rng.choice(n_opps, size=contest["field"] - 1, replace=False)
        sub_size = "all" if opp_subsets[ci] is None else f"{len(opp_subsets[ci]):,}"
        print(f"  Contest {ci}: {contests[ci]['name'][:40]:<40} field={contest['field']:>7,} → "
              f"opp subset: {sub_size}")

    # Allocate per-contest position matrices
    positions_per_contest = {}
    total_pos_mem = 0
    for ci in range(len(contests)):
        n_ci = len(per_contest_candidates[ci])
        positions_per_contest[ci] = np.empty((n_ci, n_portfolio_sims), dtype=np.int32)
        total_pos_mem += n_ci * n_portfolio_sims * 4

    total_pos_mem_gb = total_pos_mem / 1024**3
    max_ci_cands = max(len(per_contest_candidates[ci]) for ci in range(len(contests)))
    print(f"  Position matrices: {len(contests)} contests, max {max_ci_cands:,} cands × "
          f"{n_portfolio_sims:,} sims = {total_pos_mem_gb:.1f}GB total")
    print(f"  Simulating {n_portfolio_sims:,} sims across {len(contests)} contests...")

    sim_start = time.time()

    for batch_start in range(0, n_portfolio_sims, batch_size):
        bs = min(batch_size, n_portfolio_sims - batch_start)
        scores = presim_scores[phase2_offset + batch_start:phase2_offset + batch_start + bs]

        opp_scores_full = scores @ opp_matrix_f32.T

        for ci in range(len(contests)):
            cand_scores = scores @ per_contest_cand_matrix[ci].T

            subset = opp_subsets[ci]
            if subset is None:
                opp_sub = opp_scores_full
                n_opp_sub = n_opps
            else:
                opp_sub = opp_scores_full[:, subset]
                n_opp_sub = len(subset)

            positions = _rank_candidates_vectorized(opp_sub, cand_scores, n_opp_sub, n_opp_sub + 1)
            positions_per_contest[ci][:, batch_start:batch_start + bs] = positions.T

        done = batch_start + bs
        if done % 10000 == 0 or done == n_portfolio_sims:
            elapsed = time.time() - sim_start
            print(f"    {done:>7,} / {n_portfolio_sims:,} sims ({elapsed:.1f}s)")

    sim_elapsed = time.time() - sim_start
    print(f"  Phase 2 complete in {sim_elapsed:.1f}s")
    for ci in range(len(contests)):
        subset = opp_subsets[ci]
        n_opp_sub = n_opps if subset is None else len(subset)
        n_ci = len(per_contest_candidates[ci])
        print(f"    Contest {ci}: positions ({n_ci:,} × {n_portfolio_sims:,}) "
              f"(ranks among {n_opp_sub + 1})")

    del presim_scores
    del per_contest_cand_matrix

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Loop over each contest — payout assignment + portfolio selection
    #   Each contest uses its own per-contest candidate pool and cut_survival.
    #   Cross-contest exclusion is by lineup identity (not index).
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 5: Contest-specific portfolio selection")
    print(f"{'='*70}")

    results = []
    # Derive event name from contest name (strip "PGA TOUR" prefix and prize info)
    import re as _re_evt
    raw_name = contests[0]["name"] if contests else "Unknown"
    evt_match = _re_evt.match(r'PGA TOUR\s+(.*?)(?:\s*\[)', raw_name)
    event_name = evt_match.group(1).strip() if evt_match else raw_name
    for ci, contest_def in enumerate(contests):
        cid = contest_def["cid"]
        cname = contest_def["name"]
        field_size = contest_def["field"]
        max_entries = contest_def["max_entries"]
        entry_fee = contest_def["fee"]
        profile = contest_def["profile"]
        payout_table = profile["payouts"]
        ci_candidates = per_contest_candidates[ci]

        print(f"\n  {'─'*60}")
        print(f"  [{ci+1}/{len(contests)}] {cname[:50]}")
        print(f"  Fee: ${entry_fee} | Field: {field_size:,} | Max entries: {max_entries}")
        print(f"  Candidates: {len(ci_candidates):,} (per-contest pool)")
        print(f"  Payout spots: {profile['payout_spots']:,} | "
              f"1st: ${profile['first_place_prize']:,.0f} | "
              f"Pool: ${profile['prize_pool']:,.0f}")
        print(f"  {'─'*60}")

        # Build payout lookup matching simulated field size
        subset = opp_subsets[ci]
        sim_field = (n_opps + 1) if subset is None else (len(subset) + 1)
        payout_by_pos = build_payout_lookup(payout_table, sim_field)

        # Direct positions from per-contest opponent subsample (no scaling)
        payouts = payout_by_pos[positions_per_contest[ci]]  # (n_ci_cands, n_portfolio_sims)

        roi = (payouts.mean(axis=1) - entry_fee) / entry_fee * 100

        print(f"  Candidate ROI: mean={roi.mean():+.1f}% | best={roi.max():+.1f}% | "
              f"+EV={int((roi > 0).sum())}/{len(roi)}")

        # w* log-utility diagnostics
        w_star, p_cash, kelly_frac = compute_w_star(payouts, entry_fee)
        finite_w = w_star[w_star > -np.inf]
        if len(finite_w) > 0:
            print(f"  w* stats: best={w_star.max():.6f} | median={np.median(finite_w):.6f} | "
                  f"+w*={int((w_star > 0).sum())}/{len(w_star)}")

        # Ownership-adjusted w*: discount by expected duplicate count
        # adj_w* = raw_w* - ln(1 + expected_dupes) × penalty
        DUPE_PENALTY_WEIGHT = 0.05
        expected_dupes_arr = np.zeros(len(ci_candidates), dtype=np.float64)
        for ci_i, lu in enumerate(ci_candidates):
            p_exact = 1.0
            for pidx in lu:
                p_exact *= max(players[pidx]["proj_ownership"], 0.01) / 100.0
            expected_dupes_arr[ci_i] = p_exact * field_size
        adj_w_star = w_star - np.log1p(expected_dupes_arr) * DUPE_PENALTY_WEIGHT
        n_adj = int((adj_w_star < w_star).sum())
        print(f"  Ownership-adjusted w*: {n_adj}/{len(w_star)} penalized "
              f"(max expected dupes: {expected_dupes_arr.max():.1f})")

        # Compute cut_survival for THIS contest's candidate pool (freed after each iteration)
        cut_survival = compute_cut_survival(ci_candidates, players, n_portfolio_sims)

        # Duplicate budget: track high-dupe lineups in greedy selector
        # Max 20% of portfolio with >10 expected dupes, max 5% with >25 expected dupes
        DUPE_BUDGET_HIGH = 0.20   # max 20% of portfolio with >10 expected dupes
        DUPE_BUDGET_VERY_HIGH = 0.05  # max 5% with >25 expected dupes
        dupe_high_mask = expected_dupes_arr > 10
        dupe_very_high_mask = expected_dupes_arr > 25
        n_dupe_high = int(dupe_high_mask.sum())
        n_dupe_very_high = int(dupe_very_high_mask.sum())
        max_high_dupe = max(1, int(max_entries * DUPE_BUDGET_HIGH))
        max_very_high_dupe = max(1, int(max_entries * DUPE_BUDGET_VERY_HIGH))
        if n_dupe_high > 0:
            print(f"  Dupe budget: {n_dupe_high} candidates >10 dupes "
                  f"(cap: {max_high_dupe}/{max_entries}), "
                  f"{n_dupe_very_high} >25 dupes "
                  f"(cap: {max_very_high_dupe}/{max_entries})")

        # Apply dupe penalty to payouts: scale down high-dupe lineups slightly
        # This biases the greedy selector away from heavily duplicated lineups
        dupe_scale = 1.0 / (1.0 + expected_dupes_arr * 0.01)
        payouts_adj = payouts * dupe_scale[:, None]

        # Select portfolio via optimizer
        portfolio = optimize_portfolio(
            payouts_adj, entry_fee, max_entries, ci_candidates,
            n_players=n_players,
            method="greedy",
            diversity_weight=0.4,
            waves=waves,
            min_early_pct=0.3,
            min_late_pct=0.2,
            cut_survival=cut_survival,
            edge_sources=edge_sources,
            edge_diversity_weight=5.0,
        )

        selected = portfolio.selected_indices

        # Free cut_survival for this contest (saves ~1.2GB vs holding all simultaneously)
        del cut_survival

        # Build lineups with player dicts for CSV export
        lineups = []
        for sel_idx in selected:
            player_indices = ci_candidates[sel_idx]
            lineup = [players[i].copy() for i in player_indices]
            lineup.sort(key=lambda p: p["salary"], reverse=True)
            lineups.append(lineup)

        # Portfolio metrics from optimizer output
        total_cost = entry_fee * len(selected)

        # Per-lineup stats (from raw payouts for display)
        sel_payouts = payouts[selected]
        lineup_stats = []
        for li, (sel_idx, lineup) in enumerate(zip(selected, lineups)):
            lu_payouts = sel_payouts[li]
            lu_roi = float((lu_payouts.mean() - entry_fee) / entry_fee * 100)
            lu_cash = float((lu_payouts > 0).mean() * 100)
            lu_sal = sum(p["salary"] for p in lineup)
            lu_proj = sum(p["projected_points"] for p in lineup)
            lu_ceil = sum(p["ceiling"] for p in lineup)
            owns = [max(p["proj_ownership"], 0.01) for p in lineup]
            lu_geomean_own = float(np.exp(np.mean(np.log(owns))))
            # Expected duplicates: P(exact match) × field_size
            p_exact = 1.0
            for p in lineup:
                p_exact *= max(p["proj_ownership"], 0.01) / 100.0
            expected_dupes = p_exact * field_size
            lineup_stats.append({
                "roi": lu_roi,
                "cash_rate": lu_cash,
                "salary": lu_sal,
                "projection": lu_proj,
                "ceiling": lu_ceil,
                "geomean_own": lu_geomean_own,
                "expected_dupes": expected_dupes,
                "players": [p["name"] for p in lineup],
            })

        avg_proj = np.mean([s["projection"] for s in lineup_stats])
        avg_ceil = np.mean([s["ceiling"] for s in lineup_stats])
        avg_own = np.mean([s["geomean_own"] for s in lineup_stats])

        # Exposure
        exposure = {}
        for lu in lineups:
            for p in lu:
                exposure[p["name"]] = exposure.get(p["name"], 0) + 1

        print(f"  Portfolio: {len(lineups)} lineups | Cost: ${total_cost:,.0f}")
        print(f"  E[max] ROI: {portfolio.expected_roi:+.1f}% | Cash: {portfolio.cash_rate:.1f}%")
        print(f"  Avg proj: {avg_proj:.1f} | Avg ceiling: {avg_ceil:.1f} | Avg own: {avg_own:.1f}%")
        print(f"  Dead lineup rate: {portfolio.dead_lineup_rate:.1f}%")
        print(f"  Wave split: AM {portfolio.wave_split['early']:.0f}% / PM {portfolio.wave_split['late']:.0f}%")
        if portfolio.edge_source_split:
            edge_str = " | ".join(f"{k} {v:.0f}%" for k, v in
                                   sorted(portfolio.edge_source_split.items(), key=lambda x: -x[1]))
            print(f"  Edge sources: {edge_str}")

        top_exp = sorted(exposure.items(), key=lambda x: -x[1])[:5]
        top_exp_str = ", ".join(f"{name} {cnt/len(lineups)*100:.0f}%" for name, cnt in top_exp)
        print(f"  Top exposure: {top_exp_str}")

        # Save CSV — derive short slug from contest name
        import re as _re
        slug_match = _re.search(r'\$(\d+[KMkm]?)', cname)
        slug = slug_match.group(1).lower() if slug_match else cid
        if max_entries == 1:
            slug += "_se"
        csv_filename = f"lineups_{cid}_{len(lineups)}.csv"
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_filename)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["G"] * ROSTER_SIZE + ["Salary", "Projection", "Ceiling", "ROI%", "Cash%", "GeoOwn%"])
            for lineup, stats in zip(lineups, lineup_stats):
                writer.writerow(
                    [p.get("name_id", p["name"]) for p in lineup] +
                    [stats["salary"], round(stats["projection"], 1), round(stats["ceiling"], 1),
                     round(stats["roi"], 1), round(stats["cash_rate"], 1), round(stats["geomean_own"], 2)]
                )
        print(f"  CSV: {csv_filename}")

        results.append({
            "contest_id": cid,
            "name": f"{cname} (${entry_fee})",
            "entry_fee": entry_fee,
            "n_lineups": len(lineups),
            "total_cost": total_cost,
            "expected_roi": portfolio.expected_roi,
            "cash_rate": portfolio.cash_rate,
            "lineups": lineups,
            "lineup_stats": lineup_stats,
            "profile": profile,
            "avg_proj": avg_proj,
            "avg_ceiling": avg_ceil,
            "avg_ownership": avg_own,
            "dead_lineup_rate": portfolio.dead_lineup_rate,
            "wave_split": portfolio.wave_split,
        })

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: Summary
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PORTFOLIO SUMMARY — ALL CONTESTS")
    print(f"{'='*70}")

    print(f"\n  {'Contest':<30} {'Fee':>5} {'LU':>4} {'Cost':>8} {'ROI':>8} {'Cash':>6} "
          f"{'AvgPrj':>7} {'AvgCeil':>8} {'AvgOwn':>7}")
    print(f"  {'-'*30} {'-'*5} {'-'*4} {'-'*8} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*7}")

    total_investment = 0
    total_expected = 0
    for r in results:
        print(f"  {r['name']:<30} ${r['entry_fee']:>4} {r['n_lineups']:>4} "
              f"${r['total_cost']:>7,} {r['expected_roi']:>+7.1f}% {r['cash_rate']:>5.1f}% "
              f"{r['avg_proj']:>7.1f} {r['avg_ceiling']:>8.1f} {r['avg_ownership']:>6.1f}%")
        total_investment += r["total_cost"]
        total_expected += r["total_cost"] * (1 + r["expected_roi"] / 100)

    total_profit = total_expected - total_investment
    overall_roi = total_profit / total_investment * 100 if total_investment > 0 else 0

    print(f"\n  Total Investment: ${total_investment:,.0f}")
    print(f"  Expected Profit:  ${total_profit:+,.0f}")
    print(f"  Overall ROI:      {overall_roi:+.1f}%")

    # Combined golfer exposure
    global_exposure = {}
    total_lineups = sum(r["n_lineups"] for r in results)
    for r in results:
        for lu in r["lineups"]:
            for p in lu:
                global_exposure[p["name"]] = global_exposure.get(p["name"], 0) + 1

    print(f"\n  COMBINED PLAYER EXPOSURE ({total_lineups} total lineups)")
    print(f"  {'Player':<28} {'Count':>6} {'Pct':>7}")
    print(f"  {'-'*28} {'-'*6} {'-'*7}")
    for name, cnt in sorted(global_exposure.items(), key=lambda x: -x[1])[:20]:
        print(f"  {name:<28} {cnt:>6} {cnt/total_lineups*100:>6.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: Portfolio Diversity & Duplication Diagnostics
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PORTFOLIO DIVERSITY & DUPLICATION DIAGNOSTICS")
    print(f"{'='*70}")

    for ri, r in enumerate(results):
        cname = r["name"][:40]
        lineups = r["lineups"]
        stats = r.get("lineup_stats", [])
        n_lu = len(lineups)
        if n_lu < 2:
            continue
        field_size = r["profile"]["max_entries"] if "profile" in r else 10000

        print(f"\n  ── {cname} ({n_lu} lineups) ──")

        # 1. Exact duplicate check
        lu_tuples = [tuple(sorted(p["name"] for p in lu)) for lu in lineups]
        n_unique = len(set(lu_tuples))
        n_exact_dupes = n_lu - n_unique
        print(f"  Exact duplicates: {n_exact_dupes} ({n_unique}/{n_lu} unique)")

        # 2. Pairwise overlap matrix
        from itertools import combinations as _comb
        overlap_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        for i, j in _comb(range(n_lu), 2):
            shared = len(set(lu_tuples[i]) & set(lu_tuples[j]))
            overlap_counts[shared] = overlap_counts.get(shared, 0) + 1
        n_pairs = n_lu * (n_lu - 1) // 2
        near_dupes = overlap_counts.get(5, 0) + overlap_counts.get(6, 0)
        mean_overlap = sum(k * v for k, v in overlap_counts.items()) / max(n_pairs, 1)
        print(f"  Pairwise overlap ({n_pairs:,} pairs): mean={mean_overlap:.2f} players")
        for k in range(7):
            if overlap_counts[k] > 0:
                pct = overlap_counts[k] / n_pairs * 100
                flag = " ⚠" if k >= 5 else ""
                print(f"    {k} shared: {overlap_counts[k]:>6,} ({pct:.1f}%){flag}")
        if near_dupes > 0:
            print(f"  ⚠ {near_dupes} near-duplicate pairs (5-6 shared players)")

        # 3. Expected duplicates distribution
        if stats:
            dupes = [s.get("expected_dupes", 0) for s in stats]
            high_dupe = sum(1 for d in dupes if d > 10)
            very_high = sum(1 for d in dupes if d > 25)
            print(f"  Expected dupes: mean={np.mean(dupes):.2f} | max={max(dupes):.1f}")
            print(f"    >10 expected dupes: {high_dupe}/{n_lu} "
                  f"({high_dupe/n_lu*100:.0f}%, budget: 20%)")
            print(f"    >25 expected dupes: {very_high}/{n_lu} "
                  f"({very_high/n_lu*100:.0f}%, budget: 5%)")

        # 4. Outcome correlation & effective portfolio size
        # Use sim payouts to compute correlation matrix
        ci_idx = ri  # contest index matches result index
        if ci_idx in positions_per_contest:
            ci_cands = per_contest_candidates[ci_idx]
            ci_selected = r.get("_selected_indices", None)
            # We need the selected indices within the per-contest pool
            # Reconstruct from lineup identity matching
            sel_lu_keys = [tuple(sorted(p["name"] for p in lu)) for lu in lineups]
            ci_lu_keys = [tuple(sorted(players[idx]["name"] for idx in lu)) for lu in ci_cands]
            key_to_idx = {}
            for ki, k in enumerate(ci_lu_keys):
                if k not in key_to_idx:
                    key_to_idx[k] = ki
            sel_idx_in_pool = [key_to_idx.get(k, -1) for k in sel_lu_keys]
            valid_sel = [i for i in sel_idx_in_pool if i >= 0]

            if len(valid_sel) >= 2:
                # Get payout rows for selected lineups
                subset = opp_subsets[ci_idx]
                sim_field = (n_opps + 1) if subset is None else (len(subset) + 1)
                payout_by_pos = build_payout_lookup(
                    r["profile"]["payouts"], sim_field)
                sel_positions = positions_per_contest[ci_idx][valid_sel]
                sel_payouts_mat = payout_by_pos[sel_positions]  # (n_sel, n_sims)

                # Pairwise Pearson correlation
                corr = np.corrcoef(sel_payouts_mat)
                # Extract upper triangle (exclude diagonal)
                triu_idx = np.triu_indices(len(valid_sel), k=1)
                pairwise_r = corr[triu_idx]
                n_high_corr = int((pairwise_r > 0.80).sum())

                print(f"  Outcome correlation ({len(valid_sel)} lineups):")
                print(f"    Mean r={np.mean(pairwise_r):.3f} | "
                      f"Max r={np.max(pairwise_r):.3f} | "
                      f"Pairs r>0.80: {n_high_corr}")

                # Effective portfolio size via eigenvalue decomposition
                eigvals = np.linalg.eigvalsh(corr)
                eigvals = np.maximum(eigvals, 0)  # numerical safety
                eigvals_sorted = np.sort(eigvals)[::-1]
                cum_var = np.cumsum(eigvals_sorted) / eigvals_sorted.sum()
                eff_size_90 = int(np.searchsorted(cum_var, 0.90)) + 1
                eff_size_95 = int(np.searchsorted(cum_var, 0.95)) + 1
                print(f"    Effective portfolio size: "
                      f"{eff_size_90} (90% var) / {eff_size_95} (95% var) "
                      f"of {len(valid_sel)} lineups")

    # ── Export to DKEntries CSV for direct upload ──
    if results and args.entries:
        print(f"\n{'='*70}")
        print(f"  EXPORTING TO DK ENTRIES CSV")
        print(f"{'='*70}")
        try:
            # Read original DKEntries template
            with open(args.entries, "r") as f:
                dk_reader = csv.reader(f)
                dk_header = next(dk_reader)
                dk_rows = list(dk_reader)

            # Build contest_id → optimized lineups mapping
            # Each lineup stored as list of "Name (DK_ID)" strings
            contest_lineups = {}
            for r in results:
                cid = r["contest_id"]
                lineup_name_ids = []
                for lu in r["lineups"]:
                    name_ids = [p.get("name_id", f"{p['name']} ({p['dk_id']})") for p in lu]
                    lineup_name_ids.append(name_ids)
                contest_lineups[cid] = lineup_name_ids

            # Assign optimized lineups to DKEntries rows
            # Track how many lineups we've assigned per contest
            contest_lineup_idx = {cid: 0 for cid in contest_lineups}
            assigned = 0
            for row in dk_rows:
                if len(row) < 10:
                    continue
                row_cid = row[2].strip() if len(row) > 2 else ""
                if row_cid in contest_lineups:
                    idx = contest_lineup_idx[row_cid]
                    lus = contest_lineups[row_cid]
                    if idx < len(lus):
                        # Replace columns 4-9 (G slots) with optimized lineup
                        for slot in range(6):
                            row[4 + slot] = lus[idx][slot]
                        contest_lineup_idx[row_cid] += 1
                        assigned += 1

            # Write updated DKEntries
            dk_output = args.entries.replace(".csv", "_optimized.csv")
            with open(dk_output, "w", newline="") as f:
                dk_writer = csv.writer(f)
                dk_writer.writerow(dk_header)
                dk_writer.writerows(dk_rows)
            print(f"  Assigned {assigned} optimized lineups across {len(contest_lineups)} contests")
            for cid, idx in contest_lineup_idx.items():
                cname = next((r["name"] for r in results if r["contest_id"] == cid), cid)
                print(f"    {cname[:45]}: {idx} lineups")
            print(f"  Output: {dk_output}")
        except Exception as e:
            print(f"  DKEntries export ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ── Export to Google Sheets ──
    if args.sheets and results:
        print(f"\n{'='*70}")
        print(f"  EXPORTING TO GOOGLE SHEETS")
        print(f"{'='*70}")
        try:
            sheet_url = export_all_to_sheets(
                event_name=event_name,
                results=results,
                players=players,
                global_exposure=global_exposure,
                total_lineups=total_lineups,
            )
            print(f"\n  Spreadsheet: {sheet_url}")

            # Add per-contest Detail tabs with lineup stats
            import re as _re
            import gspread
            from google.oauth2.service_account import Credentials

            SHEETS_CREDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials.json")
            creds = Credentials.from_service_account_file(SHEETS_CREDS, scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ])
            client = gspread.authorize(creds)
            spreadsheet = client.open_by_key("12YBOTHuRWX6EK6mGz6Uu-VfstwtRWuHlGwl3Zg5UHbc")

            HEADER_FMT = {
                "backgroundColor": {"red": 0.15, "green": 0.15, "blue": 0.15},
                "textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1}, "bold": True},
                "horizontalAlignment": "CENTER",
            }

            for ri, r in enumerate(results):
                stats = r.get("lineup_stats", [])
                if not stats:
                    continue

                # Derive tab prefix
                m = _re.search(r'\$(\d+[KMkm]?)', r["name"])
                tab_prefix = m.group(1).upper() if m else f"C{ri+1}"
                max_e = r["profile"].get("max_entries_per_user", 0)
                if max_e == 1:
                    tab_prefix += " SE"

                tab_name = f"{tab_prefix} | Detail"

                # Ensure unique
                existing = [ws.title for ws in spreadsheet.worksheets()]
                if tab_name in existing:
                    tab_name = f"{tab_prefix} ${r['entry_fee']} | Detail"

                n_rows = len(stats) + 2
                ws = spreadsheet.add_worksheet(title=tab_name, rows=n_rows, cols=13)

                detail_header = [
                    "LU#", "G1", "G2", "G3", "G4", "G5", "G6",
                    "Salary", "Projection", "Ceiling", "ROI %", "Cash %", "GeoOwn%"
                ]
                rows = [detail_header]
                for i, s in enumerate(stats, 1):
                    rows.append([
                        i,
                        s["players"][0] if len(s["players"]) > 0 else "",
                        s["players"][1] if len(s["players"]) > 1 else "",
                        s["players"][2] if len(s["players"]) > 2 else "",
                        s["players"][3] if len(s["players"]) > 3 else "",
                        s["players"][4] if len(s["players"]) > 4 else "",
                        s["players"][5] if len(s["players"]) > 5 else "",
                        s["salary"],
                        round(s["projection"], 1),
                        round(s["ceiling"], 1),
                        round(s["roi"], 1),
                        round(s["cash_rate"], 1),
                        round(s["geomean_own"], 2),
                    ])

                ws.update(rows, value_input_option="RAW")
                ws.format("A1:M1", HEADER_FMT)
                print(f"    Detail tab: {tab_name} ({len(stats)} lineups)")
                time.sleep(3)

        except Exception as e:
            print(f"\n  Sheets ERROR: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Done in {elapsed:.0f}s — {total_lineups} lineups across {len(results)} contests")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

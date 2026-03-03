#!/usr/bin/env python3
"""Cognizant Classic — Ceiling-Maximizing Portfolio Builder.

Builds lineups that maximize upside using projections CSV with CEILING, FLOOR,
mean projection, ownership, wave, and tee-time data. Uses wave-aware correlation
(same-wave players share weather/conditions) and ceiling-weighted objectives.

Simulates against all 6 Cognizant Classic contests:
  - Main (188255232): 71K field, 150 entries, $25
  - Albatross SE (188323193): 3.9K field, 1 entry, $25
  - mini-MAX (188323207): 95K field, 150 entries, $5
  - Birdie (188323208): 27.7K field, 20 entries, $15
  - Dogleg SE (188323215): 2.1K field, 1 entry, $15
  - Full Round (188323221): 4.7K field, 18 entries, $10

Usage:
    python3 -u run_cognizant.py
    python3 -u run_cognizant.py --sheets          # Also push to Google Sheets
    python3 -u run_cognizant.py --sims 5000        # Fewer sims for speed
"""
import sys
import os
import csv
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (ROSTER_SIZE, SALARY_CAP,
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
    "188508560",   # mini-MAX (Arnold Palmer Invitational)
]

CSV_PATH = "/Users/rhbot/Downloads/draftkings_main_projections (5).csv"
DECOMP_CSV = "/Users/rhbot/Downloads/dg_decomposition (1).csv"
CEILING_FILTER = 110
CEILING_WEIGHT = 0.0
CANDIDATE_POOL = 20000


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

                # Name ID
                name_id = row.get("DK ID + NAME", f"{dk_id}:{name}").strip()

                # Make cut probability (from CSV if available)
                mc = 0.0
                if "make_cut" in cols and row.get("make_cut", "").strip():
                    mc_raw = float(row["make_cut"])
                    mc = mc_raw / 100.0 if mc_raw > 1.0 else mc_raw

                # Scoring/finish breakdown for bimodal
                scoring_pts = 0.0
                if "scoring_points" in cols and row.get("scoring_points", "").strip():
                    scoring_pts = float(row["scoring_points"])

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
                }
                if mc > 0:
                    player["p_make_cut"] = mc

                players.append(player)
            except (ValueError, KeyError) as e:
                print(f"  Warning: skipping row {row.get('dk_name', '?')}: {e}")
    return players


def main():
    parser = argparse.ArgumentParser(description="Cognizant Classic — Ceiling Portfolio Builder")
    parser.add_argument("--sims", type=int, default=10000, help="Monte Carlo sims for harvesting (default: 10000)")
    parser.add_argument("--presim-factor", type=int, default=5,
                        help="Portfolio sim multiplier vs --sims (default: 5x)")
    parser.add_argument("--candidates", type=int, default=CANDIDATE_POOL, help="Candidate pool size")
    parser.add_argument("--sheets", action="store_true", help="Export to Google Sheets")
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Path to projections CSV")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("  COGNIZANT CLASSIC — CEILING-MAXIMIZING PORTFOLIO BUILDER")
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

    # ── Fetch P(make_cut) from DataGolf for mixture distribution ──
    # Players with make_cut already set from CSV skip the API lookup
    mc_from_csv = sum(1 for p in players if p.get("p_make_cut", 0) > 0)
    if mc_from_csv > 0:
        print(f"\n  make_cut from CSV: {mc_from_csv}/{len(players)} players")
    print(f"  Fetching DataGolf make_cut probabilities...")
    mc_matched = mc_from_csv
    try:
        predictions = get_predictions()
        # Build lookup: "First Last" → make_cut probability
        mc_lookup = {}
        for entry in predictions.get("baseline", []):
            raw_name = entry.get("player_name", "")
            mc_val = entry.get("make_cut", 0)
            if mc_val and mc_val > 0:
                # Convert "Last, First" → "First Last"
                if ", " in raw_name:
                    parts = raw_name.split(", ", 1)
                    clean_name = f"{parts[1]} {parts[0]}"
                else:
                    clean_name = raw_name
                # DG returns make_cut as %, convert to probability
                mc_lookup[clean_name] = mc_val / 100.0 if mc_val > 1 else mc_val

        # Build fuzzy matching indexes for unmatched players
        # Build normalized DG lookup + last-name-only lookup
        norm_lookup = {}
        norm_to_name = {}  # normalized → original DG name (for logging)
        last_name_lookup = {}  # last_name → [(full_name, mc_val)]
        for dg_name, mc_val in mc_lookup.items():
            normed = _normalize_name(dg_name)
            norm_lookup[normed] = mc_val
            norm_to_name[normed] = dg_name
            # Use normalized last name (handles "Capan III" → last token "iii" vs real last "capan")
            normed_last = normed.split()[-1] if normed.split() else ""
            last_name_lookup.setdefault(normed_last, []).append((dg_name, mc_val))

        fuzzy_matched = []
        for p in players:
            # Skip if already set from CSV
            if p.get("p_make_cut", 0) > 0:
                continue
            # 1. Exact match
            mc = mc_lookup.get(p["name"], 0)
            if mc > 0:
                p["p_make_cut"] = mc
                mc_matched += 1
                continue

            # 2. Normalized match (handles hyphens, suffixes, middle initials)
            p_norm = _normalize_name(p["name"])
            mc = norm_lookup.get(p_norm, 0)
            if mc > 0:
                p["p_make_cut"] = mc
                mc_matched += 1
                fuzzy_matched.append(f"{p['name']} → {norm_to_name[p_norm]}")
                continue

            # 3. Last-name unique match (if exactly one DG player shares the last name)
            p_last = p_norm.split()[-1] if p_norm.split() else ""
            candidates_ln = last_name_lookup.get(p_last, [])
            if len(candidates_ln) == 1:
                p["p_make_cut"] = candidates_ln[0][1]
                mc_matched += 1
                fuzzy_matched.append(f"{p['name']} → {candidates_ln[0][0]}")
                continue

            # 4. Last-name + first-name-prefix or initial match
            if len(candidates_ln) > 1:
                p_first = p_norm.split()[0] if p_norm.split() else ""
                for dg_name, mc_val in candidates_ln:
                    dg_norm = _normalize_name(dg_name)
                    dg_first = dg_norm.split()[0] if dg_norm.split() else ""
                    # Match if one first name starts with the other (Dan↔Daniel)
                    if (p_first.startswith(dg_first) or dg_first.startswith(p_first)) and len(min(p_first, dg_first)) >= 2:
                        p["p_make_cut"] = mc_val
                        mc_matched += 1
                        fuzzy_matched.append(f"{p['name']} → {dg_name}")
                        break
                    # Match by initials: "Seonghyeon" → "S.H." (first letter of DG matches)
                    # or DG initials expand to player first name initial
                    if p_first[0] == dg_first[0] and (len(dg_first) <= 2 or len(p_first) <= 2):
                        p["p_make_cut"] = mc_val
                        mc_matched += 1
                        fuzzy_matched.append(f"{p['name']} → {dg_name}")
                        break
                if p.get("p_make_cut", 0) > 0:
                    continue

            p["p_make_cut"] = 0

        print(f"  Matched make_cut for {mc_matched}/{len(players)} players")
        if fuzzy_matched:
            print(f"  Fuzzy matched: {', '.join(fuzzy_matched)}")
    except Exception as e:
        print(f"  Warning: Could not fetch make_cut data: {e}")
        for p in players:
            p["p_make_cut"] = 0

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

    # ── Load DataGolf decomposition for edge-source diversity ──
    EDGE_CATEGORIES = ["baseline", "driving_dist", "driving_acc"]
    EDGE_COL_MAP = {
        "driving_dist_adj": "driving_dist",
        "driving_acc_adj": "driving_acc",
    }
    edge_sources = None
    try:
        decomp_lookup = {}
        with open(DECOMP_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_name = row["player_name"]
                if ", " in raw_name:
                    parts = raw_name.split(", ", 1)
                    clean_name = f"{parts[1]} {parts[0]}"
                else:
                    clean_name = raw_name
                # Find primary edge: highest positive non-baseline adjustment
                adj_vals = {}
                for col, cat in EDGE_COL_MAP.items():
                    if cat != "baseline":
                        adj_vals[cat] = float(row.get(col, 0))
                best_cat = max(adj_vals, key=lambda k: adj_vals[k])
                primary = best_cat if adj_vals[best_cat] > 0 else "baseline"
                decomp_lookup[clean_name] = primary

        # Match to players using same fuzzy logic
        player_edges = []
        edge_matched = 0
        for p in players:
            edge = decomp_lookup.get(p["name"])
            if edge is None:
                # Try fuzzy: normalized name
                p_norm = _normalize_name(p["name"])
                for dname, dedge in decomp_lookup.items():
                    if _normalize_name(dname) == p_norm:
                        edge = dedge
                        break
            if edge is None:
                # Last name unique match
                p_last = p["name"].split()[-1].lower()
                last_matches = [(dn, de) for dn, de in decomp_lookup.items()
                                if dn.split()[-1].lower() == p_last]
                if len(last_matches) == 1:
                    edge = last_matches[0][1]
            if edge:
                player_edges.append(edge)
                edge_matched += 1
            else:
                player_edges.append("baseline")  # default

        edge_sources = {"primary": player_edges, "categories": EDGE_CATEGORIES}
        # Show distribution
        from collections import Counter
        edge_dist = Counter(player_edges)
        dist_str = " | ".join(f"{k} {v}" for k, v in edge_dist.most_common())
        print(f"  Edge sources: {edge_matched}/{len(players)} matched ({dist_str})")
    except Exception as e:
        print(f"  Warning: Could not load decomposition data: {e}")

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
    # STEP 3: Generate opponent field (largest contest size, reusable)
    # ══════════════════════════════════════════════════════════════════
    max_field = max(c["field"] for c in contests)
    print(f"\n{'='*70}")
    print(f"  STEP 3: Generate {max_field:,} archetype-based opponent lineups")
    print(f"{'='*70}")

    generated_field = generate_field_archetypes(
        players, max_field,
        archetype_weights=DEFAULT_ARCHETYPE_WEIGHTS,
        ownership_tolerance=0.03,
        max_iterations=10,
    )
    opponents = field_to_index_lists(generated_field)
    print(f"  Generated {len(opponents):,} opponent lineups")
    for arch, cnt in sorted(generated_field.archetype_distribution.items(), key=lambda x: -x[1]):
        print(f"    {arch:<12} {cnt:>7,} ({cnt/len(opponents)*100:.0f}%)")

    # Dump opponent field to CSV
    field_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opponent_field.csv")
    with open(field_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Lineup#", "G1", "G2", "G3", "G4", "G5", "G6", "Salary", "Proj"])
        for li, lu in enumerate(opponents, 1):
            names = [players[idx]["name"] for idx in lu]
            sal = sum(players[idx]["salary"] for idx in lu)
            proj = sum(players[idx]["projected_points"] for idx in lu)
            writer.writerow([li] + names + [sal, round(proj, 1)])
    print(f"  Field CSV: {field_csv}")

    # Ownership calibration
    n = len(players)
    counts = [0] * n
    for lu in opponents:
        for idx in lu:
            counts[idx] += 1
    print(f"\n  Ownership calibration (top 10):")
    print(f"  {'Player':<28} {'Target':>8} {'Sim':>8}")
    print(f"  {'-'*28} {'-'*8} {'-'*8}")
    top_own = sorted(range(n), key=lambda i: players[i]["proj_ownership"], reverse=True)[:10]
    for i in top_own:
        sim_own = counts[i] / len(opponents) * 100
        print(f"  {players[i]['name']:<28} {players[i]['proj_ownership']:>7.1f}% {sim_own:>7.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Pre-simulate player scores + Monte Carlo contest sim
    # ══════════════════════════════════════════════════════════════════
    # Sim count driven by field size (10x multiplier from player_sim.py design)
    # Memory-aware: caps total (candidates × sims) to fit in RAM
    #   Position matrix: int32 (4B), optimizer arrays: ~24B per element
    #   Budget: 300GB max working set on 512GB machine
    # ══════════════════════════════════════════════════════════════════
    MAX_MEMORY_GB = 40  # actual measured: OOM at 80GB est. Real overhead ~2x due to
    # positions(int32) + payouts(float64) + cut_survival(float32) + effective(float32)
    # + improvement(float32) + score_buf + opponent arrays + Python heap + numpy temps
    BYTES_PER_ELEMENT = 28  # total per (candidate × sim) element across optimizer arrays

    max_field = max(c["field"] for c in contests)
    field_driven_sims = max_field * PLAYER_SIM_MULTIPLIER  # 71K × 10 = 710K
    n_portfolio_sims = max(args.presim_factor * args.sims, field_driven_sims)

    # Check if we need to cap sims to fit candidates × sims in memory
    n_cands_est = len(candidates) + 500  # +500 for harvest
    max_elements = int(MAX_MEMORY_GB * 1024**3 / BYTES_PER_ELEMENT)
    if n_cands_est * n_portfolio_sims > max_elements:
        n_portfolio_sims = max_elements // n_cands_est
        print(f"  ⚠ Memory cap: {n_cands_est:,} candidates × sims capped to "
              f"{n_portfolio_sims:,} sims ({n_cands_est * n_portfolio_sims * BYTES_PER_ELEMENT / 1024**3:.0f}GB)")

    # Harvest phase: 10% of portfolio sims (enough for opponent ranking)
    n_harvest_sims = max(args.sims, n_portfolio_sims // 10)
    n_presim = n_harvest_sims + n_portfolio_sims

    print(f"\n{'='*70}")
    print(f"  STEP 4: Pre-simulate {n_presim:,} player score scenarios")
    print(f"  Field-driven: {max_field:,} entries × {PLAYER_SIM_MULTIPLIER}x = {field_driven_sims:,} target")
    print(f"  Portfolio sims: {n_portfolio_sims:,} | Harvest sims: {n_harvest_sims:,}")
    print(f"  Memory est: {n_cands_est * n_portfolio_sims * BYTES_PER_ELEMENT / 1024**3:.0f}GB "
          f"({n_cands_est:,} cands × {n_portfolio_sims:,} sims)")
    print(f"  Same-wave corr: {SAME_WAVE_CORRELATION} | Cross-wave corr: {DIFF_WAVE_CORRELATION}")
    print(f"{'='*70}")

    n_players = len(players)
    n_cands = len(candidates)
    n_opps = len(opponents)

    # Build SEPARATE candidate and opponent lineup matrices
    cand_matrix = np.zeros((n_cands, n_players), dtype=np.float32)
    for i, lu in enumerate(candidates):
        for idx in lu:
            cand_matrix[i, idx] = 1.0
    opp_matrix = np.zeros((n_opps, n_players), dtype=np.float32)
    for i, lu in enumerate(opponents):
        for idx in lu:
            opp_matrix[i, idx] = 1.0

    means = np.array([p["projected_points"] for p in players], dtype=np.float64)
    sigmas = np.array([_get_sigma(p) for p in players], dtype=np.float64)

    # Wave-aware covariance matrix
    waves_arr = np.array(waves)
    same_wave = (waves_arr[:, None] == waves_arr[None, :])
    corr_matrix = np.where(same_wave, SAME_WAVE_CORRELATION, DIFF_WAVE_CORRELATION)
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

    # ── Phase 1: Harvest simulation (first n_harvest_sims scenarios) ──
    positions_matrix = np.empty((n_cands, n_harvest_sims), dtype=np.int32)
    opp_score_sum = np.zeros(n_opps, dtype=np.float64)
    print(f"\n  Phase 1: Simulating {n_harvest_sims:,} contests for opponent harvesting...")
    print(f"  {n_cands:,} candidates vs {n_opps:,} opponents (opponent-only ranking)")

    sim_start = time.time()
    batch_size = 500
    cand_matrix_f32 = cand_matrix.astype(np.float32)
    opp_matrix_f32 = opp_matrix.astype(np.float32)

    for batch_start in range(0, n_harvest_sims, batch_size):
        bs = min(batch_size, n_harvest_sims - batch_start)
        scores = presim_scores[batch_start:batch_start + bs]

        cand_scores = scores @ cand_matrix_f32.T
        opp_scores = scores @ opp_matrix_f32.T

        opp_score_sum += opp_scores.sum(axis=0).astype(np.float64)

        # Vectorized ranking: all sims × all candidates in one searchsorted call
        positions = _rank_candidates_vectorized(opp_scores, cand_scores, n_opps, n_opps + 1)
        positions_matrix[:, batch_start:batch_start + bs] = positions.T

    sim_elapsed = time.time() - sim_start
    print(f"  Phase 1 complete in {sim_elapsed:.1f}s")
    print(f"  Position matrix: {positions_matrix.shape} (ranks among {n_opps + 1} participants)")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4b: Harvest top opponent lineups into candidate pool
    # ══════════════════════════════════════════════════════════════════
    HARVEST_COUNT = 500
    opp_mean_scores = opp_score_sum / n_harvest_sims

    print(f"\n{'='*70}")
    print(f"  STEP 4b: Harvest top field lineups into candidate pool")
    print(f"{'='*70}")
    print(f"  Opponent mean score range: {opp_mean_scores.min():.1f} – {opp_mean_scores.max():.1f}")

    # Existing candidate set for dedup
    cand_set = set(tuple(sorted(c)) for c in candidates)

    # Rank opponents by mean simulated score, pick top N unique ones
    top_opp_idx = np.argsort(-opp_mean_scores)
    harvested = []
    harvested_details = []
    for oi in top_opp_idx:
        opp_lu = opponents[oi]
        opp_key = tuple(sorted(opp_lu))
        if opp_key in cand_set:
            continue
        # Salary/projection check (same constraints as our candidates)
        opp_sal = sum(players[idx]["salary"] for idx in opp_lu)
        opp_proj = sum(players[idx]["projected_points"] for idx in opp_lu)
        if (opp_floors.salary_floor and opp_sal < opp_floors.salary_floor) or \
           (opp_floors.proj_floor and opp_proj < opp_floors.proj_floor):
            continue
        harvested.append(list(opp_lu))
        cand_set.add(opp_key)
        harvested_details.append({
            "score": float(opp_mean_scores[oi]),
            "salary": opp_sal,
            "proj": opp_proj,
            "names": [players[idx]["name"] for idx in opp_lu],
        })
        if len(harvested) >= HARVEST_COUNT:
            break

    print(f"  Harvested {len(harvested)} unique field lineups (from {n_opps:,} opponents)")
    if harvested_details:
        print(f"  Top harvested: score={harvested_details[0]['score']:.1f} "
              f"proj={harvested_details[0]['proj']:.1f} "
              f"sal=${harvested_details[0]['salary']:,}")
        print(f"  Bottom harvested: score={harvested_details[-1]['score']:.1f} "
              f"proj={harvested_details[-1]['proj']:.1f} "
              f"sal=${harvested_details[-1]['salary']:,}")
        # Show a few examples
        for h in harvested_details[:5]:
            names = ", ".join(h["names"][:6])
            print(f"    {h['score']:.1f} pts | ${h['salary']:,} | {h['proj']:.1f} proj | {names}")

    # Merge harvested lineups into candidate pool
    n_orig_cands = len(candidates)
    candidates = candidates + harvested
    n_cands = len(candidates)
    print(f"  Candidate pool: {n_orig_cands:,} original + {len(harvested)} harvested = {n_cands:,} total")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4c: Re-simulate with expanded candidate pool (Phase 2 presim)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEP 4c: Portfolio simulation ({n_portfolio_sims:,} sims, {args.presim_factor}x presim)")
    print(f"{'='*70}")

    # Rebuild candidate matrix with expanded pool (opponent matrix unchanged)
    cand_matrix = np.zeros((n_cands, n_players), dtype=np.float32)
    for i, lu in enumerate(candidates):
        for idx in lu:
            cand_matrix[i, idx] = 1.0

    positions_matrix = np.empty((n_cands, n_portfolio_sims), dtype=np.int32)
    print(f"  Simulating {n_portfolio_sims:,} contests: {n_cands:,} candidates vs {n_opps:,} opponents...")
    print(f"  Ranking: opponent-only (candidates ranked independently against field)")

    sim_start = time.time()
    cand_matrix_f32 = cand_matrix.astype(np.float32)
    phase2_offset = n_harvest_sims  # use different scenarios than Phase 1

    for batch_start in range(0, n_portfolio_sims, batch_size):
        bs = min(batch_size, n_portfolio_sims - batch_start)
        scores = presim_scores[phase2_offset + batch_start:phase2_offset + batch_start + bs]

        cand_scores = scores @ cand_matrix_f32.T
        opp_scores = scores @ opp_matrix_f32.T

        # Vectorized ranking: all sims × all candidates in one searchsorted call
        positions = _rank_candidates_vectorized(opp_scores, cand_scores, n_opps, n_opps + 1)
        positions_matrix[:, batch_start:batch_start + bs] = positions.T

        done = batch_start + bs
        if done % 10000 == 0 or done == n_portfolio_sims:
            elapsed = time.time() - sim_start
            print(f"    {done:>7,} / {n_portfolio_sims:,} sims ({elapsed:.1f}s)")

    sim_elapsed = time.time() - sim_start
    print(f"  Phase 2 complete in {sim_elapsed:.1f}s")
    print(f"  Position matrix: {positions_matrix.shape} (ranks among {n_opps + 1} participants)")

    # Free presim scores — no longer needed, reclaim memory before portfolio optimizer
    del presim_scores

    # ══════════════════════════════════════════════════════════════════
    # STEP 4d: Filter candidates by w* for optimizer speed + memory
    #   Always filter to top W*_CAP candidates (speed-driven).
    #   Also respects memory ceiling as a hard cap.
    # ══════════════════════════════════════════════════════════════════
    W_STAR_CAP = 1000  # speed cap: 6-7× headroom over 150-lineup portfolio
    MAX_WORKING_GB = 40
    bytes_per_element = 24
    mem_cap = int(MAX_WORKING_GB * 1024**3 / (n_portfolio_sims * bytes_per_element))
    max_candidates = min(W_STAR_CAP, mem_cap)

    # Use primary contest payout structure for w* ranking
    primary_contest = contests[0]
    primary_field = primary_contest["field"]
    primary_fee = primary_contest["fee"]
    primary_payout_by_pos = build_payout_lookup(
        primary_contest["profile"]["payouts"], primary_field
    )
    sim_field = n_opps + 1
    pos_scale = primary_field / sim_field

    # Compute w* in batches
    w_star_all = np.full(n_cands, -np.inf, dtype=np.float64)
    filter_batch = 1000
    for b_start in range(0, n_cands, filter_batch):
        b_end = min(b_start + filter_batch, n_cands)
        batch_pos = positions_matrix[b_start:b_end]
        scaled = np.rint(batch_pos * pos_scale).astype(np.int32)
        np.clip(scaled, 1, primary_field, out=scaled)
        batch_payouts = primary_payout_by_pos[scaled]
        w_batch, _, _ = compute_w_star(batch_payouts, primary_fee)
        w_star_all[b_start:b_end] = w_batch

    n_positive_w = int((w_star_all > 0).sum())
    actual_cap = min(max_candidates, n_cands)
    top_idx = np.argsort(-w_star_all)[:actual_cap]
    positions_filtered = positions_matrix[top_idx]
    candidates_filtered = [candidates[i] for i in top_idx]

    mem_est = actual_cap * n_portfolio_sims * bytes_per_element / 1024**3
    print(f"\n  w* filter: {n_cands:,} → {actual_cap:,} candidates (cap={max_candidates:,}) | "
          f"+w*={n_positive_w}/{n_cands} | best={w_star_all.max():.6f} | "
          f"est. {mem_est:.1f}GB")
    if actual_cap < n_cands:
        # Show w* cutoff
        cutoff_w = w_star_all[top_idx[-1]] if len(top_idx) > 0 else 0
        print(f"  w* cutoff: {cutoff_w:.6f} (#{actual_cap} of {n_cands:,})")
        del positions_matrix

    # Compute cut survival once for filtered candidates
    cut_survival = compute_cut_survival(candidates_filtered, players, n_portfolio_sims)

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Loop over each contest — payout assignment + portfolio selection
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
    used_candidate_indices = set()  # Track lineups already assigned to prior contests

    for ci, contest_def in enumerate(contests):
        cid = contest_def["cid"]
        cname = contest_def["name"]
        field_size = contest_def["field"]
        max_entries = contest_def["max_entries"]
        entry_fee = contest_def["fee"]
        profile = contest_def["profile"]
        payout_table = profile["payouts"]

        print(f"\n  {'─'*60}")
        print(f"  [{ci+1}/{len(contests)}] {cname[:50]}")
        print(f"  Fee: ${entry_fee} | Field: {field_size:,} | Max entries: {max_entries}")
        print(f"  Payout spots: {profile['payout_spots']:,} | "
              f"1st: ${profile['first_place_prize']:,.0f} | "
              f"Pool: ${profile['prize_pool']:,.0f}")
        print(f"  {'─'*60}")

        # Build payout lookup
        payout_by_pos = build_payout_lookup(payout_table, field_size)

        # Scale positions from opponent-only ranking (n_opps + 1) to contest field size
        sim_field = n_opps + 1
        scale = field_size / sim_field
        scaled = np.rint(positions_filtered * scale).astype(np.int32)
        np.clip(scaled, 1, field_size, out=scaled)
        payouts = payout_by_pos[scaled]  # (n_filtered, n_portfolio_sims)

        roi = (payouts.mean(axis=1) - entry_fee) / entry_fee * 100

        print(f"  Candidate ROI: mean={roi.mean():+.1f}% | best={roi.max():+.1f}% | "
              f"+EV={int((roi > 0).sum())}/{len(roi)}")

        # w* log-utility diagnostics
        w_star, p_cash, kelly_frac = compute_w_star(payouts, entry_fee)
        finite_w = w_star[w_star > -np.inf]
        if len(finite_w) > 0:
            print(f"  w* stats: best={w_star.max():.6f} | median={np.median(finite_w):.6f} | "
                  f"+w*={int((w_star > 0).sum())}/{len(w_star)}")

        # Select portfolio via optimizer (exclude lineups already in other contests)
        portfolio = optimize_portfolio(
            payouts, entry_fee, max_entries, candidates_filtered,
            n_players=n_players,
            method="greedy",
            diversity_weight=0.4,
            waves=waves,
            min_early_pct=0.3,
            min_late_pct=0.2,
            cut_survival=cut_survival,
            edge_sources=edge_sources,
            edge_diversity_weight=5.0,
            excluded_indices=used_candidate_indices,
        )

        selected = portfolio.selected_indices

        # Track these lineups so subsequent contests can't reuse them
        used_candidate_indices.update(selected)

        # Build lineups with player dicts for CSV export
        lineups = []
        for sel_idx in selected:
            player_indices = candidates_filtered[sel_idx]
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
            lineup_stats.append({
                "roi": lu_roi,
                "cash_rate": lu_cash,
                "salary": lu_sal,
                "projection": lu_proj,
                "ceiling": lu_ceil,
                "geomean_own": lu_geomean_own,
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
            writer.writerow(["G"] * ROSTER_SIZE)
            for lineup in lineups:
                writer.writerow([p.get("name_id", p["name"]) for p in lineup])
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

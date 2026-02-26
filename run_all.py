#!/usr/bin/env python3
"""Run the optimizer against every playable PGA TOUR Classic contest on DraftKings.

Shares candidate generation, opponent field, and Monte Carlo scoring across
all contests. Field-size-adjusted payout assignment and cross-contest joint
portfolio selection with global exposure caps.

Usage:
    python run_all.py                    # Run all playable contests
    python run_all.py --sheets           # Run all + export to Google Sheets
    python run_all.py --min-pool 10000   # Only contests with $10K+ prize pool
    python run_all.py --budget 5000      # Cap total investment at $5K
"""
import sys
import os
import csv
import time
import heapq
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ROSTER_SIZE, SALARY_CAP, CVAR_LAMBDA, BASE_CORRELATION
from datagolf_client import get_fantasy_projections, get_predictions, find_current_event
from dk_contests import fetch_contest, fetch_golf_contests
from engine import generate_field, generate_candidates, select_portfolio, _get_sigma
import re
import gspread
from google.oauth2.service_account import Credentials

SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEETS_CREDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials.json")
SHARE_EMAIL = None  # Set to your email to auto-share new spreadsheets


# Skip these contest types / name patterns
SKIP_PATTERNS = [
    "tiers", "single stat", "snake", "satellite", "qualifier",
    "must fill", "heavy hitter", "nosebleed", "thunderdome",
    "dp world", "3-player", "50-50", "double up",
]


def filter_contests(contests, min_pool=5000, max_fee=1000):
    """Filter to playable PGA TOUR Classic GPP/SE contests."""
    filtered = []
    for c in contests:
        name_lower = c["name"].lower()

        # Skip non-Classic formats
        if any(pat in name_lower for pat in SKIP_PATTERNS):
            continue

        # Skip tiny prize pools
        if c["prize_pool"] < min_pool:
            continue

        # Skip ultra-high-roller (>$1K entry)
        if c["entry_fee"] > max_fee:
            continue

        filtered.append(c)

    # Sort by prize pool descending
    filtered.sort(key=lambda c: c["prize_pool"], reverse=True)
    return filtered


def simulate_positions(candidates, opponents, players, n_sims=10000, seed=None,
                       mixture_params=None):
    """Run Monte Carlo and return the position matrix (shared across contests).

    Ranks each candidate independently against opponents only (not against
    other candidates), preventing candidate-pool inflation of ranks.

    Returns:
        positions_matrix: (n_candidates, n_sims) array of finish positions (1-indexed)
        n_field: opponent field size + 1 (the effective contest field per candidate)
    """
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

    # Covariance with baseline correlation
    base_corr = BASE_CORRELATION
    cov = np.outer(sigmas, sigmas) * base_corr
    np.fill_diagonal(cov, sigmas ** 2)

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += np.eye(n_players) * 1.0
        L = np.linalg.cholesky(cov)

    positions_matrix = np.empty((n_cands, n_sims), dtype=np.int32)

    # Mixture distribution support
    use_mix = (mixture_params is not None and mixture_params[5].any())
    if use_mix:
        from engine import transform_mixture_scores
        mix_p_miss, mix_mu_miss, mix_sigma_miss, mix_mu_make, mix_sigma_make, mix_flag = mixture_params
        print(f"  Mixture distribution: {int(mix_flag.sum())}/{n_players} players with bimodal scores")

    print(f"  Simulating {n_sims:,} contests: {n_cands} candidates vs {n_opps:,} opponents...")
    print(f"  Ranking: opponent-only (candidates ranked independently against field)")
    rng = np.random.default_rng(seed)
    batch_size = 500
    cand_matrix_f32 = cand_matrix.astype(np.float32)
    opp_matrix_f32 = opp_matrix.astype(np.float32)

    for batch_start in range(0, n_sims, batch_size):
        bs = min(batch_size, n_sims - batch_start)
        Z = rng.standard_normal((bs, n_players))
        X = Z @ L.T
        if use_mix:
            scores = transform_mixture_scores(
                X, sigmas, mix_p_miss, mix_mu_miss, mix_sigma_miss,
                mix_mu_make, mix_sigma_make, mix_flag)
        else:
            scores = means[None, :] + X
            np.maximum(scores, 0.0, out=scores)

        # Score candidates and opponents separately
        cand_scores = scores.astype(np.float32) @ cand_matrix_f32.T   # (bs, n_cands)
        opp_scores = scores.astype(np.float32) @ opp_matrix_f32.T     # (bs, n_opps)

        # Sort opponent scores ascending for searchsorted
        opp_sorted = np.sort(opp_scores, axis=1)

        # Rank each candidate against opponents only
        for s in range(bs):
            insert_idx = np.searchsorted(opp_sorted[s], cand_scores[s], side='left')
            positions_matrix[:, batch_start + s] = (n_opps - insert_idx + 1).astype(np.int32)

    n_field = n_opps + 1
    return positions_matrix, n_field


def build_payout_lookup(payout_table, field_size):
    """Build a position→payout numpy array for a contest.

    Returns:
        payout_by_pos: array of size (field_size + 1,) where payout_by_pos[pos] = prize
    """
    payout_by_pos = np.zeros(field_size + 1, dtype=np.float64)
    for min_pos, max_pos, prize in sorted(payout_table, key=lambda x: x[0]):
        for pos in range(min_pos, min(max_pos, field_size) + 1):
            payout_by_pos[pos] = prize
    return payout_by_pos


def assign_payouts(positions_matrix, payout_table, n_total, n_sims,
                   contest_field_size=None):
    """Convert position matrix to payout matrix for a specific contest.

    Scales positions from the simulation field (n_total) to the contest's actual
    field size using proportional rank mapping: scaled_pos = round(pos * F / n_total).
    This preserves percentile ranks and is the expected value of the hypergeometric
    rank distribution under random opponent subsampling.
    """
    F = contest_field_size if contest_field_size is not None else n_total
    scale = F / n_total

    payout_by_pos = build_payout_lookup(payout_table, F)

    # Vectorized: scale positions, clip, fancy-index lookup
    scaled = np.rint(positions_matrix * scale).astype(np.int32)
    np.clip(scaled, 1, F, out=scaled)
    payouts = payout_by_pos[scaled]  # (n_cands, n_sims) float64

    return payouts


def _compute_best_for_contest(positions_matrix, n_total, payout_by_pos,
                              field_size, entry_fee, running_max,
                              alive, selected_set):
    """Find the best alive candidate for a contest (lazy payout computation).

    Returns:
        (net_score, candidate_idx, candidate_payouts_row)
    """
    scale = field_size / n_total

    # Scale positions → payouts for all candidates
    scaled = np.rint(positions_matrix * scale).astype(np.int32)
    np.clip(scaled, 1, field_size, out=scaled)
    payouts = payout_by_pos[scaled]  # (n_cands, n_sims)

    # Marginal E[max] improvement
    delta = payouts - running_max  # broadcast (n_cands, n_sims) - (n_sims,)
    np.maximum(delta, 0.0, out=delta)
    marginal = delta.mean(axis=1)  # (n_cands,)
    net = marginal - entry_fee

    # Mask dead + already selected in this contest
    net[~alive] = -np.inf
    for idx in selected_set:
        net[idx] = -np.inf

    best = int(np.argmax(net))
    return float(net[best]), best, payouts[best]


def select_cross_contest_portfolio(positions_matrix, n_total, contest_configs,
                                   candidates, n_players, n_sims,
                                   max_exposure=None, budget=None,
                                   max_contest_appearances=None):
    """Cross-contest joint portfolio selection with global exposure caps.

    Uses interleaved E[max] greedy selection: each round picks the single best
    (candidate, contest) pair across all contests. Score = marginal E[max
    improvement within that contest] - entry_fee. Naturally fills +EV contests
    first and skips -EV ones.

    Per-candidate contest cap forces lineup diversity across contests: once a
    lineup is entered in K different contests, it's removed from consideration
    for remaining contests. This prevents the same lineup from appearing in
    every SE contest and reduces dollar-weighted concentration risk.

    Args:
        positions_matrix: (n_cands, n_sims) shared position matrix
        n_total: simulation field size
        contest_configs: list of dicts with keys:
            cid, payout_by_pos, field_size, entry_fee, max_entries, name
        candidates: list of lineups (list of player indices)
        n_players: total number of players
        n_sims: number of simulations
        max_exposure: global exposure cap (fraction of total entries)
        budget: optional total bankroll constraint
        max_contest_appearances: max contests a single lineup can appear in

    Returns:
        dict of {cid: list of selected candidate indices}
    """
    if max_exposure is None:
        max_exposure = 1.0

    n_cands = len(candidates)
    n_contests = len(contest_configs)

    # Per-candidate contest cap: default to n_contests // 5, min 2
    if max_contest_appearances is None:
        max_contest_appearances = max(2, n_contests // 5)

    # Total entries across all contests for global exposure
    total_max_entries = sum(cc["max_entries"] for cc in contest_configs)
    max_appearances = max(1, int(total_max_entries * max_exposure))

    # Build player → candidate index for fast exposure removal
    player_to_cands = [[] for _ in range(n_players)]
    for ci, lineup in enumerate(candidates):
        for pidx in lineup:
            player_to_cands[pidx].append(ci)

    # Global state
    alive = np.ones(n_cands, dtype=bool)
    appearances = np.zeros(n_players, dtype=np.int32)
    cand_contest_count = np.zeros(n_cands, dtype=np.int32)
    total_cost = 0.0
    total_selected = 0

    # Per-contest state
    states = {}
    for cc in contest_configs:
        cid = cc["cid"]
        states[cid] = {
            "running_max": np.full(n_sims, -np.inf, dtype=np.float64),
            "selected": [],
            "selected_set": set(),
            "max_entries": cc["max_entries"],
            "entry_fee": cc["entry_fee"],
            "payout_by_pos": cc["payout_by_pos"],
            "field_size": cc["field_size"],
            "name": cc["name"],
            "closed": False,
        }

    # Heap: (-net_score, cid, candidate_idx, generation, candidate_payouts)
    # Use generation counter to invalidate stale entries
    generation = {cc["cid"]: 0 for cc in contest_configs}
    heap = []

    print(f"  Computing initial best candidates per contest...")

    # Initial pass: compute best candidate for each contest
    for cid, st in states.items():
        score, cidx, cand_payouts = _compute_best_for_contest(
            positions_matrix, n_total, st["payout_by_pos"],
            st["field_size"], st["entry_fee"], st["running_max"],
            alive, st["selected_set"],
        )
        if score > 0:
            heapq.heappush(heap, (-score, cid, cidx, 0, cand_payouts))

    print(f"  Selecting across {n_contests} contests "
          f"(lineup contest cap: {max_contest_appearances}, "
          f"player exposure cap: {max_appearances})...")

    while heap:
        neg_score, cid, cidx, gen, cand_payouts = heapq.heappop(heap)
        score = -neg_score
        st = states[cid]

        # Skip if contest closed or stale generation
        if st["closed"] or gen < generation[cid]:
            continue

        # Skip if candidate no longer alive (killed by global exposure)
        if not alive[cidx]:
            generation[cid] += 1
            new_score, new_cidx, new_payouts = _compute_best_for_contest(
                positions_matrix, n_total, st["payout_by_pos"],
                st["field_size"], st["entry_fee"], st["running_max"],
                alive, st["selected_set"],
            )
            if new_score > 0 and len(st["selected"]) < st["max_entries"]:
                heapq.heappush(heap, (-new_score, cid, new_cidx,
                                      generation[cid], new_payouts))
            continue

        # Net score <= 0 means no more +EV entries
        if score <= 0:
            break

        # Contest full?
        if len(st["selected"]) >= st["max_entries"]:
            st["closed"] = True
            continue

        # Budget check
        if budget is not None and total_cost + st["entry_fee"] > budget:
            st["closed"] = True
            continue

        # Already selected in this contest? (shouldn't happen with heap, but safety)
        if cidx in st["selected_set"]:
            generation[cid] += 1
            new_score, new_cidx, new_payouts = _compute_best_for_contest(
                positions_matrix, n_total, st["payout_by_pos"],
                st["field_size"], st["entry_fee"], st["running_max"],
                alive, st["selected_set"],
            )
            if new_score > 0 and len(st["selected"]) < st["max_entries"]:
                heapq.heappush(heap, (-new_score, cid, new_cidx,
                                      generation[cid], new_payouts))
            continue

        # ── Accept selection ──
        st["selected"].append(cidx)
        st["selected_set"].add(cidx)
        np.maximum(st["running_max"], cand_payouts, out=st["running_max"])
        total_cost += st["entry_fee"]
        total_selected += 1

        # Per-candidate contest cap: remove lineup from future contests
        cand_contest_count[cidx] += 1
        if cand_contest_count[cidx] >= max_contest_appearances:
            alive[cidx] = False

        # Update global player exposure
        for pidx in candidates[cidx]:
            appearances[pidx] += 1
            if appearances[pidx] >= max_appearances:
                for ci in player_to_cands[pidx]:
                    alive[ci] = False

        # Progress logging
        if total_selected % 50 == 0 or total_selected <= 5:
            print(f"    [{total_selected}] ${st['entry_fee']} {st['name'][:35]} | "
                  f"net=${score:.2f} | alive={int(alive.sum()):,} | "
                  f"cost=${total_cost:,.0f}")

        # Recompute this contest's next-best and re-push
        if len(st["selected"]) < st["max_entries"]:
            generation[cid] += 1
            new_score, new_cidx, new_payouts = _compute_best_for_contest(
                positions_matrix, n_total, st["payout_by_pos"],
                st["field_size"], st["entry_fee"], st["running_max"],
                alive, st["selected_set"],
            )
            if new_score > 0:
                heapq.heappush(heap, (-new_score, cid, new_cidx,
                                      generation[cid], new_payouts))

    # Summary
    print(f"\n  Selection complete: {total_selected} entries across "
          f"{sum(1 for s in states.values() if s['selected'])} contests | "
          f"Cost: ${total_cost:,.0f}")

    # Per-contest summary
    for cid, st in states.items():
        if st["selected"]:
            print(f"    {st['name'][:50]:<50} ${st['entry_fee']:>5} × "
                  f"{len(st['selected']):>4} = ${st['entry_fee'] * len(st['selected']):>8,.0f}")

    return {cid: st["selected"] for cid, st in states.items()}


EXISTING_SHEET_ID = "12YBOTHuRWX6EK6mGz6Uu-VfstwtRWuHlGwl3Zg5UHbc"


def export_all_to_sheets(event_name, results, players, global_exposure, total_lineups):
    """Export all contest results to the shared Google Sheet.

    Clears all existing tabs and writes:
      - Summary tab (cross-contest portfolio overview)
      - Projections tab (shared player projections)
      - One Lineups tab per contest (DK upload format)

    Rate-limits API calls to stay under Google's 60 writes/minute quota.
    Returns the spreadsheet URL.
    """
    creds = Credentials.from_service_account_file(SHEETS_CREDS_PATH, scopes=SHEETS_SCOPES)
    client = gspread.authorize(creds)

    HEADER_FMT = {
        "backgroundColor": {"red": 0.15, "green": 0.15, "blue": 0.15},
        "textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1}, "bold": True},
        "horizontalAlignment": "CENTER",
    }

    # Open existing spreadsheet and clear all tabs
    spreadsheet = client.open_by_key(EXISTING_SHEET_ID)
    spreadsheet.update_title(f"DFS Golf — {event_name} — All Contests")

    # Delete all worksheets except the first, then repurpose the first
    worksheets = spreadsheet.worksheets()
    if len(worksheets) > 1:
        for ws in worksheets[1:]:
            try:
                spreadsheet.del_worksheet(ws)
            except Exception:
                pass
            time.sleep(1)
    worksheets[0].clear()
    time.sleep(2)
    print(f"\n  Cleared existing spreadsheet")

    time.sleep(2)

    # ── Tab 1: Summary ──
    summary_ws = spreadsheet.sheet1
    summary_ws.update_title("Summary")

    summary_rows = [
        ["CROSS-CONTEST PORTFOLIO SUMMARY", "", "", "", "", "", ""],
        [f"Event: {event_name}", "", "", "", "", "", ""],
        ["", "", "", "", "", "", ""],
        ["Contest", "Fee", "Lineups", "Cost", "ROI %", "Cash %", "Field"],
    ]
    for r in results:
        summary_rows.append([
            r["name"],
            r["entry_fee"],
            r["n_lineups"],
            r["total_cost"],
            round(r["expected_roi"], 1),
            round(r["cash_rate"], 1),
            r["profile"].get("max_entries", ""),
        ])

    total_inv = sum(r["total_cost"] for r in results)
    total_exp = sum(r["total_cost"] * (1 + r["expected_roi"] / 100) for r in results)
    total_profit = total_exp - total_inv
    overall_roi = total_profit / total_inv * 100 if total_inv > 0 else 0

    summary_rows.append(["", "", "", "", "", "", ""])
    summary_rows.append(["TOTALS", "", sum(r["n_lineups"] for r in results),
                         total_inv, round(overall_roi, 1), "", ""])
    summary_rows.append([f"Expected Profit: ${total_profit:+,.0f}", "", "", "", "", "", ""])
    summary_rows.append(["", "", "", "", "", "", ""])
    summary_rows.append(["GLOBAL PLAYER EXPOSURE", "", "", "", "", "", ""])
    summary_rows.append(["Player", "Appearances", "Exposure %", "", "", "", ""])
    top_global = sorted(global_exposure.items(), key=lambda x: -x[1])[:20]
    for name, count in top_global:
        summary_rows.append([name, count, round(count / total_lineups * 100, 1),
                            "", "", "", ""])

    summary_ws.update(summary_rows, value_input_option="RAW")
    summary_ws.format("A1:G1", HEADER_FMT)
    summary_ws.format("A4:G4", HEADER_FMT)
    time.sleep(2)

    # ── Tab 2: Projections ──
    proj_ws = spreadsheet.add_worksheet(title="Projections",
                                        rows=len(players) + 1, cols=7)
    proj_header = ["Player", "Salary", "Proj Pts", "Value", "Own%", "Std Dev", "Name ID"]
    proj_rows = [proj_header]
    for p in players:
        proj_rows.append([
            p["name"], p["salary"], p["projected_points"],
            p.get("value", 0), p.get("proj_ownership", 0),
            p.get("std_dev", 0), p.get("name_id", p["name"]),
        ])
    proj_ws.update(proj_rows, value_input_option="RAW")
    proj_ws.format("A1:G1", HEADER_FMT)
    time.sleep(2)

    # ── Per-contest lineup tabs ──
    for i, r in enumerate(results):
        lineups = r["lineups"]
        if not lineups:
            continue

        # Derive short tab name from contest
        m = re.search(r'\$(\d+(?:\.\d+)?[MKmk]?)', r["name"])
        tab_prefix = m.group(1).upper() if m else f"C{i+1}"
        max_e = r["profile"].get("max_entries_per_user", 0)
        if max_e == 1:
            tab_prefix += " SE"
        tab_name = f"{tab_prefix} | Lineups"

        # Ensure tab name is unique (Google Sheets requires unique names)
        existing = [ws.title for ws in spreadsheet.worksheets()]
        if tab_name in existing:
            tab_name = f"{tab_prefix} ${r['entry_fee']} | Lineups"

        ws = spreadsheet.add_worksheet(title=tab_name,
                                       rows=len(lineups) + 2, cols=6)
        lineup_rows = [["G", "G", "G", "G", "G", "G"]]
        for lineup in lineups:
            lineup_rows.append([p.get("name_id", p["name"]) for p in lineup])
        ws.update(lineup_rows, value_input_option="RAW")
        ws.format("A1:F1", HEADER_FMT)

        print(f"    Tab: {tab_name} ({len(lineups)} lineups)")

        # Rate limit: ~3 API calls per tab, stay under 60/min
        time.sleep(3)

    return spreadsheet.url


def main():
    parser = argparse.ArgumentParser(description="DFS Golf — Run All Contests")
    parser.add_argument("--candidates", type=int, default=5000,
                        help="Candidate pool size (default: 5000)")
    parser.add_argument("--sims", type=int, default=10000,
                        help="Monte Carlo simulations (default: 10000)")
    parser.add_argument("--field-size", type=int, default=None,
                        help="Override opponent field size")
    parser.add_argument("--min-pool", type=int, default=5000,
                        help="Minimum prize pool (default: $5000)")
    parser.add_argument("--max-fee", type=int, default=1000,
                        help="Maximum entry fee (default: $1000)")
    parser.add_argument("--budget", type=float, default=None,
                        help="Total bankroll constraint across all contests")
    parser.add_argument("--max-contest-appearances", type=int, default=None,
                        help="Max contests a single lineup can appear in (default: n_contests//5)")
    parser.add_argument("--sheets", action="store_true",
                        help="Export all results to Google Sheets")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("  DFS GOLF — CROSS-CONTEST PORTFOLIO OPTIMIZER")
    print("=" * 70)

    # ── Fetch lobby ──
    print(f"\n  Fetching DraftKings Golf lobby...")
    all_contests = fetch_golf_contests()
    playable = filter_contests(all_contests, min_pool=args.min_pool, max_fee=args.max_fee)
    print(f"  Total contests: {len(all_contests)}")
    print(f"  Playable Classic contests: {len(playable)}")

    if not playable:
        print("  No playable contests found.")
        return

    print(f"\n  {'Contest':<50} {'Fee':>6} {'Pool':>10} {'Field':>8} {'MaxE':>5}")
    print(f"  {'-'*50} {'-'*6} {'-'*10} {'-'*8} {'-'*5}")
    for c in playable:
        print(f"  {c['name'][:50]:<50} ${c['entry_fee']:>5} ${c['prize_pool']:>9,} "
              f"{c['max_entries']:>8,} {c['max_entries_per_user']:>5}")

    # ── Fetch DataGolf projections ──
    print(f"\n  Fetching DataGolf projections...")
    fantasy_data = get_fantasy_projections()
    dg_projections = fantasy_data.get("projections", []) if isinstance(fantasy_data, dict) else fantasy_data
    event_name = fantasy_data.get("event_name", "Unknown") if isinstance(fantasy_data, dict) else "Unknown"
    print(f"  Event: {event_name}")

    # Build player list
    players = []
    for dg in dg_projections:
        salary = dg.get("salary", 0)
        proj_pts = dg.get("proj_points_total", 0)
        if not salary or salary <= 0 or proj_pts <= 0:
            continue
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

    # Synthesize ownership if needed
    has_ownership = any(p["proj_ownership"] > 0 for p in players)
    if not has_ownership:
        print(f"  Ownership not live — synthesizing from projections")
        projs = np.array([p["projected_points"] for p in players])
        exp_projs = np.exp((projs - projs.mean()) / max(projs.std(), 1))
        synth_own = exp_projs / exp_projs.sum() * 100 * ROSTER_SIZE
        for i, p in enumerate(players):
            p["proj_ownership"] = round(float(synth_own[i]), 2)

    print(f"  Players: {len(players)}")

    # ── Fetch P(make_cut) for mixture distribution ──
    print(f"\n  Fetching DataGolf make_cut probabilities...")
    mixture_params_data = None
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

        mc_matched = 0
        for p in players:
            mc = mc_lookup.get(p["name"], 0)
            if mc > 0:
                p["p_make_cut"] = mc
                mc_matched += 1
            else:
                p["p_make_cut"] = 0
        print(f"  Matched make_cut for {mc_matched}/{len(players)} players")

        from engine import compute_mixture_params
        mixture_params_data = compute_mixture_params(players)
        n_mixture = int(mixture_params_data[5].sum())
        print(f"  Mixture distribution: {n_mixture}/{len(players)} players with bimodal scores")
    except Exception as e:
        print(f"  Warning: Could not fetch make_cut data: {e}")

    # ── SHARED: Generate field + candidates ──
    max_field = args.field_size or max(c["max_entries"] for c in playable)
    max_field = min(max_field, 75000)  # cap for performance

    print(f"\n{'='*70}")
    print(f"  SHARED: Generating {max_field:,} opponents + {args.candidates:,} candidates")
    print(f"{'='*70}")

    opponents = generate_field(players, max_field)
    print(f"  Opponents: {len(opponents):,}")

    candidates = generate_candidates(players, pool_size=args.candidates)
    print(f"  Unique candidates: {len(candidates):,}")

    # ── SHARED: Monte Carlo simulation (positions only) ──
    print(f"\n{'='*70}")
    print(f"  SHARED: Monte Carlo simulation ({args.sims:,} sims)")
    print(f"{'='*70}")

    positions_matrix, n_field = simulate_positions(
        candidates, opponents, players, n_sims=args.sims,
        mixture_params=mixture_params_data,
    )
    n_total = n_field  # For downstream scaling (opponent-only ranking base)
    print(f"  Position matrix: {positions_matrix.shape}")
    print(f"  Ranking base: {n_field:,} (opponent-only, {n_field - 1:,} opponents + 1 candidate)")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Fetch all contest details and build payout lookups
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Fetching contest details + field-size-adjusted payouts")
    print(f"{'='*70}")

    contest_configs = []
    contest_profiles = {}

    for ci, contest_summary in enumerate(playable):
        cid = contest_summary["contest_id"]
        cname = contest_summary["name"]
        contest_field = contest_summary["max_entries"]
        scale = contest_field / n_total

        try:
            profile = fetch_contest(cid)
        except Exception as e:
            print(f"  [{ci+1}/{len(playable)}] ERROR fetching {cname[:40]}: {e}")
            continue

        payout_by_pos = build_payout_lookup(profile["payouts"], contest_field)

        # Quick diagnostic: sample ROI with field-size scaling
        sample_scaled = np.rint(positions_matrix[:, 0] * scale).astype(np.int32)
        np.clip(sample_scaled, 1, contest_field, out=sample_scaled)
        sample_payouts = payout_by_pos[sample_scaled]
        cash_pct = (sample_payouts > 0).sum() / len(sample_payouts) * 100

        contest_configs.append({
            "cid": cid,
            "name": cname,
            "entry_fee": profile["entry_fee"],
            "max_entries": contest_summary["max_entries_per_user"],
            "field_size": contest_field,
            "payout_by_pos": payout_by_pos,
        })
        contest_profiles[cid] = profile

        print(f"  [{ci+1}/{len(playable)}] {cname[:45]:<45} "
              f"field={contest_field:>6,} scale={scale:.4f} ~{cash_pct:.0f}% cash")

    if not contest_configs:
        print("  No contest details fetched successfully.")
        return

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Cross-contest joint portfolio selection
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Cross-contest joint portfolio selection")
    total_possible = sum(cc["max_entries"] for cc in contest_configs)
    print(f"  {len(contest_configs)} contests | {total_possible} max entries | "
          f"{len(candidates):,} candidates")
    if args.budget:
        print(f"  Budget: ${args.budget:,.0f}")
    print(f"{'='*70}")

    selections = select_cross_contest_portfolio(
        positions_matrix=positions_matrix,
        n_total=n_total,
        contest_configs=contest_configs,
        candidates=candidates,
        n_players=len(players),
        n_sims=args.sims,
        budget=args.budget,
        max_contest_appearances=args.max_contest_appearances,
    )

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Build outputs per contest
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Building outputs")
    print(f"{'='*70}")

    results = []
    for cc in contest_configs:
        cid = cc["cid"]
        selected = selections.get(cid, [])
        if not selected:
            continue

        profile = contest_profiles[cid]
        entry_fee = cc["entry_fee"]
        field_size = cc["field_size"]

        # Compute payouts for selected lineups
        sel_positions = positions_matrix[selected]
        scale = field_size / n_total
        scaled = np.rint(sel_positions * scale).astype(np.int32)
        np.clip(scaled, 1, field_size, out=scaled)
        sel_payouts = cc["payout_by_pos"][scaled]

        # Build lineups
        lineups = []
        for sel_idx in selected:
            player_indices = candidates[sel_idx]
            lineup = [players[i].copy() for i in player_indices]
            lineup.sort(key=lambda p: p["salary"], reverse=True)
            lineups.append(lineup)

        # Portfolio metrics
        total_cost = entry_fee * len(selected)
        port_returns = sel_payouts.sum(axis=0) - total_cost
        port_roi = port_returns / total_cost * 100 if total_cost > 0 else np.zeros_like(port_returns)

        # Player exposure
        exposure = {}
        for lu in lineups:
            for p in lu:
                exposure[p["name"]] = exposure.get(p["name"], 0) + 1

        top_exp = sorted(exposure.items(), key=lambda x: -x[1])[:5]
        top_exp_str = ", ".join(f"{n} {c/len(lineups)*100:.0f}%" for n, c in top_exp)

        contest_result = {
            "contest_id": cid,
            "name": cc["name"],
            "entry_fee": entry_fee,
            "n_lineups": len(lineups),
            "total_cost": total_cost,
            "expected_roi": float(port_roi.mean()),
            "cash_rate": float((port_returns > 0).mean() * 100),
            "p5": float(np.percentile(port_roi, 5)),
            "p95": float(np.percentile(port_roi, 95)),
            "lineups": lineups,
            "profile": profile,
        }
        results.append(contest_result)

        print(f"\n  {cc['name'][:50]}")
        print(f"  Fee: ${entry_fee} | Lineups: {len(lineups)} | "
              f"Field: {field_size:,} | Cost: ${total_cost:,.0f}")
        print(f"  ROI: {contest_result['expected_roi']:+.1f}% | "
              f"Cash: {contest_result['cash_rate']:.1f}%")
        print(f"  Top exposure: {top_exp_str}")

        # Save CSV
        event_slug = event_name.replace(" ", "_").replace("'", "")
        fee_label = f"${entry_fee}"
        csv_filename = f"lineups_{event_slug}_{fee_label}_{len(lineups)}.csv"
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_filename)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["G"] * ROSTER_SIZE)
            for lineup in lineups:
                writer.writerow([p.get("name_id", p["name"]) for p in lineup])
        print(f"  CSV: {csv_filename}")

    # ── Summary ──
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  CROSS-CONTEST PORTFOLIO SUMMARY")
    print(f"{'='*70}")

    if not results:
        print(f"\n  No +EV entries found across any contest.")
        print(f"  Done in {elapsed:.0f}s")
        print(f"{'='*70}\n")
        return

    print(f"\n  {'Contest':<45} {'Fee':>6} {'LU':>4} {'Cost':>8} {'ROI':>8} {'Cash':>6}")
    print(f"  {'-'*45} {'-'*6} {'-'*4} {'-'*8} {'-'*8} {'-'*6}")

    total_investment = 0
    total_expected = 0
    for r in results:
        print(f"  {r['name'][:45]:<45} ${r['entry_fee']:>5} {r['n_lineups']:>4} "
              f"${r['total_cost']:>7,} {r['expected_roi']:>+7.1f}% {r['cash_rate']:>5.1f}%")
        total_investment += r["total_cost"]
        total_expected += r["total_cost"] * (1 + r["expected_roi"] / 100)

    skipped = len(contest_configs) - len(results)
    total_profit = total_expected - total_investment
    overall_roi = total_profit / total_investment * 100 if total_investment > 0 else 0

    # Global player exposure
    global_exposure = {}
    total_lineups = sum(r["n_lineups"] for r in results)
    for r in results:
        for lu in r["lineups"]:
            for p in lu:
                global_exposure[p["name"]] = global_exposure.get(p["name"], 0) + 1
    top_global = sorted(global_exposure.items(), key=lambda x: -x[1])[:8]

    print(f"\n  Total Investment: ${total_investment:,.0f}")
    print(f"  Total Expected:  ${total_expected:,.0f}")
    print(f"  Expected Profit: ${total_profit:+,.0f}")
    print(f"  Overall ROI:     {overall_roi:+.1f}%")
    print(f"  Contests Entered: {len(results)} | Skipped (-EV): {skipped}")
    print(f"  Total Lineups:   {total_lineups}")

    print(f"\n  Global Player Exposure ({total_lineups} total lineups):")
    for name, count in top_global:
        print(f"    {name:<25} {count:>4}/{total_lineups} = {count/total_lineups*100:.1f}%")

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
        except Exception as e:
            print(f"\n  Sheets ERROR: {e}")

    print(f"\n  Done in {elapsed:.0f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

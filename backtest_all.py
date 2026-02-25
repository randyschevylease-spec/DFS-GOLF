#!/usr/bin/env python3
"""DFS Golf — Full Pipeline Backtester.

Replays historical PGA events through the complete optimizer pipeline and scores
against actual DraftKings results. Uses the DataGolf historical DFS API for real
salaries, ownership, and fantasy points.

Usage:
    python backtest_all.py                              # Backtest 20 recent events
    python backtest_all.py --year 2025 --events 40      # All 2025 events
    python backtest_all.py --events 10 --sweep           # Parameter sweep
    python backtest_all.py --sheets --csv results.csv    # Full run with exports
"""
import sys
import os
import csv
import json
import time
import argparse
import numpy as np
from itertools import product
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ROSTER_SIZE, SALARY_CAP, SALARY_FLOOR, MAX_EXPOSURE
from engine import generate_field, generate_candidates, select_portfolio, _get_sigma, empirical_sigma_from_projection
from run_all import simulate_positions, build_payout_lookup, assign_payouts
from backtest import _cached_get, CACHE_DIR
from datagolf_client import _get


# ── Empirical Std Dev by Make-Cut Tier (from calibration_data.json, n=4746) ──
# Computed from actual DK score variance per make_cut probability bucket.

EMPIRICAL_STDDEV = {
    0: 15.3, 1: 17.6, 2: 23.6, 3: 24.9, 4: 27.7,
    5: 28.7, 6: 30.7, 7: 29.2, 8: 29.8, 9: 20.9,
}


def empirical_std_dev(make_cut_prob):
    """Data-driven std_dev from calibration data, grouped by make_cut decile."""
    tier = min(int(make_cut_prob * 10), 9)
    return EMPIRICAL_STDDEV[tier]


# ── Skill Ratings Cache ──────────────────────────────────────────────────────

_skill_ratings_cache = None

def get_skill_ratings_lookup():
    """Fetch and cache current DG skill ratings (sg_total, sg_app, etc.)."""
    global _skill_ratings_cache
    if _skill_ratings_cache is not None:
        return _skill_ratings_cache

    cache_path = os.path.join(CACHE_DIR, "_skill_ratings.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
    else:
        try:
            data = _get("/preds/skill-ratings", {"display": "value"})
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"  Warning: Could not fetch skill ratings: {e}")
            _skill_ratings_cache = {}
            return _skill_ratings_cache

    lookup = {}
    for p in data.get("players", []):
        lookup[p["dg_id"]] = {
            "sg_total": p.get("sg_total", 0) or 0,
            "sg_app": p.get("sg_app", 0) or 0,
            "sg_ott": p.get("sg_ott", 0) or 0,
            "sg_arg": p.get("sg_arg", 0) or 0,
            "sg_putt": p.get("sg_putt", 0) or 0,
        }
    _skill_ratings_cache = lookup
    return lookup


# ── Average DK points by finish range (empirical calibration) ────────────────

FINISH_RANGE_PTS = {
    "win": 135, "top_3": 120, "top_5": 110, "top_10": 95,
    "top_20": 80, "top_30": 68, "made_cut": 55, "missed_cut": 28,
}

# Events to skip (team events, non-standard)
SKIP_KEYWORDS = [
    "zurich classic", "team", "olympic", "tour championship",
    "q-school", "barracuda", "presidents cup", "ryder cup",
]


# ── FantasyCruncher Projection Loader ────────────────────────────────────────

FC_DIR = "/Users/rhbot/Downloads/untitled folder"
_fc_event_map = None


def _load_fc_event_map():
    """Load FC file → DG event mapping."""
    global _fc_event_map
    if _fc_event_map is not None:
        return _fc_event_map
    map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history", "fc_event_map.json")
    if os.path.exists(map_path):
        with open(map_path) as f:
            _fc_event_map = json.load(f)
    else:
        _fc_event_map = {}
    return _fc_event_map


def load_fc_projections(event_id, year):
    """Load FC pre-tournament projections for a historical event.

    Returns dict of {player_name: {fc_proj, salary, own_proj, ownership}} or None.
    """
    mapping = _load_fc_event_map()

    # Find FC file for this event
    for fname, info in mapping.items():
        if info["event_id"] == event_id and info["year"] == year:
            path = os.path.join(FC_DIR, fname)
            if not os.path.exists(path):
                continue

            result = {}
            with open(path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Player", "").strip()
                    if not name:
                        continue
                    result[name] = {
                        "fc_proj": float(row.get("FC Proj", 0) or 0),
                        "salary": int(row.get("Salary", 0) or 0),
                        "own_proj": float(row.get("Own Proj", 0) or 0),
                        "ownership": float(row.get("Ownership", 0) or 0),
                        "actual_pts": float(row.get("FPs", 0) or 0),
                    }
            return result if result else None

    return None


def build_players_from_fc(event_id, year):
    """Build player list entirely from FC data — no derived projections.

    Uses FC Proj as projected_points, FC ownership for opponent field,
    FC FPs as actual_dk_pts. No DataGolf projection model involved.

    Returns:
        players: list of player dicts (ready for engine.py), or None
    """
    fc_lookup = load_fc_projections(event_id, year)
    if not fc_lookup:
        return None

    players = []
    for name, fc in fc_lookup.items():
        salary = fc["salary"]
        proj = fc["fc_proj"]
        actual = fc["actual_pts"]
        if salary <= 0 or proj <= 0:
            continue

        # Ownership: prefer pre-tournament Own Proj, fall back to actual Ownership
        own = fc["own_proj"] if fc["own_proj"] > 0 else fc["ownership"]

        players.append({
            "name": name,
            "salary": salary,
            "projected_points": round(proj, 2),
            "std_dev": round(empirical_sigma_from_projection(proj), 1),
            "proj_ownership": round(own, 2),
            "actual_dk_pts": actual,
            "value": round(proj / (salary / 1000), 2) if salary > 0 else 0,
        })

    if not players:
        return None

    players.sort(key=lambda p: p["projected_points"], reverse=True)

    # Scale ownership to sum to ROSTER_SIZE * 100%
    total_own = sum(p["proj_ownership"] for p in players)
    if total_own > 0:
        target_total = ROSTER_SIZE * 100
        scale = target_total / total_own
        for p in players:
            p["proj_ownership"] = round(max(p["proj_ownership"] * scale, 0.1), 2)

    # Apply ownership leverage for candidate generation
    _apply_adaptive_leverage(players)

    fc_count = sum(1 for p in players if p["projected_points"] > 0)
    print(f"    FC data: {fc_count} players loaded directly from FantasyCruncher")

    return players


def load_fc_payout_table(event_id, year):
    """Load real payout table from the FC All Entrants file for this event.

    Finds the All Entrants CSV matching the same event, parses it to extract
    entry fee, field size, and per-position payout structure.

    Returns (payout_table, entry_fee, prize_pool, field_size) or None.
    """
    mapping = _load_fc_event_map()

    # Find the Player Exposures file for this event
    pe_fname = None
    for fname, info in mapping.items():
        if info["event_id"] == event_id and info["year"] == year:
            pe_fname = fname
            break
    if not pe_fname:
        return None

    # Find matching All Entrants file (same contest name prefix)
    # PE: "PGA TOUR $300K Drive the Green [$50K to 1st] Player Exposures.csv"
    # AE: "PGA TOUR $300K Drive the Green [$50K to 1st] All Entrants.csv"
    ae_fname = pe_fname.replace("Player Exposures", "All Entrants")
    ae_path = os.path.join(FC_DIR, ae_fname)
    if not os.path.exists(ae_path):
        # Try variants (copy, copy 2, etc.)
        base = pe_fname.split("Player Exposures")[0].strip()
        for f in os.listdir(FC_DIR):
            if f.startswith(base) and "All Entrants" in f and f.endswith(".csv"):
                ae_path = os.path.join(FC_DIR, f)
                break
        else:
            return None

    # Parse All Entrants to extract payout structure
    try:
        with open(ae_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = list(reader)

        if not rows:
            return None

        total_lineups = sum(int(r[2]) for r in rows)

        # Entry fee from single-lineup losers (Lineups=1, #Cash=0)
        entry_fees = []
        for r in rows:
            if r[2] == "1" and r[3] == "0":
                entry_fees.append(abs(float(r[4])))
        if not entry_fees:
            return None
        import statistics
        entry_fee = statistics.median(entry_fees)

        # Find stated pool from filename
        import re
        m = re.search(r'\$([0-9.]+)([MK])', os.path.basename(ae_path))
        if m:
            val = float(m.group(1))
            prize_pool = val * (1_000_000 if m.group(2).upper() == "M" else 1_000)
        else:
            prize_pool = entry_fee * total_lineups * 0.85

        # Last cashing place
        last_cash_place = 0
        for r in rows:
            if int(r[3]) > 0:
                last_cash_place = max(last_cash_place, int(r[6]))

        # Single-lineup cashers → per-position payouts
        single_cashers = []
        for r in rows:
            if r[2] == "1" and r[3] == "1":
                gross = float(r[4]) + entry_fee
                place = int(r[6])
                single_cashers.append((place, gross))

        if not single_cashers:
            return None

        # Build payout table: group by position ranges, take median payout
        from collections import defaultdict
        TIERS = [
            (1, 1), (2, 2), (3, 3), (4, 5), (6, 10),
            (11, 25), (26, 50), (51, 100), (101, 250), (251, 500),
            (501, 1000), (1001, 2500), (2501, 5000), (5001, last_cash_place),
        ]

        tier_payouts = defaultdict(list)
        for place, gross in single_cashers:
            for lo, hi in TIERS:
                if lo <= place <= hi:
                    tier_payouts[(lo, hi)].append(gross)
                    break

        # Stated 1st prize from filename
        m2 = re.search(r'\[.*?\$([0-9.]+)([MK])\s+to\s+1st', os.path.basename(ae_path), re.IGNORECASE)
        if m2:
            first_val = float(m2.group(1))
            stated_first = first_val * (1_000_000 if m2.group(2).upper() == "M" else 1_000)
        else:
            stated_first = None

        payout_table = []
        for lo, hi in TIERS:
            if hi < lo or lo > last_cash_place:
                continue
            actual_hi = min(hi, last_cash_place)
            if (lo, hi) in tier_payouts:
                median_payout = statistics.median(tier_payouts[(lo, hi)])
                payout_table.append((lo, actual_hi, round(median_payout)))
            elif lo == 1 and stated_first:
                payout_table.append((1, 1, round(stated_first)))

        if not payout_table:
            return None

        return payout_table, entry_fee, prize_pool, total_lineups

    except Exception as e:
        print(f"    Warning: Could not parse All Entrants: {e}")
        return None


# ── Data Fetching (cached) ───────────────────────────────────────────────────

def get_backtest_events_dfs(year=None, max_events=None):
    """Get PGA events with DK salary + ownership data for backtesting."""
    events = _cached_get("/historical-dfs-data/event-list", {})

    filtered = []
    for e in events:
        # Must have DK salary and ownership data
        if e.get("dk_salaries") != "yes" or e.get("dk_ownerships") != "yes":
            continue
        if e.get("tour") != "pga":
            continue

        name_lower = e.get("event_name", "").lower()
        if any(kw in name_lower for kw in SKIP_KEYWORDS):
            continue

        cal_year = e.get("calendar_year", 0)
        if year and cal_year != year:
            continue

        filtered.append(e)

    # Sort by date descending (most recent first)
    filtered.sort(key=lambda e: e.get("date", ""), reverse=True)

    if max_events:
        filtered = filtered[:max_events]

    return filtered


def fetch_historical_dfs_points(event_id, year):
    """Fetch actual DK salaries, ownership, and fantasy points for a historical event."""
    data = _cached_get("/historical-dfs-data/points", {
        "tour": "pga",
        "site": "draftkings",
        "event_id": str(event_id),
        "year": str(year),
        "market": "classic",
        "file_format": "json",
    })
    return data


def fetch_predictions_archive(event_id, year):
    """Fetch DG pre-tournament predictions for a historical event."""
    return _cached_get("/preds/pre-tournament-archive", {
        "tour": "pga",
        "event_id": str(event_id),
        "year": str(year),
        "odds_format": "percent",
    })


# ── Projection Model ─────────────────────────────────────────────────────────
# Empirical average DK points per finish-range bucket (from backtest_event.py).
# Simple probability-weighted model: better out-of-sample than complex fits.

FINISH_RANGE_PTS = {
    "win": 135.0, "top_3": 120.0, "top_5": 110.0, "top_10": 95.0,
    "top_20": 80.0, "top_30": 68.0, "made_cut": 55.0, "missed_cut": 28.0,
}


def project_dk_points_from_probs(pred, salary=0, sg_total=0):
    """Derive projected DK points from DG pre-tournament probability buckets.

    Uses probability-weighted finish-range averages plus skill rating adjustment.
    """
    win = pred.get("win", 0) or 0
    top_3 = pred.get("top_3", 0) or 0
    top_5 = pred.get("top_5", 0) or 0
    top_10 = pred.get("top_10", 0) or 0
    top_20 = pred.get("top_20", 0) or 0
    top_30 = pred.get("top_30", 0) or 0
    make_cut = pred.get("make_cut", 0) or 0

    # Normalize to 0-1 if given as percentages
    if make_cut > 1:
        win /= 100; top_3 /= 100; top_5 /= 100
        top_10 /= 100; top_20 /= 100; top_30 /= 100
        make_cut /= 100

    F = FINISH_RANGE_PTS
    proj_pts = (
        win * F["win"]
        + max(0, top_3 - win) * F["top_3"]
        + max(0, top_5 - top_3) * F["top_5"]
        + max(0, top_10 - top_5) * F["top_10"]
        + max(0, top_20 - top_10) * F["top_20"]
        + max(0, top_30 - top_20) * F["top_30"]
        + max(0, make_cut - top_30) * F["made_cut"]
        + max(0, 1.0 - make_cut) * F["missed_cut"]
    )

    # Skill rating adjustment: sg_total ranges ~-2 to +3, optimal weight ~5 pts/SG
    proj_pts += sg_total * 5.0

    return max(proj_pts, 0.0)


# ── Player Builder ────────────────────────────────────────────────────────────

def _synthesize_ownership(players):
    """Synthesize ownership from projections using tempered softmax.

    Calibrated so max ownership ≈ 25-30% (matching real DK distributions).
    Total ownership = ROSTER_SIZE * 100% (6 players per lineup = 600%).
    """
    projs = np.array([p["projected_points"] for p in players])
    # Temperature = 3*std gives realistic ownership spread (max ~25-30%)
    temperature = max(projs.std() * 3, 1)
    exp_projs = np.exp((projs - projs.mean()) / temperature)
    synth_own = exp_projs / exp_projs.sum() * 100 * ROSTER_SIZE
    for i, p in enumerate(players):
        p["proj_ownership"] = round(float(synth_own[i]), 2)


def _gini_coefficient(values):
    """Compute Gini coefficient of a distribution (0=equal, 1=concentrated)."""
    arr = np.sort(np.array(values, dtype=np.float64))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


def _apply_adaptive_leverage(players):
    """Apply ownership leverage with power derived from ownership concentration.

    Leverage adjusts the MIP objective (mip_value) for candidate generation,
    NOT projected_points (which drives Monte Carlo simulation means).

    Concentrated ownership → higher leverage (exploit chalk/contrarian gap).
    Spread ownership → lower leverage (less alpha in fading chalk).
    """
    ownerships = [p["proj_ownership"] for p in players if p["proj_ownership"] > 0]
    if not ownerships:
        for p in players:
            p["mip_value"] = p["projected_points"]
        return

    gini = _gini_coefficient(ownerships)
    leverage_power = 0.25 + 0.15 * gini  # Range: ~0.30-0.40

    median_own = np.median(ownerships)
    if median_own <= 0:
        for p in players:
            p["mip_value"] = p["projected_points"]
        return

    for p in players:
        own = max(p["proj_ownership"], 0.1)
        mult = (median_own / own) ** leverage_power
        mult = max(0.70, min(1.50, mult))  # Floor/cap
        p["leverage_mult"] = round(mult, 3)
        # MIP objective gets leverage; projected_points stays clean for simulation
        p["mip_value"] = round(p["projected_points"] * mult, 2)


def build_players_for_backtest(predictions, dfs_data, event_id=None, year=None):
    """Build player list by merging DG predictions with actual DK data.

    From predictions: derive projected_points, std_dev, synthesized ownership
    From dfs_data: actual salary, actual total_pts
    Optionally uses FC projections when available for the event.

    Returns:
        players: list of player dicts (ready for engine.py functions)
    """
    # Build DFS lookup by dg_id
    dfs_points = dfs_data.get("dfs_points", dfs_data) if isinstance(dfs_data, dict) else dfs_data
    dfs_lookup = {}
    for p in dfs_points:
        dg_id = p.get("dg_id")
        if dg_id and p.get("salary") and p["salary"] > 0:
            dfs_lookup[dg_id] = p

    # Get prediction baseline
    pred_field = predictions.get("baseline", []) if isinstance(predictions, dict) else predictions

    # Skill ratings for enhanced projections
    sr_lookup = get_skill_ratings_lookup()

    # FC projections (when available for this event)
    fc_lookup = load_fc_projections(event_id, year) if event_id else None
    fc_used = 0

    players = []
    for pred in pred_field:
        dg_id = pred.get("dg_id")
        if dg_id not in dfs_lookup:
            continue

        dfs = dfs_lookup[dg_id]
        salary = dfs["salary"]
        if salary <= 0:
            continue

        # Skill rating enhancement
        sr = sr_lookup.get(dg_id, {})
        sg_total = sr.get("sg_total", 0)

        # Project DK points from pre-tournament probabilities + skill
        proj_pts = project_dk_points_from_probs(pred, salary=salary, sg_total=sg_total)

        # Name normalization (needed early for FC lookup)
        raw_name = pred.get("player_name", "")
        if ", " in raw_name:
            parts = raw_name.split(", ", 1)
            name = f"{parts[1]} {parts[0]}"
        else:
            name = raw_name

        # Blend with FC projection if available (60% FC, 40% DG model)
        if fc_lookup and name in fc_lookup:
            fc_proj = fc_lookup[name]["fc_proj"]
            if fc_proj > 0:
                proj_pts = 0.6 * fc_proj + 0.4 * proj_pts
                fc_used += 1

        # Empirical std_dev from calibration data
        make_cut = pred.get("make_cut", 0) or 0
        if make_cut > 1:
            make_cut /= 100
        std_dev = empirical_std_dev(make_cut)

        # FC pre-tournament ownership (if available)
        fc_own = 0
        if fc_lookup and name in fc_lookup:
            fc_own = fc_lookup[name].get("own_proj", 0) or 0

        players.append({
            "name": name,
            "dg_id": dg_id,
            "salary": salary,
            "projected_points": round(proj_pts, 2),
            "std_dev": round(std_dev, 1),
            "proj_ownership": 0,  # will be set below (FC or synthesized)
            "fc_own_proj": fc_own,
            "value": round(proj_pts / (salary / 1000), 2) if salary > 0 else 0,
            "actual_dk_pts": dfs.get("total_pts", 0) or 0,
            "actual_finish": dfs.get("fin_text", ""),
        })

    players.sort(key=lambda p: p["projected_points"], reverse=True)

    if fc_used > 0:
        print(f"    FC projections used for {fc_used}/{len(players)} players")

    # Use FC pre-tournament ownership if enough players have it; otherwise synthesize
    fc_own_count = sum(1 for p in players if p.get("fc_own_proj", 0) > 0)
    if players:
        if fc_own_count >= len(players) * 0.5:
            # Scale FC ownership to sum to ROSTER_SIZE * 100%
            total_fc_own = sum(p["fc_own_proj"] for p in players)
            target_total = ROSTER_SIZE * 100  # 600% total
            scale = target_total / total_fc_own if total_fc_own > 0 else 1
            for p in players:
                p["proj_ownership"] = round(max(p["fc_own_proj"] * scale, 0.1), 2)
            print(f"    FC ownership used for {fc_own_count}/{len(players)} players")
        else:
            _synthesize_ownership(players)
        _apply_adaptive_leverage(players)

    return players


# ── Payout Table ─────────────────────────────────────────────────────────────
# Calibrated from 18 real DraftKings PGA GPP contests via FantasyCruncher data.
# Entry fees: $5-$25, Fields: 26K-78K lineups.

def build_synthetic_payout_table(field_size, entry_fee):
    """GPP payout table calibrated from real DraftKings PGA contests.

    Empirical structure from FantasyCruncher All Entrants data:
    - 1st place: 25% of pool (was 17% in old synthetic)
    - Cash rate: ~21.5%
    - 14 payout tiers with realistic decay
    - Min-cash: ~0.66x entry fee

    Returns:
        (payout_table, entry_fee, prize_pool)
        payout_table: list of (min_pos, max_pos, prize) tuples
    """
    prize_pool = entry_fee * field_size * 0.85
    payout_spots = max(1, int(field_size * 0.215))

    payouts = [
        (1, 1, round(prize_pool * 0.2500)),         # 1st: 25.0%
        (2, 2, round(prize_pool * 0.0750)),          # 2nd: 7.5%
        (3, 3, round(prize_pool * 0.0333)),          # 3rd: 3.3%
        (4, 5, round(prize_pool * 0.0150)),          # 4-5: 1.5% each
        (6, 10, round(prize_pool * 0.00625)),        # 6-10: 0.625% each
        (11, 25, round(prize_pool * 0.001833)),      # 11-25
        (26, 50, round(prize_pool * 0.000646)),      # 26-50
        (51, 100, round(prize_pool * 0.000347)),     # 51-100
        (101, 250, round(prize_pool * 0.000128)),    # 101-250
        (251, 500, round(prize_pool * 0.000080)),    # 251-500
        (501, 1000, round(prize_pool * 0.000055)),   # 501-1000
        (1001, 2500, round(prize_pool * 0.000043)),  # 1001-2500
        (2501, 5000, round(prize_pool * 0.000036)),  # 2501-5000
    ]

    # Tail: min-cash
    if payout_spots > 5000:
        min_cash = max(1, round(entry_fee * 0.66, 2))
        payouts.append((5001, payout_spots, min_cash))

    # Trim tiers past payout_spots
    payouts = [(lo, min(hi, payout_spots), amt)
               for lo, hi, amt in payouts if lo <= payout_spots]

    return payouts, entry_fee, prize_pool


# ── Actual Result Scoring ─────────────────────────────────────────────────────

def compute_actual_roi(selected_lineups, opponent_lineups, players,
                       payout_table, entry_fee, contest_field_size):
    """Score selected lineups against actual results within opponent field.

    1. Score all lineups (ours + opponents) using actual DK points
    2. Rank all lineups by actual score
    3. Scale positions to contest field size
    4. Assign payouts from payout table
    5. Compute ROI
    """
    n_ours = len(selected_lineups)
    n_opps = len(opponent_lineups)
    n_total = n_ours + n_opps

    # Score all lineups using actual DK points
    all_lineups = list(selected_lineups) + list(opponent_lineups)
    all_scores = np.array([
        sum(players[idx]["actual_dk_pts"] for idx in lu)
        for lu in all_lineups
    ], dtype=np.float64)

    # Rank: highest score = position 1
    order = np.argsort(-all_scores)
    positions = np.empty(n_total, dtype=np.int32)
    positions[order] = np.arange(1, n_total + 1)

    # Scale positions to contest field size
    scale = contest_field_size / n_total
    our_positions = positions[:n_ours].copy()
    scaled_positions = np.rint(our_positions * scale).astype(np.int32)
    np.clip(scaled_positions, 1, contest_field_size, out=scaled_positions)

    # Build payout lookup and assign
    payout_by_pos = build_payout_lookup(payout_table, contest_field_size)
    lineup_payouts = payout_by_pos[scaled_positions]

    total_cost = entry_fee * n_ours
    total_payout = float(lineup_payouts.sum())
    roi_pct = (total_payout - total_cost) / total_cost * 100 if total_cost > 0 else 0
    cash_rate = float((lineup_payouts > 0).sum() / n_ours * 100) if n_ours > 0 else 0

    # Per-lineup details
    lineup_details = []
    for i in range(n_ours):
        lineup_details.append({
            "actual_score": float(all_scores[i]),
            "position": int(our_positions[i]),
            "scaled_position": int(scaled_positions[i]),
            "payout": float(lineup_payouts[i]),
            "players": [players[idx]["name"] for idx in selected_lineups[i]],
        })
    lineup_details.sort(key=lambda x: x["actual_score"], reverse=True)

    return {
        "total_cost": total_cost,
        "total_payout": total_payout,
        "roi_pct": roi_pct,
        "cash_rate": cash_rate,
        "n_lineups": n_ours,
        "avg_actual_score": float(all_scores[:n_ours].mean()),
        "best_score": float(all_scores[:n_ours].max()),
        "worst_score": float(all_scores[:n_ours].min()),
        "lineup_details": lineup_details,
    }


# ── Single Event Pipeline ────────────────────────────────────────────────────

def backtest_single_event(event, params):
    """Run full optimizer pipeline against a single historical event.

    Returns EventResult dict or None if data unavailable.
    """
    event_id = event["event_id"]
    year = event.get("calendar_year") or event.get("year")
    name = event.get("event_name", f"Event {event_id}")

    # Step 1: Fetch data
    try:
        predictions = fetch_predictions_archive(event_id, year)
        dfs_data = fetch_historical_dfs_points(event_id, year)
    except Exception as e:
        print(f"    Skip: {e}")
        return None

    # Step 2: Build player list
    players = build_players_for_backtest(predictions, dfs_data, event_id=event_id, year=year)
    min_field = params.get("min_field_size", 30)
    if len(players) < min_field:
        print(f"    Skip: only {len(players)} players (min: {min_field})")
        return None

    print(f"    Players: {len(players)} | "
          f"Top proj: {players[0]['name']} ({players[0]['projected_points']:.1f}pts)")

    # Step 3: Generate opponent field
    seed = params.get("seed", 42)
    opponents = generate_field(players, params["field_size"], seed=seed)
    print(f"    Opponents: {len(opponents):,}")

    # Step 4: Generate candidate lineups (using projected points)
    candidates = generate_candidates(players, pool_size=params["n_candidates"], seed=seed)
    n_lineups = min(params["n_lineups"], len(candidates))
    if n_lineups < 5:
        print(f"    Skip: only {len(candidates)} candidates")
        return None
    print(f"    Candidates: {len(candidates):,}")

    # Step 5: Simulate positions (Monte Carlo)
    positions_matrix, n_total = simulate_positions(
        candidates, opponents, players, n_sims=params["n_sims"], seed=seed
    )

    # Step 6: Build payout table (prefer real FC data, fallback to synthetic)
    fc_payouts = load_fc_payout_table(event_id, year)
    if fc_payouts:
        payout_table, entry_fee, prize_pool, fc_field_size = fc_payouts
        contest_field_size = fc_field_size
        print(f"    Real FC payouts: fee=${entry_fee:.0f} pool=${prize_pool:,.0f} "
              f"field={contest_field_size:,}")
    else:
        payout_table, entry_fee, prize_pool = build_synthetic_payout_table(
            params["contest_field_size"], params["entry_fee"]
        )
        contest_field_size = params["contest_field_size"]

    # Step 7: Assign simulated payouts for portfolio selection
    payouts = assign_payouts(
        positions_matrix, payout_table, n_total, params["n_sims"],
        contest_field_size=contest_field_size
    )

    # Step 8: Select portfolio via E[max] greedy
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

    # Add event metadata
    result["event_name"] = name
    result["event_id"] = event_id
    result["year"] = year
    result["n_players"] = len(players)
    result["n_candidates"] = len(candidates)
    result["n_opponents"] = len(opponents)
    result["prize_pool"] = prize_pool
    result["fc_data"] = fc_payouts is not None

    # Projection accuracy metrics
    all_proj = [players[idx]["projected_points"] for lu in selected_lineups for idx in lu]
    all_actual = [players[idx]["actual_dk_pts"] for lu in selected_lineups for idx in lu]
    result["avg_proj_pts_player"] = float(np.mean(all_proj))
    result["avg_actual_pts_player"] = float(np.mean(all_actual))
    if len(all_proj) > 2:
        result["projection_corr"] = float(np.corrcoef(all_proj, all_actual)[0, 1])
    else:
        result["projection_corr"] = 0.0

    return result


# ── Multi-Event Loop ──────────────────────────────────────────────────────────

def run_backtest(events, params):
    """Run full pipeline backtest across multiple events."""
    results = []

    for i, event in enumerate(events):
        name = event.get("event_name", "Unknown")
        year = event.get("calendar_year") or event.get("year")

        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(events)}] {name} {year}")
        print(f"{'='*70}")

        t0 = time.time()
        result = backtest_single_event(event, params)
        elapsed = time.time() - t0

        if result is None:
            continue

        results.append(result)

        print(f"    ROI: {result['roi_pct']:+.1f}% | Cash: {result['cash_rate']:.1f}% | "
              f"Lineups: {result['n_lineups']} | Avg Score: {result['avg_actual_score']:.1f} | "
              f"{elapsed:.0f}s")

    return results


# ── Summary Reporting ─────────────────────────────────────────────────────────

def print_summary(results, params):
    """Print aggregate backtest summary."""
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

    print(f"\n{'='*70}")
    print(f"  BACKTEST SUMMARY — {len(results)} events")
    print(f"{'='*70}")
    print(f"\n  Parameters:")
    print(f"    Lineups: {params['n_lineups']} | Sims: {params['n_sims']:,} | "
          f"Field: {params['field_size']:,} | Contest Field: {params['contest_field_size']:,}")
    print(f"    Entry Fee: ${params['entry_fee']:.0f} | Max Exposure: {params['max_exposure']:.0%}")

    print(f"\n  Per-Event Statistics:")
    print(f"    Mean ROI:        {mean_roi:+.1f}%")
    print(f"    Median ROI:      {median_roi:+.1f}%")
    print(f"    Std Dev ROI:     {std_roi:.1f}%")
    print(f"    Sharpe Ratio:    {sharpe:.3f}")
    print(f"    Mean Cash Rate:  {np.mean(cash_rates):.1f}%")
    print(f"    Win Rate:        {sum(1 for r in rois if r > 0)}/{len(rois)} "
          f"({sum(1 for r in rois if r > 0)/len(rois)*100:.0f}%)")
    print(f"    Best Event:      {max(rois):+.1f}%")
    print(f"    Worst Event:     {min(rois):+.1f}%")

    print(f"\n  Dollar-Weighted (pooled across all events):")
    print(f"    Total Cost:      ${total_cost:,.0f}")
    print(f"    Total Payout:    ${total_payout:,.0f}")
    print(f"    Net Profit:      ${total_payout - total_cost:+,.0f}")
    print(f"    Overall ROI:     {overall_roi:+.1f}%")

    avg_proj_corr = np.mean([r["projection_corr"] for r in results])
    print(f"\n  Projection Quality:")
    print(f"    Avg Proj-Actual Correlation: {avg_proj_corr:.4f}")

    # FC data stats
    fc_count = sum(1 for r in results if r.get("fc_data"))
    if fc_count > 0:
        print(f"\n  Data Source:")
        print(f"    FC-enhanced events: {fc_count}/{len(results)} "
              f"(real ownership + payouts)")

    # Per-event table
    print(f"\n  {'Event':<35} {'Year':>5} {'ROI':>8} {'Cash%':>7} {'LU':>4} "
          f"{'Avg Scr':>8} {'Best':>7} {'Worst':>7} {'Src':>4}")
    print(f"  {'-'*35} {'-'*5} {'-'*8} {'-'*7} {'-'*4} "
          f"{'-'*8} {'-'*7} {'-'*7} {'-'*4}")
    for r in results:
        src = "FC" if r.get("fc_data") else "SYN"
        print(f"  {r['event_name'][:35]:<35} {r['year']:>5} "
              f"{r['roi_pct']:>+7.1f}% {r['cash_rate']:>6.1f}% "
              f"{r['n_lineups']:>4} {r['avg_actual_score']:>8.1f} "
              f"{r['best_score']:>7.1f} {r['worst_score']:>7.1f} {src:>4}")


# ── Parameter Sweep ───────────────────────────────────────────────────────────

def run_parameter_sweep(events, base_params, sweep_config):
    """Grid search over parameter combinations to find optimal settings."""
    param_names = list(sweep_config.keys())
    param_values = list(sweep_config.values())

    sweep_results = []
    total_combos = 1
    for v in param_values:
        total_combos *= len(v)

    print(f"\n{'#'*70}")
    print(f"  PARAMETER SWEEP — {total_combos} combinations × {len(events)} events")
    print(f"{'#'*70}")

    for combo_idx, combo in enumerate(product(*param_values)):
        params = base_params.copy()
        for name, val in zip(param_names, combo):
            params[name] = val

        label = ", ".join(f"{n}={v}" for n, v in zip(param_names, combo))
        print(f"\n  [{combo_idx+1}/{total_combos}] {label}")

        results = run_backtest(events, params)

        if results:
            rois = [r["roi_pct"] for r in results]
            total_cost = sum(r["total_cost"] for r in results)
            total_payout = sum(r["total_payout"] for r in results)
            overall_roi = (total_payout - total_cost) / total_cost * 100 if total_cost > 0 else 0

            sweep_results.append({
                "params": dict(zip(param_names, combo)),
                "mean_roi": float(np.mean(rois)),
                "median_roi": float(np.median(rois)),
                "std_roi": float(np.std(rois)),
                "sharpe": float(np.mean(rois) / np.std(rois)) if np.std(rois) > 0 else 0,
                "overall_roi": overall_roi,
                "mean_cash": float(np.mean([r["cash_rate"] for r in results])),
                "n_events": len(results),
            })

    # Print ranked results
    sweep_results.sort(key=lambda x: x["overall_roi"], reverse=True)

    print(f"\n{'='*70}")
    print(f"  PARAMETER SWEEP RESULTS (ranked by overall ROI)")
    print(f"{'='*70}")
    print(f"  {'Parameters':<45} {'ROI':>8} {'Sharpe':>8} {'Cash%':>7} {'Events':>7}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
    for sr in sweep_results:
        params_str = ", ".join(f"{k}={v}" for k, v in sr["params"].items())
        print(f"  {params_str:<45} {sr['overall_roi']:>+7.1f}% "
              f"{sr['sharpe']:>7.3f} {sr['mean_cash']:>6.1f}% {sr['n_events']:>7}")

    return sweep_results


# ── Export ────────────────────────────────────────────────────────────────────

def export_results_csv(results, filepath):
    """Export backtest results to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Event", "Year", "ROI%", "Cash Rate%", "Lineups",
            "Total Cost", "Total Payout", "Avg Actual Score",
            "Best Score", "Worst Score", "Players", "Candidates",
            "Proj Correlation",
        ])
        for r in results:
            writer.writerow([
                r["event_name"], r["year"], round(r["roi_pct"], 2),
                round(r["cash_rate"], 1), r["n_lineups"],
                round(r["total_cost"], 2), round(r["total_payout"], 2),
                round(r["avg_actual_score"], 1),
                round(r["best_score"], 1), round(r["worst_score"], 1),
                r["n_players"], r["n_candidates"],
                round(r["projection_corr"], 4),
            ])
    print(f"  CSV saved: {filepath}")


def export_results_sheets(results, params, sweep_results=None):
    """Export backtest results to Google Sheets."""
    from google_sheets import _get_client, _get_or_create_worksheet, SHEET_ID, HEADER_FMT

    client = _get_client()
    spreadsheet = client.open_by_key(SHEET_ID)

    # ── Tab: Backtest Summary ──
    ws = _get_or_create_worksheet(spreadsheet, "BT Summary", len(results) + 25, 13)

    rois = [r["roi_pct"] for r in results]
    total_cost = sum(r["total_cost"] for r in results)
    total_payout = sum(r["total_payout"] for r in results)
    overall_roi = (total_payout - total_cost) / total_cost * 100 if total_cost > 0 else 0

    rows = [
        ["FULL PIPELINE BACKTEST", "", "", "", "", "", "", "", "", "", "", "", ""],
        [f"Events: {len(results)}",
         f"Overall ROI: {overall_roi:+.1f}%",
         f"Mean ROI: {np.mean(rois):+.1f}%",
         f"Sharpe: {np.mean(rois)/np.std(rois):.3f}" if np.std(rois) > 0 else "Sharpe: N/A",
         f"Total Cost: ${total_cost:,.0f}",
         f"Total Payout: ${total_payout:,.0f}",
         "", "", "", "", "", "", ""],
        [f"Params: {params['n_lineups']}LU, {params['n_sims']}sims, "
         f"${params['entry_fee']}fee, {params['max_exposure']:.0%}exp",
         "", "", "", "", "", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", "", "", "", "", "", ""],
        ["Event", "Year", "ROI%", "Cash%", "LU", "Cost", "Payout",
         "Avg Score", "Best", "Worst", "Players", "Candidates", "Proj Corr"],
    ]

    for r in results:
        rows.append([
            r["event_name"], r["year"], round(r["roi_pct"], 1),
            round(r["cash_rate"], 1), r["n_lineups"],
            round(r["total_cost"]), round(r["total_payout"]),
            round(r["avg_actual_score"], 1),
            round(r["best_score"], 1), round(r["worst_score"], 1),
            r["n_players"], r["n_candidates"],
            round(r["projection_corr"], 4),
        ])

    ws.update(rows, value_input_option="RAW")
    ws.format("A1:M1", HEADER_FMT)
    ws.format("A5:M5", HEADER_FMT)

    time.sleep(3)

    # ── Tab: Sweep Results (if available) ──
    if sweep_results:
        sw_ws = _get_or_create_worksheet(spreadsheet, "BT Sweep", len(sweep_results) + 5, 8)
        sw_rows = [
            ["PARAMETER SWEEP RESULTS", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["Parameters", "Overall ROI%", "Mean ROI%", "Median ROI%",
             "Std ROI%", "Sharpe", "Cash%", "Events"],
        ]
        for sr in sweep_results:
            params_str = ", ".join(f"{k}={v}" for k, v in sr["params"].items())
            sw_rows.append([
                params_str, round(sr["overall_roi"], 1), round(sr["mean_roi"], 1),
                round(sr["median_roi"], 1), round(sr["std_roi"], 1),
                round(sr["sharpe"], 3), round(sr["mean_cash"], 1), sr["n_events"],
            ])
        sw_ws.update(sw_rows, value_input_option="RAW")
        sw_ws.format("A1:H1", HEADER_FMT)
        sw_ws.format("A3:H3", HEADER_FMT)

    return spreadsheet.url


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DFS Golf — Full Pipeline Backtester")
    parser.add_argument("--events", type=int, default=20,
                        help="Number of recent events to backtest (default: 20)")
    parser.add_argument("--year", type=int, default=None,
                        help="Filter to specific year")
    parser.add_argument("--candidates", type=int, default=5000,
                        help="Candidate pool size (default: 5000)")
    parser.add_argument("--sims", type=int, default=10000,
                        help="Monte Carlo simulations (default: 10000)")
    parser.add_argument("--field-size", type=int, default=50000,
                        help="Opponent field size (default: 50000)")
    parser.add_argument("--contest-field-size", type=int, default=47000,
                        help="Synthetic contest field size for payouts (default: 47000)")
    parser.add_argument("--lineups", type=int, default=150,
                        help="Lineups to select per event (default: 150)")
    parser.add_argument("--entry-fee", type=float, default=25.0,
                        help="Synthetic entry fee (default: $25)")
    parser.add_argument("--max-exposure", type=float, default=0.60,
                        help="Max exposure cap (default: 0.60)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sensitivity sweep")
    parser.add_argument("--sheets", action="store_true",
                        help="Export results to Google Sheets")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export results CSV to this path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--min-field-size", type=int, default=30,
                        help="Skip events with fewer than N players (default: 30)")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("  DFS GOLF — FULL PIPELINE BACKTESTER")
    print("=" * 70)

    params = {
        "n_candidates": args.candidates,
        "n_sims": args.sims,
        "field_size": args.field_size,
        "contest_field_size": args.contest_field_size,
        "n_lineups": args.lineups,
        "entry_fee": args.entry_fee,
        "max_exposure": args.max_exposure,
        "seed": args.seed,
        "min_field_size": args.min_field_size,
    }

    # Get events
    print(f"\n  Fetching historical event list...")
    events = get_backtest_events_dfs(year=args.year, max_events=args.events)
    print(f"  Found {len(events)} events with DK data")

    if not events:
        print("  No events found. Try a different year or increase --events.")
        return

    # Print event list
    print(f"\n  {'Event':<40} {'Year':>5} {'Date':>12}")
    print(f"  {'-'*40} {'-'*5} {'-'*12}")
    for e in events[:10]:
        print(f"  {e['event_name'][:40]:<40} {e.get('calendar_year', '?'):>5} "
              f"{e.get('date', '?'):>12}")
    if len(events) > 10:
        print(f"  ... and {len(events) - 10} more")

    print(f"\n  Parameters:")
    print(f"    Lineups: {params['n_lineups']} | Candidates: {params['n_candidates']:,} | "
          f"Sims: {params['n_sims']:,}")
    print(f"    Field: {params['field_size']:,} | Contest Field: {params['contest_field_size']:,}")
    print(f"    Entry Fee: ${params['entry_fee']:.0f} | Max Exposure: {params['max_exposure']:.0%}")
    print(f"    Seed: {params['seed']} | Min Field: {params['min_field_size']}")

    sweep_results = None

    if args.sweep:
        sweep_config = {
            "n_lineups": [50, 100, 150],
            "max_exposure": [0.40, 0.60, 0.80],
        }
        sweep_events = events[:min(10, len(events))]
        sweep_results = run_parameter_sweep(sweep_events, params, sweep_config)

        # Also run default params for full summary
        print(f"\n{'='*70}")
        print(f"  Running full backtest with default parameters...")
        print(f"{'='*70}")
        results = run_backtest(events, params)
    else:
        results = run_backtest(events, params)

    print_summary(results, params)

    # Save results JSON
    base = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base, "history")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "backtest_all_results.json")
    summary = {
        "params": params,
        "n_events": len(results),
        "overall_roi": (sum(r["total_payout"] for r in results) - sum(r["total_cost"] for r in results))
                       / sum(r["total_cost"] for r in results) * 100 if results else 0,
        "mean_roi": float(np.mean([r["roi_pct"] for r in results])) if results else 0,
        "per_event": [{k: v for k, v in r.items() if k != "lineup_details"} for r in results],
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    if args.csv:
        export_results_csv(results, args.csv)

    if args.sheets:
        print(f"\n  Exporting to Google Sheets...")
        url = export_results_sheets(results, params, sweep_results=sweep_results)
        print(f"  Sheet: {url}")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Done in {elapsed:.0f}s — {len(results)} events backtested")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

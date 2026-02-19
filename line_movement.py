"""Vegas Line Movement — Prediction Snapshot Tracking

Save/load/compare prediction snapshots to identify sharp money movers
(players whose make_cut probability shifts significantly between early
and late-week predictions).
"""
import os
import json
from datetime import datetime

from config import MOVEMENT_THRESHOLD_PCT, MOVEMENT_ADJ_CAP

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history", "snapshots")


def save_prediction_snapshot(predictions):
    """Save current predictions to history/snapshots/ with date stamp.

    predictions: raw DataGolf predictions response dict.
    """
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    event_name = predictions.get("event_name", "unknown")
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    slug = event_name.replace(" ", "_").replace("'", "")
    filename = f"snapshot_{slug}_{date_str}.json"
    filepath = os.path.join(SNAPSHOT_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"  Snapshot saved: {filepath}")
    return filepath


def load_latest_snapshot(event_name):
    """Load the most recent snapshot for a given event name.

    Returns the parsed predictions dict, or None if no snapshot found.
    """
    if not os.path.isdir(SNAPSHOT_DIR):
        return None

    slug = event_name.replace(" ", "_").replace("'", "")
    matching = []
    for fname in os.listdir(SNAPSHOT_DIR):
        if fname.startswith(f"snapshot_{slug}_") and fname.endswith(".json"):
            matching.append(fname)

    if not matching:
        return None

    # Sort by filename (which includes date) and pick the most recent
    matching.sort()
    latest = matching[-1]
    filepath = os.path.join(SNAPSHOT_DIR, latest)

    with open(filepath) as f:
        return json.load(f)


def compare_snapshots(early, current):
    """Compare two prediction snapshots and identify significant movers.

    Returns list of dicts with player_name, dg_id, early_mc, current_mc, delta_mc
    for players whose make_cut% changed by more than MOVEMENT_THRESHOLD_PCT.
    Sorted by absolute delta descending.
    """
    # Build lookup from early snapshot
    early_field = early.get("baseline", [])
    early_lookup = {}
    for p in early_field:
        dg_id = p.get("dg_id")
        if dg_id is not None:
            early_lookup[dg_id] = p

    # Compare with current
    current_field = current.get("baseline", [])
    movers = []
    for p in current_field:
        dg_id = p.get("dg_id")
        if dg_id is None or dg_id not in early_lookup:
            continue

        early_mc = early_lookup[dg_id].get("make_cut", 0) or 0
        current_mc = p.get("make_cut", 0) or 0
        delta = current_mc - early_mc

        if abs(delta) >= MOVEMENT_THRESHOLD_PCT:
            movers.append({
                "player_name": p.get("player_name", "Unknown"),
                "dg_id": dg_id,
                "early_mc": early_mc,
                "current_mc": current_mc,
                "delta_mc": delta,
            })

    movers.sort(key=lambda m: abs(m["delta_mc"]), reverse=True)
    return movers


def get_movement_adjustment(dg_id, movement_data):
    """Get an optional conservative adjustment based on line movement.

    movement_data: list from compare_snapshots().
    Returns ±MOVEMENT_ADJ_CAP pts adjustment (very conservative).
    """
    for m in movement_data:
        if m["dg_id"] == dg_id:
            # Positive delta = sharp money on player (MC% rose)
            # Negative delta = sharp money against (MC% fell)
            if m["delta_mc"] > 0:
                return min(MOVEMENT_ADJ_CAP, m["delta_mc"] * 0.1)
            else:
                return max(-MOVEMENT_ADJ_CAP, m["delta_mc"] * 0.1)
    return 0.0


def get_movement_adjustments(movement_data):
    """Convert movement report to {dg_id: adjustment_pts} lookup.

    Bulk version of get_movement_adjustment() for wiring into the
    projection pipeline. Returns dict mapping dg_id → point adjustment.
    """
    adjustments = {}
    for m in movement_data:
        adj = m["delta_mc"] * 0.1
        adj = max(-MOVEMENT_ADJ_CAP, min(MOVEMENT_ADJ_CAP, adj))
        adjustments[m["dg_id"]] = round(adj, 2)
    return adjustments

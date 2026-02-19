import requests
from config import DATAGOLF_API_KEY, DATAGOLF_BASE_URL, TOUR, ODDS_FORMAT


def _get(endpoint, params=None):
    """Make an authenticated GET request to the DataGolf API."""
    if params is None:
        params = {}
    params["key"] = DATAGOLF_API_KEY
    url = f"{DATAGOLF_BASE_URL}{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def american_to_implied_prob(odds):
    """Convert American odds to implied probability."""
    if odds is None:
        return 0.0
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 0.0


def decimal_to_implied_prob(odds):
    """Convert decimal odds to implied probability."""
    if odds is None or odds <= 0:
        return 0.0
    return 1.0 / float(odds)


def get_player_list():
    """Fetch the full DataGolf player list with dg_id mappings."""
    data = _get("/get-player-list")
    # Returns a list of dicts: {dg_id, player_name, country, ...}
    return data


def get_predictions():
    """Fetch pre-tournament predictions for the current week.

    Returns dict with 'baseline' and 'baseline_history' keys.
    Each player entry has: dg_id, player_name, win, top_5, top_10, top_20, make_cut, etc.
    """
    data = _get("/preds/pre-tournament", params={
        "tour": TOUR,
        "odds_format": "percent",
    })
    return data


def get_event_list():
    """Fetch the historical DFS event list for looking up event IDs."""
    data = _get("/historical-dfs-data/event-list", params={
        "tour": TOUR,
    })
    return data


def get_prediction_archive(event_id, year):
    """Fetch archived pre-tournament predictions for a past event.

    Used for course history analysis.
    """
    data = _get("/preds/pre-tournament-archive", params={
        "tour": TOUR,
        "event_id": event_id,
        "year": year,
        "odds_format": "percent",
    })
    return data


def get_historical_dfs(event_id, year, market="draftkings"):
    """Fetch historical DFS salary and results data for a past event."""
    data = _get("/historical-dfs-data/dfs-points", params={
        "tour": TOUR,
        "event_id": event_id,
        "year": year,
        "market": market,
    })
    return data


def find_event_id(event_list, event_name):
    """Search the event list for an event by name (partial match)."""
    event_name_lower = event_name.lower()
    for event in event_list:
        if event_name_lower in event.get("event_name", "").lower():
            return event.get("event_id")
    return None


def get_player_decompositions():
    """Fetch player prediction decompositions for the current week.

    Returns per-player breakdown including:
    - baseline_pred, final_pred
    - total_fit_adjustment, driving_accuracy_adjustment, driving_distance_adjustment
    - strokes_gained_category_adjustment, total_course_history_adjustment
    - timing_adjustment (recent form signal)
    """
    data = _get("/preds/player-decompositions", params={
        "tour": TOUR,
    })
    return data


def get_skill_ratings():
    """Fetch player skill ratings (strokes gained breakdown).

    Returns per-player: sg_ott, sg_app, sg_arg, sg_putt, sg_total
    """
    data = _get("/preds/skill-ratings", params={
        "display": "value",
    })
    return data


def get_fantasy_projections():
    """Fetch DFS fantasy projections including ownership.

    Returns per-player: proj_ownership, proj_points_total, std_dev,
    early_late_wave, r1_teetime
    """
    data = _get("/preds/fantasy-projection-defaults", params={
        "tour": TOUR,
        "site": "draftkings",
        "slate": "main",
    })
    return data


def find_current_event(predictions):
    """Extract current event info from predictions response."""
    return {
        "event_name": predictions.get("event_name", "Unknown"),
        "event_id": predictions.get("event_id"),
        "course": predictions.get("course_name", "Unknown"),
    }

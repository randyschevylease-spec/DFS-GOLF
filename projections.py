"""DFS Golf Projections — DataGolf pass-through with situational edges.

Uses DataGolf's fantasy-projection-defaults as the baseline projection.
Applies additive adjustments for edges the market underweights:
  - Weather/wave advantage (from weather.py)
  - Line movement signal (from line_movement.py)
"""


def build_projection(dg_proj, wave_adj=0.0, movement_adj=0.0):
    """Build player projection from DataGolf fantasy data + situational edges.

    dg_proj: dict from DataGolf fantasy-projection-defaults endpoint
        Required keys: proj_points_total, salary, player_name, site_name_id
        Optional keys: std_dev, proj_ownership, proj_points_finish,
                       proj_points_scoring, early_late_wave, r1_teetime
    wave_adj: weather/wave advantage in DK pts (from weather.py)
    movement_adj: line movement signal in DK pts (from line_movement.py)

    Returns player dict ready for the optimizer.
    """
    proj_pts = dg_proj["proj_points_total"] + wave_adj + movement_adj
    salary = dg_proj["salary"]

    # Flip "Last, First" → "First Last" for display
    raw_name = dg_proj["player_name"]
    if ", " in raw_name:
        parts = raw_name.split(", ", 1)
        display_name = f"{parts[1]} {parts[0]}"
    else:
        display_name = raw_name

    return {
        "name": display_name,
        "name_id": dg_proj["site_name_id"],
        "salary": salary,
        "projected_points": round(proj_pts, 2),
        "std_dev": dg_proj.get("std_dev", 0),
        "proj_ownership": dg_proj.get("proj_ownership", 0),
        "dg_proj_raw": round(dg_proj["proj_points_total"], 2),
        "proj_points_finish": round(dg_proj.get("proj_points_finish", 0), 2),
        "proj_points_scoring": round(dg_proj.get("proj_points_scoring", 0), 2),
        "value": round(proj_pts / (salary / 1000.0), 2) if salary > 0 else 0,
        "wave": "AM" if dg_proj.get("early_late_wave") == 1 else "PM",
        "wave_adj": round(wave_adj, 2),
        "movement_adj": round(movement_adj, 2),
        "r1_teetime": dg_proj.get("r1_teetime", ""),
    }


def build_fallback_projection(name, name_id, salary, dk_avg_points):
    """Build a fallback projection for DK players not in DataGolf.

    Uses DK AvgPointsPerGame as the projection. These players will have
    no std_dev (optimizer falls back to heuristic sigma estimation).
    """
    proj_pts = dk_avg_points if dk_avg_points and dk_avg_points > 0 else 0
    return {
        "name": name,
        "name_id": name_id,
        "salary": salary,
        "projected_points": round(proj_pts, 2),
        "std_dev": 0,
        "proj_ownership": 0,
        "dg_proj_raw": 0,
        "proj_points_finish": 0,
        "proj_points_scoring": 0,
        "value": round(proj_pts / (salary / 1000.0), 2) if salary > 0 else 0,
        "wave": "?",
        "wave_adj": 0,
        "movement_adj": 0,
        "r1_teetime": "",
        "dg_matched": False,
    }

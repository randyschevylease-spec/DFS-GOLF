#!/usr/bin/env python3
"""BOT 2: DataGolf Data Pipeline — Collects all player data."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import *

log = setup_logger("bot2", "bot2_data.log")

ENDPOINTS = [
    ("fantasy_projections", "preds/fantasy-projection-defaults", {"tour": "pga", "site": "draftkings"}),
    ("pre_tournament", "preds/pre-tournament", {"tour": "pga", "odds_format": "percent"}),
    ("skill_decompositions", "preds/player-decompositions", {"tour": "pga"}),
    ("skill_ratings", "preds/skill-ratings", {}),
    ("field_updates", "field-updates", {"tour": "pga"}),
    ("approach_skill", "preds/approach-skill", {"tour": "pga"}),
    ("dg_rankings", "preds/get-dg-rankings", {}),
]

def _get_wave(field_entry, fantasy_entry):
    """Extract R1 wave from field updates teetimes or fantasy data."""
    # Try field updates teetimes (has explicit wave)
    teetimes = field_entry.get("teetimes", [])
    for tt in teetimes:
        if tt.get("round_num") == 1:
            w = tt.get("wave", "")
            if w:
                return w.upper() if isinstance(w, str) else w
    # Try fantasy early_late_wave (1=AM, 2=PM)
    elw = fantasy_entry.get("early_late_wave")
    if elw == 1:
        return "EARLY"
    elif elw == 2:
        return "LATE"
    elif elw == 0:
        return "UNKNOWN"
    return "UNKNOWN"


def run():
    log.info("=" * 60)
    log.info("BOT 2: DataGolf Data Pipeline")
    log.info("=" * 60)

    raw_data = {}

    for name, endpoint, params in ENDPOINTS:
        try:
            log.info(f"Fetching {name}...")
            data = dg_api(endpoint, params)
            raw_data[name] = data
            count = len(data) if isinstance(data, list) else "ok"
            log.info(f"  {name}: {count}")
        except Exception as e:
            log.warning(f"  {name}: {e}")
            raw_data[name] = []

    # Build unified player pool
    fantasy_raw = raw_data.get("fantasy_projections", [])
    # Handle both list and dict formats (API returns dict with 'projections' key)
    if isinstance(fantasy_raw, dict):
        fantasy = fantasy_raw.get("projections", [])
    elif isinstance(fantasy_raw, list):
        fantasy = fantasy_raw
    else:
        fantasy = []
    if not fantasy:
        log.error("No fantasy projections — cannot proceed")
        sys.exit(1)

    pre_tourney = raw_data.get("pre_tournament", {})
    pre_tourney_list = pre_tourney if isinstance(pre_tourney, list) else pre_tourney.get("baseline", [])
    pre_lookup = {str(p.get("dg_id", "")): p for p in pre_tourney_list} if pre_tourney_list else {}

    decomp = raw_data.get("skill_decompositions", [])
    if isinstance(decomp, dict):
        decomp = decomp.get("players", decomp.get("decompositions", []))
    decomp_lookup = {}
    if isinstance(decomp, list):
        decomp_lookup = {str(p.get("dg_id", "")): p for p in decomp if isinstance(p, dict)}

    # Skill ratings (has per-category SG data)
    skill = raw_data.get("skill_ratings", {})
    skill_list = skill.get("players", skill) if isinstance(skill, dict) else skill
    skill_lookup = {}
    if isinstance(skill_list, list):
        skill_lookup = {str(p.get("dg_id", "")): p for p in skill_list if isinstance(p, dict)}

    field = raw_data.get("field_updates", {})
    field_list = field if isinstance(field, list) else field.get("field", [])
    field_lookup = {str(p.get("dg_id", "")): p for p in field_list} if field_list else {}

    players = []
    for fp in fantasy:
        dg_id = str(fp.get("dg_id", ""))
        name = fp.get("player_name", "")
        salary = fp.get("salary", 0)

        if salary <= 0:
            continue

        pt = pre_lookup.get(dg_id, {})
        sk = skill_lookup.get(dg_id, {})
        dc = decomp_lookup.get(dg_id, {})
        fl = field_lookup.get(dg_id, {})

        # Estimate ceiling/floor from std_dev if not provided
        std_dev = fp.get("std_dev", 0) or 0
        proj_pts = fp.get("proj_points_total", 0)
        ceiling = fp.get("ceiling", proj_pts + 2.0 * std_dev if std_dev else proj_pts * 1.3)
        floor = fp.get("floor", max(proj_pts - 1.5 * std_dev, 0) if std_dev else proj_pts * 0.3)

        players.append({
            "name": name,
            "dg_id": dg_id,
            "dk_salary": salary,
            "site_name_id": fp.get("site_name_id", ""),
            "dg_projection": {
                "proj_dk_pts": proj_pts,
                "proj_ownership_pct": fp.get("proj_ownership", 0),
                "ceiling": round(ceiling, 1),
                "floor": round(floor, 1),
                "std_dev": std_dev,
            },
            "dg_model_probs": {
                "win": pt.get("win", pt.get("win_prob", 0)),
                "top_5": pt.get("top_5", pt.get("top_5_prob", 0)),
                "top_10": pt.get("top_10", pt.get("top_10_prob", 0)),
                "top_20": pt.get("top_20", pt.get("top_20_prob", 0)),
                "make_cut": pt.get("make_cut", pt.get("make_cut_prob", 0.5)),
            },
            "sg_predictions": {
                "sg_total": sk.get("sg_total", dc.get("sg_total", 0)),
                "sg_ott": sk.get("sg_ott", 0),
                "sg_app": sk.get("sg_app", 0),
                "sg_arg": sk.get("sg_arg", 0),
                "sg_putt": sk.get("sg_putt", 0),
            },
            "tee_time_info": {
                "wave": _get_wave(fl, fp),
                "r1_tee_time": fp.get("r1_teetime", ""),
            },
        })

    output = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "player_count": len(players),
            "endpoints_fetched": len([k for k, v in raw_data.items() if v]),
        },
        "players": players,
    }

    SHARED.mkdir(parents=True, exist_ok=True)
    with open(SHARED / "player_pool.json", "w") as f:
        json.dump(output, f, indent=2)
    with open(SHARED / "datagolf_raw.json", "w") as f:
        json.dump(raw_data, f, indent=2, default=str)
    Path(SHARED / "raw_data_ready.flag").touch()

    log.info(f"\nPlayer pool: {len(players)} players saved to shared/player_pool.json")

if __name__ == "__main__":
    run()

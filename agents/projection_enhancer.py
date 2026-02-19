#!/usr/bin/env python3
"""BOT 3: DK-Specific Projection Enhancement + Weather Wave Adjustment."""
import json, sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_utils import *
from weather import get_wave_adjustment
from datagolf_client import get_predictions, find_current_event

log = setup_logger("bot3", "bot3_projections.log")

# Tunable parameters (adjusted by BOT 7 backtesting)
BIRDIE_CLUSTER_COEFF = 0.15
BOGEY_FREE_RATE_ADJ = 0.08
SPREAD_DIVISOR = 2.56

def enhance_player(p):
    """Add DK-specific scoring adjustments to DG baseline projection."""
    dg = p.get("dg_projection", {})
    sg = p.get("sg_predictions", {})
    probs = p.get("dg_model_probs", {})

    base_pts = dg.get("proj_dk_pts", 0)
    base_ceil = dg.get("ceiling", 0)
    base_floor = dg.get("floor", 0)

    if base_pts <= 0:
        base_pts = 30  # Fallback minimum

    # DK Bonus Estimation
    # Birdie streak bonus: correlates with birdie rate and SG:APP
    sg_app = sg.get("sg_app", 0)
    streak_bonus = max(sg_app * BIRDIE_CLUSTER_COEFF * 3, 0)  # 3 pts per streak

    # Bogey-free round bonus: correlates with SG:ARG
    sg_arg = sg.get("sg_arg", 0)
    bogey_free_prob = min(max(0.12 + sg_arg * BOGEY_FREE_RATE_ADJ, 0), 0.35)
    bf_bonus = bogey_free_prob * 4 * 3  # P(BF) x 4 rounds x 3 pts

    # Under-70 bonus: top players only
    under70_bonus = 0
    if probs.get("top_10", 0) > 0.15:
        under70_bonus = probs["top_10"] * 0.3 * 5  # Conditional probability x 5 pts

    total_adjustment = streak_bonus + bf_bonus + under70_bonus

    final_proj = base_pts + total_adjustment
    final_ceil = max(base_ceil + total_adjustment * 1.5, final_proj * 1.3)
    final_floor = max(base_floor, final_proj * 0.3)

    # Value scoring
    salary = p.get("dk_salary", 8000)
    value_per_k = final_proj / (salary / 1000) if salary > 0 else 0

    # Tier classification
    if salary >= 9500:
        tier = "STUD"
    elif salary >= 8000:
        tier = "MID"
    elif salary >= 6800:
        tier = "VALUE"
    else:
        tier = "PUNT"

    p["enhanced_projection"] = {
        "final_proj_dk_pts": round(final_proj, 1),
        "final_ceiling": round(final_ceil, 1),
        "final_floor": round(final_floor, 1),
        "dk_bonus_estimate": round(total_adjustment, 2),
        "streak_bonus": round(streak_bonus, 2),
        "bogey_free_bonus": round(bf_bonus, 2),
        "under70_bonus": round(under70_bonus, 2),
        "value_per_1k": round(value_per_k, 2),
    }
    p["tier"] = tier
    return p

def run():
    log.info("=" * 60)
    log.info("BOT 3: Projection Enhancer")
    log.info("=" * 60)

    with open(SHARED / "player_pool.json") as f:
        data = json.load(f)

    players = data.get("players", [])
    enhanced = [enhance_player(p) for p in players]

    # Weather wave adjustment
    am_advantage = 0.0
    event_name = "Unknown"
    try:
        predictions = get_predictions()
        event_info = find_current_event(predictions)
        event_name = event_info["event_name"]
        am_advantage, details, wave_conditions = get_wave_adjustment(event_name)
    except Exception as e:
        log.warning(f"Weather fetch failed: {e}")

    if abs(am_advantage) >= 0.5:
        wave_adj_half = am_advantage / 2.0
        n_am, n_pm = 0, 0
        for p in enhanced:
            wave = p.get("tee_time_info", {}).get("wave", "").upper()
            if wave in ("EARLY", "AM"):
                p["enhanced_projection"]["final_proj_dk_pts"] = round(
                    p["enhanced_projection"]["final_proj_dk_pts"] + wave_adj_half, 1)
                p["enhanced_projection"]["wave_adj"] = round(wave_adj_half, 1)
                n_am += 1
            elif wave in ("LATE", "PM"):
                p["enhanced_projection"]["final_proj_dk_pts"] = round(
                    p["enhanced_projection"]["final_proj_dk_pts"] - wave_adj_half, 1)
                p["enhanced_projection"]["wave_adj"] = round(-wave_adj_half, 1)
                n_pm += 1
            else:
                p["enhanced_projection"]["wave_adj"] = 0.0
        log.info(f"  Weather: {event_name} — AM advantage {am_advantage:+.1f} pts")
        log.info(f"  Applied ±{abs(wave_adj_half):.1f} pts to {n_am} AM / {n_pm} PM players")
    else:
        for p in enhanced:
            p["enhanced_projection"]["wave_adj"] = 0.0
        log.info(f"  Weather: neutral wave conditions (adj={am_advantage:+.1f})")

    # Sort by projected points
    enhanced.sort(key=lambda p: p.get("enhanced_projection", {}).get("final_proj_dk_pts", 0), reverse=True)

    output = {
        "metadata": {"player_count": len(enhanced), "source": "bot3_enhanced"},
        "players": enhanced,
    }

    with open(SHARED / "projections.json", "w") as f:
        json.dump(output, f, indent=2)
    Path(SHARED / "projections_ready.flag").touch()

    # Summary
    for tier in ["STUD", "MID", "VALUE", "PUNT"]:
        tier_players = [p for p in enhanced if p.get("tier") == tier]
        if tier_players:
            avg = sum(p["enhanced_projection"]["final_proj_dk_pts"] for p in tier_players) / len(tier_players)
            log.info(f"  {tier}: {len(tier_players)} players, avg proj {avg:.1f} pts")

    log.info(f"\n{len(enhanced)} projections saved to shared/projections.json")

if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""BOT 4: Ownership Projection & Game Theory Leverage."""
import json, sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import *

log = setup_logger("bot4", "bot4_ownership.log")

def calculate_leverage(proj_pts, ceiling, ownership_pct, contest_type="LARGE_GPP"):
    """Calculate game-theory leverage score."""
    if ownership_pct <= 0:
        ownership_pct = 0.5

    if contest_type == "CASH":
        return proj_pts  # Cash doesn't care about leverage

    # GPP leverage: ceiling relative to ownership
    # High ceiling + low ownership = maximum leverage
    leverage_raw = ceiling / max(ownership_pct, 0.5)

    # Log-based ownership penalty (from Haugh & Singal)
    own_penalty = math.log(5 / max(ownership_pct, 0.5)) / math.log(5) + 1

    return round(ceiling * own_penalty, 2)

def run():
    log.info("=" * 60)
    log.info("BOT 4: Ownership & Game Theory Engine")
    log.info("=" * 60)

    with open(SHARED / "projections.json") as f:
        data = json.load(f)

    players = data.get("players", [])

    for p in players:
        dg = p.get("dg_projection", {})
        enh = p.get("enhanced_projection", {})

        own_pct = dg.get("proj_ownership_pct", 5.0)
        proj_pts = enh.get("final_proj_dk_pts", 0)
        ceiling = enh.get("final_ceiling", 0)

        p["projected_ownership_pct"] = round(own_pct, 1)
        p["leverage_score"] = calculate_leverage(proj_pts, ceiling, own_pct)

        # Classify leverage
        if own_pct < 3 and ceiling > proj_pts * 1.3:
            p["leverage_tag"] = "CONTRARIAN_UPSIDE"
        elif own_pct > 20:
            p["leverage_tag"] = "CHALK"
        elif own_pct > 10 and ceiling < proj_pts * 1.15:
            p["leverage_tag"] = "CHALK_FADE"
        else:
            p["leverage_tag"] = "NEUTRAL"

    # Sort by leverage score
    players.sort(key=lambda p: p.get("leverage_score", 0), reverse=True)

    output = {"players": players, "metadata": {"source": "bot4_ownership"}}

    with open(SHARED / "ownership.json", "w") as f:
        json.dump(output, f, indent=2)
    Path(SHARED / "ownership_ready.flag").touch()

    # Print top leverage plays
    log.info(f"\nTop 10 Leverage Plays:")
    for p in players[:10]:
        log.info(f"  {p['name']:<25} Lev={p['leverage_score']:>7.1f}  Own={p['projected_ownership_pct']:>5.1f}%  Tag={p['leverage_tag']}")

    log.info(f"\n{len(players)} players with ownership + leverage saved")

if __name__ == "__main__":
    run()

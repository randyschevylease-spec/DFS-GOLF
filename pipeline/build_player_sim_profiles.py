"""
Build merged sim profiles from:
  Tier 1: Historical player profiles (player_profiles_2025.csv)
  Tier 2: Synthetic profiles from DataGolf pre-tournament API

Output: data/cache/sim_profiles_current.csv
"""

import csv
import json
import os
import urllib.request

API_KEY = "85acc5b95e8b7ead122cab0c8020"
PRE_TOURNAMENT_URL = (
    "https://feeds.datagolf.com/preds/pre-tournament"
    "?tour=pga&add_position=1,5,10,20&file_format=json"
    f"&key={API_KEY}"
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_PATH = os.path.join(PROJECT_ROOT, "data", "cache", "player_profiles_2025.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "cache", "sim_profiles_current.csv")

# DK finish pts schedule
# Weighted averages for position buckets:
#   top_5 (positions 2-5): (20+18+16+14)/4 = 17.0, but top_5 prob includes win
#   so incremental top_5 contribution = top_5_prob - win_prob
#   Similarly for each tier
FINISH_PTS = {
    "win": 30,
    "top_5": 17.0,    # avg of 2nd-5th: (20+18+16+14)/4
    "top_10": 9.6,    # avg of 6th-10th: (12+10+9+8+7)/5 (shifted to be exact for 6-10)
    "top_20": 5.5,    # avg of 11th-20th: (6+5+4.5+4+3.5+3+2.5+2+1.5+1)/10 ≈ rough
}

FIELDS = [
    "player_name", "dg_id", "tier", "appearances", "made_cut", "missed_cut",
    "made_cut_rate", "avg_pts_made", "avg_pts_missed", "gap", "avg_salary",
    "std_pts_made", "std_pts_missed",
    "min_pts_made", "max_pts_made", "min_pts_missed", "max_pts_missed",
    "dg_win", "dg_make_cut", "dg_top_5", "dg_top_10", "dg_top_20",
]


def load_tier1():
    """Load historical player profiles."""
    profiles = {}
    with open(PROFILES_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            profiles[row["player_name"]] = row
    return profiles


def fetch_dg_pretournament():
    """Pull live pre-tournament predictions from DataGolf."""
    print(f"Fetching DataGolf pre-tournament data...")
    with urllib.request.urlopen(PRE_TOURNAMENT_URL) as resp:
        data = json.loads(resp.read())

    event = data.get("event_name", "unknown")
    updated = data.get("last_updated", "unknown")
    print(f"  Event: {event}")
    print(f"  Updated: {updated}")

    # Use baseline_history_fit (course-history adjusted)
    model = "baseline_history_fit"
    if model not in data:
        print(f"  Warning: {model} not available, falling back to baseline")
        model = "baseline"

    players = data[model]
    print(f"  Model: {model} ({len(players)} players)")
    return players, event


def compute_tier1_averages(tier1):
    """Compute tour-average std and pts for Tier 2 synthetic profiles."""
    stds = []
    missed_stds = []
    avg_made_pts = []
    avg_missed_pts = []

    for row in tier1.values():
        if row.get("std_pts_made"):
            stds.append(float(row["std_pts_made"]))
        if row.get("std_pts_missed"):
            missed_stds.append(float(row["std_pts_missed"]))
        if row.get("avg_pts_made"):
            avg_made_pts.append(float(row["avg_pts_made"]))
        if row.get("avg_pts_missed"):
            avg_missed_pts.append(float(row["avg_pts_missed"]))

    return {
        "mean_std_made": sum(stds) / len(stds),
        "mean_std_missed": sum(missed_stds) / len(missed_stds),
        "mean_pts_made": sum(avg_made_pts) / len(avg_made_pts),
        "mean_pts_missed": sum(avg_missed_pts) / len(avg_missed_pts),
    }


def estimate_synthetic_pts(dg_player, tour_avgs):
    """
    Estimate avg made-cut pts from DG finish probabilities.

    Expected finish pts using incremental probability bands:
      win_prob × 30
      (top5 - win) × 17.0    (avg of positions 2-5)
      (top10 - top5) × 9.6   (avg of positions 6-10)
      (top20 - top10) × 5.5  (avg of positions 11-20)

    Then add tour-average hole_score_pts scaled by skill tier.
    """
    win = dg_player.get("win", 0)
    top5 = dg_player.get("top_5", 0)
    top10 = dg_player.get("top_10", 0)
    top20 = dg_player.get("top_20", 0)
    make_cut = dg_player.get("make_cut", 0)

    # Incremental finish pts expectation (conditional on making cut)
    exp_finish = (
        win * 30
        + max(0, top5 - win) * 17.0
        + max(0, top10 - top5) * 9.6
        + max(0, top20 - top10) * 5.5
    )

    # Scale to conditional on making cut
    if make_cut > 0:
        exp_finish_given_cut = exp_finish / make_cut
    else:
        exp_finish_given_cut = 0

    # From our Q4 analysis: for made-cut players, hole_score_pts ≈ 88% of total
    # finish_pts ≈ 6-10% for top 20, ~0% for 50+
    # Tour avg made-cut pts ≈ 78, with finish_pts ≈ 5 on average
    # So hole_score component ≈ tour_avg_made - avg_finish
    base_hole_pts = tour_avgs["mean_pts_made"] - 5.0  # rough tour-avg finish component

    # Scale hole score by skill: better players score more
    # Use make_cut rate as skill proxy (range ~0.4 to 0.9)
    # Tour avg make_cut ≈ 0.55
    skill_scale = make_cut / 0.55 if make_cut > 0 else 0.8
    skill_scale = min(skill_scale, 1.5)  # cap scaling

    scaled_hole_pts = base_hole_pts * skill_scale
    avg_pts_made = scaled_hole_pts + exp_finish_given_cut

    # Missed cut pts: relatively stable across players
    avg_pts_missed = tour_avgs["mean_pts_missed"]

    return round(avg_pts_made, 2), round(avg_pts_missed, 2)


def build_tier2_profile(dg_player, tour_avgs):
    """Build a synthetic sim profile for a player not in Tier 1."""
    avg_made, avg_missed = estimate_synthetic_pts(dg_player, tour_avgs)

    make_cut = dg_player.get("make_cut", 0.5)

    # Tier 2 std override: elite players have tighter distributions
    if avg_made > 100:
        std_made = 15.0
    elif avg_made > 90:
        std_made = 17.0
    else:
        std_made = round(tour_avgs["mean_std_made"], 2)

    return {
        "player_name": dg_player["player_name"],
        "dg_id": dg_player["dg_id"],
        "tier": 2,
        "appearances": 0,
        "made_cut": "",
        "missed_cut": "",
        "made_cut_rate": round(make_cut, 4),
        "avg_pts_made": avg_made,
        "avg_pts_missed": avg_missed,
        "gap": round(avg_made - avg_missed, 2),
        "avg_salary": "",
        "std_pts_made": std_made,
        "std_pts_missed": round(tour_avgs["mean_std_missed"], 2),
        "min_pts_made": round(avg_made - 2 * std_made, 1),
        "max_pts_made": round(avg_made + 2 * std_made, 1),
        "min_pts_missed": round(avg_missed - 2 * tour_avgs["mean_std_missed"], 1),
        "max_pts_missed": round(avg_missed + 2 * tour_avgs["mean_std_missed"], 1),
        "dg_win": dg_player.get("win", ""),
        "dg_make_cut": make_cut,
        "dg_top_5": dg_player.get("top_5", ""),
        "dg_top_10": dg_player.get("top_10", ""),
        "dg_top_20": dg_player.get("top_20", ""),
    }


def main():
    # Load Tier 1
    tier1 = load_tier1()
    print(f"Tier 1: {len(tier1)} players from historical profiles")

    # Compute tour averages for Tier 2 synthesis
    tour_avgs = compute_tier1_averages(tier1)
    print(f"  Tour avg std_made: {tour_avgs['mean_std_made']:.2f}")
    print(f"  Tour avg std_missed: {tour_avgs['mean_std_missed']:.2f}")
    print(f"  Tour avg pts_made: {tour_avgs['mean_pts_made']:.1f}")
    print(f"  Tour avg pts_missed: {tour_avgs['mean_pts_missed']:.1f}")

    # Fetch DG pre-tournament
    dg_players, event = fetch_dg_pretournament()

    # Build dg_id lookup for Tier 1
    tier1_by_dg_id = {}
    # We don't have dg_id in Tier 1 profiles, so match by name
    tier1_names = set(tier1.keys())

    # Merge
    merged = []
    tier1_matched = 0
    tier2_created = 0

    for dg_p in dg_players:
        name = dg_p["player_name"]
        dg_id = dg_p["dg_id"]

        if name in tier1:
            # Tier 1: use historical profile, enrich with DG probs
            row = dict(tier1[name])
            row["dg_id"] = dg_id
            row["tier"] = 1
            row["dg_win"] = dg_p.get("win", "")
            row["dg_make_cut"] = dg_p.get("make_cut", "")
            row["dg_top_5"] = dg_p.get("top_5", "")
            row["dg_top_10"] = dg_p.get("top_10", "")
            row["dg_top_20"] = dg_p.get("top_20", "")
            merged.append(row)
            tier1_matched += 1
        else:
            # Tier 2: synthetic profile
            row = build_tier2_profile(dg_p, tour_avgs)
            merged.append(row)
            tier2_created += 1

    # Global caps
    MAX_PTS_MADE = 145
    MIN_PTS_MADE = 35
    BLEND_STD_THRESHOLD = 30
    BLEND_MAX_THRESHOLD = 145

    flagged = []
    for row in merged:
        std_made = float(row.get("std_pts_made") or 0)
        max_made = float(row.get("max_pts_made") or 0)
        name = row["player_name"]
        tier = row["tier"]

        # Flag Tier 1 players with unreliable small-sample stats
        if str(tier) == "1" and (max_made > BLEND_MAX_THRESHOLD or std_made > BLEND_STD_THRESHOLD):
            reason = []
            if max_made > BLEND_MAX_THRESHOLD:
                reason.append(f"max={max_made:.1f}")
            if std_made > BLEND_STD_THRESHOLD:
                reason.append(f"std={std_made:.1f}")

            # Blend std with tour average: 50/50
            old_std = std_made
            blended_std = round((std_made + tour_avgs["mean_std_made"]) / 2, 2)
            row["std_pts_made"] = blended_std

            # Recalculate min/max from blended std
            avg_made = float(row.get("avg_pts_made") or 0)
            row["min_pts_made"] = round(avg_made - 2 * blended_std, 1)
            row["max_pts_made"] = round(avg_made + 2 * blended_std, 1)

            flagged.append({
                "name": name,
                "reason": ", ".join(reason),
                "old_std": old_std,
                "new_std": blended_std,
                "appearances": row.get("appearances", ""),
            })

        # Apply global caps to all players
        if row.get("max_pts_made") != "":
            row["max_pts_made"] = min(float(row["max_pts_made"]), MAX_PTS_MADE)
        if row.get("min_pts_made") != "":
            row["min_pts_made"] = max(float(row["min_pts_made"]), MIN_PTS_MADE)

    # Print flagged players
    if flagged:
        print(f"\nFlagged Tier 1 players (blended with tour avg std):")
        print(f"  {'Player':<30} {'Apps':>4} {'Reason':<20} {'OldStd':>7} {'NewStd':>7}")
        print(f"  {'-'*72}")
        for f in flagged:
            print(f"  {f['name']:<30} {f['appearances']:>4} {f['reason']:<20} {f['old_std']:>7.1f} {f['new_std']:>7.1f}")

    # Sort by DG win probability descending
    merged.sort(key=lambda r: float(r.get("dg_win") or 0), reverse=True)

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)

    # Summary
    field_size = len(dg_players)
    total = len(merged)
    print(f"\n{'='*50}")
    print(f"Event: {event}")
    print(f"Field size (DG):     {field_size}")
    print(f"Tier 1 (historical): {tier1_matched} ({tier1_matched/field_size*100:.0f}%)")
    print(f"Tier 2 (synthetic):  {tier2_created} ({tier2_created/field_size*100:.0f}%)")
    print(f"Total profiles:      {total}")
    print(f"Coverage:            {total}/{field_size} ({total/field_size*100:.0f}%)")
    print(f"\nSaved to: {OUTPUT_PATH}")

    # Show top 10
    print(f"\nTop 10 by win probability:")
    print(f"{'Player':<30} {'Tier':>4} {'Win%':>6} {'Cut%':>6} {'AvgMade':>8} {'Std':>6}")
    print("-" * 66)
    for r in merged[:10]:
        win = float(r.get("dg_win") or 0) * 100
        cut = float(r.get("dg_make_cut") or r.get("made_cut_rate", 0)) * 100
        avg = float(r.get("avg_pts_made") or 0)
        std = float(r.get("std_pts_made") or 0)
        print(f"{r['player_name']:<30} T{r['tier']:>3} {win:>5.1f}% {cut:>5.1f}% {avg:>8.1f} {std:>6.1f}")


if __name__ == "__main__":
    main()

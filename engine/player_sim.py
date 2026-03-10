"""
player_sim.py — Hole-level Monte Carlo simulation engine for golf DFS.

Architecture:
  Per player, per iteration, per round:
    1. Draw sg_ott, sg_app, sg_arg, sg_putt from player's distribution
    2. For each of 18 holes:
       - Load base outcome probabilities from course hole profiles
       - Adjust birdie/eagle/bogey rates by player's drawn SG vs tour avg
       - Sample hole outcome from adjusted distribution
    3. Sum 18 hole outcomes -> round DK pts + bonuses

  After round 2: apply cut (top 65 + ties by 2-round stroke total)
  After round 4: rank field, assign finish_pts
"""

import csv
import math
import os
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SG_STDS_PATH = os.path.join(PROJECT_ROOT, "data", "cache", "player_sg_stds_2024.csv")
HOLE_PROFILES_PATH = os.path.join(PROJECT_ROOT, "data", "cache", "tpc_sawgrass_hole_profiles.csv")

# Tour median stds (fallback for unknown players)
TOUR_MEDIANS = {
    "sg_ott_mean": 0.0, "sg_ott_std": 1.01,
    "sg_app_mean": 0.0, "sg_app_std": 1.56,
    "sg_arg_mean": 0.0, "sg_arg_std": 1.04,
    "sg_putt_mean": 0.0, "sg_putt_std": 1.66,
}

# DK scoring per outcome
DK_PTS = {"eagle": 8, "birdie": 3, "par": 0.5, "bogey": -0.5, "double": -1}

# DK bonuses
BOGEY_FREE_BONUS = 3
SUB_70_BONUS = 5  # round score < 70 on par 72

# DK finish points schedule
FINISH_PTS = {
    1: 30, 2: 20, 3: 18, 4: 16, 5: 14,
    6: 12, 7: 10, 8: 9, 9: 8, 10: 7,
    11: 6, 12: 5.5, 13: 5, 14: 4.5, 15: 4,
    16: 3.5, 17: 3, 18: 2.5, 19: 2, 20: 1.5,
    21: 1, 22: 1, 23: 1, 24: 1, 25: 1,
    26: 0.5, 27: 0.5, 28: 0.5, 29: 0.5, 30: 0.5,
}

SG_CATS = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]

# Outcome indices for vectorized hole sim
EAGLE, BIRDIE, PAR, BOGEY, DOUBLE = 0, 1, 2, 3, 4


def load_hole_profiles():
    """Load TPC Sawgrass 22-year hole profiles."""
    holes = []
    with open(HOLE_PROFILES_PATH) as f:
        for row in csv.DictReader(f):
            holes.append({
                "hole_num": int(row["hole_num"]),
                "hole_par": int(row["hole_par"]),
                "eagle_rate": float(row["eagle_rate"]),
                "birdie_rate": float(row["birdie_rate"]),
                "par_rate": float(row["par_rate"]),
                "bogey_rate": float(row["bogey_rate"]),
                "double_rate": float(row["double_rate"]),
            })
    return holes


def load_sg_profiles():
    """Load player SG profiles. Returns dict keyed by dg_id and by name."""
    by_id = {}
    by_name = {}
    if not os.path.exists(SG_STDS_PATH):
        return by_id, by_name
    with open(SG_STDS_PATH) as f:
        for row in csv.DictReader(f):
            prof = {
                "player_name": row["player_name"],
                "dg_id": int(row["dg_id"]),
                "num_rounds": int(row["num_rounds"]),
            }
            for cat in SG_CATS:
                prof[f"{cat}_mean"] = float(row[f"{cat}_mean"])
                prof[f"{cat}_std"] = float(row[f"{cat}_std"])
            by_id[prof["dg_id"]] = prof
            by_name[prof["player_name"]] = prof
    return by_id, by_name


def get_profile(player, by_id, by_name):
    """Resolve player to SG profile. Player can be dg_id (int) or name (str)."""
    if isinstance(player, int) and player in by_id:
        return by_id[player]
    if isinstance(player, str) and player in by_name:
        return by_name[player]
    return dict(TOUR_MEDIANS)


def draw_sg(profile, rng):
    """Draw one round of SG components."""
    sg = {}
    for cat in SG_CATS:
        mu = profile.get(f"{cat}_mean", 0.0)
        sigma = profile.get(f"{cat}_std", 1.0)
        sg[cat] = rng.gauss(mu, sigma)
    sg["sg_total"] = sum(sg[cat] for cat in SG_CATS)
    return sg


def adjust_hole_probs(hole, sg, course_offset=0.0):
    """
    Adjust base hole probabilities based on player's drawn SG and course difficulty.

    Args:
        hole: base hole profile from course data
        sg: dict of drawn SG components for this round
        course_offset: per-hole difficulty shift derived from course_avg_score.
            Negative = easier (more birdies), positive = harder (more bogeys).
            Computed as (course_avg_score - 72.0) / 18 per hole.

    Adjustment logic:
      - sg_app & sg_putt boost/reduce birdie rate on par 4s and 3s
      - sg_ott & sg_app boost eagle rate on par 5s
      - Worse SG increases bogey/double rates
      - course_offset shifts birdie down and bogey up (or vice versa)
      - All rates re-normalized to sum to 1.0

    The adjustment is proportional: each +1.0 sg_total shifts birdie
    probability by ~4% (from empirical birdie slope of 0.397/18 holes).
    """
    sg_total = sg["sg_total"]
    par = hole["hole_par"]

    # Base rates
    eagle = hole["eagle_rate"]
    birdie = hole["birdie_rate"]
    par_r = hole["par_rate"]
    bogey = hole["bogey_rate"]
    double = hole["double_rate"]

    # Course difficulty offset: shifts scoring distribution
    # Each +0.1 strokes over par per hole ≈ -2.2% birdie, +1.9% bogey, +0.55% double
    birdie_course = -0.022 * (course_offset / 0.0556) if course_offset != 0 else 0
    bogey_course = 0.019 * (course_offset / 0.0556) if course_offset != 0 else 0
    double_course = 0.0055 * (course_offset / 0.0556) if course_offset != 0 else 0
    # Simplify: 0.0556 = 1/18, so course_offset/0.0556 = course_offset * 18
    # But course_offset is already (avg_score - 72) / 18
    # So course_offset / 0.0556 = (avg_score - 72), the total stroke diff
    # Net: use the same per-stroke slopes as SG adjustment
    birdie_course = -0.022 * course_offset * 18
    bogey_course = 0.019 * course_offset * 18
    double_course = 0.0055 * course_offset * 18

    # Per-hole birdie shift: empirical ~0.022 per +1.0 sg_total per hole
    # (0.397 birdies/round / 18 holes)
    birdie_shift = 0.022 * sg_total + birdie_course

    # Eagle shift on par 5s: driven by sg_ott + sg_app
    eagle_shift = 0.0
    if par == 5:
        sg_power = sg["sg_ott"] + sg["sg_app"]
        eagle_shift = 0.008 * sg_power  # ~0.8% per +1.0 combined

    # Bogey shift: negative SG increases bogey risk
    # Empirical: -0.019 bogeys/hole per +1.0 sg_total
    bogey_shift = -0.019 * sg_total + bogey_course

    # Double shift: -0.0055 per +1.0 sg_total per hole
    double_shift = -0.0055 * sg_total + double_course

    # Apply shifts
    eagle = max(0, eagle + eagle_shift)
    birdie = max(0.001, birdie + birdie_shift)
    bogey = max(0.001, bogey + bogey_shift)
    double = max(0, double + double_shift)

    # Par absorbs the remainder
    total = eagle + birdie + bogey + double
    par_r = max(0.01, 1.0 - total)

    # Normalize
    total = eagle + birdie + par_r + bogey + double
    return (
        eagle / total,
        birdie / total,
        par_r / total,
        bogey / total,
        double / total,
    )


def sample_hole(probs, rng):
    """Sample one hole outcome from probability distribution."""
    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            return i
    return PAR  # fallback


def sim_round(holes, sg, rng, course_offset=0.0):
    """
    Simulate one 18-hole round.

    Args:
        holes: list of hole profile dicts
        sg: drawn SG components for this round
        rng: random generator
        course_offset: per-hole difficulty shift (default 0.0)

    Returns:
        scores: list of 18 outcome indices (EAGLE..DOUBLE)
        dk_pts: total DK hole-score points for the round
        strokes: total strokes for the round
    """
    scores = []
    dk_pts = 0.0
    strokes = 0
    bogey_free = True

    dk_vals = [DK_PTS["eagle"], DK_PTS["birdie"], DK_PTS["par"],
               DK_PTS["bogey"], DK_PTS["double"]]
    stroke_vals = [-2, -1, 0, 1, 2]  # relative to par

    for hole in holes:
        probs = adjust_hole_probs(hole, sg, course_offset)
        outcome = sample_hole(probs, rng)
        scores.append(outcome)
        dk_pts += dk_vals[outcome]
        strokes += hole["hole_par"] + stroke_vals[outcome]

        if outcome >= BOGEY:
            bogey_free = False

    # Bonuses
    if bogey_free:
        dk_pts += BOGEY_FREE_BONUS

    if strokes < 70:  # sub-70 on par 72
        dk_pts += SUB_70_BONUS

    return scores, dk_pts, strokes


def get_finish_pts(position):
    """Lookup DK finish points."""
    if position in FINISH_PTS:
        return FINISH_PTS[position]
    if position <= 40:
        return 0.5
    return 0


def simulate_tournament(players, n_iterations=10000, cut_size=65, seed=42,
                        dg_cut_probs=None, course_avg_score=72.0):
    """
    Full tournament Monte Carlo simulation.

    Args:
        players: list of player identifiers (dg_id or name)
        n_iterations: number of MC iterations
        cut_size: number of players making the cut (top N + ties)
        seed: random seed
        dg_cut_probs: optional dict of player -> make_cut probability
            from DataGolf baseline_history_fit model. If provided, cut
            is determined by independent probability draw per player
            instead of stroke-based ranking.
        course_avg_score: course average score (default 72.0 = neutral).
            Values < 72 make the course easier (more birdies).
            Values > 72 make it harder (more bogeys).
            TPC Sawgrass historical avg: ~72.38.

    Returns:
        dict: player -> list of total DK pts per iteration
    """
    rng = random.Random(seed)
    by_id, by_name = load_sg_profiles()
    holes = load_hole_profiles()

    # Compute per-hole difficulty offset from course_avg_score
    # course_avg_score=72.0 -> offset=0.0 (neutral, base probs unchanged)
    # course_avg_score=73.0 -> offset=+0.0556/hole (harder)
    # course_avg_score=71.0 -> offset=-0.0556/hole (easier)
    course_offset = (course_avg_score - 72.0) / 18.0

    # Resolve all player profiles
    profiles = {}
    for p in players:
        profiles[p] = get_profile(p, by_id, by_name)

    results = {p: [] for p in players}
    field_size = len(players)

    for _iter in range(n_iterations):
        # --- Draw SG for all 4 rounds upfront per player ---
        player_sgs = {}
        for p in players:
            player_sgs[p] = [draw_sg(profiles[p], rng) for _ in range(4)]

        # --- Rounds 1 & 2 ---
        r12_data = {}
        for p in players:
            rd1_scores, rd1_pts, rd1_strokes = sim_round(holes, player_sgs[p][0], rng, course_offset)
            rd2_scores, rd2_pts, rd2_strokes = sim_round(holes, player_sgs[p][1], rng, course_offset)
            r12_data[p] = {
                "dk_pts": rd1_pts + rd2_pts,
                "strokes": rd1_strokes + rd2_strokes,
                "rd_scores": [rd1_scores, rd2_scores],
            }

        # --- Apply cut ---
        if dg_cut_probs is not None:
            # DG probability-based cut: each player draws independently
            made_cut = set()
            for p in players:
                cut_prob = dg_cut_probs.get(p, 0.5)  # default 50% if missing
                if rng.random() < cut_prob:
                    made_cut.add(p)
        else:
            # Stroke-based cut: top N by 2-round score
            sorted_field = sorted(r12_data.items(), key=lambda x: x[1]["strokes"])
            if field_size > cut_size:
                cut_stroke = sorted_field[cut_size - 1][1]["strokes"]
                made_cut = {p for p, d in r12_data.items() if d["strokes"] <= cut_stroke}
            else:
                made_cut = set(players)

        # --- Rounds 3 & 4 for made cut ---
        final_data = {}
        for p in players:
            if p not in made_cut:
                final_data[p] = {
                    "dk_pts": r12_data[p]["dk_pts"],
                    "strokes": r12_data[p]["strokes"],
                    "made_cut": False,
                }
                continue

            rd3_scores, rd3_pts, rd3_strokes = sim_round(holes, player_sgs[p][2], rng, course_offset)
            rd4_scores, rd4_pts, rd4_strokes = sim_round(holes, player_sgs[p][3], rng, course_offset)

            final_data[p] = {
                "dk_pts": r12_data[p]["dk_pts"] + rd3_pts + rd4_pts,
                "strokes": r12_data[p]["strokes"] + rd3_strokes + rd4_strokes,
                "made_cut": True,
            }

        # --- Rank and assign finish pts ---
        cut_players = [(p, d) for p, d in final_data.items() if d["made_cut"]]
        cut_players.sort(key=lambda x: x[1]["strokes"])

        pos = 1
        i = 0
        while i < len(cut_players):
            j = i
            while j < len(cut_players) and cut_players[j][1]["strokes"] == cut_players[i][1]["strokes"]:
                j += 1
            # All tied players get same position's finish pts
            fp = get_finish_pts(pos)
            for k in range(i, j):
                final_data[cut_players[k][0]]["dk_pts"] += fp
            pos = j + 1
            i = j

        # --- Record ---
        for p in players:
            results[p].append(final_data[p]["dk_pts"])

    return results


def build_default_field(by_id, target_ids):
    """Build a realistic ~148 player field from SG profiles."""
    # Use top 148 players by num_rounds (most active)
    all_players = sorted(by_id.values(), key=lambda p: -p["num_rounds"])
    field = [p["dg_id"] for p in all_players[:148]]
    # Ensure targets are included
    for tid in target_ids:
        if tid not in field:
            field.append(tid)
    return field


def main():
    """Test: Scheffler vs Burns, 10k iterations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    SCHEFFLER = 18417
    BURNS = 19483
    N_ITER = 10000

    by_id, by_name = load_sg_profiles()
    field = build_default_field(by_id, [SCHEFFLER, BURNS])

    print(f"Simulating {N_ITER} tournament iterations...")
    print(f"  Field size: {len(field)} players")
    print(f"  Scheffler (dg_id={SCHEFFLER})")
    print(f"  Burns (dg_id={BURNS})")

    # Show their SG profiles
    for name, did in [("Scheffler", SCHEFFLER), ("Burns", BURNS)]:
        p = by_id[did]
        print(f"\n  {name} SG profile ({p['num_rounds']} rounds):")
        for cat in SG_CATS:
            print(f"    {cat}: mean={p[f'{cat}_mean']:+.3f}  std={p[f'{cat}_std']:.3f}")

    results = simulate_tournament(field, n_iterations=N_ITER, seed=42)

    scheffler_pts = results[SCHEFFLER]
    burns_pts = results[BURNS]

    print(f"\n{'='*60}")
    print(f"Results ({N_ITER} iterations):")
    for name, pts in [("Scheffler", scheffler_pts), ("Burns", burns_pts)]:
        arr = np.array(pts)
        print(f"\n  {name}:")
        print(f"    Mean:          {arr.mean():.1f}")
        print(f"    Std:           {arr.std():.1f}")
        print(f"    Median:        {np.median(arr):.1f}")
        print(f"    P10 / P90:     {np.percentile(arr, 10):.1f} / {np.percentile(arr, 90):.1f}")
        print(f"    Min / Max:     {arr.min():.1f} / {arr.max():.1f}")
        print(f"    Made cut rate: {(arr > 40).sum() / len(arr):.1%}")
        print(f"    100+ pts rate: {(arr >= 100).sum() / len(arr):.1%}")
        print(f"    120+ pts rate: {(arr >= 120).sum() / len(arr):.1%}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    bins = np.arange(-20, 200, 5)

    for ax, (name, pts, color) in zip(axes, [
        ("Scheffler", scheffler_pts, "#2ecc71"),
        ("Burns", burns_pts, "#3498db"),
    ]):
        arr = np.array(pts)
        ax.hist(arr, bins=bins, alpha=0.8, color=color, edgecolor="white")
        ax.axvline(arr.mean(), color="red", linestyle="--", lw=2,
                   label=f"Mean: {arr.mean():.1f}")
        ax.axvline(np.percentile(arr, 90), color="orange", linestyle=":",
                   lw=2, label=f"P90: {np.percentile(arr, 90):.1f}")
        cut_rate = (arr > 40).sum() / len(arr)
        rate100 = (arr >= 100).sum() / len(arr)
        ax.set_title(
            f"{name}\nstd={arr.std():.1f} | cut={cut_rate:.0%} | 100+={rate100:.0%}",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Total DK Points", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Hole-Level Monte Carlo Sim — {N_ITER} Iterations | "
        f"TPC Sawgrass | Field: {len(field)} players",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    plot_path = os.path.join(PROJECT_ROOT, "reports", "sim_validation_scheffler_burns.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")

    # Overlay
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    s = np.array(scheffler_pts)
    b = np.array(burns_pts)
    ax2.hist(s, bins=bins, alpha=0.6, color="#2ecc71", label="Scheffler", edgecolor="white")
    ax2.hist(b, bins=bins, alpha=0.6, color="#3498db", label="Burns", edgecolor="white")
    ax2.axvline(s.mean(), color="#27ae60", linestyle="--", lw=2)
    ax2.axvline(b.mean(), color="#2980b9", linestyle="--", lw=2)
    ax2.set_xlabel("Total DK Points", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Scheffler vs Burns — Hole-Level Sim Distribution", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    overlay_path = os.path.join(PROJECT_ROOT, "reports", "sim_validation_overlay.png")
    plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
    print(f"Overlay saved to {overlay_path}")


if __name__ == "__main__":
    main()

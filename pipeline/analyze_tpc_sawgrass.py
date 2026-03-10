"""Analyze TPC Sawgrass hole-level data (2003-2025)."""

import csv
import os
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "data", "cache", "tpc_sawgrass_all_years.csv")
HOLE_PROFILES = os.path.join(PROJECT_ROOT, "data", "cache", "tpc_sawgrass_hole_profiles.csv")
PLOT_PATH = os.path.join(PROJECT_ROOT, "reports", "tpc_sawgrass_hole_profiles.png")

DK = {"eagle": 6, "birdie": 3, "par": 0, "bogey": -0.5, "double": -1}


def safe_int(v, default=0):
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def main():
    with open(DATA) as f:
        rows = list(csv.DictReader(f))

    # Step 1: Convert counts to rates
    for r in rows:
        eagles = safe_int(r["eagles_or_better"])
        birdies = safe_int(r["birdies"])
        pars = safe_int(r["pars"])
        bogeys = safe_int(r["bogeys"])
        doubles = safe_int(r["doubles_or_worse"])
        total = eagles + birdies + pars + bogeys + doubles
        r["field_size"] = total
        r["eagle_rate"] = eagles / total if total else 0
        r["birdie_rate"] = birdies / total if total else 0
        r["par_rate"] = pars / total if total else 0
        r["bogey_rate"] = bogeys / total if total else 0
        r["double_rate"] = doubles / total if total else 0

    # Step 2: 22-year averages per hole
    hole_data = defaultdict(list)
    for r in rows:
        hole_data[int(r["hole_num"])].append(r)

    hole_profiles = []
    for h in sorted(hole_data.keys()):
        rds = hole_data[h]
        n = len(rds)
        yds_vals = [safe_int(r["hole_yardage"]) for r in rds if r["hole_yardage"] != "null"]
        prof = {
            "hole_num": h,
            "hole_par": safe_int(rds[0]["hole_par"]),
            "avg_yardage": round(sum(yds_vals) / len(yds_vals), 0) if yds_vals else 0,
            "avg_scoring": round(sum(safe_float(r["scoring_avg"]) for r in rds) / n, 4),
            "avg_rel_scoring": round(sum(safe_float(r["rel_scoring_avg"]) for r in rds) / n, 4),
            "eagle_rate": round(sum(r["eagle_rate"] for r in rds) / n, 5),
            "birdie_rate": round(sum(r["birdie_rate"] for r in rds) / n, 5),
            "par_rate": round(sum(r["par_rate"] for r in rds) / n, 5),
            "bogey_rate": round(sum(r["bogey_rate"] for r in rds) / n, 5),
            "double_rate": round(sum(r["double_rate"] for r in rds) / n, 5),
        }
        hole_profiles.append(prof)

    # Step 3: Expected DK pts per hole
    for prof in hole_profiles:
        prof["exp_dk_pts"] = round(
            prof["eagle_rate"] * DK["eagle"]
            + prof["birdie_rate"] * DK["birdie"]
            + prof["par_rate"] * DK["par"]
            + prof["bogey_rate"] * DK["bogey"]
            + prof["double_rate"] * DK["double"],
            4,
        )

    # Step 4: Year-to-year variance
    for i, h in enumerate(sorted(hole_data.keys())):
        rds = hole_data[h]
        year_avgs = defaultdict(list)
        for r in rds:
            year_avgs[r["tournament_year"]].append(safe_float(r["scoring_avg"]))
        yearly_means = [sum(v) / len(v) for v in year_avgs.values()]
        if len(yearly_means) > 1:
            mu = sum(yearly_means) / len(yearly_means)
            std = math.sqrt(sum((x - mu) ** 2 for x in yearly_means) / (len(yearly_means) - 1))
        else:
            std = 0
        hole_profiles[i]["scoring_std_across_years"] = round(std, 4)

    # Save hole profiles
    prof_fields = list(hole_profiles[0].keys())
    with open(HOLE_PROFILES, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=prof_fields)
        writer.writeheader()
        writer.writerows(hole_profiles)
    print(f"Saved hole profiles to {HOLE_PROFILES}")

    # --- Print ranked holes ---
    print(f"\n{'='*90}")
    print("HOLES RANKED BY EXPECTED DK PTS")
    print(f"{'='*90}")
    ranked = sorted(hole_profiles, key=lambda p: -p["exp_dk_pts"])
    print(
        f"{'Hole':>4} {'Par':>3} {'Yds':>5} {'ScAvg':>6} {'Rel':>7} "
        f"{'Eagle%':>7} {'Bird%':>7} {'Par%':>6} {'Bog%':>6} {'Dbl%':>6} "
        f"{'ExpDK':>7} {'StdYr':>6}"
    )
    print("-" * 90)
    for p in ranked:
        print(
            f"{p['hole_num']:>4} {p['hole_par']:>3} {p['avg_yardage']:>5.0f} "
            f"{p['avg_scoring']:>6.3f} {p['avg_rel_scoring']:>+7.3f} "
            f"{p['eagle_rate']*100:>6.2f}% {p['birdie_rate']*100:>6.2f}% "
            f"{p['par_rate']*100:>5.1f}% {p['bogey_rate']*100:>5.1f}% "
            f"{p['double_rate']*100:>5.1f}% "
            f"{p['exp_dk_pts']:>+7.4f} {p['scoring_std_across_years']:>6.3f}"
        )

    # --- Most setup-dependent ---
    print(f"\n{'='*60}")
    print("MOST SETUP-DEPENDENT HOLES")
    print(f"{'='*60}")
    by_var = sorted(hole_profiles, key=lambda p: -p["scoring_std_across_years"])
    for p in by_var[:5]:
        print(
            f"  Hole {p['hole_num']:>2} (par {p['hole_par']}, {p['avg_yardage']:.0f}y): "
            f"std={p['scoring_std_across_years']:.4f}  avg_rel={p['avg_rel_scoring']:+.3f}"
        )
    print("\nMost stable:")
    for p in sorted(by_var[-3:], key=lambda p: p["scoring_std_across_years"]):
        print(
            f"  Hole {p['hole_num']:>2} (par {p['hole_par']}, {p['avg_yardage']:.0f}y): "
            f"std={p['scoring_std_across_years']:.4f}  avg_rel={p['avg_rel_scoring']:+.3f}"
        )

    # --- Hole 17 deep dive ---
    print(f"\n{'='*60}")
    print("HOLE 17 (ISLAND GREEN) -- SCORING BY YEAR")
    print(f"{'='*60}")
    h17 = [r for r in rows if int(r["hole_num"]) == 17]
    year_data = defaultdict(list)
    for r in h17:
        year_data[r["tournament_year"]].append(r)

    years_sorted = sorted(year_data.keys(), key=int)
    h17_yearly = []
    for y in years_sorted:
        rds = year_data[y]
        avg_score = sum(safe_float(r["scoring_avg"]) for r in rds) / len(rds)
        avg_bird = sum(r["birdie_rate"] for r in rds) / len(rds)
        avg_bog = sum(r["bogey_rate"] for r in rds) / len(rds)
        avg_dbl = sum(r["double_rate"] for r in rds) / len(rds)
        yds = safe_int(rds[0]["hole_yardage"])
        h17_yearly.append((y, avg_score, avg_bird, avg_bog, avg_dbl, yds))
        print(
            f"  {y}: avg={avg_score:.3f}  birdie={avg_bird*100:.1f}%  "
            f"bogey={avg_bog*100:.1f}%  double+={avg_dbl*100:.1f}%  yds={yds}"
        )

    x = np.array([int(y) for y, *_ in h17_yearly])
    y_scores = np.array([s for _, s, *_ in h17_yearly])
    slope = np.polyfit(x, y_scores, 1)[0]
    direction = "getting harder" if slope > 0 else "getting easier"
    print(f"\n  Linear trend: {slope:+.4f} strokes/year ({direction})")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(16, 8))

    holes = [p["hole_num"] for p in hole_profiles]
    exp_pts = [p["exp_dk_pts"] for p in hole_profiles]
    par_labels = [p["hole_par"] for p in hole_profiles]
    yds_labels = [p["avg_yardage"] for p in hole_profiles]

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in exp_pts]
    bars = ax.bar(holes, exp_pts, color=colors, edgecolor="white", linewidth=0.8, width=0.7)

    for h, v, p, y in zip(holes, exp_pts, par_labels, yds_labels):
        offset = 0.015 if v >= 0 else -0.015
        va = "bottom" if v >= 0 else "top"
        ax.text(h, v + offset, f"P{p}\n{y:.0f}y", ha="center", va=va,
                fontsize=7.5, color="#555", fontweight="bold")

    # Highlight hole 17
    h17_idx = holes.index(17)
    bars[h17_idx].set_edgecolor("#f39c12")
    bars[h17_idx].set_linewidth(3)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Hole Number", fontsize=13)
    ax.set_ylabel("Expected DK Points per Player", fontsize=13)
    ax.set_title(
        "TPC Sawgrass -- Expected DK Points by Hole (2003-2025, 22 Years)\n"
        "Green = Birdie Opportunity | Red = Danger Hole | Gold Border = #17 Island Green",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(holes)
    ax.grid(axis="y", alpha=0.3)

    avg_dk = sum(exp_pts) / len(exp_pts)
    ax.axhline(avg_dk, color="#3498db", linestyle="--", alpha=0.7,
               label=f"Course avg: {avg_dk:+.3f} pts/hole")
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {PLOT_PATH}")
    print(f"Total expected DK pts per 18 holes: {sum(exp_pts):+.3f}")


if __name__ == "__main__":
    main()

"""CSV Export — All lineup export formats in one place.

Handles DK upload format, detail CSV, and upload-with-stats CSV.
No more 60 lines of duplicated export code in main().
"""
import csv
import numpy as np
from collections import Counter
from config import ROSTER_SIZE


def export_all(selected_indices, candidates, players, payouts,
               entry_fee, contest_id, portfolio_roi, sim_elapsed):
    """Export all CSV files for a completed pipeline run.

    Args:
        selected_indices: list of indices into candidates array
        candidates: list of lineups (each a list of player indices)
        players: list of player dicts
        payouts: (n_candidates, n_sims) payout matrix
        entry_fee: contest entry fee
        contest_id: DK contest ID (for filenames)
        portfolio_roi: computed portfolio ROI %
        sim_elapsed: simulation time in seconds

    Returns:
        dict of {filename: filepath} for all exported files
    """
    sel = selected_indices
    n_sel = len(sel)

    # Build exposure counts
    player_counts = Counter()
    for si in sel:
        for idx in candidates[si]:
            player_counts[idx] += 1

    # Portfolio stats
    sel_payouts = payouts[sel]
    portfolio_max = sel_payouts.max(axis=0)
    portfolio_cost = entry_fee * n_sel

    files = {}

    # 1. DK Upload CSV (minimal — just IDs)
    lineup_file = f"lineups_showdown_{contest_id}.csv"
    with open(lineup_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["G"] * ROSTER_SIZE)
        for si in sel:
            w.writerow([players[idx]["dk_id"] for idx in candidates[si]])
    files["lineup"] = lineup_file
    print(f"\n  Exported {lineup_file}")

    # 2. Detail CSV (per-lineup breakdown)
    mean_pay = payouts.mean(axis=1)
    detail_roi = (mean_pay - entry_fee) / entry_fee * 100
    detail_file = f"showdown_{contest_id}_detail.csv"
    with open(detail_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lineup_num", "players", "salary", "projection",
                     "mean_payout", "roi_pct"])
        for rank, si in enumerate(sel, 1):
            lu = candidates[si]
            w.writerow([
                rank,
                " | ".join(players[i]["name"] for i in lu),
                sum(players[i]["salary"] for i in lu),
                f"{sum(players[i]['projected_points'] for i in lu):.1f}",
                f"{mean_pay[si]:.2f}",
                f"{detail_roi[si]:.1f}",
            ])
    files["detail"] = detail_file
    print(f"  Exported {detail_file}")

    # 3. Upload CSV with exposure summary and portfolio stats
    upload_file = f"showdown_{contest_id}_upload.csv"
    with open(upload_file, "w", newline="") as f:
        w = csv.writer(f)
        # Lineups
        w.writerow(["G"] * ROSTER_SIZE)
        for si in sel:
            w.writerow([players[idx]["dk_id"] for idx in candidates[si]])

        # Exposure summary
        w.writerow([])
        w.writerow(["EXPOSURE SUMMARY", "", "", "", "", ""])
        w.writerow(["Player", "Salary", "Projection", "Count", "Exposure %", "DK ID"])

        exposure_sorted = sorted(player_counts.items(), key=lambda x: -x[1])
        for idx, count in exposure_sorted:
            w.writerow([players[idx]["name"], f"${players[idx]['salary']:,}",
                        f"{players[idx]['projected_points']:.1f}",
                        count, f"{count/n_sel*100:.1f}%", players[idx]["dk_id"]])
        for i in range(len(players)):
            if i not in player_counts:
                w.writerow([players[i]["name"], f"${players[i]['salary']:,}",
                            f"{players[i]['projected_points']:.1f}",
                            0, "0.0%", players[i]["dk_id"]])

        # Portfolio stats
        w.writerow([])
        w.writerow(["PORTFOLIO STATS", "", "", "", "", ""])
        w.writerow(["Lineups", n_sel, "", "", "", ""])
        w.writerow(["Investment", f"${portfolio_cost:.0f}", "", "", "", ""])
        w.writerow(["Portfolio ROI", f"{portfolio_roi:+.1f}%", "", "", "", ""])
        w.writerow(["Mean Best Payout", f"${portfolio_max.mean():.2f}", "", "", "", ""])
        w.writerow(["Unique Players", f"{len(player_counts)}/{len(players)}", "", "", "", ""])
        w.writerow(["Sim Time", f"{sim_elapsed:.1f}s", "", "", "", ""])

    files["upload"] = upload_file
    print(f"  Exported {upload_file}")

    return files

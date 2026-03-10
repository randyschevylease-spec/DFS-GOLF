"""
export_dk.py -- Convert portfolio_selected.csv to DraftKings upload format.

Input:  data/cache/portfolio_selected.csv (150 lineups)
Output: data/outputs/dk_upload_PLAYERS_2026.csv

DK format: Entry ID,Contest Name,Contest ID,Entry Fee,G,G,G,G,G,G
Player name format: "First Last (dk_id)"
"""

import csv
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(PROJECT_ROOT, "data", "cache", "portfolio_selected.csv")
OUTPUT = os.path.join(PROJECT_ROOT, "data", "outputs", "dk_upload_PLAYERS_2026.csv")


def flip_name(name):
    """Convert 'Last, First' to 'First Last'."""
    if "," in name:
        parts = name.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name.strip()


def export():
    rows = []
    with open(INPUT) as f:
        for row in csv.DictReader(f):
            players = []
            for i in range(1, 7):
                dk_id = row[f"p{i}_id"]
                dk_name = flip_name(row[f"p{i}_name"])
                players.append(f"{dk_name} ({dk_id})")
            rows.append(players)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Entry ID", "Contest Name", "Contest ID", "Entry Fee",
                         "G", "G", "G", "G", "G", "G"])
        for players in rows:
            writer.writerow(["", "", "", ""] + players)

    print(f"Exported {len(rows)} lineups to {OUTPUT}")
    print(f"\nSample (first 3 rows):")
    for i, players in enumerate(rows[:3], 1):
        print(f"  Row {i}: {' | '.join(players)}")


if __name__ == "__main__":
    export()

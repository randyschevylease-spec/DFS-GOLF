"""
export_dk.py — DraftKings upload CSV formatter.

Converts selected portfolio lineups into the exact CSV format
required by DraftKings for bulk lineup upload.

DK Golf format:
  - Header: G,G,G,G,G,G
  - Each row: 6 player names or DK player IDs
  - Player format: "FirstName LastName (SALARY)"

Inputs:
  - data/cache/portfolio_selected.csv
  - DK salary export (for exact name matching)

Output:
  - data/export/dk_upload.csv (ready for DK import)
"""

import csv
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTFOLIO_PATH = os.path.join(PROJECT_ROOT, "data", "cache", "portfolio_selected.csv")
EXPORT_DIR = os.path.join(PROJECT_ROOT, "data", "export")
OUTPUT = os.path.join(EXPORT_DIR, "dk_upload.csv")


def flip_name(name):
    """Convert 'Last, First' to 'First Last' for DK format."""
    if "," in name:
        parts = name.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name


def load_portfolio(path=None):
    """Load selected portfolio lineups."""
    if path is None:
        path = PORTFOLIO_PATH
    lineups = []
    with open(path) as f:
        for row in csv.DictReader(f):
            players = [row[f"p{i}"] for i in range(1, 7)]
            lineups.append({
                "portfolio_rank": int(row["portfolio_rank"]),
                "lineup_id": int(row["lineup_id"]),
                "strategy": row["strategy"],
                "total_salary": int(row["total_salary"]),
                "players": players,
            })
    return lineups


def load_dk_names(dk_salary_path):
    """
    Load DK salary file to get exact player name mapping.

    DK salary files typically have columns: Name, Salary, Position, etc.
    Returns dict of normalized_name -> dk_exact_name.
    """
    mapping = {}
    with open(dk_salary_path) as f:
        for row in csv.DictReader(f):
            dk_name = row.get("Name", "")
            if dk_name:
                # Normalize for matching
                norm = dk_name.strip().lower()
                mapping[norm] = dk_name
    return mapping


def match_name(player_name, dk_mapping=None):
    """
    Match a player name from our data to DK's exact format.

    Our format: "Last, First"
    DK format: "First Last"
    """
    flipped = flip_name(player_name)

    if dk_mapping:
        norm = flipped.strip().lower()
        if norm in dk_mapping:
            return dk_mapping[norm]

    return flipped


def export(dk_salary_path=None, portfolio_path=None):
    """
    Export portfolio to DK upload format.

    Args:
        dk_salary_path: optional path to DK salary CSV for exact name matching
        portfolio_path: optional path to portfolio CSV

    Returns:
        path to exported file
    """
    lineups = load_portfolio(portfolio_path)

    dk_mapping = None
    if dk_salary_path and os.path.exists(dk_salary_path):
        dk_mapping = load_dk_names(dk_salary_path)
        print(f"Loaded {len(dk_mapping)} DK player names for matching")

    os.makedirs(EXPORT_DIR, exist_ok=True)

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        # DK header: 6 golfer slots
        writer.writerow(["G", "G", "G", "G", "G", "G"])

        for lineup in lineups:
            dk_names = [match_name(p, dk_mapping) for p in lineup["players"]]
            writer.writerow(dk_names)

    print(f"Exported {len(lineups)} lineups to {OUTPUT}")
    print(f"  Format: DraftKings bulk upload CSV")

    # Also print for review
    print(f"\nLineups:")
    for lineup in lineups:
        names = [flip_name(p) for p in lineup["players"]]
        print(f"  #{lineup['portfolio_rank']}: {' | '.join(names)}  (${lineup['total_salary']:,})")

    return OUTPUT


if __name__ == "__main__":
    import sys
    dk_path = sys.argv[1] if len(sys.argv) > 1 else None
    export(dk_salary_path=dk_path)

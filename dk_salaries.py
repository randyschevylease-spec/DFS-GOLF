import os
import glob
import pandas as pd
from difflib import SequenceMatcher


def find_latest_csv(salaries_dir="salaries"):
    """Find the most recently modified CSV file in the salaries directory."""
    pattern = os.path.join(salaries_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_dk_csv(filepath):
    """Parse a DraftKings salary CSV export.

    DraftKings CSVs are typically comma-delimited with columns like:
    Position, Name+ID, Name, ID, Roster Position, Salary, Game Info, TeamAbbrev, AvgPointsPerGame
    """
    # Detect header row — some DK exports have a title row before the actual header
    skip = 0
    with open(filepath) as f:
        for i, line in enumerate(f):
            if "Name" in line and "Salary" in line:
                skip = i
                break

    # Try comma first, then tab
    try:
        df = pd.read_csv(filepath, sep=",", skiprows=skip)
        if len(df.columns) < 3:
            df = pd.read_csv(filepath, sep="\t", skiprows=skip)
    except Exception:
        df = pd.read_csv(filepath, sep="\t", skiprows=skip)

    # Normalize column names (strip whitespace, lowercase)
    df.columns = [c.strip() for c in df.columns]

    # Map common DK column name variations
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "")
        if cl in ("name", "playername"):
            col_map[c] = "name"
        elif cl in ("salary",):
            col_map[c] = "salary"
        elif cl in ("id",):
            col_map[c] = "dk_id"
        elif cl in ("avgpointspergame", "avgpts", "fppg"):
            col_map[c] = "avg_points"
        elif cl in ("name+id", "nameid"):
            col_map[c] = "name_id"
        elif cl in ("teamabbrev",):
            col_map[c] = "team"
        elif cl in ("gameinfo",):
            col_map[c] = "game_info"
        elif cl in ("rosterposition",):
            col_map[c] = "roster_position"

    df = df.rename(columns=col_map)

    # Ensure required columns exist
    if "name" not in df.columns:
        # Try extracting from Name+ID if available
        if "name_id" in df.columns:
            df["name"] = df["name_id"].str.extract(r"^(.+?)\s*\(")[0]

    if "salary" not in df.columns:
        raise ValueError(f"Could not find 'Salary' column in {filepath}. Columns: {list(df.columns)}")

    # Clean salary (remove $ and commas)
    df["salary"] = pd.to_numeric(
        df["salary"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce"
    )

    # Clean avg_points
    if "avg_points" in df.columns:
        df["avg_points"] = pd.to_numeric(df["avg_points"], errors="coerce").fillna(0.0)
    else:
        df["avg_points"] = 0.0

    # Drop rows with no name or salary
    df = df.dropna(subset=["name", "salary"])
    df["salary"] = df["salary"].astype(int)

    # Normalize player names for matching
    df["name_normalized"] = df["name"].apply(normalize_name)

    # Build Name + ID string for DK upload format
    if "dk_id" in df.columns:
        df["name_id"] = df["name"] + " (" + df["dk_id"].astype(str) + ")"
    elif "name_id" not in df.columns:
        df["name_id"] = df["name"]

    cols = ["name", "name_normalized", "salary", "avg_points", "name_id"]
    if "dk_id" in df.columns:
        cols.append("dk_id")
    return df[cols].reset_index(drop=True)


def normalize_name(name):
    """Normalize a player name for matching: lowercase, flip 'Last, First' to 'First Last', strip suffixes."""
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    # Convert "Last, First" → "First Last"
    if "," in name:
        parts = name.split(",", 1)
        name = f"{parts[1].strip()} {parts[0].strip()}"
    # Remove common suffixes
    for suffix in [" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
            break
    # Remove periods and extra spaces
    name = name.replace(".", "").replace("  ", " ")
    return name


def match_players_exact(dk_players, dg_projections):
    """Match DK players to DG fantasy projections via site_name_id ↔ name_id.

    Uses exact string match on DataGolf's site_name_id field which matches
    DraftKings' "Player Name (DK_ID)" format.

    dk_players: DataFrame from parse_dk_csv() with 'name_id' column
    dg_projections: list of dicts from DG fantasy-projection-defaults endpoint

    Returns (matches, unmatched):
        matches: dict mapping DK DataFrame index → DG projection dict
        unmatched: list of unmatched DK player names
    """
    dg_lookup = {p["site_name_id"]: p for p in dg_projections}

    matches = {}
    unmatched = []
    for idx, row in dk_players.iterrows():
        dk_name_id = row["name_id"]
        if dk_name_id in dg_lookup:
            matches[idx] = dg_lookup[dk_name_id]
        else:
            unmatched.append(row["name"])

    return matches, unmatched


def match_players(dk_players, dg_players):
    """Match DraftKings players to DataGolf players by name.

    dk_players: DataFrame with 'name', 'name_normalized' columns
    dg_players: list of dicts with 'player_name' and 'dg_id'

    Returns a dict mapping DK index → dg_id, and a list of unmatched DK names.
    """
    # Build a lookup from normalized DataGolf names
    dg_lookup = {}
    for p in dg_players:
        pname = p.get("player_name", "")
        norm = normalize_name(pname)
        dg_lookup[norm] = p

    matches = {}
    unmatched = []

    for idx, row in dk_players.iterrows():
        dk_norm = row["name_normalized"]

        # Exact match
        if dk_norm in dg_lookup:
            matches[idx] = dg_lookup[dk_norm]
            continue

        # Try "Last, First" → "First Last" conversion
        # DataGolf typically uses "First Last", DK might use either
        if "," in dk_norm:
            parts = dk_norm.split(",", 1)
            flipped = f"{parts[1].strip()} {parts[0].strip()}"
            if flipped in dg_lookup:
                matches[idx] = dg_lookup[flipped]
                continue

        # Fuzzy match — find best match above threshold
        best_score = 0.0
        best_key = None
        for dg_norm in dg_lookup:
            score = SequenceMatcher(None, dk_norm, dg_norm).ratio()
            if score > best_score:
                best_score = score
                best_key = dg_norm
        if best_score >= 0.80:
            matches[idx] = dg_lookup[best_key]
        else:
            unmatched.append(row["name"])

    return matches, unmatched

"""Player data parsing, validation, and std dev derivation.

All player data comes from the uploaded DataGolf projection CSV.
NO API calls — the CSV is the single source of truth.
"""
import csv
import numpy as np
from config import (ROSTER_SIZE, DK_PTS_PER_SG, SINGLE_ROUND_DIVISOR,
                    FALLBACK_STD_DEV_PCT)


def parse_projections(csv_path):
    """Parse showdown projections CSV into player dicts.

    Expected CSV columns (from DataGolf export):
        dk_name, datagolf_name, dk_salary, dk_id, total_points,
        value, morning_wave, position, std_dev (optional),
        proj_ownership (optional), p_make_cut (optional),
        ceiling (optional)

    Returns list of player dicts, sorted by projected_points descending.
    """
    players = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        available_cols = reader.fieldnames or []

        for row_num, row in enumerate(reader, start=2):
            proj = float(row.get("total_points", 0))
            if proj <= 0:
                continue

            name = row.get("dk_name", row.get("datagolf_name", ""))
            salary = int(row.get("dk_salary", 0))
            dk_id = str(row.get("dk_id", ""))

            if not name or salary <= 0 or not dk_id:
                print(f"  ⚠ Row {row_num}: skipped (missing name/salary/dk_id)")
                continue

            player = {
                "name": name,
                "datagolf_name": row.get("datagolf_name", ""),
                "salary": salary,
                "projected_points": proj,
                "dk_id": dk_id,
                "value": float(row.get("value", 0)),
                "wave": 1 if row.get("morning_wave", "").upper() in ("TRUE", "1", "YES") else 0,
                "position": row.get("position", ""),
            }

            # Optional fields from CSV (preferred over derivation)
            if "std_dev" in available_cols and row.get("std_dev"):
                try:
                    sd = float(row["std_dev"])
                    if sd > 0:
                        player["std_dev"] = sd * DK_PTS_PER_SG / SINGLE_ROUND_DIVISOR
                except (ValueError, TypeError):
                    pass

            if "proj_ownership" in available_cols and row.get("proj_ownership"):
                try:
                    own = float(row["proj_ownership"])
                    if own > 0:
                        player["proj_ownership"] = own
                except (ValueError, TypeError):
                    pass

            if "p_make_cut" in available_cols and row.get("p_make_cut"):
                try:
                    player["p_make_cut"] = float(row["p_make_cut"])
                except (ValueError, TypeError):
                    pass

            if "ceiling" in available_cols and row.get("ceiling"):
                try:
                    player["ceiling"] = float(row["ceiling"])
                except (ValueError, TypeError):
                    pass

            players.append(player)

    # Sort by projection descending for consistent indexing
    players.sort(key=lambda p: p["projected_points"], reverse=True)
    return players


def validate_players(players):
    """Validate player data and report issues.

    Checks for:
      - Minimum player count for ROSTER_SIZE lineups
      - Missing or invalid salaries
      - Duplicate DK IDs
      - Projection range sanity
      - Wave assignment coverage

    Returns (is_valid: bool, issues: list[str])
    """
    issues = []

    if len(players) < ROSTER_SIZE:
        issues.append(f"FATAL: Only {len(players)} players, need at least {ROSTER_SIZE}")
        return False, issues

    # Check for duplicate DK IDs
    dk_ids = [p["dk_id"] for p in players]
    dupes = [did for did in set(dk_ids) if dk_ids.count(did) > 1]
    if dupes:
        issues.append(f"WARNING: Duplicate DK IDs: {dupes}")

    # Salary range check
    sals = [p["salary"] for p in players]
    if min(sals) <= 0:
        issues.append(f"FATAL: Players with zero/negative salary found")
        return False, issues

    # Projection range sanity
    projs = [p["projected_points"] for p in players]
    if max(projs) > 200:
        issues.append(f"WARNING: Max projection {max(projs):.1f} seems high for single-round")
    if min(projs) < 1:
        issues.append(f"WARNING: Min projection {min(projs):.1f} — near-zero players in pool")

    # Wave coverage
    wave_counts = {0: 0, 1: 0}
    for p in players:
        wave_counts[p["wave"]] = wave_counts.get(p["wave"], 0) + 1
    if wave_counts[1] == 0:
        issues.append("INFO: No morning wave players detected (all PM or wave data missing)")

    is_valid = not any(i.startswith("FATAL") for i in issues)
    return is_valid, issues


def derive_std_devs(players):
    """Derive single-round std devs from CSV data.

    Priority:
      1. std_dev already parsed from CSV (DG skill decomposition)
      2. Fallback: projected_points * FALLBACK_STD_DEV_PCT

    Logs which players used fallback so you can debug.
    """
    from_csv = 0
    fallback = 0
    fallback_names = []

    for p in players:
        if "std_dev" in p and p["std_dev"] > 0:
            from_csv += 1
        else:
            p["std_dev"] = p["projected_points"] * FALLBACK_STD_DEV_PCT
            fallback += 1
            fallback_names.append(p["name"])

    print(f"  Std devs: {from_csv}/{len(players)} from CSV, {fallback} fallback")
    if fallback_names:
        print(f"  ⚠ Fallback players: {', '.join(fallback_names[:10])}"
              f"{'...' if len(fallback_names) > 10 else ''}")


def compute_ownership(players):
    """Compute projected ownership from salary/projection/position blend.

    Only used when proj_ownership is NOT in the CSV.
    If CSV has ownership data, this step is skipped.
    """
    # Check if ownership already loaded from CSV
    has_ownership = sum(1 for p in players if "proj_ownership" in p)
    if has_ownership == len(players):
        owns = [p["proj_ownership"] for p in players]
        print(f"  Ownership from CSV: {min(owns):.1f}% – {max(owns):.1f}% "
              f"(total {sum(owns):.0f}%)")
        return
    elif has_ownership > 0:
        print(f"  ⚠ Partial ownership data ({has_ownership}/{len(players)}), "
              f"computing for all")

    n = len(players)
    sals = np.array([p["salary"] for p in players], dtype=np.float64)
    projs = np.array([p["projected_points"] for p in players], dtype=np.float64)

    # Normalize to [0, 1]
    sal_norm = (sals - sals.min()) / max(sals.max() - sals.min(), 1)
    proj_norm = (projs - projs.min()) / max(projs.max() - projs.min(), 1)

    # Position bonus (leaders get more ownership)
    pos_score = np.zeros(n)
    for i, p in enumerate(players):
        pos = p.get("position", "")
        try:
            pos_num = int(pos.replace("T", ""))
            pos_score[i] = max(0, 1.0 - pos_num / 70.0)
        except (ValueError, AttributeError):
            pos_score[i] = 0.3

    # Blended score
    blend = 0.50 * proj_norm + 0.25 * sal_norm + 0.25 * pos_score
    temperature = 0.4
    exp_blend = np.exp(blend / temperature)
    ownership = exp_blend / exp_blend.sum()

    # Scale to ~600% total (6-player rosters, ~100% per slot)
    ownership *= ROSTER_SIZE * 100

    for i, p in enumerate(players):
        p["proj_ownership"] = float(ownership[i])

    print(f"  Ownership computed: {ownership.min():.1f}% – {ownership.max():.1f}% "
          f"(total {ownership.sum():.0f}%)")


def enrich_from_sg(players, sg_csv_path):
    """Enrich player std_dev and ownership from live strokes-gained CSV.

    Must be called AFTER derive_std_devs() and compute_ownership().

    Ownership boost (scaled to field position):
        mean_own = mean(proj_ownership)
        boost = sg_total * 0.5 * (proj_ownership / mean_own)
        Renormalize to preserve ROSTER_SIZE * 100 total.

    Std dev adjustment (driving/putting divergence):
        std_dev *= (1 + abs(sg_ott - sg_putt) / projected_points * 0.1)

    Args:
        players: list of player dicts (already have std_dev and proj_ownership)
        sg_csv_path: path to SG CSV with player_name, sg_total, sg_ott, sg_putt
    """
    # Load SG data keyed by "Last, First" name
    sg_lookup = {}
    with open(sg_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("player_name", "").strip()
            if not name:
                continue
            try:
                sg_lookup[name] = {
                    "sg_total": float(row.get("sg_total", 0)),
                    "sg_ott": float(row.get("sg_ott", 0)),
                    "sg_putt": float(row.get("sg_putt", 0)),
                }
            except (ValueError, TypeError):
                continue

    # Also build reverse lookup: "First Last" → SG data (fallback for dk_name match)
    reverse_lookup = {}
    for name_lf, sg in sg_lookup.items():
        parts = name_lf.split(", ", 1)
        if len(parts) == 2:
            reverse_lookup[f"{parts[1]} {parts[0]}"] = sg

    # Match players
    matched = 0
    mean_own = np.mean([p["proj_ownership"] for p in players])

    for p in players:
        sg = sg_lookup.get(p.get("datagolf_name", ""))
        if sg is None:
            sg = reverse_lookup.get(p["name"])
        if sg is None:
            continue

        matched += 1

        # Std dev: widen for players with driving/putting divergence
        divergence = abs(sg["sg_ott"] - sg["sg_putt"])
        p["std_dev"] *= (1 + divergence / p["projected_points"] * 0.1)

        # Ownership: scaled boost by sg_total
        boost = sg["sg_total"] * 0.5 * (p["proj_ownership"] / mean_own)
        p["proj_ownership"] += boost

    # Renormalize ownership to preserve ROSTER_SIZE * 100 total
    total_target = ROSTER_SIZE * 100
    current_total = sum(p["proj_ownership"] for p in players)
    if current_total > 0:
        scale = total_target / current_total
        for p in players:
            p["proj_ownership"] *= scale

    owns = [p["proj_ownership"] for p in players]
    print(f"  SG enrichment: {matched}/{len(players)} matched from {len(sg_lookup)} SG rows")
    print(f"  Ownership after SG: {min(owns):.1f}% – {max(owns):.1f}% "
          f"(total {sum(owns):.0f}%)")

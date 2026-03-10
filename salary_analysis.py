import os
import json
import pandas as pd


HISTORY_DIR = "history"

# Default salary tier breakpoints
SALARY_TIERS = {
    "elite": (9500, 99999),    # $9,500+
    "high": (8500, 9499),      # $8,500-$9,499
    "mid_high": (7500, 8499),  # $7,500-$8,499
    "mid": (7000, 7499),       # $7,000-$7,499
    "mid_low": (6500, 6999),   # $6,500-$6,999
    "value": (6000, 6499),     # $6,000-$6,499
}

# Default historical ROI by salary tier (points per $1K)
# These are reasonable PGA DFS baselines when no history is available
DEFAULT_TIER_ROI = {
    "elite": 6.5,      # High floor, high ceiling
    "high": 6.2,
    "mid_high": 6.0,
    "mid": 5.8,
    "mid_low": 5.5,    # Often best value tier
    "value": 5.0,      # High variance
}


def get_salary_tier(salary):
    """Determine which salary tier a player falls into."""
    for tier, (low, high) in SALARY_TIERS.items():
        if low <= salary <= high:
            return tier
    if salary < 6000:
        return "value"
    return "elite"


def load_historical_data():
    """Load all historical salary + results data from the history directory.

    Returns a DataFrame with columns: event, year, player, salary, actual_points.
    """
    records = []
    if not os.path.exists(HISTORY_DIR):
        return pd.DataFrame(columns=["event", "year", "player", "salary", "actual_points"])

    for filename in os.listdir(HISTORY_DIR):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(HISTORY_DIR, filename)
        try:
            with open(filepath) as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                records.append({
                    "event": entry.get("event", ""),
                    "year": entry.get("year", ""),
                    "player": entry.get("player", ""),
                    "salary": entry.get("salary", 0),
                    "actual_points": entry.get("actual_points", 0),
                })
        except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
            continue

    return pd.DataFrame(records)


def save_event_results(event_name, year, results):
    """Save event results to history for future analysis.

    results: list of dicts with {player, salary, actual_points}
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    for r in results:
        r["event"] = event_name
        r["year"] = year

    filename = f"{event_name.replace(' ', '_')}_{year}.json"
    filepath = os.path.join(HISTORY_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def calculate_tier_roi(history_df):
    """Calculate historical points-per-$1K by salary tier.

    Returns a dict of tier → avg ROI.
    """
    if history_df.empty:
        return DEFAULT_TIER_ROI.copy()

    history_df = history_df.copy()
    history_df["tier"] = history_df["salary"].apply(get_salary_tier)
    history_df["roi"] = history_df["actual_points"] / (history_df["salary"] / 1000.0)

    tier_roi = {}
    for tier in SALARY_TIERS:
        tier_data = history_df[history_df["tier"] == tier]
        if len(tier_data) >= 5:
            tier_roi[tier] = tier_data["roi"].mean()
        else:
            tier_roi[tier] = DEFAULT_TIER_ROI[tier]

    return tier_roi


def calculate_salary_value_score(salary, projected_points, tier_roi=None):
    """Calculate a salary value adjustment for a player.

    Compares the player's projected points-per-$1K against historical tier average.
    Players priced below their expected output get a positive adjustment.

    Returns a point adjustment value.
    """
    if tier_roi is None:
        tier_roi = DEFAULT_TIER_ROI

    tier = get_salary_tier(salary)
    expected_roi = tier_roi.get(tier, 5.5)

    if salary <= 0:
        return 0.0

    player_roi = projected_points / (salary / 1000.0)

    # Difference from expected ROI for their tier
    roi_diff = player_roi - expected_roi

    # Convert to point adjustment (scale by salary tier weight)
    # Higher-salary players: small ROI diff = bigger absolute point impact
    salary_k = salary / 1000.0
    adjustment = roi_diff * salary_k * 0.15

    # Cap the adjustment
    return max(-5.0, min(5.0, adjustment))


def analyze_salary_distribution(players_df):
    """Analyze the salary distribution of a player pool.

    Returns summary stats useful for lineup construction guidance.
    """
    if players_df.empty:
        return {}

    stats = {
        "min_salary": int(players_df["salary"].min()),
        "max_salary": int(players_df["salary"].max()),
        "mean_salary": int(players_df["salary"].mean()),
        "median_salary": int(players_df["salary"].median()),
    }

    # Count players per tier
    tier_counts = {}
    for _, row in players_df.iterrows():
        tier = get_salary_tier(row["salary"])
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    stats["tier_counts"] = tier_counts

    return stats

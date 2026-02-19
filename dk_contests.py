"""DraftKings Contest API — Fetch contest details, classify, and derive optimizer parameters.

Usage:
    from dk_contests import fetch_contest, classify_contest, derive_optimizer_params

    profile = fetch_contest("188100564")
    metrics = classify_contest(profile)
    params = derive_optimizer_params(metrics, profile)
"""
import math
import requests


DK_CONTEST_URL = "https://api.draftkings.com/contests/v1/contests/{contest_id}?format=json"
DK_LOBBY_URL = "https://www.draftkings.com/lobby/getcontests?sport=GOLF"

CONTEST_TYPES = {
    "large_gpp": "Large GPP",
    "small_gpp": "Small GPP",
    "single_entry": "Single Entry",
    "cash": "Cash/50-50",
    "satellite": "Satellite",
}


def fetch_contest(contest_id):
    """Fetch contest details from DraftKings API.

    Returns a cleaned contest profile dict, or raises on failure.
    """
    url = DK_CONTEST_URL.format(contest_id=contest_id)
    resp = requests.get(url, timeout=15, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    })
    resp.raise_for_status()
    raw = resp.json()

    # Contest data is nested under 'contestDetail'
    data = raw.get("contestDetail", raw)

    # Extract payout structure
    payouts = []
    payout_spots = 0
    first_place_prize = 0

    payout_summary = data.get("payoutSummary", [])
    for tier in payout_summary:
        min_pos = tier.get("minPosition", 0)
        max_pos = tier.get("maxPosition", 0)

        # Extract cash prize — first try payoutDescriptions (has numeric value)
        prize = 0
        for pd in tier.get("payoutDescriptions", []):
            val = pd.get("value", 0)
            if val and val > 0:
                prize = val
                break

        # Fallback: tierPayoutDescriptions (may be formatted string like "$200,000.00")
        if not prize:
            tier_descs = tier.get("tierPayoutDescriptions", {})
            for desc_val in tier_descs.values():
                if isinstance(desc_val, (int, float)):
                    prize = desc_val
                elif isinstance(desc_val, str):
                    try:
                        prize = float(desc_val.replace("$", "").replace(",", ""))
                    except ValueError:
                        pass
                if prize > 0:
                    break

        if prize > 0:
            payouts.append((min_pos, max_pos, prize))
            payout_spots = max(payout_spots, max_pos)
            if min_pos == 1:
                first_place_prize = prize

    return {
        "contest_id": str(contest_id),
        "name": data.get("name", "Unknown Contest"),
        "entry_fee": data.get("entryFee", 0),
        "prize_pool": data.get("totalPayouts", 0),
        "max_entries": data.get("maximumEntries", 0),
        "max_entries_per_user": data.get("maximumEntriesPerUser", 1),
        "entries": data.get("entries", 0),
        "payout_spots": payout_spots,
        "first_place_prize": first_place_prize,
        "payouts": payouts,
    }


def fetch_golf_contests():
    """Fetch all current golf contests from DraftKings lobby.

    Returns list of contest summary dicts (lighter than full fetch_contest).
    """
    resp = requests.get(DK_LOBBY_URL, timeout=15, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    })
    resp.raise_for_status()
    data = resp.json()

    contests = []
    for c in data.get("Contests", []):
        max_entries = c.get("m", 0) or 0     # current/max entries
        max_entries_per_user = c.get("mec", 1)
        entry_fee = c.get("a", 0)
        prize_pool = c.get("po", 0)
        name = c.get("n", "Unknown")

        # Quick contest type classification from name hints
        contest_type = "GPP"
        if max_entries_per_user == 1 or "single entry" in name.lower():
            contest_type = "SE"
        elif "50/50" in name or "double up" in name.lower():
            contest_type = "Cash"
        elif "satellite" in name.lower() or "qualifier" in name.lower():
            contest_type = "Sat"

        contests.append({
            "contest_id": str(c.get("id", "")),
            "name": name,
            "entry_fee": entry_fee,
            "prize_pool": prize_pool,
            "max_entries": max_entries,
            "max_entries_per_user": max_entries_per_user,
            "contest_type": contest_type,
            "draft_group_id": c.get("dg"),
        })

    return contests


def classify_contest(profile):
    """Classify a contest and derive payout metrics.

    Returns dict with contest_type, payout_skew (0-1), and component metrics.
    """
    prize_pool = profile["prize_pool"] or 1
    max_entries = profile["max_entries"] or 1
    max_per_user = profile["max_entries_per_user"]
    payout_spots = profile["payout_spots"] or 0
    first_place = profile["first_place_prize"] or 0

    # Core metrics
    payout_pct = payout_spots / max_entries if max_entries > 0 else 0
    top_heavy_ratio = first_place / prize_pool if prize_pool > 0 else 0

    # Field factor: log-scaled (100 entries → 0.3, 1K → 0.5, 10K → 0.7, 100K → 0.9)
    field_factor = min(1.0, math.log10(max(max_entries, 10)) / 5.5)

    # Contest type classification
    if max_per_user == 1:
        contest_type = "single_entry"
    elif payout_pct > 0.40 and top_heavy_ratio < 0.05:
        contest_type = "cash"
    elif payout_pct < 0.05 or top_heavy_ratio > 0.30:
        contest_type = "satellite"
    elif max_entries > 5000:
        contest_type = "large_gpp"
    else:
        contest_type = "small_gpp"

    # Payout skew: master dial (0 = flat/cash, 1 = winner-take-all)
    payout_skew = (
        top_heavy_ratio * 0.5
        + (1 - payout_pct) * 0.3
        + field_factor * 0.2
    )
    payout_skew = max(0.0, min(1.0, payout_skew))

    return {
        "contest_type": contest_type,
        "contest_type_display": CONTEST_TYPES.get(contest_type, contest_type),
        "payout_skew": round(payout_skew, 3),
        "payout_pct": round(payout_pct, 4),
        "top_heavy_ratio": round(top_heavy_ratio, 4),
        "field_factor": round(field_factor, 3),
    }


def derive_optimizer_params(metrics, profile):
    """Derive optimizer parameters from contest metrics.

    All parameters are interpolated from payout_skew (0-1):
    - 0 = cash game (maximize floor, minimal leverage)
    - 1 = satellite/winner-take-all (maximize ceiling, aggressive leverage)

    Returns dict of parameters ready to pass to optimize_lineup().
    """
    skew = metrics["payout_skew"]
    max_per_user = profile["max_entries_per_user"]
    contest_type = metrics["contest_type"]

    # Single-entry special case
    if contest_type == "single_entry":
        return {
            "num_lineups": 1,
            "max_exposure": 1.0,
            "leverage_power": round(0.20 + 0.15 * skew, 3),
            "leverage_floor": 0.60,
            "leverage_cap": 1.80,
            "kelly_fraction": 0.50,
            "lambda_base": 0.0,
            "lambda_penalty_cap": 0.0,
            "n_sims": 5000,
            "max_overlap_early": 6,
            "max_overlap_late": 6,
        }

    # Multi-entry: interpolate all dials from payout_skew
    num_lineups = max_per_user

    return {
        "num_lineups": num_lineups,
        "max_exposure": round(0.35 + 0.45 * skew, 3),
        "leverage_power": round(0.05 + 0.60 * skew, 3),
        "leverage_floor": round(0.90 - 0.40 * skew, 3),
        "leverage_cap": round(1.10 + 0.90 * skew, 3),
        "kelly_fraction": round(0.15 + 0.35 * skew, 3),
        "lambda_base": round(0.003 + 0.007 * skew, 4),
        "lambda_penalty_cap": round(0.05 + 0.15 * skew, 3),
        "n_sims": 5000 + int(10000 * skew),
        "max_overlap_early": max(2, 4 - int(2 * (1 - skew))),
        "max_overlap_late": max(3, 5 - int(2 * (1 - skew))),
    }


def format_contest_summary(profile, metrics, params):
    """Format contest details and derived parameters for display."""
    lines = []
    lines.append(f"  Contest: {profile['name']}")

    fee = profile['entry_fee']
    pool = profile['prize_pool']
    field = profile['max_entries']
    entries = profile['entries']
    max_e = profile['max_entries_per_user']
    spots = profile['payout_spots']
    first = profile['first_place_prize']

    lines.append(f"  Entry Fee: ${fee:,} | Prize Pool: ${pool:,.0f} | Field: {field:,} ({entries:,} entered)")
    lines.append(f"  Max Entries: {max_e} | Payout Spots: {spots:,} ({metrics['payout_pct']*100:.1f}%)")
    if first > 0:
        lines.append(f"  1st Place: ${first:,.0f} ({metrics['top_heavy_ratio']*100:.1f}% of pool)")

    lines.append(f"")
    lines.append(f"  Contest Type: {metrics['contest_type_display'].upper()} (payout skew: {metrics['payout_skew']:.2f})")
    lines.append(f"")
    lines.append(f"  AUTO-TUNED PARAMETERS:")
    lines.append(f"    Lineups:      {params['num_lineups']:<10} Leverage:     {params['leverage_power']:.2f}")
    lines.append(f"    Max Exposure:  {params['max_exposure']:<10.2f} Kelly Frac:   {params['kelly_fraction']:.2f}")
    lines.append(f"    Lambda Base:   {params['lambda_base']:<10.4f} Overlap:      {params['max_overlap_early']}/{params['max_overlap_late']}")
    lines.append(f"    Sims:          {params['n_sims']:<10}")

    return "\n".join(lines)


def format_contest_list(contests):
    """Format lobby contest list for display."""
    if not contests:
        return "  No golf contests found."

    lines = []
    lines.append(f"  {'ID':<14} {'Name':<44} {'Fee':>6} {'Pool':>12} {'Field':>8} {'MaxE':>5} {'Type':<6}")
    lines.append(f"  {'-'*14} {'-'*44} {'-'*6} {'-'*12} {'-'*8} {'-'*5} {'-'*6}")

    for c in sorted(contests, key=lambda x: x["prize_pool"], reverse=True):
        pool_str = f"${c['prize_pool']:,.0f}" if c['prize_pool'] >= 1000 else f"${c['prize_pool']:.0f}"
        lines.append(
            f"  {c['contest_id']:<14} {c['name'][:44]:<44} ${c['entry_fee']:>4} {pool_str:>12} "
            f"{c['max_entries']:>8,} {c['max_entries_per_user']:>5} {c['contest_type']:<6}"
        )

    return "\n".join(lines)

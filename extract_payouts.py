#!/usr/bin/env python3
"""
Extract REAL DraftKings payout tables from FantasyCruncher "All Entrants" CSVs.

Strategy:
  For each position, DK publishes a payout amount. When lineups tie, DK averages
  the payouts across the tied positions. So the reported Profit for any user is:
    sum(DK_payout[pos] for pos in tied_range) / n_tied_lineups - entry_fee

  For our backtest payout table, we need per-TIER allocation as % of prize pool.
  We can compute this by:
    1. For each contest, summing up ALL observed payouts (from all users, not just
       single-lineup), to get total_paid_out.
    2. For each tier, collecting SINGLE-LINEUP observations (they are clean: the
       gross payout = Profit + entry_fee, regardless of ties).
    3. The per-spot payout includes tie-averaging, but over many contests this
       converges to the actual DK per-spot payout.
    4. For 1st place: we use the stated prize from the filename (known ground truth).
    5. For 2nd-10th: very few single-LU observations, high variance from ties.
       Instead, we back-derive from the allocation check: (100% - 1st% - tail%) = top2-10%.
"""

import csv
import os
import re
import statistics
from collections import defaultdict

FOLDER = "/Users/rhbot/Downloads/untitled folder/"

SYNTHETIC = {
    "1st":        0.170,
    "2nd":        0.080,
    "3rd":        0.050,
    "4-5":        0.025,
    "6-10":       0.012,
    "cash_rate":  0.22,
    "rake":       0.15,
}

TIERS = [
    ("1st",      1,    1),
    ("2nd",      2,    2),
    ("3rd",      3,    3),
    ("4-5",      4,    5),
    ("6-10",     6,   10),
    ("11-25",   11,   25),
    ("26-50",   26,   50),
    ("51-100",  51,  100),
    ("101-250", 101,  250),
    ("251-500", 251,  500),
    ("501-1000",501, 1000),
    ("1001-2500",1001,2500),
    ("2501-5000",2501,5000),
    ("5001+",   5001, 999999),
]


def get_tier(place):
    for label, lo, hi in TIERS:
        if lo <= place <= hi:
            return label
    return None


def extract_stated_first_prize(filename):
    m = re.search(r'\[.*?\$([0-9.]+)([MK])\s+to\s+1st', filename, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val * (1_000_000 if m.group(2).upper() == "M" else 1_000)
    return None


def extract_stated_pool(filename):
    m = re.search(r'\$([0-9.]+)([MK])', filename)
    if m:
        val = float(m.group(1))
        return val * (1_000_000 if m.group(2).upper() == "M" else 1_000)
    return None


def parse_contest(filepath):
    """Parse a single All Entrants CSV."""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader)
        all_rows = list(reader)

    if not all_rows:
        return None

    total_lineups = sum(int(r[2]) for r in all_rows)

    entry_fees = [abs(float(r[4])) for r in all_rows if r[2] == "1" and r[3] == "0"]
    if not entry_fees:
        return None
    entry_fee = statistics.median(entry_fees)

    # Single-lineup cashers: clean per-position payout data
    single_cashers = []
    for r in all_rows:
        if r[2] == "1" and r[3] == "1":
            gross = float(r[4]) + entry_fee
            place = int(r[6])
            single_cashers.append((place, gross))

    # Last cashing place across ALL users
    last_cash_place = 0
    for r in all_rows:
        if int(r[3]) > 0:
            last_cash_place = max(last_cash_place, int(r[6]))

    # Total paid out (from ALL users)
    total_paid_out = 0.0
    for r in all_rows:
        profit = float(r[4])
        lineups = int(r[2])
        total_cost = entry_fee * lineups
        total_paid_out += profit + total_cost  # gross payout for this user

    basename = os.path.basename(filepath)
    stated_pool = extract_stated_pool(basename)
    stated_first = extract_stated_first_prize(basename)
    gross_revenue = entry_fee * total_lineups

    if stated_pool:
        implied_rake = 1.0 - (stated_pool / gross_revenue) if gross_revenue > 0 else 0.15
    else:
        implied_rake = 0.15
    prize_pool = stated_pool if stated_pool else gross_revenue * 0.85

    return {
        "filename": basename,
        "entry_fee": entry_fee,
        "total_lineups": total_lineups,
        "prize_pool": prize_pool,
        "gross_revenue": gross_revenue,
        "implied_rake": implied_rake,
        "last_cash_place": last_cash_place,
        "cash_rate": last_cash_place / total_lineups if total_lineups > 0 else 0,
        "single_cashers": single_cashers,
        "stated_first": stated_first,
        "total_paid_out": total_paid_out,
        "n_single_cashers": len(single_cashers),
        "n_single_losers": len(entry_fees),
    }


def build_tier_table(contest):
    """Build per-tier median payout and allocation for a contest.

    For positions 11+: excellent single-LU data, use median per spot.
    For 1st: use stated prize from filename.
    For 2nd-10th: use single-LU data when available (ties are already averaged by DK).

    Returns dict: tier_label -> { per_spot_pct, avg_gross, n_obs, source }
    """
    prize_pool = contest["prize_pool"]
    entry_fee = contest["entry_fee"]

    tier_data = defaultdict(list)
    for place, gross in contest["single_cashers"]:
        tier = get_tier(place)
        if tier:
            tier_data[tier].append(gross)

    result = {}
    for label, lo, hi in TIERS:
        if label in tier_data:
            payouts = tier_data[label]
            med = statistics.median(payouts)
            result[label] = {
                "avg_gross": med,
                "per_spot_pct": med / prize_pool if prize_pool > 0 else 0,
                "n_observations": len(payouts),
                "entry_fee_multiple": med / entry_fee if entry_fee > 0 else 0,
                "source": "single-LU",
            }

    # Override 1st with stated prize
    if contest["stated_first"]:
        stated = contest["stated_first"]
        result["1st"] = {
            "avg_gross": stated,
            "per_spot_pct": stated / prize_pool,
            "n_observations": 0,
            "entry_fee_multiple": stated / entry_fee,
            "source": "stated",
        }

    return result


def compute_tier_allocation(contest, tier_table):
    """For each tier, compute total allocation = per_spot_pct * n_spots.

    Returns dict: tier_label -> fraction of prize pool allocated.
    """
    alloc = {}
    for label, lo, hi in TIERS:
        if label not in tier_table:
            continue
        pct = tier_table[label]["per_spot_pct"]
        if hi < 999999:
            actual_hi = min(hi, contest["last_cash_place"])
            if lo > contest["last_cash_place"]:
                continue
            n = actual_hi - lo + 1
        else:
            if lo > contest["last_cash_place"]:
                continue
            n = contest["last_cash_place"] - lo + 1
        alloc[label] = pct * n
    return alloc


def aggregate_tier_allocations(all_contests, all_tier_tables):
    """For each tier, compute the fraction of prize pool allocated across contests.

    This is (per_spot_pct * n_spots_in_tier) for each contest, then median.
    """
    tier_allocs = defaultdict(list)
    tier_per_spot = defaultdict(list)

    for contest, tier_table in zip(all_contests, all_tier_tables):
        alloc = compute_tier_allocation(contest, tier_table)
        for label in alloc:
            tier_allocs[label].append(alloc[label])
        for label in tier_table:
            tier_per_spot[label].append(tier_table[label]["per_spot_pct"])

    return tier_allocs, tier_per_spot


def main():
    csv_files = []
    for f in sorted(os.listdir(FOLDER)):
        if "All Entrants" in f and f.endswith(".csv"):
            csv_files.append(os.path.join(FOLDER, f))

    print(f"Found {len(csv_files)} All Entrants CSV files\n")

    all_contests = []
    all_tier_tables = []

    for filepath in csv_files:
        contest = parse_contest(filepath)
        if contest is None:
            print(f"  SKIPPED: {os.path.basename(filepath)}")
            continue

        tier_table = build_tier_table(contest)
        all_contests.append(contest)
        all_tier_tables.append(tier_table)

    # ── Per-contest summaries ─────────────────────────────────────────────
    for contest, tier_table in zip(all_contests, all_tier_tables):
        alloc = compute_tier_allocation(contest, tier_table)
        total_alloc = sum(alloc.values())

        print(f"\n{'='*85}")
        name = contest["filename"][:80]
        print(f"  {name}")
        print(f"  Fee=${contest['entry_fee']:.0f}  Lineups={contest['total_lineups']:,}  "
              f"Pool=${contest['prize_pool']:,.0f}  "
              f"Rake={contest['implied_rake']*100:.1f}%  "
              f"CashRt={contest['cash_rate']*100:.2f}%")
        if contest["stated_first"]:
            print(f"  Stated 1st: ${contest['stated_first']:,.0f} "
                  f"({contest['stated_first']/contest['prize_pool']*100:.1f}% of pool)")
        print(f"  Total paid out (all users): ${contest['total_paid_out']:,.0f} "
              f"({contest['total_paid_out']/contest['prize_pool']*100:.1f}% of stated pool)")
        print(f"  {'='*83}")

        print(f"  {'Tier':<14} {'Median$':>10} {'%Pool/spot':>10} {'xEntry':>7} "
              f"{'TierAlloc':>10} {'Obs':>5}  {'Source':<10}")
        print(f"  {'-'*75}")

        for label, lo, hi in TIERS:
            if label in tier_table:
                t = tier_table[label]
                a = alloc.get(label, 0)
                print(f"  {label:<14} ${t['avg_gross']:>8,.0f} "
                      f"{t['per_spot_pct']*100:>9.4f}% "
                      f"{t['entry_fee_multiple']:>6.0f}x "
                      f"{a*100:>9.3f}% "
                      f"{t['n_observations']:>5}  {t['source']:<10}")

        print(f"  {'TOTAL':<14} {'':>10} {'':>10} {'':>7} "
              f"{total_alloc*100:>9.1f}%")

    # ── Aggregate analysis ────────────────────────────────────────────────
    print(f"\n\n{'#'*85}")
    print(f"#  AGGREGATE ANALYSIS ({len(all_contests)} contests)")
    print(f"{'#'*85}")

    cash_rates = [c["cash_rate"] for c in all_contests]
    rakes = [c["implied_rake"] for c in all_contests]

    min_cash_multiples = []
    for c in all_contests:
        if c["single_cashers"]:
            min_gross = min(g for _, g in c["single_cashers"])
            min_cash_multiples.append(min_gross / c["entry_fee"])

    # Payout verification: total_paid_out / prize_pool
    payout_ratios = [c["total_paid_out"] / c["prize_pool"] for c in all_contests
                     if c["prize_pool"] > 0]

    print(f"\n  Cash Rate: median={statistics.median(cash_rates)*100:.2f}%  "
          f"mean={statistics.mean(cash_rates)*100:.2f}%")
    print(f"  Rake:      median={statistics.median(rakes)*100:.1f}%  "
          f"mean={statistics.mean(rakes)*100:.1f}%")
    if min_cash_multiples:
        print(f"  Min Cash:  median={statistics.median(min_cash_multiples):.2f}x  "
              f"mean={statistics.mean(min_cash_multiples):.2f}x")
    print(f"  Payout Verification (total_paid / stated_pool): "
          f"median={statistics.median(payout_ratios)*100:.1f}%  "
          f"mean={statistics.mean(payout_ratios)*100:.1f}%")

    tier_allocs, tier_per_spot = aggregate_tier_allocations(all_contests, all_tier_tables)

    # ── Tier allocation table ─────────────────────────────────────────────
    print(f"\n  TIER ALLOCATION (% of prize pool allocated to entire tier range)")
    print(f"  {'Tier':<14} {'Spots':>6} {'Median':>10} {'Mean':>10} {'Min':>10} {'Max':>10} {'#Ctst':>6}")
    print(f"  {'-'*70}")

    total_median_alloc = 0.0
    for label, lo, hi in TIERS:
        if label in tier_allocs:
            vals = tier_allocs[label]
            n_spots = hi - lo + 1 if hi < 999999 else "var"
            med = statistics.median(vals)
            total_median_alloc += med
            print(f"  {label:<14} {str(n_spots):>6} {med*100:>9.3f}% "
                  f"{statistics.mean(vals)*100:>9.3f}% "
                  f"{min(vals)*100:>9.3f}% {max(vals)*100:>9.3f}% "
                  f"{len(vals):>6}")

    print(f"  {'TOTAL':<14} {'':>6} {total_median_alloc*100:>9.1f}%")

    # ── Per-spot payout table ─────────────────────────────────────────────
    print(f"\n  PER-SPOT PAYOUT (% of prize pool per position)")
    print(f"  {'Tier':<14} {'Median':>10} {'Mean':>10} {'Min':>10} {'Max':>10} {'#Ctst':>6}")
    print(f"  {'-'*65}")

    for label, lo, hi in TIERS:
        if label in tier_per_spot:
            vals = tier_per_spot[label]
            print(f"  {label:<14} {statistics.median(vals)*100:>9.4f}% "
                  f"{statistics.mean(vals)*100:>9.4f}% "
                  f"{min(vals)*100:>9.4f}% {max(vals)*100:>9.4f}% "
                  f"{len(vals):>6}")

    # ── Separate contest types ────────────────────────────────────────────
    mill_idx = [i for i, c in enumerate(all_contests) if "Millionaire" in c["filename"]]
    non_mill_idx = [i for i, c in enumerate(all_contests) if "Millionaire" not in c["filename"]]

    for group_name, indices in [("Millionaire Contests", mill_idx),
                                 ("Non-Millionaire Contests", non_mill_idx)]:
        if not indices:
            continue

        print(f"\n  --- {group_name} ({len(indices)} contests) ---")
        sub_allocs = defaultdict(list)
        sub_perspot = defaultdict(list)
        for i in indices:
            alloc = compute_tier_allocation(all_contests[i], all_tier_tables[i])
            for label in alloc:
                sub_allocs[label].append(alloc[label])
            for label in all_tier_tables[i]:
                sub_perspot[label].append(all_tier_tables[i][label]["per_spot_pct"])

        print(f"  1st % of pool: median={statistics.median(sub_perspot.get('1st', [0]))*100:.1f}%")
        sub_cash = [all_contests[i]["cash_rate"] for i in indices]
        print(f"  Cash rate:     median={statistics.median(sub_cash)*100:.2f}%")

        total = 0
        for label, lo, hi in TIERS:
            if label in sub_allocs:
                med = statistics.median(sub_allocs[label])
                total += med
                if label in ["1st", "2nd", "3rd", "4-5", "6-10"]:
                    print(f"    {label}: tier alloc median = {med*100:.2f}%")
        print(f"    Total allocation: {total*100:.1f}%")

    # ── Side-by-side comparison with synthetic ────────────────────────────
    print(f"\n\n{'#'*85}")
    print(f"#  COMPARISON: Synthetic vs Empirical")
    print(f"{'#'*85}")

    # Use non-Millionaire contests as they better represent typical GPP structure
    # (Millionaires are ultra-top-heavy with $1M guaranteed first)
    print(f"\n  NOTE: Using NON-MILLIONAIRE contests ({len(non_mill_idx)}) for comparison")
    print(f"  (Millionaire contests have $1M guaranteed 1st regardless of pool size,")
    print(f"   making them 25-40% top-heavy, which is not representative of $25 GPPs)\n")

    nm_perspot = defaultdict(list)
    nm_allocs = defaultdict(list)
    for i in non_mill_idx:
        alloc = compute_tier_allocation(all_contests[i], all_tier_tables[i])
        for label in alloc:
            nm_allocs[label].append(alloc[label])
        for label in all_tier_tables[i]:
            nm_perspot[label].append(all_tier_tables[i][label]["per_spot_pct"])

    nm_cash_rates = [all_contests[i]["cash_rate"] for i in non_mill_idx]

    synthetic_map = {
        "1st":  SYNTHETIC["1st"],
        "2nd":  SYNTHETIC["2nd"],
        "3rd":  SYNTHETIC["3rd"],
        "4-5":  SYNTHETIC["4-5"],
        "6-10": SYNTHETIC["6-10"],
    }

    print(f"  {'Tier':<14} {'Synthetic':>12} {'Emp(perSpot)':>14} {'Emp(tierAlloc)':>15}  Notes")
    print(f"  {'-'*80}")

    # For each tier, show: synthetic per-spot, empirical per-spot, empirical tier alloc
    emp_total_alloc = 0.0
    for label, lo, hi in TIERS:
        syn_val = synthetic_map.get(label)
        emp_ps = statistics.median(nm_perspot[label]) if label in nm_perspot else None
        emp_ta = statistics.median(nm_allocs[label]) if label in nm_allocs else None
        if emp_ta:
            emp_total_alloc += emp_ta

        syn_str = f"{syn_val*100:.4f}%" if syn_val is not None else "       --"
        ps_str = f"{emp_ps*100:.4f}%" if emp_ps is not None else "       --"
        ta_str = f"{emp_ta*100:.3f}%" if emp_ta is not None else "        --"

        if syn_val is not None and emp_ps is not None:
            ratio = emp_ps / syn_val if syn_val > 0 else 0
            note = f"  {ratio:.2f}x"
            if ratio > 1.5:
                note += " HIGHER"
            elif ratio < 0.5:
                note += " LOWER"
        else:
            note = ""

        print(f"  {label:<14} {syn_str:>12} {ps_str:>14} {ta_str:>15} {note}")

    print(f"  {'TOTAL alloc':<14} {'':>12} {'':>14} {emp_total_alloc*100:>14.1f}%")

    # Cash rate and rake
    emp_cash = statistics.median(nm_cash_rates)
    emp_rake_val = statistics.median([all_contests[i]["implied_rake"] for i in non_mill_idx])
    print(f"\n  {'Cash Rate':<14} {SYNTHETIC['cash_rate']*100:>11.1f}% "
          f"{emp_cash*100:>13.2f}%")
    print(f"  {'Rake':<14} {SYNTHETIC['rake']*100:>11.1f}% "
          f"{emp_rake_val*100:>13.1f}%")

    # ── Back-derive 2nd-10th from total allocation ────────────────────────
    print(f"\n\n{'#'*85}")
    print(f"#  DERIVING TOP-10 STRUCTURE FROM ALLOCATION BUDGET")
    print(f"{'#'*85}")

    # For non-Millionaire contests:
    # We know total allocation should be ~100% of pool
    # We have reliable data for 11+ tiers
    # We can back-derive how much goes to top 10

    tail_alloc = 0.0  # tiers 11+
    tail_tiers = {}
    for label, lo, hi in TIERS:
        if lo >= 11 and label in nm_allocs:
            med = statistics.median(nm_allocs[label])
            tail_alloc += med
            tail_tiers[label] = med

    # 1st place from stated prizes
    first_pcts = [all_contests[i]["stated_first"] / all_contests[i]["prize_pool"]
                  for i in non_mill_idx if all_contests[i]["stated_first"]]
    first_alloc = statistics.median(first_pcts) if first_pcts else 0.25

    # Budget for 2nd-10th = 100% - 1st - tail
    top2_10_budget = 1.0 - first_alloc - tail_alloc

    print(f"\n  Non-Millionaire contest allocation budget:")
    print(f"    1st place (stated):       {first_alloc*100:.1f}%")
    print(f"    Tiers 11+ (empirical):    {tail_alloc*100:.1f}%")
    print(f"    Budget for 2nd-10th:      {top2_10_budget*100:.1f}%")
    print(f"    (This should be ~35-40% in a typical DK GPP)")

    # Now let's use the data we DO have for 2nd-10th
    # For each non-Millionaire contest, look at what single-LU data exists for 2-10
    print(f"\n  Available single-LU data for positions 2-10 (non-Millionaire):")
    for label in ["2nd", "3rd", "4-5", "6-10"]:
        if label in nm_perspot:
            vals = nm_perspot[label]
            print(f"    {label}: {len(vals)} contests, "
                  f"median={statistics.median(vals)*100:.4f}%, "
                  f"values={[round(v*100, 3) for v in sorted(vals)]}")

    # The 2-10 per-spot data from single-LU users is somewhat noisy due to ties
    # but should be roughly correct. Let's use them as-is and see what the
    # implied top-10 allocation would be.
    pct_2nd_ps = statistics.median(nm_perspot["2nd"]) if "2nd" in nm_perspot else 0
    pct_3rd_ps = statistics.median(nm_perspot["3rd"]) if "3rd" in nm_perspot else 0
    pct_45_ps = statistics.median(nm_perspot["4-5"]) if "4-5" in nm_perspot else 0
    pct_610_ps = statistics.median(nm_perspot["6-10"]) if "6-10" in nm_perspot else 0

    implied_2_10 = pct_2nd_ps + pct_3rd_ps + pct_45_ps * 2 + pct_610_ps * 5
    print(f"\n  Implied 2nd-10th allocation from single-LU data:")
    print(f"    2nd: {pct_2nd_ps*100:.3f}% x 1 = {pct_2nd_ps*100:.3f}%")
    print(f"    3rd: {pct_3rd_ps*100:.3f}% x 1 = {pct_3rd_ps*100:.3f}%")
    print(f"    4-5: {pct_45_ps*100:.3f}% x 2 = {pct_45_ps*2*100:.3f}%")
    print(f"    6-10: {pct_610_ps*100:.3f}% x 5 = {pct_610_ps*5*100:.3f}%")
    print(f"    Total 2nd-10th: {implied_2_10*100:.1f}%")
    print(f"    Budget was:     {top2_10_budget*100:.1f}%")
    print(f"    Gap:            {(top2_10_budget - implied_2_10)*100:.1f}%")

    # Scale 2nd-10th to fit the budget
    if implied_2_10 > 0:
        scale = top2_10_budget / implied_2_10
        print(f"\n  Scaling factor to fit budget: {scale:.2f}x")
        print(f"  Scaled values:")
        print(f"    2nd: {pct_2nd_ps*scale*100:.4f}%")
        print(f"    3rd: {pct_3rd_ps*scale*100:.4f}%")
        print(f"    4-5: {pct_45_ps*scale*100:.4f}% each")
        print(f"    6-10: {pct_610_ps*scale*100:.4f}% each")

    # ── Final recommended values ──────────────────────────────────────────
    print(f"\n\n{'#'*85}")
    print(f"#  RECOMMENDED EMPIRICAL PAYOUT TABLE")
    print(f"{'#'*85}")

    # Use the non-Millionaire data, with scale adjustment for 2-10th
    # For 1st: use median stated 1st as % of pool across non-Millionaire
    # For 2-10: use single-LU medians, potentially scaled
    # For 11+: use single-LU medians directly (high confidence, many obs)

    rec_1st = first_alloc
    # For 2nd-10th, use the raw single-LU values (not scaled).
    # The "gap" represents prize pool allocated to positions where no single-LU
    # user exists in these contests (e.g., pure multi-entry winners at 2-3rd).
    # The tie-averaged values from single-LU data ARE the correct expected payouts.
    rec_2nd = pct_2nd_ps if pct_2nd_ps > 0 else 0.050
    rec_3rd = pct_3rd_ps if pct_3rd_ps > 0 else 0.025
    rec_45 = pct_45_ps if pct_45_ps > 0 else 0.015
    rec_610 = pct_610_ps if pct_610_ps > 0 else 0.005

    # For 11+ use non-Millionaire medians
    rec_rest = {}
    for label, lo, hi in TIERS:
        if lo >= 11 and label in nm_perspot:
            rec_rest[label] = statistics.median(nm_perspot[label])

    # Compute total allocation with these values
    total = rec_1st + rec_2nd + rec_3rd + rec_45 * 2 + rec_610 * 5

    # Estimate typical field for tail tiers
    avg_field = statistics.mean([all_contests[i]["total_lineups"] for i in non_mill_idx])
    avg_cash_spots = avg_field * emp_cash

    for label, lo, hi in TIERS:
        if lo >= 11 and label in rec_rest:
            if hi < 999999:
                n = min(hi, int(avg_cash_spots)) - lo + 1
                if n > 0:
                    total += rec_rest[label] * n
            else:
                n = max(0, int(avg_cash_spots) - lo + 1)
                if n > 0:
                    total += rec_rest[label] * n

    print(f"\n  Estimated total allocation: {total*100:.1f}%")
    print(f"  (Should be close to 100% for a well-calibrated table)\n")

    # Print the recommended dictionary
    print("# =====================================================================")
    print("# Empirical payout table derived from real DraftKings PGA GPP contests")
    print(f"# Based on {len(non_mill_idx)} non-Millionaire contests")
    print(f"# Entry fees: $5-$25, Fields: 26K-78K lineups")
    print("# =====================================================================")
    print()
    print("EMPIRICAL_PAYOUT_PCT = {")
    all_labels = [t[0] for t in TIERS]
    vals_dict = {}
    vals_dict["1st"] = rec_1st
    vals_dict["2nd"] = rec_2nd
    vals_dict["3rd"] = rec_3rd
    vals_dict["4-5"] = rec_45
    vals_dict["6-10"] = rec_610
    for label in all_labels:
        if label in rec_rest:
            vals_dict[label] = rec_rest[label]

    for label, lo, hi in TIERS:
        if label in vals_dict:
            v = vals_dict[label]
            n = hi - lo + 1 if hi < 999999 else "var"
            if hi < 999999:
                tier_tot = v * (hi - lo + 1)
                extra = f"tier={tier_tot*100:.3f}%"
            else:
                extra = "tail"
            print(f'    "{label}": {v:.6f},  '
                  f'# {v*100:.4f}% each, {n} spots, {extra}')
    print("}")

    print(f"\nEMPIRICAL_CASH_RATE = {emp_cash:.4f}  # {emp_cash*100:.2f}%")
    print(f"EMPIRICAL_RAKE = {emp_rake_val:.4f}      # {emp_rake_val*100:.1f}%")

    if min_cash_multiples:
        nm_mincash = [min_cash_multiples[i] for i in range(len(all_contests))
                      if i in set(non_mill_idx)]
        if nm_mincash:
            med_mc = statistics.median(nm_mincash)
            print(f"EMPIRICAL_MIN_CASH_MULTIPLE = {med_mc:.2f}")

    # ── Drop-in replacement function ──────────────────────────────────────
    print(f"\n\n{'#'*85}")
    print(f"#  DROP-IN REPLACEMENT FOR backtest_all.py::build_synthetic_payout_table()")
    print(f"{'#'*85}")

    print(f"""
def build_empirical_payout_table(field_size, entry_fee):
    \"\"\"GPP payout table calibrated from {len(non_mill_idx)} real DraftKings PGA contests.

    Key differences from old synthetic table:
      - 1st place: {rec_1st*100:.1f}% of pool (was 17.0%)
      - Much more detail: 14 tiers instead of 5
      - Cash rate: {emp_cash*100:.1f}% (was 22.0%)
    \"\"\"
    prize_pool = entry_fee * field_size * {1.0 - emp_rake_val:.4f}
    payout_spots = max(1, int(field_size * {emp_cash:.4f}))

    payouts = []""")

    for label, lo, hi in TIERS:
        if label in vals_dict:
            v = vals_dict[label]
            if hi < 999999:
                print(f'    payouts.append(({lo}, min({hi}, payout_spots), '
                      f'round(prize_pool * {v:.6f})))  # {label}')
            else:
                nm_mc = [min_cash_multiples[i] for i in range(len(all_contests))
                         if i in set(non_mill_idx)]
                med_mc = statistics.median(nm_mc) if nm_mc else 1.0
                print(f"""
    # Tail: min-cash
    if payout_spots > 5000:
        min_cash = max(1, round(entry_fee * {med_mc:.2f}, 2))
        payouts.append((5001, payout_spots, min_cash))

    # Remove tiers past payout_spots
    payouts = [(lo, min(hi, payout_spots), amt)
               for lo, hi, amt in payouts if lo <= payout_spots]

    return payouts, entry_fee, prize_pool""")

    # ── Contest summary ───────────────────────────────────────────────────
    print(f"\n\n{'#'*85}")
    print(f"#  CONTEST SUMMARY TABLE")
    print(f"{'#'*85}\n")

    print(f"  {'#':<3} {'Type':<5} {'Contest':<50} {'Fee':>5} {'Lineups':>9} "
          f"{'Pool':>12} {'Rake':>6} {'CashRt':>7} {'1stPz':>12} {'1st%':>6}")
    print(f"  {'-'*125}")

    for i, c in enumerate(all_contests, 1):
        name = c["filename"].replace(" All Entrants", "").replace(".csv", "")
        if len(name) > 48:
            name = name[:45] + "..."
        first_str = f"${c['stated_first']:,.0f}" if c["stated_first"] else "?"
        first_pct = c["stated_first"] / c["prize_pool"] * 100 if c["stated_first"] else 0
        ctype = "MILL" if "Millionaire" in c["filename"] else "GPP"
        print(f"  {i:<3} {ctype:<5} {name:<50} ${c['entry_fee']:>4.0f} "
              f"{c['total_lineups']:>9,} ${c['prize_pool']:>10,.0f} "
              f"{c['implied_rake']*100:>5.1f}% {c['cash_rate']*100:>6.2f}% "
              f"{first_str:>12} {first_pct:>5.1f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FantasyCruncher CSV Data Analysis for DFS Golf Calibration
Analyzes Player Exposures and All Entrants files to evaluate
FC projection accuracy and contest characteristics.
"""

import csv
import os
import math
import re
from collections import defaultdict

DATA_DIR = "/Users/rhbot/Downloads/untitled folder/"

# ── helpers ──────────────────────────────────────────────────────────────

def pearson_r(xs, ys):
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 3:
        return float('nan')
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx)**2 for x in xs) / (n - 1))
    sy = math.sqrt(sum((y - my)**2 for y in ys) / (n - 1))
    if sx == 0 or sy == 0:
        return float('nan')
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    return cov / (sx * sy)


def _rank(vals):
    """Average ranks for tied values."""
    n = len(vals)
    indexed = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and vals[indexed[j]] == vals[indexed[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j
    return ranks


def spearman_r(xs, ys):
    """Spearman rank correlation coefficient."""
    return pearson_r(_rank(xs), _rank(ys))


def mae(xs, ys):
    """Mean absolute error."""
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


def rmse(xs, ys):
    return math.sqrt(sum((x - y)**2 for x, y in zip(xs, ys)) / len(xs))


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return float('nan')
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def read_csv(path):
    """Read CSV, return list of dicts. Handles quoted fields."""
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def safe_float(val, default=0.0):
    try:
        return float(val.replace(',', '').replace('$', '').strip())
    except (ValueError, AttributeError):
        return default


# ── discover files ───────────────────────────────────────────────────────

def discover_files(data_dir):
    """
    Find all Player Exposures and All Entrants CSV pairs.
    Returns list of dicts with keys: base_name, pe_path, ae_path
    """
    all_files = os.listdir(data_dir)
    pe_files = sorted([f for f in all_files if 'Player Exposures' in f and f.endswith('.csv')])
    ae_files = sorted([f for f in all_files if 'All Entrants' in f and f.endswith('.csv')])

    results = []
    for pf in pe_files:
        full_pe = os.path.join(data_dir, pf)
        # derive base name: strip "Player Exposures" and any copy/number suffix
        base = pf.replace('Player Exposures', '').strip()
        # Remove copy markers and .csv
        base_clean = base.replace('.csv', '').strip()

        # find matching All Entrants file
        # The base contest name is everything before "Player Exposures"
        contest_prefix = pf.split('Player Exposures')[0].strip()
        # Try to find a matching AE file with same prefix
        matching_ae = None
        # Match using the same suffix (copy, copy 2, (1), etc.)
        suffix = pf.split('Player Exposures')[1].replace('.csv', '').strip()
        # Build the expected AE filename
        expected_ae = contest_prefix + 'All Entrants' + (' ' + suffix if suffix else '') + '.csv'
        if expected_ae in ae_files:
            matching_ae = os.path.join(data_dir, expected_ae)
        else:
            # Try without suffix
            expected_ae_base = contest_prefix + 'All Entrants.csv'
            if expected_ae_base in ae_files:
                matching_ae = os.path.join(data_dir, expected_ae_base)
            else:
                # try with the same suffix
                for af in ae_files:
                    if af.startswith(contest_prefix) and af.endswith('.csv'):
                        matching_ae = os.path.join(data_dir, af)
                        break

        results.append({
            'filename': pf,
            'contest_prefix': contest_prefix.strip(),
            'suffix': suffix,
            'pe_path': full_pe,
            'ae_path': matching_ae,
        })

    return results


# ── identify tournament from player field ────────────────────────────────

# Known PGA fields: player combos that hint at specific tournaments
TOURNAMENT_HINTS = {
    # The Masters
    frozenset(['Jon Rahm', 'Jordan Spieth', 'Justin Thomas', 'Dustin Johnson',
               'Bryson DeChambeau']): 'The Masters (Augusta)',
    # Players known for specific 2021-2022 events
}

def identify_tournament_from_players(players, contest_name, salary_range=None):
    """
    Attempt to identify which real PGA tournament a contest corresponds to,
    using the combination of player field and contest name.
    """
    # We mainly rely on the unique player pool per week
    # Return the player set signature for grouping
    return frozenset(players)


# ── analyze one Player Exposures file ────────────────────────────────────

def analyze_pe_file(pe_path):
    """Parse a Player Exposures CSV and return structured data."""
    rows = read_csv(pe_path)
    records = []
    for row in rows:
        rec = {
            'player': row.get('Player', '').strip(),
            'salary': safe_float(row.get('Salary', '0')),
            'fc_proj': safe_float(row.get('FC Proj', '0')),
            'own_proj': safe_float(row.get('Own Proj', '0')),
            'fps': safe_float(row.get('FPs', '0')),
            'value': safe_float(row.get('Value', '0')),
            'count': safe_float(row.get('Count', '0')),
            'ownership': safe_float(row.get('Ownership', '0')),
        }
        if rec['player']:
            records.append(rec)
    return records


# ── analyze one All Entrants file ────────────────────────────────────────

def analyze_ae_file(ae_path):
    """Parse an All Entrants CSV and return contest metadata."""
    if ae_path is None or not os.path.exists(ae_path):
        return None

    rows = read_csv(ae_path)
    if not rows:
        return None

    # Extract data
    total_entrants = len(rows)
    profits = []
    scores = []
    places = []
    lineups_total = 0

    for row in rows:
        profit = safe_float(row.get('Profit', '0'))
        score = safe_float(row.get('Score', '0'))
        place = safe_float(row.get('Place', '0'))
        lineups = safe_float(row.get('Lineups', '0'))
        profits.append(profit)
        scores.append(score)
        places.append(place)
        lineups_total += int(lineups)

    # Infer entry fee from single-entry users who did NOT cash (pure losers)
    entry_fee = None
    for row in rows:
        lineups = safe_float(row.get('Lineups', '0'))
        profit = safe_float(row.get('Profit', '0'))
        cash = safe_float(row.get('# Cash', '0'))
        if lineups == 1 and cash == 0 and profit < 0:
            entry_fee = abs(profit)
            break

    # Payout structure: use single-entry users to get clean per-place prizes
    # For each place, find users with 1 lineup to get exact prize amount
    place_prizes_clean = defaultdict(list)
    place_prizes_all = defaultdict(list)
    for row in rows:
        place = int(safe_float(row.get('Place', '0')))
        profit = safe_float(row.get('Profit', '0'))
        lineups = safe_float(row.get('Lineups', '0'))
        cash = safe_float(row.get('# Cash', '0'))
        if entry_fee is not None:
            gross = profit + lineups * entry_fee
        else:
            gross = profit
        if gross > 0 and lineups > 0:
            per_lineup = gross / lineups
            place_prizes_all[place].append(per_lineup)
            # Prefer single-entry users for clean numbers
            if lineups == 1 and cash >= 1:
                place_prizes_clean[place].append(gross)

    # Summarize payout tiers (prefer clean single-entry data)
    payout_tiers = {}
    for place in sorted(place_prizes_all.keys()):
        if place in place_prizes_clean and place_prizes_clean[place]:
            payout_tiers[place] = median(place_prizes_clean[place])
        else:
            payout_tiers[place] = median(place_prizes_all[place])

    # How many unique entrants cashed
    cashed = sum(1 for row in rows if safe_float(row.get('# Cash', '0')) > 0)

    # Max place that won money
    max_paying_place = max(place_prizes_all.keys()) if place_prizes_all else 0

    return {
        'total_unique_entrants': total_entrants,
        'total_lineups': lineups_total,
        'entry_fee': entry_fee,
        'cashed_entrants': cashed,
        'max_paying_place': max_paying_place,
        'top_score': max(scores) if scores else 0,
        'median_score': median(scores),
        'payout_tiers': payout_tiers,
    }


# ── main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("  FANTASYCRUNCHER DFS GOLF DATA ANALYSIS")
    print("  Calibration of FC Projections vs Actual Fantasy Points")
    print("=" * 90)
    print()

    # Discover files
    file_sets = discover_files(DATA_DIR)
    print(f"Found {len(file_sets)} Player Exposures files\n")

    # ── Per-file analysis ────────────────────────────────────────────────
    all_fc_proj = []
    all_fps = []
    all_salary = []
    all_ownership = []
    all_own_proj = []

    # Group by player set to identify unique tournaments
    tournament_groups = defaultdict(list)

    per_file_stats = []

    for i, fs in enumerate(file_sets):
        records = analyze_pe_file(fs['pe_path'])
        if not records:
            print(f"  WARNING: No data in {fs['filename']}")
            continue

        fc_proj = [r['fc_proj'] for r in records]
        fps = [r['fps'] for r in records]
        salary = [r['salary'] for r in records]
        ownership = [r['ownership'] for r in records]
        own_proj = [r['own_proj'] for r in records]
        players = [r['player'] for r in records]

        # Correlations for this file
        pr = pearson_r(fc_proj, fps)
        sr = spearman_r(fc_proj, fps)
        file_mae = mae(fc_proj, fps)
        file_rmse = rmse(fc_proj, fps)

        # Ownership projection accuracy
        own_pr = pearson_r(own_proj, ownership) if len(own_proj) >= 3 else float('nan')

        # Contest info
        ae_info = analyze_ae_file(fs['ae_path'])

        # Group by player set (top 10 players as signature)
        top_players = tuple(sorted(players[:min(15, len(players))]))
        tournament_groups[top_players].append(fs)

        stat = {
            'idx': i + 1,
            'filename': fs['filename'],
            'contest_prefix': fs['contest_prefix'],
            'n_players': len(records),
            'pearson': pr,
            'spearman': sr,
            'mae': file_mae,
            'rmse': file_rmse,
            'own_pearson': own_pr,
            'avg_fc_proj': sum(fc_proj) / len(fc_proj),
            'avg_fps': sum(fps) / len(fps),
            'players': players,
            'ae_info': ae_info,
        }
        per_file_stats.append(stat)

        # Accumulate
        all_fc_proj.extend(fc_proj)
        all_fps.extend(fps)
        all_salary.extend(salary)
        all_ownership.extend(ownership)
        all_own_proj.extend(own_proj)

    # ── Identify unique tournaments ──────────────────────────────────────
    # Group files by overlapping player pools
    print("-" * 90)
    print("  TOURNAMENT IDENTIFICATION (by player pool)")
    print("-" * 90)

    # Better grouping: compare player sets with Jaccard similarity
    n = len(per_file_stats)
    player_sets = [set(s['players']) for s in per_file_stats]
    assigned_group = [-1] * n
    group_id = 0

    for i in range(n):
        if assigned_group[i] >= 0:
            continue
        assigned_group[i] = group_id
        for j in range(i + 1, n):
            if assigned_group[j] >= 0:
                continue
            intersection = len(player_sets[i] & player_sets[j])
            union = len(player_sets[i] | player_sets[j])
            jaccard = intersection / union if union > 0 else 0
            if jaccard > 0.5:
                assigned_group[j] = group_id
        group_id += 1

    tournament_map = defaultdict(list)
    for i, g in enumerate(assigned_group):
        tournament_map[g].append(i)

    print(f"\nIdentified {len(tournament_map)} unique tournaments across {n} files:\n")

    for gid in sorted(tournament_map.keys()):
        indices = tournament_map[gid]
        stats_in_group = [per_file_stats[i] for i in indices]
        # representative players (top 5 by salary from first file)
        rep_players = stats_in_group[0]['players'][:5]
        print(f"  Tournament {gid + 1}:")
        print(f"    Files ({len(indices)}):")
        for s in stats_in_group:
            print(f"      - {s['filename']}")
        print(f"    Top players: {', '.join(rep_players)}")
        print(f"    Player pool size: {stats_in_group[0]['n_players']}")
        print()

    # ── Per-file detailed table ──────────────────────────────────────────
    print("-" * 90)
    print("  PER-FILE STATISTICS")
    print("-" * 90)
    print()
    print(f"{'#':>3} {'Contest':50s} {'N':>4} {'Pearson':>8} {'Spearman':>9} {'MAE':>7} {'RMSE':>7}")
    print(f"{'':>3} {'':50s} {'':>4} {'(proj)':>8} {'(proj)':>9} {'':>7} {'':>7}")
    print("-" * 90)

    for s in per_file_stats:
        short_name = s['filename'][:50]
        print(f"{s['idx']:3d} {short_name:50s} {s['n_players']:4d} "
              f"{s['pearson']:8.4f} {s['spearman']:9.4f} {s['mae']:7.2f} {s['rmse']:7.2f}")

    print("-" * 90)
    print()

    # ── Ownership projection accuracy per file ───────────────────────────
    print("-" * 90)
    print("  OWNERSHIP PROJECTION ACCURACY (Own Proj vs Actual Ownership)")
    print("-" * 90)
    print()
    print(f"{'#':>3} {'Contest':50s} {'Own Pearson':>12}")
    print("-" * 70)
    for s in per_file_stats:
        short_name = s['filename'][:50]
        own_str = f"{s['own_pearson']:.4f}" if not math.isnan(s['own_pearson']) else "N/A"
        print(f"{s['idx']:3d} {short_name:50s} {own_str:>12}")
    print()

    # ── Contest / All Entrants info ──────────────────────────────────────
    print("-" * 90)
    print("  CONTEST INFORMATION (from All Entrants files)")
    print("-" * 90)
    print()

    for s in per_file_stats:
        ae = s['ae_info']
        print(f"  File: {s['filename']}")
        if ae is None:
            print(f"    (No matching All Entrants file found)")
        else:
            print(f"    Unique entrants:   {ae['total_unique_entrants']:,}")
            print(f"    Total lineups:     {ae['total_lineups']:,}")
            if ae['entry_fee'] is not None:
                print(f"    Entry fee:         ${ae['entry_fee']:.2f}")
                implied_pool = ae['total_lineups'] * ae['entry_fee']
                print(f"    Implied pool:      ${implied_pool:,.0f}")
            print(f"    Entrants cashed:   {ae['cashed_entrants']:,} ({100*ae['cashed_entrants']/ae['total_unique_entrants']:.1f}%)")
            print(f"    Paying places:     through place {ae['max_paying_place']:,}")
            print(f"    Top score:         {ae['top_score']}")
            print(f"    Median score:      {ae['median_score']:.1f}")

            # Show top payout tiers
            tiers = ae['payout_tiers']
            if tiers:
                print(f"    Payout structure (top tiers, gross prize per lineup):")
                shown = 0
                for place in sorted(tiers.keys()):
                    if shown >= 15:
                        remaining = len(tiers) - shown
                        print(f"      ... and {remaining} more tiers down to place {max(tiers.keys()):,}")
                        break
                    print(f"      Place {place:>6,}: ${tiers[place]:>12,.2f}")
                    shown += 1
        print()

    # ── Overall aggregate statistics ─────────────────────────────────────
    N = len(all_fc_proj)
    print("=" * 90)
    print("  AGGREGATE STATISTICS (across ALL files)")
    print("=" * 90)
    print()
    print(f"  Total Player Exposures files analyzed:   {len(per_file_stats)}")
    print(f"  Unique tournaments identified:           {len(tournament_map)}")
    print(f"  Total player-events (rows):              {N:,}")
    print()

    overall_pearson = pearson_r(all_fc_proj, all_fps)
    overall_spearman = spearman_r(all_fc_proj, all_fps)
    overall_mae_val = mae(all_fc_proj, all_fps)
    overall_rmse_val = rmse(all_fc_proj, all_fps)

    print(f"  FC Proj vs Actual FPs:")
    print(f"    Pearson r:           {overall_pearson:.4f}")
    print(f"    Spearman rho:        {overall_spearman:.4f}")
    print(f"    MAE:                 {overall_mae_val:.2f}")
    print(f"    RMSE:                {overall_rmse_val:.2f}")
    print()

    # Summary stats on projections and actuals
    print(f"  FC Proj distribution:")
    print(f"    Mean:    {sum(all_fc_proj)/N:.2f}")
    print(f"    Median:  {median(all_fc_proj):.2f}")
    print(f"    Min:     {min(all_fc_proj):.2f}")
    print(f"    Max:     {max(all_fc_proj):.2f}")
    print(f"    Std:     {math.sqrt(sum((x - sum(all_fc_proj)/N)**2 for x in all_fc_proj)/(N-1)):.2f}")
    print()

    print(f"  Actual FPs distribution:")
    print(f"    Mean:    {sum(all_fps)/N:.2f}")
    print(f"    Median:  {median(all_fps):.2f}")
    print(f"    Min:     {min(all_fps):.2f}")
    print(f"    Max:     {max(all_fps):.2f}")
    print(f"    Std:     {math.sqrt(sum((x - sum(all_fps)/N)**2 for x in all_fps)/(N-1)):.2f}")
    print()

    # Salary vs FPs
    sal_fps_pearson = pearson_r(all_salary, all_fps)
    sal_fps_spearman = spearman_r(all_salary, all_fps)
    print(f"  Salary vs Actual FPs:")
    print(f"    Pearson r:           {sal_fps_pearson:.4f}")
    print(f"    Spearman rho:        {sal_fps_spearman:.4f}")
    print()

    # Ownership projection overall
    own_pearson_all = pearson_r(all_own_proj, all_ownership)
    own_spearman_all = spearman_r(all_own_proj, all_ownership)
    own_mae_all = mae(all_own_proj, all_ownership)
    print(f"  Ownership Proj vs Actual Ownership:")
    print(f"    Pearson r:           {own_pearson_all:.4f}")
    print(f"    Spearman rho:        {own_spearman_all:.4f}")
    print(f"    MAE:                 {own_mae_all:.2f}%")
    print()

    # ── Per-tournament correlations (deduped) ────────────────────────────
    print("-" * 90)
    print("  PER-TOURNAMENT CORRELATIONS (one representative file per tournament)")
    print("-" * 90)
    print()

    dedup_pearsons = []
    dedup_spearmans = []
    dedup_maes = []
    dedup_rmses = []

    for gid in sorted(tournament_map.keys()):
        indices = tournament_map[gid]
        # Pick the file with the most players as representative
        best_idx = max(indices, key=lambda i: per_file_stats[i]['n_players'])
        s = per_file_stats[best_idx]
        dedup_pearsons.append(s['pearson'])
        dedup_spearmans.append(s['spearman'])
        dedup_maes.append(s['mae'])
        dedup_rmses.append(s['rmse'])
        print(f"  Tournament {gid + 1:2d} | Pearson={s['pearson']:.4f}  Spearman={s['spearman']:.4f}  "
              f"MAE={s['mae']:.2f}  RMSE={s['rmse']:.2f}  N={s['n_players']}")
        print(f"               | File: {s['filename']}")
        print(f"               | Top 3: {', '.join(s['players'][:3])}")
        print()

    avg_p = sum(dedup_pearsons) / len(dedup_pearsons)
    avg_s = sum(dedup_spearmans) / len(dedup_spearmans)
    avg_m = sum(dedup_maes) / len(dedup_maes)
    avg_r = sum(dedup_rmses) / len(dedup_rmses)

    print(f"  Average across {len(dedup_pearsons)} unique tournaments:")
    print(f"    Pearson r:     {avg_p:.4f}  (range: {min(dedup_pearsons):.4f} to {max(dedup_pearsons):.4f})")
    print(f"    Spearman rho:  {avg_s:.4f}  (range: {min(dedup_spearmans):.4f} to {max(dedup_spearmans):.4f})")
    print(f"    MAE:           {avg_m:.2f}   (range: {min(dedup_maes):.2f} to {max(dedup_maes):.2f})")
    print(f"    RMSE:          {avg_r:.2f}   (range: {min(dedup_rmses):.2f} to {max(dedup_rmses):.2f})")
    print()

    # ── Comparison with DG probability model ─────────────────────────────
    print("=" * 90)
    print("  COMPARISON: FC PROJECTIONS vs DATA GOLF PROBABILITY MODEL")
    print("=" * 90)
    print()
    print(f"  FC Projections (this analysis):")
    print(f"    Per-tournament avg Pearson:   {avg_p:.4f}")
    print(f"    Per-tournament avg Spearman:  {avg_s:.4f}")
    print(f"    Pooled Pearson:               {overall_pearson:.4f}")
    print(f"    Pooled Spearman:              {overall_spearman:.4f}")
    print()
    print(f"  DG Probability Model (reference benchmark):")
    print(f"    Typical Pearson range:         0.31 - 0.41")
    print(f"    Typical Spearman range:        ~0.30 - 0.40")
    print()
    if avg_p > 0.41:
        print(f"  --> FC projections EXCEED the DG benchmark (Pearson {avg_p:.4f} > 0.41)")
    elif avg_p > 0.31:
        print(f"  --> FC projections are WITHIN the DG benchmark range (Pearson {avg_p:.4f})")
    else:
        print(f"  --> FC projections are BELOW the DG benchmark (Pearson {avg_p:.4f} < 0.31)")
    print()

    # ── Bias analysis ────────────────────────────────────────────────────
    print("-" * 90)
    print("  BIAS ANALYSIS: FC Proj - Actual FPs")
    print("-" * 90)
    print()
    biases = [fc - fp for fc, fp in zip(all_fc_proj, all_fps)]
    mean_bias = sum(biases) / len(biases)
    median_bias = median(biases)
    pct_over = sum(1 for b in biases if b > 0) / len(biases) * 100
    pct_under = sum(1 for b in biases if b < 0) / len(biases) * 100

    print(f"  Mean bias (FC - Actual):    {mean_bias:+.2f} pts")
    print(f"  Median bias:                {median_bias:+.2f} pts")
    print(f"  % over-projected:           {pct_over:.1f}%")
    print(f"  % under-projected:          {pct_under:.1f}%")
    print()

    # Bias by salary tier
    print(f"  Bias by salary tier:")
    tiers = [
        ("$10K+", 10000, 99999),
        ("$9K-$10K", 9000, 9999),
        ("$8K-$9K", 8000, 8999),
        ("$7K-$8K", 7000, 7999),
        ("$6K-$7K", 6000, 6999),
        ("Under $6K", 0, 5999),
    ]
    for label, lo, hi in tiers:
        tier_biases = [fc - fp for fc, sal, fp in zip(all_fc_proj, all_salary, all_fps) if lo <= sal <= hi]
        tier_fc = [fc for fc, sal in zip(all_fc_proj, all_salary) if lo <= sal <= hi]
        tier_fps_list = [fp for fp, sal in zip(all_fps, all_salary) if lo <= sal <= hi]
        if tier_biases:
            tier_mean = sum(tier_biases) / len(tier_biases)
            tier_pr = pearson_r(tier_fc, tier_fps_list) if len(tier_fc) >= 3 else float('nan')
            print(f"    {label:12s}: bias={tier_mean:+6.2f}  n={len(tier_biases):4d}  pearson={tier_pr:.4f}")
    print()

    # ── Value metric analysis ────────────────────────────────────────────
    print("-" * 90)
    print("  FC VALUE vs ACTUAL VALUE (FPs/Salary*1000)")
    print("-" * 90)
    print()

    # Compute actual value
    actual_values = [fp / sal * 1000 if sal > 0 else 0 for fp, sal in zip(all_fps, all_salary)]
    fc_values = [fc / sal * 1000 if sal > 0 else 0 for fc, sal in zip(all_fc_proj, all_salary)]

    val_pearson = pearson_r(fc_values, actual_values)
    val_spearman = spearman_r(fc_values, actual_values)
    print(f"  FC implied value vs actual value:")
    print(f"    Pearson r:     {val_pearson:.4f}")
    print(f"    Spearman rho:  {val_spearman:.4f}")
    print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 90)
    print("  EXECUTIVE SUMMARY")
    print("=" * 90)
    print()
    print(f"  Data scope: {len(per_file_stats)} files, {len(tournament_map)} unique tournaments, {N:,} player-events")
    print()
    print(f"  FC Projection Accuracy (per-tournament avg):")
    print(f"    Pearson r  = {avg_p:.4f}")
    print(f"    Spearman r = {avg_s:.4f}")
    print(f"    MAE        = {avg_m:.2f} DK points")
    print(f"    RMSE       = {avg_r:.2f} DK points")
    print()
    print(f"  Ownership Projection Accuracy:")
    print(f"    Pearson r  = {own_pearson_all:.4f}")
    print(f"    MAE        = {own_mae_all:.2f} percentage points")
    print()
    print(f"  Salary as predictor:  Pearson r = {sal_fps_pearson:.4f}")
    print(f"  Projection bias:      {mean_bias:+.2f} pts (FC tends to {'over' if mean_bias > 0 else 'under'}-project)")
    print()
    print(f"  Note: These files appear to be from DraftKings PGA TOUR contests")
    print(f"  (mostly GPPs / millionaire makers from the 2021-2022 seasons)")
    print(f"  spanning multiple different tournament weeks.")
    print()
    print("=" * 90)
    print("  END OF ANALYSIS")
    print("=" * 90)


if __name__ == '__main__':
    main()

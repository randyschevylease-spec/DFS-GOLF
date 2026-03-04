#!/usr/bin/env python3
"""Analyze which lineups benefit most from wave + course-fit correlation.

Computes a per-lineup correlation score based on same-wave and same-edge
pairwise boosts, then reports top/bottom lineups, wave/edge stacking stats,
and the relationship between correlation and ROI.

Usage:
    python3 analyze_correlation.py <projections_csv> <lineups_csv>
    python3 analyze_correlation.py projections.csv lineups_188375740_150.csv
"""
import csv
import re
import sys
import numpy as np


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <projections_csv> <lineups_csv>")
        sys.exit(1)

    proj_csv = sys.argv[1]
    lineups_csv = sys.argv[2]

    # Parse players
    players = {}
    with open(proj_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dk_id = int(row['dk_id'])
            wave = int(row.get('early_late_wave', 0) or 0)
            dd = float(row.get('DRIVE DIST', 0) or 0)
            da = float(row.get('DRIVE ACC', 0) or 0)
            if dd > 0 and dd >= da:
                edge = 'dist'
            elif da > 0:
                edge = 'acc'
            else:
                edge = 'base'
            players[dk_id] = {'name': row['dk_name'].strip(), 'wave': wave, 'edge': edge}

    # Parse lineups
    lineups = []
    with open(lineups_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 12:
                continue
            ids = []
            for cell in row[:6]:
                m = re.search(r'\((\d+)\)', cell)
                if m:
                    ids.append(int(m.group(1)))
            if len(ids) != 6:
                continue
            roi = float(row[9])
            cash = float(row[10])
            proj = float(row[7])
            ceil = float(row[8])

            waves = [players[i]['wave'] for i in ids]
            edges = [players[i]['edge'] for i in ids]
            names = [players[i]['name'] for i in ids]

            # Count same-wave pairs (out of 15 possible), excluding wave=0
            wave_pairs = sum(1 for a in range(6) for b in range(a+1, 6)
                             if waves[a] == waves[b] and waves[a] != 0)
            # Count same-edge pairs
            edge_pairs = sum(1 for a in range(6) for b in range(a+1, 6)
                             if edges[a] == edges[b])

            # Correlation score: wave_pairs * 0.2 + edge_pairs * 0.1 (total pairwise boost)
            corr_score = wave_pairs * 0.2 + edge_pairs * 0.1

            am_ct = sum(1 for w in waves if w == 1)
            pm_ct = sum(1 for w in waves if w == 2)
            dist_ct = edges.count('dist')
            acc_ct = edges.count('acc')
            base_ct = edges.count('base')

            lineups.append({
                'names': names, 'roi': roi, 'cash': cash, 'proj': proj, 'ceil': ceil,
                'wave_pairs': wave_pairs, 'edge_pairs': edge_pairs, 'corr_score': corr_score,
                'am': am_ct, 'pm': pm_ct, 'dist': dist_ct, 'acc': acc_ct, 'base': base_ct,
            })

    if not lineups:
        print("No lineups parsed.")
        sys.exit(1)

    # Sort by correlation score
    lineups.sort(key=lambda x: -x['corr_score'])
    show = min(15, len(lineups))

    print(f'TOP {show} LINEUPS BY CORRELATION BENEFIT')
    print(f'{"#":>3} {"CorrScr":>7} {"WvPrs":>5} {"EdPrs":>5} {"AM/PM":>5} {"D/A/B":>7} {"ROI%":>7} {"Cash%":>6} {"Ceil":>6}  Players')
    print(f'{"─"*3} {"─"*7} {"─"*5} {"─"*5} {"─"*5} {"─"*7} {"─"*7} {"─"*6} {"─"*6}  {"─"*55}')
    for i, lu in enumerate(lineups[:show]):
        names_short = ', '.join(n.split()[-1] for n in lu['names'])
        print(f'{i+1:>3} {lu["corr_score"]:>7.1f} {lu["wave_pairs"]:>5} {lu["edge_pairs"]:>5} '
              f'{lu["am"]}/{lu["pm"]:>1} {lu["dist"]}/{lu["acc"]}/{lu["base"]} '
              f'{lu["roi"]:>+7.1f} {lu["cash"]:>5.1f}% {lu["ceil"]:>6.1f}  {names_short}')

    print()
    print(f'BOTTOM {show} LINEUPS BY CORRELATION BENEFIT (most diversified)')
    print(f'{"#":>3} {"CorrScr":>7} {"WvPrs":>5} {"EdPrs":>5} {"AM/PM":>5} {"D/A/B":>7} {"ROI%":>7} {"Cash%":>6} {"Ceil":>6}  Players')
    print(f'{"─"*3} {"─"*7} {"─"*5} {"─"*5} {"─"*5} {"─"*7} {"─"*7} {"─"*6} {"─"*6}  {"─"*55}')
    for i, lu in enumerate(lineups[-show:]):
        names_short = ', '.join(n.split()[-1] for n in lu['names'])
        idx = len(lineups) - show + i
        print(f'{idx+1:>3} {lu["corr_score"]:>7.1f} {lu["wave_pairs"]:>5} {lu["edge_pairs"]:>5} '
              f'{lu["am"]}/{lu["pm"]:>1} {lu["dist"]}/{lu["acc"]}/{lu["base"]} '
              f'{lu["roi"]:>+7.1f} {lu["cash"]:>5.1f}% {lu["ceil"]:>6.1f}  {names_short}')

    print()

    # Summary stats
    corr_scores = [lu['corr_score'] for lu in lineups]
    rois = [lu['roi'] for lu in lineups]
    ceils = [lu['ceil'] for lu in lineups]
    print(f'Correlation score range: {min(corr_scores):.1f} – {max(corr_scores):.1f}')
    print(f'Mean: {np.mean(corr_scores):.1f} | Median: {np.median(corr_scores):.1f}')

    corr_roi = np.corrcoef(corr_scores, rois)[0, 1]
    corr_ceil = np.corrcoef(corr_scores, ceils)[0, 1]
    print(f'Correlation(corr_score, ROI%): {corr_roi:+.3f}')
    print(f'Correlation(corr_score, Ceiling): {corr_ceil:+.3f}')

    # Wave stacking
    all_am = [lu for lu in lineups if lu['am'] == 6]
    all_pm = [lu for lu in lineups if lu['pm'] == 6]
    five_plus_am = [lu for lu in lineups if lu['am'] >= 5]
    five_plus_pm = [lu for lu in lineups if lu['pm'] >= 5]
    mixed = [lu for lu in lineups if lu['am'] == 3 and lu['pm'] == 3]
    print(f'\nWave stacking:')
    print(f'  All-AM (6/0): {len(all_am)} lineups', end='')
    if all_am:
        print(f' | avg ROI: {np.mean([lu["roi"] for lu in all_am]):+.1f}%')
    else:
        print()
    print(f'  All-PM (0/6): {len(all_pm)} lineups', end='')
    if all_pm:
        print(f' | avg ROI: {np.mean([lu["roi"] for lu in all_pm]):+.1f}%')
    else:
        print()
    if five_plus_am:
        print(f'  5+ AM: {len(five_plus_am)} lineups | avg ROI: {np.mean([lu["roi"] for lu in five_plus_am]):+.1f}%')
    if five_plus_pm:
        print(f'  5+ PM: {len(five_plus_pm)} lineups | avg ROI: {np.mean([lu["roi"] for lu in five_plus_pm]):+.1f}%')
    if mixed:
        print(f'  3/3 split: {len(mixed)} lineups | avg ROI: {np.mean([lu["roi"] for lu in mixed]):+.1f}%')

    # Edge stacking
    all_dist = [lu for lu in lineups if lu['dist'] >= 5]
    all_acc = [lu for lu in lineups if lu['acc'] >= 5]
    print(f'\nEdge stacking:')
    if all_dist:
        print(f'  5+ driving_dist: {len(all_dist)} lineups | avg ROI: {np.mean([lu["roi"] for lu in all_dist]):+.1f}%')
    else:
        print('  5+ driving_dist: 0 lineups')
    if all_acc:
        print(f'  5+ driving_acc: {len(all_acc)} lineups | avg ROI: {np.mean([lu["roi"] for lu in all_acc]):+.1f}%')
    else:
        print('  5+ driving_acc: 0 lineups')


if __name__ == "__main__":
    main()

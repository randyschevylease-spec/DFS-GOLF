#!/usr/bin/env python3
"""Player Score Pre-Simulation — Standalone Data Asset Generator.

Generates N correlated player score scenarios using the same Cholesky + mixture
distribution pipeline as the contest simulator, but stores them as a persistent
numpy array for downstream analysis and consumption.

The output is a (N, n_players) float32 matrix where each row is one complete
"state of the world" — all players' scores in a single tournament scenario.
Correlations between players (wave-aware) and bimodal miss/make-cut distributions
are baked in.

Usage:
    python3 presim.py                          # default 100K scenarios
    python3 presim.py --scenarios 500000       # 500K scenarios
    python3 presim.py --scenarios 1500000      # 1.5M for full 150K-entry 10x
    python3 presim.py --stats                  # print per-player distribution stats
    python3 presim.py --query "Ryan Gerard"    # inspect one player's distribution
"""
import sys
import os
import csv
import time
import argparse
import numpy as np
from scipy.stats import norm as sp_norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SAME_WAVE_CORRELATION, DIFF_WAVE_CORRELATION
from datagolf_client import get_predictions
from engine import _get_sigma, compute_mixture_params, transform_mixture_scores

# ── Defaults ──────────────────────────────────────────────────────────────
CSV_PATH = "/Users/rhbot/Downloads/draftkings_main_projections (2).csv"
CEILING_FILTER = 110
DEFAULT_SCENARIOS = 100_000
BATCH_SIZE = 5000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_players(csv_path):
    """Parse projections CSV. Returns list of player dicts."""
    players = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                player = {
                    "name": row["dk_name"].strip(),
                    "dk_id": int(row["dk_id"]),
                    "salary": int(row["dk_salary"]),
                    "projected_points": float(row["total_points"]),
                    "std_dev": float(row["std_dev"]),
                    "wave": int(row["early_late_wave"]),
                    "floor": float(row["FLOOR"]),
                    "ceiling": float(row["CEILING"]),
                    "proj_ownership": float(row["projected_ownership"]),
                }
                players.append(player)
            except (ValueError, KeyError) as e:
                print(f"  Warning: skipping {row.get('dk_name', '?')}: {e}")
    return players


def fetch_make_cut_probs(players):
    """Fetch P(make_cut) from DataGolf and match to players."""
    import re

    def _normalize(name):
        n = name.strip()
        n = re.sub(r'\s+(III|II|IV|Jr\.?|Sr\.?)\s*$', '', n)
        n = re.sub(r'\s+[A-Z]\.\s+', ' ', n)
        n = re.sub(r'\s+[A-Z]\s+', ' ', n)
        n = n.replace('-', '').replace('.', '').replace("'", '')
        n = n.lower().strip()
        n = re.sub(r'\s+', ' ', n)
        return n

    matched = 0
    try:
        predictions = get_predictions()
        mc_lookup = {}
        for entry in predictions.get("baseline", []):
            raw_name = entry.get("player_name", "")
            mc_val = entry.get("make_cut", 0)
            if mc_val and mc_val > 0:
                if ", " in raw_name:
                    parts = raw_name.split(", ", 1)
                    clean_name = f"{parts[1]} {parts[0]}"
                else:
                    clean_name = raw_name
                mc_lookup[clean_name] = mc_val / 100.0 if mc_val > 1 else mc_val

        # Build normalized + last-name lookups
        norm_lookup = {}
        last_name_lookup = {}
        for dg_name, mc_val in mc_lookup.items():
            normed = _normalize(dg_name)
            norm_lookup[normed] = mc_val
            last = normed.split()[-1] if normed.split() else ""
            last_name_lookup.setdefault(last, []).append((dg_name, mc_val))

        for p in players:
            mc = mc_lookup.get(p["name"], 0)
            if mc > 0:
                p["p_make_cut"] = mc
                matched += 1
                continue

            p_norm = _normalize(p["name"])
            mc = norm_lookup.get(p_norm, 0)
            if mc > 0:
                p["p_make_cut"] = mc
                matched += 1
                continue

            p_last = p_norm.split()[-1] if p_norm.split() else ""
            candidates_ln = last_name_lookup.get(p_last, [])
            if len(candidates_ln) == 1:
                p["p_make_cut"] = candidates_ln[0][1]
                matched += 1
                continue

            if len(candidates_ln) > 1:
                p_first = p_norm.split()[0] if p_norm.split() else ""
                for dg_name, mc_val in candidates_ln:
                    dg_first = _normalize(dg_name).split()[0]
                    if (p_first.startswith(dg_first) or dg_first.startswith(p_first)) and len(min(p_first, dg_first)) >= 2:
                        p["p_make_cut"] = mc_val
                        matched += 1
                        break
                if p.get("p_make_cut", 0) > 0:
                    continue

            p["p_make_cut"] = 0

        print(f"  Make-cut matched: {matched}/{len(players)} players")
    except Exception as e:
        print(f"  Warning: Could not fetch make_cut data: {e}")
        for p in players:
            p["p_make_cut"] = 0
    return matched


def build_covariance(players):
    """Build wave-aware covariance matrix and Cholesky factor."""
    n = len(players)
    sigmas = np.array([_get_sigma(p) for p in players], dtype=np.float64)
    waves = np.array([p["wave"] for p in players])

    same_wave = (waves[:, None] == waves[None, :])
    corr = np.where(same_wave, SAME_WAVE_CORRELATION, DIFF_WAVE_CORRELATION)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(sigmas, sigmas) * corr

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += np.eye(n) * 1.0
        L = np.linalg.cholesky(cov)

    return sigmas, L


def generate_presim(players, n_scenarios, sigmas, L, mixture_params, seed=None):
    """Generate correlated player score scenarios.

    Returns:
        scores: (n_scenarios, n_players) float32 array
    """
    n_players = len(players)
    means = np.array([p["projected_points"] for p in players], dtype=np.float64)

    use_mix = (mixture_params is not None and mixture_params[5].any())
    if use_mix:
        p_miss, mu_miss, sigma_miss, mu_make, sigma_make, mix_flag = mixture_params
        n_mix = int(mix_flag.sum())
        print(f"  Mixture: {n_mix}/{n_players} players with bimodal scores")

    scores = np.empty((n_scenarios, n_players), dtype=np.float32)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    for batch_start in range(0, n_scenarios, BATCH_SIZE):
        bs = min(BATCH_SIZE, n_scenarios - batch_start)
        Z = rng.standard_normal((bs, n_players))
        X = Z @ L.T

        if use_mix:
            batch_scores = transform_mixture_scores(
                X, sigmas, p_miss, mu_miss, sigma_miss,
                mu_make, sigma_make, mix_flag)
        else:
            batch_scores = means[None, :] + X
            np.maximum(batch_scores, 0.0, out=batch_scores)

        scores[batch_start:batch_start + bs] = batch_scores.astype(np.float32)

        done = batch_start + bs
        if done % 50000 == 0 or done == n_scenarios:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    {done:>10,} / {n_scenarios:,} scenarios  "
                  f"({elapsed:.1f}s, {rate:,.0f}/s)")

    return scores


def compute_player_stats(scores, players):
    """Compute per-player distribution statistics from pre-simulated scores.

    Returns list of dicts with distribution characteristics per player.
    """
    n_scenarios = scores.shape[0]
    stats = []
    for i, p in enumerate(players):
        s = scores[:, i]
        pct = np.percentile(s, [1, 5, 10, 25, 50, 75, 90, 95, 99])

        # Bimodality detection: score < 40 is likely a miss-cut scenario
        miss_cut_rate = float((s < 40).mean())
        make_cut_scores = s[s >= 40]
        miss_cut_scores = s[s < 40]

        stat = {
            "name": p["name"],
            "salary": p["salary"],
            "proj": p["projected_points"],
            "ceiling_csv": p["ceiling"],
            "floor_csv": p["floor"],
            "ownership": p["proj_ownership"],
            "p_make_cut_dg": p.get("p_make_cut", 0),
            # Empirical distribution
            "sim_mean": float(s.mean()),
            "sim_std": float(s.std()),
            "sim_skew": float(((s - s.mean()) ** 3).mean() / max(s.std() ** 3, 1e-9)),
            "sim_kurtosis": float(((s - s.mean()) ** 4).mean() / max(s.std() ** 4, 1e-9) - 3),
            # Quantiles
            "p01": float(pct[0]),
            "p05": float(pct[1]),
            "p10": float(pct[2]),
            "p25": float(pct[3]),
            "p50": float(pct[4]),
            "p75": float(pct[5]),
            "p90": float(pct[6]),
            "p95": float(pct[7]),
            "p99": float(pct[8]),
            # Tail frequencies
            "miss_cut_rate": miss_cut_rate,
            "prob_80_plus": float((s >= 80).mean()),
            "prob_90_plus": float((s >= 90).mean()),
            "prob_100_plus": float((s >= 100).mean()),
            "sim_max": float(s.max()),
            "sim_min": float(s.min()),
            # Conditional stats
            "make_cut_mean": float(make_cut_scores.mean()) if len(make_cut_scores) > 0 else 0,
            "miss_cut_mean": float(miss_cut_scores.mean()) if len(miss_cut_scores) > 0 else 0,
        }
        stats.append(stat)
    return stats


def print_stats_table(stats):
    """Print formatted per-player distribution table."""
    print(f"\n  {'Player':<26} {'Proj':>5} {'SimMu':>6} {'SimSD':>6} {'Skew':>6} "
          f"{'P10':>5} {'P50':>5} {'P90':>5} {'P99':>5} "
          f"{'MC%':>5} {'80+':>5} {'90+':>5} {'100+':>5} {'Max':>5}")
    print(f"  {'-'*26} {'-'*5} {'-'*6} {'-'*6} {'-'*6} "
          f"{'-'*5} {'-'*5} {'-'*5} {'-'*5} "
          f"{'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")

    for s in stats:
        mc_pct = (1 - s["miss_cut_rate"]) * 100
        print(f"  {s['name']:<26} {s['proj']:>5.1f} {s['sim_mean']:>6.1f} {s['sim_std']:>6.1f} "
              f"{s['sim_skew']:>6.2f} "
              f"{s['p10']:>5.1f} {s['p50']:>5.1f} {s['p90']:>5.1f} {s['p99']:>5.1f} "
              f"{mc_pct:>5.1f} {s['prob_80_plus']*100:>5.1f} {s['prob_90_plus']*100:>5.1f} "
              f"{s['prob_100_plus']*100:>5.1f} {s['sim_max']:>5.0f}")


def print_player_detail(scores, players, player_name):
    """Print detailed distribution analysis for a single player."""
    # Find player by partial name match
    idx = None
    for i, p in enumerate(players):
        if player_name.lower() in p["name"].lower():
            idx = i
            break
    if idx is None:
        print(f"  Player '{player_name}' not found.")
        return

    p = players[idx]
    s = scores[:, idx]
    n = len(s)

    print(f"\n  {'='*60}")
    print(f"  {p['name']} — ${p['salary']:,} — Wave {'AM' if p['wave']==1 else 'PM'}")
    print(f"  {'='*60}")
    print(f"  CSV projection: {p['projected_points']:.1f}")
    print(f"  CSV floor/ceiling: {p['floor']:.1f} / {p['ceiling']:.1f}")
    print(f"  CSV ownership: {p['proj_ownership']:.1f}%")
    print(f"  DG P(make cut): {p.get('p_make_cut', 0)*100:.1f}%")

    print(f"\n  Simulated Distribution ({n:,} scenarios)")
    print(f"  {'─'*40}")
    print(f"  Mean:     {s.mean():.2f}")
    print(f"  Std Dev:  {s.std():.2f}")
    skew = float(((s - s.mean()) ** 3).mean() / max(s.std() ** 3, 1e-9))
    kurt = float(((s - s.mean()) ** 4).mean() / max(s.std() ** 4, 1e-9) - 3)
    print(f"  Skewness: {skew:.3f}  ({'left-skewed' if skew < -0.3 else 'right-skewed' if skew > 0.3 else 'symmetric'})")
    print(f"  Kurtosis: {kurt:.3f}  ({'heavy tails' if kurt > 1 else 'light tails' if kurt < -0.5 else 'normal tails'})")

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(s, pcts)
    print(f"\n  Quantiles")
    print(f"  {'─'*40}")
    for pct, val in zip(pcts, vals):
        bar = '█' * max(0, int(val / 2))
        print(f"  P{pct:>2}: {val:>7.1f}  {bar}")

    print(f"\n  Tail Frequencies")
    print(f"  {'─'*40}")
    thresholds = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    for t in thresholds:
        count = int((s >= t).sum())
        pct = count / n * 100
        label = f"  >= {t:>3}: {count:>7,} ({pct:>6.2f}%)"
        if t == 40:
            label += "  ← make cut threshold"
        if t == 80:
            label += "  ← GPP contention"
        if t == 100:
            label += "  ← GPP ceiling"
        print(label)

    # Bimodal analysis
    miss = s[s < 40]
    make = s[s >= 40]
    print(f"\n  Bimodal Split")
    print(f"  {'─'*40}")
    print(f"  Miss cut (<40): {len(miss):,} scenarios ({len(miss)/n*100:.1f}%)")
    if len(miss) > 0:
        print(f"    Mean: {miss.mean():.1f} | Std: {miss.std():.1f} | Range: {miss.min():.1f} – {miss.max():.1f}")
    print(f"  Make cut (>=40): {len(make):,} scenarios ({len(make)/n*100:.1f}%)")
    if len(make) > 0:
        print(f"    Mean: {make.mean():.1f} | Std: {make.std():.1f} | Range: {make.min():.1f} – {make.max():.1f}")

    # ASCII histogram
    print(f"\n  Score Distribution (histogram)")
    print(f"  {'─'*40}")
    bins = np.arange(0, max(160, int(s.max()) + 10), 5)
    hist, _ = np.histogram(s, bins=bins)
    max_count = hist.max()
    bar_width = 50
    for i in range(len(hist)):
        lo, hi = bins[i], bins[i + 1]
        bar_len = int(hist[i] / max_count * bar_width) if max_count > 0 else 0
        bar = '█' * bar_len
        if hist[i] > 0:
            print(f"  {lo:>5.0f}-{hi:>3.0f} │{bar} {hist[i]:,}")


def save_asset(scores, players, output_dir):
    """Save presim scores + player metadata as a numpy archive."""
    path = os.path.join(output_dir, "presim_scores.npz")
    names = [p["name"] for p in players]
    salaries = np.array([p["salary"] for p in players], dtype=np.int32)
    projections = np.array([p["projected_points"] for p in players], dtype=np.float32)
    ownerships = np.array([p["proj_ownership"] for p in players], dtype=np.float32)
    ceilings = np.array([p["ceiling"] for p in players], dtype=np.float32)
    floors = np.array([p["floor"] for p in players], dtype=np.float32)
    waves = np.array([p["wave"] for p in players], dtype=np.int8)
    dk_ids = np.array([p["dk_id"] for p in players], dtype=np.int64)
    make_cut = np.array([p.get("p_make_cut", 0) for p in players], dtype=np.float32)

    np.savez_compressed(
        path,
        scores=scores,
        names=names,
        salaries=salaries,
        projections=projections,
        ownerships=ownerships,
        ceilings=ceilings,
        floors=floors,
        waves=waves,
        dk_ids=dk_ids,
        make_cut=make_cut,
    )
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved: {path} ({size_mb:.1f} MB)")
    print(f"  Shape: {scores.shape} ({scores.shape[0]:,} scenarios × {scores.shape[1]} players)")
    return path


def load_asset(path=None):
    """Load a saved presim asset. Returns (scores, player_meta dict)."""
    if path is None:
        path = os.path.join(OUTPUT_DIR, "presim_scores.npz")
    data = np.load(path, allow_pickle=True)
    meta = {
        "names": list(data["names"]),
        "salaries": data["salaries"],
        "projections": data["projections"],
        "ownerships": data["ownerships"],
        "ceilings": data["ceilings"],
        "floors": data["floors"],
        "waves": data["waves"],
        "dk_ids": data["dk_ids"],
        "make_cut": data["make_cut"],
    }
    return data["scores"], meta


def main():
    parser = argparse.ArgumentParser(description="Player Score Pre-Simulation")
    parser.add_argument("--scenarios", type=int, default=DEFAULT_SCENARIOS,
                        help=f"Number of score scenarios to generate (default: {DEFAULT_SCENARIOS:,})")
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Path to projections CSV")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--stats", action="store_true", help="Print per-player distribution stats")
    parser.add_argument("--query", type=str, default=None, help="Detailed analysis for one player")
    parser.add_argument("--load", action="store_true", help="Load existing presim instead of generating")
    parser.add_argument("--no-save", action="store_true", help="Don't save to disk (analysis only)")
    args = parser.parse_args()

    print("=" * 70)
    print("  PLAYER SCORE PRE-SIMULATION")
    print("=" * 70)

    if args.load:
        print(f"\n  Loading existing presim asset...")
        scores, meta = load_asset()
        print(f"  Loaded: {scores.shape[0]:,} scenarios × {scores.shape[1]} players")

        # Rebuild minimal player dicts for stats/query
        players = []
        for i, name in enumerate(meta["names"]):
            players.append({
                "name": name,
                "salary": int(meta["salaries"][i]),
                "projected_points": float(meta["projections"][i]),
                "ceiling": float(meta["ceilings"][i]),
                "floor": float(meta["floors"][i]),
                "proj_ownership": float(meta["ownerships"][i]),
                "wave": int(meta["waves"][i]),
                "p_make_cut": float(meta["make_cut"][i]),
                "dk_id": int(meta["dk_ids"][i]),
            })
    else:
        # ── Parse players ──
        print(f"\n  Parsing players from {args.csv}")
        all_players = parse_players(args.csv)
        players = [p for p in all_players if p["ceiling"] >= CEILING_FILTER]
        print(f"  {len(all_players)} total → {len(players)} after CEILING >= {CEILING_FILTER} filter")

        # ── Fetch make-cut probabilities ──
        print(f"\n  Fetching DataGolf make-cut probabilities...")
        fetch_make_cut_probs(players)

        # ── Build covariance ──
        print(f"\n  Building wave-aware covariance matrix...")
        sigmas, L = build_covariance(players)
        mixture_params = compute_mixture_params(players)
        n_mix = int(mixture_params[5].sum())
        print(f"  {n_mix}/{len(players)} players with mixture distributions")

        # ── Generate scenarios ──
        mem_mb = args.scenarios * len(players) * 4 / 1024 / 1024
        print(f"\n  Generating {args.scenarios:,} scenarios for {len(players)} players...")
        print(f"  Memory: {mem_mb:.0f} MB (float32)")
        t0 = time.time()
        scores = generate_presim(players, args.scenarios, sigmas, L, mixture_params, seed=args.seed)
        elapsed = time.time() - t0
        print(f"  Complete in {elapsed:.1f}s ({args.scenarios/elapsed:,.0f} scenarios/sec)")

        # ── Save ──
        if not args.no_save:
            print(f"\n  Saving presim asset...")
            save_asset(scores, players, OUTPUT_DIR)

    # ── Stats ──
    if args.stats or args.query:
        stats = compute_player_stats(scores, players)

    if args.stats:
        # Sort by projection descending
        stats.sort(key=lambda s: s["proj"], reverse=True)
        print_stats_table(stats)

    if args.query:
        print_player_detail(scores, players, args.query)

    # Always print summary correlation check
    if not args.load or args.stats:
        print(f"\n  Correlation spot check (top 5 by projection):")
        top5 = sorted(range(len(players)), key=lambda i: players[i]["projected_points"], reverse=True)[:5]
        print(f"  {'':>20}", end="")
        for i in top5:
            print(f"  {players[i]['name'][:8]:>8}", end="")
        print()
        for i in top5:
            print(f"  {players[i]['name'][:20]:<20}", end="")
            for j in top5:
                r = np.corrcoef(scores[:, i].astype(np.float64), scores[:, j].astype(np.float64))[0, 1]
                print(f"  {r:>8.3f}", end="")
            print()

    print(f"\n{'='*70}")
    print(f"  Done.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

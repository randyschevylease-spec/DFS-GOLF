#!/usr/bin/env python3
"""DFS Golf Backtester — Validate and tune projections against historical results.

Usage:
    python backtest.py                    # Backtest last 20 events
    python backtest.py --events 50        # Backtest last 50 events
    python backtest.py --tune             # Backtest + tune model weights
    python backtest.py --year 2024        # Backtest only 2024 events
"""
import sys
import os
import json
import time
import argparse
import math
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datagolf_client import _get

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history", "cache")

# Default backtest projection weights (legacy, for backtest tuning only)
WEIGHT_DATAGOLF = 0.50
WEIGHT_COURSE_HISTORY = 0.15
WEIGHT_DK_AVG = 0.25
WEIGHT_SALARY_VALUE = 0.10

# ── DraftKings Finish Bonus Scoring ──────────────────────────────────────────

FINISH_BONUS = {
    1: 30, 2: 20, 3: 18, 4: 16, 5: 14,
    6: 12, 7: 10, 8: 9, 9: 8, 10: 7,
}
for _i in range(11, 16):
    FINISH_BONUS[_i] = 6
for _i in range(16, 21):
    FINISH_BONUS[_i] = 5
for _i in range(21, 26):
    FINISH_BONUS[_i] = 4
for _i in range(26, 31):
    FINISH_BONUS[_i] = 3
for _i in range(31, 41):
    FINISH_BONUS[_i] = 2
for _i in range(41, 51):
    FINISH_BONUS[_i] = 1


def estimate_finish_distribution(p_win, p_top5, p_top10, p_top20, p_make_cut):
    """Estimate probability of finishing in each position 1-50+ from DataGolf odds."""
    dist = {}
    p_win = min(p_win, p_top5) if p_top5 > 0 else p_win
    p_top5 = min(p_top5, p_top10) if p_top10 > 0 else p_top5
    p_top10 = min(p_top10, p_top20) if p_top20 > 0 else p_top10
    p_top20 = min(p_top20, p_make_cut) if p_make_cut > 0 else p_top20
    dist[1] = p_win
    for pos in range(2, 6):
        dist[pos] = max(0, p_top5 - p_win) / 4.0
    for pos in range(6, 11):
        dist[pos] = max(0, p_top10 - p_top5) / 5.0
    for pos in range(11, 21):
        dist[pos] = max(0, p_top20 - p_top10) / 10.0
    for pos in range(21, 51):
        dist[pos] = max(0, p_make_cut - p_top20) / 30.0
    dist["MC"] = max(0, 1.0 - p_make_cut)
    return dist


def expected_finish_bonus(dist):
    """Calculate expected DraftKings finish bonus points from a finish distribution."""
    ev = 0.0
    for pos, prob in dist.items():
        if pos == "MC":
            continue
        ev += prob * FINISH_BONUS.get(pos, 0)
    return ev


def estimate_per_hole_points(p_make_cut, dk_avg_points):
    """Estimate expected scoring points (excluding finish bonus) via piecewise interpolation."""
    anchors = [
        (0.00, 19.0), (0.05, 21.0), (0.10, 27.0), (0.15, 34.0), (0.20, 32.0),
        (0.25, 39.0), (0.30, 43.0), (0.35, 42.0), (0.40, 52.0), (0.45, 51.0),
        (0.50, 53.0), (0.55, 58.0), (0.60, 60.0), (0.65, 62.0), (0.70, 64.0),
        (0.75, 67.0), (0.80, 68.0), (0.85, 68.0), (0.90, 72.0), (0.95, 73.0),
        (1.00, 73.0),
    ]
    mc = max(0.0, min(1.0, p_make_cut))
    model_estimate = anchors[-1][1]
    for i in range(len(anchors) - 1):
        x0, y0 = anchors[i]
        x1, y1 = anchors[i + 1]
        if x0 <= mc <= x1:
            t = (mc - x0) / (x1 - x0) if x1 != x0 else 0
            model_estimate = y0 + t * (y1 - y0)
            break
    if dk_avg_points and dk_avg_points > 0:
        dk_scoring_estimate = dk_avg_points * 0.80
        scoring_points = 0.50 * model_estimate + 0.50 * dk_scoring_estimate
    else:
        scoring_points = model_estimate
    return scoring_points


# ── DraftKings Fantasy Point Calculator ──────────────────────────────────────

def calc_dk_fantasy_points(player_rounds, fin_text):
    """Calculate exact DraftKings fantasy points from round-level scoring data.

    DK PGA Scoring:
        Eagle: +8, Birdie: +3, Par: +0.5, Bogey: -0.5, Double+: -1
        Streak 3+ birdies (per round): +3
        Bogey-free round: +3
        All 4 rounds under 70: +5
        Finish: 1st=30, 2nd=20, 3rd=18, ..., 41-50=1
    """
    total = 0.0
    rounds_played = 0
    all_under_70 = True
    round_scores = []

    for rd_key in ["round_1", "round_2", "round_3", "round_4"]:
        rd = player_rounds.get(rd_key)
        if rd is None or rd.get("score") is None:
            all_under_70 = False
            continue

        rounds_played += 1
        score = rd["score"]
        birdies = rd.get("birdies", 0) or 0
        eagles = rd.get("eagles_or_better", 0) or 0
        pars = rd.get("pars", 0) or 0
        bogeys = rd.get("bogies", 0) or 0
        doubles = rd.get("doubles_or_worse", 0) or 0

        # Per-hole scoring
        total += eagles * 8.0
        total += birdies * 3.0
        total += pars * 0.5
        total += bogeys * (-0.5)
        total += doubles * (-1.0)

        # Bogey-free round bonus
        if bogeys == 0 and doubles == 0:
            total += 3.0

        # Birdie streak bonus (we don't have hole-by-hole, so estimate)
        # With N birdies in 18 holes, probability of a 3+ streak ≈
        # We'll approximate: if 5+ birdies, likely had a streak
        if birdies >= 5:
            total += 3.0
        elif birdies >= 3:
            # ~40% chance of streak with 3-4 birdies
            total += 1.2

        if score >= 70:
            all_under_70 = False
        round_scores.append(score)

    # All 4 rounds under 70 bonus
    if rounds_played == 4 and all_under_70:
        total += 5.0

    # Finish position bonus
    finish_bonus = _parse_finish_bonus(fin_text)
    total += finish_bonus

    return round(total, 1), rounds_played


def _parse_finish_bonus(fin_text):
    """Parse finish text and return DK finish bonus points."""
    if not fin_text or fin_text in ("CUT", "WD", "DQ", "MDF"):
        return 0.0

    try:
        pos_str = fin_text.replace("T", "")
        pos = int(pos_str)
    except (ValueError, TypeError):
        return 0.0

    return float(FINISH_BONUS.get(pos, 0))


# ── Data Fetching with Cache ─────────────────────────────────────────────────

def _cache_path(endpoint, params):
    """Generate a cache file path for an API call."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{endpoint}_{params.get('event_id', '')}_{params.get('year', '')}"
    key = key.replace("/", "_").replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{key}.json")


def _cached_get(endpoint, params):
    """Fetch from API with local file cache to avoid repeated calls."""
    path = _cache_path(endpoint, params)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    data = _get(endpoint, params)
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def fetch_event_predictions(event_id, year):
    """Fetch archived pre-tournament predictions for a past event."""
    return _cached_get("/preds/pre-tournament-archive", {
        "tour": "pga",
        "event_id": str(event_id),
        "year": str(year),
        "odds_format": "percent",
    })


def fetch_event_rounds(event_id, year):
    """Fetch round-level scoring data for a past event."""
    return _cached_get("/historical-raw-data/rounds", {
        "tour": "pga",
        "event_id": str(event_id),
        "year": str(year),
    })


def get_backtest_events(year=None, max_events=None):
    """Get list of events suitable for backtesting.

    Filters out team events, Q-School, and non-standard tournaments.
    """
    events = _cached_get("/historical-raw-data/event-list", {"tour": "pga"})

    # Filter to requested year(s)
    if year:
        events = [e for e in events if e["calendar_year"] == year]
    else:
        events = [e for e in events if e["calendar_year"] in (2024, 2025)]

    # Filter out non-standard events
    skip_keywords = ["q-school", "zurich classic", "team", "olympic", "tour championship"]
    filtered = []
    for e in events:
        name_lower = e["event_name"].lower()
        if any(kw in name_lower for kw in skip_keywords):
            continue
        filtered.append(e)

    # Sort by date descending (most recent first)
    filtered.sort(key=lambda e: e["date"], reverse=True)

    if max_events:
        filtered = filtered[:max_events]

    return filtered


# ── Projection Model (parameterized for tuning) ─────────────────────────────

def project_player_backtest(preds, dk_avg_proxy=None, weights=None):
    """Run projection model for a single player during backtesting.

    preds: dict with win, top_5, top_10, top_20, make_cut (as 0-1 probabilities)
    dk_avg_proxy: estimated DK avg points (from scoring data proxy)
    weights: dict with w_dg, w_course, w_dk, w_sal weight overrides

    Returns projected fantasy points.
    """
    if weights is None:
        weights = {
            "w_dg": WEIGHT_DATAGOLF,
            "w_course": WEIGHT_COURSE_HISTORY,
            "w_dk": WEIGHT_DK_AVG,
            "w_sal": WEIGHT_SALARY_VALUE,
        }

    p_win = preds.get("win", 0) or 0
    p_top5 = preds.get("top_5", 0) or 0
    p_top10 = preds.get("top_10", 0) or 0
    p_top20 = preds.get("top_20", 0) or 0
    p_make_cut = preds.get("make_cut", 0) or 0

    # Ensure probabilities are 0-1 range
    if p_make_cut > 1:
        p_win /= 100; p_top5 /= 100; p_top10 /= 100; p_top20 /= 100; p_make_cut /= 100

    dist = estimate_finish_distribution(p_win, p_top5, p_top10, p_top20, p_make_cut)
    e_finish = expected_finish_bonus(dist)
    e_scoring = estimate_per_hole_points(p_make_cut, dk_avg_proxy)

    dg_estimate = e_finish + e_scoring
    dk_estimate = dk_avg_proxy if dk_avg_proxy and dk_avg_proxy > 0 else dg_estimate

    # Blend (no course history or salary data in backtest — redistribute those weights)
    w_dg = weights["w_dg"] + weights["w_course"] + weights["w_sal"]  # absorb unavailable weights
    w_dk = weights["w_dk"]
    total_w = w_dg + w_dk
    w_dg /= total_w
    w_dk /= total_w

    projected = w_dg * dg_estimate + w_dk * dk_estimate
    return projected


# ── Backtest Runner ──────────────────────────────────────────────────────────

def backtest_event(event, weights=None, verbose=False):
    """Backtest our projection model against a single historical event.

    Returns dict with accuracy metrics for this event.
    """
    event_id = event["event_id"]
    year = event["calendar_year"]
    name = event["event_name"]

    # Fetch predictions and actual results
    try:
        preds_data = fetch_event_predictions(event_id, year)
        rounds_data = fetch_event_rounds(event_id, year)
    except Exception as e:
        if verbose:
            print(f"  Skip {name} {year}: {e}")
        return None

    pred_field = preds_data.get("baseline", [])
    actual_scores = rounds_data.get("scores", [])

    if not pred_field or not actual_scores:
        return None

    # Build lookup of actual results by dg_id
    actual_lookup = {}
    for p in actual_scores:
        dg_id = p.get("dg_id")
        if dg_id:
            dk_pts, rounds_played = calc_dk_fantasy_points(p, p.get("fin_text", ""))
            actual_lookup[dg_id] = {
                "dk_points": dk_pts,
                "fin_text": p.get("fin_text", ""),
                "rounds_played": rounds_played,
                "player_name": p.get("player_name", ""),
            }

    # Run projections and compare
    results = []
    for pred in pred_field:
        dg_id = pred.get("dg_id")
        if dg_id not in actual_lookup:
            continue

        actual = actual_lookup[dg_id]

        # Use a rough DK avg proxy based on make_cut probability
        p_mc = pred.get("make_cut", 0) or 0
        if p_mc > 1:
            p_mc /= 100
        # Proxy calibrated from backtest: avg total DK pts by MC bucket
        # MC=0.3→44, MC=0.5→55, MC=0.7→67, MC=0.9→111
        dk_avg_proxy = 20 + p_mc * 65

        projected = project_player_backtest(pred, dk_avg_proxy=dk_avg_proxy, weights=weights)
        actual_pts = actual["dk_points"]

        results.append({
            "player": actual["player_name"],
            "dg_id": dg_id,
            "projected": round(projected, 1),
            "actual": actual_pts,
            "error": round(projected - actual_pts, 1),
            "abs_error": round(abs(projected - actual_pts), 1),
            "fin_text": actual["fin_text"],
        })

    if not results:
        return None

    # Calculate metrics
    errors = [r["error"] for r in results]
    abs_errors = [r["abs_error"] for r in results]
    projected_vals = [r["projected"] for r in results]
    actual_vals = [r["actual"] for r in results]

    mae = sum(abs_errors) / len(abs_errors)
    bias = sum(errors) / len(errors)
    rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
    corr = _correlation(projected_vals, actual_vals)

    # Rank correlation (how well do we rank players?)
    rank_corr = _rank_correlation(projected_vals, actual_vals)

    return {
        "event": name,
        "year": year,
        "players": len(results),
        "mae": round(mae, 2),
        "bias": round(bias, 2),
        "rmse": round(rmse, 2),
        "correlation": round(corr, 4),
        "rank_correlation": round(rank_corr, 4),
        "details": results,
    }


def _correlation(x, y):
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(sum((xi - mx)**2 for xi in x) / n)
    sy = math.sqrt(sum((yi - my)**2 for yi in y) / n)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
    return cov / (sx * sy)


def _rank_correlation(x, y):
    """Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = _ranks(x)
    ry = _ranks(y)
    return _correlation(rx, ry)


def _ranks(values):
    """Convert values to ranks (1 = highest)."""
    indexed = sorted(enumerate(values), key=lambda t: t[1], reverse=True)
    ranks = [0] * len(values)
    for rank, (idx, _) in enumerate(indexed, 1):
        ranks[idx] = rank
    return ranks


# ── Weight Tuning ────────────────────────────────────────────────────────────

def tune_weights(events, metric="rank_correlation"):
    """Grid search over projection weight combinations to find optimal weights.

    Tests different w_dg vs w_dk blends (course history and salary value
    aren't available in backtest, so we tune the core DG vs DK balance).
    """
    print("\n  Tuning model weights...")
    print(f"  Optimizing for: {metric}")
    print(f"  Testing across {len(events)} events\n")

    # Grid: w_dg from 0.5 to 0.95, w_dk is the remainder
    best_score = -999
    best_weights = None
    all_results = []

    for w_dg_pct in range(50, 100, 5):
        w_dg = w_dg_pct / 100.0
        w_dk = 1.0 - w_dg

        weights = {"w_dg": w_dg, "w_course": 0.0, "w_dk": w_dk, "w_sal": 0.0}

        event_metrics = []
        for event in events:
            result = backtest_event(event, weights=weights)
            if result:
                event_metrics.append(result[metric])

        if not event_metrics:
            continue

        avg_metric = sum(event_metrics) / len(event_metrics)
        all_results.append({
            "w_dg": w_dg,
            "w_dk": w_dk,
            "avg_score": round(avg_metric, 4),
            "n_events": len(event_metrics),
        })

        if avg_metric > best_score:
            best_score = avg_metric
            best_weights = weights.copy()

        print(f"    w_dg={w_dg:.2f}  w_dk={w_dk:.2f}  avg_{metric}={avg_metric:.4f}  ({len(event_metrics)} events)")

    print(f"\n  Best weights: w_dg={best_weights['w_dg']:.2f}, w_dk={best_weights['w_dk']:.2f}")
    print(f"  Best avg {metric}: {best_score:.4f}")

    return best_weights, all_results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DFS Golf Model Backtester")
    parser.add_argument("--events", type=int, default=20,
                        help="Number of recent events to backtest (default: 20)")
    parser.add_argument("--year", type=int, default=None,
                        help="Backtest only events from this year")
    parser.add_argument("--tune", action="store_true",
                        help="Run weight tuning after backtest")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-player details")
    args = parser.parse_args()

    print("=" * 70)
    print("  DFS GOLF MODEL BACKTESTER")
    print("=" * 70)

    # Get events
    print(f"\n  Fetching event list...")
    events = get_backtest_events(year=args.year, max_events=args.events)
    print(f"  Found {len(events)} events to backtest")

    # Run backtest on each event
    print(f"\n  Running backtest with current model weights...")
    print(f"  (w_dg={WEIGHT_DATAGOLF}, w_course={WEIGHT_COURSE_HISTORY}, "
          f"w_dk={WEIGHT_DK_AVG}, w_sal={WEIGHT_SALARY_VALUE})\n")

    all_results = []
    all_player_results = []

    for i, event in enumerate(events):
        name = event["event_name"]
        year = event["calendar_year"]
        sys.stdout.write(f"  [{i+1}/{len(events)}] {name} {year}...")
        sys.stdout.flush()

        result = backtest_event(event, verbose=args.verbose)
        if result is None:
            print(" SKIPPED (no data)")
            continue

        all_results.append(result)
        all_player_results.extend(result["details"])

        print(f" MAE={result['mae']:.1f}  Corr={result['correlation']:.3f}  "
              f"RankCorr={result['rank_correlation']:.3f}  Bias={result['bias']:+.1f}  "
              f"({result['players']} players)")

        # Be polite to the API
        time.sleep(0.3)

    if not all_results:
        print("\n  No events could be backtested.")
        return

    # Aggregate metrics
    print(f"\n{'='*70}")
    print("  BACKTEST SUMMARY")
    print(f"{'='*70}")
    print(f"  Events backtested: {len(all_results)}")
    print(f"  Total player-events: {len(all_player_results)}")

    avg_mae = sum(r["mae"] for r in all_results) / len(all_results)
    avg_bias = sum(r["bias"] for r in all_results) / len(all_results)
    avg_rmse = sum(r["rmse"] for r in all_results) / len(all_results)
    avg_corr = sum(r["correlation"] for r in all_results) / len(all_results)
    avg_rank_corr = sum(r["rank_correlation"] for r in all_results) / len(all_results)

    print(f"\n  Mean Absolute Error (MAE): {avg_mae:.2f} pts")
    print(f"  Root Mean Sq Error (RMSE): {avg_rmse:.2f} pts")
    print(f"  Avg Bias:                  {avg_bias:+.2f} pts {'(over-projects)' if avg_bias > 0 else '(under-projects)'}")
    print(f"  Avg Correlation:           {avg_corr:.4f}")
    print(f"  Avg Rank Correlation:      {avg_rank_corr:.4f}")

    # Per-player aggregate accuracy
    all_proj = [r["projected"] for r in all_player_results]
    all_actual = [r["actual"] for r in all_player_results]
    overall_corr = _correlation(all_proj, all_actual)
    overall_rank_corr = _rank_correlation(all_proj, all_actual)
    overall_mae = sum(abs(p - a) for p, a in zip(all_proj, all_actual)) / len(all_proj)

    print(f"\n  Overall (pooled across all events):")
    print(f"    MAE:              {overall_mae:.2f} pts")
    print(f"    Correlation:      {overall_corr:.4f}")
    print(f"    Rank Correlation: {overall_rank_corr:.4f}")

    # Accuracy by finish tier
    print(f"\n  Accuracy by finish position:")
    tiers = [
        ("Top 10", lambda r: _finish_pos(r["fin_text"]) is not None and _finish_pos(r["fin_text"]) <= 10),
        ("11-30", lambda r: _finish_pos(r["fin_text"]) is not None and 11 <= _finish_pos(r["fin_text"]) <= 30),
        ("31-50", lambda r: _finish_pos(r["fin_text"]) is not None and 31 <= _finish_pos(r["fin_text"]) <= 50),
        ("MC/WD/DQ", lambda r: r["fin_text"] in ("CUT", "WD", "DQ", "MDF") or _finish_pos(r["fin_text"]) is None),
    ]
    for label, filt in tiers:
        tier_results = [r for r in all_player_results if filt(r)]
        if tier_results:
            tier_mae = sum(r["abs_error"] for r in tier_results) / len(tier_results)
            tier_bias = sum(r["error"] for r in tier_results) / len(tier_results)
            print(f"    {label:<12}  n={len(tier_results):>5}  MAE={tier_mae:.1f}  Bias={tier_bias:+.1f}")

    if args.verbose:
        # Show worst predictions for diagnostic
        print(f"\n  Biggest misses (|error| > 40):")
        big_misses = sorted(all_player_results, key=lambda r: r["abs_error"], reverse=True)
        for r in big_misses[:15]:
            print(f"    {r['player']:<25} Proj={r['projected']:>6.1f}  Actual={r['actual']:>6.1f}  "
                  f"Error={r['error']:>+6.1f}  Fin={r['fin_text']}")

    # Save results
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history", "backtest_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    summary = {
        "events_tested": len(all_results),
        "player_events": len(all_player_results),
        "avg_mae": round(avg_mae, 2),
        "avg_rmse": round(avg_rmse, 2),
        "avg_bias": round(avg_bias, 2),
        "avg_correlation": round(avg_corr, 4),
        "avg_rank_correlation": round(avg_rank_corr, 4),
        "overall_correlation": round(overall_corr, 4),
        "overall_rank_correlation": round(overall_rank_corr, 4),
        "per_event": [{k: v for k, v in r.items() if k != "details"} for r in all_results],
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Weight tuning
    if args.tune:
        print(f"\n{'='*70}")
        print("  WEIGHT TUNING")
        print(f"{'='*70}")
        best_weights, tune_results = tune_weights(events)

        print(f"\n  Recommended config.py update:")
        print(f"    WEIGHT_DATAGOLF = {best_weights['w_dg']:.2f}")
        print(f"    WEIGHT_DK_AVG = {best_weights['w_dk']:.2f}")

    print(f"\n{'='*70}")
    print("  BACKTEST COMPLETE")
    print(f"{'='*70}\n")


def _finish_pos(fin_text):
    """Parse finish position from fin_text, returns int or None."""
    if not fin_text or fin_text in ("CUT", "WD", "DQ", "MDF"):
        return None
    try:
        return int(fin_text.replace("T", ""))
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    main()

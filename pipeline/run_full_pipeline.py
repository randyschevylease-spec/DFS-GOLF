"""
run_full_pipeline.py -- Single command to run the entire AceEdge pipeline.

Steps:
  1. generate_field.py      -> field_1/2/3.npy
  2. build_candidates.py    -> candidates.csv (7.7M)
  3. filter_candidates.py   -> candidates_filtered.csv (100K)
  4. contest_sim.py         -> contest_sim_results.csv
  5. portfolio_select.py    -> portfolio_selected.csv
  6. export_dk.py           -> dk_upload_PLAYERS_2026.csv

Usage: py pipeline/run_full_pipeline.py
"""

import os
import sys
import time
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pipeline"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "engine"))


def run_step(step_num, name, func):
    """Run a pipeline step with timing and error handling. Returns elapsed seconds."""
    print(f"\n{'='*70}")
    print(f"  STEP {step_num}/6: {name}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")

    t0 = time.perf_counter()
    func()
    elapsed = time.perf_counter() - t0

    print(f"\n  -> {name} completed in {elapsed:.1f}s")
    return elapsed


def main():
    t_total = time.perf_counter()
    today = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'#'*70}")
    print(f"  AceEdge Golf DFS Pipeline -- {today}")
    print(f"{'#'*70}")

    timings = {}

    # Step 1
    from generate_field import generate_fields
    timings["generate_field"] = run_step(1, "Generate synthetic fields", generate_fields)

    # Step 2
    from build_candidates import build_candidates
    timings["build_candidates"] = run_step(2, "Build candidate lineups", build_candidates)

    # Step 3
    from filter_candidates import filter_candidates
    timings["filter_candidates"] = run_step(3, "Filter candidates to 100K", filter_candidates)

    # Step 4
    from contest_sim import run_contest_sim
    timings["contest_sim"] = run_step(4, "Contest simulation", run_contest_sim)

    # Step 5
    from portfolio_select import run as portfolio_run
    timings["portfolio_select"] = run_step(5, "Portfolio selection", portfolio_run)

    # Step 6
    from export_dk import export
    timings["export_dk"] = run_step(6, "DraftKings export", export)

    # Final summary
    total_elapsed = time.perf_counter() - t_total

    # Read summary stats from output files
    import csv

    # Candidates generated
    cand_path = os.path.join(PROJECT_ROOT, "data", "cache", "candidates.csv")
    with open(cand_path) as f:
        n_candidates = sum(1 for _ in f) - 1

    # Candidates after filter
    filt_path = os.path.join(PROJECT_ROOT, "data", "cache", "candidates_filtered.csv")
    with open(filt_path) as f:
        n_filtered = sum(1 for _ in f) - 1

    # Portfolio stats
    port_path = os.path.join(PROJECT_ROOT, "data", "cache", "portfolio_selected.csv")
    evs = []
    overlaps = []
    with open(port_path) as f:
        for row in csv.DictReader(f):
            evs.append(float(row["mean_ev"]))
            overlaps.append(float(row["avg_portfolio_overlap"]))

    output_path = os.path.join(PROJECT_ROOT, "data", "outputs", "dk_upload_PLAYERS_2026.csv")

    print(f"\n\n{'#'*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'#'*70}")

    print(f"\n  Step timings:")
    for name, elapsed in timings.items():
        print(f"    {name:<22s} {elapsed:>8.1f}s")
    print(f"    {'TOTAL':<22s} {total_elapsed:>8.1f}s")

    print(f"\n  Results:")
    print(f"    Candidates generated:  {n_candidates:>12,}")
    print(f"    Candidates filtered:   {n_filtered:>12,}")
    print(f"    Portfolio lineups:     {len(evs):>12,}")
    print(f"    Portfolio mean EV:     ${sum(evs)/len(evs):>11.2f}")
    print(f"    Portfolio mean overlap: {sum(overlaps)/len(overlaps):>11.3f}")
    print(f"    Output: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n  PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

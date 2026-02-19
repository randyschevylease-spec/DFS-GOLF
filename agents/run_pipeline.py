#!/usr/bin/env python3
"""BOT 1: Pipeline Orchestrator — runs all bots in sequence."""
import json, sys, time, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import *

log = setup_logger("pipeline", "orchestrator.log")

PIPELINE = [
    ("BOT 2 -- Data Pipeline",    "datagolf_pipeline.py",    180),
    ("BOT 3 -- Projections",      "projection_enhancer.py",   60),
    ("BOT 4 -- Ownership",        "ownership_engine.py",      60),
    ("BOT 6 -- Correlations",     "correlation_engine.py",    60),
    ("BOT 5 -- Contest Simulator", "contest_simulator.py",   600),
]

def run_bot(name, script, timeout):
    path = AGENTS / script
    if not path.exists():
        log.error(f"{name}: {script} not found!")
        return False

    log.info(f"\n{'='*50}")
    log.info(f"STARTING: {name}")
    log.info(f"{'='*50}")

    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode == 0:
            log.info(f"{name} COMPLETE")
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-5:]:
                    log.info(f"  {line}")
            return True
        else:
            log.error(f"{name} FAILED")
            log.error(result.stderr[-500:] if result.stderr else "No error output")
            return False
    except subprocess.TimeoutExpired:
        log.error(f"{name} TIMEOUT ({timeout}s)")
        return False
    except Exception as e:
        log.error(f"{name} ERROR: {e}")
        return False

def main():
    log.info("=" * 60)
    log.info("PGA DFS PIPELINE -- STARTING")
    log.info("=" * 60)

    start = time.time()

    for name, script, timeout in PIPELINE:
        ok = run_bot(name, script, timeout)
        if not ok:
            if "Correlation" in name:
                log.warning(f"  {name} failed (optional) -- continuing")
                continue
            else:
                log.error(f"\nPIPELINE HALTED at {name}")
                sys.exit(1)

    # Push to Google Sheets with sim results
    log.info(f"\n{'='*50}")
    log.info("Exporting to Google Sheets...")
    log.info(f"{'='*50}")
    try:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "optimizer.py"), "--pipeline", "--sheets"],
            capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode == 0:
            log.info("Google Sheets export complete")
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-3:]:
                    log.info(f"  {line}")
        else:
            log.error(f"Sheets export failed: {result.stderr[-300:] if result.stderr else 'No error'}")
    except Exception as e:
        log.error(f"Sheets export error: {e}")

    elapsed = time.time() - start
    log.info(f"\n{'='*60}")
    log.info(f"PIPELINE COMPLETE in {elapsed:.0f}s")
    log.info(f"Lineups ready at: {SHARED / 'final_export.csv'}")
    log.info(f"{'='*60}")

if __name__ == "__main__":
    main()

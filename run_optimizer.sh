#!/bin/bash
# DFS Golf Optimizer — Auto-run Script
# Checks for salary CSV, runs optimizer, sends macOS notification, opens output CSV.
#
# Installation:
#   chmod +x ~/Desktop/dfs-golf/run_optimizer.sh
#   cp ~/Desktop/dfs-golf/com.dfs-golf.optimizer.plist ~/Library/LaunchAgents/
#   launchctl load ~/Library/LaunchAgents/com.dfs-golf.optimizer.plist

set -euo pipefail

PROJECT_DIR="$HOME/Desktop/dfs-golf"
SALARIES_DIR="$PROJECT_DIR/salaries"
LOG_FILE="$PROJECT_DIR/history/optimizer_$(date +%Y%m%d_%H%M).log"
PYTHON="python3"

# Ensure log directory exists
mkdir -p "$PROJECT_DIR/history"

# Check for salary CSV
LATEST_CSV=$(ls -t "$SALARIES_DIR"/*.csv 2>/dev/null | head -1)
if [ -z "$LATEST_CSV" ]; then
    osascript -e 'display notification "No salary CSV found in salaries/ folder" with title "DFS Golf Optimizer" subtitle "ERROR"'
    echo "ERROR: No salary CSV found in $SALARIES_DIR/" >> "$LOG_FILE"
    exit 1
fi

echo "=== DFS Golf Optimizer Auto-Run ===" >> "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "Using CSV: $LATEST_CSV" >> "$LOG_FILE"

# Run optimizer
cd "$PROJECT_DIR"
$PYTHON optimizer.py --lineups 150 >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    # Find the latest output CSV
    OUTPUT_CSV=$(ls -t "$PROJECT_DIR"/lineups_*.csv 2>/dev/null | head -1)

    osascript -e "display notification \"150 lineups generated successfully\" with title \"DFS Golf Optimizer\" subtitle \"Complete\""

    # Open the output CSV
    if [ -n "$OUTPUT_CSV" ]; then
        open "$OUTPUT_CSV"
    fi

    echo "Completed successfully: $(date)" >> "$LOG_FILE"
else
    osascript -e "display notification \"Optimizer failed — check log\" with title \"DFS Golf Optimizer\" subtitle \"ERROR\""
    echo "FAILED with exit code $EXIT_CODE: $(date)" >> "$LOG_FILE"
    exit $EXIT_CODE
fi

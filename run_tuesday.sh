#!/bin/bash
# Re-run Cognizant Classic with live DG ownership
# Scheduled for Tuesday Feb 25, 2025 at 3:00 PM

cd /Users/rhbot/Desktop/dfs-golf

echo "=========================================="
echo "  Checking DG ownership availability..."
echo "  $(date)"
echo "=========================================="

# Check if ownership is live
OWN_CHECK=$(python3 -c "
from datagolf_client import get_fantasy_projections
data = get_fantasy_projections()
projs = data.get('projections', []) if isinstance(data, dict) else data
has_own = sum(1 for p in projs if (p.get('proj_ownership') or 0) > 0)
print(has_own)
")

if [ "$OWN_CHECK" -eq 0 ]; then
    echo "  Ownership not live yet. Retrying in 1 hour..."
    # Schedule a retry in 1 hour
    echo "python3 /Users/rhbot/Desktop/dfs-golf/run.py --contest 188255235 --lineups 150 --candidates 5000 --sims 10000 --sheets" | at "now + 1 hour" 2>/dev/null
    exit 0
fi

echo "  Ownership is LIVE ($OWN_CHECK players). Running optimizer..."

python3 run.py --contest 188255235 --lineups 150 --candidates 5000 --sims 10000 --sheets

echo ""
echo "  Done at $(date)"

DATAGOLF_API_KEY = "85acc5b95e8b7ead122cab0c8020"
DATAGOLF_BASE_URL = "https://feeds.datagolf.com"

# Default settings
TOUR = "pga"
ODDS_FORMAT = "american"
NUM_LINEUPS = 150

# DraftKings Classic PGA constraints
ROSTER_SIZE = 6
SALARY_CAP = 50000
SALARY_FLOOR = 48000  # Minimum salary usage (96% of cap)

# Multi-lineup exposure limit (max % of lineups a single player can appear in)
MAX_EXPOSURE = 0.60

# CVaR tail-risk penalty (λ). Controls upside vs downside tradeoff:
#   0.0  = pure E[max] — best for GPP (highest ROI, best diversification)
#   0.15 = light hedge (slightly softer worst-case, lower ceiling)
#   0.5  = heavy hedge (sacrifices significant upside for tail protection)
# For top-heavy GPP contests, 0.0 is optimal. Use --cvar-lambda to override.
CVAR_LAMBDA = 0.0

# Ownership leverage
LEVERAGE_POWER = 0.35
LEVERAGE_MULT_FLOOR = 0.70
LEVERAGE_MULT_CAP = 1.50

# Vegas line movement
MOVEMENT_THRESHOLD_PCT = 5.0
MOVEMENT_ADJ_CAP = 1.0

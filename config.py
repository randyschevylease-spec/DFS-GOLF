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

BASE_CORRELATION = 0.11
SAME_WAVE_CORRELATION = 0.25   # same AM/PM wave → shared conditions
DIFF_WAVE_CORRELATION = 0.05   # cross-wave → different conditions

# CVaR tail-risk penalty (λ). Controls upside vs downside tradeoff:
#   0.0  = pure E[max] — best for GPP (highest ROI, best diversification)
#   0.15 = light hedge (slightly softer worst-case, lower ceiling)
#   0.5  = heavy hedge (sacrifices significant upside for tail protection)
# For top-heavy GPP contests, 0.0 is optimal. Use --cvar-lambda to override.
CVAR_LAMBDA = 0.0

# Portfolio selection method
PORTFOLIO_METHOD = "emax"         # "mpt" or "emax"
MPT_FRONTIER_TOLERANCE = 0.02     # Sharpe tolerance for near-frontier inclusion
MPT_MIN_FRONTIER_SIZE = 200       # Minimum frontier candidates
MPT_SIGMA_BINS = 50               # Risk bins for envelope sweep
MPT_FRONTIER_MAX = 1500           # Hard cap on frontier size (bounds cov matrix)

# Vegas line movement
MOVEMENT_THRESHOLD_PCT = 5.0
MOVEMENT_ADJ_CAP = 1.0

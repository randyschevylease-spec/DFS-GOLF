"""
scoring.py — DraftKings golf scoring rules.

Converts raw tournament results into DK fantasy points:
  - Hole scoring: eagle (-3), birdie (-2), par, bogey (+1), double+
  - Finish bonus: 1st (30), 2nd (20), 3rd (18), etc.
  - Streak bonus: 3+ consecutive birdies
  - Bogey-free round bonus
  - Hole-in-one bonus
  - Under 70 round bonus

Note: Historical data already includes actual points, so this module
is primarily for validation and forward-looking projections.
"""

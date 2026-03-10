"""
results.py — Leaderboard and statistical analysis of bot battles.

After a tournament run, this module:
  - Builds a leaderboard ranked by ROI
  - Calculates per-bot stats: avg score, ceiling, floor, std dev
  - Runs statistical significance tests (are differences real or noise?)
  - Exports results to reports/battle_results.csv
"""

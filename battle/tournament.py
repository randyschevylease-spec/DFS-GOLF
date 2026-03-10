"""
tournament.py — Runs bot battles across N historical contests.

For each historical event in the cache:
  1. Gives each bot the pre-contest player pool
  2. Collects their lineups
  3. Passes lineups + actual results to contest_sim
  4. Records profit/loss per bot

Tracks cumulative ROI, win rate, and consistency across the full sample.
"""

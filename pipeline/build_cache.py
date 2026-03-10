"""
build_cache.py — Transform processed data into bot-ready cache files.

Reads from data/processed/, builds optimized lookup structures:
  - Per-event player pools with salary + ownership + actual points
  - Historical player stat aggregates (avg pts, consistency, ceiling)
  - Salary-adjusted value metrics

Writes to data/cache/ in formats bots can quickly load.
Bots should NEVER call APIs directly — only read from cache.
"""

"""
clean_data.py — Normalize and validate raw DataGolf API responses.

Reads JSON files from data/raw/, applies:
  - Consistent player name formatting
  - Missing value handling (null ownership, WD/CUT players)
  - Schema validation (expected fields present per site)
  - Deduplication checks

Outputs cleaned CSVs to data/processed/.
"""

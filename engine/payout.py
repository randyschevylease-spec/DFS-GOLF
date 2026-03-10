"""
payout.py — Contest payout structure logic.

Models different DFS contest types:
  - GPP (tournament): Top-heavy, ~20% of field cashes
  - Cash (50/50, double-up): ~50% of field cashes, flat payout

Given a lineup's rank in the contest field, returns the payout.
Used by contest_sim.py to calculate bot profit/loss.
"""

# DraftKings Milly Maker — $20 entry, $1M to 1st
# Format: (rank_start, rank_end, payout)
MILLY_MAKER_PAYOUTS = [
    (1, 1, 1_000_000),
    (2, 2, 100_000),
    (3, 3, 40_000),
    (4, 4, 20_000),
    (5, 5, 15_000),
    (6, 6, 10_000),
    (7, 8, 7_000),
    (9, 10, 5_000),
    (11, 12, 3_000),
    (13, 15, 2_000),
    (16, 20, 1_500),
    (21, 25, 1_250),
    (26, 30, 1_000),
    (31, 40, 800),
    (41, 50, 700),
    (51, 75, 600),
    (76, 100, 500),
    (101, 125, 400),
    (126, 175, 300),
    (176, 250, 200),
    (251, 350, 150),
    (351, 500, 100),
    (501, 700, 80),
    (701, 1_000, 70),
    (1_001, 1_500, 60),
    (1_501, 2_250, 55),
    (2_251, 3_500, 50),
    (3_501, 6_900, 45),
    (6_901, 21_250, 40),
]

MILLY_MAKER_ENTRY = 25
MILLY_MAKER_FIELD = 105_800  # total entries (21,250 cash)


def get_payout(rank, payout_table=None):
    """
    Get payout for a given rank.

    Args:
        rank: 1-indexed finish position in contest
        payout_table: list of (start, end, payout) tuples.
            Defaults to MILLY_MAKER_PAYOUTS.

    Returns:
        payout amount (0 if outside payout range)
    """
    if payout_table is None:
        payout_table = MILLY_MAKER_PAYOUTS

    for start, end, payout in payout_table:
        if start <= rank <= end:
            return payout
    return 0


def get_roi(rank, entry_fee=MILLY_MAKER_ENTRY, payout_table=None):
    """Get ROI for a given rank. Returns (payout - entry_fee)."""
    return get_payout(rank, payout_table) - entry_fee


def cash_line(payout_table=None):
    """Return the last rank that receives a payout."""
    if payout_table is None:
        payout_table = MILLY_MAKER_PAYOUTS
    return payout_table[-1][1]


def payout_summary(payout_table=None, entry_fee=None):
    """Print summary stats for a payout structure."""
    if payout_table is None:
        payout_table = MILLY_MAKER_PAYOUTS
    if entry_fee is None:
        entry_fee = MILLY_MAKER_ENTRY

    total_cashing = payout_table[-1][1]
    total_prize = 0

    for start, end, payout in payout_table:
        n = end - start + 1
        total_prize += n * payout

    total_spots = MILLY_MAKER_FIELD
    total_entries_revenue = total_spots * entry_fee

    print(f"Contest Summary:")
    print(f"  Entry fee:       ${entry_fee}")
    print(f"  Max entries:     {total_spots:,}")
    print(f"  Entry revenue:   ${total_entries_revenue:,.0f}")
    print(f"  Total prize:     ${total_prize:,.0f}")
    print(f"  Rake:            ${total_entries_revenue - total_prize:,.0f} ({(total_entries_revenue - total_prize) / total_entries_revenue:.1%})")
    print(f"  Cashing spots:   {total_cashing:,} ({total_cashing / total_spots:.1%})")
    print(f"  1st place:       ${payout_table[0][2]:,.0f}")
    print(f"  Cash line:       ${payout_table[-1][2]} (rank {payout_table[-1][1]:,})")
    print(f"  Min cash ROI:    {payout_table[-1][2] / entry_fee:.1f}x")
    print(f"  1st place ROI:   {payout_table[0][2] / entry_fee:,.0f}x")


if __name__ == "__main__":
    payout_summary()

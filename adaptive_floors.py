"""Contest-adaptive projection and salary floors.

Computes candidate and opponent floors based on contest structure
(entry fee, field size, payout percentage) instead of static constants.
"""
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class FloorConfig:
    """Floor thresholds for lineup filtering."""
    proj_floor: Optional[float]
    salary_floor: Optional[int]


def _detect_contest_type(payout_pct: Optional[float]) -> str:
    """Return 'cash' if payout_pct > 0.40, else 'gpp'."""
    if payout_pct is not None and payout_pct > 0.40:
        return "cash"
    return "gpp"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def get_candidate_floors(optimal_proj: float, field_size: int,
                         entry_fee: float,
                         payout_pct: Optional[float] = None) -> FloorConfig:
    """Compute candidate lineup floors based on contest structure.

    GPP:  proj_pct = clamp(101.6 + 0.38·ln(fee) − 1.31·ln(field), 85, 98)
    Cash: proj_pct = 96%, salary_floor = 49500
    """
    ctype = _detect_contest_type(payout_pct)
    if ctype == "cash":
        return FloorConfig(
            proj_floor=optimal_proj * 0.96,
            salary_floor=49500,
        )
    # GPP
    proj_pct = _clamp(
        101.6 + 0.38 * math.log(max(entry_fee, 0.01))
             - 1.31 * math.log(max(field_size, 1)),
        85, 98,
    ) / 100.0
    return FloorConfig(
        proj_floor=optimal_proj * proj_pct,
        salary_floor=None,
    )


def get_opponent_floors(optimal_proj: float, field_size: int,
                        entry_fee: float,
                        payout_pct: Optional[float] = None) -> FloorConfig:
    """Compute opponent harvest floors based on contest structure.

    GPP:  proj_pct = clamp(79.2 + 0.56·ln(fee) − 0.22·ln(field), 70, 85)
    Cash: no opponent filtering (None floors).
    """
    ctype = _detect_contest_type(payout_pct)
    if ctype == "cash":
        return FloorConfig(proj_floor=None, salary_floor=None)
    # GPP
    proj_pct = _clamp(
        79.2 + 0.56 * math.log(max(entry_fee, 0.01))
            - 0.22 * math.log(max(field_size, 1)),
        70, 85,
    ) / 100.0
    return FloorConfig(
        proj_floor=optimal_proj * proj_pct,
        salary_floor=None,
    )


def log_floor_config(floors: FloorConfig, optimal_proj: float,
                     label: str = "GPP") -> None:
    """Print diagnostic line for adaptive floor configuration."""
    if floors.proj_floor is not None:
        pct = floors.proj_floor / optimal_proj * 100 if optimal_proj else 0
        proj_str = f"proj >= {floors.proj_floor:.1f} pts ({pct:.1f}% of {optimal_proj:.1f})"
    else:
        proj_str = "proj: none"
    sal_str = f"${floors.salary_floor:,}" if floors.salary_floor else "default"
    print(f"  Adaptive floors ({label}): {proj_str} | salary: {sal_str}")

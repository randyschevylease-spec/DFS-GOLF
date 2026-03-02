"""MIP Lineup Solver — Single HiGHS implementation for all MIP needs.

Used by both candidate_generator and field_generator for
optimization-based lineup construction.
"""
import numpy as np
from highspy import Highs, ObjSense, HighsModelStatus
from config import ROSTER_SIZE, SALARY_CAP, SALARY_FLOOR


def solve_mip(players, obj, proj_pts=None, proj_floor=None,
              salary_floor_override=None):
    """Solve a single lineup MIP using HiGHS.

    Args:
        players: list of player dicts
        obj: objective coefficients (one per player)
        proj_pts: array of projected points per player (for floor constraint)
        proj_floor: minimum total projected points for the lineup
        salary_floor_override: optional override for minimum salary usage

    Returns:
        tuple of sorted player indices, or None if infeasible
    """
    n = len(players)
    h = Highs()
    h.silent()

    for i in range(n):
        h.addVariable(0.0, 1.0, float(obj[i]))
    h.changeColsIntegrality(n, np.arange(n, dtype=np.int32),
                            np.array([1]*n, dtype=np.uint8))
    h.changeObjectiveSense(ObjSense.kMaximize)

    # Exactly ROSTER_SIZE players
    h.addRow(float(ROSTER_SIZE), float(ROSTER_SIZE), n,
             np.arange(n, dtype=np.int32), np.ones(n))

    # Salary bounds
    sal_floor = salary_floor_override if salary_floor_override is not None else SALARY_FLOOR
    salaries = np.array([float(p["salary"]) for p in players])
    h.addRow(float(sal_floor), float(SALARY_CAP), n,
             np.arange(n, dtype=np.int32), salaries)

    # Minimum projection floor (ensures lineup quality)
    if proj_pts is not None and proj_floor is not None:
        h.addRow(float(proj_floor), float(1e9), n,
                 np.arange(n, dtype=np.int32),
                 proj_pts.astype(np.float64))

    h.run()
    if h.getModelStatus() != HighsModelStatus.kOptimal:
        return None

    sol = h.getSolution()
    return tuple(sorted(i for i in range(n) if sol.col_value[i] > 0.5))

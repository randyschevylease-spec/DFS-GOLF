#!/usr/bin/env python3
"""BOT 6: Wave & Skill Correlation Engine.
Builds an N×N player correlation matrix from wave assignments and SG profiles."""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import *

log = setup_logger("bot6", "bot6_correlations.log")

# --- Tunable parameters ---
BASELINE_CORR = 0.04
SAME_WAVE_BONUS = 0.18
OPP_WAVE_PENALTY = -0.04
SG_SIMILARITY_SCALE = 0.08
SG_KEYS = ("sg_ott", "sg_app", "sg_arg", "sg_putt")
MIN_EIGENVALUE = 1e-4


def build_sg_cosine_matrix(players):
    """Compute N×N cosine similarity from 4-component SG vectors."""
    n = len(players)
    vecs = np.zeros((n, len(SG_KEYS)))
    for i, p in enumerate(players):
        sg = p.get("sg_predictions", {})
        for j, key in enumerate(SG_KEYS):
            vecs[i, j] = sg.get(key, 0.0)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    unit = vecs / norms
    cosine = unit @ unit.T
    return cosine


def build_correlation_matrix(players):
    """Assemble full N×N correlation matrix: baseline + wave + SG similarity.
    Guarantees PSD via eigenvalue clipping."""
    n = len(players)

    # Wave labels — normalise to 'AM' / 'PM' / None
    waves = []
    for p in players:
        w = p.get("tee_time_info", {}).get("wave", "")
        if isinstance(w, str):
            w = w.upper()
        if w in ("AM", "EARLY", 1, "1"):
            waves.append("AM")
        elif w in ("PM", "LATE", 2, "2"):
            waves.append("PM")
        else:
            waves.append(None)

    # Vectorised wave component
    wave_matrix = np.zeros((n, n))
    for i in range(n):
        if waves[i] is None:
            continue
        for j in range(i + 1, n):
            if waves[j] is None:
                continue
            val = SAME_WAVE_BONUS if waves[i] == waves[j] else OPP_WAVE_PENALTY
            wave_matrix[i, j] = val
            wave_matrix[j, i] = val

    # SG cosine similarity
    sg_cos = build_sg_cosine_matrix(players)

    # Combine
    corr = np.full((n, n), BASELINE_CORR) + wave_matrix + SG_SIMILARITY_SCALE * sg_cos
    np.fill_diagonal(corr, 1.0)

    # Clamp to valid correlation range
    corr = np.clip(corr, -1.0, 1.0)

    # PSD safety net — clip negative eigenvalues
    eigvals, eigvecs = np.linalg.eigh(corr)
    min_eig = float(eigvals.min())
    if min_eig < MIN_EIGENVALUE:
        eigvals = np.maximum(eigvals, MIN_EIGENVALUE)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Restore unit diagonal
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

    return corr, min_eig


def run():
    log.info("=" * 60)
    log.info("BOT 6: Correlation Engine")
    log.info("=" * 60)

    with open(SHARED / "projections.json") as f:
        data = json.load(f)

    players = data.get("players", [])
    n = len(players)
    log.info(f"Building {n}×{n} correlation matrix")

    corr, min_eig = build_correlation_matrix(players)
    is_psd = bool(min_eig >= MIN_EIGENVALUE)

    player_order = [p["name"] for p in players]

    output = {
        "metadata": {
            "player_count": n,
            "params": {
                "baseline": BASELINE_CORR,
                "same_wave_bonus": SAME_WAVE_BONUS,
                "opp_wave_penalty": OPP_WAVE_PENALTY,
                "sg_similarity_scale": SG_SIMILARITY_SCALE,
            },
            "min_eigenvalue": round(min_eig, 4),
            "is_psd": is_psd,
        },
        "player_order": player_order,
        "correlation_matrix": [[round(float(corr[i, j]), 6) for j in range(n)] for i in range(n)],
    }

    with open(SHARED / "correlations.json", "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"Correlation matrix saved ({n}×{n}, min_eig={min_eig:.4f}, PSD={is_psd})")
    print(f"BOT 6: {n}×{n} correlation matrix saved (PSD={is_psd}, min_eig={min_eig:.4f})")


if __name__ == "__main__":
    run()

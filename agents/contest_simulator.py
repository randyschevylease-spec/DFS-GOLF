#!/usr/bin/env python3
"""BOT 5 v3.0: Full Contest Simulation + Efficient Frontier.
Vectorized numpy — handles 47K opponent fields."""
import json, sys, time
import numpy as np
from pathlib import Path
from highspy import Highs, ObjSense, HighsModelStatus
sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import *

log = setup_logger("bot5", "bot5_contest_sim.log")

def load_data():
    with open(SHARED / "projections.json") as f:
        proj = json.load(f)
    with open(SHARED / "ownership.json") as f:
        own = json.load(f)
    own_lookup = {p['name']: p for p in own.get('players', [])}

    players = []
    for pp in proj.get('players', []):
        name = pp.get('name', '')
        salary = pp.get('dk_salary', 0)
        if salary <= 0: continue
        op = own_lookup.get(name, {})
        enh = pp.get('enhanced_projection', {})
        probs = pp.get('dg_model_probs', {})
        players.append({
            'name': name, 'dg_id': pp.get('dg_id', ''), 'salary': salary,
            'proj_pts': enh.get('final_proj_dk_pts', 0),
            'ceiling': enh.get('final_ceiling', 0),
            'floor': enh.get('final_floor', 0),
            'ownership_pct': op.get('projected_ownership_pct',
                             pp.get('dg_projection', {}).get('proj_ownership_pct', 5)),
            'cut_prob': probs.get('make_cut', 0.5),
            'wave': pp.get('tee_time_info', {}).get('wave', 'UNKNOWN'),
        })
    return players


def load_correlation_matrix(players):
    """Load BOT 6 correlation matrix, aligned to current player ordering.
    Falls back to identity matrix if correlations.json is missing."""
    corr_path = SHARED / "correlations.json"
    n = len(players)

    if not corr_path.exists():
        log.warning("correlations.json not found — using identity (independent sampling)")
        return np.eye(n)

    with open(corr_path) as f:
        data = json.load(f)

    corr_order = data.get("player_order", [])
    raw_matrix = np.array(data.get("correlation_matrix", []))

    if raw_matrix.shape[0] != len(corr_order):
        log.warning("Correlation matrix shape mismatch — using identity")
        return np.eye(n)

    # Build name → index lookup for the correlation file's ordering
    corr_idx = {name: i for i, name in enumerate(corr_order)}

    # Align to BOT 5's player ordering
    aligned = np.eye(n)
    for i, pi in enumerate(players):
        ci = corr_idx.get(pi['name'])
        if ci is None:
            continue
        for j, pj in enumerate(players):
            cj = corr_idx.get(pj['name'])
            if cj is None:
                continue
            aligned[i, j] = raw_matrix[ci, cj]

    log.info(f"Loaded {raw_matrix.shape[0]}×{raw_matrix.shape[1]} correlation matrix, "
             f"aligned to {n} players")
    return aligned

def _generate_batch(players, n_entries, probs, alpha_scale):
    """Generate a batch of lineups using Dirichlet-multinomial sampling."""
    n_players = len(players)
    min_sal = min(p['salary'] for p in players)
    alpha = np.maximum(probs * alpha_scale * n_players, 0.01)
    lineups = []

    generated = 0
    attempts = 0
    while generated < n_entries and attempts < n_entries * 15:
        attempts += 1
        try:
            draw = np.random.dirichlet(alpha)
        except Exception:
            draw = probs

        selected = []
        budget = SALARY_CAP
        avail = np.ones(n_players, dtype=bool)
        ok = True

        for slot in range(ROSTER_SIZE):
            rem = ROSTER_SIZE - slot - 1
            min_rem = rem * min_sal if rem > 0 else 0
            max_aff = budget - min_rem
            afford = np.array([players[i]['salary'] <= max_aff for i in range(n_players)])
            valid = avail & afford
            if not np.any(valid):
                ok = False; break
            vp = draw * valid
            if vp.sum() <= 0:
                ok = False; break
            vp /= vp.sum()
            try:
                c = np.random.choice(n_players, p=vp)
            except Exception:
                ok = False; break
            selected.append(c)
            budget -= players[c]['salary']
            avail[c] = False

        if ok and len(selected) == ROSTER_SIZE and sum(players[i]['salary'] for i in selected) <= SALARY_CAP:
            lineups.append(selected)
            generated += 1

    return lineups


def generate_opponents(players, opp_profile, n_total):
    """Generate opponent lineups calibrated to DG projected ownership.

    Uses a pilot batch to measure salary-constraint bias, then adjusts
    selection probabilities so the full field converges on DG targets.
    """
    n_players = len(players)
    target_own = np.array([max(p['ownership_pct'], 0.1) for p in players])
    target_own /= target_own.sum()  # normalise to probability

    ALPHA_SCALE = 12.0  # controls lineup-to-lineup variance
    PILOT_SIZE = min(5000, n_total)
    CALIBRATION_ROUNDS = 5

    # Iterative calibration: pilot → measure → adjust → repeat
    adjusted_probs = target_own.copy()
    for rnd in range(CALIBRATION_ROUNDS):
        pilot = _generate_batch(players, PILOT_SIZE, adjusted_probs, ALPHA_SCALE)
        if not pilot:
            break

        # Measure resulting ownership
        counts = np.zeros(n_players)
        for lu in pilot:
            for idx in lu:
                counts[idx] += 1
        pilot_own = counts / len(pilot) / ROSTER_SIZE  # per-slot frequency
        pilot_own = np.maximum(pilot_own, 1e-6)

        # Adjust with damping to prevent oscillation
        ratio = target_own / pilot_own
        damped_ratio = ratio ** 0.6
        adjusted_probs = adjusted_probs * damped_ratio
        adjusted_probs /= adjusted_probs.sum()

        log.info(f"  Calibration round {rnd + 1}: pilot RMSE "
                 f"{np.sqrt(np.mean((pilot_own * target_own.sum() / pilot_own.sum() - target_own)**2)) * 100 * n_players / ROSTER_SIZE:.2f}%")

    # Generate full field with calibrated probabilities
    all_lineups = _generate_batch(players, n_total, adjusted_probs, ALPHA_SCALE)
    log.info(f"  Generated {len(all_lineups)}/{n_total} opponent lineups")

    return all_lineups

def _solve_mip(players, obj):
    """Solve a single lineup MIP using HiGHS. Returns tuple of indices or None."""
    n = len(players)
    h = Highs()
    h.silent()
    for i in range(n):
        h.addVariable(0.0, 1.0, float(obj[i]))
    h.changeColsIntegrality(n, np.arange(n, dtype=np.int32), np.array([1]*n, dtype=np.uint8))
    h.changeObjectiveSense(ObjSense.kMaximize)
    # Roster size
    h.addRow(float(ROSTER_SIZE), float(ROSTER_SIZE), n, np.arange(n, dtype=np.int32), np.ones(n))
    # Salary bounds
    salaries = np.array([float(p['salary']) for p in players])
    h.addRow(42000.0, float(SALARY_CAP), n, np.arange(n, dtype=np.int32), salaries)
    h.run()
    if h.getModelStatus() != HighsModelStatus.kOptimal:
        return None
    sol = h.getSolution()
    return tuple(sorted(i for i in range(n) if sol.col_value[i] > 0.5))


def generate_candidates(players, n_candidates):
    """Generate diverse candidate lineup pool via randomized MIP solves (HiGHS)."""
    n = len(players)
    base_obj = np.array([p['proj_pts'] for p in players])
    candidates = set()
    rng = np.random.default_rng()

    # Phase 1: Noise-injected MIP solves
    for _ in range(n_candidates):
        noise = np.exp(rng.normal(0.0, 0.15, size=n))
        noisy_obj = base_obj * noise
        sel = _solve_mip(players, noisy_obj)
        if sel is not None:
            candidates.add(sel)

    # Phase 2: Diversity — exclude top players
    top_players = sorted(range(n), key=lambda i: base_obj[i], reverse=True)[:10]
    for excluded in top_players:
        for _ in range(50):
            noise = np.exp(rng.normal(0.0, 0.15, size=n))
            noisy_obj = base_obj * noise
            noisy_obj[excluded] = -1e6
            sel = _solve_mip(players, noisy_obj)
            if sel is not None:
                candidates.add(sel)

    return [list(c) for c in candidates]

def simulate(players, our_lineups, opp_lineups, n_sims, payout_structure, entry_fee, corr_matrix):
    """Vectorized contest simulation with correlated player sampling."""
    n_players = len(players)
    n_ours = len(our_lineups)
    n_total = n_ours + len(opp_lineups)

    # Build binary lineup matrix
    all_lu = our_lineups + opp_lineups
    matrix = np.zeros((n_total, n_players), dtype=np.float32)
    for i, lu in enumerate(all_lu):
        for idx in lu:
            matrix[i, idx] = 1.0

    means = np.array([p['proj_pts'] for p in players], dtype=np.float64)
    stds = np.array([max(p['ceiling'] - p['floor'], 2.0) / 2.56 for p in players], dtype=np.float64)
    cut_probs = np.array([p['cut_prob'] for p in players])

    # Build covariance matrix from correlation matrix and player stds
    cov = np.outer(stds, stds) * corr_matrix

    # Pre-generate all correlated samples at once
    rng = np.random.default_rng()
    all_samples = rng.multivariate_normal(means, cov, size=n_sims)  # (n_sims, n_players)

    payout_thresholds = sorted(payout_structure.items(), reverse=True)

    payouts = np.zeros((n_ours, n_sims), dtype=np.float32)

    log.info(f"Running {n_sims} sims: {n_ours} ours vs {len(opp_lineups)} opponents...")
    start = time.time()

    for sim in range(n_sims):
        if sim % 3000 == 0 and sim > 0:
            rate = sim / (time.time() - start)
            log.info(f"  Sim {sim}/{n_sims} ({rate:.0f}/s)")

        # Use pre-generated correlated samples
        scores = all_samples[sim].copy()
        scores += np.random.exponential(stds * 0.15)  # Positive skew

        missed = np.random.random(n_players) > cut_probs
        scores[missed] *= 0.4

        scores = np.maximum(scores, 0).astype(np.float32)

        # Score all lineups (single matmul)
        lu_scores = matrix @ scores

        # Percentiles for our lineups
        sorted_all = np.sort(lu_scores)
        our_scores = lu_scores[:n_ours]
        ranks = np.searchsorted(sorted_all, our_scores, side='left')
        percentiles = (ranks / n_total) * 100

        # Payouts
        sim_payouts = np.zeros(n_ours, dtype=np.float32)
        for pct_thresh, mult in payout_thresholds:
            mask = (percentiles >= pct_thresh) & (sim_payouts == 0)
            sim_payouts[mask] = mult * entry_fee

        payouts[:, sim] = sim_payouts

    elapsed = time.time() - start
    log.info(f"Simulations complete in {elapsed:.1f}s ({n_sims/elapsed:.0f} sims/s)")

    mean_payouts = payouts.mean(axis=1)
    roi = (mean_payouts - entry_fee) / entry_fee * 100
    roi_var = payouts.var(axis=1) / (entry_fee ** 2) * 10000
    cash_rate = (payouts > 0).mean(axis=1) * 100

    return roi, roi_var, cash_rate, mean_payouts, payouts

def select_frontier(roi, roi_var, n_select):
    """Select lineups from efficient frontier."""
    n = len(roi)
    on_frontier = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j: continue
            if roi[j] > roi[i] and roi_var[j] <= roi_var[i]:
                dominated = True; break
        if not dominated:
            on_frontier.append(i)

    # Score: ROI - 0.3 * sqrt(variance)
    scores = {i: roi[i] - 0.3 * np.sqrt(roi_var[i]) for i in on_frontier}
    selected = sorted(scores, key=scores.get, reverse=True)[:n_select]

    # Fill with top ROI if not enough frontier lineups
    if len(selected) < n_select:
        remaining = [i for i in np.argsort(roi)[::-1] if i not in selected]
        selected.extend(remaining[:n_select - len(selected)])

    return selected, on_frontier

def run():
    log.info("=" * 70)
    log.info("BOT 5 v3.0: CONTEST SIMULATOR + EFFICIENT FRONTIER")
    log.info("=" * 70)

    players = load_data()
    if not players:
        log.error("No player data!"); return

    # Load contest config
    config_path = SHARED / "contest_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        log.info(f"Loaded contest config: {config.get('field_size')} field, {config.get('max_entries_per_user')} max")
    else:
        log.warning("No contest_config.json -- using defaults")
        config = {
            'field_size': 10000, 'max_entries_per_user': 150, 'entry_fee': 20,
            'prize_pool': 170000, 'payout_structure': {
                99.9: 500, 99.5: 150, 99: 75, 98: 30, 95: 12, 90: 5, 85: 3, 80: 2, 0: 0
            },
            'opponent_field': build_default_opponents(10000),
            'simulation_params': {'n_simulations': 10000, 'n_candidates': 800, 'target_lineups': 150},
        }

    field_size = config['field_size']
    max_entries = config.get('max_entries_per_user', 150)
    entry_fee = config['entry_fee']
    n_sims = config.get('simulation_params', {}).get('n_simulations', 10000)
    n_cands = config.get('simulation_params', {}).get('n_candidates', 800)
    target_lu = config.get('simulation_params', {}).get('target_lineups', max_entries)
    payout = config['payout_structure']
    # Ensure payout keys are floats
    payout = {float(k): float(v) for k, v in payout.items()}

    opp_profile = config.get('opponent_field', {})

    print(f"Players: {len(players)} | Field: {field_size:,} | Max: {max_entries} | Fee: ${entry_fee}")

    # Load correlation matrix from BOT 6
    corr_matrix = load_correlation_matrix(players)

    # Generate opponents
    print(f"\n[1/4] Generating {field_size:,} opponent lineups...")
    opp_lineups = generate_opponents(players, opp_profile, field_size)
    print(f"  {len(opp_lineups):,} opponent lineups generated")

    # Generate candidates
    print(f"\n[2/4] Generating {n_cands} candidate lineups...")
    our_candidates = generate_candidates(players, n_cands)
    print(f"  {len(our_candidates)} unique candidates")

    # Simulate
    print(f"\n[3/4] Simulating {n_sims:,} contests (vectorized)...")
    roi, roi_var, cash_rate, mean_payouts, raw_payouts = simulate(
        players, our_candidates, opp_lineups, n_sims, payout, entry_fee, corr_matrix
    )

    print(f"  Avg ROI: {roi.mean():.1f}% | Best: {roi.max():.1f}% | Avg cash rate: {cash_rate.mean():.1f}%")

    # Select from frontier
    print(f"\n[4/4] Building efficient frontier, selecting {target_lu} lineups...")
    selected_idx, frontier_idx = select_frontier(roi, roi_var, target_lu)
    print(f"  Frontier: {len(frontier_idx)} lineups | Selected: {len(selected_idx)}")

    # Build output
    output_lineups = []
    for rank, idx in enumerate(selected_idx):
        lu = our_candidates[idx]
        output_lineups.append({
            'lineup_id': rank + 1,
            'players': [
                {'name': players[i]['name'], 'dg_id': players[i]['dg_id'],
                 'salary': players[i]['salary'], 'proj_pts': round(players[i]['proj_pts'], 1),
                 'ceiling': round(players[i]['ceiling'], 1),
                 'ownership_pct': round(players[i]['ownership_pct'], 1)}
                for i in sorted(lu, key=lambda i: players[i]['salary'], reverse=True)
            ],
            'total_salary': sum(players[i]['salary'] for i in lu),
            'total_proj': round(sum(players[i]['proj_pts'] for i in lu), 1),
            'mean_roi_pct': round(float(roi[idx]), 2),
            'roi_std': round(float(np.sqrt(roi_var[idx])), 2),
            'cash_rate_pct': round(float(cash_rate[idx]), 1),
            'mean_payout': round(float(mean_payouts[idx]), 2),
            'avg_ownership': round(float(np.mean([players[i]['ownership_pct'] for i in lu])), 1),
        })

    output = {
        'metadata': {
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'contest_field_size': field_size,
            'max_entries': max_entries,
            'entry_fee': entry_fee,
            'n_sims': n_sims,
            'frontier_size': len(frontier_idx),
            'methodology': 'Full contest simulation with Dirichlet-multinomial opponent field',
        },
        'large_gpp': {
            'lineups': output_lineups,
            'simulation_summary': {
                'avg_roi': round(float(roi.mean()), 2),
                'best_roi': round(float(roi.max()), 2),
                'avg_cash_rate': round(float(cash_rate.mean()), 1),
            },
        },
    }

    with open(SHARED / "optimized_lineups.json", "w") as f:
        json.dump(output, f, indent=2)

    # DK CSV export
    lines = ["G,G,G,G,G,G"]
    for lu in output_lineups:
        ids = [str(p['dg_id']) for p in lu['players']]
        lines.append(",".join(ids))
    (SHARED / "final_export.csv").write_text("\n".join(lines))

    Path(SHARED / "lineups_ready.flag").touch()

    print(f"\n{'='*60}")
    print(f"CONTEST SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Selected: {len(output_lineups)} lineups from frontier")
    if output_lineups:
        print(f"  Best ROI: {output_lineups[0]['mean_roi_pct']}%")
        print(f"  Avg cash rate: {np.mean([l['cash_rate_pct'] for l in output_lineups]):.1f}%")

        top5 = output_lineups[:5]
        print(f"\n  TOP 5 LINEUPS:")
        for lu in top5:
            names = ', '.join(p['name'] for p in lu['players'])
            print(f"  #{lu['lineup_id']} ROI={lu['mean_roi_pct']}% Cash={lu['cash_rate_pct']}% "
                  f"Own={lu['avg_ownership']}% Sal=${lu['total_salary']}")
            print(f"     {names}")

    print(f"\n  Output: {SHARED / 'optimized_lineups.json'}")
    print(f"  DK CSV: {SHARED / 'final_export.csv'}")

def build_default_opponents(n):
    return {'composition': {
        'casual': {'n_entries': n, 'dirichlet_alpha_scale': 15,
                   'ownership_modifier': 1.0, 'ceiling_weight': 0.0}
    }}

if __name__ == "__main__":
    run()

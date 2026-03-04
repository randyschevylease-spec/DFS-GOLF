#!/usr/bin/env python3
"""Explore tail-weighted E[max] — how different alpha values change lineup selection."""
import sys, os, time
import numpy as np
sys.path.insert(0, "/Users/rhbot/Desktop/dfs-golf")
sys.path.insert(0, "/Users/rhbot/Desktop/dfs-core")
os.chdir("/Users/rhbot/Desktop/dfs-golf")

from run_cognizant import parse_players, CSV_PATH, MIN_PROJ_PCT, SALARY_FLOOR_CUSTOM, PROJ_FLOOR_CUSTOM
from engine import generate_candidates, compute_mixture_params, transform_mixture_scores, _get_sigma
from field_generator import generate_field as generate_field_archetypes, field_to_index_lists, DEFAULT_ARCHETYPE_WEIGHTS
from config import ROSTER_SIZE, SALARY_CAP, SAME_WAVE_CORRELATION, DIFF_WAVE_CORRELATION

# ── Load players ──
players = parse_players(CSV_PATH)

# Get make_cut from DG
from datagolf_client import get_predictions
preds = get_predictions()
baseline = preds.get('baseline_history_fit', preds.get('baseline', []))
last_name_lookup = {}
for entry in baseline:
    dg_name = entry.get('player_name', '')
    mc = entry.get('make_cut', 0)
    if ',' in dg_name:
        last = dg_name.split(',')[0].strip()
        last_name_lookup.setdefault(last, []).append((dg_name, mc))
for p in players:
    parts = p['name'].split()
    p_last = parts[-1] if parts else ''
    candidates_ln = last_name_lookup.get(p_last, [])
    if len(candidates_ln) == 1:
        p['p_make_cut'] = candidates_ln[0][1]
    elif len(candidates_ln) > 1:
        for dg_name, mc_val in candidates_ln:
            first = dg_name.split(',')[1].strip().split()[0] if ',' in dg_name else ''
            if first.lower() in p['name'].lower():
                p['p_make_cut'] = mc_val
                break

n = len(players)
print(f"Loaded {n} players")

# ── Build correlation matrix + Cholesky ──
waves = [p["wave"] for p in players]
waves_arr = np.array(waves)
corr = np.full((n, n), DIFF_WAVE_CORRELATION)
for i in range(n):
    for j in range(n):
        if i == j:
            corr[i, j] = 1.0
        elif waves[i] == waves[j]:
            corr[i, j] = SAME_WAVE_CORRELATION
L = np.linalg.cholesky(corr)

# Mixture params
mix_params = compute_mixture_params(players)
sigmas = np.array([_get_sigma(p) for p in players])
means = np.array([p["projected_points"] for p in players])
use_mix = mix_params is not None

# ── Generate candidates (small pool for speed) ──
print("Generating candidates...")
ceiling_pts = [p["ceiling"] for p in players]
candidates = generate_candidates(
    players, pool_size=10000, min_proj_pct=MIN_PROJ_PCT,
    ceiling_pts=ceiling_pts, ceiling_weight=0.0,
    salary_floor_override=SALARY_FLOOR_CUSTOM,
    proj_floor_override=PROJ_FLOOR_CUSTOM,
)
print(f"  {len(candidates)} unique candidates")

# ── Generate opponent field ──
print("Generating opponent field...")
max_field = 10000  # smaller for speed
generated_field = generate_field_archetypes(
    players, max_field,
    archetype_weights=DEFAULT_ARCHETYPE_WEIGHTS,
    ownership_tolerance=0.03,
    max_iterations=10,
)
opponents = field_to_index_lists(generated_field)
n_opps = len(opponents)
print(f"  {n_opps} opponents")

# ── Pre-sim scores ──
n_sims = 5000
n_cands = len(candidates)
rng = np.random.default_rng(42)

print(f"Simulating {n_sims} contests...")

# Build lineup-player matrices
cand_matrix = np.zeros((n_cands, n), dtype=np.float32)
for i, lu in enumerate(candidates):
    for j in lu:
        cand_matrix[i, j] = 1.0

opp_matrix = np.zeros((n_opps, n), dtype=np.float32)
for i, lu in enumerate(opponents):
    for j in lu:
        opp_matrix[i, j] = 1.0

n_total = n_cands + n_opps

# Simple payout structure (top-heavy GPP for 10K field)
payout_structure = np.zeros(n_total)
# Top 1: $10,000
payout_structure[0] = 10000
# Top 2-10: $1,000-$500
for i in range(1, 10):
    payout_structure[i] = 1000 - i * 50
# Top 11-50: $200-$50
for i in range(10, 50):
    payout_structure[i] = 200 - (i - 10) * 3.75
# Top 51-200: $30-$25
for i in range(50, 200):
    payout_structure[i] = 30 - (i - 50) * 0.033
# Top 201-1000: $25
for i in range(200, 1000):
    payout_structure[i] = 25.0

entry_fee = 25.0

# Simulate
payouts = np.zeros((n_cands, n_sims), dtype=np.float32)

t0 = time.time()
batch_size = 500
for batch_start in range(0, n_sims, batch_size):
    bs = min(batch_size, n_sims - batch_start)
    Z = rng.standard_normal((bs, n))
    X = Z @ L.T

    if use_mix:
        scores = transform_mixture_scores(
            X, sigmas,
            mix_params[0], mix_params[1], mix_params[2],
            mix_params[3], mix_params[4], mix_params[5]
        )
    else:
        scores = means[None, :] + X
        np.maximum(scores, 0.0, out=scores)

    scores = scores.astype(np.float32)

    # Lineup scores
    cand_scores = cand_matrix @ scores.T  # (n_cands, bs)
    opp_scores = opp_matrix @ scores.T    # (n_opps, bs)

    # Rank each candidate against opponents only
    for s in range(bs):
        sim_idx = batch_start + s
        opp_s = opp_scores[:, s]
        cand_s = cand_scores[:, s]
        # For each candidate, count how many opponents score higher
        # Position = 1 + count(opponents with higher score)
        opp_sorted = np.sort(opp_s)[::-1]  # descending
        for ci in range(n_cands):
            pos = np.searchsorted(-opp_sorted, -cand_s[ci]) + 1
            if pos <= len(payout_structure):
                payouts[ci, sim_idx] = payout_structure[pos - 1]

elapsed = time.time() - t0
print(f"  Simulated in {elapsed:.1f}s")

# ── Now test different alpha values ──
print("\n" + "=" * 90)
print("TAIL-WEIGHTED E[MAX] EXPLORATION")
print("=" * 90)
print("""
Current E[max] greedy: score = mean(max(0, payout - running_max))
Tail-weighted:         score = mean(max(0, payout - running_max)^α)

α = 1.0: standard E[max] (equal weight to all improvements)
α = 1.5: moderate tail preference (big improvements worth ~3x more per $)
α = 2.0: strong tail preference (big improvements worth ~10x more per $)
α = 3.0: extreme tail preference (almost only cares about largest improvements)
""")

for alpha in [1.0, 1.25, 1.5, 2.0, 3.0]:
    print(f"\n{'─' * 80}")
    print(f"  α = {alpha}")
    print(f"{'─' * 80}")

    n_select = 20  # select 20 lineups for this experiment
    alive = np.ones(n_cands, dtype=bool)
    selected = []
    running_max = np.zeros(n_sims, dtype=np.float64)
    
    # Track player appearances
    appearances = np.zeros(n, dtype=int)
    
    for rnd in range(n_select):
        # Compute marginal improvements
        improvement = np.maximum(payouts - running_max[None, :], 0.0)  # (n_cands, n_sims)
        
        if alpha == 1.0:
            scores = improvement.mean(axis=1)
        else:
            # Tail-weighted: mean(improvement^alpha)
            scores = np.power(improvement, alpha).mean(axis=1)
        
        scores[~alive] = -np.inf
        
        best_idx = int(np.argmax(scores))
        selected.append(best_idx)
        
        # Update running max
        running_max = np.maximum(running_max, payouts[best_idx].astype(np.float64))
        alive[best_idx] = False
        
        # Track appearances
        for pidx in candidates[best_idx]:
            appearances[pidx] += 1
    
    # Compute portfolio stats
    sel_payouts = payouts[selected]  # (n_select, n_sims)
    best_per_sim = sel_payouts.max(axis=0)
    total_cost = entry_fee * n_select
    roi = (best_per_sim.mean() - total_cost) / total_cost * 100
    cash_rate = (best_per_sim > total_cost).mean() * 100
    
    # Player exposure
    exposure = {}
    for si in selected:
        for pidx in candidates[si]:
            name = players[pidx]["name"]
            exposure[name] = exposure.get(name, 0) + 1
    
    # Sort by exposure
    top_exp = sorted(exposure.items(), key=lambda x: -x[1])
    
    print(f"  ROI: {roi:+.1f}%  |  Cash rate: {cash_rate:.1f}%")
    print(f"  E[best payout]: ${best_per_sim.mean():.2f}  |  Cost: ${total_cost:.0f}")
    print(f"\n  Top player exposures ({n_select} lineups):")
    for name, count in top_exp[:15]:
        pct = count / n_select * 100
        own = next((p["proj_ownership"] for p in players if p["name"] == name), 0)
        proj = next((p["projected_points"] for p in players if p["name"] == name), 0)
        bar = "█" * int(pct / 5)
        print(f"    {name:<28} {count:>3}/{n_select} ({pct:>5.1f}%) Own={own:>5.1f}%  Proj={proj:>4.1f}  {bar}")
    
    # Unique players used
    unique_players = len(exposure)
    print(f"\n  Unique players: {unique_players}/{n}")
    
    # Lineup diversity: avg pairwise Jaccard
    jaccards = []
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            si = set(candidates[selected[i]])
            sj = set(candidates[selected[j]])
            jac = len(si & sj) / len(si | sj)
            jaccards.append(jac)
    avg_jac = np.mean(jaccards) if jaccards else 0
    max_jac = np.max(jaccards) if jaccards else 0
    print(f"  Avg pairwise Jaccard: {avg_jac:.3f}  |  Max: {max_jac:.3f}")
    
    # Count lineups with 0, 1, 2, 3 chalk players (top 3 projected)
    top3_idx = sorted(range(n), key=lambda i: players[i]["projected_points"], reverse=True)[:3]
    top3_set = set(top3_idx)
    chalk_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for si in selected:
        chalk_count = len(set(candidates[si]) & top3_set)
        chalk_dist[chalk_count] += 1
    print(f"  Chalk distribution (top 3 players per lineup):")
    for k in range(4):
        pct = chalk_dist[k] / n_select * 100
        print(f"    {k} chalk: {chalk_dist[k]:>3} lineups ({pct:.0f}%)")

print("\n\nDone.")

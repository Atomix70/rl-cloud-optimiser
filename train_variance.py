"""
train_variance.py — training-run variance for the sla-focused config.

The canonical results come from a single (unseeded) training run. Deep RL is
sensitive to training randomness (Henderson et al. 2018), so this script
trains the same configuration 3 more times with fixed seeds and evaluates
each policy on the same 30 paired evaluation seeds as notebook 10. The
canonical model/numbers are untouched; this adds robustness evidence.

Run:  venv/bin/python train_variance.py
"""
import json
import random
import time

import numpy as np
import torch

from env import CloudClusterEnv
from agent import ActorCritic, train_ppo
from evaluate import run_ppo

TRAIN_SEEDS = [1, 2, 3]
EVAL_SEEDS = [1000 + i for i in range(30)]   # same paired seeds as notebook 10
CFG = {'cost': 0.3, 'sla': 2.5, 'util': 0.1}  # sla-focused (canonical config)

stats = json.load(open('trace_params.json'))['stats']
variance_results = {}

for s in TRAIN_SEEDS:
    print(f"\n{'='*55}\nTraining sla-focused, training seed {s}\n{'='*55}")
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    env = CloudClusterEnv(stats, lambda_cost=CFG['cost'], lambda_sla=CFG['sla'],
                          lambda_util=CFG['util'], seed=100 + s)
    net = ActorCritic()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    t0 = time.time()
    episode_rewards, convergence_log = train_ppo(env, net, optimizer,
                                                 total_steps=250_000)
    print(f"Trained in {(time.time()-t0)/60:.1f} min")

    torch.save(net.state_dict(), f'ppo_sla-focused_seed{s}.pth')
    json.dump(convergence_log,
              open(f'ppo_sla-focused_seed{s}_convergence.json', 'w'))

    net.eval()
    runs = [run_ppo(CloudClusterEnv(stats, seed=es), net) for es in EVAL_SEEDS]
    variance_results[f'seed{s}'] = runs
    c = [r['cost'] for r in runs]; b = [r['breaches'] for r in runs]
    print(f"seed {s}: cost {np.mean(c):.1f} ± {np.std(c):.1f}   "
          f"breaches {np.mean(b):.0f} ± {np.std(b):.0f}")

json.dump(variance_results, open('training_variance_results.json', 'w'), indent=2)

# summary across training runs (including the canonical run from notebook 10)
canonical = json.load(open('significance_results.json'))['PPO']
print(f"\n{'='*60}\nTRAINING-RUN VARIANCE — sla-focused, 30 eval seeds each\n{'='*60}")
rows = [('canonical', canonical)] + [(k, v) for k, v in variance_results.items()]
run_costs, run_breaches = [], []
for name, runs in rows:
    c = np.mean([r['cost'] for r in runs]); b = np.mean([r['breaches'] for r in runs])
    run_costs.append(c); run_breaches.append(b)
    print(f"  {name:<10}: cost {c:7.1f}   breaches {b:6.0f}")
print(f"\nAcross {len(rows)} independent training runs:")
print(f"  cost     {np.mean(run_costs):.1f} ± {np.std(run_costs):.1f}  "
      f"(range {min(run_costs):.1f}-{max(run_costs):.1f})")
print(f"  breaches {np.mean(run_breaches):.0f} ± {np.std(run_breaches):.0f}  "
      f"(range {min(run_breaches):.0f}-{max(run_breaches):.0f})")
print("\nSaved training_variance_results.json")

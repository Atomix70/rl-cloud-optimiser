"""
evaluate.py — shared evaluation functions for all agents.
Runs a trained agent (or rule-based HPA) through episodes and returns metrics.
"""
import numpy as np
import torch
from agent import DISCRETE_ACTIONS


def run_hpa(env, up=0.70, down=0.30):
    """Rule-based HPA baseline: scale on CPU thresholds."""
    env.reset()
    cost = breaches = util = 0.0; vms = []; steps = 0
    from env import STEPS_PER_WEEK
    for t in range(STEPS_PER_WEEK):
        cpu = env.history[-1][0]
        if cpu > up:     a = np.array([0.2])
        elif cpu < down: a = np.array([-0.2])
        else:            a = np.array([0.0])
        _, r, done, _, info = env.step(a)
        cost += info['cost']; breaches += info['breaches']
        util += info['utilisation']; vms.append(info['active_vms']); steps += 1
        if done: break
    return {'cost': cost, 'breaches': breaches,
            'util': util/steps, 'vms': float(np.mean(vms))}


def run_ppo(env, net):
    """Evaluate a trained PPO agent (deterministic mean action)."""
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    cost = breaches = util = 0.0; vms = []; steps = 0
    from env import STEPS_PER_WEEK
    for t in range(STEPS_PER_WEEK):
        with torch.no_grad():
            mean, _ = net.forward(obs.unsqueeze(0))
        obs, r, done, _, info = env.step(mean.squeeze(0).numpy())
        obs = torch.tensor(obs, dtype=torch.float32)
        cost += info['cost']; breaches += info['breaches']
        util += info['utilisation']; vms.append(info['active_vms']); steps += 1
        if done: break
    return {'cost': cost, 'breaches': breaches,
            'util': util/steps, 'vms': float(np.mean(vms))}


def run_dqn(env, q_net):
    """Evaluate a trained DQN agent (greedy)."""
    obs, _ = env.reset()
    cost = breaches = util = 0.0; vms = []; steps = 0
    from env import STEPS_PER_WEEK
    for t in range(STEPS_PER_WEEK):
        with torch.no_grad():
            qv = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            idx = qv.argmax().item()
        obs, r, done, _, info = env.step(np.array([DISCRETE_ACTIONS[idx] / 5.0]))
        cost += info['cost']; breaches += info['breaches']
        util += info['utilisation']; vms.append(info['active_vms']); steps += 1
        if done: break
    return {'cost': cost, 'breaches': breaches,
            'util': util/steps, 'vms': float(np.mean(vms))}


def run_many(env_factory, agent_fn, n_episodes=5):
    """Run an agent over several episodes, return list of metric dicts.
    env_factory() must return a fresh env each call."""
    return [agent_fn(env_factory()) for _ in range(n_episodes)]
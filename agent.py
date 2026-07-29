"""
agent.py — RL agent networks and training algorithms.
Shared across all notebooks (imported, not pasted).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

# ============================================================
# PPO — Actor-Critic network
# ============================================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim=32, action_dim=1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),       nn.Tanh(),
        )
        self.actor_mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(256, 1)

    def forward(self, state):
        x = self.shared(state)
        return self.actor_mean(x), self.critic(x)

    def get_action(self, state):
        mean, value = self.forward(state)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, state, action):
        mean, value = self.forward(state)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value.squeeze(-1), entropy

    def act(self, state):
        """Deterministic action (mean) for evaluation/deployment."""
        with torch.no_grad():
            mean, _ = self.forward(state)
        return mean


# ============================================================
# DQN — Q-network (discrete-action baseline)
# ============================================================
DISCRETE_ACTIONS = [-5, -2, 0, +2, +5]
N_ACTIONS = len(DISCRETE_ACTIONS)

class DQN(nn.Module):
    def __init__(self, state_dim=32, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, n_actions),
        )
    def forward(self, state):
        return self.net(state)


# ============================================================
# PPO training algorithm
# ============================================================
def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0.0
    values = values + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns

def ppo_update(net, optimizer, states, actions, old_log_probs, advantages, returns,
               clip_eps=0.2, epochs=4, value_coef=0.5, entropy_coef=0.01):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    for _ in range(epochs):
        new_log_probs, values, entropy = net.evaluate_actions(states, actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = ((values - returns)**2).mean()
        loss = actor_loss + value_coef*critic_loss - entropy_coef*entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()
    return actor_loss.item(), critic_loss.item(), entropy.mean().item()

def train_ppo(env, net, optimizer, total_steps=250_000, rollout_len=2048):
    """Train PPO. Returns (episode_rewards, convergence_log)."""
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    steps_done = 0
    episode_reward = 0
    episode_rewards = []
    convergence_log = []
    while steps_done < total_steps:
        states, actions, log_probs = [], [], []
        rewards, values, dones = [], [], []
        for _ in range(rollout_len):
            with torch.no_grad():
                action, log_prob, value = net.get_action(state.unsqueeze(0))
            a = action.squeeze(0).numpy()
            next_state, reward, done, tr, info = env.step(a)
            states.append(state); actions.append(action.squeeze(0))
            log_probs.append(log_prob.squeeze(0)); rewards.append(float(reward))
            values.append(value.item()); dones.append(1.0 if done else 0.0)
            episode_reward += reward
            state = torch.tensor(next_state, dtype=torch.float32)
            steps_done += 1
            if done:
                episode_rewards.append(episode_reward); episode_reward = 0
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            _, _, next_value = net.get_action(state.unsqueeze(0))
            next_value = next_value.item()
        advantages, returns = compute_gae(rewards, values, dones, next_value)
        b_states = torch.stack(states); b_actions = torch.stack(actions)
        b_old_lp = torch.stack(log_probs)
        b_adv = torch.tensor(advantages, dtype=torch.float32)
        b_ret = torch.tensor(returns, dtype=torch.float32)
        ppo_update(net, optimizer, b_states, b_actions, b_old_lp, b_adv, b_ret)
        recent = np.mean(episode_rewards[-5:]) if episode_rewards else float('nan')
        convergence_log.append({'steps': steps_done, 'reward': float(recent)})
        print(f"steps {steps_done:6d} | recent ep reward {recent:8.1f}")
    return episode_rewards, convergence_log


# ============================================================
# DQN training algorithm
# ============================================================
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(ns), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32))
    def __len__(self):
        return len(self.buffer)

def train_dqn(env, total_steps=250_000, batch_size=64, gamma=0.99,
              lr=1e-3, target_update=1000,
              eps_start=1.0, eps_end=0.05, eps_decay=50_000):
    """Train DQN. Returns (q_net, episode_rewards, convergence_log)."""
    q_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer()
    state, _ = env.reset()
    steps_done = 0
    episode_reward = 0
    episode_rewards = []
    convergence_log = []
    while steps_done < total_steps:
        eps = max(eps_end, eps_start - (eps_start - eps_end) * steps_done / eps_decay)
        if random.random() < eps:
            action_idx = random.randrange(N_ACTIONS)
        else:
            with torch.no_grad():
                qv = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                action_idx = qv.argmax().item()
        env_action = np.array([DISCRETE_ACTIONS[action_idx] / 5.0])
        next_state, reward, done, tr, info = env.step(env_action)
        buffer.push(state, action_idx, reward, next_state, float(done))
        episode_reward += reward
        state = next_state
        steps_done += 1
        if done:
            episode_rewards.append(episode_reward); episode_reward = 0
            state, _ = env.reset()
        if len(buffer) >= batch_size:
            s, a, r, ns, d = buffer.sample(batch_size)
            q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_next_q = target_net(ns).max(1)[0]
                target = r + gamma * max_next_q * (1 - d)
            loss = nn.functional.mse_loss(q_vals, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if steps_done % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())
        if steps_done % 10000 == 0:
            recent = np.mean(episode_rewards[-5:]) if episode_rewards else float('nan')
            convergence_log.append({'steps': steps_done, 'reward': float(recent)})
            print(f"steps {steps_done:6d} | eps {eps:.2f} | recent reward {recent:8.1f}")
    return q_net, episode_rewards, convergence_log
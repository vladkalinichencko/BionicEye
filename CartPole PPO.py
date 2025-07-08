"""
Алгоритм PPO (Clipped Surrogate):
1. Собираем переходы в буфер заданной длины rollouts_per_env.
2. Вычисляем GAE (Generalized Advantage Estimation).
3. Обновляем policy и value network по мини-батчам на несколько эпох.
"""
import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="gym.utils.passive_env_checker"
)
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from gym.vector import AsyncVectorEnv
from tqdm.auto import tqdm

device = torch.device("mps" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def make_env(env_id: str, seed: int):
    def _thunk():
        env = gym.make(env_id)
        env.action_space.seed(seed)
        env.seed(seed)
        return env
    return _thunk

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh()
        )
        self.policy_head = nn.Linear(hidden_size, act_dim)
        self.value_head = nn.Linear(hidden_size, 1)
    def forward(self, x):
        z = self.shared(x)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value

def rollout(envs, model, rollouts_per_env):
    obs, _ = envs.reset()
    obs = torch.from_numpy(obs).float().to(device)
    mb_obs, mb_actions, mb_logps, mb_rewards, mb_dones, mb_values = [], [], [], [], [], []
    for _ in range(rollouts_per_env):
        logits, values = model(obs)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        logps = dist.log_prob(actions)
        next_obs, rewards, terminations, truncations, _ = envs.step(actions.cpu().numpy())
        dones = terminations | truncations
        mb_obs.append(obs)
        mb_actions.append(actions)
        mb_logps.append(logps)
        mb_values.append(values)
        mb_rewards.append(torch.from_numpy(rewards).float().to(device))
        mb_dones.append(torch.from_numpy(dones.astype(np.float32)).to(device))
        obs = torch.from_numpy(next_obs).float().to(device)
    _, last_values = model(obs) #последний value для GAE
    return mb_obs, mb_actions, mb_logps, mb_values, mb_rewards, mb_dones, last_values

def compute_gae(mb_rewards, mb_values, mb_dones, last_values, gamma, gae_lambda):
    mb_advantages = []
    adv = torch.zeros(n_envs, device=device)
    values = mb_values + [last_values]
    for t in reversed(range(len(mb_rewards))):
        mask = 1.0 - mb_dones[t]
        delta = mb_rewards[t] + gamma * values[t+1] * mask - values[t]
        adv = delta + gamma * gae_lambda * mask * adv
        mb_advantages.insert(0, adv)
    mb_returns = [adv + val for adv, val in zip(mb_advantages, mb_values)]
    return mb_returns, mb_advantages

if __name__ == "__main__":
    env_id = "CartPole-v1"
    n_envs = 4
    env_fns = [make_env(env_id, seed=1000 + i) for i in range(n_envs)]
    envs = AsyncVectorEnv(env_fns)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.n

    rollouts_per_env = 32    
    total_steps = 200_000   
    update_epochs = 4      
    minibatch_size = 64      
    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    policy_lr = 1e-4
    value_lr = 1e-3

    model = ActorCritic(obs_dim, act_dim).to(device)
    optim_policy = optim.Adam(model.policy_head.parameters(), lr=policy_lr)
    optim_value = optim.Adam(model.value_head.parameters(), lr=value_lr)

    steps_done = 0
    pbar = tqdm(total=total_steps, desc="PPO gooooo")
    while steps_done < total_steps:
        mb_obs, mb_actions, mb_logps, mb_values, mb_rewards, mb_dones, last_values = rollout(envs, model, rollouts_per_env)
        mb_returns, mb_advantages = compute_gae(mb_rewards, mb_values, mb_dones, last_values, gamma, gae_lambda)
        obs_batch = torch.cat(mb_obs)
        actions_batch = torch.cat(mb_actions)
        old_logps_batch = torch.cat(mb_logps).detach()
        returns_batch = torch.cat(mb_returns).detach()
        adv_batch = torch.cat(mb_advantages).detach()
        adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
        batch_size_total = obs_batch.size(0)
        idxs = np.arange(batch_size_total)
        for epoch in range(update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size_total, minibatch_size):
                mb_idx = idxs[start:start+minibatch_size]
                obs_mb = obs_batch[mb_idx]
                act_mb = actions_batch[mb_idx]
                ret_mb = returns_batch[mb_idx]
                adv_mb = adv_batch[mb_idx]
                old_logp_mb = old_logps_batch[mb_idx]

                logits, values = model(obs_mb)
                dist = Categorical(logits=logits)
                logp_mb = dist.log_prob(act_mb)
                ratio = (logp_mb - old_logp_mb).exp()
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (ret_mb - values).pow(2).mean()
                entropy_loss = -dist.entropy().mean() * 0.01
                optim_policy.zero_grad()
                optim_value.zero_grad()
                total_loss = policy_loss + value_loss + entropy_loss
                total_loss.backward()
                optim_policy.step()
                optim_value.step()
        steps_done += n_envs * rollouts_per_env
        pbar.update(n_envs * rollouts_per_env)
        pbar.set_postfix({'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()})

    pbar.close()
    envs.close()

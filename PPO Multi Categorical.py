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
from gym.vector import SyncVectorEnv
from torchvision import datasets
from tqdm.auto import tqdm
import cv2
from gym import spaces
import os

import random

class CIFAR10WindowEnv(gym.Env):
    def __init__(self, data, labels, patch_size=8, max_steps=15, log_file="agent_movement_log_3.log"):
        super().__init__()
        self.data = data
        self.labels = labels
        self.patch_size = patch_size
        self.max_steps = max_steps
        self.log_file = log_file
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as log:
                log.write("step, x, y, zoom, label, image_index\n")

        self.action_space = spaces.MultiDiscrete([
            2, 2, 2, 2,
            2, 2,
            10
        ])

        self.observation_space = spaces.Box(
            0.0, 1.0,
            shape=(6, patch_size, patch_size),
            dtype=np.float32
        )

        self.min_zoom = 1
        self.max_zoom = 4

    def reset(self, *, seed=None, **kwargs):
        super().reset(seed=seed)
        idx = self.np_random.integers(len(self.data))
        self.img = self.data[idx].astype(np.uint8)
        self.label = int(self.labels[idx])
        self.image_index = idx

        self.zoom = 1
        max_off = 32 - self.patch_size * self.zoom
        self.x = self.np_random.integers(max_off)
        self.y = self.np_random.integers(max_off)

        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        mv_up, mv_dn, mv_lf, mv_rt, zi, zo, cls = action
        reward = 0.0
        done = False

        if mv_up:
            self.y = max(0, self.y - 1)
        if mv_dn:
            self.y = min(32 - self.patch_size * self.zoom, self.y + 1)
        if mv_lf:
            self.x = max(0, self.x - 1)
        if mv_rt:
            self.x = min(32 - self.patch_size * self.zoom, self.x + 1)

        if zi:
            self.zoom = min(self.max_zoom, self.zoom * 2)
        if zo:
            self.zoom = max(self.min_zoom, self.zoom // 2)

        self.steps += 1
        if self.steps >= self.max_steps:
            reward = 1.0 if cls == self.label else 0.0
            done = True

        self.log_step()
        return self._get_obs(), reward, done, False, {}

    def log_step(self):
        with open(self.log_file, 'a') as log:
            log.write(f"{self.steps}, {self.x}, {self.y}, {self.zoom}, {self.label}, {self.image_index}\n")

    def _get_patch(self):
        p = self.patch_size * self.zoom
        sub = self.img[self.y:self.y + p, self.x:self.x + p]
        sub = cv2.resize(sub, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
        patch = sub.transpose(2, 0, 1).astype(np.float32) / 255.0
        return patch

    def _get_obs(self):
        patch = self._get_patch()
        H, W = self.patch_size, self.patch_size
        max_off = 32 - self.patch_size * self.zoom

        x_plane = np.full((1, H, W), self.x / max_off if max_off > 0 else 0, dtype=np.float32)
        y_plane = np.full((1, H, W), self.y / max_off if max_off > 0 else 0, dtype=np.float32)
        z_plane = np.full((1, H, W), (self.zoom - self.min_zoom) / (self.max_zoom - self.min_zoom), dtype=np.float32)

        return np.concatenate([patch, x_plane, y_plane, z_plane], axis=0)

def make_env(data, labels, patch_size, max_steps):
    return CIFAR10WindowEnv(data, labels, patch_size, max_steps)

train_ds = datasets.CIFAR10(root="./data", train=True, download=True)
train_data = train_ds.data
train_labels = np.array(train_ds.targets)

test_ds = datasets.CIFAR10(root="./data", train=False, download=True)
test_data = test_ds.data
test_labels = np.array(test_ds.targets)

n_envs = 1024
patch_size = 8
max_steps = 20
rollouts_per_env = 128
total_steps = 100_000_000
update_epochs = 8
minibatch_size = 4096
gamma = 0.97
gae_lambda = 0.95
clip_eps = 0.2
policy_lr = 1e-4
value_lr = 1e-4

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", device)

log_file = "agent_movement_log_3.log"
train_env_fns = [
    (lambda: CIFAR10WindowEnv(train_data, train_labels, patch_size, max_steps, log_file))
    for _ in range(n_envs)
]
train_envs = SyncVectorEnv(train_env_fns)

obs_shape = train_envs.single_observation_space.shape
nvec = train_envs.single_action_space.nvec
act_dim = int(nvec.sum())

n_eval_envs = 8
train_eval_fns = [
    (lambda: make_env(train_data, train_labels, patch_size, max_steps))
    for _ in range(n_eval_envs)
]
train_eval_envs = SyncVectorEnv(train_eval_fns)

test_eval_fns = [
    (lambda: make_env(test_data, test_labels, patch_size, max_steps))
    for _ in range(n_eval_envs)
]
test_eval_envs = SyncVectorEnv(test_eval_fns)

class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_shape, nvec, hidden_size=256, lstm_layers=1):
        super().__init__()
        C, H, W = obs_shape
        self.nvec = nvec
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * H * W, hidden_size),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers
        )
        self.policy = nn.Linear(hidden_size, int(nvec.sum()))
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x, hx, cx):
        seq_len, bsz, C, H, W = x.shape
        flat = x.view(seq_len * bsz, C, H, W)
        feat = self.encoder(flat)
        feat = feat.view(seq_len, bsz, -1)
        out, (hn, cn) = self.lstm(feat, (hx, cx))
        last = out[-1]
        logits = self.policy(last)
        value = self.value(last).squeeze(-1)
        return logits, value, (hn, cn)

class MultiCategorical:
    def __init__(self, logits, nvec):
        self.nvec = nvec
        splits = torch.split(logits, nvec.tolist(), dim=-1)
        self.dists = [Categorical(logits=s) for s in splits]

    def sample(self):
        samples = [d.sample() for d in self.dists]
        return torch.stack(samples, dim=-1)

    def log_prob(self, actions):
        logps = [d.log_prob(actions[..., i]) for i, d in enumerate(self.dists)]
        return torch.stack(logps, dim=-1).sum(dim=-1)

    def entropy(self):
        ents = [d.entropy() for d in self.dists]
        return torch.stack(ents, dim=-1).sum(dim=-1)

model = RecurrentActorCritic(obs_shape, nvec).to(device)
optimizer = optim.Adam(model.parameters(), lr=policy_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps // (n_envs * rollouts_per_env)
)

def rollout(envs, model, rollouts):
    obs_np, _ = envs.reset()
    obs = torch.tensor(obs_np, device=device, dtype=torch.float32)
    obs = obs.view(envs.num_envs, *obs_shape)

    num_layers = model.lstm.num_layers
    hidden_size = model.lstm.hidden_size
    hx = torch.zeros(num_layers, envs.num_envs, hidden_size, device=device)
    cx = torch.zeros_like(hx)

    mb = {"obs": [], "acts": [], "logps": [], "vals": [],
          "rews": [], "dns": [], "hxs": [], "cxs": []}

    for _ in range(rollouts):
        mb["hxs"].append(hx)
        mb["cxs"].append(cx)

        x_in = obs.unsqueeze(0)
        logits, vals, (hx, cx) = model(x_in, hx, cx)

        dist = MultiCategorical(logits, nvec)
        acts = dist.sample()
        logps = dist.log_prob(acts)

        next_obs_np, rews, terms, truns, _ = envs.step(acts.cpu().numpy())
        done = terms | truns

        mb["obs"].append(obs)
        mb["acts"].append(acts)
        mb["logps"].append(logps)
        mb["vals"].append(vals)
        mb["rews"].append(torch.tensor(rews, device=device, dtype=torch.float32))
        mb["dns"].append(torch.tensor(done.astype(np.float32), device=device))

        mask = (1.0 - torch.tensor(done, device=device, dtype=torch.float32)).view(1, envs.num_envs, 1)
        hx = hx * mask
        cx = cx * mask

        obs = torch.tensor(next_obs_np, device=device, dtype=torch.float32)
        obs = obs.view(envs.num_envs, *obs_shape)

    x_in = obs.unsqueeze(0)
    _, last_vals, _ = model(x_in, hx, cx)
    return mb, last_vals

def compute_gae(mb, last_vals, num_envs):
    mb_adv = []
    adv = torch.zeros(num_envs, device=device)
    values = mb["vals"] + [last_vals]
    for t in reversed(range(len(mb["rews"]))):
        mask = (1.0 - mb["dns"][t])
        delta = mb["rews"][t] + gamma * values[t + 1] * mask - values[t]
        adv = delta + gamma * gae_lambda * mask * adv
        mb_adv.insert(0, adv)
    mb_ret = [a + v for a, v in zip(mb_adv, mb["vals"])]
    return mb_ret, mb_adv

def evaluate_accuracy(envs, model, episodes=100):
    obs_np, _ = envs.reset()
    obs = torch.tensor(obs_np, device=device, dtype=torch.float32)
    obs = obs.view(envs.num_envs, *obs_shape)

    num_layers = model.lstm.num_layers
    hidden_size = model.lstm.hidden_size
    hx = torch.zeros(num_layers, envs.num_envs, hidden_size, device=device)
    cx = torch.zeros_like(hx)

    correct = 0
    total = 0
    while total < episodes:
        with torch.no_grad():
            x_in = obs.unsqueeze(0)
            logits, _, (hx, cx) = model(x_in, hx, cx)
            dist = MultiCategorical(logits, nvec)
            acts = dist.sample()
        next_obs_np, rews, terms, truns, _ = envs.step(acts.cpu().numpy())
        done = terms | truns
        correct += float(np.sum(rews))
        total += int(np.sum(done))
        mask = (1.0 - torch.tensor(done, device=device, dtype=torch.float32)).view(1, envs.num_envs, 1)
        hx = hx * mask
        cx = cx * mask
        obs = torch.tensor(next_obs_np, device=device, dtype=torch.float32)
        obs = obs.view(envs.num_envs, *obs_shape)
    return correct / max(total, 1)

steps = 0
pbar = tqdm(total=total_steps, desc="PPO on CIFAR10")

log_metrics_file = "P3_training_metrics.log"

# Initialize the log file with headers if it does not exist
if not os.path.exists(log_metrics_file):
    with open(log_metrics_file, 'w') as log_file:
        log_file.write("steps | policy_loss | value_loss | entropy | train_acc | test_acc\n")
        
entropy_coeffs = np.array([0.05, 0.05, 0.05, 0.05, 0.005, 0.005, 0.1])

while steps < total_steps:
    mb, last_vals = rollout(train_envs, model, rollouts_per_env)
    mb_ret, mb_adv = compute_gae(mb, last_vals, train_envs.num_envs)

    obs_b = torch.cat(mb["obs"])
    acts_b = torch.cat(mb["acts"])
    old_lp = torch.cat(mb["logps"]).detach()
    ret_b = torch.cat(mb_ret).detach()
    adv_b = torch.cat(mb_adv).detach()
    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

    hxs = torch.stack(mb["hxs"], dim=1)
    cxs = torch.stack(mb["cxs"], dim=1)
    layers, T, B, H = hxs.shape
    hxs = hxs.reshape(layers, T * B, H)
    cxs = cxs.reshape(layers, T * B, H)

    batch_size = obs_b.size(0)
    idxs = np.arange(batch_size)

    for _ in range(update_epochs):
        np.random.shuffle(idxs)
        for start in range(0, batch_size, minibatch_size):
            mb_idx = idxs[start:start + minibatch_size]
            o_mb = obs_b[mb_idx]
            a_mb = acts_b[mb_idx]
            r_mb = ret_b[mb_idx]
            adv_mb = adv_b[mb_idx]
            old_lp_mb = old_lp[mb_idx]
            hx_mb = hxs[:, mb_idx].detach()
            cx_mb = cxs[:, mb_idx].detach()

            x_in = o_mb.unsqueeze(0)
            logits, vals, _ = model(x_in, hx_mb, cx_mb)
            dist = MultiCategorical(logits, nvec)

            lp = dist.log_prob(a_mb)
            ratio = (lp - old_lp_mb).exp()

            loss_p = -torch.min(ratio * adv_mb,
                                torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_mb).mean()
            loss_v = (r_mb - vals).pow(2).mean()
            entropy_vals = torch.stack([d.entropy() for d in dist.dists], dim=-1)
            entropy_coeffs_tensor = torch.tensor(entropy_coeffs, device=device).float()
            loss_e = -(entropy_vals * entropy_coeffs_tensor).mean()

            total_loss = loss_p + loss_v + loss_e

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

    steps += train_envs.num_envs * rollouts_per_env
    pbar.update(train_envs.num_envs * rollouts_per_env)

    train_acc = evaluate_accuracy(train_eval_envs, model, episodes=100)
    test_acc = evaluate_accuracy(test_eval_envs, model, episodes=100)
    # Log metrics to the .log file
    with open(log_metrics_file, 'a') as log_file:
        log_file.write(f"{steps} | {loss_p.item():.3f} | {loss_v.item():.3f} | "
                       f"{dist.entropy().mean().item():.3f} | {train_acc * 100:.1f}% | "
                       f"{test_acc * 100:.1f}%\n")
    pbar.set_postfix({
        "policy_loss": f"{loss_p.item():.3f}",
        "value_loss": f"{loss_v.item():.3f}",
        "entropy": f"{dist.entropy().mean().item():.3f}",
        "train_acc": f"{train_acc*100:.1f}%",
        "test_acc": f"{test_acc*100:.1f}%"
    })

pbar.close()
train_envs.close()
train_eval_envs.close()
test_eval_envs.close()
print("Training finished")
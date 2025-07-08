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
from torchvision import datasets
from tqdm.auto import tqdm
import cv2

class CIFAR10WindowEnv(gym.Env):
    def __init__(self, data, labels, patch_size=8, max_steps=15):
        super().__init__()
        self.data = data
        self.labels = labels
        self.patch_size = patch_size
        self.max_steps = max_steps
        # 4 movements + zoom in/out + 10 classes
        self.action_space = gym.spaces.Discrete(4 + 2 + 10)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(3, patch_size, patch_size),
            dtype=np.float32
        )
        self.min_zoom = 1
        self.max_zoom = 4
        self.zoom = 1

    def reset(self, *, seed=None, **kwargs):
        super().reset(seed=seed)
        idx = self.np_random.integers(len(self.data))
        img = self.data[idx]
        self.img = img.astype(np.uint8)
        self.label = int(self.labels[idx])
        self.x = (32 - self.patch_size) // 2
        self.y = (32 - self.patch_size) // 2
        self.steps = 0
        self.zoom = 1
        return self._get_patch(), {}

    def step(self, action):
        done = False
        reward = 0.0
        info = {}

        # classification
        if action >= 6:
            pred = action - 6
            reward = 1.0 if pred == self.label else 0.0
            done = True
            return self._get_patch(), reward, done, False, info

        # movements
        if action == 0:
            self.y = max(0, self.y - 1)
        elif action == 1:
            self.y = min(32 - self.patch_size * self.zoom, self.y + 1)
        elif action == 2:
            self.x = max(0, self.x - 1)
        elif action == 3:
            self.x = min(32 - self.patch_size * self.zoom, self.x + 1)
        # zoom in / zoom out
        elif action == 4:
            self.zoom = min(self.max_zoom, self.zoom * 2)
        elif action == 5:
            self.zoom = max(self.min_zoom, self.zoom // 2)

        self.steps += 1
        if self.steps >= self.max_steps:
            flat = self._get_patch().reshape(-1)
            pred = int(flat.argmax() % 10)
            reward = 1.0 if pred == self.label else 0.0
            done = True

        return self._get_patch(), reward, done, False, info

    def _get_patch(self):
        p = self.patch_size * self.zoom
        patch = self.img[self.y:self.y+p, self.x:self.x+p, :]
        patch = cv2.resize(patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
        patch = patch.transpose(2, 0, 1)  # HWC -> CHW
        return patch.astype(np.float32) / 255.0

    def render(self, mode="human", zoom_win: int = 4):
        full = self.img.copy()
        p = self.patch_size * self.zoom
        x1, y1 = self.x, self.y
        x2, y2 = x1 + p, y1 + p
        cv2.rectangle(full, (x1, y1), (x2, y2), (0, 255, 0), 1)
        patch = (self._get_patch().transpose(1, 2, 0) * 255).astype(np.uint8)
        patch_zoom = cv2.resize(patch, (p * zoom_win, p * zoom_win), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Full Image", cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
        cv2.imshow(f"Patch x{zoom_win}", cv2.cvtColor(patch_zoom, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


train_ds = datasets.CIFAR10(root="./data", train=True, download=True)
train_data = train_ds.data
train_labels = np.array(train_ds.targets)

test_ds = datasets.CIFAR10(root="./data", train=False, download=True)
test_data = test_ds.data
test_labels = np.array(test_ds.targets)

n_envs = 16
patch_size = 8
max_steps = 40
rollouts_per_env = 200
total_steps = 100_000_000
update_epochs = 4
minibatch_size = 1024
gamma = 0.99
gae_lambda = 0.95
clip_eps = 0.2
policy_lr = 1e-4
value_lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_env_fns = [
    (lambda seed=1000 + i: lambda: CIFAR10WindowEnv(train_data, train_labels, patch_size, max_steps))()
    for i in range(n_envs)
]
train_envs = AsyncVectorEnv(train_env_fns)
obs_shape = train_envs.single_observation_space.shape
obs_dim = int(np.prod(obs_shape))
act_dim = train_envs.single_action_space.n

n_eval_envs = 8
train_eval_fns = [
    (lambda seed=2000 + i: lambda: CIFAR10WindowEnv(train_data, train_labels, patch_size, max_steps))()
    for i in range(n_eval_envs)
]
train_eval_envs = AsyncVectorEnv(train_eval_fns)

test_eval_fns = [
    (lambda seed=3000 + i: lambda: CIFAR10WindowEnv(test_data, test_labels, patch_size, max_steps))()
    for i in range(n_eval_envs)
]
test_eval_envs = AsyncVectorEnv(test_eval_fns)

class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_shape, act_dim, hidden_size=256, lstm_layers=1):
        super().__init__()
        C, H, W = obs_shape
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
        self.policy = nn.Linear(hidden_size, act_dim)
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

model = RecurrentActorCritic(obs_shape, act_dim).to(device)
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

        dist = Categorical(logits=logits)
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
            dist = Categorical(logits=logits)
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
            dist = Categorical(logits=logits)

            lp = dist.log_prob(a_mb)
            ratio = (lp - old_lp_mb).exp()

            loss_p = -torch.min(ratio * adv_mb,
                                torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_mb).mean()
            loss_v = (r_mb - vals).pow(2).mean()
            loss_e = -dist.entropy().mean() * 0.01
            total_loss = loss_p + loss_v + loss_e

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()

    steps += train_envs.num_envs * rollouts_per_env
    pbar.update(train_envs.num_envs * rollouts_per_env)

    train_acc = evaluate_accuracy(train_eval_envs, model, episodes=200)
    test_acc = evaluate_accuracy(test_eval_envs, model, episodes=200)
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
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

class CIFAR10WindowEnv(gym.Env):
    def __init__(self, data, labels, patch_size=8, max_steps=15):
        super().__init__()
        self.data = data
        self.labels = labels
        self.patch_size = patch_size
        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(4 + 10)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(patch_size, patch_size, 3),
            dtype=np.float32
        )

    def reset(self, *, seed=None, **kwargs):
        super().reset(seed=seed)
        idx = self.np_random.integers(len(self.data))
        self.img   = self.data[idx]
        self.label = int(self.labels[idx])
        self.x = (32 - self.patch_size) // 2
        self.y = (32 - self.patch_size) // 2
        self.steps = 0
        return self._get_patch(), {}

    def step(self, action):
        done = False
        reward = 0.0
        info = {}

        # классификация
        if action >= 4:
            pred = action - 4
            reward = 1.0 if pred == self.label else 0.0
            done = True
            return self._get_patch(), reward, done, False, info

        # движение
        if action == 0:
            self.y = max(0, self.y - 1)
        elif action == 1:
            self.y = min(32 - self.patch_size, self.y + 1)
        elif action == 2:
            self.x = max(0, self.x - 1)
        elif action == 3:
            self.x = min(32 - self.patch_size, self.x + 1)

        self.steps += 1
        if self.steps >= self.max_steps:
            flat = self._get_patch().reshape(-1)
            pred = int(flat.argmax() % 10)
            reward = 1.0 if pred == self.label else 0.0
            done = True

        return self._get_patch(), reward, done, False, info

    def _get_patch(self):
        p = self.patch_size
        patch = self.img[self.y:self.y+p, self.x:self.x+p, :]
        return patch.astype(np.float32) / 255.0

train_ds = datasets.CIFAR10(root="./data", train=True, download=True)
data   = train_ds.data
labels = np.array(train_ds.targets)

n_envs           = 16
patch_size       = 8
max_steps        = 30
rollouts_per_env = 100
total_steps      = 20_000_000
update_epochs    = 4
minibatch_size   = 256
gamma            = 0.99
gae_lambda       = 0.95
clip_eps         = 0.2
policy_lr        = 1e-4
value_lr         = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

env_fns = [
    (lambda seed=1000+i: lambda: CIFAR10WindowEnv(data, labels, patch_size, max_steps))()
    for i in range(n_envs)
]
envs = AsyncVectorEnv(env_fns)
obs_shape = envs.single_observation_space.shape
obs_dim   = int(np.prod(obs_shape))
act_dim   = envs.single_action_space.n

class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256, lstm_layers=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers
        )
        self.policy = nn.Linear(hidden_size, act_dim)
        self.value  = nn.Linear(hidden_size, 1)

    def forward(self, x, hx, cx):
        seq_len, bsz, _ = x.shape
        flat = x.reshape(seq_len * bsz, -1)
        feat = self.encoder(flat)
        feat = feat.view(seq_len, bsz, -1)
        out, (hn, cn) = self.lstm(feat, (hx, cx))
        last = out[-1]
        logits = self.policy(last)
        value  = self.value(last).squeeze(-1)
        return logits, value, (hn, cn)


model      = RecurrentActorCritic(obs_dim, act_dim).to(device)
opt_policy = optim.AdamW(model.policy.parameters(), lr=policy_lr)
opt_value  = optim.AdamW(model.value.parameters(),  lr=value_lr)

def rollout(envs, model, rollouts):
    obs_np, _ = envs.reset()
    obs = torch.tensor(obs_np, device=device, dtype=torch.float32).view(n_envs, obs_dim)

    num_layers  = model.lstm.num_layers
    hidden_size = model.lstm.hidden_size
    hx = torch.zeros(num_layers, n_envs, hidden_size, device=device)
    cx = torch.zeros_like(hx)
    mb = {'obs':[], 'acts':[], 'logps':[], 'vals':[],
          'rews':[], 'dns':[], 'hxs':[], 'cxs':[]}
    for _ in range(rollouts):
        mb['hxs'].append(hx)
        mb['cxs'].append(cx)
        logits, vals, (hx, cx) = model(obs.unsqueeze(0), hx, cx)
        dist = Categorical(logits=logits)
        acts = dist.sample()
        logps = dist.log_prob(acts)
        next_obs_np, rews, terms, truns, _ = envs.step(acts.cpu().numpy())
        done = terms | truns
        mb['obs'].append(obs)
        mb['acts'].append(acts)
        mb['logps'].append(logps)
        mb['vals'].append(vals)
        mb['rews'].append(torch.tensor(rews, device=device, dtype=torch.float32))
        mb['dns'].append(torch.tensor(done.astype(np.float32), device=device))
        mask = (1.0 - torch.tensor(done, device=device, dtype=torch.float32))
        mask = mask.view(1, -1, 1)
        hx = hx * mask
        cx = cx * mask
        obs = torch.tensor(next_obs_np, device=device, dtype=torch.float32).view(n_envs, obs_dim)
    _, last_vals, _ = model(obs.unsqueeze(0), hx, cx)
    return mb, last_vals

def compute_gae(mb, last_vals):
    mb_adv = []
    adv = torch.zeros(n_envs, device=device)
    values = mb['vals'] + [last_vals]
    for t in reversed(range(len(mb['rews']))):
        mask = 1.0 - mb['dns'][t]
        delta = mb['rews'][t] + gamma * values[t+1] * mask - values[t]
        adv = delta + gamma * gae_lambda * mask * adv
        mb_adv.insert(0, adv)
    mb_ret = [a + v for a, v in zip(mb_adv, mb['vals'])]
    return mb_ret, mb_adv
steps = 0
pbar = tqdm(total=total_steps, desc="PPO CIFAR10")
while steps < total_steps:
    mb, last_vals = rollout(envs, model, rollouts_per_env)
    mb_ret, mb_adv = compute_gae(mb, last_vals)

    obs_b  = torch.cat(mb['obs'])
    acts_b = torch.cat(mb['acts'])
    old_lp = torch.cat(mb['logps']).detach()
    ret_b  = torch.cat(mb_ret).detach()
    adv_b  = torch.cat(mb_adv).detach()
    adv_b  = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
    rews_all = torch.cat(mb['rews']).cpu()
    done_all = torch.cat(mb['dns']).cpu()
    episodes = done_all.sum().item()
    correct  = rews_all.sum().item()
    acc = correct / episodes if episodes > 0 else 0.0
    hxs = torch.stack(mb['hxs'], dim=1) 
    cxs = torch.stack(mb['cxs'], dim=1)
    layers, T, B, H = hxs.shape
    hxs = hxs.permute(0,1,2,3).reshape(layers, T*B, H)
    cxs = cxs.permute(0,1,2,3).reshape(layers, T*B, H)

    batch_size = obs_b.size(0)
    idxs = np.arange(batch_size)

    for _ in range(update_epochs):
        np.random.shuffle(idxs)
        for start in range(0, batch_size, minibatch_size):
            mb_idx = idxs[start:start+minibatch_size]
            o_mb = obs_b[mb_idx]
            a_mb = acts_b[mb_idx]
            r_mb = ret_b[mb_idx]
            adv_mb = adv_b[mb_idx]
            old_lp_mb = old_lp[mb_idx]
            hx_mb = hxs[:, mb_idx].detach() # detach hidden-state
            cx_mb = cxs[:, mb_idx].detach()
            logits, vals, _ = model(o_mb.unsqueeze(0), hx_mb, cx_mb)
            dist = Categorical(logits=logits)
            lp = dist.log_prob(a_mb)
            ratio = (lp - old_lp_mb).exp()
            s1 = ratio * adv_mb
            s2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv_mb
            loss_p = -torch.min(s1, s2).mean()
            loss_v = (r_mb - vals).pow(2).mean()
            loss_e = -dist.entropy().mean() * 0.01
            total_loss = loss_p + loss_v + loss_e
            #print(f"[Batch {start}-{start+len(mb_idx)}] total_loss = {total_loss.item():.6f}")
            opt_policy.zero_grad()
            opt_value.zero_grad()
            total_loss.backward()
            opt_policy.step()
            opt_value.step()

    steps += n_envs * rollouts_per_env
    pbar.update(n_envs * rollouts_per_env)
    pbar.set_postfix({
        "policy_loss": loss_p.item(),
        "value_loss" : loss_v.item(),
        "accuracy"   : f"{acc:.3f}"
    })

pbar.close()
envs.close()
print("Training finished")



import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm import tqdm
from gym.vector import SyncVectorEnv
import gym
from functools import lru_cache
from einops import rearrange
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CFG = {
    'env_steps': 10_000_000,
    'rollout_length': 128,
    'update_epochs': 4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.1,  
    'lr': 1e-5,          
    'batch_size': 128,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 4,
    'initial_patch': 16,
    'min_patch': 8,
    'max_patch': 32,
    'entropy_coef': 0.01,
    'checkpoint_interval': 5_000,
    'eval_interval': 10_000,
    'max_episode_steps': 100,
    'dataset_seed': 80411,
    'n_envs': 16,
    'n_move': 4,
}

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'avg_return', 'goal': 'maximize'},
    'parameters': {
        'lr': {'min': 1e-5, 'max': 1e-4},
        'entropy_coef': {'min': 1e-2, 'max': 1e-1},
        'batch_size': {'values': [128]},
        'update_epochs': {'values': [2, 4]},
        'rollout_length': {'values': [128]},
        'clip_epsilon': {'min': 0.1, 'max': 0.3},
        'gamma': {'min': 0.9, 'max': 0.999},
        'gae_lambda': {'min': 0.8, 'max': 0.99},
        'correct_bonus': {'values': [1, 2, 3]},
        'wrong_penalty': {'values': [0]},
        'move_penalty': {'values': [0]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="cifar_active_sweep")

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

@dataclass
class DatasetSplit:
    images: np.ndarray
    targets: np.ndarray
    image_shape: tuple[int, int, int]

    def __len__(self):
        return len(self.targets)

    @property
    def n_classes(self):
        return len(np.unique(self.targets))

class Dataset:
    def __init__(self, seed: int, cache_path: str = 'cifar10.npz'):
        if os.path.exists(cache_path):
            logger.info(f"Loading CIFAR-10 from local cache `{cache_path}`...")
            data = np.load(cache_path)
            X, y = data['X'], data['y']
        else:
            logger.info("Downloading CIFAR-10 from OpenML...")
            X, y = fetch_openml('CIFAR_10', version=1, return_X_y=True, as_frame=False)
            X = X.astype(np.float32)
            y = y.astype(int)
            np.savez(cache_path, X=X, y=y)
            logger.info(f"Saved CIFAR-10 to local cache `{cache_path}`")

        X = X / 255.0 
        shape = (3, 32, 32)  
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=10000, random_state=seed
        )
        self.train = DatasetSplit(train_X, train_y, shape)
        self.test  = DatasetSplit(test_X,  test_y,  shape)
        logger.info(f"CIFAR-10 ready: train {train_X.shape}, test {test_X.shape}")


_ds = Dataset(25151)
train_imgs = _ds.train.images.reshape(-1, *_ds.train.image_shape)
images = torch.tensor(train_imgs, dtype=torch.float32, device=device)
targets = _ds.train.targets
logger.info(f"Dataset: images {images.shape}, targets {len(targets)} entries")


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0, "embed_dim must be even"
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    assert embed_dim % 2 == 0, "embed_dim must be even"
    dim_half = embed_dim // 2
    gh = np.arange(grid_size, dtype=np.float32)
    gw = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(gw, gh), axis=0).reshape(2, -1)
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_half, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_half, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

def build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    np_pe = get_2d_sincos_pos_embed(embed_dim, grid_size)
    return torch.from_numpy(np_pe).float()

class PatchEmbedding(nn.Module):
    def __init__(self, in_ch: int, p_size: int, d_model: int, img_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=p_size, stride=p_size)
        n_patches = (img_size // p_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        pe = build_2d_sincos_pos_embed(d_model, img_size // p_size)
        pe = torch.cat([torch.zeros(1, d_model), pe], dim=0)
        self.register_buffer('pos_embed', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self.pos_embed


class ViT(nn.Module):
    def __init__(self, in_ch: int, img_size: int, p_size: int, d_model: int, n_layers: int, n_heads: int, n_actions: int):
        super().__init__()
        self.patch_emb = PatchEmbedding(in_ch, p_size, d_model, img_size)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.actor = nn.Linear(d_model, n_actions)
        self.critic = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor = None):  
        tokens = self.patch_emb(x)
        out = self.transformer(tokens)
        cls = out[:, 0]
        return self.actor(cls), self.critic(cls).squeeze(-1)
        
def compute_ppo_loss(logits, values, actions, old_log_probs, advantages, returns, entropy_coef, clip_epsilon):
    dist = torch.distributions.Categorical(logits.softmax(-1))
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    ratio = torch.exp(new_log_probs - old_log_probs)
    clip_advantage = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(ratio * advantages, clip_advantage).mean()
    value_loss = F.mse_loss(values, returns)
    
    total_loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
    return total_loss, policy_loss, value_loss, entropy

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                gamma: float, lam: float):
    T, N = rewards.shape
    advantages = torch.zeros((T, N), device=device)
    last_adv = torch.zeros(N, device=device)
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + values[:-1]
    return advantages, returns

class CIFAREnv(gym.Env):
    def __init__(self, data, labels, init_patch, min_patch, max_patch, max_steps, n_cls):
        super().__init__()
        self.data, self.labels = data, labels
        self.N = data.size(0)
        self.init_patch = init_patch
        self.patch_size = init_patch
        self.min_patch = min_patch
        self.max_patch = max_patch
        self.max_steps = max_steps
        self.n_move = CFG['n_move']
        self.action_space = gym.spaces.Discrete(self.n_move + n_cls)  # Дискретное пространство действий
        C, _, _ = data.shape[1:]
        self.observation_space = gym.spaces.Box(0.0, 1.0, (C, init_patch, init_patch), dtype=np.float32)
        self.ptr = 0
        self.model = None

    def reset(self):
        idx = self.ptr % self.N
        self.ptr += 1
        self.img = self.data[idx]
        self.label = int(self.labels[idx])
        self.patch_size = self.init_patch
        self.steps = 0
        self.y = np.random.randint(0, 32 - self.patch_size)
        self.x = np.random.randint(0, 32 - self.patch_size)
        return self._get_patch(), {}

    def _get_patch(self):
        H, W = self.img.shape[1:]
        self.x = max(0, min(self.x, W - self.patch_size))
        self.y = max(0, min(self.y, H - self.patch_size))
        patch = self.img[:, self.y:self.y + self.patch_size, self.x:self.x + self.patch_size].unsqueeze(0)
        patch = F.interpolate(patch, size=self.init_patch, mode='bilinear', align_corners=False)[0]
        return patch.cpu().numpy()

    def adjust_position(self, action):
        H, W = self.img.shape[1:]
        patch_size = self.patch_size
        if action == 0:
            self.y = max(0, self.y - patch_size)
        elif action == 1:
            self.y = min(H - patch_size, self.y + patch_size)
        elif action == 2:
            self.x = max(0, self.x - patch_size)
        elif action == 3:
            self.x = min(W - patch_size, self.x + patch_size)

    def step(self, action):
        cfg = wandb.config
        self.steps += 1
        external_reward = 0.0
        done = False

        if action < self.n_move:
            self.adjust_position(action)
            external_reward -= cfg.move_penalty
        else:
            pred = action - self.n_move
            if pred == self.label:
                external_reward += cfg.correct_bonus
            else:
                external_reward -= cfg.wrong_penalty
            done = True

        obs_raw = self.img[:, self.y:self.y + self.patch_size, self.x:self.x + self.patch_size].unsqueeze(0)
        obs_raw = F.interpolate(obs_raw, size=self.init_patch, mode='bilinear', align_corners=False)[0]
        t_obs = obs_raw.unsqueeze(0)
        time_tensor = torch.tensor([self.steps], device=t_obs.device, dtype=torch.long)
        logits, vals = self.model(t_obs, time_tensor)
        pred = logits.argmax(dim=-1)
        target = torch.zeros_like(logits)
        target.scatter_(1, pred.unsqueeze(-1), 1)
        loss = F.mse_loss(logits, target)

        total_reward = external_reward
        obs = self._get_patch()
        truncated = (self.steps >= self.max_steps)
        return obs, total_reward, done, truncated, {}

def make_env(seed):
    def thunk():
        return CIFAREnv(images, targets, CFG['initial_patch'], CFG['min_patch'], CFG['max_patch'], CFG['max_episode_steps'], _ds.train.n_classes)
    return thunk

envs = SyncVectorEnv([make_env(i) for i in range(CFG['n_envs'])])

def train_ppo(agent):
    wandb.init()
    cfg = wandb.config
    opt = optim.AdamW(agent.parameters(), lr=cfg.lr)
    buf = {k: [] for k in ['obs', 'act', 'logp', 'val', 'rew', 'done', 'time']}
    step_count, ep_count = 0, 0
    ep_rewards, ep_lengths = [], []
    cur_rewards = np.zeros(CFG['n_envs'], dtype=float)
    cur_lengths = np.zeros(CFG['n_envs'], dtype=int)
    next_log = 1000

    obs_batch, _ = envs.reset()
    obs_batch = torch.tensor(obs_batch, device=device)
    time_batch = torch.zeros(CFG['n_envs'], dtype=torch.long, device=device)
    pbar = tqdm(total=CFG['env_steps'], desc='Training')

    while step_count < CFG['env_steps']:
        #rollout
        for _ in range(cfg.rollout_length):
            logits, vals = agent(obs_batch, time_batch)
            dist = torch.distributions.Categorical(logits.softmax(-1))
            acts = dist.sample()
            lp = dist.log_prob(acts)
            nxt, rews, term, tru, _ = envs.step(acts.cpu().numpy())
            dones = np.logical_or(term, tru)

            cur_rewards += rews
            cur_lengths += 1
            for i, done in enumerate(dones):
                if done:
                    ep_rewards.append(cur_rewards[i])
                    ep_lengths.append(cur_lengths[i])
                    cur_rewards[i] = 0.0
                    cur_lengths[i] = 0

            buf['obs'].append(obs_batch)
            buf['act'].append(acts)
            buf['logp'].append(lp.detach())
            buf['val'].append(vals.detach())
            buf['rew'].append(torch.tensor(rews, device=device))
            buf['done'].append(torch.tensor(dones, device=device, dtype=torch.float32))
            buf['time'].append(time_batch)

            obs_batch = torch.tensor(nxt, device=device)
            time_batch = torch.where(torch.tensor(dones, device=device), torch.zeros_like(time_batch), time_batch + 1)
            step_count += CFG['n_envs']
            pbar.update(CFG['n_envs'])
            if dones.any():
                ep_count += dones.sum().item()
                pbar.set_postfix(ep=int(ep_count))

        _, last_val = agent(obs_batch, time_batch)
        buf['val'].append(last_val.detach())

        rews = torch.stack(buf['rew'])
        vals = torch.stack(buf['val'])
        dones_vec = torch.stack(buf['done'])
        advs, rets = compute_gae(rews, vals, dones_vec, cfg.gamma, cfg.gae_lambda)
        rets = rets.detach().reshape(-1)
        advs = advs.detach().reshape(-1)

        obs_f = torch.cat(buf['obs'])
        act_f = torch.cat(buf['act'])
        lp_f = torch.cat(buf['logp'])
        time_f = torch.cat(buf['time'])

        tot_pl = tot_vl = tot_ent = cnt = 0
        correct_predictions = 0

        for _ in range(cfg.update_epochs):
            perm = torch.randperm(len(obs_f))
            for i in range(0, len(obs_f), cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                b_obs = obs_f[idx]
                b_act = act_f[idx]
                b_old = lp_f[idx]
                b_ret = rets[idx]
                b_adv = advs[idx]
                b_time = time_f[idx]

                l, v = agent(b_obs, b_time)
                d = torch.distributions.Categorical(l.softmax(-1))

                new_lp = d.log_prob(b_act)
                ent = d.entropy().mean()

                ratio = torch.exp(new_lp - b_old)
                clip = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * b_adv

                pl = -torch.min(ratio * b_adv, clip).mean()
                vl = F.mse_loss(v, b_ret)
                loss = pl + 0.5 * vl - cfg.entropy_coef * ent

                opt.zero_grad()
                loss.backward()
                opt.step()

                tot_pl += pl.item()
                tot_vl += vl.item()
                tot_ent += ent.item()
                cnt += 1

                pred_classes = l.argmax(dim=-1)
                correct_predictions += (pred_classes == b_act).sum().item()

        for k in buf:
            buf[k].clear()

        avg_pl = tot_pl / cnt
        avg_vl = tot_vl / cnt
        avg_ent = tot_ent / cnt

        accuracy = correct_predictions / (len(obs_f) * cfg.batch_size)

        wandb.log({
            "step": step_count,
            "policy_loss": avg_pl,
            "value_loss": avg_vl,
            "entropy": avg_ent,
            "accuracy": accuracy
        }, step=step_count)

        if ep_count >= next_log:
            avg_return = np.mean(ep_rewards[-next_log:])
            avg_length = np.mean(ep_lengths[-next_log:])
            wandb.log({
                "avg_return": avg_return,
                "avg_length": avg_length
            }, step=step_count)
            next_log += cfg.rollout_length

    pbar.close()


if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    in_ch = _ds.train.image_shape[0]  
    agent = ViT(in_ch=in_ch,
                img_size=CFG['initial_patch'],
                p_size=CFG['initial_patch'],
                d_model=CFG['d_model'],
                n_layers=CFG['n_layers'],
                n_heads=CFG['n_heads'],
                n_actions=CFG['n_move'] + _ds.train.n_classes).to(device)
    for e in envs.envs:
        e.model = agent
    wandb.agent(sweep_id, function=lambda: train_ppo(agent), count=20)

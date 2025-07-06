import os
import logging
import numpy as np
import math
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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CFG = {
    'env_steps': 300_000,
    'rollout_length': 128,
    'update_epochs': 4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'lr': 1e-4,
    'batch_size': 32,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 14,
    'initial_patch': 16,
    'min_patch': 8,
    'max_patch': 32,
    'intrinsic_coef': 0.01,
    'surprise_coef': 0.1,
    'recon_coef': 0.1,
    'entropy_coef': 0.01,
    'checkpoint_interval': 5_000,
    'eval_interval': 10_000,
    'max_episode_steps': 100,
    'dataset_seed': 80411,
    'n_envs': 16,
    'n_zoom': 2,
    'n_move': 4,
}

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'avg_return', 'goal': 'maximize'},
    'parameters': {
        'lr':            {'min': 1e-5,  'max': 1e-4},
        'entropy_coef':  {'min': 1e-2,  'max': 1e-1},
        'batch_size':    {'values': [128]},
        'update_epochs': {'values': [2, 4]},
        'rollout_length':{'values': [128]},
        'clip_epsilon':  {'min': 0.1,   'max': 0.3},
        'gamma':         {'min': 0.9,   'max': 0.999},
        'gae_lambda':    {'min': 0.8,   'max': 0.99},
        'correct_bonus': {'values': [1, 2]},
        'wrong_penalty': {'values': [0, 1]},
        'zoom_bonus':    {'values': [1, 2]},
        'move_penalty':  {'values': [0, 1]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="cifar_active_sweep")

@dataclass
class DatasetSplit:
    images: np.ndarray
    targets: np.ndarray
    image_shape: tuple[int,int,int]
    def __len__(self): return len(self.targets)
    @property
    def n_classes(self): return len(np.unique(self.targets))

class Dataset:
    def __init__(self, seed: int, cache_path: str = 'cifar10.npz'):
        import os
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split

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

        # normalize and split
        X = X / 255.0
        shape = (3, 32, 32)
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=10000, random_state=seed
        )
        self.train = DatasetSplit(train_X, train_y, shape)
        self.test  = DatasetSplit(test_X,  test_y,  shape)
        logger.info(f"CIFAR-10 ready: train {train_X.shape}, test {test_X.shape}")

_ds = Dataset(CFG['dataset_seed'])
train_imgs = _ds.train.images.reshape(-1, *_ds.train.image_shape)
images = torch.tensor(train_imgs, dtype=torch.float32, device=device)
targets = _ds.train.targets
logger.info(f"Dataset: images {images.shape}, targets {len(targets)} entries")

# Time Positional 2d + Friquence improved Legendre Memory Helpers
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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, T: int) -> torch.Tensor:
    BTF, N, M = x.shape
    B = shift.shape[0]
    x = rearrange(x, '(b t) n m -> b t n m', b=B, t=T)
    x = x * (1 + scale.unsqueeze(1).unsqueeze(2)) + shift.unsqueeze(1).unsqueeze(2)
    return rearrange(x, 'b t n m -> (b t) n m', b=B, t=T)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_emb: int = 256):
        super().__init__()
        self.freq_emb = freq_emb
        self.mlp = nn.Sequential(
            nn.Linear(freq_emb, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    @staticmethod
    def timestep_embedding(t: torch.LongTensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args  = t.unsqueeze(1).float() * freqs.unsqueeze(0)
        emb   = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb
    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        pe = self.timestep_embedding(t, self.freq_emb)
        return self.mlp(pe)

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, drop_p: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(num_classes + int(drop_p > 0), hidden_size)
        self.p     = drop_p
    def forward(self, labels: torch.LongTensor, train: bool, force: torch.LongTensor = None) -> torch.Tensor:
        if (train and self.p > 0) or force is not None:
            if force is None:
                drop_mask = (torch.rand(labels.shape, device=labels.device) < self.p)
            else:
                drop_mask = force.bool()
            labels = torch.where(drop_mask,
                                 torch.tensor(self.embed.num_embeddings - 1, device=labels.device),
                                 labels)
        return self.embed(labels)

@lru_cache()
def build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    np_pe = get_2d_sincos_pos_embed(embed_dim, grid_size)
    return torch.from_numpy(np_pe).float()

#patch embedding + SinCos PE
class PatchEmbedding(nn.Module):
    def __init__(self, in_ch: int, p_size: int, d_model: int, img_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=p_size, stride=p_size)
        n_patches = (img_size // p_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        pe = build_2d_sincos_pos_embed(d_model, img_size // p_size)
        pe = torch.cat([torch.zeros(1, d_model), pe], dim=0)
        self.register_buffer('pos_embed', pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1,2)
        cls = self.cls_token.expand(B,-1,-1)
        x = torch.cat([cls, x], dim=1)
        return x + self.pos_embed

class ViTWithTimeAndFiLM(nn.Module):
    def __init__(self, in_ch: int, img_size: int, p_size: int,
                 d_model: int, n_layers: int, n_heads: int,
                 n_actions: int, max_steps: int,
                 use_film: bool=False, film_classes: int=None, film_drop_p: float=0.0):
        super().__init__()
        self.patch_emb  = PatchEmbedding(in_ch, p_size, d_model, img_size)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads,
                                               dim_feedforward=4*d_model,
                                               batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.time_emb   = TimestepEmbedder(hidden_size=d_model, freq_emb=d_model)
        self.max_steps  = max_steps
        self.use_film   = use_film
        if use_film:
            assert film_classes is not None
            self.shift_embed = LabelEmbedder(film_classes, d_model, drop_p=film_drop_p)
            self.scale_embed = LabelEmbedder(film_classes, d_model, drop_p=film_drop_p)
        self.actor      = nn.Linear(d_model, n_actions)
        self.critic     = nn.Linear(d_model, 1)
        self.pred_patch = nn.Linear(d_model, in_ch * p_size * p_size)
        self.decoder    = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.ReLU(),
            nn.Linear(4*d_model, in_ch * p_size * p_size)
        )

    def forward(self, x: torch.Tensor, t: torch.LongTensor, film_idx: torch.LongTensor=None):
        tokens = self.patch_emb(x)
        t_pe = self.time_emb(t)
        tokens[:,0] = tokens[:,0] + t_pe
        if self.use_film and (film_idx is not None):
            shift = self.shift_embed(film_idx, train=self.training)
            scale = self.scale_embed(film_idx, train=self.training)
            B, N, M = tokens.shape
            tokens = modulate(tokens.view(-1, N, M), shift, scale, T=1).view(B, N, M)
        out = self.transformer(tokens)
        cls = out[:,0]
        return (
            self.actor(cls),
            self.critic(cls).squeeze(-1),
            self.pred_patch(cls),
            self.decoder(cls)
        )


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
        self.n_zoom   = CFG['n_zoom']
        self.n_move   = CFG['n_move']
        self.action_space = gym.spaces.Discrete(self.n_zoom + self.n_move + n_cls)
        C, _, _ = data.shape[1:]
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (C + 2, init_patch, init_patch), dtype=np.float32
        )
        self.ptr = 0
        self.model = None  

    def reset(self):
        idx = self.ptr % self.N
        self.ptr += 1
        self.img = self.data[idx]
        self.label = int(self.labels[idx])
        self.patch_size = self.init_patch
        self.steps = 0
        # Случайные координаты для начала
        self.y = np.random.randint(0, 32 - self.patch_size)
        self.x = np.random.randint(0, 32 - self.patch_size)
        self.last_surprise = 0.0
        self.last_uncertainty = 0.0
        return self._get_patch(), {}

    def _get_patch(self):
        H, W = self.img.shape[1:]
        self.x = max(0, min(self.x, W - self.patch_size))
        self.y = max(0, min(self.y, H - self.patch_size))
        
        # Извлечение патча
        patch = self.img[:, self.y:self.y + self.patch_size, self.x:self.x + self.patch_size].unsqueeze(0)
        
        # Заполнение остальной части изображения нулями
        full_img = torch.zeros(3, 32, 32)
        full_img[:, self.y:self.y + self.patch_size, self.x:self.x + self.patch_size] = patch.squeeze(0)
        
        # Масштабируем патч
        patch = F.interpolate(full_img.unsqueeze(0), size=self.init_patch, mode='bilinear', align_corners=False)[0]
        
        # Добавление "surprise" и "uncertainty"
        extra = torch.tensor([self.last_surprise, self.last_uncertainty], device=patch.device).view(2, 1, 1)
        extra = extra.expand(2, self.init_patch, self.init_patch)
        patch = torch.cat([patch, extra], dim=0)
        return patch.cpu().numpy()

    def adjust_patch_size(self, action):
        """
        Динамическое изменение размера фрагмента в зависимости от действия.
        action: 0 - уменьшить размер, 1 - увеличить размер.
        """
        if action == 0 and self.patch_size > self.min_patch:
            self.patch_size //= 2  # Уменьшаем размер
        elif action == 1 and self.patch_size < self.max_patch:
            self.patch_size *= 2  # Увеличиваем размер

    def adjust_position(self, action):
        """
        Адаптивное изменение позиции фрагмента.
        action: индикатор для направления перемещения.
        """
        H, W = self.img.shape[1:]
        patch_size = self.patch_size
        if action == 0:  # Перемещение вверх
            self.y = max(0, self.y - patch_size)
        elif action == 1:  # Перемещение вниз
            self.y = min(H - patch_size, self.y + patch_size)
        elif action == 2:  # Перемещение влево
            self.x = max(0, self.x - patch_size)
        elif action == 3:  # Перемещение вправо
            self.x = min(W - patch_size, self.x + patch_size)

    def step(self, action):
        cfg = wandb.config
        self.steps += 1
        H, W = self.img.shape[1:]
        external_reward = 0.0
        done = False

        # Увеличение/уменьшение масштаба фрагмента
        if action < self.n_zoom:
            # Масштабирование фрагмента
            self.adjust_patch_size(action)  
            external_reward += cfg.zoom_bonus
            offset = (32 - self.patch_size) // 2
            self.x = self.y = offset
        elif action < self.n_zoom + self.n_move:
            mv = action - self.n_zoom
            # Адаптивное позиционирование фрагмента
            self.adjust_position(mv)  # Позиция
            external_reward -= cfg.move_penalty
        else:
            # Классификация
            pred = action - (self.n_zoom + self.n_move)
            if pred == self.label:
                external_reward += cfg.correct_bonus
            else:
                external_reward -= cfg.wrong_penalty
            done = True

        # Интринсик вознаграждения
        obs_raw = self.img[:, self.y:self.y + self.patch_size, self.x:self.x + self.patch_size].unsqueeze(0)
        obs_raw = F.interpolate(obs_raw, size=self.init_patch, mode='bilinear', align_corners=False)[0]
        extra = torch.tensor([self.last_surprise, self.last_uncertainty], device=obs_raw.device).view(2, 1, 1)
        extra = extra.expand(2, self.init_patch, self.init_patch)
        t_obs = torch.cat([obs_raw, extra], dim=0).unsqueeze(0)
        time_tensor = torch.tensor([self.steps], device=obs_raw.device, dtype=torch.long)
        logits, _, p_pred, dec = self.model(t_obs, time_tensor)
        flat = t_obs.reshape(-1)
        surprise = F.mse_loss(p_pred.squeeze(0), flat)
        dist = torch.distributions.Categorical(logits.softmax(-1))
        uncertainty = dist.entropy().mean()
        recon = F.mse_loss(dec.squeeze(0), flat)
        self.last_surprise = surprise.item()
        self.last_uncertainty = uncertainty.item()

        intrinsic_reward = CFG['intrinsic_coef'] * (
            CFG['surprise_coef'] * surprise.item() +
            CFG['recon_coef']    * recon.item()
        )

        total_reward = external_reward + intrinsic_reward
        obs = self._get_patch()
        truncated = (self.steps >= self.max_steps)
        return obs, total_reward, done, truncated, {}


def make_env(seed):
    def thunk():
        return CIFAREnv(images, targets,
                        CFG['initial_patch'], CFG['min_patch'],
                        CFG['max_patch'], CFG['max_episode_steps'], _ds.train.n_classes)
    return thunk

envs = SyncVectorEnv([make_env(i) for i in range(CFG['n_envs'])])

def train_ppo(agent):
    wandb.init()
    cfg = wandb.config
    opt = optim.AdamW(agent.parameters(), lr=cfg.lr)
    buf = {k: [] for k in ['obs','act','logp','val','rew','done','time']}
    step_count, ep_count = 0, 0
    ep_rewards, ep_lengths = [], []
    cur_rewards = np.zeros(CFG['n_envs'], dtype=float)
    cur_lengths = np.zeros(CFG['n_envs'], dtype=int)
    next_log = 1000

    obs_batch, _ = envs.reset()
    obs_batch  = torch.tensor(obs_batch, device=device)
    time_batch = torch.zeros(CFG['n_envs'], dtype=torch.long, device=device)
    pbar = tqdm(total=CFG['env_steps'], desc='Training')

    while step_count < CFG['env_steps']:
        #// сбор rollout
        for _ in range(cfg.rollout_length):
            logits, vals, _, _ = agent(obs_batch, time_batch)
            dist = torch.distributions.Categorical(logits.softmax(-1))
            acts = dist.sample(); lp = dist.log_prob(acts)
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
            time_batch = torch.where(
                torch.tensor(dones, device=device),
                torch.zeros_like(time_batch),
                time_batch + 1
            )
            step_count += CFG['n_envs']
            pbar.update(CFG['n_envs'])
            if dones.any():
                ep_count += dones.sum().item()
                pbar.set_postfix(ep=int(ep_count))

        # PPO up
        _, last_val, _, _ = agent(obs_batch, time_batch)
        buf['val'].append(last_val.detach())
        rews      = torch.stack(buf['rew'])
        vals      = torch.stack(buf['val'])
        dones_vec = torch.stack(buf['done'])
        advs, rets = compute_gae(rews, vals, dones_vec, cfg.gamma, cfg.gae_lambda)
        rets    = rets.detach().reshape(-1)
        advs    = advs.detach().reshape(-1)
        obs_f   = torch.cat(buf['obs'])
        act_f   = torch.cat(buf['act'])
        lp_f    = torch.cat(buf['logp'])
        time_f  = torch.cat(buf['time'])
        tot_pl = tot_vl = tot_ent = cnt = 0
        correct_predictions = 0  # Инициализация счетчика правильных предсказаний

        for _ in range(cfg.update_epochs):
            perm = torch.randperm(len(obs_f))
            for i in range(0, len(obs_f), cfg.batch_size):
                idx    = perm[i:i+cfg.batch_size]
                b_obs  = obs_f[idx];  b_act = act_f[idx]
                b_old  = lp_f[idx];   b_ret = rets[idx]
                b_adv  = advs[idx];    b_time = time_f[idx]

                l, v, _, _ = agent(b_obs, b_time)
                d = torch.distributions.Categorical(l.softmax(-1))

                new_lp = d.log_prob(b_act)
                ent    = d.entropy().mean()
                ratio  = torch.exp(new_lp - b_old)
                clip   = torch.clamp(ratio, 1-cfg.clip_epsilon, 1+cfg.clip_epsilon) * b_adv

                pl = -torch.min(ratio * b_adv, clip).mean()
                vl = F.mse_loss(v, b_ret)
                loss = pl + 0.5 * vl - cfg.entropy_coef * ent

                opt.zero_grad(); loss.backward(); opt.step()

                tot_pl  += pl.item()
                tot_vl  += vl.item()
                tot_ent += ent.item()
                cnt     += 1

                # Вычисление точности классификации
                pred_classes = l.argmax(dim=-1)  # Предсказанные классы
                correct_predictions += (pred_classes == b_act).sum().item()  # Считаем правильные предсказания

        # clean buf
        for k in buf: buf[k].clear()
        avg_pl  = tot_pl / cnt
        avg_vl  = tot_vl / cnt
        avg_ent = tot_ent / cnt

        # Рассчитываем точность
        accuracy = correct_predictions / (len(obs_f) * cfg.batch_size)  # Точность классификации

        # Логируем данные в wandb
        wandb.log({
            "step":          step_count,
            "policy_loss":   avg_pl,
            "value_loss":    avg_vl,
            "entropy":       avg_ent,
            "accuracy":      accuracy  # Логируем точность классификации
        }, step=step_count)
        
        if ep_count >= next_log:
            avg_return = np.mean(ep_rewards[-next_log:])
            avg_length = np.mean(ep_lengths[-next_log:])
            wandb.log({
                "avg_return":   avg_return,
                "avg_length":   avg_length
            }, step=step_count)  # Логируем с шагом
            next_log += cfg.rollout_length  

    pbar.close()


if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    in_ch = _ds.train.image_shape[0] + 2
    agent = ViTWithTimeAndFiLM(
        in_ch=in_ch,
        img_size=CFG['initial_patch'],
        p_size=CFG['initial_patch'],
        d_model=CFG['d_model'],
        n_layers=CFG['n_layers'],
        n_heads=CFG['n_heads'],
        n_actions=CFG['n_zoom'] + CFG['n_move'] + _ds.train.n_classes,
        max_steps=CFG['max_episode_steps'],
        use_film=True,
        film_classes=_ds.train.n_classes,
        film_drop_p=0.1
    ).to(device)
    for e in envs.envs:
        e.model = agent

    wandb.agent(sweep_id, function=lambda: train_ppo(agent), count=20)
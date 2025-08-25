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
import random

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CFG = {
    'env_steps': 100_000,
    'rollout_length': 128,
    'update_epochs': 4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'lr': 2.5e-4,
    'batch_size': 32,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 14,
    'initial_patch': 8,
    'min_patch': 4,
    'max_patch': 32,
    'intrinsic_coef': 0.01,     
    'surprise_coef': 0.1,      
    'recon_coef': 0.1,          
    'entropy_coef': 0.01,
    'supervised_coef': 1.0,
    'checkpoint_interval': 5_000,
    'eval_interval': 10_000,
    'eval_episodes': 1000,
    'max_episode_steps': 10,
    'dataset_seed': 8041991,
    'n_envs': 16,
    'n_eval_envs': 4,
    'n_zoom': 2,         
    'n_move': 4,        
    'correct_bonus': 1,  
    'wrong_penalty': 1,  
    'zoom_bonus': 1,     
    'move_penalty': 0    
}

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'avg_return', 'goal': 'maximize'},
    'parameters': {
        'lr':            {'min': 1e-5,  'max': 1e-4},
        'entropy_coef':  {'min': 1e-3,  'max': 1e-1},
        'batch_size':    {'values': [128]},
        'update_epochs': {'values': [2, 4, 8, 16]},
        'rollout_length':{'values': [128, 256, 512]},
        'clip_epsilon':  {'min': 0.1,   'max': 0.5},
        'gamma':         {'min': 0.9,   'max': 0.999},
        'gae_lambda':    {'min': 0.8,   'max': 0.99},
        'correct_bonus': {'values': [1, 2]},
        'wrong_penalty': {'values': [0, 1]},
        'zoom_bonus':    {'values': [1, 2]},
        'move_penalty':  {'values': [0, 1]},
    }
}

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
        X = X / 255.0  # normalize pixel values to [0,1]
        shape = (3, 32, 32)
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=10000, random_state=seed
        )
        self.train = DatasetSplit(train_X, train_y, shape)
        self.test  = DatasetSplit(test_X,  test_y,  shape)
        logger.info(f"CIFAR-10 ready: train {train_X.shape}, test {test_X.shape}")

_ds = Dataset(CFG['dataset_seed'])
train_imgs = _ds.train.images.reshape(-1, *_ds.train.image_shape)
test_imgs  = _ds.test.images.reshape(-1, *_ds.test.image_shape)
train_images = torch.tensor(train_imgs, dtype=torch.float32, device=device)
test_images  = torch.tensor(test_imgs,  dtype=torch.float32, device=device)
train_targets = _ds.train.targets
test_targets  = _ds.test.targets
logger.info(f"Dataset: train {train_images.shape}, test {test_images.shape} entries")

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
    B = shift.shape[0]
    x = rearrange(x, '(b t) n m -> b t n m', b=B, t=T)
    x = x * (1 + scale.unsqueeze(1).unsqueeze(2)) + shift.unsqueeze(1).unsqueeze(2)
    return rearrange(x, 'b t n m -> (b t) n m', b=B, t=T)

class TimestepEmbedder(nn.Module):
    """Embeds a discrete time-step into a vector, using sinusoidal features followed by an MLP."""
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
        emb   = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
        if dim % 2:  
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb
    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        pe = self.timestep_embedding(t, self.freq_emb)
        return self.mlp(pe)

class LabelEmbedder(nn.Module):
    """Embeds class labels (with optional dropout to represent 'unknown' label)."""
    def __init__(self, num_classes: int, hidden_size: int, drop_p: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(num_classes + (1 if drop_p > 0 else 0), hidden_size)
        self.p = drop_p  
    def forward(self, labels: torch.LongTensor, train: bool, force: torch.LongTensor = None) -> torch.Tensor:
        if (train and self.p > 0) or force is not None:
            if force is None:
                drop_mask = (torch.rand(labels.shape, device=labels.device) < self.p)
            else:
                drop_mask = force.bool()
            labels = torch.where(
                drop_mask,
                torch.tensor(self.embed.num_embeddings - 1, device=labels.device),
                labels
            )
        return self.embed(labels)

@lru_cache()
def build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    np_pe = get_2d_sincos_pos_embed(embed_dim, grid_size)
    return torch.from_numpy(np_pe).float()

class PatchEmbedding(nn.Module):
    """Converts an image patch to an embedding vector (with a learnable [CLS] token and positional encoding)."""
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

class ViTWithTimeAndFiLM(nn.Module):
    """
    Vision Transformer model that includes:
    - Time embedding (to encode how many steps into the episode)
    - FiLM modulation based on a label (or label guess) embedding
    - Actor heads for explore vs classify, and a critic value head
    - Prediction heads for patch content (for curiosity)
    """
    def __init__(self, in_ch: int, img_size: int, p_size: int,
                 d_model: int, n_layers: int, n_heads: int,
                 n_zoom: int, n_move: int, n_classes: int, max_steps: int,
                 use_film: bool = False, film_classes: int = None, film_drop_p: float = 0.0):
        super().__init__()
        self.patch_emb = PatchEmbedding(in_ch, p_size, d_model, img_size)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads,
                                               dim_feedforward=4*d_model,
                                               batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.time_emb   = TimestepEmbedder(hidden_size=d_model, freq_emb=d_model)
        self.max_steps  = max_steps
        self.use_film   = use_film
        self.n_zoom     = n_zoom
        self.n_move     = n_move
        self.n_classes  = n_classes
        if use_film:
            assert film_classes is not None, "FiLM is True" 
            self.shift_embed = LabelEmbedder(film_classes, d_model, drop_p=film_drop_p)
            self.scale_embed = LabelEmbedder(film_classes, d_model, drop_p=film_drop_p)
        self.actor_explore  = nn.Linear(d_model, n_zoom + n_move)  
        self.actor_classify = nn.Linear(d_model, n_classes)      
        self.gate = nn.Sequential(
            nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, 1)
        )
        self.critic     = nn.Linear(d_model, 1)

        self.pred_patch = nn.Linear(d_model, in_ch * p_size * p_size)  
        self.decoder    = nn.Sequential(  
            nn.Linear(d_model, 4*d_model), nn.ReLU(),
            nn.Linear(4*d_model, in_ch * p_size * p_size)
        )
    def forward(self, x: torch.Tensor, t: torch.LongTensor, film_idx: torch.LongTensor = None):
        tokens = self.patch_emb(x)     
        t_pe = self.time_emb(t)   
        tokens[:, 0] = tokens[:, 0] + t_pe  
        if self.use_film and (film_idx is not None):
            shift = self.shift_embed(film_idx, train=self.training)
            scale = self.scale_embed(film_idx, train=self.training)
            B, N, M = tokens.shape
            tokens = modulate(tokens.view(-1, N, M), shift, scale, T=1).view(B, N, M)
        out = self.transformer(tokens)  
        cls = out[:, 0]                   
        gate_in = x[:, -2:].mean(dim=[2, 3]) 
        alpha = torch.sigmoid(self.gate(gate_in)) 
        explore_logits = self.actor_explore(cls) * (1 - alpha)  
        class_logits   = self.actor_classify(cls) * alpha       
        logits = torch.cat([explore_logits, class_logits], dim=-1) 
        value = self.critic(cls).squeeze(-1)
        pred_patch = self.pred_patch(cls)
        recon = self.decoder(cls)
        return logits, value, pred_patch, recon

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                gamma: float, lam: float):
    T, N = rewards.shape 
    advantages = torch.zeros((T, N), device=device)
    last_adv = torch.zeros(N, device=device)
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]  # 0 if done at step t, else 1
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
        self.n_zoom = CFG['n_zoom']
        self.n_move = CFG['n_move']
        self.action_space = gym.spaces.Discrete(self.n_zoom + self.n_move + n_cls)
        C, H, W = data.shape[1:] 
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (C + 2, init_patch, init_patch), dtype=np.float32
        )
        self.last_surprise = 0.0
        self.last_uncertainty = 0.0
        self.last_guess_idx = None 
        self.ptr = 0   
        self.model = None 
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.ptr % self.N
        self.ptr += 1
        self.img = self.data[idx]   
        self.label = int(self.labels[idx])
        self.patch_size = self.init_patch
        self.steps = 0
        self.y = (32 - self.patch_size) // 2
        self.x = (32 - self.patch_size) // 2
        self.last_surprise = 0.0
        self.last_uncertainty = 0.0
        if self.model is not None and getattr(self.model, 'use_film', False):
            self.last_guess_idx = self.model.shift_embed.embed.num_embeddings - 1
        else:
            self.last_guess_idx = None
        return self._get_patch(), {}
    def _get_patch(self):
        """Extract the current patch from self.img, resize to initial_patch size, and add extra channels."""
        C, H, W = self.img.shape
        self.x = max(0, min(self.x, W - self.patch_size))
        self.y = max(0, min(self.y, H - self.patch_size))
        patch = self.img[:, self.y:self.y+self.patch_size, self.x:self.x+self.patch_size].unsqueeze(0) 
        patch = F.interpolate(patch, size=self.init_patch, mode='bilinear', align_corners=False)[0]
        extra = torch.tensor([self.last_surprise, self.last_uncertainty], device=patch.device, dtype=torch.float32).view(2, 1, 1)
        extra = extra.expand(2, self.init_patch, self.init_patch)
        patch = torch.cat([patch, extra], dim=0)  
        return patch.cpu().numpy()
    def step(self, action):
        cfg = wandb.config
        self.steps += 1
        H, W = self.img.shape[1:] 
        external_reward = 0.0
        done = False
        info = {}
        # Interpret action
        if action < self.n_zoom:
            # Zoom 
            if action == 0 and self.patch_size > self.min_patch:
                self.patch_size //= 2 
            elif action == 1 and self.patch_size < self.max_patch:
                self.patch_size *= 2 
            external_reward += cfg.zoom_bonus
            offset = (32 - self.patch_size) // 2
            self.x = self.y = offset
        elif action < self.n_zoom + self.n_move:
            mv = action - self.n_zoom
            if mv == 0:   # up
                self.y = max(0, self.y - self.patch_size)
            elif mv == 1: # down
                self.y = min(H - self.patch_size, self.y + self.patch_size)
            elif mv == 2: # left
                self.x = max(0, self.x - self.patch_size)
            elif mv == 3: # right
                self.x = min(W - self.patch_size, self.x + self.patch_size)
            external_reward -= cfg.move_penalty
        else:
            # Classification 
            pred_class = action - (self.n_zoom + self.n_move)
            info["correct"] = int(pred_class == self.label)
            if pred_class == self.label:
                external_reward += cfg.correct_bonus
            else:
                external_reward -= cfg.wrong_penalty
            done = True
        obs_raw = self.img[:, self.y:self.y+self.patch_size, self.x:self.x+self.patch_size].unsqueeze(0)  
        obs_raw = F.interpolate(obs_raw, size=self.init_patch, mode='bilinear', align_corners=False)[0]
        extra = torch.tensor([self.last_surprise, self.last_uncertainty], device=obs_raw.device, dtype=torch.float32).view(2, 1, 1)
        extra = extra.expand(2, self.init_patch, self.init_patch)
        t_obs = torch.cat([obs_raw, extra], dim=0).unsqueeze(0) 
        time_tensor = torch.tensor([self.steps], device=obs_raw.device, dtype=torch.long)
        film_input = None
        if self.model is not None and getattr(self.model, 'use_film', False):
            if self.last_guess_idx is not None:
                film_input = torch.tensor([self.last_guess_idx], device=obs_raw.device, dtype=torch.long)
        logits, _, patch_pred, dec_pred = self.model(t_obs, time_tensor, film_idx=film_input)
        flat_patch = t_obs.reshape(-1)
        surprise = F.mse_loss(patch_pred.squeeze(0), flat_patch)
        dist = torch.distributions.Categorical(logits.softmax(-1))
        uncertainty = dist.entropy().mean()
        recon_loss = F.mse_loss(dec_pred.squeeze(0), flat_patch)
        self.last_surprise = surprise.item()
        self.last_uncertainty = uncertainty.item()
        intrinsic_reward = cfg.surprise_coef * self.last_surprise - cfg.intrinsic_coef * self.last_uncertainty #биля как важно именно вычитать!!
        total_reward = external_reward + intrinsic_reward
        obs = self._get_patch()
        truncated = (self.steps >= self.max_steps)
        info["surprise"] = self.last_surprise
        info["recon"] = recon_loss.item()
        info["label"] = self.label
        if self.model is not None and getattr(self.model, 'use_film', False):
            class_logits = logits[0, self.n_zoom + self.n_move:] 
            guessed_class = int(torch.argmax(class_logits).item())
            self.last_guess_idx = guessed_class if not done else None
        return obs, float(total_reward), done, truncated, info

def make_env(seed):
    def thunk():
        return CIFAREnv(
            train_images,
            train_targets,
            CFG['initial_patch'],
            CFG['min_patch'],
            CFG['max_patch'],
            CFG['max_episode_steps'],
            _ds.train.n_classes,
        )
    return thunk

def make_eval_env(seed):
    def thunk():
        return CIFAREnv(
            test_images,
            test_targets,
            CFG['initial_patch'],
            CFG['min_patch'],
            CFG['max_patch'],
            CFG['max_episode_steps'],
            _ds.test.n_classes,
        )
    return thunk

def evaluate(agent, envs, episodes: int) -> float:
    agent.eval()
    obs, _ = envs.reset()
    obs = torch.tensor(obs, device=device)
    time_t = torch.zeros(envs.num_envs, dtype=torch.long, device=device)
    guess_eval = torch.full((envs.num_envs,), fill_value=agent.n_classes,
                            device=device, dtype=torch.long)

    correct = 0
    completed = 0

    with torch.no_grad():
        while completed < episodes:
            # Прямой проход через модель с учетом FiLM-догадки
            logits, _, _, _ = agent(obs, time_t, film_idx=guess_eval)
            # Выбор действий
            dist = torch.distributions.Categorical(logits.softmax(-1))
            actions = dist.sample().cpu().numpy()
            # Шаг в средах
            nxt, _, dones, truncated, infos = envs.step(actions)
            terminated = np.logical_or(dones, truncated)

            # Подсчет  классификаций (для старой и новой версии gymnasium)
            if isinstance(infos, dict):
                correct_arr = infos.get("correct", np.zeros(envs.num_envs, dtype=int))
                for i, term in enumerate(terminated):
                    if term:
                        correct += int(correct_arr[i])
                        completed += 1
            else:
                for i, term in enumerate(terminated):
                    if term and "correct" in infos[i]:
                        correct += int(infos[i]["correct"])
                        completed += 1

            # Обновляем догадки агента по argmax логитов классификации
            class_logits = logits[:, agent.n_zoom + agent.n_move:]
            pred_classes = class_logits.argmax(dim=-1)
            for i in range(envs.num_envs):
                if terminated[i]:
                    # сброс догадки в новом эпизоде
                    guess_eval[i] = agent.n_classes
                else:
                    guess_eval[i] = int(pred_classes[i].item())

            # Подготовка к следующему шагу
            obs = torch.tensor(nxt, device=device)
            term_mask = torch.tensor(terminated, device=device)
            time_t = torch.where(term_mask, torch.zeros_like(time_t), time_t + 1)

    accuracy = correct / completed if completed > 0 else 0.0
    agent.train()
    return accuracy

    
def log_eval_images(obs_batch, infos, preds, step):
    # obs_batch: numpy-матрица (n_envs, C+2, H, W)
    # возьмём, например, 4 случайных env-а
    indices = random.sample(range(obs_batch.shape[0]), k=4)
    imgs = []
    for i in indices:
        # исходный патч: первые 3 канала
        patch = obs_batch[i][:3]  # shape (3, H, W)
        # переведём в HWC и в [0,255]
        img = (patch.transpose(1,2,0) * 255).astype(np.uint8)
        caption = f"pred={preds[i]}, true={infos['label'][i]}"
        imgs.append(wandb.Image(img, caption=caption))
    wandb.log({"eval_images": imgs}, step=step)
    
def train_ppo():
    import time
    wandb.init(config=CFG)
    cfg = wandb.config
    in_ch = _ds.train.image_shape[0] + 2  
    agent = ViTWithTimeAndFiLM(
        in_ch=in_ch,
        img_size=CFG['initial_patch'],
        p_size=CFG['initial_patch'],
        d_model=CFG['d_model'],
        n_layers=CFG['n_layers'],
        n_heads=CFG['n_heads'],
        n_zoom=CFG['n_zoom'],
        n_move=CFG['n_move'],
        n_classes=_ds.train.n_classes,
        max_steps=CFG['max_episode_steps'],
        use_film=True,
        film_classes=_ds.train.n_classes,
        film_drop_p=0.1
    ).to(device)

    envs = SyncVectorEnv([make_env(i) for i in range(CFG['n_envs'])])
    eval_envs = SyncVectorEnv([make_eval_env(i) for i in range(CFG['n_eval_envs'])])
    for e in envs.envs:
        e.model = agent
        if agent.use_film:
            e.last_guess_idx = agent.shift_embed.embed.num_embeddings - 1 
    for e in eval_envs.envs:
        e.model = agent
        if agent.use_film:
            e.last_guess_idx = agent.shift_embed.embed.num_embeddings - 1

    opt = optim.AdamW(agent.parameters(), lr=cfg.lr)

    buf = {k: [] for k in ['obs', 'act', 'logp', 'val', 'rew', 'done', 'time', 'label']}
    step_count = 0
    ep_count = 0
    ep_rewards, ep_lengths = [], []
    cur_rewards = np.zeros(CFG['n_envs'], dtype=float)
    cur_lengths = np.zeros(CFG['n_envs'], dtype=int)
    next_log = 1000    
    next_eval = cfg.eval_interval
    obs_batch, _ = envs.reset()
    obs_batch = torch.tensor(obs_batch, device=device)
    time_batch = torch.zeros(CFG['n_envs'], dtype=torch.long, device=device)
    guess_batch = torch.full((CFG['n_envs'],), fill_value=_ds.train.n_classes, device=device, dtype=torch.long)
    pbar = tqdm(total=CFG['env_steps'], desc='Training')
    start_time = time.time()

    while step_count < CFG['env_steps']:
        surprise_list, recon_list = [], []
        for _ in range(cfg.rollout_length):
            logits, vals, _, _ = agent(obs_batch, time_batch, film_idx=guess_batch)
            dist = torch.distributions.Categorical(logits.softmax(-1))
            acts = dist.sample()  
            lp = dist.log_prob(acts)  
            wandb.log({"action_dist": wandb.Histogram(acts.cpu().numpy())}, step=step_count)
            class_logits = logits[:, CFG['n_zoom'] + CFG['n_move']:] 
            pred_classes = class_logits.argmax(dim=-1) 
            for i, env in enumerate(envs.envs):
                env.last_guess_idx = int(pred_classes[i].item())
            nxt_obs, rewards, dones, truncated, infos = envs.step(acts.cpu().numpy())
            terminated = np.logical_or(dones, truncated)
            cur_rewards += rewards
            cur_lengths += 1
            for i, term in enumerate(terminated):
                if term:
                    ep_rewards.append(cur_rewards[i])
                    ep_lengths.append(cur_lengths[i])
                    cur_rewards[i] = 0.0
                    cur_lengths[i] = 0
                    ep_count += 1
            buf['obs'].append(obs_batch)
            buf['act'].append(acts)
            buf['logp'].append(lp.detach())
            buf['val'].append(vals.detach())
            buf['rew'].append(torch.tensor(rewards, device=device, dtype=torch.float32))
            buf['done'].append(torch.tensor(terminated, device=device, dtype=torch.float32))
            buf['time'].append(time_batch)
            
            if isinstance(infos, dict):
                labels = infos.get("label", np.zeros(CFG['n_envs']))
                surps = infos.get("surprise", np.zeros(CFG['n_envs']))
                recons = infos.get("recon", np.zeros(CFG['n_envs']))
            else:
                # Older gym style: infos is a list of dicts
                labels = [inf.get("label", 0) for inf in infos]
                surps = [inf.get("surprise", 0.0) for inf in infos]
                recons = [inf.get("recon", 0.0) for inf in infos]
            buf['label'].append(torch.tensor(labels, device=device, dtype=torch.long))
            surprise_list.extend(np.array(surps).tolist())
            recon_list.extend(np.array(recons).tolist())

            # Log external reward for this step (average across envs)
            wandb.log({"reward_external": float(np.mean(rewards))}, step=step_count)

            # Prepare for next step
            obs_batch = torch.tensor(nxt_obs, device=device)
            # Reset time and guess for environments that ended, increment for others
            term_mask = torch.tensor(terminated, device=device, dtype=torch.bool)
            time_batch = torch.where(term_mask, torch.zeros_like(time_batch), time_batch + 1)
            # Use current predictions as next guess, but unknown for those that ended
            guess_batch = pred_classes.clone().to(device)
            guess_batch[term_mask] = _ds.train.n_classes  # unknown label index

            step_count += CFG['n_envs']
            pbar.update(CFG['n_envs'])

        # After rollout, log mean/std of surprise and reconstruction error
        wandb.log({
            "surprise_mean": float(np.mean(surprise_list)) if surprise_list else 0.0,
            "surprise_std": float(np.std(surprise_list)) if surprise_list else 0.0,
            "recon_mean": float(np.mean(recon_list)) if recon_list else 0.0,
            "recon_std": float(np.std(recon_list)) if recon_list else 0.0
        }, step=step_count)

        # Compute advantage estimates and returns (adding one extra value for bootstrap)
        with torch.no_grad():
            _, last_val, _, _ = agent(obs_batch, time_batch, film_idx=guess_batch)
        buf['val'].append(last_val.detach())
        # Stack rollout tensors
        rewards_tensor = torch.stack(buf['rew'])        # shape (T, n_envs)
        values_tensor = torch.stack(buf['val'])         # shape (T+1, n_envs)
        dones_tensor = torch.stack(buf['done'])         # shape (T, n_envs)
        # Calculate GAE advantages and returns
        advantages, returns = compute_gae(rewards_tensor, values_tensor, dones_tensor, cfg.gamma, cfg.gae_lambda)
        advantages = advantages.detach().reshape(-1)
        returns = returns.detach().reshape(-1)
        # Flatten observations and actions across the rollout
        obs_flat = torch.cat(buf['obs'])    # shape (T*n_envs, C+2, patch, patch)
        act_flat = torch.cat(buf['act'])    # shape (T*n_envs,)
        logp_flat = torch.cat(buf['logp'])  # shape (T*n_envs,)
        time_flat = torch.cat(buf['time'])  # shape (T*n_envs,)
        label_flat = torch.cat(buf['label'])  # shape (T*n_envs,)

        # PPO policy and value network update phase
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_cls_loss = 0.0
        count = 0
        for _ in range(cfg.update_epochs):
            permutation = torch.randperm(len(obs_flat))
            for i in range(0, len(obs_flat), cfg.batch_size):
                idx = permutation[i:i+cfg.batch_size]
                batch_obs = obs_flat[idx]
                batch_act = act_flat[idx]
                batch_old_logp = logp_flat[idx]
                batch_ret = returns[idx]
                batch_adv = advantages[idx]
                batch_time = time_flat[idx]
                batch_labels = label_flat[idx]

                # Forward pass (note: we do not provide FiLM here, to avoid using true labels for training).
                # We set film_idx=None (or could set all to unknown) during policy update to prevent label leakage.
                logits, value_pred, _, _ = agent(batch_obs, batch_time, film_idx=None)
                dist = torch.distributions.Categorical(logits.softmax(-1))
                new_logp = dist.log_prob(batch_act)
                entropy = dist.entropy().mean()
                # PPO clipped objective
                ratio = torch.exp(new_logp - batch_old_logp)
                clipped_ratio = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon)
                policy_loss = -torch.min(ratio * batch_adv, clipped_ratio * batch_adv).mean()
                value_loss = F.mse_loss(value_pred, batch_ret)
                # Supervised classification loss for experiences where a classification was taken
                class_mask = batch_act >= (CFG['n_zoom'] + CFG['n_move'])
                if class_mask.any():
                    cls_logits = logits[class_mask, CFG['n_zoom'] + CFG['n_move']:]
                    cls_targets = batch_labels[class_mask]
                    cls_loss = F.cross_entropy(cls_logits, cls_targets)
                else:
                    cls_loss = torch.tensor(0.0, device=device)

                # Combined loss
                loss = policy_loss + 0.5 * value_loss - cfg.entropy_coef * entropy + cfg.supervised_coef * cls_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_cls_loss += cls_loss.item()
                count += 1

        # Log average losses after policy update
        if count > 0:
            wandb.log({
                "policy_loss": total_policy_loss / count,
                "value_loss": total_value_loss / count,
                "entropy": total_entropy / count,
                "cls_loss": total_cls_loss / count
            }, step=step_count)

        # Clear rollout buffers for next iteration
        for k in buf:
            buf[k].clear()

        # Periodically evaluate the policy on test set
        if ep_count >= next_eval:
            eval_acc = evaluate(agent, eval_envs, cfg.eval_episodes)
            log_eval_images(nxt_obs, infos, pred_classes, step_count)
            wandb.log({"eval_accuracy": eval_acc}, step=step_count)
            next_eval += cfg.eval_interval

        # Periodically log training episode returns and lengths
        if ep_count >= next_log:
            # Compute average return and length over last 1000 episodes (or fewer if not enough)
            last_n = min(len(ep_rewards), 1000)
            avg_return = float(np.mean(ep_rewards[-last_n:])) if last_n > 0 else 0.0
            avg_length = float(np.mean(ep_lengths[-last_n:])) if last_n > 0 else 0.0
            wandb.log({
                "avg_return": avg_return,
                "avg_length": avg_length
            }, step=step_count)
            next_log += 1000

    pbar.close()
    elapsed = time.time() - start_time
    wandb.log({"fps": step_count / elapsed if elapsed > 0 else 0.0}, step=step_count)
    wandb.finish()
    envs.close()
    eval_envs.close()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    import gc
    gc.collect()

import multiprocessing as mp

def worker(rank: int, project_name: str):
    # Launch a WandB sweep agent (for hyperparameter tuning) that runs train_ppo
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=train_ppo, count=40)

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    project_name = "cifar_active_sweep_1_bonus_Acc"
    mp.set_start_method('spawn', force=True)
    processes = []
    for i in range(6):
        p = mp.Process(target=worker, args=(i, project_name))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

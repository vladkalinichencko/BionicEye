import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm import tqdm
from gymnasium.vector import SyncVectorEnv
import gymnasium
from functools import lru_cache
from einops import rearrange
import wandb
from PIL import Image, ImageDraw, ImageFont
import imageio
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="gym.utils.passive_env_checker"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CFG = {
    'env_steps': 10_000_000,  # полноценное обучение
    'max_episode_steps': 20,  # эпизод ограничен 20 шагами
    'n_envs': 16,  # больше параллелизма для 4 V100
    'patch_size': 8,
    'd_model': 256,  # больше для лучшего качества
    'n_heads': 8,  # больше для 4 V100
    'n_layers': 6,  # больше слоев
    'lr': 0.0001,  # уменьшил для стабильности
    'batch_size': 256,  # большой batch для GPU
    'update_epochs': 3,  # уменьшил для предотвращения переобучения
    'gamma': 0.99,  # увеличил для долгосрочного планирования
    'gae_lambda': 0.98,  # увеличил для сглаживания
    'clip_epsilon': 0.15,  # уменьшил для более консервативных обновлений
    'entropy_coef': 0.02,  # увеличил для большего исследования
    'entropy_coef_classification': 0.005,  # увеличил пропорционально
    'recon_coef': 0.005,  # еще меньше фокуса на реконструкции
    'grad_clip_norm': 0.3,  # уменьшил для предотвращения больших градиентов
    'correct_bonus': 3.0,  # увеличил бонус за правильную классификацию
    'interp_mode': 'bilinear',  # быстрее чем bicubic
    'dataset_seed': 80411,
    'rollout_length': 256,  # увеличил для более стабильных обновлений
    'entropy_coef_decision': 0.0,
    'entropy_coef_actions': 0.0,
    'temperature': 0.3,  # уменьшил для более четких решений
    'dropout': 0.1,  # dropout для регуляризации
    'weight_decay': 1e-4,  # L2 регуляризация
}

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'validation/accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'lr': {
            'min': 1e-5,
            'max': 1e-4
        },
        'gamma': {
            'min': 0.9,
            'max': 0.99
        },
        'gae_lambda': {
            'min': 0.8,
            'max': 0.98
        },
        'entropy_coef': {
            'min': 0.01,
            'max': 0.2
        },
        'entropy_coef_classification': {
            'min': 0.001,
            'max': 0.05
        },
        'clip_epsilon': {
            'min': 0.1,
            'max': 0.3
        },
        'update_epochs': {
            'values': [2, 4, 8]
        },
        'batch_size': {
            'values': [64, 128]
        },
        'rollout_length': {
            'values': [128, 256]
        },
        'correct_bonus': {
            'values': [1, 2, 3]
        },
        'wrong_penalty': {
            'values': [0, -1]
        },
        'step_penalty': {
            'values': [0, 0.05, 0.1]
        },
        'viz_interval': {
            'values': [20000]
        },
        'interp_mode': {
            'values': ['bilinear', 'nearest', 'bicubic']
        },
        'recon_coef': {
            'min': 0.01,
            'max': 0.1
        },
        'grad_clip_norm': {
            'min': 0.3,
            'max': 0.7
        },
    },
    'program': 'current_main_script.py'
}

# Standard CIFAR10 transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # [0, 255] -> [0, 1] and converts to torch.FloatTensor
])

@dataclass
class DatasetSplit:
    images: torch.Tensor
    targets: np.ndarray
    image_shape: tuple[int, int, int]

    def __len__(self):
        return len(self.targets)

    @property
    def n_classes(self):
        return 10  # CIFAR10 has 10 classes

class CIFAR10Dataset:
    """Clean wrapper around torchvision CIFAR10."""
    def __init__(self, root='data'):
        self.train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        self.test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        
        # Convert to tensors
        train_images = torch.stack([img for img, _ in self.train_ds])
        test_images = torch.stack([img for img, _ in self.test_ds])
        
        train_targets = np.array([label for _, label in self.train_ds])
        test_targets = np.array([label for _, label in self.test_ds])
        
        # Store as (N, C, H, W) tensors
        self.train = DatasetSplit(train_images, train_targets, (3, 32, 32))
        self.test = DatasetSplit(test_images, test_targets, (3, 32, 32))
        
        logger.info(f"CIFAR-10 loaded: train {train_images.shape}, test {test_images.shape}")

# Load dataset
_ds = CIFAR10Dataset()
# Используем полный CIFAR-10 датасет для максимального обучения
images = _ds.train.images.to(device)  # Все 50000 тренировочных картинок
train_targets = _ds.train.targets
test_images = _ds.test.images.to(device)  # Все 10000 тестовых картинок
test_targets = _ds.test.targets

logger.info(f"Dataset: train image {images.shape}, test {test_images.shape}, target shape {train_targets.shape}")

def fourier_coord_embedding(cx, cy, sz, num_freqs=4):
    """
    cx, cy, sz: (...,) тензоры или скаляры в [0,1]
    num_freqs: сколько частот использовать (например, 4 -> k=1,2,4,8)
    Возвращает: (..., 24) эмбеддинг (3 координаты * 2 * num_freqs)
    """
    import math
    device = cx.device if torch.is_tensor(cx) else 'cpu'
    ks = torch.tensor([1,2,4,8], dtype=torch.float32, device=device)[:num_freqs]  # (num_freqs,)
    coords = torch.stack([cx, cy, sz], dim=-1)  # (..., 3)
    # shape: (..., 3, num_freqs)
    angles = 2 * math.pi * coords.unsqueeze(-1) * ks  # (..., 3, num_freqs)
    sin = torch.sin(angles)  # (..., 3, num_freqs)
    cos = torch.cos(angles)  # (..., 3, num_freqs)
    emb = torch.cat([sin, cos], dim=-1)  # (..., 3, num_freqs)
    emb = emb.reshape(*emb.shape[:-2], 3*2*num_freqs)  # (..., 3*2*num_freqs)
    return emb

class AcceleratedTanh(nn.Module):
    def __init__(self, gamma=0.5):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, x):
        y = torch.tanh(x)
        return torch.sign(y) * torch.abs(y) ** self.gamma

class PatchEmbedding(nn.Module):
    def __init__(self, in_ch: int, p_size: int, d_model: int, img_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=p_size, stride=p_size)
        self.img_size = img_size
        self.patch_size = p_size
        self.coord_proj = nn.Linear(3*2*4, d_model)  # 3 координаты, 4 частоты, sin+cos

    def forward(self, x: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor, sz: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), cx/cy/sz: (B,) - один патч
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, n_patches, d_model)
        # Обычно n_patches=1, оставим универсальность
        coord_emb = fourier_coord_embedding(cx, cy, sz)  # (B, 24)
        coord_emb = self.coord_proj(coord_emb)  # (B, d_model)
        coord_emb = coord_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x = x + coord_emb
        return x

class MemoryTransformer(nn.Module):
    """Memory transformer для обработки истории патчей."""
    def __init__(self, in_ch: int, p_size: int, d_model: int, img_size: int):
        super().__init__()
        self.patch_emb = PatchEmbedding(in_ch, p_size, d_model, img_size)

    def forward(self, x: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor, sz: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W), cx/cy/sz: (B, T) - история патчей
        B, T = x.shape[:2]
        
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        cx_flat = rearrange(cx, 'b t -> (b t)')
        cy_flat = rearrange(cy, 'b t -> (b t)')
        sz_flat = rearrange(sz, 'b t -> (b t)')

        tokens_flat = self.patch_emb(x_flat, cx_flat, cy_flat, sz_flat)
        
        tokens = rearrange(tokens_flat, '(b t) n d -> b (t n) d', b=B)
        return tokens

class ViT(nn.Module):
    """ViT backbone + hybrid heads for Decision / Class / Move / Zoom and Critic."""
    def __init__(self, in_ch: int, img_size: int, p_size: int, d_model: int, n_layers: int, n_heads: int, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = p_size
        self.patch_emb = PatchEmbedding(in_ch, p_size, d_model, img_size)
        self.memory_transformer = MemoryTransformer(in_ch, p_size, d_model, img_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 
            dim_feedforward=4 * d_model, 
            batch_first=True, 
            norm_first=True,
            dropout=CFG.get('dropout', 0.1)
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        # Heads
        self.decision_head = nn.Sequential(
            nn.Dropout(CFG.get('dropout', 0.1)),
            nn.Linear(d_model, 1)
        )
        self.class_head    = nn.Sequential(
            nn.Dropout(CFG.get('dropout', 0.1)),
            nn.Linear(d_model, n_classes)
        )
        self.move_head     = nn.Sequential(
            nn.Dropout(CFG.get('dropout', 0.1)),
            nn.Linear(d_model, 2), 
            AcceleratedTanh(0.8)
        )
        self.zoom_head     = nn.Sequential(
            nn.Dropout(CFG.get('dropout', 0.1)),
            nn.Linear(d_model, 1), 
            AcceleratedTanh(0.8)
        )
        self.critic        = nn.Sequential(
            nn.Dropout(CFG.get('dropout', 0.1)),
            nn.Linear(d_model, 1)
        )
        self.decoder       = nn.Sequential(
            nn.Dropout(CFG.get('dropout', 0.1)),
            nn.Linear(d_model, 4*d_model), 
            nn.ReLU(), 
            nn.Dropout(CFG.get('dropout', 0.1)),
            nn.Linear(4*d_model, 3*p_size*p_size)
        )

    def forward(self, x: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor, sz: torch.Tensor):
        B = x.shape[0]
        # Определяем, история это или один патч по размерности x
        if len(x.shape) == 4:  # (B, C, H, W) - один патч
            tokens = self.patch_emb(x, cx, cy, sz)  # (B, n_patches, d_model)
        elif len(x.shape) == 5:  # (B, T, C, H, W) - история
            tokens = self.memory_transformer(x, cx, cy, sz)  # (B, T*n_patches, d_model)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
            
        out = self.transformer(tokens)
        cls = out[:, 0]  # Первый токен (CLS)
        # Clamp logits to avoid extreme values that may overflow after sigmoid
        decision_logit = torch.clamp(self.decision_head(cls).squeeze(-1), -10.0, 10.0)  # (B,)
        class_logits   = self.class_head(cls)                 # (B,K)
        move_tanh      = self.move_head(cls)                  # (B,2) -> in [-1, 1]
        zoom_tanh      = self.zoom_head(cls)                  # (B,1) -> in [-1, 1]
        value          = self.critic(cls).squeeze(-1)
        recon_pred     = self.decoder(cls)
        # unpack move/zoom params
        mu_move = (move_tanh + 1.0) / 2.0 # Rescale from [-1,1] to [0,1]
        mu_zoom_norm = (zoom_tanh + 1.0) / 2.0 # Rescale from [-1,1] to [0,1]

        return {
            'decision_logit': decision_logit,
            'class_logits': class_logits,
            'mu_move': mu_move, # (B, 2)
            'mu_zoom_norm': mu_zoom_norm.squeeze(-1), # (B,) in [0,1]
            'raw_move': move_tanh,          # (B,2) в [-1,1]
            'raw_zoom': zoom_tanh.squeeze(-1), # (B,) в [-1,1]
            'value': value,
            'recon': recon_pred,
            'tokens': tokens, # Возвращаем токены для анализа
        }

# ---------------- Distribution -------------------------------------------------

class HybridActionDist:
    """Bernoulli (decision) + Categorical (class) + Gaussians (move, zoom)."""
    def __init__(self, params: dict, is_eval: bool = False):
        # --- Decision branch ---
        p_raw = torch.sigmoid(params['decision_logit'].clamp(-10.0, 10.0))
        if not torch.isfinite(p_raw).all():
            raise RuntimeError("NaN or Inf detected in decision probability. Check model stability.")
        self.p = p_raw
        self.bernoulli = torch.distributions.Bernoulli(probs=self.p)

        # --- Classification branch ---
        self.cat = torch.distributions.Categorical(logits=params['class_logits'] / CFG.get('temperature', 1.0))

        # --- Move / Zoom --- УБИРАЕМ ГАУССИАНЫ
        self.mu_move = params['mu_move']
        self.mu_zoom_norm = params['mu_zoom_norm']

    def sample(self):
        decision = self.bernoulli.sample().long()                # (B,)
        class_sample = self.cat.sample()
        # Просто возвращаем предсказанные значения
        return {
            'decision': decision,
            'class': class_sample,
            'move': self.mu_move,
            'zoom': self.mu_zoom_norm
        }

    def log_prob(self, actions):
        decision = actions['decision'].float()
        lp = self.bernoulli.log_prob(decision)
        mask_class = (decision == 1)
        mask_move  = (decision == 0)

        # Add log_prob for the classification action, if taken
        if mask_class.any():
            # We must create a new distribution over the subset of logits
            # corresponding to the actions where decision == 1, because the
            # shapes must match for log_prob.
            masked_logits = self.cat.logits[mask_class]
            masked_actions = actions['class'][mask_class]
            masked_cat_dist = torch.distributions.Categorical(logits=masked_logits)
            class_log_prob = masked_cat_dist.log_prob(masked_actions)
            lp[mask_class] += class_log_prob

        # Add log_prob for the exploration actions, if taken
        if mask_move.any():
            # For Normal distributions, we can compute for all and then mask the result.
            # lp_move = self.move_dist.log_prob(actions['move']).sum(-1) # УБИРАЕМ
            # lp_zoom = self.zoom_dist.log_prob(actions['zoom'].unsqueeze(-1)).squeeze(-1) # УБИРАЕМ
            lp[mask_move] += 0 # УБИРАЕМ
            
        return lp

    def entropy(self):
        # Возвращаем энтропию только для стохастических частей
        # и разделяем их по типам действий
        ent_decision = self.bernoulli.entropy()
        ent_classification = self.cat.entropy()  # Энтропия классификации отдельно
        return ent_decision, ent_classification

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

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float):
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

class CIFAREnv(gymnasium.Env):
    """Environment with hybrid action. Intrinsic rewards are calculated externally."""
    def __init__(self, data, labels, patch_size, max_steps, n_cls):
        super().__init__()
        self.data, self.labels = data, labels
        self.N = data.size(0)
        self.patch_size = patch_size   # output tensor resolution (e.g., 16)
        self.img_hw = data.shape[-1]   # CIFAR size (square)
        # maximum zoom factor (see docstring)
        self.z_max = self.img_hw / self.patch_size
        self.max_steps = max_steps
        self.n_classes = n_cls
        self.action_space = gymnasium.spaces.Dict({
            'decision': gymnasium.spaces.Discrete(2),
            'class':    gymnasium.spaces.Discrete(n_cls),
            'move':     gymnasium.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            'zoom':     gymnasium.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        C, _, _ = data.shape[1:]
        self.observation_space = gymnasium.spaces.Box(0.0, 1.0, (3, patch_size, patch_size), dtype=np.float32)
        self.ptr = 0
        self.interp_mode = CFG.get('interp_mode', 'bilinear')
        self.history = []
        # start with zoom=2 to make movement visible
        self.cur_view_size = self.img_hw // 2  # Start with 16x16 view instead of 32x32
        self.last_pos = None
        # --- История для memory transformer ---
        self.obs_history = []
        self.cx_history = []
        self.cy_history = []
        self.sz_history = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.ptr % self.N
        self.ptr += 1
        self.img = self.data[idx]
        self.label = int(self.labels[idx])
        self.steps = 0
        # start from centre with initial zoom
        centre = (self.img_hw - self.cur_view_size)//2
        self.y = self.x = centre
        self.last_pos = np.array([self.x, self.y])
        
        # Start history for visualization
        self.history = [{'full_image': self.img.cpu().numpy()}]
        # --- История для memory transformer ---
        self.obs_history = []
        self.cx_history = []
        self.cy_history = []
        self.sz_history = []
        # Добавляем первый патч и координаты в историю
        patch = self._get_patch()
        cx, cy, sz = self.get_patch_coords()
        self.obs_history.append(patch)
        self.cx_history.append(cx)
        self.cy_history.append(cy)
        self.sz_history.append(sz)
        return patch, {}

    def _get_patch(self):
        H, W = self.img.shape[1:]
        self.x = max(0, min(self.x, W - self.cur_view_size))
        self.y = max(0, min(self.y, H - self.cur_view_size))
        patch = self.img[:, self.y:self.y + self.cur_view_size, self.x:self.x + self.cur_view_size].unsqueeze(0)
        patch = F.interpolate(patch, size=self.patch_size, mode=self.interp_mode, align_corners=False)[0]
        return patch.cpu().numpy()

    def adjust_position(self, action):
        H, W = self.img.shape[1:]
        patch_size = self.cur_view_size
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
        reward = 0.0
        info = {}

        decision = int(action['decision'])
        terminated = False  # будем завершать эпизод вручную
        truncated = False   # больше не используем тайм-аут по шагам

        if decision == 1:  # classify now – завершаем эпизод
            pred_class = int(action['class'])
            if pred_class == self.label:
                reward = cfg.correct_bonus
            terminated = True  # ключевое изменение

            # Log action for visualization (добавили reward)
            self.history.append({
                'decision': 1,
                'class': pred_class,
                'view_size': self.cur_view_size,
                'x': self.x,
                'y': self.y,
                'reward': reward,
                'true_class': self.label,  # добавили реальный класс
            })
        else:
            # continue exploring: use move & zoom
            x_norm_raw, y_norm_raw = action['move']
            x_norm = np.clip(x_norm_raw, 0.0, 1.0)
            y_norm = np.clip(y_norm_raw, 0.0, 1.0)

            zoom_norm_action = float(action['zoom'])  # now in [0,1]
            z_factor = 1.0 + (self.z_max - 1.0) * zoom_norm_action
            
            view_size = int(round(self.img_hw / z_factor))
            view_size = max(self.patch_size, min(self.img_hw, view_size))

            cx = int(round(x_norm * (self.img_hw - 1)))
            cy = int(round(y_norm * (self.img_hw - 1)))
            # centre the viewport on chosen point while keeping inside image
            x0 = max(0, min(self.img_hw - view_size, cx - view_size // 2))
            y0 = max(0, min(self.img_hw - view_size, cy - view_size // 2))
            
            self.x, self.y = x0, y0
            self.cur_view_size = view_size
            
            # Log action for visualization (добавили reward)
            self.history.append({
                'decision': 0,
                'view_size': view_size,
                'x': x0,
                'y': y0,
                'reward': reward,
            })

        obs = self._get_patch()
        cx, cy, sz = self.get_patch_coords()
        # --- Добавляем в историю ---
        self.obs_history.append(obs)
        self.cx_history.append(cx)
        self.cy_history.append(cy)
        self.sz_history.append(sz)

        # Добавляем информацию о текущем состоянии для анализа
        info['step'] = self.steps
        info['current_pos'] = (self.x, self.y)
        info['current_view_size'] = self.cur_view_size
        info['decision'] = decision
        if decision == 1:
            info['predicted_class'] = int(action['class'])
            info['true_class'] = self.label
            info['correct'] = (int(action['class']) == self.label)

        # Если эпизод не завершён, проверяем ограничение по шагам
        if not terminated and self.steps >= self.max_steps:
            truncated = True

        # Если эпизод завершён, прикладываем историю (history уже содержит reward)
        if terminated or truncated:
            info['episode_history'] = self.history

        return obs, reward, terminated, truncated, info

    def get_patch_coords(self):
        # Центр текущего патча (x, y) и масштаб (z)
        # x, y — левый верхний угол, cur_view_size — размер окна
        cx = self.x + self.cur_view_size / 2
        cy = self.y + self.cur_view_size / 2
        img_hw = self.img_hw
        # Нормализация центра
        cx_norm = cx / (img_hw - 1)
        cy_norm = cy / (img_hw - 1)
        # Масштаб (z = img_hw / cur_view_size)
        z = img_hw / self.cur_view_size
        # Логарифм и нормализация (максимальный зум — z_max = img_hw / patch_size)
        sz = np.log2(z)
        sz_max = np.log2(self.z_max)
        sz_norm = sz / sz_max if sz_max > 0 else 0.0
        return cx_norm, cy_norm, sz_norm

    def get_history(self):
        # Вернуть всю историю патчей и pos-эмбеддингов (np.array)
        return (np.stack(self.obs_history, axis=0),
                np.array(self.cx_history, dtype=np.float32),
                np.array(self.cy_history, dtype=np.float32),
                np.array(self.sz_history, dtype=np.float32))

def get_batch_history(envs):
    # Вернёт батч истории для всех envs: (B, T, ...)
    obs_list, cx_list, cy_list, sz_list = [], [], [], []
    for env in envs.envs:
        obs, cx, cy, sz = env.get_history()
        obs_list.append(obs)
        cx_list.append(cx)
        cy_list.append(cy)
        sz_list.append(sz)
    # Привести к одинаковой длине (padding, если нужно)
    max_T = max(len(o) for o in obs_list)
    def pad(arr, shape, value=0):
        out = np.full(shape, value, dtype=arr.dtype)
        out[:len(arr)] = arr
        return out
    obs_batch = np.stack([pad(o, (max_T,)+o.shape[1:]) for o in obs_list], axis=0)
    cx_batch = np.stack([pad(c, (max_T,)) for c in cx_list], axis=0)
    cy_batch = np.stack([pad(c, (max_T,)) for c in cy_list], axis=0)
    sz_batch = np.stack([pad(s, (max_T,)) for s in sz_list], axis=0)
    return obs_batch, cx_batch, cy_batch, sz_batch

def evaluate_on_test_set(agent, n_episodes=100, verbose=False):
    """Evaluate agent on test set without training.

    If verbose=True, prints how many episodes ended with classification and
    basic accuracy statistics to facilitate debugging.
    """
    # Create test environments
    test_envs = SyncVectorEnv([lambda: CIFAREnv(test_images, test_targets, CFG['patch_size'], CFG['max_episode_steps'], _ds.test.n_classes) for _ in range(min(CFG['n_envs'], 8))])
    
    agent.eval()
    total_rewards = []
    total_episode_lengths = []

    cls_total = 0   # сколько раз классифицировали
    cls_correct = 0 # сколько из них верны
    step_total = 0  # общее число шагов (кроме нулевого)
    step_correct = 0
    
    episodes_done = 0
    obs_batch, _ = test_envs.reset()
    obs_batch = torch.tensor(obs_batch, device=device)
    
    max_steps = n_episodes * CFG['max_episode_steps'] * 2  # защита от зависания
    steps_taken = 0
    
    while episodes_done < n_episodes and steps_taken < max_steps:
        steps_taken += 1
        # --- Получаем всю историю для каждого env ---
        obs_hist, cx_hist, cy_hist, sz_hist = get_batch_history(test_envs)
        obs_hist = torch.tensor(obs_hist, device=device, dtype=torch.float32)
        cx_hist = torch.tensor(cx_hist, device=device, dtype=torch.float32)
        cy_hist = torch.tensor(cy_hist, device=device, dtype=torch.float32)
        sz_hist = torch.tensor(sz_hist, device=device, dtype=torch.float32)
        with torch.no_grad():
            params = agent(obs_hist, cx_hist, cy_hist, sz_hist)
            dist = HybridActionDist(params)
            act_dict = dist.sample()
        
        env_actions = {
            'decision': act_dict['decision'].cpu().numpy(),
            'class':    act_dict['class'].cpu().numpy(),
            'move':     act_dict['move'].cpu().numpy(),
            'zoom':     act_dict['zoom'].cpu().numpy()
        }
        obs_batch, rewards, terminated, truncated, infos = test_envs.step(env_actions)
        obs_batch = torch.tensor(obs_batch, device=device)
        
        dones = terminated | truncated
        for i, done in enumerate(dones):
            if done and episodes_done < n_episodes:
                # Extract episode info - более безопасная проверка
                history_available = (infos and isinstance(infos, (list, tuple)) and len(infos) > i and 
                                             infos[i] and isinstance(infos[i], dict) and 'episode_history' in infos[i])
                if history_available:
                    history = infos[i]['episode_history']
                    episode_reward = sum([step.get('reward', 0) for step in history[1:]])  # Skip first entry
                    episode_length = len(history) - 1
                    
                    # Check if final decision was correct
                    final_step = history[-1]
                    if final_step.get('decision') == 1:
                        pred_class = final_step.get('class', -1)
                        true_label = final_step.get('true_class', -1)  # берем из истории
                        accuracy = 1.0 if pred_class == true_label else 0.0
                        cls_correct += accuracy
                        cls_total += 1
                    else:
                        # не было классификации в истории – отложим, решим ниже
                        accuracy = None
                else:
                    # Если нет истории эпизода, используем дефолтные значения
                    history = None
                    episode_reward = 0.0
                    episode_length = 1.0
                    accuracy = None

                # Если accuracy ещё не вычислена (нет классификации) – делаем forced argmax
                if accuracy is None:
                    # берём предсказание из последних logits
                    forced_pred = int(params['class_logits'][i].argmax().item())
                    true_label = test_envs.envs[i].label
                    accuracy = 1.0 if forced_pred == true_label else 0.0
                    cls_correct += accuracy
                    cls_total += 1

                # step totals
                step_total += episode_length
                step_correct += accuracy  # accuracy is 1 or 0

                total_rewards.append(episode_reward)
                total_episode_lengths.append(episode_length)
                episodes_done += 1
            else:
                # Если нет истории эпизода, используем дефолтные значения
                total_rewards.append(0.0)
                total_episode_lengths.append(1.0)
                episodes_done += 1
                    
    if steps_taken >= max_steps:
        logger.warning(f"Test evaluation timeout after {steps_taken} steps, got {episodes_done} episodes")
    
    test_envs.close()
    agent.train()
    
    # Если не получили ни одного результата, возвращаем дефолтные значения
    if not total_rewards:
        return {
            'test_accuracy': 0.0,
            'test_avg_return': 0.0,
            'test_avg_length': 0.0,
            'test_episodes': 0
        }
    
    class_acc = cls_correct / max(cls_total, 1)
    step_acc = step_correct / max(step_total, 1)
    results = {
        'val_class_accuracy': class_acc,
        'val_step_accuracy': step_acc,
        'test_avg_return': np.mean(total_rewards),
        'test_avg_length': np.mean(total_episode_lengths),
        'test_episodes': len(total_rewards)
    }

    if verbose:
        print(f"[EVAL DEBUG] Episodes: {len(total_rewards)} | Classifications attempted: {cls_total} | ClassAcc: {class_acc:.3f} | StepAcc: {step_acc:.3f}")

    return results

def make_env(seed):
    def thunk():
        return CIFAREnv(images, train_targets, CFG['patch_size'], CFG['max_episode_steps'], _ds.train.n_classes)
    return thunk

envs = SyncVectorEnv([make_env(i) for i in range(CFG['n_envs'])])

# --- Визуализация одного шага ---
step_vis_counter = 0  # глобальный step для визуализации

def log_single_step_visualization(step_info, full_image_np, step_count):
    img = (full_image_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    base_img = Image.fromarray(img).convert("RGBA").resize((512, 512), Image.NEAREST)
    overlay = Image.new('RGBA', base_img.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    view_size = step_info['view_size']
    x, y = step_info['x'], step_info['y']
    scale_factor = 512 / 32
    box_x0 = max(0, min(512, x * scale_factor))
    box_y0 = max(0, min(512, y * scale_factor))
    box_x1 = max(0, min(512, (x + view_size) * scale_factor))
    box_y1 = max(0, min(512, (y + view_size) * scale_factor))
    if box_x1 > box_x0 and box_y1 > box_y0:
        draw_overlay.rectangle([box_x0, box_y0, box_x1, box_y1], fill=(255, 255, 0, 100), outline=(255, 255, 0, 200), width=3)
    frame = Image.alpha_composite(base_img, overlay)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    draw_text = ImageDraw.Draw(frame)
    decision = "Classify" if step_info['decision'] == 1 else "Explore"
    caption = f"Step: {step_count} | {decision}"
    if 'class' in step_info:
        caption += f" (Class {step_info['class']})"
    caption += f" | View: {view_size}x{view_size} | Pos: ({x},{y})"
    lines = caption.split(" | ")
    for j, line in enumerate(lines):
        draw_text.text((10, 10 + j*25), line, font=font, fill="white", stroke_width=2, stroke_fill="black")
    import wandb
    wandb.log({"diagnostics/step_visualization": wandb.Image(frame, caption=caption)}, step=step_count)

def log_token_count_bar(n_tokens, step):
    import wandb
    fig, ax = plt.subplots()
    ax.bar(['tokens'], [n_tokens], color='purple')
    ax.set_ylim(0, n_tokens+2)
    ax.set_title('Number of tokens (ViT input)')
    plt.tight_layout()
    wandb.log({'n_tokens_bar': wandb.Image(fig)}, step=step)
    plt.close(fig)

def log_tsne_embeddings(tokens, step):
    import wandb
    # tokens: (N, d_model), N = num tokens
    if tokens.ndim == 1:
        tokens = tokens.reshape(1, -1)
    if tokens.shape[0] < 2:
        return  # t-SNE не имеет смысла для одного токена
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, tokens.shape[0]-1))
    emb_2d = tsne.fit_transform(tokens)
    fig, ax = plt.subplots()
    ax.scatter(emb_2d[0,0], emb_2d[0,1], color='red', label='CLS', s=80, marker='*')
    if tokens.shape[0] > 1:
        ax.scatter(emb_2d[1:,0], emb_2d[1:,1], color='blue', label='patch tokens', s=40)
    ax.set_title('t-SNE of input tokens (CLS=red star)')
    ax.legend()
    plt.tight_layout()
    wandb.log({'tsne_tokens': wandb.Image(fig)}, step=step)
    plt.close(fig)

def train_ppo(agent):
    cfg = wandb.config
    opt = optim.AdamW(
        agent.parameters(), 
        lr=cfg.lr,
        weight_decay=CFG.get('weight_decay', 1e-4)
    )
    buf = {k: [] for k in ['obs', 'cx', 'cy', 'sz', 'act', 'logp', 'val', 'rew', 'done']} # Добавляем координаты
    step_count, ep_count = 0, 0
    ep_rewards, ep_lengths = [], []
    cur_rewards = np.zeros(CFG['n_envs'], dtype=float)
    cur_lengths = np.zeros(CFG['n_envs'], dtype=int)
    next_log = 1000
    # next_test_eval = 50000  # убираем - больше не нужно

    obs_batch, info = envs.reset()
    obs_batch = torch.tensor(obs_batch, device=device)
    pbar = tqdm(total=CFG['env_steps'], desc='Training')

    # Track detailed reward components
    external_rewards = []

    while step_count < CFG['env_steps']:
        #rollout
        for _ in range(cfg.rollout_length):
            # --- Получаем всю историю для каждого env ---
            obs_hist, cx_hist, cy_hist, sz_hist = get_batch_history(envs)
            obs_hist = torch.tensor(obs_hist, device=device, dtype=torch.float32)  # (B, T, C, H, W)
            cx_hist = torch.tensor(cx_hist, device=device, dtype=torch.float32)    # (B, T)
            cy_hist = torch.tensor(cy_hist, device=device, dtype=torch.float32)    # (B, T)
            sz_hist = torch.tensor(sz_hist, device=device, dtype=torch.float32)    # (B, T)
            with torch.no_grad():
                params = agent(obs_hist, cx_hist, cy_hist, sz_hist)
                dist = HybridActionDist(params)
                act_dict = dist.sample()
                lp = dist.log_prob(act_dict)
                vals = params['value']

            env_actions = {
                'decision': act_dict['decision'].cpu().numpy(),
                'class':    act_dict['class'].cpu().numpy(),
                'move':     act_dict['move'].cpu().numpy(),
                'zoom':     act_dict['zoom'].cpu().numpy()
            }
            nxt, rews, terminated, truncated, infos = envs.step(env_actions)

            total_rewards = torch.tensor(rews, device=device, dtype=torch.float32)
            
            # Track reward components
            external_rewards.extend(rews)
            
            dones = terminated | truncated
            
            cur_rewards += total_rewards.cpu().numpy()
            cur_lengths += 1
            for i, done in enumerate(dones):
                if done:
                    ep_rewards.append(cur_rewards[i])
                    ep_lengths.append(cur_lengths[i])
                    cur_rewards[i] = 0.0
                    cur_lengths[i] = 0

            packed_actions = torch.cat([
                act_dict['decision'].unsqueeze(-1).float(),
                act_dict['class'].unsqueeze(-1).float(),
                act_dict['move'],
                act_dict['zoom'].unsqueeze(-1)
            ], dim=-1)
            buf['act'].append(packed_actions)

            buf['obs'].append(obs_batch)
            # Сохраняем координаты последних патчей
            buf['cx'].append(cx_hist[:, -1])  # (B,)
            buf['cy'].append(cy_hist[:, -1])  # (B,)
            buf['sz'].append(sz_hist[:, -1])  # (B,)
            buf['logp'].append(lp)
            buf['val'].append(vals)
            buf['rew'].append(total_rewards)
            buf['done'].append(torch.tensor(dones, device=device, dtype=torch.float32))

            obs_batch = torch.tensor(nxt, device=device)
            step_count += CFG['n_envs']
            pbar.update(CFG['n_envs'])
            if dones.any():
                ep_count += dones.sum().item()
                pbar.set_postfix(ep=int(ep_count))

            # Log single step visualization для каждого env
            for env_idx, env in enumerate(envs.envs):
                if hasattr(env, 'history') and len(env.history) > 1:
                    # Базовые метрики логируем на каждом шаге
                    step_info = env.history[-1]
                    decision_logit = params['decision_logit'][env_idx].item()
                    p_classify = 1/(1+np.exp(-decision_logit))
                    
                    # Дополнительные метрики для анализа
                    class_probs = F.softmax(params['class_logits'][env_idx], dim=0)
                    max_class_prob = class_probs.max().item()
                    class_entropy = -torch.sum(class_probs * torch.log(class_probs + 1e-8)).item()
                    
                    log_payload = {
                        "actions/p_classify": p_classify,
                        "vitals/n_tokens": params['tokens'][env_idx].shape[0],
                        "confidence/max_class_prob": max_class_prob,
                        "confidence/class_entropy": class_entropy,
                    }

                    if step_info['decision'] == 0: # Если было исследование
                        move_vals = act_dict['move'][env_idx].cpu().numpy()
                        zoom_val = act_dict['zoom'][env_idx].cpu().item()
                        raw_move_vals = params['raw_move'][env_idx].cpu().numpy()
                        raw_zoom_val  = params['raw_zoom'][env_idx].cpu().item()
                        log_payload.update({
                            "actions/move_x": move_vals[0],
                            "actions/move_y": move_vals[1],
                            "actions/zoom_norm": zoom_val, # this is now the [0,1] value
                            "actions/raw_move_x": raw_move_vals[0],
                            "actions/raw_move_y": raw_move_vals[1],
                            "actions/raw_zoom": raw_zoom_val,
                            "diagnostics/view_size": step_info['view_size'],
                        })
                    elif step_info['decision'] == 1: # Если была классификация
                        pred_class = act_dict['class'][env_idx].cpu().item()
                        true_class = env.label
                        is_correct = (pred_class == true_class)
                        log_payload.update({
                            "classification/predicted_class": pred_class,
                            "classification/true_class": true_class,
                            "classification/is_correct": float(is_correct),
                        })
                    
                    wandb.log(log_payload, step=step_count)
                    
                    # "Тяжелые" визуализации логируем ОЧЕНЬ редко для ускорения
                    if step_count > 0 and step_count % 5000 == 0 and env_idx == 0:  # только первый env, очень редко
                        full_image_np = env.history[0]['full_image']
                        log_single_step_visualization(step_info, full_image_np, step_count)
                        
                        # class
                        class_logits = params['class_logits'][env_idx].cpu().numpy()
                        true_class = env.label
                        log_class_histogram(class_logits, true_class, step_count)
                        
                        # t-SNE только очень редко - очень медленный
                        # input_tokens = params['tokens'][env_idx]
                        # log_tsne_embeddings(input_tokens.cpu().numpy(), step_count)


        with torch.no_grad():
            obs_hist, cx_hist, cy_hist, sz_hist = get_batch_history(envs)
            obs_hist = torch.tensor(obs_hist, device=device, dtype=torch.float32)
            cx_hist = torch.tensor(cx_hist, device=device, dtype=torch.float32)
            cy_hist = torch.tensor(cy_hist, device=device, dtype=torch.float32)
            sz_hist = torch.tensor(sz_hist, device=device, dtype=torch.float32)
            last_val = agent(obs_hist, cx_hist, cy_hist, sz_hist)['value']
        
        buf['val'].append(last_val)

        advs, rets = compute_gae(torch.stack(buf['rew']), torch.stack(buf['val']), torch.stack(buf['done']), cfg.gamma, cfg.gae_lambda)
        
        obs_f   = torch.cat(buf['obs']).view(-1, *buf['obs'][0].shape[1:])
        cx_f    = torch.cat(buf['cx']).view(-1)
        cy_f    = torch.cat(buf['cy']).view(-1)
        sz_f    = torch.cat(buf['sz']).view(-1)
        act_f   = torch.cat(buf['act']).view(-1, buf['act'][0].shape[-1])
        lp_f    = torch.cat(buf['logp']).view(-1)
        advs_f  = advs.view(-1)
        rets_f  = rets.view(-1)

        for k in buf: buf[k].clear()
        
        tot_pl, tot_vl, tot_ent, cnt, tot_rec = 0,0,0,0,0
        correct_step = 0      # правильные предсказания на всех шагах
        cls_total = 0          # сколько раз классифицировали
        cls_correct = 0        # сколько раз верно классифицировали
        
        for _ in range(cfg.update_epochs):
            perm = torch.randperm(len(obs_f))
            for i in range(0, len(obs_f), cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                b_obs = obs_f[idx]
                b_cx = cx_f[idx]
                b_cy = cy_f[idx]
                b_sz = sz_f[idx]
                b_packed = act_f[idx]
                
                b_decision = b_packed[:,0].long()
                b_class    = b_packed[:,1].long()
                b_move     = b_packed[:,2:4]
                b_zoom     = b_packed[:,4]
                b_act_dict = {'decision': b_decision, 'class': b_class, 'move': b_move, 'zoom': b_zoom}
                
                b_old_lp = lp_f[idx]
                b_ret = rets_f[idx]
                b_adv = advs_f[idx]

                # Передаём отдельные патчи с реальными координатами в agent
                # b_obs имеет форму (batch_size, C, H, W)
                # b_cx, b_cy, b_sz имеют форму (batch_size,) - реальные координаты
                params_b = agent(b_obs, b_cx, b_cy, b_sz)
                dist_b   = HybridActionDist(params_b)
                new_lp   = dist_b.log_prob(b_act_dict)
                ent_decision, ent_classification = dist_b.entropy()
                # Используем отдельные коэффициенты энтропии
                ent_loss = cfg.entropy_coef * ent_decision.mean() + cfg.entropy_coef_classification * ent_classification.mean()

                ratio = torch.exp(new_lp - b_old_lp)
                clip = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * b_adv
                
                pl = -torch.min(ratio * b_adv, clip).mean()
                vl = F.mse_loss(params_b['value'], b_ret)
                
                recon_flat = params_b['recon']
                true_flat  = b_obs.view(b_obs.size(0), -1)
                r_loss = F.mse_loss(recon_flat, true_flat)

                opt.zero_grad()
                loss = pl + 0.5 * vl - ent_loss + cfg.recon_coef * r_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=wandb.config.grad_clip_norm)
                opt.step()

                tot_pl += pl.item()
                tot_vl += vl.item()
                tot_ent += ent_loss.item() # Увеличиваем tot_ent на ent_loss
                tot_rec += r_loss.item()
                cnt += 1

                pred_classes = params_b['class_logits'].argmax(dim=-1)
                is_cls = (b_decision == 1)
                step_correct = (is_cls & (pred_classes == b_class)).sum()
                correct_step += step_correct.item()
                cls_correct += step_correct.item()
                cls_total += is_cls.sum().item()
        
        avg_pl = tot_pl / cnt
        avg_vl = tot_vl / cnt
        avg_ent = tot_ent / cnt
        avg_rec = tot_rec / cnt
        step_accuracy = correct_step / (len(obs_f) * cfg.update_epochs)
        class_accuracy = cls_correct / max(cls_total, 1)

        # Log detailed reward breakdown every 1000 steps
        if step_count % 1000 == 0 and external_rewards:
            print(f"DEBUG: step_count={step_count}, logging reward breakdown")
            wandb.log({
                "reward_external_mean": np.mean(external_rewards[-1000:]),
            }, step=step_count)
            
        # Валидация после каждого rollout
        n_val_episodes = max(1, int(0.02 * len(test_targets)))  # 2% от тестового датасета
        print(f"DEBUG: step_count={step_count}, running validation with {n_val_episodes} episodes")
        logger.info(f"Running validation on {n_val_episodes} episodes at step {step_count}")
        val_results = evaluate_on_test_set(agent, n_episodes=n_val_episodes)
        print(f"DEBUG: Validation results: {val_results}")
        wandb.log({
            "validation/class_accuracy": val_results['val_class_accuracy'],
            "validation/step_accuracy": val_results['val_step_accuracy'],
            "validation/avg_return": val_results['test_avg_return'],
            "validation/avg_length": val_results['test_avg_length'],
            "validation/episodes": val_results['test_episodes']
        }, step=step_count)
        logger.info(f"Validation at step {step_count}: ClassAcc={val_results['val_class_accuracy']:.3f}, StepAcc={val_results['val_step_accuracy']:.3f}, Return={val_results['test_avg_return']:.3f}")
        print(f"DEBUG: Logged validation to wandb at step {step_count}")

        # Вычисляем реальные значения энтропии для логирования
        with torch.no_grad():
            sample_params = agent(obs_f[:32], cx_f[:32], cy_f[:32], sz_f[:32])  # небольшой sample
            sample_dist = HybridActionDist(sample_params)
            ent_decision_val, ent_classification_val = sample_dist.entropy()
            
        wandb.log({
            "step": step_count,
            "training/policy_loss": avg_pl,
            "training/value_loss": avg_vl,
            "training/entropy_total": avg_ent,
            "training/entropy_decision_value": ent_decision_val.mean().item(),  # реальные значения
            "training/entropy_classification_value": ent_classification_val.mean().item(),  # реальные значения
            "training/entropy_decision_coef": cfg.entropy_coef,  # коэффициенты для справки
            "training/entropy_classification_coef": cfg.entropy_coef_classification,
            "training/step_accuracy": step_accuracy,
            "training/class_accuracy": class_accuracy,
            "training/recon_loss": avg_rec,
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
    
    # Final test set evaluation - быстрая проверка
    logger.info("Running final test set evaluation...")
    final_test_results = evaluate_on_test_set(agent, n_episodes=20)  # быстрая проверка
    wandb.log({
        "final_test_class_accuracy": final_test_results['val_class_accuracy'],
        "final_test_step_accuracy": final_test_results['val_step_accuracy'],
        "final_test_avg_return": final_test_results['test_avg_return'],
        "final_test_avg_length": final_test_results['test_avg_length']
    }, step=step_count)
    
    # Detailed reward analysis
    logger.info(f"Final Test Results:")
    logger.info(f"  Class Accuracy: {final_test_results['val_class_accuracy']:.3f}")
    logger.info(f"  Step Accuracy: {final_test_results['val_step_accuracy']:.3f}")
    logger.info(f"  Avg Return: {final_test_results['test_avg_return']:.3f}")
    logger.info(f"  Avg Episode Length: {final_test_results['test_avg_length']:.1f}")
    
    if external_rewards:
        logger.info(f"Training Reward Analysis (last 1000 steps):")
        logger.info(f"  External Rewards: {np.mean(external_rewards[-1000:]):.3f}")

def create_step_by_step_visualization(history_data, save_dir="visualizations"):
    """Creates a GIF visualization showing each step of agent's exploration on a single image."""
    if not history_data:
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    
    full_image_np = history_data[0].get('full_image')
    if full_image_np is None:
        return None

    # Convert from (C, H, W) to (H, W, C) and scale to 0-255
    full_image_np = (full_image_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    base_img = Image.fromarray(full_image_np).convert("RGBA").resize((512, 512), Image.NEAREST)  # Увеличиваем размер для лучшей видимости
    
    frames = []
    
    # Use a basic font
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Увеличиваем размер шрифта
    except IOError:
        font = ImageFont.load_default()

    # Добавляем первый кадр с исходным изображением
    frame = base_img.copy()
    draw = ImageDraw.Draw(frame)
    draw.text((10, 10), "Step 0: Initial view", font=font, fill="white", stroke_width=2, stroke_fill="black")
    frames.append(frame)

    for i, step_info in enumerate(history_data[1:]): # Skip the first entry which is just the base image
        frame = base_img.copy()
        draw = ImageDraw.Draw(frame)
        
        # Create a transparent overlay for drawing
        overlay = Image.new('RGBA', frame.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        view_size = step_info['view_size']
        x, y = step_info['x'], step_info['y']
        
        # Scale coords to the 512x512 image
        scale_factor = 512 / 32
        box_x0 = max(0, min(512, x * scale_factor))
        box_y0 = max(0, min(512, y * scale_factor))
        box_x1 = max(0, min(512, (x + view_size) * scale_factor))
        box_y1 = max(0, min(512, (y + view_size) * scale_factor))
        
        # Only draw rectangle if it has valid dimensions
        if box_x1 > box_x0 and box_y1 > box_y0:
            # Draw semi-transparent rectangle for viewport
            draw_overlay.rectangle([box_x0, box_y0, box_x1, box_y1], fill=(255, 255, 0, 100), outline=(255, 255, 0, 200), width=3)
        frame = Image.alpha_composite(frame, overlay)
        
        # Add text with more detailed information
        draw_text = ImageDraw.Draw(frame)
        decision = "Classify" if step_info['decision'] == 1 else "Explore"
        caption = f"Step {i+1}: {decision}"
        if 'class' in step_info:
            caption += f" (Class {step_info['class']})"
        caption += f" | View: {view_size}x{view_size} | Pos: ({x},{y})"

        # Разбиваем текст на несколько строк для лучшей читаемости
        lines = caption.split(" | ")
        for j, line in enumerate(lines):
            draw_text.text((10, 10 + j*25), line, font=font, fill="white", stroke_width=2, stroke_fill="black")
        
        frames.append(frame)

    if not frames:
        return None
        
    # Save GIF
    timestamp = int(wandb.run.step)
    save_path = os.path.join(save_dir, f"step_by_step_{timestamp}.gif")
    imageio.mimsave(save_path, frames, duration=1.0)  # Увеличиваем длительность кадра
    
    return wandb.Image(save_path, caption=f"Step-by-step exploration at step {timestamp}")

def save_checkpoint(agent, path="checkpoints/latest_checkpoint.pth"):
    """Saves the agent's state dictionary, overwriting the previous one."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(agent.state_dict(), path)
    logger.info(f"Saved checkpoint to {path}")

def visualize_and_checkpoint(agent, infos, step_count):
    """
    Проверяем, есть ли завершённый эпизод среди словарей infos.
    Если найдена история эпизода – создаём GIF, логируем его в wandb
    и сохраняем чекпойнт модели.
    Возвращает True, если визуализация/чекпойнт были залогированы.
    """
    # `infos` от SyncVectorEnv – это список словарей длиной n_envs
    if infos is None:
        return False

    if isinstance(infos, dict):  # edge-case, когда vec.env агрегирует
        # Попробуем извлечь list из ключа 'final_info' (Gym>=0.26)
        if 'final_info' in infos and isinstance(infos['final_info'], (list, tuple)):
            info_dicts = infos['final_info']
        else:
            info_dicts = [infos]
    else:
        info_dicts = infos  # уже list/tuple

    history_to_log = None
    for info_dict in info_dicts:
        if info_dict and 'episode_history' in info_dict:
            history_to_log = info_dict['episode_history']
            break

    if not history_to_log:
        return False

    video = create_step_by_step_visualization(history_to_log)
    if video:
        wandb.log({"episode_visualization": video}, step=step_count)
    save_checkpoint(agent)
    return True

def log_class_histogram(logits, true_class, step):
    import wandb
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    fig, ax = plt.subplots()
    bars = ax.bar(np.arange(len(probs)), probs, color=['orange' if i==true_class else 'blue' for i in range(len(probs))])
    ax.set_title('Class logits (orange = true class)')
    ax.set_xlabel('Class')
    ax.set_ylabel('Probability')
    plt.tight_layout()
    wandb.log({"diagnostics/class_histogram": wandb.Image(fig)}, step=step)
    plt.close(fig)

# ---------------- Entry point ------------------

def run_sweep(config=None):
    """Create a fresh model for each sweep run and train it."""
    with wandb.init(config=config):
        os.makedirs('checkpoints', exist_ok=True)
        in_ch = _ds.train.image_shape[0]
        model = ViT(in_ch=in_ch,
                    img_size=CFG['patch_size'],
                    p_size=CFG['patch_size'],
                    d_model=CFG['d_model'],
                    n_layers=CFG['n_layers'],
                    n_heads=CFG['n_heads'],
                    n_classes=_ds.train.n_classes).to(device)
        train_ppo(model)

def main():
    """Запускает один тренировочный прогон с конфигурацией из CFG."""
    # Инициализируем wandb для одного запуска
    wandb.init(project="cifar_active_sweep", config=CFG, name="manual-good-params-run")

    # Создаём модель
    in_ch = _ds.train.image_shape[0]
    model = ViT(in_ch=in_ch,
                img_size=CFG['patch_size'],
                p_size=CFG['patch_size'],
                d_model=CFG['d_model'],
                n_layers=CFG['n_layers'],
                n_heads=CFG['n_heads'],
                n_classes=_ds.train.n_classes).to(device)

    # Запускаем тренировку
    train_ppo(model)

if __name__ == '__main__':
    # --- Выберите одно из действий ---

    # 1. Чтобы запустить один эксперимент с настройками из CFG (ДЕЙСТВИЕ ПО УМОЛЧАНИЮ)
    main()

    # 2. Чтобы создать новый sweep, раскомментируйте следующую строку:
    # sweep_id = wandb.sweep(sweep_config, project="cifar_active_sweep")
    # print(f"Created sweep with ID: {sweep_id}")
    # print(f"Sweep URL: https://wandb.ai/vladsteam/cifar_active_sweep/sweeps/{sweep_id}")

    # 3. Чтобы запустить sweep-агент, закомментируйте main() выше и раскомментируйте следующие строки:
    # sweep_id = "PASTE_YOUR_SWEEP_ID_HERE"  # <--- Вставьте сюда ID вашего sweep'а
    # wandb.agent(sweep_id, function=run_sweep)

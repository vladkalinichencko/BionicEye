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
from PIL import Image, ImageDraw, ImageFont
import imageio

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
    'patch_size': 16,              # retina resolution (output H=W)
    'step_penalty': 0.1,           # >0 ; env subtracts this each non-terminal step
    'entropy_coef': 0.01,   # PPO entropy bonus
    'intrinsic_coef': 0.05, # coefficient for intrinsic reward (-entropy)
    'recon_coef': 0.1,      # weight for reconstruction loss in total loss
    'checkpoint_interval': 5_000,
    'eval_interval': 10_000,
    'max_episode_steps': 100,
    'dataset_seed': 80411,
    'n_envs': 16,
    'n_move': 4,
    'interp_mode': 'bilinear',     # resize mode: nearest, bilinear, bicubic, area
    'surprise_coef': 0.1,          # Add surprise coefficient
    'viz_interval': 20000,         # How often to log visualization (in steps)
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
        'step_penalty': {'values': [0.0, 0.05, 0.1]},
        'intrinsic_coef': {'min': 0.0, 'max': 0.1},
        'surprise_coef': {'min': 0.0, 'max': 0.2}, # Add surprise to sweep
        'recon_coef': {'min':0.0,'max':0.2},
        'interp_mode': {'values': ['nearest','bilinear','bicubic','area']},
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
    """ViT backbone + hybrid heads for Decision / Class / Move / Zoom and Critic."""
    def __init__(self, in_ch: int, img_size: int, p_size: int, d_model: int, n_layers: int, n_heads: int, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.patch_emb = PatchEmbedding(in_ch, p_size, d_model, img_size)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        # Heads
        self.decision_head = nn.Linear(d_model, 1)
        self.class_head    = nn.Linear(d_model, n_classes)
        self.move_head     = nn.Linear(d_model, 4)  # μx, μy, logσx, logσy
        self.zoom_head     = nn.Linear(d_model, 2)  # μz, logσz
        self.critic        = nn.Linear(d_model, 1)
        self.decoder       = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, 3*p_size*p_size))

    def forward(self, x: torch.Tensor):
        tokens = self.patch_emb(x)
        out = self.transformer(tokens)
        cls = out[:, 0]
        decision_logit = self.decision_head(cls).squeeze(-1)  # (B,)
        class_logits   = self.class_head(cls)                 # (B,K)
        move_params    = self.move_head(cls)                  # (B,4)
        zoom_params    = self.zoom_head(cls)                  # (B,2)
        value          = self.critic(cls).squeeze(-1)
        recon_pred     = self.decoder(cls)
        # unpack move/zoom params
        mu_move, log_sigma_move = move_params[:, :2], move_params[:, 2:]
        mu_zoom, log_sigma_zoom = zoom_params[:, :1], zoom_params[:, 1:]
        return {
            'decision_logit': decision_logit,
            'class_logits': class_logits,
            'mu_move': mu_move,
            'log_sigma_move': log_sigma_move,
            'mu_zoom': mu_zoom,
            'log_sigma_zoom': log_sigma_zoom,
            'value': value,
            'recon': recon_pred
        }

# ---------------- Distribution -------------------------------------------------

class HybridActionDist:
    """Bernoulli (decision) + Categorical (class) + Gaussians (move, zoom)."""
    def __init__(self, params: dict):
        self.p = torch.sigmoid(params['decision_logit'])          # (B,)
        self.bernoulli = torch.distributions.Bernoulli(probs=self.p)
        self.cat = torch.distributions.Categorical(logits=params['class_logits'])
        sigma_move = torch.exp(params['log_sigma_move'])          # (B,2)
        self.move_dist = torch.distributions.Normal(params['mu_move'], sigma_move)
        sigma_zoom = torch.exp(params['log_sigma_zoom'])          # (B,1)
        self.zoom_dist = torch.distributions.Normal(params['mu_zoom'], sigma_zoom)

    def sample(self):
        decision = self.bernoulli.sample().long()                # (B,)
        class_sample = self.cat.sample()
        move_sample  = self.move_dist.sample()
        zoom_sample  = self.zoom_dist.sample().squeeze(-1)
        return {
            'decision': decision,
            'class': class_sample,
            'move': move_sample,
            'zoom': zoom_sample
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
            lp_move = self.move_dist.log_prob(actions['move']).sum(-1)
            # The zoom distribution has batch_shape (B, 1), so the action needs to be unsqueezed.
            lp_zoom = self.zoom_dist.log_prob(actions['zoom'].unsqueeze(-1)).squeeze(-1)
            lp[mask_move] += lp_move[mask_move] + lp_zoom[mask_move]
            
        return lp

    def entropy(self):
        # average entropy of components
        return self.bernoulli.entropy() + self.cat.entropy() + self.move_dist.entropy().sum(-1) + self.zoom_dist.entropy().squeeze(-1)

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
    """Environment with hybrid action. Intrinsic rewards are calculated externally."""
    def __init__(self, data, labels, patch_size, max_steps, n_cls):
        super().__init__()
        self.data, self.labels = data, labels
        self.N = data.size(0)
        self.patch_size = patch_size   # output tensor resolution (e.g., 16)
        self.img_hw = 32               # CIFAR size (square)
        # maximum zoom factor (see docstring)
        self.z_max = self.img_hw / self.patch_size
        self.max_steps = max_steps
        self.n_classes = n_cls
        self.action_space = gym.spaces.Dict({
            'decision': gym.spaces.Discrete(2),
            'class':    gym.spaces.Discrete(n_cls),
            'move':     gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            'zoom':     gym.spaces.Box(low=1.0, high=self.z_max, shape=(1,), dtype=np.float32),
        })
        C, _, _ = data.shape[1:]
        self.observation_space = gym.spaces.Box(0.0, 1.0, (3, patch_size, patch_size), dtype=np.float32)
        self.ptr = 0
        self.interp_mode = CFG.get('interp_mode', 'bilinear')
        self.history = []
        # start with full view (z=1)
        self.cur_view_size = self.img_hw

    def reset(self):
        idx = self.ptr % self.N
        self.ptr += 1
        self.img = self.data[idx]
        self.label = int(self.labels[idx])
        self.steps = 0
        # start from centre with z=1
        centre = (self.img_hw - self.cur_view_size)//2
        self.y = self.x = centre
        
        # Start history for visualization
        self.history = [{'full_image': self.img.cpu().numpy()}]
        
        return self._get_patch(), {}

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
        done = False
        info = {}

        decision = int(action['decision'])
        if decision == 1:  # classify now
            pred_class = int(action['class'])
            if pred_class == self.label:
                reward += cfg.correct_bonus
            else:
                reward -= cfg.wrong_penalty
            done = True
            # Log action for visualization
            self.history.append({'decision': 1, 'class': pred_class, 'view_size': self.cur_view_size, 'x': self.x, 'y': self.y})
        else:
            # continue exploring: use move & zoom
            x_norm, y_norm = action['move']
            z_factor = float(action['zoom'])
            z_factor = max(1.0, min(self.z_max, z_factor))
            view_size = int(round(self.img_hw / z_factor))
            view_size = max(self.patch_size, min(self.img_hw, view_size))

            cx = int(round(x_norm * (self.img_hw - 1)))
            cy = int(round(y_norm * (self.img_hw - 1)))
            x0 = max(0, min(self.img_hw - view_size, cx - view_size // 2))
            y0 = max(0, min(self.img_hw - view_size, cy - view_size // 2))
            self.x, self.y = x0, y0
            self.cur_view_size = view_size
            reward -= cfg.step_penalty
            # Log action for visualization
            self.history.append({'decision': 0, 'view_size': view_size, 'x': x0, 'y': y0})

        obs = self._get_patch()
        
        truncated = (self.steps >= self.max_steps)
        
        if done or truncated:
            info['episode_history'] = self.history

        return obs, reward, done, truncated, info

def make_env(seed):
    def thunk():
        return CIFAREnv(images, targets, CFG['patch_size'], CFG['max_episode_steps'], _ds.train.n_classes)
    return thunk

envs = SyncVectorEnv([make_env(i) for i in range(CFG['n_envs'])])

def train_ppo(agent):
    wandb.init()
    cfg = wandb.config
    opt = optim.AdamW(agent.parameters(), lr=cfg.lr)
    buf = {k: [] for k in ['obs', 'act', 'logp', 'val', 'rew', 'done']} # Removed aux and time
    step_count, ep_count = 0, 0
    ep_rewards, ep_lengths = [], []
    cur_rewards = np.zeros(CFG['n_envs'], dtype=float)
    cur_lengths = np.zeros(CFG['n_envs'], dtype=int)
    next_log = 1000
    next_log_and_checkpoint_step = cfg.viz_interval

    obs_batch, info = envs.reset()
    obs_batch = torch.tensor(obs_batch, device=device)
    pbar = tqdm(total=CFG['env_steps'], desc='Training')

    while step_count < CFG['env_steps']:
        #rollout
        for _ in range(cfg.rollout_length):
            with torch.no_grad():
                params = agent(obs_batch)
                dist = HybridActionDist(params)
                act_dict = dist.sample()
                lp = dist.log_prob(act_dict)
                vals = params['value']

                # --- Intrinsic Reward Calculation ---
                # Surprise: reconstruction loss of the CURRENT observation
                recon_pred = params['recon']
                true_flat = obs_batch.view(obs_batch.size(0), -1)
                surprise = F.mse_loss(recon_pred, true_flat, reduction='none').mean(dim=1)

                # Uncertainty: policy entropy for the CURRENT observation
                uncertainty = dist.entropy()

            env_actions = {
                'decision': act_dict['decision'].cpu().numpy(),
                'class':    act_dict['class'].cpu().numpy(),
                'move':     act_dict['move'].cpu().numpy(),
                'zoom':     act_dict['zoom'].cpu().numpy()
            }
            nxt, ext_rews, term, tru, infos = envs.step(env_actions)

            # Combine rewards
            intrinsic_reward = cfg.surprise_coef * surprise - cfg.intrinsic_coef * uncertainty
            total_rewards = torch.tensor(ext_rews, device=device, dtype=torch.float32) + intrinsic_reward
            
            dones = term | tru

            # --- Visualization & Checkpointing ---
            if dones.any() and step_count >= next_log_and_checkpoint_step:
                logged = visualize_and_checkpoint(agent, infos, step_count)
                if logged:
                    next_log_and_checkpoint_step += cfg.viz_interval
            
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

        with torch.no_grad():
            last_val = agent(obs_batch)['value']
        buf['val'].append(last_val)

        advs, rets = compute_gae(torch.stack(buf['rew']), torch.stack(buf['val']), torch.stack(buf['done']), cfg.gamma, cfg.gae_lambda)
        
        obs_f   = torch.cat(buf['obs']).view(-1, *buf['obs'][0].shape[1:])
        act_f   = torch.cat(buf['act']).view(-1, buf['act'][0].shape[-1])
        lp_f    = torch.cat(buf['logp']).view(-1)
        advs_f  = advs.view(-1)
        rets_f  = rets.view(-1)

        for k in buf: buf[k].clear()
        
        tot_pl, tot_vl, tot_ent, cnt, tot_rec = 0,0,0,0,0
        correct_predictions = 0
        
        for _ in range(cfg.update_epochs):
            perm = torch.randperm(len(obs_f))
            for i in range(0, len(obs_f), cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                b_obs = obs_f[idx]
                b_packed = act_f[idx]
                
                b_decision = b_packed[:,0].long()
                b_class    = b_packed[:,1].long()
                b_move     = b_packed[:,2:4]
                b_zoom     = b_packed[:,4]
                b_act_dict = {'decision': b_decision, 'class': b_class, 'move': b_move, 'zoom': b_zoom}
                
                b_old_lp = lp_f[idx]
                b_ret = rets_f[idx]
                b_adv = advs_f[idx]

                params_b = agent(b_obs)
                dist_b   = HybridActionDist(params_b)
                new_lp   = dist_b.log_prob(b_act_dict)
                ent      = dist_b.entropy().mean()

                ratio = torch.exp(new_lp - b_old_lp)
                clip = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * b_adv
                
                pl = -torch.min(ratio * b_adv, clip).mean()
                vl = F.mse_loss(params_b['value'], b_ret)
                
                recon_flat = params_b['recon']
                true_flat  = b_obs.view(b_obs.size(0), -1)
                r_loss = F.mse_loss(recon_flat, true_flat)

                opt.zero_grad()
                loss = pl + 0.5 * vl - cfg.entropy_coef * ent + cfg.recon_coef * r_loss
                loss.backward()
                opt.step()

                tot_pl += pl.item()
                tot_vl += vl.item()
                tot_ent += ent.item()
                tot_rec += r_loss.item()
                cnt += 1

                pred_classes = params_b['class_logits'].argmax(dim=-1)
                correct_predictions += ((b_decision==1) & (pred_classes==b_class)).sum().item()
        
        avg_pl = tot_pl / cnt
        avg_vl = tot_vl / cnt
        avg_ent = tot_ent / cnt
        avg_rec = tot_rec / cnt
        accuracy = correct_predictions / (len(obs_f) * cfg.update_epochs)

        wandb.log({
            "step": step_count,
            "policy_loss": avg_pl,
            "value_loss": avg_vl,
            "entropy": avg_ent,
            "accuracy": accuracy,
            "recon_loss": avg_rec
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

def create_episode_visualization(history_data, save_dir="visualizations"):
    """Creates a GIF visualization of an agent's episode."""
    if not history_data:
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    
    full_image_np = history_data[0].get('full_image')
    if full_image_np is None:
        return None

    # Convert from (C, H, W) to (H, W, C) and scale to 0-255
    full_image_np = (full_image_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    base_img = Image.fromarray(full_image_np).convert("RGBA").resize((256, 256), Image.NEAREST)
    
    frames = []
    
    # Use a basic font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    for i, step_info in enumerate(history_data[1:]): # Skip the first entry which is just the base image
        frame = base_img.copy()
        draw = ImageDraw.Draw(frame)
        
        # Create a transparent overlay for drawing
        overlay = Image.new('RGBA', frame.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        view_size = step_info['view_size']
        x, y = step_info['x'], step_info['y']
        
        # Scale coords to the 256x256 image
        scale_factor = 256 / 32
        box_x0 = x * scale_factor
        box_y0 = y * scale_factor
        box_x1 = (x + view_size) * scale_factor
        box_y1 = (y + view_size) * scale_factor
        
        # Draw semi-transparent rectangle for viewport
        draw_overlay.rectangle([box_x0, box_y0, box_x1, box_y1], fill=(255, 255, 0, 100), outline=(255, 255, 0, 200), width=2)
        frame = Image.alpha_composite(frame, overlay)
        
        # Add text
        draw_text = ImageDraw.Draw(frame)
        decision = "Classify" if step_info['decision'] == 1 else "Explore"
        caption = f"Step {i}: {decision}"
        if 'class' in step_info:
            caption += f" (Class {step_info['class']})"

        draw_text.text((10, 10), caption, font=font, fill="white", stroke_width=1, stroke_fill="black")
        
        frames.append(frame)

    if not frames:
        return None
        
    # Save GIF
    timestamp = int(wandb.run.step)
    save_path = os.path.join(save_dir, f"episode_{timestamp}.gif")
    imageio.mimsave(save_path, frames, duration=0.5)
    
    return wandb.Video(save_path, fps=2, format="gif")

def save_checkpoint(agent, path="checkpoints/latest_checkpoint.pth"):
    """Saves the agent's state dictionary, overwriting the previous one."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(agent.state_dict(), path)
    logger.info(f"Saved checkpoint to {path}")

def visualize_and_checkpoint(agent, infos, step_count):
    """
    If a completed episode is found in infos, creates a visualization, logs it,
    and saves a model checkpoint. Returns True if successful, False otherwise.
    """
    if not infos.get("_final_info", np.zeros(CFG['n_envs'], dtype=bool)).any():
        return False
        
    final_infos = infos["final_info"][infos["_final_info"]]
    history_to_log = None
    for info_dict in final_infos:
        if info_dict and "episode_history" in info_dict:
            history_to_log = info_dict["episode_history"]
            break
    
    if history_to_log:
        video = create_episode_visualization(history_to_log)
        if video:
            wandb.log({"episode_visualization": video}, step=step_count)
        
        save_checkpoint(agent)
        return True

    return False

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    in_ch = _ds.train.image_shape[0]  
    agent = ViT(in_ch=in_ch,
                img_size=CFG['patch_size'],
                p_size=CFG['patch_size'],
                d_model=CFG['d_model'],
                n_layers=CFG['n_layers'],
                n_heads=CFG['n_heads'],
                n_classes=_ds.train.n_classes).to(device)
    wandb.agent(sweep_id, function=lambda: train_ppo(agent), count=20)

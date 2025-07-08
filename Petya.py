from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import random_split
from tqdm import tqdm

@dataclass
class SimpleDataset:
    train: torch.utils.data.Dataset
    val: torch.utils.data.Dataset
    test: torch.utils.data.Dataset
    image_shape: Tuple[int, int, int]
    n_classes: int

    @staticmethod
    def create(seed: Optional[int] = None, root: str = "./data", val_frac: float = 0.1) -> "SimpleDataset":
        transform = T.Compose([T.ToTensor()])
        full_train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        val_size = int(len(full_train) * val_frac)
        train_size = len(full_train) - val_size
        train, val = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed or 0),
        )
        return SimpleDataset(train, val, test, (3, 32, 32), 10)

class ImageBuffer:
    def __init__(self, dataset: datasets.VisionDataset, thumb_hw: int = 16, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.images = torch.stack([img for img, _ in dataset]).numpy()
        self.targets = np.array([tar for _, tar in dataset], dtype=np.int64)
        resize = T.Resize(thumb_hw, antialias=True)
        self.thumbnails = torch.stack([resize(img) for img, _ in dataset]).numpy()
        self._size = len(self.targets)

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, self._size, n)
        return idx, self.images[idx], self.thumbnails[idx], self.targets[idx]

def _get_pos_range(img_shape: Tuple[int, int, int], obs_hw: int) -> np.ndarray:
    _, h, w = img_shape
    oh, ow = obs_hw, obs_hw
    return np.array([[0, 0], [max(0, h - oh), max(0, w - ow)]])

def _get_obs(images: np.ndarray, pos: np.ndarray, obs_hw: int, max_hw: int) -> np.ndarray:
    bsz, n_ch = images.shape[0], images.shape[1]
    out = np.zeros((bsz, n_ch, max_hw, max_hw), dtype=float)
    for i, (img, p) in enumerate(zip(images, pos)):
        r, cpos = int(p[0]), int(p[1])
        h_img, w_img = img.shape[1], img.shape[2]
        r    = np.clip(r,    0, max(0, h_img - obs_hw))
        cpos = np.clip(cpos, 0, max(0, w_img - obs_hw))
        patch = img[:, r : r + obs_hw, cpos : cpos + obs_hw]
        out[i, :, :obs_hw, :obs_hw] = patch
    return out

def _to_one_hot_pos(pos: np.ndarray, pos_range: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    max_h, max_w = pos_range[1]
    res = np.zeros((len(pos), 1 + max_h + 1 + max_w + 1), float)
    for i, (r, c) in enumerate(pos):
        if hidden[i]:
            res[i, 0] = 1.0
        else:
            res[i, 1 + r] = 1.0
            res[i, 1 + max_h + 1 + c] = 1.0
    return res

class ImageEnvironment:
    def __init__(
        self,
        ds: SimpleDataset,
        *,
        num_envs: int = 8,
        obs_hw_levels: Tuple[int, ...] = (8, 16, 32),
        thumb_hw: int = 16,
        max_steps: int = 3,
        step_reward: float = 0.0,
        answer_reward: Tuple[float, float] = (1.0, -1.0),
        seed: Optional[int] = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.data = ImageBuffer(ds.train, thumb_hw=thumb_hw, seed=seed)
        self.num_envs = num_envs
        self.img_shape = ds.image_shape
        self.obs_hw_levels = obs_hw_levels
        self.thumb_hw = thumb_hw
        self.max_hw = max(max(obs_hw_levels), thumb_hw)
        self.pos_ranges = [_get_pos_range(self.img_shape, hw) for hw in obs_hw_levels]
        self.global_pos_range = _get_pos_range(self.img_shape, min(obs_hw_levels))
        self.scale_dim = len(obs_hw_levels) + 1
        self.pos_dim = 1 + self.global_pos_range[1].sum() + 2
        self.obs_dim = self.img_shape[0] * self.max_hw * self.max_hw
        self.total_obs_size = self.scale_dim + self.pos_dim + self.obs_dim
        self.max_steps = max_steps
        self.step_reward = step_reward
        self.answer_reward = answer_reward
        self._step = np.zeros(num_envs, dtype=int)
        self._pos = np.zeros((num_envs, 2), dtype=int)
        self._img = np.empty((num_envs, *self.img_shape), dtype=float)
        self._th = np.empty((num_envs, self.img_shape[0], thumb_hw, thumb_hw), dtype=float)
        self._tar = np.empty(num_envs, dtype=int)
        self._done = np.ones(num_envs, dtype=bool)

    def reset(self) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
        self._step[:] = 0
        self._pos[:] = 0
        _, self._img[:], self._th[:], self._tar[:] = self.data.sample(self.num_envs)
        self._done[:] = False
        scale = np.zeros((self.num_envs, self.scale_dim), float)
        scale[:, 0] = 1.0
        pos = _to_one_hot_pos(self._pos, self.global_pos_range, np.ones(self.num_envs, bool))
        obs = _get_obs(self._th, self._pos, self.thumb_hw, self.max_hw)
        return (scale, pos, obs.reshape(self.num_envs, -1)), {}

    def step(self, action: np.ndarray):
        assert action.shape[1] == 5
        what = action[:, 0]
        cls = action[:, 1]
        row = action[:, 2]
        col = action[:, 3]
        scale_idx = action[:, 4]
        reset_mask = self._done
        n_reset = int(reset_mask.sum())
        if n_reset:
            _, self._img[reset_mask], self._th[reset_mask], self._tar[reset_mask] = self.data.sample(n_reset)
            self._step[reset_mask] = 0
            self._pos[reset_mask] = 0
            self._done[reset_mask] = False
        self._step += 1
        move_mask = what == 1
        for i in range(self.num_envs):
            if move_mask[i]:
                hw = self.obs_hw_levels[scale_idx[i]]
                max_r, max_c = self.pos_ranges[scale_idx[i]][1]
                self._pos[i, 0] = min(row[i], max_r)
                self._pos[i, 1] = min(col[i], max_c)
        terminated = what == 2
        truncated = self._step >= self.max_steps
        done = np.logical_or(terminated, truncated)
        reward = np.full(self.num_envs, self.step_reward, float)
        correct = np.logical_and(terminated, cls == self._tar)
        reward[terminated] = self.answer_reward[1]
        reward[correct] = self.answer_reward[0]
        scale = np.zeros((self.num_envs, self.scale_dim), float)
        scale[:, 0] = what == 0
        for i in range(self.num_envs):
            if move_mask[i]:
                scale[i, scale_idx[i] + 1] = 1.0
        obs_img = np.empty((self.num_envs, self.img_shape[0], self.max_hw, self.max_hw), float)
        for i in range(self.num_envs):
            if what[i] == 0:
                obs_img[i] = _get_obs(self._th[i : i + 1], self._pos[i : i + 1], self.thumb_hw, self.max_hw)[0]
            else:
                hw = self.obs_hw_levels[scale_idx[i]]
                obs_img[i] = _get_obs(self._img[i : i + 1], self._pos[i : i + 1], hw, self.max_hw)[0]
        pos = _to_one_hot_pos(self._pos, self.global_pos_range, what == 0)
        self._done = done
        info = {"n_correct": int(correct.sum()), "reset_mask": reset_mask}
        return (scale, pos, obs_img.reshape(self.num_envs, -1)), reward, terminated, truncated, info


class RlAgent(nn.Module):
    def __init__(
        self,
        scale_dim: int,
        pos_dim: int,
        max_hw: int,
        n_classes: int,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.scale_dim = scale_dim
        self.pos_dim = pos_dim
        self.n_classes = n_classes

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        conv_out = 64 * (max_hw // 4) * (max_hw // 4)
        self.img_fc = nn.Sequential(nn.Linear(conv_out, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.aux_fc = nn.Sequential(nn.Linear(scale_dim + pos_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.rnn = nn.LSTM(2 * hidden, hidden, batch_first=True)
        pi_dim = 3 + n_classes + (pos_dim - 3) + (scale_dim - 1)
        self.pi = nn.Linear(hidden, pi_dim)
        self.val = nn.Linear(hidden, 1)

    def forward(self, obs_img: torch.Tensor, aux: torch.Tensor, state=None):
        batch = obs_img.size(0)
        z_img = self.conv(obs_img)
        z_img = z_img.view(batch, -1)
        z_img = self.img_fc(z_img)
        z_aux = self.aux_fc(aux)
        x = torch.cat([z_img, z_aux], dim=1).unsqueeze(1)
        out, state = self.rnn(x, state)
        h = out.squeeze(1)
        pi = self.pi(h)
        v = self.val(h).squeeze(-1)
        return pi, v, state

    def act(self, obs_img: np.ndarray, aux: np.ndarray, state=None, greedy=False):
        img_t = torch.tensor(obs_img, dtype=torch.float32)
        aux_t = torch.tensor(aux, dtype=torch.float32)
        logits, v, state = self.forward(img_t, aux_t, state)
        mv_end = 3
        cl_end = mv_end + self.n_classes
        row_end = cl_end + (self.pos_dim - 3) // 2
        col_end = row_end + (self.pos_dim - 3) // 2
        sc_end = col_end + (self.scale_dim - 1)
        zma = logits[:, :mv_end]
        cls = logits[:, mv_end:cl_end]
        row = logits[:, cl_end:row_end]
        col = logits[:, row_end:col_end]
        sc = logits[:, col_end:sc_end]
        d_zma, d_cls = Categorical(logits=zma), Categorical(logits=cls)
        d_row, d_col = Categorical(logits=row), Categorical(logits=col)
        d_sc = Categorical(logits=sc)

        if greedy:
            a0 = zma.argmax(-1)
            a1 = cls.argmax(-1)
            a2 = row.argmax(-1)
            a3 = col.argmax(-1)
            a4 = sc.argmax(-1)
        else:
            a0 = d_zma.sample()
            a1 = d_cls.sample()
            a2 = d_row.sample()
            a3 = d_col.sample()
            a4 = d_sc.sample()

        logp = d_zma.log_prob(a0) + d_row.log_prob(a2) + d_col.log_prob(a3) + d_cls.log_prob(a1) + d_sc.log_prob(a4)
        a = torch.stack([a0, a1, a2, a3, a4], dim=-1)
        return a.numpy(), v.detach().numpy(), logp, state

def reset_state(state, done: np.ndarray):
    if state is None:
        return None
    mask = torch.tensor(~done).view(1, -1, 1)
    h, c = state
    h = (h * mask).detach()
    c = (c * mask).detach()
    return (h, c)

def train(env: ImageEnvironment, agent: RlAgent, *, steps: int = 1000, gamma: float = 0.99):
    optimiser = optim.Adam(agent.parameters(), lr=1e-3)
    obs, _ = env.reset()
    state = None
    for _ in tqdm(range(steps), desc="Training"):
        scale, pos, flat_img = obs
        img = flat_img.reshape(env.num_envs, env.img_shape[0], env.max_hw, env.max_hw)
        act, val, logp, state = agent.act(img, np.hstack([scale, pos]), state)
        obs_next, r, term, trunc, info = env.step(act)
        done = term | trunc
        with torch.no_grad():
            scale_n, pos_n, flat_img_n = obs_next
            img_n = flat_img_n.reshape(env.num_envs, env.img_shape[0], env.max_hw, env.max_hw)
            _, v_next, _ = agent.forward(
                torch.tensor(img_n, dtype=torch.float32),
                torch.tensor(np.hstack([scale_n, pos_n]), dtype=torch.float32),
                state,
            )

        reward = torch.tensor(r, dtype=torch.float32)
        done_t = torch.tensor(done, dtype=torch.float32)
        val_t = torch.tensor(val, dtype=torch.float32)
        target = reward + gamma * (1 - done_t) * v_next.detach()
        advantage = target - val_t
        policy_loss = -(logp * advantage.detach()).mean()
        value_loss = 0.5 * advantage.pow(2).mean()
        loss = policy_loss + value_loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        obs = obs_next
        state = reset_state(state, done)


def evaluate(env: ImageEnvironment, agent: RlAgent, episodes: int = 100):
    correct = 0
    total = 0
    for _ in tqdm(range(episodes), desc="Evaluating"):
        obs, _ = env.reset()
        state = None
        while True:
            scale, pos, flat_img = obs
            img = flat_img.reshape(env.num_envs, env.img_shape[0], env.max_hw, env.max_hw)
            act, _, _, state = agent.act(img, np.hstack([scale, pos]), state, greedy=True)
            obs, _, term, trunc, info = env.step(act)
            done = term | trunc
            if done.all():
                correct += info["n_correct"]
                total += env.num_envs
                break
        state = reset_state(state, done)
    acc = correct / total if total > 0 else 0.0
    print(f"\nTest Accuracy over {episodes * env.num_envs} episodes: {acc:.4f}")

if __name__ == "__main__":
    ds = SimpleDataset.create(seed=42, val_frac=0.1)
    train_env = ImageEnvironment(ds, num_envs=8)
    agent = RlAgent(
        scale_dim=train_env.scale_dim,
        pos_dim=train_env.pos_dim,
        max_hw=train_env.max_hw,
        n_classes=ds.n_classes,
        hidden=128,
    )
    train(train_env, agent, steps=1000)
    test_env = ImageEnvironment(ds, num_envs=8)
    test_env.data = ImageBuffer(ds.test, thumb_hw=test_env.thumb_hw, seed=0)
    evaluate(test_env, agent, episodes=100)
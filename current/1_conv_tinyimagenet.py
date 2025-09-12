# %%
# %env CLEARML_WEB_HOST=https://app.clear.ml/
# %env CLEARML_API_HOST=https://api.clear.ml
# %env CLEARML_FILES_HOST=https://files.clear.ml
# %env CLEARML_API_ACCESS_KEY=RDZ5AJHQ3GK7OXRMR0E6WGSFINIEXE
# %env CLEARML_API_SECRET_KEY=1tEo6bSWU_JAFa1eIh0zQO8opoC2pimE8hRue0G_BIVLSLVflilPDIiPI6gUfwWjnpc
# %%
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from clearml import Task
import optuna
from optuna.storages import RDBStorage
import gc
from plotly.subplots import make_subplots
import plotly.graph_objects as go

LATENT_DIM = 512
NUM_CLASSES = 100
IMAGE_SIZE = 64
EPOCHS = 60
VAL_INTERVAL = 100
LOG_INTERVAL = 10
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

# ClearML defaults (override via env or function params)
CLEARML_PROJECT = os.environ.get('CLEARML_PROJECT', 'BionicEye')
SWEEP_EXPERIMENT_BASE = os.environ.get('SWEEP_EXPERIMENT_BASE', 'Tiny ImageNet Sweep (optuna)')
FIXED_EXPERIMENT_NAME = os.environ.get('FIXED_EXPERIMENT_NAME', 'Tiny ImageNet Final')

HPARAMS = {
    'encoder': 'conv',
    'batch_size': 160,
    'learning_rate': 3e-3,
    'weight_decay': 1e-4,
    'dropout': 0.2,
    'conv_depth': 5,
    'head_depth': 2,
    'use_autoaugment': False,
    # lighter augmentation knobs
    'crop_scale_min': 0.8,
    'erasing_p': 0.1,
    'label_smoothing': 0.1,
    'residual': True,
}

# Cache for normalization stats computed on train split once
TRAIN_MEAN_STD = None

class _IdentityTfm:
    def __call__(self, x):
        return x

class ToRGB:
    def __call__(self, img):
        return img.convert('RGB')

class TinyImageNetTorch(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index]['image']
        label = self.data[index]['label']

        if self.transform:
            image = self.transform(image)
        
        return image, label

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        attention_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attention_output)
        ffn_output = self.ffn(x)
        return self.norm2(x + ffn_output)

class ResidualDown(nn.Module):
    def __init__(self, in_c, out_c, p):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p),
        )
        self.proj = nn.Conv2d(in_c, out_c, 1, stride=2, bias=False)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.block(x) + self.proj(x))

class ConvEncoder(nn.Module):
    def __init__(self, input_channels=3, conv_depth=4, residual=False):
        super().__init__()
        self.residual = residual
        channels = [64, 128, 256, 512, 512]
        layers = []
        in_c = input_channels
        for i in range(conv_depth):
            out_c = channels[i]
            if self.residual:
                layers.append(ResidualDown(in_c, out_c, HPARAMS['dropout']))
            else:
                layers.append(nn.Sequential(
                    nn.Conv2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(HPARAMS['dropout']),
                ))
            in_c = out_c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_proj = nn.Linear(in_c, LATENT_DIM)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.final_proj(x)

class ViTEncoder(nn.Module):
    def __init__(self, input_channels=3, patch_size=8, num_layers=6, num_heads=8, hidden_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (IMAGE_SIZE // patch_size) ** 2
        patch_dim = input_channels * patch_size * patch_size
        self.patch_embedding = nn.Linear(patch_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.final_proj = nn.Linear(hidden_dim, LATENT_DIM)
        self.transformer_layers = nn.ModuleList([TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)])
    
    def forward(self, x):
        patches = self.extract_patches(x)
        patch_embeddings = self.patch_embedding(patches)
        b = patch_embeddings.size(0)
        cls_token = self.cls_token.expand(b, -1, -1)
        patch_embeddings = torch.cat([cls_token, patch_embeddings], dim=1)
        embeddings = patch_embeddings + self.pos_embedding

        for transformer_layer in self.transformer_layers:
            embeddings = transformer_layer(embeddings)
        
        cls_output = embeddings[:, 0, :]
        return self.final_proj(cls_output)
    
    def extract_patches(self, x):
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(patches.size(0), -1, self.patch_size * self.patch_size * 3)
        return patches

# ------------------------------
# Minimal ResNet-like encoder (our own copy)
# CIFAR-style stem (3x3 s=1) + 4 stages with residual blocks (2-2-2-2).
# ------------------------------

def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, layers_cfg=(2,2,2,2), width: float = 1.0, input_channels: int = 3):
        super().__init__()
        c1, c2, c3, c4 = [int(v * width) for v in (64, 128, 256, 512)]
        # CIFAR stem: keep stride=1 at input resolution 64x64
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.inplanes = c1
        self.layer1 = self._make_layer(BasicBlock, c1, layers_cfg[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, c2, layers_cfg[1], stride=2)  # downsample x2
        self.layer3 = self._make_layer(BasicBlock, c3, layers_cfg[2], stride=2)  # downsample x2
        self.layer4 = self._make_layer(BasicBlock, c4, layers_cfg[3], stride=2)  # downsample x2
        last_dim = c4 * BasicBlock.expansion

        self.features = nn.Sequential(self.stem, self.layer1, self.layer2, self.layer3, self.layer4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_proj = nn.Linear(last_dim, LATENT_DIM)

    def _make_layer(self, block, planes, blocks, stride):
        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.final_proj(x)

def encoder(encoder_type, input_channels=3, conv_depth=4, residual=False):
    if encoder_type == 'conv':
        # Our own residual CNN with a classic 2-2-2-2 layout (no variants)
        return ResNetEncoder(layers_cfg=(2, 2, 2, 2), input_channels=input_channels)
    elif encoder_type == 'vit':
        return ViTEncoder(input_channels)

def classifier_head(head_depth=2):
    layers = []
    in_f = LATENT_DIM
    hidden = 256
    for _ in range(max(0, head_depth - 1)):
        layers += [nn.Linear(in_f, hidden), nn.ReLU(inplace=True), nn.Dropout(HPARAMS['dropout'])]
        in_f = hidden
    layers += [nn.Linear(in_f, NUM_CLASSES)]
    return nn.Sequential(*layers)

def get_dataset(split):
    def load_tinyimagenet(split):
        dataset = load_dataset('zh-plus/tiny-imagenet', split=split)
        dataset = dataset.filter(lambda x: x['label'] < NUM_CLASSES)
        return dataset
    
    def compute_statistics(dataset):
        n_channels = 3
        mean = torch.zeros(n_channels)
        std = torch.zeros(n_channels)
        n_samples = len(dataset)

        for sample in dataset:
            img = sample['image']

            img = transforms.ToTensor()(img)
            mean += img.mean(dim=(1, 2))
            std += img.std(dim=(1, 2))
        
        mean /= n_samples
        std /= n_samples

        return mean.numpy(), std.numpy()
    
    # compute train stats once and reuse for valid/test
    global TRAIN_MEAN_STD
    if TRAIN_MEAN_STD is None:
        train_raw = load_tinyimagenet('train')
        TRAIN_MEAN_STD = compute_statistics(train_raw)
    mean, std = TRAIN_MEAN_STD

    raw_dataset = load_tinyimagenet(split)

    train_tfms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(HPARAMS.get('crop_scale_min', 0.8), 1.0)),
        transforms.RandomHorizontalFlip(),
        AutoAugment(AutoAugmentPolicy.IMAGENET) if HPARAMS['use_autoaugment'] else _IdentityTfm(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        transforms.RandomErasing(p=HPARAMS.get('erasing_p', 0.1), scale=(0.02, 0.1), value='random'),
    ])

    valid_tfms = transforms.Compose([
        ToRGB(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    tfms = train_tfms if split == 'train' else valid_tfms
    return TinyImageNetTorch(raw_dataset, transform=tfms)

def _build_cosine_warmup(optimizer, total_steps: int, warmup_ratio: float = 0.1, min_lr_scale: float = 0.01):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        rem = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(rem)
        # cosine from 1.0 to min_lr_scale
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train(model, train_loader, val_loader, optimizer, criterion, logger=None, checkpoint_path: str = None):
    def validate(model, loader, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        if monitor:
            monitor.begin_validation()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                bs = x.size(0)
                total_loss += loss.item() * bs
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += bs
                # confusion matrix update delegated
                if monitor:
                    monitor.update_confusion_batch(pred, y)
        
        return total_loss / total, correct / total

    model.to(DEVICE)
    best_val = 0.0
    global_step = 0

    total_steps = EPOCHS * max(1, len(train_loader))
    scheduler = _build_cosine_warmup(optimizer, total_steps, warmup_ratio=0.1, min_lr_scale=0.01)

    for epoch in range(EPOCHS):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            if monitor:
                monitor.before_step(model)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            # optional: gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            bs = x.size(0)
            running_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            running_total += bs
            global_step += 1

            # batch-level metrics
            batch_acc = (preds == y).float().mean().item()
            if monitor:
                monitor.log_step(
                    model=model,
                    epoch=epoch,
                    step=global_step,
                    batch_loss=float(loss.item()),
                    batch_acc=float(batch_acc),
                    lr=float(optimizer.param_groups[0]['lr']),
                    grad_norm=float(total_grad_norm.item() if hasattr(total_grad_norm, 'item') else float(total_grad_norm)),
                )

            if logger and global_step % LOG_INTERVAL == 0:
                acc = running_correct / running_total
                logger.report_scalar("train_loss", "step", value=running_loss / running_total, iteration=global_step)
                logger.report_scalar("train_acc", "step", value=acc, iteration=global_step)
                logger.report_scalar("lr", "step", value=optimizer.param_groups[0]['lr'], iteration=global_step)
                print(f"Epoch {epoch} | Step {global_step}: Train loss: {running_loss/running_total:.4f}, Acc: {acc:.4f}")

            if global_step % VAL_INTERVAL == 0:
                val_loss, val_acc = validate(model, val_loader, criterion)
                if logger:
                    logger.report_scalar("val_loss", "step", value=val_loss, iteration=global_step)
                    logger.report_scalar("val_acc", "step", value=val_acc, iteration=global_step)
                if monitor:
                    monitor.end_validation(step=global_step, val_loss=float(val_loss), val_acc=float(val_acc))
                
                print(f"Epoch {epoch} | [Validation at step {global_step}] Val loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                if val_acc > best_val:
                    best_val = val_acc
                    if checkpoint_path:
                        save_checkpoint(model, checkpoint_path, extra={"best_val_acc": best_val, "global_step": global_step})

    return best_val

def _free_memory():
    try:
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()

def objective(trial: optuna.trial.Trial) -> float:
    hp = {
        'encoder': 'conv',
        'batch_size': trial.suggest_categorical('batch_size', [96, 128, 160]),
        'learning_rate': trial.suggest_float('learning_rate', 3e-4, 5e-3, log=True),
        'weight_decay': trial.suggest_categorical('weight_decay', [0.0, 1e-5, 1e-4, 3e-4]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
        'conv_depth': trial.suggest_int('conv_depth', 2, 5),
        'head_depth': trial.suggest_int('head_depth', 1, 3),
        'use_autoaugment': trial.suggest_categorical('use_autoaugment', [True, False]),
        'label_smoothing': 0.1,
        'residual': trial.suggest_categorical('residual', [True, False]),
    }

    global HPARAMS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, DROPOUT_P
    HPARAMS = hp
    BATCH_SIZE = hp['batch_size']
    LEARNING_RATE = hp['learning_rate']
    WEIGHT_DECAY = hp['weight_decay']
    DROPOUT_P = hp['dropout']

    trial_task = Task.init(project_name=CLEARML_PROJECT, task_name=f"{SWEEP_EXPERIMENT_BASE} / trial_{trial.number:04d}", reuse_last_task_id=False)
    trial_logger = trial_task.get_logger()
    trial_task.connect(HPARAMS, name='HPARAMS')

    train_data = get_dataset('train')
    val_data = get_dataset('valid')
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=(DEVICE.type=='cuda'))
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=(DEVICE.type=='cuda'))

    enc = encoder(HPARAMS['encoder'], conv_depth=HPARAMS['conv_depth'], residual=HPARAMS['residual'])
    clf = classifier_head(head_depth=HPARAMS['head_depth'])
    model = nn.Sequential(enc, clf).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=HPARAMS['label_smoothing'])

    best = train(model, train_loader, val_loader, optimizer, criterion, trial_logger)

    del model, enc, clf, optimizer, criterion
    del train_loader, val_loader, train_data, val_data
    _free_memory()

    trial_task.flush()
    trial_task.close()
    return best


def save_checkpoint(model: nn.Module, path: str, extra: dict = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)


def _to_uint8_rgb(img3chw: torch.Tensor) -> np.ndarray:
    x = img3chw.detach().cpu()
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    x = (x * 255.0).clamp(0, 255).byte().numpy()
    return np.transpose(x, (1, 2, 0))


def plot_feature_maps_cnn(enc: nn.Module, x: torch.Tensor, n_maps: int = 8, title: str = "Feature Maps") -> go.Figure:
    # capture last Conv2d activations
    target_conv = None
    for m in reversed(list(enc.features.modules())):
        if isinstance(m, nn.Conv2d):
            target_conv = m
            break
    assert target_conv is not None, "No Conv2d layer found in encoder.features"

    acts = {}
    def fwd_hook(module, inp, out):
        acts['feat'] = out.detach()
    h = target_conv.register_forward_hook(fwd_hook)
    was_training = enc.training
    enc.eval()
    with torch.no_grad():
        _ = enc(x.to(DEVICE)) # run forward pass
    if was_training:
        enc.train()
    h.remove()

    feat = acts['feat'][0]  # [C,H,W]
    c = min(n_maps, feat.size(0))
    maps = feat[:c]
    # per-map minmax
    maps = (maps - maps.amin(dim=(1,2), keepdim=True)) / (maps.amax(dim=(1,2), keepdim=True) - maps.amin(dim=(1,2), keepdim=True) + 1e-6)

    rows = int(np.ceil(c / 4))
    cols = min(4, c)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"ch {i}" for i in range(c)], vertical_spacing=0.08)
    r = c
    for i in range(c):
        rr = i // cols + 1
        cc = i % cols + 1
        z = maps[i].detach().cpu().numpy()
        fig.add_trace(go.Heatmap(z=z, colorscale='Inferno', showscale=False), row=rr, col=cc)
    for i in range(rows*cols):
        rr = i // cols + 1
        cc = i % cols + 1
        fig.update_xaxes(visible=False, row=rr, col=cc)
        fig.update_yaxes(visible=False, row=rr, col=cc)
    fig.update_layout(height=max(400, rows*220), width=900, title=title, showlegend=False)
    return fig


def plot_gradcam_cnn(enc: nn.Module, clf: nn.Module, x: torch.Tensor, target_class: int = None, alpha: float = 0.45, title: str = "Grad-CAM") -> go.Figure:
    # find last conv
    target_conv = None
    for m in reversed(list(enc.features.modules())):
        if isinstance(m, nn.Conv2d):
            target_conv = m
            break
    assert target_conv is not None, "No Conv2d layer found in encoder.features"

    feats, grads = {}, {}
    def fwd_hook(module, inp, out):
        feats['v'] = out
    def bwd_hook(module, gin, gout):
        grads['v'] = gout[0]
    h1 = target_conv.register_forward_hook(fwd_hook)
    h2 = target_conv.register_full_backward_hook(bwd_hook)

    model = nn.Sequential(enc, clf).to(DEVICE)
    model.eval()

    x = x.to(DEVICE).requires_grad_(True)
    logits = model(x)
    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())
    model.zero_grad()
    logits[0, target_class].backward()

    A = feats['v'][0]       # [C,H,W]
    dA = grads['v'][0]      # [C,H,W]
    w = dA.mean(dim=(1,2))  # [C]
    cam = (w[:, None, None] * A).sum(dim=0)
    cam = torch.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam = torch.nn.functional.interpolate(cam[None, None], size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)[0,0]

    img_rgb = _to_uint8_rgb(x[0].detach())

    h1.remove(); h2.remove()

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Image(z=img_rgb))
    fig.add_trace(go.Heatmap(z=cam.detach().cpu().numpy(), colorscale='Jet', opacity=alpha, showscale=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=420, width=420, title=title)
    return fig


def plot_saliency(model: nn.Module, x: torch.Tensor, target_class: int = None, alpha: float = 0.45, title: str = "Saliency") -> go.Figure:
    model = model.to(DEVICE)
    model.eval()

    x = x.to(DEVICE).requires_grad_(True)
    logits = model(x)
    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())
    model.zero_grad()
    logits[0, target_class].backward()

    g = x.grad.detach()[0]  # [3,H,W]
    sal = g.abs().amax(dim=0)  # [H,W]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)

    img_rgb = _to_uint8_rgb(x[0].detach())

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Image(z=img_rgb))
    fig.add_trace(go.Heatmap(z=sal.cpu().numpy(), colorscale='Viridis', opacity=alpha, showscale=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=420, width=420, title=title)
    return fig


def plot_vit_attention(vit: nn.Module, model: nn.Module, x: torch.Tensor, layer: int = -1, alpha: float = 0.5, title: str = "ViT CLS Attention") -> go.Figure:
    # capture attention weights from a given TransformerBlock.attention
    attn_store = []
    layers = vit.transformer_layers
    idx = layer if layer >= 0 else (len(layers) + layer)
    idx = max(0, min(idx, len(layers)-1))
    blk = layers[idx]

    def attn_hook(module, inp, out):
        # out = (attn_output, attn_weights)
        attn_store.append(out[1].detach())
    h = blk.attention.register_forward_hook(attn_hook)

    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        _ = model(x.to(DEVICE))
    h.remove()

    assert len(attn_store) > 0, "No attention captured; ensure forward ran."
    A = attn_store[0][0]  # [T,T] after averaging heads
    cls_to_all = A[0]     # [T]
    patch_scores = cls_to_all[1:]  # ignore CLS self-attn
    grid = int(np.sqrt(vit.num_patches))
    attn_map = patch_scores[:grid*grid].reshape(grid, grid)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)
    attn_up = torch.nn.functional.interpolate(attn_map[None,None], size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)[0,0]

    img_rgb = _to_uint8_rgb(x[0].detach())

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Image(z=img_rgb))
    fig.add_trace(go.Heatmap(z=attn_up.cpu().numpy(), colorscale='Viridis', opacity=alpha, showscale=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=420, width=420, title=title)
    return fig


def run_sweep(experiment_name: str = None, project_name: str = None):
    """Run Optuna sweep with ClearML logging. Set experiment base name via param."""
    global SWEEP_EXPERIMENT_BASE, CLEARML_PROJECT
    if experiment_name:
        SWEEP_EXPERIMENT_BASE = experiment_name
    if project_name:
        CLEARML_PROJECT = project_name
    study_name = os.environ.get('STUDY', 'tiny_imagenet_sweep')
    storage = RDBStorage(
        url=f"sqlite:///{os.path.abspath('sweep.db')}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)
    n_trials = int(os.environ.get('N_TRIALS', '10'))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)


def run_fixed(experiment_name: str = None, project_name: str = None):
    # ClearML Task for fixed training
    exp_name = experiment_name or FIXED_EXPERIMENT_NAME
    proj_name = project_name or CLEARML_PROJECT
    task = Task.init(project_name=proj_name, task_name=exp_name, reuse_last_task_id=False)
    logger = task.get_logger()
    task.connect(HPARAMS, name='HPARAMS')

    # data & loaders
    train_data = get_dataset('train')
    val_data = get_dataset('valid')
    train_loader = DataLoader(train_data, batch_size=HPARAMS['batch_size'], shuffle=True,
                              num_workers=4, pin_memory=(DEVICE.type=='cuda'))
    val_loader = DataLoader(val_data, batch_size=HPARAMS['batch_size'], shuffle=False,
                            num_workers=4, pin_memory=(DEVICE.type=='cuda'))

    # model
    enc = encoder(HPARAMS['encoder'], conv_depth=HPARAMS.get('conv_depth', 4), residual=HPARAMS.get('residual', False))
    clf = classifier_head(head_depth=HPARAMS.get('head_depth', 2))
    model = nn.Sequential(enc, clf).to(DEVICE)

    # resume from checkpoint if available (minimalistic)
    ckpt_path = os.path.join('checkpoints', 'best_fixed.pt')
    if os.path.isfile(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=DEVICE)
            sd = state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state
            model.load_state_dict(sd, strict=False)
            print(f"Resumed weights from {ckpt_path}")
        except Exception as e:
            print(f"Could not load checkpoint {ckpt_path}: {e}")

    # optim & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS['learning_rate'], weight_decay=HPARAMS['weight_decay'])
    criterion = nn.CrossEntropyLoss(label_smoothing=HPARAMS['label_smoothing'])

    # train
    print("Training with fixed hyperparameters:", HPARAMS)
    best = train(model, train_loader, val_loader, optimizer, criterion, logger=logger, checkpoint_path=os.path.join('checkpoints', 'best_fixed.pt'))
    logger.report_scalar("best_val_acc", "final", best, iteration=0)
    print(f"Best val acc: {best:.4f}. Checkpoint saved on improvement at checkpoints/best_fixed.pt")

    # sample one batch and visualize
    xb, yb = next(iter(val_loader))
    xb = xb[:1]  # one sample

    if HPARAMS['encoder'] == 'conv':
        # Feature maps
        fig_fm = plot_feature_maps_cnn(enc, xb, n_maps=8, title="CNN Feature Maps")
        fig_fm.show()
        # Grad-CAM
        fig_cam = plot_gradcam_cnn(enc, clf, xb, target_class=None, alpha=0.45, title="CNN Grad-CAM")
        fig_cam.show()
        # Saliency
        fig_sal = plot_saliency(model, xb, target_class=None, alpha=0.45, title="CNN Saliency")
        fig_sal.show()
    else:
        # ViT attention + general saliency (optional)
        fig_attn = plot_vit_attention(enc, model, xb, layer=-1, alpha=0.5, title="ViT Attention (CLS→patches)")
        fig_attn.show()
        fig_sal = plot_saliency(model, xb, target_class=None, alpha=0.45, title="ViT Saliency")
        fig_sal.show()

    task.flush()
    task.close()


if __name__ == '__main__':
    # run_sweep()
    run_fixed()

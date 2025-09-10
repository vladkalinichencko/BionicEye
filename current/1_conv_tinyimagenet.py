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

LATENT_DIM = 512
NUM_CLASSES = 100
IMAGE_SIZE = 64
EPOCHS = 10
VAL_INTERVAL = 100
LOG_INTERVAL = 10
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

HPARAMS = {
    'encoder': 'conv',
    'batch_size': 128,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'conv_depth': 4,
    'head_depth': 2,
    'use_autoaugment': True,
    'label_smoothing': 0.1,
    'residual': False,
}

class _IdentityTfm:
    def __call__(self, x):
        return x

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

def encoder(encoder_type, input_channels=3, conv_depth=4, residual=False):
    if encoder_type == 'conv':
        return ConvEncoder(input_channels=input_channels, conv_depth=conv_depth, residual=residual)
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
    
    raw_dataset = load_tinyimagenet(split)
    mean, std = compute_statistics(raw_dataset)

    train_tfms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        AutoAugment(AutoAugmentPolicy.IMAGENET) if HPARAMS['use_autoaugment'] else _IdentityTfm(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value='random'),
    ])

    valid_tfms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    tfms = train_tfms if split == 'train' else valid_tfms
    return TinyImageNetTorch(raw_dataset, transform=tfms)

def train(model, train_loader, val_loader, optimizer, criterion, logger=None):
    def validate(model, loader, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

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
        
        return total_loss / total, correct / total

    model.to(DEVICE)
    best_val = 0.0
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            bs = x.size(0)
            running_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            running_total += bs
            global_step += 1

            if logger and global_step % LOG_INTERVAL == 0:
                acc = running_correct / running_total
                logger.report_scalar("train_loss", "step", value=running_loss / running_total, iteration=global_step)
                logger.report_scalar("train_acc", "step", value=acc, iteration=global_step)
                print(f"Epoch {epoch} | Step {global_step}: Train loss: {running_loss/running_total:.4f}, Acc: {acc:.4f}")

            if global_step % VAL_INTERVAL == 0:
                val_loss, val_acc = validate(model, val_loader, criterion)
                if logger:
                    logger.report_scalar("val_loss", "step", value=val_loss, iteration=global_step)
                    logger.report_scalar("val_acc", "step", value=val_acc, iteration=global_step)
                
                print(f"Epoch {epoch} | [Validation at step {global_step}] Val loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                if val_acc > best_val:
                    best_val = val_acc

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

# ---- Optuna objective & launcher (parallel-friendly via SQLite) ----

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

    # apply HPs
    global HPARAMS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, DROPOUT_P
    HPARAMS = hp
    BATCH_SIZE = hp['batch_size']
    LEARNING_RATE = hp['learning_rate']
    WEIGHT_DECAY = hp['weight_decay']
    DROPOUT_P = hp['dropout']

    # clearml task per trial (for nice Scalars graphs)
    trial_task = Task.init(project_name="BionicEye", task_name=f"Tiny ImageNet Sweep (optuna) / trial_{trial.number:04d}", reuse_last_task_id=False)
    trial_logger = trial_task.get_logger()
    trial_task.connect(HPARAMS, name='HPARAMS')

    # data & loaders
    train_data = get_dataset('train')
    val_data = get_dataset('valid')
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # model
    enc = encoder(HPARAMS['encoder'], conv_depth=HPARAMS['conv_depth'], residual=HPARAMS['residual'])
    clf = classifier_head(head_depth=HPARAMS['head_depth'])
    model = nn.Sequential(enc, clf).to(DEVICE)

    # optim & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=HPARAMS['label_smoothing'])

    # train & return best val acc
    best = train(model, train_loader, val_loader, optimizer, criterion, trial_logger)

    # cleanup to avoid device/loader locks between trials
    del model, enc, clf, optimizer, criterion
    del train_loader, val_loader, train_data, val_data
    _free_memory()

    trial_task.flush()
    trial_task.close()
    return best

if __name__ == '__main__':
    study_name = os.environ.get('STUDY', 'tiny_imagenet_sweep')
    storage = RDBStorage(
        url=f"sqlite:///{os.path.abspath('sweep.db')}",
        engine_kwargs={"connect_args": {"timeout": 60}},  # wait up to 60s on busy DB
    )
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)
    n_trials = int(os.environ.get('N_TRIALS', '10'))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
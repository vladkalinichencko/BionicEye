import argparse
import os

import torch
import torch.nn as nn

from current.data.tinyimagenet import tinyimagenet_half_loaders
from current.models.encoders import build_encoder
from current.models.heads import MLPHead
from current.utils.common import get_device, save_checkpoint, set_seed
from current.utils.transforms import get_imagenet_transforms
from current.train.train_cls import train_one_epoch, evaluate


class EncoderWithHead(nn.Module):
    def __init__(self, encoder_name: str, num_classes: int, hidden_dim: int | None, pretrained: bool):
        super().__init__()
        self.encoder = build_encoder(encoder_name, pretrained=pretrained)
        self.head = MLPHead(self.encoder.feature_dim, num_classes, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.head(feats)


def parse_args():
    p = argparse.ArgumentParser(description="Classification baseline on TinyImageNet half-classes")
    p.add_argument('--data-root', type=str, required=True, help='Path to TinyImageNet root containing train/ and val/')
    p.add_argument('--output-dir', type=str, default='./checkpoints/cls')
    p.add_argument('--encoder', type=str, default='resnet18', choices=['resnet18', 'vit_b16', 'conv_ae'])
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--half-index', type=int, default=0, choices=[0, 1])
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--pretrained', type=lambda x: str(x).lower() in ['1', 'true', 'yes'], default=True)
    p.add_argument('--hidden-dim', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    if args.encoder == 'conv_ae' and args.img_size != 64:
        print('[Info] For conv_ae, overriding img_size to 64 for stable shapes.')
        args.img_size = 64

    train_tf, val_tf = get_imagenet_transforms(img_size=args.img_size)
    train_loader, val_loader, num_classes, selected = tinyimagenet_half_loaders(
        root=args.data_root,
        train_transform=train_tf,
        val_transform=val_tf,
        half_index=args.half_index,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    print(f"Selected {num_classes} classes (half_index={args.half_index}). Example: {selected[:5]} ...")

    model = EncoderWithHead(args.encoder, num_classes=num_classes, hidden_dim=args.hidden_dim, pretrained=args.pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader, device)
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} acc={train_acc:.2f} | val_loss={val_loss:.4f} acc={val_acc:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, 'best.pth')
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'args': vars(args),
            }, ckpt_path)
            print(f"[Saved] {ckpt_path} (val_acc={best_val_acc:.2f})")

    print(f"Done. Best val_acc={best_val_acc:.2f}")


if __name__ == '__main__':
    main()

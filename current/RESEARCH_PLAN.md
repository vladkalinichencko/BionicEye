# Active Vision Research Plan

This project explores Active Vision through three progressively stronger baselines on classification (TinyImageNet-200, half of classes) and, later, VQA (Cocoa-QA).

- Baseline 1: Global image encoding for classification
  - Encoders: ResNet-18, ViT-B/16, simple Conv AutoEncoder (encoder part)
  - Head: MLP classifier (1–2 linear layers)
  - Dataset: TinyImageNet-200, use half of classes (A or B split)
  - Task: Standard top-1 accuracy on validation
- Baseline 2: Single-patch heuristic (no RL)
  - Fixed patch selection (center or random) akin to initial JEPA-like sampling
  - Encode the cropped patch; classify with MLP
  - Compare vs. global encoding under same encoders
- Baseline 3: RL-driven patch movement (Active Vision)
  - Agent learns to move/scale a glimpse window to maximize representation quality or classification accuracy in few steps
  - Start with simple PPO; reuse utilities from `previous/` as inspiration

## Hierarchical roadmap

- Phase A: Classification with global encoding (TinyImageNet half-classes)
  - A1. Data loaders for TinyImageNet in ImageFolder format; deterministic A/B class split
  - A2. Encoders: ResNet-18, ViT-B/16, Conv-AE encoder
  - A3. MLP head and trainer loop (metrics, checkpointing)
  - A4. Ablations: encoder choice, head depth, input size, augmentation
- Phase B: Single-patch heuristic
  - B1. Patch extractor (center/random; size as hyperparam)
  - B2. Reuse encoders and head; evaluate vs. Phase A
  - B3. Ablations: patch size/position distributions
- Phase C: Active Vision with RL
  - C1. Environment for patch movement (actions: up/down/left/right/zoom)
  - C2. Reward shaping: classification logit margin / correctness / info gain proxy
  - C3. PPO or equivalent policy optimization
  - C4. Few-step budget evaluation curves
- Phase D: VQA (Cocoa-QA)
  - D1. Text encoder + image encoder fusion (late/simple first)
  - D2. Start with global/patch heuristics; extend to RL glimpses later

## Running the first baseline (example)

```
python current/main_cls.py \
  --data-root /path/to/tinyimagenet \
  --output-dir ./checkpoints/cls_resnet18_A \
  --encoder resnet18 \
  --img-size 224 \
  --batch-size 128 \
  --epochs 50 \
  --lr 1e-3 \
  --half-index 0 \
  --pretrained true
```

Switch encoders with `--encoder vit_b16` or `--encoder conv_ae` (adjust `--img-size` to 64 for `conv_ae`).

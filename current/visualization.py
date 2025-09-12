import os
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _to_uint8_rgb(img3chw: torch.Tensor) -> np.ndarray:
    x = img3chw.detach().cpu()
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    x = (x * 255.0).clamp(0, 255).byte().numpy()
    return np.transpose(x, (1, 2, 0))


class ModelInterpreter:
    """Encapsulates model interpretability visualizations (CNN feature maps, Grad-CAM, saliency, ViT attention)."""

    def __init__(self, image_size: int, device: torch.device):
        self.image_size = image_size
        self.device = device

    def feature_maps_cnn(self, enc: nn.Module, x: torch.Tensor, n_maps: int = 8, title: str = "Feature Maps") -> go.Figure:
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
            _ = enc(x.to(self.device))
        if was_training:
            enc.train()
        h.remove()

        feat = acts['feat'][0]
        c = min(n_maps, feat.size(0))
        maps = feat[:c]
        maps = (maps - maps.amin(dim=(1,2), keepdim=True)) / (maps.amax(dim=(1,2), keepdim=True) - maps.amin(dim=(1,2), keepdim=True) + 1e-6)

        rows = int(np.ceil(c / 4))
        cols = min(4, c)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"ch {i}" for i in range(c)], vertical_spacing=0.08)
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

    def gradcam_cnn(self, enc: nn.Module, clf: nn.Module, x: torch.Tensor, target_class: Optional[int] = None, alpha: float = 0.45, title: str = "Grad-CAM") -> go.Figure:
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

        model = nn.Sequential(enc, clf).to(self.device)
        model.eval()

        x = x.to(self.device).requires_grad_(True)
        logits = model(x)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        model.zero_grad()
        logits[0, target_class].backward()

        A = feats['v'][0]
        dA = grads['v'][0]
        w = dA.mean(dim=(1,2))
        cam = (w[:, None, None] * A).sum(dim=0)
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        cam = torch.nn.functional.interpolate(cam[None, None], size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)[0,0]

        img_rgb = _to_uint8_rgb(x[0].detach())

        h1.remove(); h2.remove()

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Image(z=img_rgb))
        fig.add_trace(go.Heatmap(z=cam.detach().cpu().numpy(), colorscale='Jet', opacity=alpha, showscale=False))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=420, width=420, title=title)
        return fig

    def saliency(self, model: nn.Module, x: torch.Tensor, target_class: Optional[int] = None, alpha: float = 0.45, title: str = "Saliency") -> go.Figure:
        model = model.to(self.device)
        model.eval()
        x = x.to(self.device).requires_grad_(True)
        logits = model(x)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        model.zero_grad()
        logits[0, target_class].backward()

        g = x.grad.detach()[0]
        sal = g.abs().amax(dim=0)
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)

        img_rgb = _to_uint8_rgb(x[0].detach())
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Image(z=img_rgb))
        fig.add_trace(go.Heatmap(z=sal.cpu().numpy(), colorscale='Viridis', opacity=alpha, showscale=False))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=420, width=420, title=title)
        return fig

    def vit_attention(self, vit: nn.Module, model: nn.Module, x: torch.Tensor, layer: int = -1, alpha: float = 0.5, title: str = "ViT CLS Attention") -> go.Figure:
        attn_store = []
        layers = vit.transformer_layers
        idx = layer if layer >= 0 else (len(layers) + layer)
        idx = max(0, min(idx, len(layers)-1))
        blk = layers[idx]

        def attn_hook(module, inp, out):
            attn_store.append(out[1].detach())
        h = blk.attention.register_forward_hook(attn_hook)

        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            _ = model(x.to(self.device))
        h.remove()

        assert len(attn_store) > 0, "No attention captured; ensure forward ran."
        A = attn_store[0][0]
        cls_to_all = A[0]
        patch_scores = cls_to_all[1:]
        grid = int(np.sqrt(vit.num_patches))
        attn_map = patch_scores[:grid*grid].reshape(grid, grid)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)
        attn_up = torch.nn.functional.interpolate(attn_map[None,None], size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)[0,0]

        img_rgb = _to_uint8_rgb(x[0].detach())
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Image(z=img_rgb))
        fig.add_trace(go.Heatmap(z=attn_up.cpu().numpy(), colorscale='Viridis', opacity=alpha, showscale=False))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=420, width=420, title=title)
        return fig


class TrainingVisualizer:
    """Tracks training/validation metrics and handles plotting/reporting.

    Responsibilities:
    - Per-step logging (loss, acc, lr, grad norm, update/weight ratio, grad noise proxy)
    - Gradient snapshotting for 3D landscape
    - Confusion matrix accumulation and plotting
    - Interpretability previews during validation
    - Final report (combined curves, grad norm, landscape, confusion) and optional ClearML upload
    """

    def __init__(
        self,
        device: torch.device,
        capture_gradients: bool = True,
        grad_sample_size: int = 8192,
        num_classes: int = 100,
        logger=None,
        interpreter: ModelInterpreter = None,
        enc: nn.Module = None,
        clf: nn.Module = None,
        encoder_type: str = 'conv',
        vis_sample: Optional[torch.Tensor] = None,
        plot_every_steps: int = 1,
    ):
        self.device = device
        self.capture_gradients = capture_gradients
        self.grad_sample_size = grad_sample_size
        self.num_classes = num_classes
        self.logger = logger

        # interpretability
        self.interpreter = interpreter
        self.enc = enc
        self.clf = clf
        self.encoder_type = encoder_type
        self.vis_sample = vis_sample
        self.plot_every_steps = max(1, int(plot_every_steps))

        # logs
        self.train_steps: List[Dict] = []
        self.val_steps: List[Dict] = []
        self._grad_snapshots: List[np.ndarray] = []
        self._last_confusion: Optional[Tuple[int, np.ndarray]] = None  # (step, cm)
        self._cm_work: Optional[np.ndarray] = None

        # for update/weight ratio
        self._prev_params: Optional[List[torch.Tensor]] = None

        # gradient noise proxy via Welford on grad_norm
        self._noise_n = 0
        self._noise_mean = 0.0
        self._noise_M2 = 0.0

    def _flatten_grad_snapshot(self, model: nn.Module) -> np.ndarray:
        if not self.capture_gradients:
            return np.array([])
        grads = []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().flatten()
            grads.append(g)
        if not grads:
            return np.array([])
        v = torch.cat(grads)
        # Downsample to fixed size deterministically for memory/plotting
        if v.numel() > self.grad_sample_size:
            idx = torch.linspace(0, v.numel()-1, steps=self.grad_sample_size, device=v.device).long()
            v = v[idx]
        return v.cpu().numpy()

    def before_step(self, model: nn.Module) -> None:
        # snapshot parameters pre-optimizer step for update/weight ratio
        self._prev_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    def _compute_update_weight_ratio(self, model: nn.Module) -> float:
        if not self._prev_params:
            return float('nan')
        num_sq = 0.0
        den_sq = 0.0
        i = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            prev = self._prev_params[i]
            i += 1
            d = (p.detach() - prev).float()
            num_sq += float((d*d).sum().item())
            den_sq += float((prev.float()*prev.float()).sum().item())
        self._prev_params = None
        if den_sq <= 0.0:
            return float('inf') if num_sq > 0 else 0.0
        return float(np.sqrt(num_sq) / (np.sqrt(den_sq) + 1e-12))

    def _update_noise(self, grad_norm: float) -> float:
        # Welford update for running std of grad_norm
        self._noise_n += 1
        delta = grad_norm - self._noise_mean
        self._noise_mean += delta / self._noise_n
        delta2 = grad_norm - self._noise_mean
        self._noise_M2 += delta * delta2
        if self._noise_n < 2:
            return 0.0
        var = self._noise_M2 / (self._noise_n - 1)
        std = float(np.sqrt(max(0.0, var)))
        return std / (abs(self._noise_mean) + 1e-12)

    def log_step(self, *, model: nn.Module, epoch: int, step: int, batch_loss: float, batch_acc: float, lr: float, grad_norm: float) -> None:
        update_ratio = self._compute_update_weight_ratio(model)
        grad_noise = self._update_noise(grad_norm)
        self.train_steps.append({
            'epoch': epoch,
            'step': step,
            'loss': float(batch_loss),
            'acc': float(batch_acc),
            'lr': float(lr),
            'grad_norm': float(grad_norm),
            'update_ratio': float(update_ratio),
            'grad_noise': float(grad_noise),
        })
        snap = self._flatten_grad_snapshot(model)
        if snap.size > 0:
            self._grad_snapshots.append(snap)
        # Report diagnostic scalars immediately
        if self.logger is not None:
            try:
                self.logger.report_scalar('grad_norm', 'step', value=float(grad_norm), iteration=step)
                if np.isfinite(update_ratio):
                    self.logger.report_scalar('update_weight_ratio', 'step', value=float(update_ratio), iteration=step)
                if np.isfinite(grad_noise):
                    self.logger.report_scalar('grad_noise_index', 'step', value=float(grad_noise), iteration=step)
            except Exception:
                pass
        # As frequently as possible: update combined curves + 3D landscape
        if self.logger is not None and (step % self.plot_every_steps == 0):
            try:
                loss_fig, acc_fig = self.plot_curves()
                self.logger.report_plotly('curves', 'loss_train_vs_val', loss_fig, iteration=step)
                self.logger.report_plotly('curves', 'acc_train_vs_val', acc_fig, iteration=step)
                land_fig = self.plot_training_landscape_3d()
                if land_fig is not None:
                    self.logger.report_plotly('diagnostics', 'gradient_landscape', land_fig, iteration=step)
            except Exception:
                pass

    def begin_validation(self) -> None:
        self._cm_work = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update_confusion_batch(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if self._cm_work is None:
            self.begin_validation()
        preds = preds.detach().view(-1).cpu().numpy()
        targets = targets.detach().view(-1).cpu().numpy()
        for t, p in zip(targets, preds):
            self._cm_work[int(t), int(p)] += 1

    def end_validation(self, *, step: int, val_loss: float, val_acc: float) -> None:
        self.val_steps.append({
            'step': step,
            'loss': float(val_loss),
            'acc': float(val_acc),
        })
        if self._cm_work is not None:
            self._last_confusion = (int(step), self._cm_work.copy())
            self._cm_work = None
        # Report confusion and interpretability previews to ClearML (optional)
        if self.logger is not None:
            try:
                cm_fig = self.plot_confusion_matrix(normalize=True)
                if cm_fig is not None:
                    self.logger.report_plotly('val', 'confusion_matrix', cm_fig, iteration=step)
            except Exception:
                pass
        if self.interpreter is not None and self.vis_sample is not None and self.logger is not None:
            try:
                xs = self.vis_sample[:1]
                if self.encoder_type == 'conv' and self.enc is not None and self.clf is not None:
                    # Feature maps (basic feature interpretation)
                    try:
                        fig_fm = self.interpreter.feature_maps_cnn(self.enc, xs, n_maps=8, title=f"Feature Maps@{step}")
                        self.logger.report_plotly('interpret', 'feature_maps', fig_fm, iteration=step)
                    except Exception:
                        pass
                    fig_cam = self.interpreter.gradcam_cnn(self.enc, self.clf, xs, target_class=None, alpha=0.45, title=f"Grad-CAM@{step}")
                    fig_sal = self.interpreter.saliency(nn.Sequential(self.enc, self.clf).to(self.device), xs, target_class=None, alpha=0.45, title=f"Saliency@{step}")
                    self.logger.report_plotly('interpret', 'gradcam', fig_cam, iteration=step)
                    self.logger.report_plotly('interpret', 'saliency', fig_sal, iteration=step)
                elif self.encoder_type == 'vit' and self.enc is not None:
                    fig_attn = self.interpreter.vit_attention(self.enc, nn.Sequential(self.enc, self.clf).to(self.device), xs, layer=-1, alpha=0.5, title=f"ViT Attention@{step}")
                    fig_sal = self.interpreter.saliency(nn.Sequential(self.enc, self.clf).to(self.device), xs, target_class=None, alpha=0.45, title=f"ViT Saliency@{step}")
                    self.logger.report_plotly('interpret', 'vit_attention', fig_attn, iteration=step)
                    self.logger.report_plotly('interpret', 'saliency', fig_sal, iteration=step)
            except Exception:
                pass

        # Also report combined and separate curves + diagnostics each validation
        if self.logger is not None:
            try:
                loss_fig, acc_fig = self.plot_curves()
                self.logger.report_plotly('curves', 'loss_train_vs_val', loss_fig, iteration=step)
                self.logger.report_plotly('curves', 'acc_train_vs_val', acc_fig, iteration=step)
                tr_loss_fig, tr_acc_fig = self.plot_train_only()
                va_loss_fig, va_acc_fig = self.plot_val_only()
                self.logger.report_plotly('curves', 'train_loss_only', tr_loss_fig, iteration=step)
                self.logger.report_plotly('curves', 'train_acc_only', tr_acc_fig, iteration=step)
                self.logger.report_plotly('curves', 'val_loss_only', va_loss_fig, iteration=step)
                self.logger.report_plotly('curves', 'val_acc_only', va_acc_fig, iteration=step)
                grad_fig = self.plot_grad_norm()
                upd_fig = self.plot_update_ratio()
                noise_fig = self.plot_grad_noise()
                self.logger.report_plotly('diagnostics', 'grad_norm', grad_fig, iteration=step)
                self.logger.report_plotly('diagnostics', 'update_weight_ratio', upd_fig, iteration=step)
                self.logger.report_plotly('diagnostics', 'grad_noise_index', noise_fig, iteration=step)
                land_fig = self.plot_training_landscape_3d()
                if land_fig is not None:
                    self.logger.report_plotly('diagnostics', 'gradient_landscape', land_fig, iteration=step)
            except Exception:
                pass

    def plot_curves(self) -> Tuple[go.Figure, go.Figure]:
        # Loss figure (train vs val in one plot)
        steps = [d['step'] for d in self.train_steps]
        train_loss = [d['loss'] for d in self.train_steps]
        vsteps = [d['step'] for d in self.val_steps]
        val_loss = [d['loss'] for d in self.val_steps]

        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(x=steps, y=train_loss, mode='lines', name='train_loss'))
        loss_fig.add_trace(go.Scatter(x=vsteps, y=val_loss, mode='lines+markers', name='val_loss'))
        loss_fig.update_layout(title='Loss (train vs val)', xaxis_title='step', yaxis_title='loss')

        # Accuracy figure (train vs val in one plot)
        train_acc = [max(0.0, min(1.0, float(d['acc']))) for d in self.train_steps]
        val_acc = [max(0.0, min(1.0, float(d['acc']))) for d in self.val_steps]
        acc_fig = go.Figure()
        acc_fig.add_trace(go.Scatter(x=steps, y=train_acc, mode='lines', name='train_acc'))
        acc_fig.add_trace(go.Scatter(x=vsteps, y=val_acc, mode='lines+markers', name='val_acc'))
        acc_fig.update_layout(title='Accuracy (train vs val)', xaxis_title='step', yaxis_title='accuracy', yaxis=dict(range=[0, 1]))

        return loss_fig, acc_fig

    def plot_train_only(self) -> Tuple[go.Figure, go.Figure]:
        steps = [d['step'] for d in self.train_steps]
        loss = [d['loss'] for d in self.train_steps]
        acc = [max(0.0, min(1.0, float(d['acc']))) for d in self.train_steps]
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(x=steps, y=loss, mode='lines', name='train_loss'))
        fig_l.update_layout(title='Train Loss', xaxis_title='step', yaxis_title='loss')
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=steps, y=acc, mode='lines', name='train_acc'))
        fig_a.update_layout(title='Train Accuracy', xaxis_title='step', yaxis_title='accuracy', yaxis=dict(range=[0,1]))
        return fig_l, fig_a

    def plot_val_only(self) -> Tuple[go.Figure, go.Figure]:
        steps = [d['step'] for d in self.val_steps]
        loss = [d['loss'] for d in self.val_steps]
        acc = [max(0.0, min(1.0, float(d['acc']))) for d in self.val_steps]
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(x=steps, y=loss, mode='lines+markers', name='val_loss'))
        fig_l.update_layout(title='Validation Loss', xaxis_title='step', yaxis_title='loss')
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=steps, y=acc, mode='lines+markers', name='val_acc'))
        fig_a.update_layout(title='Validation Accuracy', xaxis_title='step', yaxis_title='accuracy', yaxis=dict(range=[0,1]))
        return fig_l, fig_a

    def plot_grad_norm(self) -> go.Figure:
        steps = [d['step'] for d in self.train_steps]
        norms = [d['grad_norm'] for d in self.train_steps]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=norms, mode='lines', name='grad_norm'))
        fig.update_layout(title='Gradient Norm over Steps', xaxis_title='step', yaxis_title='||grad||')
        return fig

    def plot_update_ratio(self) -> go.Figure:
        steps = [d['step'] for d in self.train_steps]
        ratios = [d['update_ratio'] for d in self.train_steps]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=ratios, mode='lines', name='update/weight'))
        fig.update_layout(title='Update/Weight Ratio', xaxis_title='step', yaxis_title='||Δw||/||w||')
        return fig

    def plot_grad_noise(self) -> go.Figure:
        steps = [d['step'] for d in self.train_steps]
        noise = [d['grad_noise'] for d in self.train_steps]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=noise, mode='lines', name='grad_noise_index'))
        fig.update_layout(title='Gradient Noise Index (std/mean of ||grad||)', xaxis_title='step', yaxis_title='std/mean')
        return fig

    def plot_training_landscape_3d(self) -> Optional[go.Figure]:
        if len(self._grad_snapshots) < 3:
            return None
        G = np.stack(self._grad_snapshots, axis=0)  # [T, D]
        # Center
        Gc = G - G.mean(axis=0, keepdims=True)
        # PCA via SVD, take first 2 PCs
        try:
            U, S, Vt = np.linalg.svd(Gc, full_matrices=False)
            PCs = Vt[:2].T  # [D, 2]
            XY = Gc @ PCs     # [T, 2]
        except Exception:
            # Fallback: random projection
            rng = np.random.default_rng(0)
            R = rng.standard_normal(size=(G.shape[1], 2))
            XY = Gc @ R

        steps = [d['step'] for d in self.train_steps[:XY.shape[0]]]
        losses = [d['loss'] for d in self.train_steps[:XY.shape[0]]]
        xs, ys = XY[:, 0], XY[:, 1]

        # Build a coarse surface via bin-averaged losses
        nx, ny = 40, 40
        # Pad ranges for readability
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        x_pad = max(1e-6, 0.2 * (x_max - x_min))
        y_pad = max(1e-6, 0.2 * (y_max - y_min))
        x_edges = np.linspace(x_min - x_pad, x_max + x_pad, nx + 1)
        y_edges = np.linspace(y_min - y_pad, y_max + y_pad, ny + 1)
        counts, _, _ = np.histogram2d(xs, ys, bins=(x_edges, y_edges))
        sums, _, _ = np.histogram2d(xs, ys, bins=(x_edges, y_edges), weights=np.array(losses))
        Z = sums / (counts + 1e-12)
        Z[counts < 1] = np.nan
        Xc = 0.5 * (x_edges[:-1] + x_edges[1:])
        Yc = 0.5 * (y_edges[:-1] + y_edges[1:])

        surface = go.Surface(x=Xc, y=Yc, z=Z.T, colorscale='Viridis', opacity=0.65, showscale=True, colorbar=dict(title='loss'))
        path = go.Scatter3d(
            x=xs, y=ys, z=losses,
            mode='lines+markers',
            marker=dict(size=2, color='blue', opacity=0.8),
            line=dict(width=6, color='rgba(30, 90, 200, 0.9)')
        )

        start = go.Scatter3d(x=[xs[0]], y=[ys[0]], z=[losses[0]], mode='markers', marker=dict(size=6, color='green'), name='start')
        end = go.Scatter3d(x=[xs[-1]], y=[ys[-1]], z=[losses[-1]], mode='markers', marker=dict(size=6, color='red'), name='end')

        fig = go.Figure(data=[surface, path, start, end])
        fig.update_layout(
            title='Gradient-derived Training Landscape (surface + path)',
            width=1100, height=700,
            scene=dict(
                xaxis_title='PC1(grad)', yaxis_title='PC2(grad)', zaxis_title='loss',
                xaxis=dict(range=[x_edges[0], x_edges[-1]]),
                yaxis=dict(range=[y_edges[0], y_edges[-1]]),
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig

    def plot_confusion_matrix(self, normalize: bool = True) -> Optional[go.Figure]:
        if not self._last_confusion:
            return None
        step, cm = self._last_confusion
        cm_plot = cm.astype(np.float64)
        if normalize:
            row_sums = cm_plot.sum(axis=1, keepdims=True) + 1e-12
            cm_plot = cm_plot / row_sums
        fig = go.Figure(data=go.Heatmap(z=cm_plot, colorscale='Blues', colorbar=dict(title='freq')))
        fig.update_layout(title=f'Confusion Matrix (step {step})', xaxis_title='Pred', yaxis_title='True')
        return fig

    def end_of_training(self, show: bool = True, report: bool = True) -> None:
        # Build plots
        loss_fig, acc_fig = self.plot_curves()
        grad_fig = self.plot_grad_norm()
        upd_fig = self.plot_update_ratio()
        noise_fig = self.plot_grad_noise()
        land_fig = self.plot_training_landscape_3d()
        cm_fig = self.plot_confusion_matrix(normalize=True)

        # Show locally
        if show:
            try:
                loss_fig.show(); acc_fig.show(); grad_fig.show(); upd_fig.show(); noise_fig.show()
                if land_fig is not None:
                    land_fig.show()
                if cm_fig is not None:
                    cm_fig.show()
            except Exception:
                pass

        # Report to ClearML
        if report and self.logger is not None:
            try:
                self.logger.report_plotly('curves', 'loss_train_vs_val', loss_fig, iteration=0)
                self.logger.report_plotly('curves', 'acc_train_vs_val', acc_fig, iteration=0)
                self.logger.report_plotly('diagnostics', 'grad_norm', grad_fig, iteration=0)
                self.logger.report_plotly('diagnostics', 'update_weight_ratio', upd_fig, iteration=0)
                self.logger.report_plotly('diagnostics', 'grad_noise_index', noise_fig, iteration=0)
                if land_fig is not None:
                    self.logger.report_plotly('diagnostics', 'gradient_landscape', land_fig, iteration=0)
                if cm_fig is not None:
                    self.logger.report_plotly('val', 'confusion_matrix_final', cm_fig, iteration=0)
            except Exception:
                pass

    def overfitting_indicator(self) -> Dict[str, float]:
        # Simple diagnostics: last val - train gap and trend
        if not self.val_steps:
            return {}
        # Align by nearest step
        v = self.val_steps[-1]
        # Find closest train step
        t = min(self.train_steps, key=lambda d: abs(d['step'] - v['step']))
        gap_loss = v['loss'] - t['loss']
        gap_acc = t['acc'] - v['acc']
        return {
            'loss_gap_last': float(gap_loss),
            'acc_gap_last': float(gap_acc),
        }

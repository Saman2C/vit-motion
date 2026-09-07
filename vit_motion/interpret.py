"""Interpretability for the ViT-Motion multimodal regression model.

This module adds three complementary, regression-aware explanations that all
run through the model's real ``forward(image, numeric_sequence, image_age)``
signature and depend only on packages the project already uses
(``torch`` / ``numpy`` / ``matplotlib``):

1. ``grad_cam``            -> WHERE in the cached RGB frame mattered
                             (Grad-CAM, Selvaraju et al. 2017; ViT token reshape).
2. ``integrated_gradients`` -> per-(step, channel) numeric attribution + per-pixel
                             image attribution, a joint attribution over the image
                             and the [B,K,5] sequence (Integrated Gradients,
                             Sundararajan 2017). NOTE: a single image-vs-numeric IG
                             scalar is confounded by dimensionality (150k pixels vs.
                             30 numerics) -- use ``temporal_attention`` for the
                             modality verdict; IG is for the *pattern* within each
                             modality. Both raw-sum and per-input shares are returned
                             for transparency but are not the headline number.
3. ``temporal_attention``  -> token importance inside the fusion encoder: the
                             single image token vs. the K numeric-step tokens
                             (Attention Rollout, Abnar & Zuidema 2020).

Because the visual encoder is average-pooled *before* fusion, the image enters
the temporal encoder as ONE token. So ``temporal_attention`` / ``modality_share``
tell you "image-as-a-whole vs. numeric", while ``grad_cam`` tells you *where* in
the image -- the two are complementary, not redundant.

A regression model has no class logit, so every method reduces the 3-D output to
a scalar via a selectable ``target`` in {"dx", "dy", "yaw", "norm"}.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Scalar target selection (regression has no class logit)
# --------------------------------------------------------------------------- #
_TARGET_INDEX = {"dx": 0, "dy": 1, "yaw": 2}


def make_scalar_target(target: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function mapping model output [B,3] -> scalar-per-batch [B]."""
    if target == "norm":
        return lambda out: out.norm(dim=1)
    if target in _TARGET_INDEX:
        idx = _TARGET_INDEX[target]
        return lambda out: out[:, idx]
    raise ValueError("target must be one of {'dx','dy','yaw','norm'}")


# ImageNet normalization used by the dataset transform; needed only to build a
# visually meaningful RGB for the overlay (model still sees normalized tensors).
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _gaussian_blur(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on a [B,1,H,W] tensor (no scipy dependency)."""
    radius = max(1, int(round(2.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=t.device, dtype=t.dtype)
    k = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    t = torch.nn.functional.conv2d(t, kx, padding=(0, radius))
    t = torch.nn.functional.conv2d(t, ky, padding=(radius, 0))
    return t


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """[3,H,W] normalized tensor -> [H,W,3] float image in [0,1] for display."""
    img = image.detach().cpu().float()
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


@dataclass
class Explanation:
    target: str
    prediction: np.ndarray                    # [3] raw (normalized-space) model output
    rgb: np.ndarray                           # [H,W,3] display image
    grad_cam: np.ndarray | None = None        # [H,W] in [0,1]
    ig_image: np.ndarray | None = None        # [H,W] absolute attribution (norm 0..1)
    ig_numeric: np.ndarray | None = None      # [K,5] signed attribution
    modality_share: dict[str, float] = field(default_factory=dict)
    token_importance: np.ndarray | None = None  # [K+1] image + K numeric steps
    meta: dict = field(default_factory=dict)


class MotionInterpreter:
    """Wraps a trained :class:`ViTMotionModel` with interpretability methods."""

    NUMERIC_CHANNELS = [
        "current_body_dx",
        "current_body_dy",
        "current_delta_yaw",
        "left_mps",
        "right_mps",
    ]

    def __init__(self, model: nn.Module, device: torch.device | str | None = None):
        self.model = model
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    # -- input helpers ------------------------------------------------------ #
    def _prep(self, image, numeric, image_age):
        image = image.to(self.device).float()
        numeric = numeric.to(self.device).float()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if numeric.ndim == 2:
            numeric = numeric.unsqueeze(0)
        if image_age is None:
            image_age = torch.zeros(image.shape[0], device=self.device)
        else:
            image_age = torch.as_tensor(image_age, device=self.device).float().reshape(-1)
        return image, numeric, image_age

    # --------------------------------------------------------------------- #
    # 1) Grad-CAM on the visual encoder
    # --------------------------------------------------------------------- #
    def grad_cam(self, image, numeric, image_age=None, target="yaw"):
        image, numeric, image_age = self._prep(image, numeric, image_age)
        scalar_fn = make_scalar_target(target)

        # Hook the last transformer block's first norm -> tokens [B, N, C].
        blocks = self.model.encoder.blocks
        layer = blocks[-1].norm1 if hasattr(blocks[-1], "norm1") else blocks[-1]
        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        def fwd_hook(_m, _i, out):
            activations["value"] = out
            out.register_hook(lambda g: gradients.__setitem__("value", g))

        handle = layer.register_forward_hook(fwd_hook)
        try:
            self.model.zero_grad(set_to_none=True)
            out = self.model(image, numeric, image_age)
            scalar = scalar_fn(out).sum()
            scalar.backward()
        finally:
            handle.remove()

        act = activations["value"].detach()      # [B, N, C]
        grad = gradients["value"].detach()        # [B, N, C]
        n_prefix = int(getattr(self.model.encoder, "num_prefix_tokens", 1))
        act = act[:, n_prefix:, :]
        grad = grad[:, n_prefix:, :]
        weights = grad.mean(dim=1, keepdim=True)  # [B,1,C]
        cam = (weights * act).sum(dim=-1)         # [B, N_patch]
        cam = torch.relu(cam)
        b, n = cam.shape
        side = int(round(n ** 0.5))
        cam = cam.reshape(b, side, side)
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1), size=image.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        cam = cam[0]
        cam = cam - cam.min()
        denom = cam.max().clamp_min(1e-8)
        return (cam / denom).cpu().numpy(), out.detach()[0].cpu().numpy()

    # --------------------------------------------------------------------- #
    # 2) Integrated Gradients over image + numeric sequence
    # --------------------------------------------------------------------- #
    def integrated_gradients(self, image, numeric, image_age=None, target="yaw", steps=32):
        image, numeric, image_age = self._prep(image, numeric, image_age)
        scalar_fn = make_scalar_target(target)
        img_base = torch.zeros_like(image)
        num_base = torch.zeros_like(numeric)

        img_grads = torch.zeros_like(image)
        num_grads = torch.zeros_like(numeric)
        alphas = torch.linspace(1.0 / steps, 1.0, steps, device=self.device)
        for a in alphas:
            img_in = (img_base + a * (image - img_base)).clone().requires_grad_(True)
            num_in = (num_base + a * (numeric - num_base)).clone().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            out = self.model(img_in, num_in, image_age)
            scalar = scalar_fn(out).sum()
            gi, gn = torch.autograd.grad(scalar, (img_in, num_in))
            img_grads += gi.detach()
            num_grads += gn.detach()

        img_attr = ((image - img_base) * img_grads / steps).detach()[0]   # [3,H,W]
        num_attr = ((numeric - num_base) * num_grads / steps).detach()[0]  # [K,5]

        img_map = img_attr.abs().sum(dim=0).cpu().numpy()                  # [H,W]
        if img_map.max() > 0:
            img_map = img_map / img_map.max()
        num_np = num_attr.cpu().numpy()

        # Modality share must NOT compare raw summed |attribution|: the image has
        # 3*H*W (~150k) inputs vs. only K*5 (=30) numeric inputs, so a raw sum makes
        # the image dominate purely by dimensionality (this is why the old share read
        # ~image 1.00 / numeric 0.00). We instead compare MEAN |attribution| per input
        # element, which is dimensionality-fair. Raw totals are kept for reference.
        img_total = float(img_attr.abs().sum().item())
        num_total = float(num_attr.abs().sum().item())
        img_mean = float(img_attr.abs().mean().item())
        num_mean = float(num_attr.abs().mean().item())
        mdenom = img_mean + num_mean + 1e-12
        modality = {
            "image": img_mean / mdenom,           # dimensionality-fair (per-input) share
            "numeric": num_mean / mdenom,
            "image_mean_abs": img_mean,
            "numeric_mean_abs": num_mean,
            "image_total_abs": img_total,          # raw sums (biased by #inputs; reference only)
            "numeric_total_abs": num_total,
        }
        return img_map, num_np, modality

    # --------------------------------------------------------------------- #
    # 3) Attention rollout across the temporal fusion encoder
    # --------------------------------------------------------------------- #
    def temporal_attention(self, image, numeric, image_age=None):
        image, numeric, image_age = self._prep(image, numeric, image_age)
        layers = list(self.model.temporal_encoder.layers)
        captured: list[torch.Tensor] = []

        def wrap(mha: nn.MultiheadAttention):
            original = mha.forward

            def patched(query, key, value, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True
                out, weights = original(query, key, value, **kwargs)
                captured.append(weights.detach())  # [B, L, L]
                return out, weights

            return original, patched

        originals = []
        for layer in layers:
            orig, patched = wrap(layer.self_attn)
            originals.append((layer, orig))
            layer.self_attn.forward = patched

        try:
            # Force the slow (python) attention path -- the fused eval kernel
            # skips self_attn.forward, so we run under grad with grad-enabled
            # tokens to disable it and actually trigger our wrapper.
            with torch.enable_grad():
                visual = self.model.encode_image(image).requires_grad_(True)
                self.model.predict_from_feature(visual, numeric, image_age)
        finally:
            for layer, orig in originals:
                layer.self_attn.forward = orig

        if not captured:
            return None, {}
        # Attention rollout: A_hat = 0.5*A + 0.5*I, row-normalize, chain-multiply.
        rollout = None
        for attn in captured:
            a = attn[0]                                  # [L, L]
            eye = torch.eye(a.shape[-1], device=a.device)
            a = 0.5 * a + 0.5 * eye
            a = a / a.sum(dim=-1, keepdim=True)
            rollout = a if rollout is None else a @ rollout
        # Prediction is read from the LAST token; its row = influence of each token.
        importance = rollout[-1].cpu().numpy()           # [L] = [image, step_1..K]
        share = {
            "image": float(importance[0]),
            "numeric": float(importance[1:].sum()),
        }
        return importance, share

    # --------------------------------------------------------------------- #
    # Input-gradient saliency (the SAME quantity the RRR fine-tune penalizes)
    # --------------------------------------------------------------------- #
    def input_saliency(self, image, numeric, image_age=None, target="yaw", smooth_sigma=3.0):
        """|d(target)/d(image)| summed over channels -> [H,W] map in [0,1].

        This is exactly the quantity finetune_rrr.py regularizes, so measuring /
        visualizing it (instead of Grad-CAM) keeps the explanation aligned with
        what was optimized. Only first-order gradients (no double backprop).
        """
        image, numeric, image_age = self._prep(image, numeric, image_age)
        scalar_fn = make_scalar_target(target)
        img = image.clone().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        out = self.model(img, numeric, image_age)
        g = torch.autograd.grad(scalar_fn(out).sum(), img)[0]      # [1,3,H,W]
        sal = g.abs().sum(dim=1, keepdim=True)                     # [1,1,H,W]
        if smooth_sigma and smooth_sigma > 0:
            sal = _gaussian_blur(sal, float(smooth_sigma))
        sal = sal[0, 0]
        sal = sal - sal.min()
        sal = sal / sal.max().clamp_min(1e-8)
        return sal.detach().cpu().numpy(), out.detach()[0].cpu().numpy()

    def saliency_map(self, image, numeric, image_age=None, target="yaw",
                     method="gradcam", smooth_sigma=3.0):
        """Unified accessor: method in {'gradcam','inputgrad'} -> ([H,W], pred)."""
        if method == "inputgrad":
            return self.input_saliency(image, numeric, image_age, target=target, smooth_sigma=smooth_sigma)
        return self.grad_cam(image, numeric, image_age, target=target)

    # --------------------------------------------------------------------- #
    # Convenience: run all three at once
    # --------------------------------------------------------------------- #
    def explain(self, image, numeric, image_age=None, target="yaw", ig_steps=32) -> Explanation:
        cam, pred = self.grad_cam(image, numeric, image_age, target=target)
        ig_img, ig_num, ig_share = self.integrated_gradients(
            image, numeric, image_age, target=target, steps=ig_steps
        )
        importance, attn_share = self.temporal_attention(image, numeric, image_age)
        img_t, _, _ = self._prep(image, numeric, image_age)
        return Explanation(
            target=target,
            prediction=pred,
            rgb=denormalize_image(img_t[0]),
            grad_cam=cam,
            ig_image=ig_img,
            ig_numeric=ig_num,
            modality_share={"integrated_gradients": ig_share, "attention": attn_share},
            token_importance=importance,
            meta={"ig_steps": ig_steps},
        )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_explanation(exp: Explanation, save_path=None, title=None):
    """Render a 4-panel figure summarizing one sample's explanation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.4), constrained_layout=True)

    axes[0].imshow(exp.rgb)
    axes[0].set_title("Cached RGB (model input)")
    axes[0].axis("off")

    axes[1].imshow(exp.rgb)
    if exp.grad_cam is not None:
        axes[1].imshow(exp.grad_cam, cmap="jet", alpha=0.45)
    axes[1].set_title(f"Grad-CAM  |  target = {exp.target}")
    axes[1].axis("off")

    # Token importance (image vs K numeric steps) + modality share.
    ax = axes[2]
    if exp.token_importance is not None:
        imp = exp.token_importance
        labels = ["image"] + [f"t-{len(imp) - 1 - i}" for i in range(1, len(imp))]
        colors = ["#EF6C00"] + ["#1565C0"] * (len(imp) - 1)
        ax.bar(range(len(imp)), imp, color=colors)
        ax.set_xticks(range(len(imp)))
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel("attention rollout weight")
        share = exp.modality_share.get("attention", {})
        ax.set_title(
            "Token importance (fusion encoder)\n"
            f"image {share.get('image', 0):.2f}  vs  numeric {share.get('numeric', 0):.2f}"
        )
    else:
        ax.set_title("Token importance (unavailable)")

    # IG numeric heatmap [K steps x 5 channels].
    ax = axes[3]
    if exp.ig_numeric is not None:
        data = exp.ig_numeric
        vmax = np.abs(data).max() or 1.0
        im = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels([f"t-{data.shape[0] - 1 - i}" for i in range(data.shape[0])])
        ax.set_xticks(range(len(MotionInterpreter.NUMERIC_CHANNELS)))
        ax.set_xticklabels(
            ["dx", "dy", "dyaw", "L m/s", "R m/s"], rotation=30, ha="right"
        )
        # NOTE: we deliberately do NOT print an image-vs-numeric scalar here. Any such
        # scalar from IG is confounded by dimensionality (3*H*W image inputs vs. K*5
        # numeric): summed |attr| favors the image, mean |attr| favors numeric. The
        # principled modality verdict is the token-level attention share (panel 3),
        # where the image is exactly ONE token vs. the K numeric tokens. This panel
        # shows only the per-(step,channel) numeric attribution pattern.
        ax.set_title("Integrated Gradients — per-channel numeric attribution")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.set_title("IG numeric (unavailable)")

    pred = exp.prediction
    suptitle = title or (
        f"prediction (normalized): dx={pred[0]:.3f}  dy={pred[1]:.3f}  dyaw={pred[2]:.3f}"
    )
    fig.suptitle(suptitle, fontsize=13)
    if save_path is not None:
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
        return save_path
    return fig

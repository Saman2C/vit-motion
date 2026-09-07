#!/usr/bin/env python3
"""Side-by-side saliency maps for any number of checkpoints, on one figure.

Built for the comparison this project actually needs: the original model, the
v1 fine-tune (top-half penalty), and the v3 guided fine-tune (non-ground penalty
plus the ground-spread term), on the same frames, with the same saliency, drawn
against the same segmentation mask.

Each panel carries the two numbers that matter, because the first version's
failure is invisible without the second one:

    distractor   share of the saliency on sky + trees + buildings + cars
    coverage     effective share of the GROUND the attention covers, exp(H)/N

v1 scores a near-perfect distractor number and a terrible coverage — that is the
"three hot spots on the grass" result, made legible.

Example:
    python compare_interpret.py --config config_run.yaml \
        --checkpoints original=artifacts/runs/v03_base/best.pt \
                      v1_tophalf=artifacts/runs/v1_half/best.pt \
                      v3_guided=artifacts/runs/v03_guided/best.pt \
        --sky-mask-dir artifacts/sky_masks --experiment 20260819_afernoon_wet_002 \
        --num-samples 4 --target yaw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vit_motion.attention_focus import region_stats, spread_stats
from vit_motion.config import load_config, resolve_from_config
from vit_motion.interpret import MotionInterpreter, denormalize_image
from vit_motion.sky_mask import (
    build_masker,
    cache_provides_ground,
    half_frame_mask,
    load_mask_quality,
    mask_cache_path,
    read_mask_png,
)
from vit_motion.validation import (
    ExperimentDataset,
    load_experiment_frame,
    load_model,
    load_normalization,
)

GROUND_LINE, SKY_LINE = "#1baf7a", "#2a78d6"
PANEL_COLORS = ["#6b7280", "#B3261E", "#1B7A3D", "#8b5cf6", "#eb6834"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="name=path pairs, in the order they should appear.")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--target", default="yaw", choices=["dx", "dy", "yaw", "norm"])
    ap.add_argument("--saliency", default="inputgrad", choices=["gradcam", "inputgrad"])
    ap.add_argument("--smooth-sigma", type=float, default=3.0)
    ap.add_argument("--sky-mask-dir", default="artifacts/sky_masks")
    ap.add_argument("--mask-on-the-fly", action="store_true")
    ap.add_argument("--masker", default="auto", choices=["auto", "segmentation", "energy"])
    ap.add_argument("--horizon-frac", type=float, default=0.5)
    ap.add_argument("--output", default="artifacts/interpretability/compare_focus.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cp = cfg["_config_path"]
    manifest_dir = resolve_from_config(cfg["data"]["manifest_dir"], cp)
    out_path = resolve_from_config(args.output, cp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_dir = resolve_from_config(args.sky_mask_dir, cp) if args.sky_mask_dir else None

    models: dict[str, Path] = {}
    for item in args.checkpoints:
        if "=" not in item:
            raise SystemExit(f"--checkpoints expects name=path, got: {item}")
        name, path = item.split("=", 1)
        models[name] = resolve_from_config(path, cp)

    frame, eid = load_experiment_frame(manifest_dir / "manifest.csv", args.experiment)
    norm = load_normalization(manifest_dir / "normalization.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    interpreters = {name: MotionInterpreter(load_model(path, cfg, device), device)
                    for name, path in models.items()}
    reference = next(iter(interpreters.values())).model

    image_size = tuple(int(x) for x in cfg["data"]["image_size"])
    sp = float(cfg["data"].get("sample_period_sec", 0.1))
    ds = ExperimentDataset(frame, norm, image_size, reference.sequence_length,
                           reference.image_update_interval, sp)

    live_masker = build_masker(args.masker) if args.mask_on_the_fly else None
    has_ground = (type(live_masker).__name__ == "SegformerSkyMasker" if live_masker
                  else (cache_provides_ground(mask_dir) if mask_dir else False))
    fallback_sky = half_frame_mask(*image_size, horizon_frac=args.horizon_frac).sky
    quality = load_mask_quality(mask_dir) if (mask_dir and not args.mask_on_the_fly) else {}

    n_rows = min(args.num_samples, len(ds))
    n_cols = 1 + len(models)
    picked = np.linspace(0, len(ds) - 1, n_rows, dtype=int)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols + 0.6, 3.7 * n_rows),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for r, i in enumerate(picked):
        item = ds[int(i)]
        rgb = denormalize_image(item["image"])
        rgb_path = ds.frame.iloc[ds.windows[int(i)][1]]["rgb_abs_path"]

        regions = None
        if live_masker is not None:
            from PIL import Image as _Image
            with _Image.open(rgb_path) as img:
                regions = live_masker(img.convert("RGB"), size=image_size)
        elif mask_dir is not None:
            path = mask_cache_path(mask_dir, eid, rgb_path)
            if Path(path).is_file():
                regions = read_mask_png(path, size=image_size, provides_ground=has_ground)
        if regions is None:
            sky, ground = fallback_sky, np.zeros_like(fallback_sky)
            other = ~fallback_sky
        else:
            sky, ground, other = regions.sky, regions.ground, regions.other
        distractor = sky | other

        ax = axes[r, 0]
        ax.imshow(rgb)
        overlay = np.zeros((*sky.shape, 4))
        overlay[ground] = [0.11, 0.69, 0.48, 0.30]
        ax.imshow(overlay)
        if ground.any():
            ax.contour(ground.astype(float), levels=[0.5], colors="white", linewidths=2.0)
            ax.contour(ground.astype(float), levels=[0.5], colors=GROUND_LINE, linewidths=1.0)
        ax.axis("off")
        trusted = quality.get((eid, Path(rgb_path).stem), True)
        title = f"ground = {ground.mean():.0%} of frame" if ground.any() else "no ground mask"
        if not trusted:
            title += "  ·  MASK NOT TRUSTED"
        ax.set_title(title, fontsize=9.5, color=(GROUND_LINE if trusted else "#B3261E"))
        if r == 0:
            ax.text(0.5, 1.14, "Cached RGB + ground mask", transform=ax.transAxes,
                    ha="center", fontsize=12.5)

        for c, (name, interp) in enumerate(interpreters.items(), start=1):
            saliency, _ = interp.saliency_map(
                item["image"], item["numeric"], item["image_age"],
                target=args.target, method=args.saliency, smooth_sigma=args.smooth_sigma,
            )
            dist = region_stats(saliency, distractor)["fraction"]
            spread = spread_stats(saliency, ground) if ground.any() else {"coverage": float("nan")}
            ax = axes[r, c]
            ax.imshow(rgb)
            ax.imshow(saliency, cmap="jet", alpha=0.45)
            if ground.any():
                ax.contour(ground.astype(float), levels=[0.5], colors="white", linewidths=1.4)
            ax.axis("off")
            colour = PANEL_COLORS[(c - 1) % len(PANEL_COLORS)]
            label = f"distractor {dist:.0%}"
            if np.isfinite(spread["coverage"]):
                label += f" · covers {spread['coverage']:.0%} of ground"
            ax.set_title(label, fontsize=9.5, color=colour)
            if r == 0:
                ax.text(0.5, 1.14, name, transform=ax.transAxes, ha="center",
                        fontsize=12.5, color=colour)

    fig.suptitle(
        f"Where the model looks, and how much of the ground it uses\n"
        f"{eid} · target={args.target} · distractor = sky + trees + buildings + cars",
        fontsize=11.5,
    )
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved comparison to: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Attention-guided fine-tuning: look at the ground, and look at all of it.

The goal is not "avoid the sky" — it is "use the surface the vehicle is driving
over".  For a tracked vehicle the motion cue lives in the mud, dirt and grass in
front of it; trees, buildings, parked cars, tents and sky are all equally
useless, and the first version of this fine-tune only ever penalized the last of
those, using the top half of the frame as a stand-in for it.

Two terms, both computed on the input-gradient saliency of the chosen target,
both bounded in [0, 1] so a single lambda each is enough:

    placement   share of the saliency mass landing in the DISTRACTOR region
                (everything that is not ground).  RobustViT (Chefer et al.,
                NeurIPS 2022) / Right for the Right Reasons (Ross et al.,
                IJCAI 2017).  -> minimize.

    spread      1 - H(saliency | ground) / log(N_ground).  -> minimize, i.e.
                maximize the entropy of the attention *within* the ground.

The second term exists because of what happened without it: the placement
penalty alone drove sky saliency from 45% to 0%, and left the attention piled
onto two or three isolated hot spots on the grass.  The penalty had no opinion
about that — it only says "not outside", never "spread out inside".

    total_loss = task_loss + lambda_rrr * placement + lambda_spread * spread

Region modes:

``--mask-mode nonground``  penalize sky + trees + buildings + cars.  The default,
                           and the one the project is actually about.  REQUIRES a
                           mask cache built by a semantic segmenter — the script
                           refuses to run on sky-only masks, because there
                           "non-ground" would include the grass itself.
``--mask-mode sky``        penalize only the segmented sky.  The intermediate
                           version, for the ablation.
``--mask-mode half``       the original fixed top-half region.  v1, reproduced
                           exactly so the three can be compared on one figure.

Set ``--lambda-rrr 0 --lambda-spread 0`` for the control run (same extra
training, no guidance) and ``--lambda-spread 0`` to isolate the spread term.

Example:
    python finetune_rrr.py --config config_run.yaml \
        --checkpoint artifacts/runs/v03_base/best.pt \
        --sky-mask-dir artifacts/sky_masks --mask-mode nonground \
        --epochs 3 --lambda-rrr 1.0 --lambda-spread 0.5 --target yaw \
        --output artifacts/runs/v03_guided/best.pt
"""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vit_motion.attention_focus import masked_fraction, normalized_entropy
from vit_motion.config import load_config, resolve_from_config
from vit_motion.dataset import MotionDataset
from vit_motion.model import ViTMotionModel
from vit_motion.sky_mask import cache_provides_ground, half_frame_mask

# The saliency penalty needs a SECOND derivative through attention. The fused
# flash / mem-efficient SDPA kernels do not implement double-backward (on CPU it
# errors outright), so we force the math kernel, which is fully differentiable on
# both CPU and CUDA. Fall back gracefully across torch versions.
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    @contextmanager
    def math_attention():
        with sdpa_kernel(SDPBackend.MATH):
            yield
except Exception:  # pragma: no cover - older torch

    @contextmanager
    def math_attention():
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
        yield


def distractor_mask(batch: dict, mode: str, fixed_half: torch.Tensor):
    """``(mask [B,1,H,W], valid [B])`` for the region whose saliency is penalized."""
    device = fixed_half.device
    batch_size = batch["image"].shape[0]
    if mode == "half":
        mask = fixed_half.expand(batch_size, -1, -1, -1)
        return mask, torch.ones(batch_size, device=device)
    has = batch["has_mask"].to(device).reshape(-1)
    if mode == "sky":
        mask = batch["sky_mask"].to(device)
    elif mode == "nonground":
        mask = 1.0 - batch["ground_mask"].to(device)
    else:
        raise ValueError("mask-mode must be one of {'nonground','sky','half'}")
    # A frame whose penalty region is empty (no sky in view) contributes no
    # gradient; excluding it keeps the logged penalty an average over frames
    # where the region actually exists.
    nonempty = (mask.sum(dim=(1, 2, 3)) > 0).float()
    return mask, has * nonempty


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--checkpoint", required=True, help="Starting checkpoint (e.g. best.pt).")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lambda-rrr", type=float, default=1.0,
                   help="Weight on the distractor-placement term. 0 = no placement penalty.")
    p.add_argument("--lambda-spread", type=float, default=0.5,
                   help="Weight on the ground-spread term. 0 = v2 behaviour (hot spots).")
    p.add_argument("--mask-mode", default="nonground", choices=["nonground", "sky", "half"])
    p.add_argument("--sky-mask-dir", default="artifacts/sky_masks")
    p.add_argument("--horizon-frac", type=float, default=0.5, help="Only for --mask-mode half.")
    p.add_argument("--area-normalize", action="store_true",
                   help="Penalize placement/area instead of placement (scale-free but noisier "
                        "on frames with a tiny region).")
    p.add_argument("--target", default="yaw", choices=["dx", "dy", "yaw", "norm"])
    p.add_argument("--max-steps", type=int, default=0, help="Cap steps/epoch (0 = all).")
    p.add_argument("--lr-scale", type=float, default=0.5)
    p.add_argument("--output", default="artifacts/runs/vit_motion_guided/best.pt")
    args = p.parse_args()

    cfg = load_config(args.config)
    config_path = cfg["_config_path"]
    data_cfg, train_cfg, model_cfg = cfg["data"], cfg["training"], cfg["model"]
    manifest_dir = resolve_from_config(data_cfg["manifest_dir"], config_path)
    out_path = resolve_from_config(args.output, config_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_size = tuple(int(x) for x in data_cfg["image_size"])
    K = int(model_cfg["sequence_length"])
    M = int(model_cfg.get("image_update_interval", 1))

    needs_masks = args.mask_mode != "half" or args.lambda_spread > 0
    mask_dir = resolve_from_config(args.sky_mask_dir, config_path) if args.sky_mask_dir else None
    if needs_masks:
        if mask_dir is None or not Path(mask_dir).is_dir():
            raise SystemExit(f"Mask cache not found: {mask_dir}\n"
                             f"Run precompute_sky_masks.py first.")
        has_ground = cache_provides_ground(mask_dir)
        if (args.mask_mode == "nonground" or args.lambda_spread > 0) and not has_ground:
            raise SystemExit(
                f"The mask cache at {mask_dir} has no usable GROUND channel "
                f"(index.json says provides_ground=false).\n"
                f"It was built by a sky-only detector, which cannot tell grass from a "
                f"tree — penalizing 'non-ground' with it would train the model to look "
                f"away from the terrain.\n"
                f"Rebuild it with --masker segmentation (needs the segmentation weights), "
                f"or use --mask-mode sky --lambda-spread 0."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_model_cfg = dict(ckpt.get("config", {}).get("model", model_cfg))
    saved_model_cfg["pretrained"] = False
    model = ViTMotionModel(**saved_model_cfg).to(device)
    model.load_state_dict(ckpt["model"])

    train_ds = MotionDataset(
        manifest_dir / "manifest.csv", manifest_dir / "normalization.json", "train",
        image_size, augment=True, sequence_length=K, image_update_interval=M,
        sky_mask_dir=mask_dir if needs_masks else None,
    )
    loader = DataLoader(train_ds, batch_size=int(train_cfg["batch_size"]),
                        shuffle=True, num_workers=int(train_cfg["num_workers"]), pin_memory=True)

    weights = torch.tensor([1.0, 1.0, float(train_cfg["yaw_loss_weight"])], device=device)

    def task_loss(pred, target):
        return (nn.functional.smooth_l1_loss(pred, target, reduction="none") * weights).mean()

    def scalar_of(pred):
        if args.target == "norm":
            return pred.norm(dim=1)
        return pred[:, {"dx": 0, "dy": 1, "yaw": 2}[args.target]]

    fixed_half = torch.from_numpy(
        half_frame_mask(*image_size, horizon_frac=args.horizon_frac).sky.astype(np.float32)
    ).to(device)[None, None]

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(train_cfg["learning_rate"]) * float(args.lr_scale),
                                  weight_decay=float(train_cfg["weight_decay"]))

    print(f"mask-mode={args.mask_mode} · lambda_rrr={args.lambda_rrr} · "
          f"lambda_spread={args.lambda_spread} · target={args.target} · "
          f"masks={mask_dir if needs_masks else 'fixed top half'}")
    history = []
    for epoch in range(args.epochs):
        model.train()
        run_task = run_place = run_spread = run_cov = 0.0
        count = covered = 0
        bar = tqdm(loader, desc=f"guided epoch {epoch + 1}")
        for step, batch in enumerate(bar):
            if args.max_steps and step >= args.max_steps:
                break
            image = batch["image"].to(device, non_blocking=True).requires_grad_(True)
            numeric = batch["numeric"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            image_age = batch["image_age"].to(device, non_blocking=True)
            mask, valid = distractor_mask(batch, args.mask_mode, fixed_half)
            ground = batch["ground_mask"].to(device) if needs_masks else None

            with math_attention():
                pred = model(image, numeric, image_age)
                t_loss = task_loss(pred, target)

                # Differentiable saliency = |d(scalar)/d(image)| summed over channels.
                grad_img = torch.autograd.grad(
                    scalar_of(pred).sum(), image, create_graph=True, retain_graph=True
                )[0]
                saliency = grad_img.abs().sum(dim=1, keepdim=True)          # [B,1,H,W]

                placement = masked_fraction(saliency, mask)                 # [B]
                if args.area_normalize:
                    area = mask.mean(dim=(1, 2, 3)).clamp_min(1e-4)
                    placement = placement / area
                place_loss = (placement * valid).sum() / valid.sum().clamp_min(1.0)

                if args.lambda_spread > 0 and ground is not None:
                    ground_valid = (batch["has_mask"].to(device).reshape(-1)
                                    * (ground.sum(dim=(1, 2, 3)) > 1).float())
                    entropy = normalized_entropy(saliency, ground)          # [B], 1 = even
                    spread_term = (1.0 - entropy)
                    spread_loss_value = ((spread_term * ground_valid).sum()
                                         / ground_valid.sum().clamp_min(1.0))
                    mean_entropy = ((entropy * ground_valid).sum()
                                    / ground_valid.sum().clamp_min(1.0))
                else:
                    spread_loss_value = torch.zeros((), device=device)
                    mean_entropy = torch.zeros((), device=device)

                loss = (t_loss
                        + args.lambda_rrr * place_loss
                        + args.lambda_spread * spread_loss_value)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
            optimizer.step()

            batch_size = image.shape[0]
            run_task += t_loss.item() * batch_size
            run_place += place_loss.item() * batch_size
            run_spread += spread_loss_value.item() * batch_size
            run_cov += mean_entropy.item() * batch_size
            count += batch_size
            covered += int(valid.sum().item())
            bar.set_postfix(task=run_task / count, place=run_place / count,
                            spread=run_spread / count, H=run_cov / count,
                            cover=f"{covered}/{count}")

        row = {"epoch": epoch + 1,
               "task_loss": run_task / count,
               "distractor_placement": run_place / count,
               "spread_loss": run_spread / count,
               "ground_normalized_entropy": run_cov / count,
               "mask_coverage": covered / max(count, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)

    torch.save({"epoch": args.epochs - 1, "model": model.state_dict(),
                "best_val_loss": float("nan"), "config": cfg,
                "guidance": {"lambda_rrr": args.lambda_rrr,
                             "lambda_spread": args.lambda_spread,
                             "mask_mode": args.mask_mode,
                             "sky_mask_dir": str(mask_dir) if needs_masks else None,
                             "horizon_frac": args.horizon_frac,
                             "area_normalize": bool(args.area_normalize),
                             "target": args.target, "history": history}},
               out_path)
    print(f"Saved guided checkpoint to: {out_path}")


if __name__ == "__main__":
    main()

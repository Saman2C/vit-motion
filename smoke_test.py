#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from vit_motion.dataset import MotionDataset
from vit_motion.model import ViTMotionModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=6)
    parser.add_argument("--image-update-interval", type=int, default=1)
    args = parser.parse_args()
    base = Path(args.manifest_dir)

    print("[1/4] Loading training dataset...", flush=True)
    step_start = time.perf_counter()
    ds = MotionDataset(
        base / "manifest.csv", base / "normalization.json", "train",
        sequence_length=args.sequence_length,
        image_update_interval=args.image_update_interval,
    )
    print(
        f"      Loaded {len(ds)} training windows "
        f"in {time.perf_counter() - step_start:.2f} s.",
        flush=True,
    )

    print("[2/4] Loading and checking the first sample...", flush=True)
    step_start = time.perf_counter()
    sample = ds[0]
    assert sample["image"].shape == (3, 224, 224)
    assert sample["numeric"].shape == (args.sequence_length, 5)
    assert sample["target"].shape == (3,)
    assert sample["image_age"].shape == ()
    print(
        f"      Sample shapes are valid "
        f"({time.perf_counter() - step_start:.2f} s).",
        flush=True,
    )

    print("[3/4] Building ViT + Temporal Transformer model...", flush=True)
    step_start = time.perf_counter()
    model = ViTMotionModel(
        pretrained=False,
        sequence_length=args.sequence_length,
        image_update_interval=args.image_update_interval,
    )
    model.eval()
    print(
        f"      Model built in {time.perf_counter() - step_start:.2f} s.",
        flush=True,
    )

    print("[4/4] Running forward pass (this may take a moment)...", flush=True)
    step_start = time.perf_counter()
    with torch.no_grad():
        out = model(
            sample["image"][None], sample["numeric"][None], sample["image_age"][None]
        )
    assert out.shape == (1, 3)
    print(
        f"      Forward pass completed in "
        f"{time.perf_counter() - step_start:.2f} s.",
        flush=True,
    )
    print(
        f"PASS: {len(ds)} training windows (K={args.sequence_length}, "
        f"M={args.image_update_interval}); "
        f"output shape={tuple(out.shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run interpretability on real samples from one experiment.

Mirrors evaluate_experiment.py: takes a config + checkpoint + experiment id,
uses the same load_model / manifest / normalization plumbing, and writes one
4-panel interpretability PNG per sampled window.

Example (Kaggle):
    python interpret_experiment.py \
        --config config_kaggle.yaml \
        --checkpoint /kaggle/working/vit_motion/artifacts/runs/vit_motion_temporal_cr_v0_2_1/best.pt \
        --experiment 20260804_morning_dry_001 \
        --num-samples 8 --target yaw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from vit_motion.config import load_config, resolve_from_config
from vit_motion.validation import (
    ExperimentDataset,
    load_experiment_frame,
    load_model,
    load_normalization,
)
from vit_motion.interpret import MotionInterpreter, render_explanation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", help="Checkpoint path; defaults to <training.output_dir>/best.pt")
    parser.add_argument("--experiment", help="Experiment ID. Omit for an interactive numbered list.")
    parser.add_argument("--output-dir", default="artifacts/interpretability")
    parser.add_argument("--num-samples", type=int, default=8, help="Evenly spaced windows to explain.")
    parser.add_argument("--target", default="yaw", choices=["dx", "dy", "yaw", "norm"])
    parser.add_argument("--ig-steps", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_path = cfg["_config_path"]
    manifest_dir = resolve_from_config(cfg["data"]["manifest_dir"], config_path)
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else resolve_from_config(cfg["training"]["output_dir"], config_path) / "best.pt"
    )
    output_root = resolve_from_config(args.output_dir, config_path)

    frame, experiment_id = load_experiment_frame(manifest_dir / "manifest.csv", args.experiment)
    normalization = load_normalization(manifest_dir / "normalization.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, cfg, device)
    image_size = tuple(int(x) for x in cfg["data"]["image_size"])
    sample_period_sec = float(cfg["data"].get("sample_period_sec", 0.1))
    dataset = ExperimentDataset(
        frame, normalization, image_size,
        model.sequence_length, model.image_update_interval, sample_period_sec,
    )

    interpreter = MotionInterpreter(model, device)
    output_dir = output_root / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int).tolist()
    print(f"Explaining {n} window(s) from '{experiment_id}' (target={args.target})...", flush=True)
    for rank, idx in enumerate(indices):
        item = dataset[idx]
        exp = interpreter.explain(
            item["image"], item["numeric"], item["image_age"],
            target=args.target, ig_steps=args.ig_steps,
        )
        out_path = output_dir / f"explain_{rank:02d}_win{idx:05d}_{args.target}.png"
        render_explanation(
            exp, save_path=out_path,
            title=f"{experiment_id}  |  window {idx}  |  target={args.target}",
        )
        print(f"  [{rank + 1}/{n}] saved {out_path.name}", flush=True)

    print(f"Saved interpretability figures to: {output_dir}")


if __name__ == "__main__":
    main()

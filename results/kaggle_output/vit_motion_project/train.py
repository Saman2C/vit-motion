#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vit_motion.config import load_config, resolve_from_config
from vit_motion.dataset import MotionDataset
from vit_motion.model import ViTMotionModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WeightedRegressionLoss(nn.Module):
    def __init__(self, kind: str, yaw_weight: float) -> None:
        super().__init__()
        self.kind = kind
        self.register_buffer("weights", torch.tensor([1.0, 1.0, yaw_weight]))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kind == "smooth_l1":
            elementwise = nn.functional.smooth_l1_loss(
                prediction, target, reduction="none"
            )
        elif self.kind == "mse":
            elementwise = nn.functional.mse_loss(prediction, target, reduction="none")
        else:
            raise ValueError("training.loss must be 'smooth_l1' or 'mse'")
        return (elementwise * self.weights).mean()


@torch.no_grad()
def evaluate(model, loader, criterion, device, description: str) -> dict[str, float]:
    model.eval()
    losses, errors, count = 0.0, torch.zeros(3, device=device), 0
    for batch in tqdm(loader, desc=description, leave=False):
        image = batch["image"].to(device, non_blocking=True)
        numeric = batch["numeric"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        image_age = batch["image_age"].to(device, non_blocking=True)
        prediction = model(image, numeric, image_age)
        losses += criterion(prediction, target).item() * len(image)
        physical_prediction = loader.dataset.denormalize_target(prediction)
        physical_target = loader.dataset.denormalize_target(target)
        errors += (physical_prediction - physical_target).abs().sum(dim=0)
        count += len(image)
    mae = (errors / count).cpu().tolist()
    return {
        "loss": losses / count,
        "mae_dx_m": mae[0],
        "mae_dy_m": mae[1],
        "mae_dyaw_rad": mae[2],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    data_cfg, train_cfg, model_cfg = cfg["data"], cfg["training"], cfg["model"]
    manifest_dir = resolve_from_config(data_cfg["manifest_dir"], cfg["_config_path"])
    output_dir = resolve_from_config(train_cfg["output_dir"], cfg["_config_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = tuple(int(x) for x in data_cfg["image_size"])
    sequence_length = int(model_cfg["sequence_length"])
    image_update_interval = int(model_cfg.get("image_update_interval", 1))

    train_ds = MotionDataset(
        manifest_dir / "manifest.csv",
        manifest_dir / "normalization.json",
        "train",
        image_size,
        augment=True,
        sequence_length=sequence_length,
        image_update_interval=image_update_interval,
    )
    all_manifest = pd.read_csv(manifest_dir / "manifest.csv")
    available_splits = set(all_manifest["split"])
    if "val" not in available_splits:
        raise SystemExit(
            "No validation experiments. Add at least 3 experiments and rerun inspect_dataset.py."
        )
    val_ds = MotionDataset(
        manifest_dir / "manifest.csv",
        manifest_dir / "normalization.json",
        "val",
        image_size,
        augment=False,
        sequence_length=sequence_length,
        image_update_interval=image_update_interval,
    )
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(train_cfg["num_workers"]),
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(train_cfg["num_workers"]),
            pin_memory=True,
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTMotionModel(**model_cfg).to(device)
    encoder_ids = {id(x) for x in model.encoder.parameters()}
    head_params = [x for x in model.parameters() if id(x) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": float(train_cfg["learning_rate"])
                * float(train_cfg["encoder_learning_rate_scale"]),
            },
            {"params": head_params, "lr": float(train_cfg["learning_rate"])},
        ],
        weight_decay=float(train_cfg["weight_decay"]),
    )
    criterion: nn.Module = WeightedRegressionLoss(
        str(train_cfg["loss"]), float(train_cfg["yaw_loss_weight"])
    ).to(device)
    scaler = torch.amp.GradScaler(
        device.type, enabled=bool(train_cfg["amp"]) and device.type == "cuda"
    )
    start_epoch, best_loss, stale = 0, float("inf"), 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved_model_cfg = checkpoint.get("config", {}).get("model", {})
        saved_k = int(saved_model_cfg.get("sequence_length", -1))
        saved_m = int(saved_model_cfg.get("image_update_interval", -1))
        if (saved_k, saved_m) != (sequence_length, image_update_interval):
            raise ValueError(
                "Resume checkpoint K/M mismatch: "
                f"checkpoint has K={saved_k}, M={saved_m}; "
                f"config requests K={sequence_length}, M={image_update_interval}."
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_val_loss"])

    history = []
    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        model.train()
        running, count = 0.0, 0
        bar = tqdm(loaders["train"], desc=f"epoch {epoch + 1}")
        for batch in bar:
            image = batch["image"].to(device, non_blocking=True)
            numeric = batch["numeric"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=bool(train_cfg["amp"]) and device.type == "cuda",
            ):
                image_age = batch["image_age"].to(device, non_blocking=True)
                prediction = model(image, numeric, image_age)
                loss = criterion(prediction, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(image)
            count += len(image)
            bar.set_postfix(loss=running / count)
        metrics = evaluate(
            model, loaders["val"], criterion, device,
            f"epoch {epoch + 1} validation"
        )
        row = {"epoch": epoch + 1, "train_loss": running / count, **metrics}
        history.append(row)
        print(json.dumps(row))
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": min(best_loss, metrics["loss"]),
            "config": cfg,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if metrics["loss"] < best_loss:
            best_loss, stale = metrics["loss"], 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            stale += 1
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        if stale >= int(train_cfg["patience"]):
            print(f"Early stopping after {stale} stale epochs.")
            break


if __name__ == "__main__":
    main()

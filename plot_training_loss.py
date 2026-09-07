#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training and validation loss from history.json.")
    parser.add_argument("--history", default="artifacts/runs/vit_motion_v1/history.json")
    parser.add_argument("--output", help="Output PNG; default is next to history.json")
    args = parser.parse_args()
    history_path = Path(args.history)
    frame = pd.DataFrame(json.loads(history_path.read_text(encoding="utf-8")))
    if frame.empty or not {"epoch", "train_loss", "loss"}.issubset(frame.columns):
        raise ValueError("history.json has no usable epoch/train_loss/loss records.")
    output = Path(args.output) if args.output else history_path.with_name("loss_curve.png")
    best_index = frame["loss"].idxmin()
    best_epoch = int(frame.loc[best_index, "epoch"])
    best_loss = float(frame.loc[best_index, "loss"])
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.plot(frame["epoch"], frame["train_loss"], label="Training loss", color="#1565C0", linewidth=1.8)
    axis.plot(frame["epoch"], frame["loss"], label="Validation loss", color="#EF6C00", linewidth=1.8)
    axis.scatter([best_epoch], [best_loss], color="#C62828", zorder=3)
    axis.annotate(f"Best validation loss = {best_loss:.6f}\nEpoch {best_epoch}", (best_epoch, best_loss),
                  xytext=(12, 16), textcoords="offset points", arrowprops={"arrowstyle": "->"})
    axis.set_title("ViT motion model training history")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weighted normalized regression loss (unitless)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    frame.to_csv(output.with_suffix(".csv"), index=False)
    print(f"Saved loss plot to: {output}")


if __name__ == "__main__":
    main()

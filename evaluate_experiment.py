#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vit_motion.config import load_config, resolve_from_config
from vit_motion.validation import DISPLAY_OUTPUTS, load_experiment_frame, load_normalization, predict_frame, regression_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one complete experiment and plot prediction versus ground truth.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", help="Checkpoint path; defaults to <training.output_dir>/best.pt")
    parser.add_argument("--experiment", help="Experiment ID. Omit for an interactive numbered list.")
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_path = cfg["_config_path"]
    manifest_dir = resolve_from_config(cfg["data"]["manifest_dir"], config_path)
    checkpoint = Path(args.checkpoint) if args.checkpoint else resolve_from_config(cfg["training"]["output_dir"], config_path) / "best.pt"
    output_root = resolve_from_config(args.output_dir, config_path)
    frame, experiment_id = load_experiment_frame(manifest_dir / "manifest.csv", args.experiment)
    normalization = load_normalization(manifest_dir / "normalization.json")
    result = predict_frame(frame, normalization, cfg, checkpoint, args.batch_size, args.num_workers)

    output_dir = output_root / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "predictions.csv", index=False)
    metrics = {"experiment_id": experiment_id, "checkpoint": str(checkpoint),
               "sequence_length": int(result.attrs.get("sequence_length", cfg["model"]["sequence_length"])),
               "image_update_interval": int(result.attrs.get("image_update_interval", cfg["model"].get("image_update_interval", 1))),
               "samples": len(result), "outputs": {}}
    for column, label, unit in DISPLAY_OUTPUTS:
        metrics["outputs"][column] = {"label": label, "unit": unit, **regression_metrics(
            result[column].to_numpy(float), result[f"predicted_{column}"].to_numpy(float))}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # Use a wide, vertically stacked layout so small changes remain visible.
    # Each output keeps an independent y scale, matching the style commonly
    # used by MATLAB System Identification comparison plots.
    fig, axes = plt.subplots(3, 1, figsize=(19, 10.5), sharex=True, constrained_layout=True)
    time = result["time_from_start_sec"].to_numpy(float)
    for axis, (column, label, unit) in zip(axes, DISPLAY_OUTPUTS):
        actual = result[column].to_numpy(float)
        predicted = result[f"predicted_{column}"].to_numpy(float)
        axis.plot(time, actual, color="#1565C0", linewidth=1.25, label="Ground truth")
        score = metrics["outputs"][column]
        r2 = "N/A" if score["r2"] is None else f'{score["r2"]:.3f}'
        fit = "N/A" if score["fit_percent"] is None else f'{score["fit_percent"]:.2f}%'
        axis.plot(time, predicted, color="#EF6C00", linewidth=1.15, label=f"Prediction — Fit (sys): {fit}")
        axis.set_title(
            f"{label}: prediction versus ground truth  |  "
            f"MAE={score['mae']:.5f} {unit}, RMSE={score['rmse']:.5f} {unit}, R²={r2}"
        )
        axis.set_ylabel(f"{label} ({unit})")
        combined = np.concatenate((actual, predicted))
        finite = combined[np.isfinite(combined)]
        if finite.size:
            lower, upper = float(finite.min()), float(finite.max())
            padding = max((upper - lower) * 0.06, 1e-6)
            axis.set_ylim(lower - padding, upper + padding)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Time from experiment start (s)")
    fig.suptitle(f"Motion prediction validation — {experiment_id}", fontsize=17)
    fig.savefig(output_dir / "prediction_vs_ground_truth.png", dpi=200)
    plt.close(fig)

    # Keep accumulated displacement as a separate diagnostic figure so it
    # cannot compress the three primary time-series comparisons.
    path_fig, path_axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    true_x = np.cumsum(result["next_body_dx"].to_numpy(float))
    true_y = np.cumsum(result["next_body_dy"].to_numpy(float))
    pred_x = np.cumsum(result["predicted_next_body_dx"].to_numpy(float))
    pred_y = np.cumsum(result["predicted_next_body_dy"].to_numpy(float))
    path_axis.plot(true_x, true_y, color="#1565C0", label="Ground truth (delta accumulation)")
    path_axis.plot(pred_x, pred_y, color="#EF6C00", label="Prediction (delta accumulation)")
    path_axis.scatter([0], [0], color="black", s=30, label="Start")
    path_axis.set_title("Accumulated body-frame displacement (diagnostic only)")
    path_axis.set_xlabel("Accumulated body x (m)")
    path_axis.set_ylabel("Accumulated body y (m)")
    path_axis.axis("equal")
    path_axis.grid(True, alpha=0.25)
    path_axis.legend()
    path_fig.suptitle(f"Accumulated displacement diagnostic — {experiment_id}", fontsize=15)
    path_fig.savefig(output_dir / "accumulated_displacement_diagnostic.png", dpi=180)
    plt.close(path_fig)
    print(f"Saved evaluation to: {output_dir}")


if __name__ == "__main__":
    main()

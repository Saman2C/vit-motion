#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vit_motion.config import load_config, resolve_from_config
from vit_motion.validation import DISPLAY_INPUTS, DISPLAY_OUTPUTS, load_experiment_frame, load_normalization, predict_frame


def put(frame, text, point, color=(235, 235, 235), scale=0.60, thickness=1):
    cv2.putText(frame, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def imread_unicode(path):
    """Read an image from a Windows/UNC path containing Unicode characters."""
    path = str(path)
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise FileNotFoundError(f"Cannot access image file:\n{path}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV cannot decode image:\n{path}")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an RGB validation video with numeric inputs and predicted/true outputs.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", help="Checkpoint path; defaults to <training.output_dir>/best.pt")
    parser.add_argument("--experiment", help="Experiment ID. Omit for an interactive numbered list.")
    parser.add_argument("--predictions-csv", help="Reuse evaluate_experiment.py output and skip inference.")
    parser.add_argument("--output", help="Output MP4 path")
    parser.add_argument("--fps", type=float, help="Output FPS; default uses data.sample_period_sec")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_path = cfg["_config_path"]
    manifest_dir = resolve_from_config(cfg["data"]["manifest_dir"], config_path)
    if args.predictions_csv:
        result = pd.read_csv(args.predictions_csv)
        experiment_id = str(result["experiment_id"].iloc[0])
    else:
        frame, experiment_id = load_experiment_frame(manifest_dir / "manifest.csv", args.experiment)
        checkpoint = Path(args.checkpoint) if args.checkpoint else resolve_from_config(cfg["training"]["output_dir"], config_path) / "best.pt"
        result = predict_frame(frame, load_normalization(manifest_dir / "normalization.json"), cfg, checkpoint,
                               args.batch_size, args.num_workers)

    output = Path(args.output) if args.output else resolve_from_config("artifacts/evaluation", config_path) / experiment_id / "validation_video.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    image_column = "model_rgb_abs_path" if "model_rgb_abs_path" in result else "rgb_abs_path"
    first = imread_unicode(result.iloc[0][image_column])
    image_h, image_w = first.shape[:2]
    display_h = 720
    display_w = int(round(image_w * display_h / image_h))
    panel_w = 640
    sample_period_sec = float(cfg["data"].get("sample_period_sec", 0.1))
    fps = args.fps or (1.0 / sample_period_sec)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (display_w + panel_w, display_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output}")

    for index, row in result.iterrows():
        image = imread_unicode(row[image_column])
        image = cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_AREA)
        panel = np.full((display_h, panel_w, 3), (28, 31, 36), dtype=np.uint8)
        put(panel, "ViT Motion Validation", (28, 42), (255, 255, 255), 0.85, 2)
        put(panel, f"Experiment: {experiment_id}", (28, 72), (190, 195, 205), 0.52)
        put(panel, f"Time: {float(row['time_from_start_sec']):7.2f} s    Frame: {index + 1}/{len(result)}", (28, 99), (190, 195, 205), 0.52)
        if "image_age_steps" in row:
            put(panel, f"Cached RGB age: {int(row['image_age_steps'])} control step(s)", (28, 122), (190, 195, 205), 0.52)
        put(panel, "LATEST NUMERIC TOKEN (physical values)", (28, 142), (80, 210, 255), 0.60, 2)
        y = 176
        for column, label, unit in DISPLAY_INPUTS:
            put(panel, f"{label:<24} {float(row[column]):+10.5f} {unit}", (38, y), (225, 225, 225), 0.55)
            y += 34
        put(panel, "MODEL OUTPUT vs GROUND TRUTH", (28, 374), (80, 210, 255), 0.64, 2)
        put(panel, "Orange: prediction", (38, 406), (0, 150, 255), 0.54, 2)
        put(panel, "Blue: ground truth", (280, 406), (255, 160, 40), 0.54, 2)
        y = 450
        for column, label, unit in DISPLAY_OUTPUTS:
            predicted = float(row[f"predicted_{column}"])
            actual = float(row[column])
            put(panel, f"{label} ({unit})", (38, y), (245, 245, 245), 0.58, 2)
            put(panel, f"Pred {predicted:+10.5f}", (245, y), (0, 150, 255), 0.56, 2)
            put(panel, f"True {actual:+10.5f}", (430, y), (255, 160, 40), 0.56, 2)
            put(panel, f"Absolute error: {abs(predicted - actual):.5f} {unit}", (58, y + 27), (165, 170, 180), 0.48)
            y += 78
        writer.write(np.hstack((image, panel)))
    writer.release()
    print(f"Saved video to: {output} ({fps:.2f} FPS)")


if __name__ == "__main__":
    main()

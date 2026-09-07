#!/usr/bin/env python3
"""Evaluate one or more checkpoints across the held-out runs, group by group.

(The file name is historical — it evaluates whichever manifest you point it at.)

``evaluate_experiment.py`` reports one experiment at a time, which is the right
granularity for debugging and the wrong one for the question this collection was
brought in to answer: *does the model still work on a different day, at a
different site, in the wet?*  This script runs every experiment in the manifest
for every checkpoint given, pools the predictions, and splits the result by
condition (wet / dry, taken from the experiment id).

Because wet and dry differ almost only in the visual channel — the 2026-08-19
target distributions match within ~4% — a gap between them is attributable to
vision, which makes it the natural test for an intervention that changed *where
the model looks*.

Example:
    python evaluate_newdata.py --config config_newdata.yaml \
        --checkpoints baseline=artifacts/runs/vit_motion_temporal_cr_v0_2_1/best.pt \
                      rrr_v3=artifacts/runs/vit_motion_rrr_v3/best_rrr.pt \
        --output-dir artifacts/newdata_eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vit_motion.config import load_config, resolve_from_config
from vit_motion.validation import (
    DISPLAY_OUTPUTS,
    load_normalization,
    predict_frame,
    regression_metrics,
)

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#6b7280"

#: One hue per model, assigned in a fixed order and never cycled.  With six or
#: seven arms, `BLUE if k == 0 else ORANGE` painted every fine-tune the same
#: colour and the legend became undecodable.  These seven are the validated
#: categorical order (worst adjacent CVD dE 9.1, normal-vision dE 19.6 on a light
#: surface); three of them sit below 3:1 contrast against white, so the figure
#: owes the reader relief — the group panel direct-labels every bar and
#: ``newdata_per_experiment.csv`` carries the same numbers as a table.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7")
#: The order arms are drawn in, so a standard run always gets the same colours.
#: Anything not listed keeps the order it arrived in, after these.
CANONICAL_ORDER = ("v021_old", "v03_base", "v1_tophalf", "sky_only",
                   "nonground", "guided", "control")


def series_colors(names: list[str]) -> dict[str, str]:
    """Map model name -> hue, consecutive slots so adjacent bars always separate.

    Slots are consecutive rather than pinned per name on purpose: the palette is
    validated on *adjacent* pairs, which is what a grouped bar chart shows, and
    skipping a slot could put two hues side by side that were never checked
    together.  A model past the seventh would be a new hue, so it raises instead:
    at that point the figure needs facets, not more colours.
    """
    if len(names) > len(SERIES_COLORS):
        raise SystemExit(
            f"{len(names)} models is more than the {len(SERIES_COLORS)} validated "
            "hues. Split the comparison into two figures rather than cycling colours.")
    return {name: SERIES_COLORS[i] for i, name in enumerate(names)}
HEADLINE = ["next_body_dx", "next_delta_yaw"]


def day_of(experiment_id: str) -> str:
    head = experiment_id.split("_", 1)[0]
    return head if head.isdigit() and len(head) == 8 else "unknown"


def condition_of(experiment_id: str) -> str:
    lowered = experiment_id.lower()
    if "wet" in lowered:
        return "wet"
    if "dry" in lowered:
        return "dry"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config_newdata.yaml")
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="One or more name=path pairs.")
    ap.add_argument("--normalizations", nargs="*", default=[],
                    help="Optional name=path overrides. A checkpoint trained with different "
                         "normalization statistics MUST be evaluated with its own, or its "
                         "predictions are silently rescaled and the comparison is void.")
    ap.add_argument("--split", default="test",
                    help="Manifest split to evaluate ('all' for every experiment).")
    ap.add_argument("--experiments", default="",
                    help="Comma list of experiment ids; overrides --split.")
    ap.add_argument("--output-dir", default="artifacts/newdata_eval")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-predictions", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    config_path = cfg["_config_path"]
    manifest_dir = resolve_from_config(cfg["data"]["manifest_dir"], config_path)
    out_dir = resolve_from_config(args.output_dir, config_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_dir / "manifest.csv")
    default_normalization = load_normalization(manifest_dir / "normalization.json")
    if args.experiments:
        experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    elif args.split != "all" and "split" in manifest.columns:
        experiments = sorted(
            manifest.loc[manifest["split"] == args.split, "experiment_id"].astype(str).unique())
        if not experiments:
            raise SystemExit(f"No experiments in split {args.split!r}.")
    else:
        experiments = sorted(manifest["experiment_id"].astype(str).unique())
    if "split_group" in manifest.columns:
        group_of = (manifest.drop_duplicates("experiment_id")
                    .set_index(manifest.drop_duplicates("experiment_id")["experiment_id"]
                               .astype(str))["split_group"].astype(str).to_dict())
    else:
        group_of = {}
    print(f"{len(experiments)} experiments:")
    for eid in experiments:
        print(f"   {eid:34s} group={group_of.get(eid, args.split)} "
              f"day={day_of(eid)} {condition_of(eid)}")

    models: dict[str, Path] = {}
    for item in args.checkpoints:
        if "=" not in item:
            raise SystemExit(f"--checkpoints expects name=path, got: {item}")
        name, path = item.split("=", 1)
        models[name] = resolve_from_config(path, config_path)

    normalizations: dict[str, dict] = {}
    for item in args.normalizations:
        if "=" not in item:
            raise SystemExit(f"--normalizations expects name=path, got: {item}")
        name, path = item.split("=", 1)
        if name not in models:
            raise SystemExit(f"--normalizations names {name!r}, which is not a checkpoint.")
        normalizations[name] = load_normalization(resolve_from_config(path, config_path))
        print(f"  {name}: using its own normalization from {path}")

    rows: list[dict] = []
    pooled: dict[str, list[pd.DataFrame]] = {name: [] for name in models}
    for name, checkpoint in models.items():
        print(f"\n=== {name}  ({checkpoint}) ===")
        for eid in experiments:
            frame = manifest[manifest["experiment_id"].astype(str) == eid].copy()
            result = predict_frame(frame, normalizations.get(name, default_normalization),
                                   cfg, checkpoint, args.batch_size, args.num_workers)
            result["experiment_id"] = eid
            pooled[name].append(result)
            if args.save_predictions:
                target = out_dir / name / eid
                target.mkdir(parents=True, exist_ok=True)
                result.to_csv(target / "predictions.csv", index=False)
            record = {"checkpoint": name, "experiment_id": eid,
                      "group": group_of.get(eid, args.split),
                      "day": day_of(eid),
                      "condition": condition_of(eid), "samples": int(len(result))}
            for column, _, _ in DISPLAY_OUTPUTS:
                scores = regression_metrics(result[column].to_numpy(float),
                                            result[f"predicted_{column}"].to_numpy(float))
                for key, value in scores.items():
                    record[f"{column}_{key}"] = value
            rows.append(record)
            print(f"  {eid:34s} dx R²={record['next_body_dx_r2']:.3f}  "
                  f"yaw R²={record['next_delta_yaw_r2']:.3f}  n={record['samples']}", flush=True)

    per_experiment = pd.DataFrame(rows)
    per_experiment.to_csv(out_dir / "per_experiment_metrics.csv", index=False)

    def pooled_metrics(frames: list[pd.DataFrame], key=None, value=None) -> dict:
        """Pool the predictions, optionally restricted to one slice of experiments."""
        merged = pd.concat(frames, ignore_index=True)
        if key is not None:
            selector = {"condition": condition_of, "day": day_of,
                        "group": lambda e: group_of.get(e, args.split)}[key]
            merged = merged[merged["experiment_id"].map(selector) == value]
        out = {"samples": int(len(merged))}
        if merged.empty:
            return out
        for column, _, _ in DISPLAY_OUTPUTS:
            out[column] = regression_metrics(merged[column].to_numpy(float),
                                             merged[f"predicted_{column}"].to_numpy(float))
        return out

    groups = sorted({group_of.get(e, args.split) for e in experiments})
    days = sorted({day_of(e) for e in experiments})

    summary = {
        "config": str(config_path),
        "manifest": str(manifest_dir / "manifest.csv"),
        "experiments": experiments,
        "checkpoints": {name: str(path) for name, path in models.items()},
        "normalization_overrides": {k: True for k in normalizations},
        "split": args.split,
        "groups": {g: sorted(e for e in experiments if group_of.get(e, args.split) == g)
                   for g in groups},
        "pooled": {name: pooled_metrics(frames) for name, frames in pooled.items()},
        "by_group": {
            name: {g: pooled_metrics(frames, "group", g) for g in groups}
            for name, frames in pooled.items()
        },
        "by_condition": {
            name: {condition: pooled_metrics(frames, "condition", condition)
                   for condition in ("dry", "wet")}
            for name, frames in pooled.items()
        },
        "by_day": {
            name: {d: pooled_metrics(frames, "day", d) for d in days}
            for name, frames in pooled.items()
        },
    }
    (out_dir / "newdata_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --------------------------------------------------------------- figure
    known = [n for n in CANONICAL_ORDER if n in models]
    names = known + [n for n in models if n not in known]
    palette = series_colors(names)
    fig, axes = plt.subplots(1, len(HEADLINE) + 1, figsize=(6.2 * (len(HEADLINE) + 1), 4.8),
                             constrained_layout=True)
    width = 0.8 / max(len(names), 1)
    for ax, column in zip(axes[:-1], HEADLINE):
        for k, name in enumerate(names):
            block = per_experiment[per_experiment["checkpoint"] == name]
            block = block.set_index("experiment_id").loc[experiments]
            positions = np.arange(len(experiments)) + k * width
            ax.bar(positions, block[f"{column}_r2"], width=width,
                   color=palette[name], linewidth=0.6,
                   edgecolor="white", label=name)
        ax.set_xticks(np.arange(len(experiments)) + width * (len(names) - 1) / 2)
        ax.set_xticklabels([e.replace("20260819_", "") for e in experiments],
                           rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("R²")
        ax.set_ylim(0, 1)
        ax.set_title(f"{column} — per held-out experiment")
        ax.legend(fontsize=9)

    ax = axes[-1]
    width = 0.8 / max(len(names), 1)
    for k, name in enumerate(names):
        values, labels = [], []
        for g in groups:
            entry = summary["by_group"][name][g]
            values.append(entry.get("next_delta_yaw", {}).get("r2") or 0.0)
            labels.append(g)
        positions = np.arange(len(groups)) + k * width
        ax.bar(positions, values, width=width, color=palette[name],
               linewidth=0.6, edgecolor="white", label=name)
        # Direct labels in ink, not in the series colour: with seven arms the
        # legend alone is a lot to hold, and three hues are under 3:1 on white.
        for x, v in zip(positions, values):
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                    rotation=90, color="#33322e")
    ax.set_xticks(np.arange(len(groups)) + width * (len(names) - 1) / 2)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("pooled yaw R²")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.set_title("By held-out group")
    fig.suptitle("Held-out accuracy: per experiment, and per held-out group", fontsize=14)
    fig.savefig(out_dir / "f_newdata_accuracy.png", dpi=180)
    plt.close(fig)

    def line(label, entry, indent=0):
        if not entry.get("samples"):
            return
        pad = " " * indent
        print(f"{pad}{label}: n={entry['samples']}  "
              f"dx R²={entry['next_body_dx']['r2']:.3f}  "
              f"yaw R²={entry['next_delta_yaw']['r2']:.3f}")

    print("\n=== POOLED ===")
    for name in names:
        line(name, summary["pooled"][name])
        for g in groups:
            line(g, summary["by_group"][name][g], indent=4)
        for condition in ("dry", "wet"):
            line(condition, summary["by_condition"][name][condition], indent=8)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()

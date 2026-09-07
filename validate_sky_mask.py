#!/usr/bin/env python3
"""Audit the region masks used by the interpretability pipeline, against depth.

Stage 2/3 of this project measure and then penalize "saliency that falls on the
sky".  In v2 the sky was *defined* as the top half of the frame.  This script
replaces that definition with a per-frame mask and checks the new mask against
an independent, non-learned reference that the 2026-08-19 collection provides
for free: the aligned 16-bit depth frame.

What the depth frame can and cannot say
---------------------------------------
It cannot separate sky from a distant apartment tower — both return nothing.
What it does give is a hard geometric fact: which pixels are *beyond usable
range* (invalid or > ``--far-m``) and therefore cannot carry a metric motion
cue, and which are *near* terrain (< ``--near-m``) that can.  So:

  * real sky must be a SUBSET of the depth "far" region -> containment is a
    genuine falsifiable test of the RGB sky mask;
  * the remainder of the far region is distant structure (towers, tents, tree
    line) — visible, but just as useless for egomotion as the sky;
  * the near region is the geometric "right reason" for a ground vehicle.

Outputs (in ``--output-dir``)
    mask_validation.json            pooled + per-experiment numbers
    mask_validation_per_frame.csv   one row per audited frame
    f_mask_qualitative.png          RGB | v2 top-half | RGB mask | depth bands | agreement
    f_mask_audit.png                four quantitative panels

Example:
    python validate_sky_mask.py --data-root new_data/processed \
        --num-per-exp 40 --output-dir new_data/analysis/mask_validation
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
from PIL import Image

from vit_motion.sky_mask import (
    GROUND_LABELS,
    build_masker,
    depth_regions,
    half_frame_mask,
    load_depth_png,
    mask_agreement,
    resize_mask,
)

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#2f9e63", "#6b7280"


def collect_rows(root: Path, num_per_exp: int) -> list[dict]:
    """Sample rows that have both an RGB and a depth file present on disk."""
    rows: list[dict] = []
    for csv_path in sorted(root.rglob("samples_full.csv")):
        exp_dir = csv_path.parent
        frame = pd.read_csv(csv_path)
        if "depth_path" not in frame.columns:
            print(f"  skip {exp_dir.name}: no depth_path column")
            continue
        usable = [
            (exp_dir / str(r["rgb_path"]), exp_dir / str(r["depth_path"]))
            for _, r in frame.iterrows()
        ]
        usable = [(a, b) for a, b in usable if a.is_file() and b.is_file()]
        if not usable:
            print(f"  skip {exp_dir.name}: no rgb/depth pairs on disk")
            continue
        take = sorted(set(np.linspace(0, len(usable) - 1,
                                      min(num_per_exp, len(usable)), dtype=int).tolist()))
        for i in take:
            rgb, depth = usable[i]
            rows.append({"experiment_id": exp_dir.name, "rgb": rgb, "depth": depth})
        print(f"  {exp_dir.name}: {len(take)} of {len(usable)} pairs", flush=True)
    return rows


def frac(part: np.ndarray, whole: np.ndarray) -> float:
    denominator = float(whole.sum())
    return float((part & whole).sum()) / denominator if denominator else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-dir", default="analysis/mask_validation")
    ap.add_argument("--num-per-exp", type=int, default=40)
    ap.add_argument("--masker", default="auto", choices=["auto", "segmentation", "energy"])
    ap.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    ap.add_argument("--near-m", type=float, default=8.0)
    ap.add_argument("--far-m", type=float, default=20.0)
    ap.add_argument("--horizon-frac", type=float, default=0.5,
                    help="The v2 heuristic being audited.")
    ap.add_argument("--eval-size", type=int, nargs=2, default=(224, 224))
    ap.add_argument("--qualitative", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--figure-dpi", type=int, default=120)
    ap.add_argument("--extra-ground-labels", default="",
                    help="Comma list of extra class names to count as ground. Pass the SAME "
                         "value used for precompute_sky_masks.py, or the audit describes a "
                         "different mask from the one the model is trained against.")
    args = ap.parse_args()

    extra = {s.strip().lower() for s in args.extra_ground_labels.split(",") if s.strip()}
    ground_labels = frozenset(GROUND_LABELS | extra)
    if extra:
        print(f"      ground list extended with: {', '.join(sorted(extra))}")

    root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_size = (int(args.eval_size[0]), int(args.eval_size[1]))

    print(f"[1/4] Collecting frames under {root}")
    rows = collect_rows(root, args.num_per_exp)
    if not rows:
        raise SystemExit("No rgb/depth pairs found.")
    print(f"      {len(rows)} frames / {len({r['experiment_id'] for r in rows})} experiments")

    print(f"[2/4] Building the '{args.masker}' masker")
    masker = build_masker(args.masker, model_name=args.model,
                          batch_size=args.batch_size, ground_labels=ground_labels)
    masker_name = type(masker).__name__
    print(f"      using {masker_name}")

    print("[3/4] Auditing")
    records: list[dict] = []
    examples: list[dict] = []
    example_at = set(np.linspace(0, len(rows) - 1, args.qualitative, dtype=int).tolist())

    for i, row in enumerate(rows):
        with Image.open(row["rgb"]) as img:
            image = img.convert("RGB")
        native = (image.height, image.width)

        rgbm = masker(image, size=native)               # sky / ground / other from RGB
        dep = depth_regions(load_depth_png(row["depth"]),
                            near_m=args.near_m, far_m=args.far_m)
        half = half_frame_mask(*native, horizon_frac=args.horizon_frac).sky

        sky = rgbm.sky
        rec = {
            "experiment_id": row["experiment_id"],
            "rgb": row["rgb"].name,
            "condition": "wet" if "wet" in row["experiment_id"] else "dry",
            # --- what the frame actually contains -------------------------- #
            "sky_area": float(sky.mean()),
            "depth_near_area": float(dep.near.mean()),
            "depth_mid_area": float(dep.mid.mean()),
            "depth_far_area": float(dep.far.mean()),
            "depth_invalid_frac": dep.meta["invalid_frac"],
            "depth_saturated_frac": dep.meta["saturated_frac"],
            # --- is the RGB sky mask trustworthy? -------------------------- #
            "sky_containment_in_far": frac(dep.far, sky),      # should be ~1
            "sky_share_of_far": frac(sky, dep.far),            # rest = distant structure
            "sky_leak_into_near": frac(dep.near, sky),         # should be ~0
            # --- how wrong was the v2 top-half heuristic? ------------------ #
            "half_is_sky": frac(sky, half),                    # precision of the old mask
            "half_is_near_ground": frac(dep.near, half),       # penalized real terrain
            "half_is_far": frac(dep.far, half),
            "half_recall_of_sky": frac(half, sky),
            "half_iou_sky": mask_agreement(half, sky)["iou"],
            "sky_below_midline": frac(~half, sky),
        }
        # at the geometry the model and the penalty actually operate on
        rec["sky_area_at_model_size"] = float(resize_mask(sky, eval_size).mean())
        records.append(rec)

        if i in example_at:
            examples.append({
                "rgb": np.asarray(image), "sky": sky, "ground": rgbm.ground,
                "near": dep.near, "mid": dep.mid, "far": dep.far, "half": half,
                "exp": row["experiment_id"], "rec": rec,
            })
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"      {i + 1}/{len(rows)}", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "mask_validation_per_frame.csv", index=False)

    def stat(col: str) -> dict[str, float]:
        values = df[col].to_numpy(dtype=float)
        return {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values)),
                "p05": float(np.nanpercentile(values, 5)),
                "p95": float(np.nanpercentile(values, 95))}

    summary = {
        "data_root": str(root),
        "masker": masker_name,
        "model": args.model if masker_name == "SegformerSkyMasker" else None,
        "near_m": args.near_m, "far_m": args.far_m, "horizon_frac": args.horizon_frac,
        "n_frames": int(len(df)),
        "experiments": sorted(df["experiment_id"].unique().tolist()),
        "frame_composition": {k: stat(k) for k in
                              ["sky_area", "depth_near_area", "depth_mid_area", "depth_far_area"]},
        "mask_validation": {k: stat(k) for k in
                            ["sky_containment_in_far", "sky_share_of_far", "sky_leak_into_near"]},
        "v2_heuristic_audit": {k: stat(k) for k in
                               ["half_is_sky", "half_is_near_ground", "half_is_far",
                                "half_recall_of_sky", "half_iou_sky", "sky_below_midline"]},
        "frames_without_sky": int((df["sky_area"] < 0.005).sum()),
        "per_experiment": df.groupby("experiment_id")[
            ["sky_area", "depth_far_area", "depth_near_area",
             "sky_containment_in_far", "half_is_sky"]].mean().round(4).to_dict(orient="index"),
        "wet_vs_dry": df.groupby("condition")[
            ["sky_area", "depth_near_area", "sky_containment_in_far"]].mean().round(4).to_dict(orient="index"),
    }
    (out_dir / "mask_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- figure 1
    n = len(examples)
    fig, axes = plt.subplots(n, 5, figsize=(19.5, 3.05 * n), constrained_layout=True)
    if n == 1:
        axes = axes[None, :]
    for r, ex in enumerate(examples):
        rgb = ex["rgb"]
        shape = ex["sky"].shape

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_ylabel(ex["exp"].replace("20260819_", ""), fontsize=8)
        if r == 0:
            axes[r, 0].set_title("RGB frame", fontsize=12)

        ov = np.zeros((*shape, 4)); ov[ex["half"]] = [0.92, 0.41, 0.20, 0.45]
        axes[r, 1].imshow(rgb); axes[r, 1].imshow(ov)
        axes[r, 1].set_xlabel(f"only {ex['rec']['half_is_sky']:.0%} of it is sky", fontsize=9)
        if r == 0:
            axes[r, 1].set_title("v2: “sky” = top half", fontsize=12, color=ORANGE)

        ov = np.zeros((*shape, 4))
        ov[ex["ground"]] = [0.18, 0.62, 0.39, 0.38]
        ov[ex["sky"]] = [0.16, 0.47, 0.84, 0.60]
        axes[r, 2].imshow(rgb); axes[r, 2].imshow(ov)
        axes[r, 2].set_xlabel(f"sky = {ex['rec']['sky_area']:.0%} of the frame", fontsize=9)
        if r == 0:
            axes[r, 2].set_title("v3: per-frame sky mask", fontsize=12, color=BLUE)

        ov = np.zeros((*shape, 4))
        ov[ex["near"]] = [0.18, 0.62, 0.39, 0.55]
        ov[ex["mid"]] = [0.95, 0.75, 0.20, 0.45]
        ov[ex["far"]] = [0.45, 0.45, 0.48, 0.55]
        axes[r, 3].imshow(rgb); axes[r, 3].imshow(ov)
        axes[r, 3].set_xlabel(f"near {ex['rec']['depth_near_area']:.0%} · "
                              f"far {ex['rec']['depth_far_area']:.0%}", fontsize=9)
        if r == 0:
            axes[r, 3].set_title("depth bands: near / mid / beyond range", fontsize=12)

        ov = np.zeros((*shape, 4))
        ov[ex["far"] & ~ex["sky"]] = [0.55, 0.35, 0.75, 0.60]     # distant structure
        ov[ex["sky"]] = [0.16, 0.47, 0.84, 0.65]
        ov[ex["sky"] & ~ex["far"]] = [0.90, 0.10, 0.10, 0.90]      # would falsify the mask
        axes[r, 4].imshow(rgb); axes[r, 4].imshow(ov)
        axes[r, 4].set_xlabel(f"{ex['rec']['sky_containment_in_far']:.1%} of the sky mask "
                              f"is confirmed beyond range", fontsize=9)
        if r == 0:
            axes[r, 4].set_title("check: blue sky ⊂ far · purple = distant structure",
                                 fontsize=11)
        for c in range(5):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle("What the model's “sky” penalty was actually pointing at, and what it points at now",
                 fontsize=14)
    fig.savefig(out_dir / "f_mask_qualitative.png", dpi=args.figure_dpi)
    plt.close(fig)

    # ------------------------------------------------------------- figure 2
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), constrained_layout=True)

    axes[0].hist(df["sky_area"], bins=24, range=(0, 1), color=BLUE, alpha=0.9, edgecolor="white")
    axes[0].axvline(args.horizon_frac, color=ORANGE, ls="--", lw=2,
                    label=f"v2 assumed {args.horizon_frac:.0%}")
    axes[0].axvline(df["sky_area"].mean(), color=BLUE, lw=2.5,
                    label=f"measured {df['sky_area'].mean():.1%}")
    axes[0].set_xlabel("sky share of the frame"); axes[0].set_ylabel("frames")
    axes[0].set_title("The sky is not half the frame"); axes[0].legend(fontsize=9)

    axes[1].hist(df["sky_containment_in_far"], bins=24, range=(0, 1), color=GREEN,
                 alpha=0.9, edgecolor="white")
    axes[1].axvline(df["sky_containment_in_far"].mean(), color=GREY, lw=2.5,
                    label=f"mean {df['sky_containment_in_far'].mean():.1%}")
    axes[1].set_xlabel("share of the sky mask that depth confirms is beyond range")
    axes[1].set_ylabel("frames")
    axes[1].set_title("Depth validates the RGB sky mask"); axes[1].legend(fontsize=9)

    parts = [df["half_is_sky"].mean(),
             (df["half_is_far"] - df["half_is_sky"]).clip(lower=0).mean(),
             df["half_is_near_ground"].mean()]
    parts.append(max(0.0, 1.0 - sum(parts)))
    labels = ["actual sky", "distant structure", "near ground", "mid-range"]
    colors = [BLUE, "#8b5cf6", GREEN, "#f5c142"]
    left = 0.0
    for value, label, color in zip(parts, labels, colors):
        axes[2].barh([0], [value], left=left, color=color, edgecolor="white", label=f"{label} {value:.0%}")
        left += value
    axes[2].set_yticks([]); axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("composition of the region the v2 penalty acted on")
    axes[2].set_title("What the old “sky” penalty was really penalizing")
    axes[2].legend(fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.42))

    order = sorted(df["experiment_id"].unique())
    data = [df.loc[df["experiment_id"] == e, "sky_area"].to_numpy() for e in order]
    bp = axes[3].boxplot(data, patch_artist=True, widths=0.6)
    for patch, name in zip(bp["boxes"], order):
        patch.set_facecolor(ORANGE if "wet" in name else BLUE); patch.set_alpha(0.8)
    for med in bp["medians"]:
        med.set_color("white"); med.set_linewidth(2)
    axes[3].axhline(args.horizon_frac, color=GREY, ls="--", lw=1.5)
    axes[3].set_xticks(range(1, len(order) + 1))
    axes[3].set_xticklabels([e.replace("20260819_", "") for e in order],
                            rotation=20, ha="right", fontsize=8)
    axes[3].set_ylabel("sky share of the frame"); axes[3].set_ylim(0, 1)
    axes[3].set_title("Per experiment (orange = wet)")
    fig.savefig(out_dir / "f_mask_audit.png", dpi=max(args.figure_dpi, 150))
    plt.close(fig)

    print("[4/4] Done")
    print(json.dumps({k: summary[k] for k in
                      ["masker", "n_frames", "frame_composition", "mask_validation",
                       "v2_heuristic_audit", "frames_without_sky"]}, indent=2))
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()

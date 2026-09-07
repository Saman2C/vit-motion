#!/usr/bin/env python3
"""Quantify where the visual encoder looks — and how spread out it is there.

Three numbers per checkpoint, all computed on the same input-gradient saliency
the fine-tune optimizes, against the same per-frame region masks:

    distractor_fraction   share of the saliency mass on sky + trees + buildings
                          + cars.  The thing the placement penalty minimizes.
    ground_concentration  ground saliency share / ground area share.  1.0 means
                          "attends to the terrain exactly in proportion to how
                          much of the frame it is"; above 1 means it prefers it.
    ground_coverage       exp(H)/N over the ground pixels — the effective share
                          of the terrain the attention actually covers.  This is
                          the one v2 had no answer for: it can report 0.04 (four
                          hot spots) while distractor_fraction reads a perfect
                          0.00.

The v2 top-half number is still computed alongside, so a v3 run can be lined up
against the old slide without re-deriving anything.

Example:
    python interpret_quantify.py --config config_run.yaml \
        --checkpoint artifacts/runs/v03_guided/best.pt \
        --sky-mask-dir artifacts/sky_masks \
        --experiments auto --num-per-exp 40 --target yaw --tag guided
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
import torch

from vit_motion.attention_focus import ground_focus_report
from vit_motion.config import load_config, resolve_from_config
from vit_motion.interpret import MotionInterpreter
from vit_motion.sky_mask import (
    build_masker,
    cache_provides_ground,
    half_frame_mask,
    load_mask_quality,
    mask_cache_path,
    read_mask_png,
)
from vit_motion.validation import ExperimentDataset, load_model, load_normalization

BLUE, ORANGE, GREEN, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#6b7280"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--experiments", default="auto",
                   help="'auto' = first --max-exp test experiments, or a comma list of ids.")
    p.add_argument("--max-exp", type=int, default=6)
    p.add_argument("--num-per-exp", type=int, default=40)
    p.add_argument("--target", default="yaw", choices=["dx", "dy", "yaw", "norm"])
    p.add_argument("--saliency", default="inputgrad", choices=["gradcam", "inputgrad"],
                   help="inputgrad = the quantity the guidance optimizes (recommended).")
    p.add_argument("--smooth-sigma", type=float, default=3.0)
    p.add_argument("--sky-mask-dir", default="artifacts/sky_masks")
    p.add_argument("--mask-on-the-fly", action="store_true")
    p.add_argument("--masker", default="auto", choices=["auto", "segmentation", "energy"])
    p.add_argument("--horizon-frac", type=float, default=0.5,
                   help="Legacy heuristic, reported alongside for comparability.")
    p.add_argument("--output-dir", default="artifacts/quantify")
    p.add_argument("--tag", default="baseline")
    p.add_argument("--include-untrusted", action="store_true",
                   help="Score frames whose mask the cache marks as a segmentation failure. "
                        "Off by default: they distort every region number.")
    args = p.parse_args()

    cfg = load_config(args.config)
    config_path = cfg["_config_path"]
    manifest_dir = resolve_from_config(cfg["data"]["manifest_dir"], config_path)
    out_dir = resolve_from_config(args.output_dir, config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = resolve_from_config(args.sky_mask_dir, config_path) if args.sky_mask_dir else None

    live_masker = build_masker(args.masker) if args.mask_on_the_fly else None
    if live_masker is not None:
        has_ground = getattr(live_masker, "provides_ground", None)
        has_ground = True if has_ground is None else bool(has_ground)
        has_ground = type(live_masker).__name__ == "SegformerSkyMasker"
    else:
        has_ground = cache_provides_ground(mask_dir) if mask_dir else False
    if not has_ground:
        print("NOTE: the masks have no ground channel, so ground_* and distractor_* "
              "columns will be empty. Only the sky and legacy numbers are meaningful.")

    manifest = pd.read_csv(manifest_dir / "manifest.csv")
    if args.experiments == "auto":
        pool = manifest[manifest["split"] == "test"] if (manifest["split"] == "test").any() else manifest
        exps = sorted(pool["experiment_id"].astype(str).unique())[: args.max_exp]
    else:
        exps = [e.strip() for e in args.experiments.split(",") if e.strip()]

    normalization = load_normalization(manifest_dir / "normalization.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, cfg, device)
    interp = MotionInterpreter(model, device)
    image_size = tuple(int(x) for x in cfg["data"]["image_size"])
    sample_period = float(cfg["data"].get("sample_period_sec", 0.1))
    half = half_frame_mask(*image_size, horizon_frac=args.horizon_frac).sky

    quality = load_mask_quality(mask_dir) if (mask_dir and not args.mask_on_the_fly) else {}
    if quality:
        n_bad = sum(1 for v in quality.values() if not v)
        print(f"mask cache: {n_bad}/{len(quality)} frames marked untrusted"
              f"{' (scored anyway)' if args.include_untrusted else ' (skipped)'}")

    rows: list[dict] = []
    missing_masks = untrusted_skipped = 0
    for eid in exps:
        frame = manifest[manifest["experiment_id"].astype(str) == eid].copy()
        if frame.empty:
            continue
        ds = ExperimentDataset(frame, normalization, image_size,
                               model.sequence_length, model.image_update_interval, sample_period)
        n = min(args.num_per_exp, len(ds))
        for i in np.linspace(0, len(ds) - 1, n, dtype=int):
            item = ds[int(i)]
            saliency, _ = interp.saliency_map(
                item["image"], item["numeric"], item["image_age"],
                target=args.target, method=args.saliency, smooth_sigma=args.smooth_sigma,
            )
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
                missing_masks += 1
                continue
            if (quality and not args.include_untrusted
                    and not quality.get((eid, Path(rgb_path).stem), True)):
                untrusted_skipped += 1
                continue

            row = {"experiment_id": eid, "rgb": Path(rgb_path).name}
            row.update(ground_focus_report(saliency, regions.sky, regions.ground, regions.other))
            legacy = ground_focus_report(saliency, half, np.zeros_like(half), ~half)
            row["legacy_half_fraction"] = legacy["sky_fraction"]
            row["legacy_half_concentration"] = legacy["sky_concentration"]
            rows.append(row)

        done = [r for r in rows if r["experiment_id"] == eid]
        if done:
            def m(key):
                values = np.asarray([r[key] for r in done], dtype=float)
                return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            print(f"  {eid}: distractor {m('distractor_fraction'):.3f} · "
                  f"ground {m('ground_concentration'):.2f}x · "
                  f"coverage {m('ground_coverage'):.3f}  (n={len(done)})", flush=True)

    if not rows:
        raise SystemExit("No samples scored — is the mask cache present? "
                         "Run precompute_sky_masks.py first, or pass --mask-on-the-fly.")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"focus_{args.tag}_per_sample.csv", index=False)

    def stat(column: str) -> dict[str, float | None]:
        values = df[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(finite.mean()), "std": float(finite.std()), "n": int(finite.size)}

    summary = {
        "tag": args.tag,
        "checkpoint": str(args.checkpoint),
        "target": args.target,
        "saliency": args.saliency,
        "mask_source": (f"on-the-fly:{type(live_masker).__name__}" if live_masker else str(mask_dir)),
        "provides_ground": bool(has_ground),
        "experiments": exps,
        "n_samples": int(len(df)),
        "n_missing_masks": int(missing_masks),
        "n_untrusted_skipped": int(untrusted_skipped),
        "headline": {
            "distractor_fraction": stat("distractor_fraction"),
            "ground_concentration": stat("ground_concentration"),
            "ground_coverage": stat("ground_coverage"),
        },
        "placement": {
            name: {"fraction": stat(f"{name}_fraction"),
                   "area": stat(f"{name}_area"),
                   "concentration": stat(f"{name}_concentration")}
            for name in ("sky", "ground", "other", "distractor")
        },
        "spread_on_ground": {
            "coverage": stat("ground_coverage"),
            "normalized_entropy": stat("ground_normalized_entropy"),
            "effective_pixels": stat("ground_effective_pixels"),
            "top_decile_share": stat("ground_top_decile_share"),
        },
        "legacy_top_half": {"fraction": stat("legacy_half_fraction"),
                            "concentration": stat("legacy_half_concentration"),
                            "horizon_frac": args.horizon_frac},
        "per_experiment": df.groupby("experiment_id")[
            ["distractor_fraction", "ground_concentration", "ground_coverage"]
        ].mean().round(4).to_dict(orient="index"),
    }
    (out_dir / f"focus_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    names = ["sky", "other", "ground"]
    colors = [BLUE, VIOLET, GREEN]
    means = [float(np.nanmean(df[f"{k}_concentration"])) if df[f"{k}_concentration"].notna().any()
             else np.nan for k in names]
    axes[0].bar(range(len(names)), means, color=colors, edgecolor="white")
    axes[0].axhline(1.0, color=GREY, ls="--", lw=2)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(["sky", "structure\n(trees, buildings)", "ground\n(the useful part)"])
    axes[0].set_ylabel("saliency concentration (share / area)")
    for i, value in enumerate(means):
        if np.isfinite(value):
            axes[0].text(i, value, f"{value:.2f}×", ha="center", va="bottom", fontsize=11)
    axes[0].set_title("Placement — where the attention goes")

    values = df["distractor_fraction"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size:
        axes[1].hist(values, bins=24, range=(0, 1), color=VIOLET, alpha=0.9, edgecolor="white")
        axes[1].axvline(values.mean(), color=GREY, lw=2.5, label=f"mean {values.mean():.3f}")
        axes[1].legend(fontsize=9)
    axes[1].set_xlabel("share of saliency on sky + structure")
    axes[1].set_ylabel("samples")
    axes[1].set_title("The quantity the penalty minimizes")

    values = df["ground_coverage"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size:
        axes[2].hist(values, bins=24, range=(0, 1), color=GREEN, alpha=0.9, edgecolor="white")
        axes[2].axvline(values.mean(), color=GREY, lw=2.5, label=f"mean {values.mean():.3f}")
        axes[2].legend(fontsize=9)
    axes[2].set_xlabel("effective share of the ground covered  (exp(H)/N)")
    axes[2].set_ylabel("samples")
    axes[2].set_title("Spread — is it using all of the terrain?")

    fig.suptitle(f"Attention focus — {args.tag} · target={args.target} · "
                 f"{args.saliency} · n={len(df)}", fontsize=13)
    fig.savefig(out_dir / f"focus_{args.tag}.png", dpi=180)
    plt.close(fig)

    print("\n=== ATTENTION FOCUS ===")
    print(json.dumps({k: summary[k] for k in
                      ("tag", "n_samples", "provides_ground", "headline",
                       "spread_on_ground", "legacy_top_half")}, indent=2))
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()

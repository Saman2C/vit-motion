#!/usr/bin/env python3
"""Score every experiment on image quality, so a NAS full of candidates can be triaged.

Run 1 showed the cost of skipping this: on part of the collection the frames are
washed out and colour-cast, the segmenter reads the pale terrain as anything but
ground (4-8% ground where half the frame is grass), and the attention penalty then
pushes saliency out of 96% of the image.  Bad frames do not just add noise — they
invert the training signal.

Six measurements per frame, all cheap and all interpretable:

    luminance        mean L*; very high = washed out, very low = underexposed
    contrast         std of L*; the single best predictor of a segmenter failure
    saturation       mean S in HSV; near-zero means colour has been washed out
    colour_cast      max |channel mean - grey mean| / grey mean; a magenta or
                     green cast the white balance never corrected
    clipped          share of pixels at 0 or 255 in any channel
    sharpness        variance of the Laplacian; low = motion blur or defocus

Flagging happens two ways, and the defaults are deliberately timid because a
guessed threshold throws away good data:

* **absolute** — only genuinely broken frames (a first pass at these numbers was
  calibrated by eye and flagged a third of a collection that turned out to be
  fine: outdoor scenes clip their bright sky, and clipping says nothing about
  whether the terrain is readable, so it is not a rule any more);
* **relative** — an experiment more than ``--relative-z`` robust deviations below
  its own collection's median contrast or saturation, which is what actually
  separated the failing runs in practice.

    python screen_experiments.py --data-root data/processed --num-per-exp 24 \
        --output-dir analysis/screening

The thresholds are only worth what the evidence behind them is worth, so pass
``--mask-quality artifacts/masks/quality.csv`` from a run that already built
masks: the screen is then scored against the mask failures that really happened,
and it prints the contrast threshold that best separates them.
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

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6b7280"

# Severe failures only. A healthy outdoor collection measured here sits at
# contrast 28-79, luminance 93-134, saturation 36-93, cast 0.01-0.12 — so these
# cut well below all of it and catch only data that is actually broken.
DEFAULT_RULES = {
    "contrast_min": 22.0,
    "luminance_max": 215.0,
    "luminance_min": 35.0,
    "saturation_min": 15.0,
    "colour_cast_max": 0.22,
    "sharpness_min": 20.0,
}


def robust_z(values: np.ndarray) -> np.ndarray:
    """Deviations below the median, scaled by the MAD (outlier-proof)."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    if mad < 1e-9:
        return np.zeros_like(values)
    return (values - median) / mad


def frame_stats(path: Path) -> dict[str, float]:
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"))
    values = rgb.astype(np.float32)
    grey = values @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    channel_means = values.reshape(-1, 3).mean(axis=0)
    grey_mean = float(channel_means.mean()) + 1e-6

    maximum = values.max(axis=2)
    minimum = values.min(axis=2)
    saturation = np.where(maximum > 0, (maximum - minimum) / np.maximum(maximum, 1e-6) * 255.0, 0.0)

    # 4-neighbour Laplacian, no cv2 dependency
    lap = (grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
           - 4.0 * grey[1:-1, 1:-1])
    return {
        "luminance": float(grey.mean()),
        "contrast": float(grey.std()),
        "saturation": float(saturation.mean()),
        "colour_cast": float(np.abs(channel_means - grey_mean).max() / grey_mean),
        "clipped": float(((rgb == 0) | (rgb == 255)).any(axis=2).mean()),
        "sharpness": float(lap.var()),
    }


def verdict(row: pd.Series, rules: dict[str, float]) -> str:
    reasons = []
    if row["contrast"] < rules["contrast_min"]:
        reasons.append("low-contrast")
    if row["luminance"] > rules["luminance_max"]:
        reasons.append("washed-out")
    if row["luminance"] < rules["luminance_min"]:
        reasons.append("underexposed")
    if row["saturation"] < rules["saturation_min"]:
        reasons.append("desaturated")
    if row["colour_cast"] > rules["colour_cast_max"]:
        reasons.append("colour-cast")
    if row["sharpness"] < rules["sharpness_min"]:
        reasons.append("blurred")
    return ",".join(reasons)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", nargs="+", required=True)
    ap.add_argument("--output-dir", default="analysis/screening")
    ap.add_argument("--num-per-exp", type=int, default=24)
    ap.add_argument("--mask-quality", help="quality.csv from a mask cache, to validate the rules.")
    ap.add_argument("--examples", type=int, default=4, help="Worst/best frames to render.")
    ap.add_argument("--relative-z", type=float, default=3.0,
                    help="Also flag an experiment this many robust deviations below its own "
                         "collection's median contrast or saturation. 0 disables it.")
    for key, value in DEFAULT_RULES.items():
        ap.add_argument(f"--{key.replace('_', '-')}", type=float, default=value)
    args = ap.parse_args()

    rules = {key: getattr(args, key) for key in DEFAULT_RULES}
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = []
    for root in args.data_root:
        experiments += sorted(p.parent for p in Path(root).expanduser().resolve()
                              .glob("*/samples.csv"))
    if not experiments:
        raise SystemExit("No experiments found.")
    print(f"screening {len(experiments)} experiments, {args.num_per_exp} frames each")

    rows, frame_rows = [], []
    for index, exp in enumerate(experiments, 1):
        frames = sorted((exp / "rgb").glob("*.png"))
        if not frames:
            print(f"  {exp.name}: no rgb/ — skipped")
            continue
        picks = np.linspace(0, len(frames) - 1,
                            min(args.num_per_exp, len(frames))).round().astype(int)
        stats = []
        for i in sorted(set(picks.tolist())):
            s = frame_stats(frames[i])
            s["experiment_id"] = exp.name
            s["path"] = str(frames[i])
            stats.append(s)
            frame_rows.append(s)
        block = pd.DataFrame(stats)
        row = {"experiment_id": exp.name, "n_frames_total": len(frames),
               "n_screened": len(block)}
        for key in ("luminance", "contrast", "saturation", "colour_cast", "clipped", "sharpness"):
            row[key] = float(block[key].median())
        rows.append(row)
        if index % 10 == 0 or index == len(experiments):
            print(f"  {index}/{len(experiments)}", flush=True)

    df = pd.DataFrame(rows)
    df["flags"] = df.apply(lambda r: verdict(r, rules), axis=1)
    if args.relative_z > 0 and len(df) >= 8:
        for key, label in (("contrast", "contrast-outlier"), ("saturation", "saturation-outlier")):
            z = robust_z(df[key].to_numpy(float))
            outlier = z < -args.relative_z
            df.loc[outlier, "flags"] = (df.loc[outlier, "flags"] + "," + label).str.strip(",")
            df[f"{key}_z"] = z.round(2)
    df["usable"] = df["flags"] == ""
    df = df.sort_values(["usable", "contrast"]).reset_index(drop=True)
    df.to_csv(out_dir / "experiment_screening.csv", index=False)
    pd.DataFrame(frame_rows).to_csv(out_dir / "frame_screening.csv", index=False)

    print("\ncollection profile (median across experiments):")
    print(df[["luminance", "contrast", "saturation", "colour_cast", "clipped",
              "sharpness"]].describe().loc[["min", "50%", "max"]].round(2).to_string())
    print(f"\n{int(df['usable'].sum())}/{len(df)} experiments pass")
    if (~df["usable"]).any():
        print("\nflagged:")
        print(df.loc[~df["usable"],
                     ["experiment_id", "contrast", "luminance", "saturation",
                      "colour_cast", "sharpness", "flags"]].round(2).to_string(index=False))

    summary = {
        "rules": rules,
        "n_experiments": int(len(df)),
        "n_usable": int(df["usable"].sum()),
        "medians": {k: float(df[k].median()) for k in
                    ("luminance", "contrast", "saturation", "colour_cast", "clipped", "sharpness")},
        "flag_counts": (df.loc[~df["usable"], "flags"].str.split(",").explode()
                        .value_counts().to_dict() if (~df["usable"]).any() else {}),
        "usable_experiments": df.loc[df["usable"], "experiment_id"].tolist(),
        "flagged_experiments": df.loc[~df["usable"], "experiment_id"].tolist(),
    }

    # ---- does the screen actually predict the mask failures we saw? ------------
    if args.mask_quality and Path(args.mask_quality).is_file():
        quality = pd.read_csv(args.mask_quality)
        trust = quality.groupby("experiment_id")["trusted"].mean().rename("trusted_rate")
        merged = df.merge(trust, on="experiment_id", how="inner")
        if not merged.empty:
            by_verdict = merged.groupby("usable")["trusted_rate"].agg(["size", "mean"]).round(3)
            correlation = float(merged["contrast"].corr(merged["trusted_rate"]))
            summary["validation"] = {
                "n_matched": int(len(merged)),
                "trusted_rate_by_verdict": by_verdict.to_dict(orient="index"),
                "corr_contrast_vs_trusted": correlation,
            }
            print("\nagainst the mask cache's own trusted flag:")
            print(by_verdict.to_string())
            print(f"corr(contrast, trusted rate) = {correlation:+.2f}")

            # The threshold that best separates the experiments whose masks held up
            # from the ones that did not — evidence instead of a guessed number.
            good = merged["trusted_rate"] >= 0.9
            if good.any() and (~good).any():
                candidates = np.linspace(merged["contrast"].min(), merged["contrast"].max(), 200)
                scores = [((merged["contrast"] >= t) == good).mean() for t in candidates]
                best = float(candidates[int(np.argmax(scores))])
                summary["validation"]["suggested_contrast_min"] = round(best, 1)
                summary["validation"]["separation_accuracy"] = round(float(max(scores)), 3)
                print(f"suggested --contrast-min {best:.0f} "
                      f"(separates {max(scores):.0%} of experiments correctly)")

    (out_dir / "experiment_screening.json").write_text(json.dumps(summary, indent=2),
                                                       encoding="utf-8")

    # ------------------------------------------------------------------ figures
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), constrained_layout=True)
    colours = np.where(df["usable"], GREEN, ORANGE)
    axes[0].scatter(df["contrast"], df["colour_cast"], c=colours, s=42, edgecolor="white")
    axes[0].axvline(rules["contrast_min"], color=GREY, ls="--", lw=1.5)
    axes[0].axhline(rules["colour_cast_max"], color=GREY, ls="--", lw=1.5)
    axes[0].set_xlabel("contrast  (std of L*)")
    axes[0].set_ylabel("colour cast  (max channel imbalance)")
    axes[0].set_title("Green passes · orange is flagged")

    axes[1].scatter(df["luminance"], df["saturation"], c=colours, s=42, edgecolor="white")
    axes[1].axvline(rules["luminance_max"], color=GREY, ls="--", lw=1.5)
    axes[1].axhline(rules["saturation_min"], color=GREY, ls="--", lw=1.5)
    axes[1].set_xlabel("luminance")
    axes[1].set_ylabel("saturation")
    axes[1].set_title("Washed out sits bottom-right")
    fig.suptitle(f"Image-quality screening — {int(df['usable'].sum())}/{len(df)} experiments pass",
                 fontsize=13)
    fig.savefig(out_dir / "f_screening.png", dpi=170)
    plt.close(fig)

    frames = pd.DataFrame(frame_rows).sort_values("contrast")
    picks = list(frames.head(args.examples).itertuples()) + \
            list(frames.tail(args.examples).itertuples())
    if picks:
        fig, axes = plt.subplots(2, args.examples, figsize=(3.2 * args.examples, 6.6),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)
        for k, item in enumerate(picks):
            ax = axes[k // args.examples, k % args.examples]
            with Image.open(item.path) as img:
                ax.imshow(img.convert("RGB"))
            ax.axis("off")
            ax.set_title(f"{item.experiment_id}\ncontrast {item.contrast:.0f} · "
                         f"cast {item.colour_cast:.2f}", fontsize=8.5,
                         color=(ORANGE if k < args.examples else GREEN))
        fig.suptitle("Lowest-contrast frames (top) vs highest (bottom)", fontsize=13)
        fig.savefig(out_dir / "f_screening_examples.png", dpi=140)
        plt.close(fig)

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()

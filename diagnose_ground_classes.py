#!/usr/bin/env python3
"""Which ADE20K classes is the segmenter putting on the terrain we call "other"?

The qualitative audit showed the muddy ruts and puddles left OUT of the ground
mask on the wet runs — they are the surface the vehicle is actually driving on,
and the most motion-informative texture in the frame, so having them in the
penalty region would teach the model the opposite of what we want.

Rather than guess which class name to add to ``GROUND_LABELS``, this asks the
segmenter.  It samples frames, runs the same model with the same preprocessing,
and reports the class names that occupy the region we currently call "other",
split by where in the frame they sit:

    bottom band   the lower ``--bottom`` share of the frame.  For a
                  forward-facing camera on a ground vehicle this is terrain
                  almost by construction, so a class with a large share HERE is
                  a candidate for GROUND_LABELS.
    top band      the rest.  Trees, buildings, tents, people — these SHOULD be
                  "other", and seeing them here is the control that says the
                  mapping is not simply broken.

    python diagnose_ground_classes.py --data-root <collection> --num 240 \
        --output-dir artifacts/ground_diagnosis

A class only earns a place in GROUND_LABELS if it is (a) big in the bottom band,
(b) small in the top band, and (c) actually terrain when you look at the crops
this writes.  "hill" and "mountain" will fail (b); that is the point of the test.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from vit_motion.sky_mask import GROUND_LABELS, SKY_LABELS, SegformerSkyMasker


def sample_frames(roots: list[str], num: int, seed: int) -> list[tuple[str, Path]]:
    """Spread the sample over experiments so one long run cannot dominate."""
    per_exp: dict[str, list[Path]] = {}
    for root in roots:
        for csv_path in sorted(Path(root).expanduser().resolve().glob("*/samples.csv")):
            exp = csv_path.parent
            frames = sorted((exp / "rgb").glob("*.png"))
            if frames:
                per_exp.setdefault(exp.name, frames)
    if not per_exp:
        raise SystemExit("No rgb frames found.")
    rng = random.Random(seed)
    per = max(1, num // len(per_exp))
    out: list[tuple[str, Path]] = []
    for name, frames in per_exp.items():
        for path in rng.sample(frames, min(per, len(frames))):
            out.append((name, path))
    rng.shuffle(out)
    return out[:num]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", nargs="+", required=True)
    ap.add_argument("--num", type=int, default=240)
    ap.add_argument("--size", type=int, nargs=2, default=(224, 224))
    ap.add_argument("--bottom", type=float, default=0.60,
                    help="Share of the frame height treated as the terrain band.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="artifacts/ground_diagnosis")
    ap.add_argument("--crops", type=int, default=6,
                    help="Save this many labelled crops per candidate class.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = args.size

    frames = sample_frames(args.data_root, args.num, args.seed)
    print(f"[1/3] {len(frames)} frames from "
          f"{len({e for e, _ in frames})} experiments")

    masker = SegformerSkyMasker(model_name=args.model, batch_size=args.batch_size)
    id2label = masker.id2label
    sky_set = {s.lower() for s in SKY_LABELS}
    ground_set = {s.lower() for s in GROUND_LABELS}

    def bucket(name: str) -> str:
        parts = {p.strip().lower() for p in str(name).split(",")}
        if parts & sky_set:
            return "sky"
        if parts & ground_set:
            return "ground"
        return "other"

    split = int(round(height * (1.0 - args.bottom)))
    bottom_px = collections.Counter()
    top_px = collections.Counter()
    frames_with = collections.Counter()
    examples: dict[int, list[tuple[str, Path, float]]] = collections.defaultdict(list)
    n_bottom = n_top = 0

    print("[2/3] Segmenting")
    for start in range(0, len(frames), args.batch_size):
        chunk = frames[start:start + args.batch_size]
        images = [Image.open(p).convert("RGB") for _, p in chunk]
        maps = masker.label_maps(images, size=(height, width))
        for (exp, path), labels in zip(chunk, maps):
            bot, top = labels[split:], labels[:split]
            n_bottom += bot.size
            n_top += top.size
            for value, count in zip(*np.unique(bot, return_counts=True)):
                bottom_px[int(value)] += int(count)
                share = count / bot.size
                if share > 0.02:
                    frames_with[int(value)] += 1
                    examples[int(value)].append((exp, path, float(share)))
            for value, count in zip(*np.unique(top, return_counts=True)):
                top_px[int(value)] += int(count)
        for image in images:
            image.close()

    rows = []
    for class_id in set(bottom_px) | set(top_px):
        name = str(id2label.get(class_id, class_id))
        rows.append({
            "class_id": class_id,
            "label": name,
            "current_bucket": bucket(name),
            "bottom_share": bottom_px[class_id] / max(n_bottom, 1),
            "top_share": top_px[class_id] / max(n_top, 1),
            "frames_over_2pct": frames_with[class_id],
        })
    table = pd.DataFrame(rows)
    # A terrain class lives low in the frame and not high in it.
    table["bottom_bias"] = table["bottom_share"] / (table["top_share"] + 1e-6)
    table = table.sort_values("bottom_share", ascending=False).reset_index(drop=True)
    table.to_csv(out_dir / "class_shares.csv", index=False)

    show = table[table["bottom_share"] > 0.001].copy()
    for column in ("bottom_share", "top_share"):
        show[column] = (show[column] * 100).round(2)
    show["bottom_bias"] = show["bottom_bias"].round(1)
    print("\n[3/3] classes occupying the bottom "
          f"{args.bottom:.0%} of the frame (share in %):\n")
    print(show[["label", "current_bucket", "bottom_share", "top_share",
                "bottom_bias", "frames_over_2pct"]].to_string(index=False))

    candidates = table[(table["current_bucket"] == "other")
                       & (table["bottom_share"] > 0.005)
                       & (table["bottom_bias"] > 2.0)]
    print("\ncandidates for GROUND_LABELS "
          "(currently 'other', >0.5% of the terrain band, biased low):")
    if candidates.empty:
        print("   none — the current class list already covers the terrain.")
    else:
        for _, row in candidates.iterrows():
            print(f"   {row['label']:<40s} bottom {row['bottom_share']:.1%} "
                  f"top {row['top_share']:.1%}  bias x{row['bottom_bias']:.0f}")
        print("\nLook at the crops before adding any of these.")

    # ---- crops, so the decision is made by eye and not by the table alone ----
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(exist_ok=True)
    saved = []
    for class_id in candidates["class_id"].tolist()[:8]:
        label = str(id2label.get(class_id, class_id)).split(",")[0].strip()
        safe = "".join(c if c.isalnum() else "_" for c in label)
        picks = sorted(examples[class_id], key=lambda t: -t[2])[:args.crops]
        for rank, (exp, path, share) in enumerate(picks, 1):
            image = Image.open(path).convert("RGB").resize((width, height))
            labels = masker.label_maps([image], size=(height, width))[0]
            overlay = np.asarray(image).astype(np.float32)
            hit = labels == class_id
            overlay[hit] = 0.45 * overlay[hit] + 0.55 * np.array([255.0, 40.0, 40.0])
            Image.fromarray(overlay.astype(np.uint8)).save(
                crop_dir / f"{safe}_{rank}_{exp}_{share:.2f}.png")
            saved.append(f"{safe}_{rank}_{exp}")
            image.close()

    summary = {
        "model": args.model,
        "n_frames": len(frames),
        "bottom_band": args.bottom,
        "current_ground_labels": sorted(GROUND_LABELS),
        "candidates": [
            {"label": r["label"], "class_id": int(r["class_id"]),
             "bottom_share": round(float(r["bottom_share"]), 4),
             "top_share": round(float(r["top_share"]), 4),
             "frames_over_2pct": int(r["frames_over_2pct"])}
            for _, r in candidates.iterrows()
        ],
        "crops_written": len(saved),
    }
    (out_dir / "diagnosis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{len(saved)} crops (candidate class highlighted red) -> {crop_dir}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
